"""
VCF annotations and an annotator to apply them.
"""

__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"
__date__ = "2023-05-09"

import itertools
import logging
import re
from abc import ABC, abstractmethod
from collections import Counter
from enum import Enum
from functools import cached_property
from itertools import product
from typing import List, Optional, Dict, Tuple, Callable, Literal, Iterable, cast, Any

import Bio.Data.CodonTable
import numpy as np
import pandas as pd
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from cyvcf2 import Variant, Writer, VCF
from matplotlib import pyplot as plt
from scipy.optimize import minimize, OptimizeResult

from . import Visualization, Spectrum
from .bio_handlers import FASTAHandler, GFFHandler, VCFHandler, get_major_base, get_called_bases
from .optimization import parallelize as parallelize_func, check_bounds

# get logger
logger = logging.getLogger('fastdfe')

# order of the bases important
bases = np.array(['A', 'C', 'G', 'T'])

# base indices
base_indices = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

# codon table
codon_table = Bio.Data.CodonTable.standard_dna_table.forward_table

# stop codons
stop_codons = Bio.Data.CodonTable.standard_dna_table.stop_codons

# start codons
start_codons = ['ATG']

# include stop codons
for c in stop_codons:
    codon_table[c] = 'Σ'

# The degeneracy of the site according to how many unique amino acids
# are code for when change the site within the codon.
# We count the third position of the isoleucine codon as 2-fold degenerate.
# This is the only site that would normally have 3-fold degeneracy
# https://en.wikipedia.org/wiki/Codon_degeneracy
unique_to_degeneracy = {0: 0, 1: 2, 2: 2, 3: 4}


class Annotation:

    def __init__(self):
        """
        Create a new annotation instance.
        """
        #: The logger.
        self.logger = logger.getChild(self.__class__.__name__)

        #: The annotator.
        self.annotator: Annotator | None = None

        #: The number of annotated sites.
        self.n_annotated: int = 0

    def _setup(self, annotator: 'Annotator', reader: VCF):
        """
        Provide context by passing the annotator. This should be called before the annotation starts.

        :param annotator: The annotator.
        :param reader: The VCF reader.
        """
        self.annotator = annotator

    def _teardown(self):
        """
        Finalize the annotation. Called after all sites have been annotated.
        """
        self.logger.info(f'Annotated {self.n_annotated} sites.')

    def annotate_site(self, variant: Variant):
        """
        Annotate a single site.

        :param variant: The variant to annotate.
        :return: The annotated variant.
        """
        pass

    @staticmethod
    def count_target_sites(file: str) -> Dict[str, int]:
        """
        Count the number of target sites in a GFF file.

        :param file: The path to The GFF file path, possibly gzipped or a URL
        :return: The number of target sites per chromosome/contig.
        """
        return GFFHandler(file)._count_target_sites()


class DegeneracyAnnotation(Annotation, GFFHandler, FASTAHandler):
    """
    Degeneracy annotation. We annotate the degeneracy by looking at each codon for coding variants.
    This also annotates mono-allelic sites.

    This annotation adds the info fields ``Degeneracy`` and ``Degeneracy_Info``, which hold the degeneracy
    of a site (0, 2, 4) and extra information about the degeneracy, respectively. To be used with
    :class:`~fastdfe.parser.DegeneracyStratification`.
    """

    #: The genomic positions for coding sequences that are mocked.
    _pos_mock: int = 1e100

    def __init__(
            self,
            gff_file: str,
            fasta_file: str,
            aliases: Dict[str, List[str]] = {},
            cache: bool = True
    ):
        """
        Create a new annotation instance.

        :param gff_file: The GFF file path, possibly gzipped or a URL
        :param fasta_file: The FASTA file path, possibly gzipped or a URL
        :param aliases: Dictionary of aliases for the contigs in the VCF file, e.g. ``{'chr1': ['1']}``.
            This is used to match the contig names in the VCF file with the contig names in the FASTA file and GFF file.
        :param cache: Whether to cache files that are downloaded from URLs
        """
        Annotation.__init__(self)

        GFFHandler.__init__(self, gff_file, cache=cache)

        FASTAHandler.__init__(self, fasta_file, cache=cache)

        #: Dictionary of aliases for the contigs in the VCF file
        self.aliases: Dict[str, List[str]] = aliases

        #: The current coding sequence or the closest coding sequence downstream.
        self.cd: Optional[pd.Series] = None

        #: The coding sequence following the current coding sequence.
        self.cd_next: Optional[pd.Series] = None

        #: The coding sequence preceding the current coding sequence.
        self.cd_prev: Optional[pd.Series] = None

        #: The current contig.
        self.contig: Optional[SeqRecord] = None

        #: The variants that could not be annotated correctly.
        self.mismatches: List[Variant] = []

        #: The variant that were skipped because they were not in coding regions.
        self.n_skipped = 0

        #: The variants for which the codon could not be determined.
        self.errors: List[Variant] = []

        #: The number of sites in coding sequences determined on runtime by looking at the GFF file.
        self.n_target_sites: int = 0

    def _setup(self, annotator: 'Annotator', reader: VCF):
        """
        Provide context to the annotator.

        :param annotator: The annotator.
        :param reader: The reader.
        """
        super()._setup(annotator, reader)

        # touch the cached properties to make for a nicer logging experience
        # noinspection PyStatementEffect
        self._cds

        # noinspection PyStatementEffect
        self._ref

        reader.add_info_to_header({
            'ID': 'Degeneracy',
            'Number': '.',
            'Type': 'Integer',
            'Description': 'n-fold degeneracy'
        })

        reader.add_info_to_header({
            'ID': 'Degeneracy_Info',
            'Number': '.',
            'Type': 'Integer',
            'Description': 'Additional information about degeneracy annotation'
        })

    def _parse_codon_forward(self, variant: Variant):
        """
        Parse the codon in forward direction.

        :param variant: The variant.
        :return: Codon, Codon position, Codon start position, Position within codon, and relative position.
        """
        # position relative to start of coding sequence
        pos_rel = variant.POS - (self.cd.start + int(self.cd.phase))

        # position relative to codon
        pos_codon = pos_rel % 3

        # inclusive codon start, 1-based
        codon_start = variant.POS - pos_codon

        # the codon positions
        codon_pos = [codon_start, codon_start + 1, codon_start + 2]

        if self.cd_prev is None and codon_pos[0] < self.cd.start:
            raise IndexError(f'Codon at site {variant.CHROM}:{variant.POS} overlaps with '
                             f'start position of current CDS and no previous CDS was given.')

        # Use final positions from previous coding sequence if current codon
        # starts before start position of current coding sequence
        if codon_pos[1] == self.cd.start:
            codon_pos[0] = self.cd_prev.end if self.cd_prev.strand == '+' else self.cd_prev.start
        elif codon_pos[2] == self.cd.start:
            codon_pos[1] = self.cd_prev.end if self.cd_prev.strand == '+' else self.cd_prev.start
            codon_pos[0] = self.cd_prev.end - 1 if self.cd_prev.strand == '+' else self.cd_prev.start + 1

        if self.cd_next is None and codon_pos[2] > self.cd.end:
            raise IndexError(f'Codon at site {variant.CHROM}:{variant.POS} overlaps with '
                             f'end position of current CDS and no subsequent CDS was given.')

        # use initial positions from subsequent coding sequence if current codon
        # ends before end position of current coding sequence
        if codon_pos[2] == self.cd.end + 1:
            codon_pos[2] = self.cd_next.start if self.cd_next.strand == '+' else self.cd_next.end
        elif codon_pos[1] == self.cd.end + 1:
            codon_pos[1] = self.cd_next.start if self.cd_next.strand == '+' else self.cd_next.end
            codon_pos[2] = self.cd_next.start + 1 if self.cd_next.strand == '+' else self.cd_next.end - 1

        # seq uses 0-based positions
        codon = ''.join([str(self.contig[int(pos - 1)]) for pos in codon_pos]).upper()

        return codon, codon_pos, codon_start, pos_codon, pos_rel

    def _parse_codon_backward(self, variant: Variant):
        """
        Parse the codon in reverse direction.

        :param variant: The variant.
        :return: Codon, Codon position, Codon start position, Position within codon, and relative position.
        """
        # position relative to end of coding sequence
        pos_rel = (self.cd.end - int(self.cd.phase)) - variant.POS

        # position relative to codon end
        pos_codon = pos_rel % 3

        # inclusive codon start, 1-based
        codon_start = variant.POS + pos_codon

        # the codon positions
        codon_pos = [codon_start, codon_start - 1, codon_start - 2]

        if self.cd_prev is None and codon_pos[2] < self.cd.start:
            raise IndexError(f'Codon at site {variant.CHROM}:{variant.POS} overlaps with '
                             f'start position of current CDS and no previous CDS was given.')

        # Use final positions from previous coding sequence if current codon
        # ends before start position of current coding sequence.
        if codon_pos[1] == self.cd.start:
            codon_pos[2] = self.cd_prev.end if self.cd_prev.strand == '-' else self.cd_prev.start
        elif codon_pos[0] == self.cd.start:
            codon_pos[1] = self.cd_prev.end if self.cd_prev.strand == '-' else self.cd_prev.start
            codon_pos[2] = self.cd_prev.end - 1 if self.cd_prev.strand == '-' else self.cd_prev.start + 1

        if self.cd_next is None and codon_pos[0] > self.cd.end:
            raise IndexError(f'Codon at site {variant.CHROM}:{variant.POS} overlaps with '
                             f'end position of current CDS and no subsequent CDS was given.')

        # use initial positions from subsequent coding sequence if current codon
        # starts before end position of current coding sequence
        if codon_pos[0] == self.cd.end + 1:
            codon_pos[0] = self.cd_next.start if self.cd_next.strand == '-' else self.cd_next.end
        elif codon_pos[1] == self.cd.end + 1:
            codon_pos[1] = self.cd_next.start if self.cd_next.strand == '-' else self.cd_next.end
            codon_pos[0] = self.cd_next.start + 1 if self.cd_next.strand == '-' else self.cd_next.end - 1

        # we use 0-based positions here
        codon = ''.join(str(self.contig[int(pos - 1)]) for pos in codon_pos)

        # take complement and convert to uppercase ('n' might be lowercase)
        codon = str(Seq(codon).complement()).upper()

        return codon, codon_pos, codon_start, pos_codon, pos_rel

    def _parse_codon(self, variant: Variant):
        """
        Parse the codon for the given variant.

        :param variant: The variant to parse the codon for.
        :return: Codon, Codon position, Codon start position, Position within codon, and relative position.
        """

        if self.cd.strand == '+':
            return self._parse_codon_forward(variant)

        return self._parse_codon_backward(variant)

    @staticmethod
    def _get_degeneracy(codon: str, pos: int) -> int:
        """
        Translate codon into amino acid.

        :param codon: The codon
        :param pos: The position of the variant in the codon, 0, 1, or 2
        :return: The degeneracy of the codon, 0, 2, or 4
        """
        amino_acid = codon_table[codon]

        # convert to list of characters
        codon = list(codon)

        # get the alternative bases
        alt = []
        for b in bases[bases != codon[pos]]:
            codon[pos] = b
            alt.append(codon_table[''.join(codon)])

        # noinspection PyTypeChecker
        return unique_to_degeneracy[sum(amino_acid == np.array(alt))]

    @staticmethod
    def _get_degeneracy_table() -> Dict[str, str]:
        """
        Create codon degeneracy table.

        :return: dictionary mapping codons to degeneracy
        """
        codon_degeneracy = {}
        for codon in product(bases, repeat=3):
            codon = ''.join(codon)
            codon_degeneracy[codon] = ''.join(
                [str(DegeneracyAnnotation._get_degeneracy(codon, pos)) for pos in range(0, 3)]
            )

        return codon_degeneracy

    def _fetch_cds(self, v: Variant):
        """
        Fetch the coding sequence for the given variant.

        :param v: The variant to fetch the coding sequence for.
        :raises LookupError: If no coding sequence was found.
        """
        # get the aliases for the current chromosome
        aliases = self.get_aliases(v.CHROM, self.aliases)

        # only fetch coding sequence if we are on a new chromosome or the
        # variant is not within the current coding sequence
        if self.cd is None or self.cd.seqid not in aliases or v.POS > self.cd.end:

            # reset coding sequences to mocking positions
            self.cd_prev = None
            self.cd = pd.Series({'seqid': v.CHROM, 'start': self._pos_mock, 'end': self._pos_mock})
            self.cd_next = pd.Series({'seqid': v.CHROM, 'start': self._pos_mock, 'end': self._pos_mock})

            # filter for the current chromosome
            on_contig = self._cds[(self._cds.seqid.isin(aliases))]

            # filter for positions ending after the variant
            cds = on_contig[(on_contig.end >= v.POS)]

            if not cds.empty:
                # take the first coding sequence
                self.cd = cds.iloc[0]

                self.logger.debug(f'Found coding sequence: {self.cd.seqid}:{self.cd.start}-{self.cd.end}, '
                                  f'reminder: {(self.cd.end - self.cd.start + 1) % 3}, '
                                  f'phase: {int(self.cd.phase)}, orientation: {self.cd.strand}, '
                                  f'current position: {v.CHROM}:{v.POS}')

                # filter for positions ending after the current coding sequence
                cds = on_contig[(on_contig.start > self.cd.end)]

                if not cds.empty:
                    # take the first coding sequence
                    self.cd_next = cds.iloc[0]

                # filter for positions starting before the current coding sequence
                cds = on_contig[(on_contig.end < self.cd.start)]

                if not cds.empty:
                    # take the last coding sequence
                    self.cd_prev = cds.iloc[-1]

            if self.cd.start == self._pos_mock and self.n_annotated == 0:
                self.logger.warning(f"No coding sequence found on all of contig '{v.CHROM}' and no previous "
                                    f'sites were annotated. Are you sure that this is the correct GFF file '
                                    f'and that the contig names match the chromosome names in the VCF file? '
                                    f'Note that you can also specify aliases for contig names in the VCF file.')

        # check if variant is located within coding sequence
        if self.cd is None or not (self.cd.start <= v.POS <= self.cd.end):
            raise LookupError(f"No coding sequence found, skipping record {v.CHROM}:{v.POS}")

    def _fetch_contig(self, v: Variant):
        """
        Fetch the contig for the given variant.

        :param v: The variant to fetch the contig for.
        """
        aliases = self.get_aliases(v.CHROM, self.aliases)

        # check if contig is up-to-date
        if self.contig is None or self.contig.id not in aliases:
            self.logger.debug(f"Fetching contig '{v.CHROM}'.")

            # fetch contig
            self.contig = self.get_contig(aliases)

            # add to number of target sites
            self.n_target_sites += self._compute_lengths(self._cds[(self._cds.seqid.isin(aliases))])['length'].sum()

    def _fetch(self, variant: Variant):
        """
        Fetch all required data for the given variant.

        :param variant:
        :raises LookupError: if some data could not be found.
        """
        self._fetch_cds(variant)
        self._fetch_contig(variant)

    def annotate_site(self, v: Variant):
        """
        Annotate a single site.

        :param v: The variant to annotate.
        """
        v.INFO['Degeneracy'] = '.'

        try:
            self._fetch(v)
        except LookupError:
            self.n_skipped += 1
            return

        # skip locus if not a single site
        if len(v.REF) != 1:
            self.n_skipped += 1
            return

        # annotate if record is in coding sequence
        if self.cd.seqid in self.get_aliases(v.CHROM, self.aliases) and self.cd.start <= v.POS <= self.cd.end:

            try:
                # parse codon
                codon, codon_pos, codon_start, pos_codon, pos_rel = self._parse_codon(v)

            except IndexError as e:

                # skip site on IndexError
                self.logger.warning(e)
                self.errors.append(v)
                return

            # make sure the reference allele matches with the position on the reference genome
            if str(self.contig[v.POS - 1]).upper() != v.REF.upper():
                self.logger.warning(f"Reference allele does not match with reference genome at {v.CHROM}:{v.POS}.")
                self.mismatches.append(v)
                return

            degeneracy = '.'
            if 'N' not in codon:
                degeneracy = self._get_degeneracy(codon, pos_codon)

                # increment counter of annotated sites
                self.n_annotated += 1

            v.INFO['Degeneracy'] = degeneracy
            v.INFO['Degeneracy_Info'] = f"{pos_codon},{self.cd.strand},{codon}"

            self.logger.debug(f'pos codon: {pos_codon}, pos abs: {v.POS}, '
                              f'codon start: {codon_start}, codon: {codon}, '
                              f'strand: {self.cd.strand}, ref allele: {self.contig[v.POS - 1]}, '
                              f'degeneracy: {degeneracy}, codon pos: {str(codon_pos)}, '
                              f'ref allele: {v.REF}')


class SynonymyAnnotation(DegeneracyAnnotation):
    """
    Synonymy annotation. This class annotates a variant with the synonymous/non-synonymous status.
    Use this when mono-allelic sites are not present in the VCF file.

    This annotation adds the info fields ``Synonymous`` and ``Synonymous_Info``, which hold
    the synonymous status (Synonymous (0) or non-synonymous (1)) and the codon information, respectively.
    To be used with :class:`~fastdfe.parser.SynonymyStratification`.

    Note that since we cannot determine the synonymy for monomorphic sites, we determine the number of
    target sites dynamically by adding up the number of  coding sequences per contig contained in the vcf
    file. This value is broadcast to :class:`~fastdfe.parser.Parser` if ``n_target_sites`` is not set.
    """

    def __init__(self, gff_file: str, fasta_file: str, aliases: Dict[str, List[str]] = {}):
        """
        Create a new annotation instance.

        :param gff_file: The GFF file path, possibly gzipped or a URL
        :param fasta_file: The FASTA file path, possibly gzipped or a URL
        :param aliases: Dictionary of aliases for the contigs in the VCF file, e.g. ``{'chr1': ['1']}``.
            This is used to match the contig names in the VCF file with the contig names in the FASTA file and GFF file.
        """
        super().__init__(
            gff_file=gff_file,
            fasta_file=fasta_file,
            aliases=aliases
        )

        #: The number of sites that did not match with VEP.
        self.vep_mismatches: List[Variant] = []

        #: The number of sites that did not math with the annotation provided by SnpEff
        self.snpeff_mismatches: List[Variant] = []

        #: The number of sites that were concordant with VEP.
        self.n_vep_comparisons: int = 0

        #: The number of sites that were concordant with SnpEff.
        self.n_snpeff_comparisons: int = 0

        #: The aliases for the contigs in the VCF file.
        self.aliases: Dict[str, List[str]] = aliases

    def _setup(self, annotator: 'Annotator', reader: VCF):
        """
        Provide context to the annotator.

        :param annotator: The annotator.
        :param reader: The reader.
        """
        Annotation._setup(self, annotator, reader)

        # touch the cached properties to make for a nicer logging experience
        # noinspection PyStatementEffect
        self._cds

        # noinspection PyStatementEffect
        self._ref

        reader.add_info_to_header({
            'ID': 'Synonymy',
            'Number': '.',
            'Type': 'Integer',
            'Description': 'Synonymous (0) or non-synonymous (1)'
        })

        reader.add_info_to_header({
            'ID': 'Synonymy_Info',
            'Number': '.',
            'Type': 'String',
            'Description': 'Alt codon and extra information'
        })

    def _get_alt_allele(self, variant: Variant) -> str | None:
        """
        Get the alternative allele.

        :param variant: The variant to get the alternative allele for.
        :return: The alternative allele or None if there is no alternative allele.
        """
        if len(variant.ALT) > 0:

            # assume there is at most one alternative allele
            if self.cd.strand == '-':
                return Seq(variant.ALT[0]).complement().__str__()

            return variant.ALT[0]

        return None

    @staticmethod
    def mutate(codon: str, alt: str, pos: int) -> str:
        """
        Mutate the codon at the given position to the given alternative allele.

        :param codon: The codon to mutate.
        :param alt: The alternative allele.
        :param pos: The position to mutate.
        :return: Mutated codon.
        """
        return codon[0:pos] + alt + codon[pos + 1:]

    @staticmethod
    def is_synonymous(codon1: str, codon2: str) -> bool:
        """
        Check if two codons are synonymous.

        :param codon1: The first codon.
        :param codon2: The second codon.
        :return: True if the codons are synonymous, False otherwise.
        """

        # handle case where there are stop codons
        if codon1 in stop_codons or codon2 in stop_codons:
            return codon1 in stop_codons and codon2 in stop_codons

        return codon_table[codon1] == codon_table[codon2]

    def _parse_codons_vep(self, variant: Variant) -> List[str]:
        """
        Parse the codons from the VEP annotation if present.

        :param variant: The variant.
        :return: The codons.
        """
        # match codons
        match = re.search("([actgACTG]{3})/([actgACTG]{3})", variant.INFO.get('CSQ'))

        if match is not None:
            if len(match.groups()) != 2:
                self.logger.info(f'VEP annotation has more than two codons: {variant.INFO.get("CSQ")}')

            return [m.upper() for m in [match[1], match[2]]]

        return []

    @staticmethod
    def _parse_synonymy_snpeff(variant: Variant) -> int | None:
        """
        Parse the synonymy from the annotation provided by SnpEff

        :param variant: The variant.
        :return: The codons.
        """
        ann = variant.INFO.get('ANN')

        if 'synonymous_variant' in ann:
            return 1

        if 'missense_variant' in ann:
            return 0

        return None

    def _teardown(self):
        """
        Finalize the annotation.
        """
        super()._teardown()

        if self.n_vep_comparisons != 0:
            self.logger.info(f'Number of mismatches with VEP: {len(self.vep_mismatches)}')

        if self.n_snpeff_comparisons != 0:
            self.logger.info(f'Number of mismatches with SnpEff: {len(self.snpeff_mismatches)}')

    def annotate_site(self, v: Variant):
        """
        Annotate a single site.

        :param v: The variant to annotate.
        :return: The annotated variant.
        """
        v.INFO['Synonymy'] = '.'

        if v.is_snp:
            try:
                self._fetch(v)
            except LookupError:
                self.n_skipped += 1
                return

            # annotate if record is in coding sequence
            if self.cd.start <= v.POS <= self.cd.end:

                try:
                    # parse codon
                    codon, codon_pos, codon_start, pos_codon, pos_rel = self._parse_codon(v)

                except IndexError as e:

                    # skip site on IndexError
                    self.logger.warning(e)
                    self.errors.append(v)
                    return

                # make sure the reference allele matches with the position in the reference genome
                if str(self.contig[v.POS - 1]).upper() != v.REF.upper():
                    self.logger.warning(f"Reference allele does not match with reference genome at {v.CHROM}:{v.POS}.")
                    self.mismatches.append(v)
                    return

                # fetch the alternative allele if present
                alt = self._get_alt_allele(v)

                info = ''
                synonymy, alt_codon, codons_vep = None, None, None
                if alt is not None:
                    # alternative codon, 'n' might not be uppercase
                    alt_codon = self.mutate(codon, alt, pos_codon).upper()

                    # whether the alternative codon is synonymous
                    if 'N' not in codon and 'N' not in alt_codon:
                        synonymy = int(self.is_synonymous(codon, alt_codon))

                    # append alternative codon to info field
                    info += f'{codon}/{alt_codon}'

                    # check if the alternative codon is a start codon
                    if alt_codon in start_codons:
                        info += ',start_gained'

                    # check if the alternative codon is a stop codon
                    if alt_codon in stop_codons:
                        info += ',stop_gained'

                    if v.INFO.get('CSQ') is not None:

                        # fetch the codons from the VEP annotation
                        codons_vep = self._parse_codons_vep(v)

                        if len(codons_vep) > 0:
                            # increase number of comparisons
                            self.n_vep_comparisons += 1

                            # make sure the codons determined by VEP are the same as our codons.
                            if set(codons_vep) != {codon, alt_codon}:
                                self.logger.warning(f'VEP codons do not match with codons determined by '
                                                    f'codon table for {v.CHROM}:{v.POS}')

                                self.vep_mismatches.append(v)
                                return

                if v.INFO.get('ANN') is not None:
                    synonymy_snpeff = self._parse_synonymy_snpeff(v)

                    self.n_snpeff_comparisons += 1

                    if synonymy_snpeff is not None:
                        if synonymy_snpeff != synonymy:
                            self.logger.warning(f'SnpEff annotation does not match with custom '
                                                f'annotation for {v.CHROM}:{v.POS}')

                            self.snpeff_mismatches.append(v)
                            return

                # increase number of annotated sites
                self.n_annotated += 1

                # add to info field
                v.INFO['Synonymy'] = synonymy
                v.INFO['Synonymy_Info'] = info


class AncestralAlleleAnnotation(Annotation, ABC):
    """
    Base class for ancestral allele annotation.
    """

    def _setup(self, annotator: 'Annotator', reader: VCF):
        """
        Add info fields to the header.

        :param annotator: The annotator.
        :param reader: The reader.
        """
        super()._setup(annotator, reader)

        reader.add_info_to_header({
            'ID': self.annotator.info_ancestral,
            'Number': '.',
            'Type': 'Character',
            'Description': 'Ancestral Allele'
        })


class MaximumParsimonyAncestralAnnotation(AncestralAlleleAnnotation):
    """
    Annotation of ancestral alleles using maximum parsimony. The info field ``AA`` is added to the VCF file,
    which holds the ancestral allele. To be used with :class:`~fastdfe.parser.AncestralBaseStratification` and
    :class:`~fastdfe.annotation.Annotator` or :class:`~fastdfe.parser.Parser`.

    Note that maximum parsimony is not a reliable way to determine ancestral alleles, so it is recommended
    to use this annotation together with the ancestral misidentification parameter ``eps`` or to fold
    spectra altogether. Alternatively, you can use :class:`~fastdfe.filtration.DeviantOutgroupFiltration` to
    filter out sites where the major allele among outgroups does not coincide with the major allele among ingroups.
    This annotation has the advantage of requiring no outgroup data. If outgroup data is available consider using
    :class`MLEAncestralAlleleAnnotation` instead.
    """

    def __init__(self, samples: List[str] = None):
        """
        Create a new ancestral allele annotation instance.

        :param samples: The samples to consider when determining the ancestral allele. If ``None``, all samples are
            considered.
        """
        super().__init__()

        #: The samples to consider when determining the ancestral allele.
        self.samples: List[str] | None = samples

        self.samples_mask: np.ndarray | None = None

    def _setup(self, annotator: 'Annotator', reader: VCF):
        """
        Add info fields to the header.

        :param annotator: The annotator.
        :param reader: The reader.
        """
        super()._setup(annotator, reader)

        # create mask for ingroups
        if self.samples is None:
            self.samples_mask = np.ones(len(reader.samples)).astype(bool)
        else:
            self.samples_mask = np.isin(reader.samples, self.samples)

    def annotate_site(self, variant: Variant):
        """
        Annotate a single site.

        :param variant: The variant to annotate.
        :return: The annotated variant.
        """
        # get the major base
        base = get_major_base(variant.gt_bases[self.samples_mask])

        # take base if defined
        major_allele = base if base is not None else '.'

        # set the ancestral allele
        variant.INFO[self.annotator.info_ancestral] = major_allele

        # increase the number of annotated sites
        self.n_annotated += 1


class SubstitutionModel(ABC):
    """
    Base class for substitution models.
    """

    #: The possible transitions
    transitions: np.ndarray[int, (..., ...)] = np.array([
        (base_indices['A'], base_indices['G']),
        (base_indices['G'], base_indices['A']),
        (base_indices['C'], base_indices['T']),
        (base_indices['T'], base_indices['C'])
    ])

    def __init__(
            self,
            bounds: Dict[str, Tuple[float, float]] = {},
            pool_branch_rates: bool = False,
            fixed_params: Dict[str, float] = {}
    ):
        """
        Create a new substitution model instance.

        :param bounds: The bounds for the parameters.
        :param pool_branch_rates: Whether to pool the branch rates. By default, each branch has its own rate which
            is optimized using MLE. If ``True``, the branch rates are pooled and a single rate is optimized. This is
            useful if the number of sites used is small.
        :param fixed_params: The fixed parameters. Parameters that are not fixed are optimized using MLE.
        """
        # validate bounds
        self.validate_bounds(bounds)

        #: Whether to pool the branch rates.
        self.pool_branch_rates: bool = pool_branch_rates

        #: The fixed parameters.
        self.fixed_params: Dict[str, float] = fixed_params

        #: Parameter bounds.
        self.bounds: Dict[str, Tuple[float, float]] = bounds

        #: Cache for the probabilities.
        self._cache: Dict[Tuple[int, int, int], float] | None = None

    def cache(self, params: Dict[str, float], n_branches: int):
        """
        Cache the probabilities for the given parameters.

        :param params: The parameters.
        :param n_branches: The number of branches.
        """
        self._cache = {}

        for (b1, b2, i) in itertools.product(range(4), range(4), range(n_branches)):
            self._cache[(b1, b2, i)] = self._get_prob(b1, b2, i, params)

    @staticmethod
    def get_x0(bounds: Dict[str, Tuple[float, float]], rng: np.random.Generator) -> Dict[str, float]:
        """
        Get the initial values for the parameters.

        :param bounds: The bounds for the parameters.
        :param rng: The random number generator.
        :return: The initial values.
        """
        x0 = {}

        # draw initial values from a uniform distribution
        for key, (lower, upper) in bounds.items():
            x0[key] = rng.uniform(lower, upper)

        return x0

    def get_bounds(self, anc: 'MaximumLikelihoodAncestralAnnotation') -> Dict[str, Tuple[float, float]]:
        """
        Get the bounds for the parameters.

        :param anc: The ancestral allele annotation.
        :return: The bounds.
        """
        return self.bounds

    @staticmethod
    def validate_bounds(bounds: Dict[str, Tuple[float, float]]):
        """
        Make sure the lower bounds are positive and the upper bounds are larger than the lower bounds.

        :param bounds: The bounds to validate
        :raises ValueError: If the bounds are invalid
        """
        for param, (lower, upper) in bounds.items():
            if lower < 0:
                raise ValueError(f'All lower bounds must be positive, got {lower} for {param}.')

            if lower >= upper:
                raise ValueError(f'Lower bounds must be smaller than upper bounds, got {lower} >= {upper} for {param}.')

    @abstractmethod
    def _get_prob(self, b1: int, b2: int, i: int, params: Dict[str, float]) -> float:
        """
        Get the probability of a branch using the substitution model.

        :param b1: First nucleotide state.
        :param b2: Second nucleotide state.
        :param i: The index of the branch.
        :param params: The parameters for the model.
        :return: The probability of the branch.
        """
        pass

    def get_prob(self, b1: int, b2: int, i: int, params: Dict[str, float]) -> float:
        """
        Get the probability of a branch using the substitution model.

        :param b1: First nucleotide state.
        :param b2: Second nucleotide state.
        :param i: The index of the branch.
        :param params: The parameters for the model.
        :return: The probability of the branch.
        """
        if self._cache is None:
            return self._get_prob(b1, b2, i, params)

        # return cached value
        return self._cache[(b1, b2, i)]


class JCSubstitutionModel(SubstitutionModel):
    """
    Jukes-Cantor substitution model.
    """

    def __init__(
            self,
            bounds: Dict[str, Tuple[float, float]] = {'K': (1e-5, 10)},
            pool_branch_rates: bool = False,
            fixed_params: Dict[str, float] = {}
    ):
        """
        Create a new substitution model instance.

        :param bounds: The bounds for the parameters. K is the branch rate.
        :param pool_branch_rates: Whether to pool the branch rates. By default, each branch has its own rate which
            is optimized using MLE. If ``True``, the branch rates are pooled and a single rate is optimized. This is
            useful if the number of sites used is small. If ``False``, each branch has its own rate denoted by "K{i}",
            where i is the branch index. If ``True``, the branch rate is denoted by "K".
        :param fixed_params: The fixed parameters. Parameters that are not fixed are optimized using MLE.
        """
        super().__init__(
            bounds=bounds,
            pool_branch_rates=pool_branch_rates,
            fixed_params=fixed_params
        )

    def get_bound(self, param: str) -> Tuple[float, float]:
        """
        Get the bounds for a parameter.

        :param param: The parameter.
        :return: The lower and upper bounds.
        """
        # check if the parameter is fixed
        if param in self.fixed_params:
            return self.fixed_params[param], self.fixed_params[param]

        # return the bounds if they are defined
        if param in self.bounds:
            return self.bounds[param]

        # attempt to get the bounds for the branch rates by removing the branch index
        param_no_index = re.sub(pattern=r'\d', repl='', string=param)

        return self.bounds[param_no_index]

    def get_bounds(self, anc: 'MaximumLikelihoodAncestralAnnotation') -> Dict[str, Tuple[float, float]]:
        """
        Get the bounds for the parameters.

        :param anc: The ancestral allele annotation instance.
        :return: The lower and upper bounds.
        """
        if self.pool_branch_rates:
            # pool the branch rates
            return {'K': self.get_bound('K')}

        # get the bounds for the branch lengths
        return {f"K{i}": self.get_bound(f"K{i}") for i in range(2 * anc.n_outgroups - 1)}

    def _get_prob(self, b1: int, b2: int, i: int, params: Dict[str, float]) -> float:
        """
        Get the probability of a branch using the substitution model.

        :param b1: First nucleotide state.
        :param b2: Second nucleotide state.
        :param i: The index of the branch.
        :param params: The parameters for the model.
        :return: The probability of the branch.
        """
        # evolutionary rate parameter for the branch
        K = params['K'] if self.pool_branch_rates else params[f'K{i}']

        if b1 == b2:
            return np.exp(-K) + (1 / 6) * K ** 2 * np.exp(-K)

        return (1 / 3) * K * np.exp(-K) + (1 / 9) * K ** 2 * np.exp(-K)


class K2SubstitutionModel(JCSubstitutionModel):
    """
    Kimura 2-parameter substitution model.
    """

    def __init__(
            self,
            bounds: Dict[str, Tuple[float, float]] = {'K': (1e-5, 10), 'k': (0.02, 0.2)},
            pool_branch_rates: bool = False,
            fixed_params: Dict[str, float] = {}
    ):
        """
        Create a new substitution model instance.

        :param bounds: The bounds for the parameters. K is the branch rate. k is the transition/transversion ratio.
        :param pool_branch_rates: Whether to pool the branch rates. By default, each branch has its own rate which
            is optimized using MLE. If ``True``, the branch rates are pooled and a single rate is optimized. This is
            useful if the number of sites used is small.
        :param fixed_params: The fixed parameters. Parameters that are not fixed are optimized using MLE.
        """
        super().__init__(
            bounds=bounds,
            pool_branch_rates=pool_branch_rates,
            fixed_params=fixed_params
        )

    def get_bounds(self, anc: 'MaximumLikelihoodAncestralAnnotation') -> Dict[str, Tuple[float, float]]:
        """
        Get the bounds for the parameters.

        :param anc: The ancestral allele annotation instance.
        :return: The lower and upper bounds.
        """
        bounds = super().get_bounds(anc)

        # add bounds for the transition/transversion ratio
        bounds["k"] = self.get_bound("k")

        return bounds

    def _get_prob(self, b1: int, b2: int, i: int, params: Dict[str, float]) -> float:
        """
        Get the probability of a branch using the K2 model.

        :param b1: First nucleotide state.
        :param b2: Second nucleotide state.
        :param i: The index of the branch.
        :param params: The parameters for the model.
        :return: The probability of the branch.
        """
        # evolutionary rate parameter for the branch
        K = params['K'] if self.pool_branch_rates else params[f'K{i}']

        # transition/transversion ratio
        k = params["k"]

        # if the ancestral and descendant nucleotide states are the same
        if b1 == b2:
            return np.exp(-K) * (1 + 0.5 * K ** 2 * (2 + k ** 2) / (k ** 2 + 4 * k + 4))

        # if we have a transition
        if (b1, b2) in [(0, 2), (2, 0), (1, 3), (3, 1)]:
            return K * np.exp(-K) * (k / (k + 2) + K * 1 / (k ** 2 + 4 * k + 4))

        # if we have a transversion
        return K * np.exp(-K) * (1 / (k + 2) + K * k / (k ** 2 + 4 * k + 4))


class SiteConfig(pd.Series):
    """
    Site configuration. A site configuration is a vector of nucleotide states.
    Operations on site configurations rather than individual sites is more
    efficient as there are not many different site configurations.
    """

    #: The multiplicity of the site.
    size: int

    #: The site indices.
    sites: np.ndarray[int]

    #: The number of major alleles.
    n_major: np.int8

    #: The major allele base.
    major_base: np.int8

    #: The minor base index.
    minor_base: np.int8

    #: The outgroup base indices.
    outgroup_bases: np.ndarray[np.int8]

    # The probability of the major allele being ancestral.
    p_ancestral: np.float64 | None

    # The probability of the minor allele.
    p_minor: np.float64 | None

    # The probability of the major allele.
    p_major: np.float64 | None


class BaseType(Enum):
    """
    The base type.
    """
    MINOR = 0
    MAJOR = 1


class MaximumLikelihoodAncestralAnnotation(AncestralAlleleAnnotation):
    """
    Annotation of ancestral alleles using a sophisticated method similar to EST-SFS
    (https://doi.org/10.1534/genetics.118.301120). The info field ``AA``
    is added to the VCF file, which holds the ancestral allele. To be used with
    :class:`~fastdfe.parser.AncestralBaseStratification` and :class:`~fastdfe.annotation.Annotator` or
    :class:`~fastdfe.parser.Parser`. This class can also be used independently, see the :meth:`from_dataframe`,
    :meth:`from_data` and :meth:`from_est_est` methods. In this case we pass the site configurations manually.

    Initially, the branch rates are determined using MLE. If ``use_prior`` is ``True``, the ancestral allele
    probabilities across sites are then also determined in a second optimization step using MLE. For every site,
    the probability that the major allele is ancestral is then calculated.

    .. warning::
        Still experimental. Use with caution.
    """

    #: The data types for the data frame
    dtypes = dict(
        n_major=np.int8,
        multiplicity=np.int16,
        sites=object,
        major_base="Int8",
        minor_base="Int8",
        outgroup_bases=object,
        p_ancestral=np.float64,
        p_minor=np.float64,
        p_major=np.float64
    )

    #: The columns to group by.
    group_cols = ['major_base', 'minor_base', 'outgroup_bases', 'n_major']

    def __init__(
            self,
            outgroups: List[str],
            n_ingroups: int,
            ingroups: List[str] = None,
            n_outgroups: int = 2,
            n_runs_rate: int = 10,
            n_runs_polarization: int = 1,
            model: SubstitutionModel = K2SubstitutionModel(),
            parallelize: bool = True,
            use_prior: bool = True,
            max_sites: int = np.inf,
    ):
        """
        Create a new ancestral allele annotation instance.

        :param outgroups: The outgroup samples to consider when determining the ancestral allele. A list of
            sample names as they appear in the VCF file.
        :param n_ingroups: The minimum number of ingroups that must be present at a site for it to be considered
            for ancestral allele inference. Note that a larger number of ingroups does not necessarily improve
            the accuracy of the ancestral allele inference (see ``use_prior``). A larger number of ingroups can lead
            to a large variance in the polarization probabilities, across different frequency counts. ``n_ingroups``
            should thus only be large if the number of sites used for the inference is also large. A sensible value is
            for a reasonable large number of sites (a few thousand) is 10 or perhaps 20 for larger numbers of sites.
            We subsample ``n_ingroups`` ingroups from the total number of ingroups for each site, and such values
            should provide representative subsamples in most cases.
        :param ingroups: The ingroup samples to consider when determining the ancestral allele. If ``None``,
            all (non-outgroup) samples are considered. A list of sample names as they appear in the VCF file.
        :param n_outgroups: The maximum number of outgroups that are considered for ancestral allele inference.
            More outgroups lead to a more accurate inference of the ancestral allele, but also increase the
            computational cost. Using more than 1 outgroup is recommended, but more than 3 is likely not necessary.
        :param n_runs_rate: The number of optimization runs to perform when determining the branch rates. You can
            check that the likelihoods of the different runs are similar by calling :meth:`plot_likelihoods`.
        :param n_runs_polarization: The number of runs to perform when determining the polarization parameters. One
            run should be sufficient as only one parameter is optimized.
        :param parallelize: Whether to parallelize the computation across multiple cores.
        :param use_prior: Whether to incorporate information about the general probability of the major allele
            being the ancestral allele across all sites with the same minor allele count. This is useful in general
            as it provides more information about the ancestral allele, but it can lead to a bias if the number of sites
            is small. For example, when we have no outgroup information for a particular site, it is good to know how
            likely it is that the major allele is ancestral. You can check that the polarization probabilities are
            smooth enough across frequency counts by calling :meth:`plot_polarization`.
        :param max_sites: The maximum number of sites to consider. This is useful if the number of sites is very large.
            Choosing a reasonably large subset of sites (on the order of a few thousand bi-allelic sites) can speed up
            the computation considerably as parsing can be slow. This subset is then used to calibrate the rate
            parameters, and possibly the polarization priors.
        """
        # check that we have at least one outgroup
        if len(outgroups) < 1:
            raise ValueError("Must specify at least one outgroup. If you do not have any outgroup "
                             "information, consider using MaximumParsimonyAncestralAnnotation instead.")

        # check that we have enough outgroups specified
        if len(outgroups) < n_outgroups:
            raise ValueError("The number of specified outgroup samples must be at least as large as ``n_outgroups``.")

        # check that we have enough ingroups specified if specified at all
        if ingroups is not None and len(ingroups) < n_ingroups:
            raise ValueError("The number of specified ingroup samples must be at least as large as ``n_ingroups``.")

        super().__init__()

        #: Whether to parallelize the computation.
        self.parallelize: bool = parallelize

        #: Maximum number of sites to consider
        self.max_sites: int = max_sites

        #: Whether to incorporate additional information about the ancestral allele.
        self.use_prior: bool = use_prior

        #: The samples to consider when determining the ancestral allele.
        self.ingroups: List[str] | None = ingroups

        #: The minimum number of ingroups that must be present at a site for it to be considered
        self.n_ingroups: int = n_ingroups

        #: Mask for ingroups
        self.ingroup_mask: np.ndarray[bool, (...,)] | None = None

        #: The outgroups to consider when determining the ancestral allele.
        self.outgroups: List[str] = outgroups

        #: The minimum number of outgroups that must be present at a site for it to be considered
        self.n_outgroups: int = n_outgroups

        #: Mask for outgroups
        self.outgroup_mask: np.ndarray[bool, (...,)] | None = None

        #: Number of random ML starts when determining the rate parameters
        self.n_runs_rate: int = n_runs_rate

        #: Number of random ML starts when determining the polarization parameters
        self.n_runs_polarization: int = n_runs_polarization

        #: The substitution model.
        self.model: SubstitutionModel = model

        #: The VCF reader.
        self.reader: VCF | None = None

        #: The data frame holding all site configurations.
        self.configs: pd.DataFrame = pd.DataFrame(columns=list(self.dtypes.keys()))

        #: The probability of all sites per frequency bin.
        self.p_bins: Dict[str, np.ndarray[float, (n_ingroups - 1,)]] | None = None

        #: The number of sites used for inference.
        self.n_sites: int | None = None

        #: The parameter names in the order they are passed to the optimizer.
        self.param_names: List[str] | None = None

        #: The log likelihoods for the different runs when optimizing the rate parameters.
        self.likelihoods: np.ndarray[float, (...,)] | None = None

        #: The best log likelihood when optimizing the rate parameters.
        self.likelihood: float | None = None

        #: Optimization result of the best run.
        self.result: OptimizeResult | None = None

        #: The MLE parameters.
        self.params_mle: Dict[str, float] | None = None

        #: Random number generator.
        self.rng: np.random.Generator | None = None

    def _setup(self, annotator: 'Annotator', reader: VCF):
        """
        Add info fields to the header.

        :param annotator: The annotator.
        :param reader: The reader.
        """
        super()._setup(annotator, reader)

        # add info field
        reader.add_info_to_header({
            'ID': self.annotator.info_ancestral + '_info',
            'Number': '.',
            'Type': 'String',
            'Description': 'Additional information about the ancestral allele.'
        })

        # set reader
        self.reader = reader

        # set rng
        self.rng = annotator.rng

        # prepare masks
        self.prepare_masks(reader.samples)

        # load data
        self.load_variants()

        # infer ancestral alleles
        self.infer()

    def prepare_masks(self, samples: List[str]):
        """
        Prepare the masks for ingroups and outgroups.

        :param samples: All samples.
        """
        # create mask for ingroups
        if self.ingroups is None:
            self.ingroup_mask = ~ np.isin(samples, self.outgroups)
        else:
            self.ingroup_mask = np.isin(samples, self.ingroups)

            # make sure all specified ingroups are present
            if np.sum(self.ingroup_mask) != len(self.ingroups):
                # get missing ingroups
                missing = np.array(self.ingroups)[~np.isin(self.ingroups, samples)]

                raise ValueError("Not all specified ingroups are present in the VCF file. "
                                 f"Missing ingroups: {', '.join(missing)}")

        # create mask for outgroups
        self.outgroup_mask = np.isin(samples, self.outgroups)

        # make sure all specified outgroups are present
        if np.sum(self.outgroup_mask) != len(self.outgroups):
            # get missing outgroups
            missing = np.array(self.outgroups)[~np.isin(self.outgroups, samples)]

            raise ValueError("Not all specified outgroups are present in the VCF file. "
                             f"Missing outgroups: {', '.join(missing)}")

    def subsample(self, bases: np.ndarray[Any], size: int) -> np.ndarray[Any]:
        """
        Subsample a set of bases.

        :param bases: A list of bases.
        :param size: The size of the subsample.
        :return: A subsample of the bases.
        """
        if bases.shape[0] == 0:
            return np.array([])

        return bases[self.annotator.rng.choice(bases.shape[0], size=min(size, bases.shape[0]), replace=False)]

    def parse_variant(self, variant: Variant) -> SiteConfig | None:
        """
        Parse a VCF variant. We only SNPs that are at most bi-allelic and have at least ``n_ingroups`` ingroup.

        :param variant: The variant.
        :return: The site configuration.
        """
        # only consider SNPs
        if not variant.is_snp:
            return None

        # get the called ingroup and outgroup bases
        ingroups = get_called_bases(variant.gt_bases[self.ingroup_mask])
        outgroups = get_called_bases(variant.gt_bases[self.outgroup_mask])

        # get the numer of called ingroup and outgroup bases
        n_ingroups = len(ingroups)

        # only consider sites with enough ingroups and outgroups
        if n_ingroups >= self.n_ingroups:

            # subsample ingroups and outgroups
            subsample_ingroups = self.subsample(ingroups, size=self.n_ingroups)
            subsample_outgroups = self.subsample(outgroups, size=self.n_outgroups)

            # get the counts of ingroups and outgroups
            counts_ingroups = Counter(subsample_ingroups)

            # only consider sites where the ingroups are at most bi-allelic
            if len(counts_ingroups) <= 2:

                # get the major and minor allele
                most_common = counts_ingroups.most_common()

                # get the bases
                bases: List[str] = list(counts_ingroups.keys())

                # take the other allele as the minor allele
                if len(counts_ingroups) == 2:
                    minor_base = base_indices[bases[0] if bases[0] != most_common[0][0] else bases[1]]
                else:
                    minor_base = np.nan

                # initialize outgroup bases
                outgroup_bases = np.full(self.n_outgroups, np.nan, dtype=np.int8)

                # fill with outgroup bases
                for i, base in enumerate(subsample_outgroups):
                    outgroup_bases[i] = base_indices[base]

                # create site configuration
                site = SiteConfig(dict(
                    major_base=base_indices[most_common[0][0]],
                    n_major=most_common[0][1],
                    minor_base=minor_base,
                    outgroup_bases=outgroup_bases,
                    multiplicity=1
                ))

                return site

        return None

    @classmethod
    def _parse_est_sfs(cls, data: pd.DataFrame) -> pd.DataFrame:
        """
        Parse EST-SFS data.

        :param data: The data frame.
        :param offset: The offset for the site indices.
        :return: The site configurations.
        """
        # extract the number of outgroups
        n_outgroups = data.shape[1] - 1

        # retain site index
        data['sites'] = data.index
        data['sites'] = data.sites.apply(lambda x: [x])

        # the first column contains the ingroup counts, split them
        ingroup_data = data[0].str.split(',', expand=True).astype(np.int8).to_numpy()

        # determine the number of major alleles per site
        data['n_major'] = ingroup_data.max(axis=1)

        # sort by the number of alleles
        data_sorted = ingroup_data.argsort(axis=1)

        # determine the number of major alleles per site
        data['major_base'] = data_sorted[:, -1]
        data['major_base'] = data.major_base.astype(cls.dtypes['major_base'])

        # determine the mono-allelic sites
        mono_allelic = (ingroup_data > 1).sum(axis=1) == 1

        # determine the minor alleles
        minor_bases = np.ma.zeros(data.shape[0], dtype=np.int8)
        minor_bases.mask = mono_allelic
        minor_bases[~mono_allelic] = data_sorted[:, -2][~mono_allelic]

        # assign the minor alleles
        data['minor_base'] = minor_bases
        data['minor_base'] = data.minor_base.astype(cls.dtypes['minor_base'])

        # extract outgroup data
        outgroup_data = np.full((data.shape[0], n_outgroups), np.nan, dtype=np.int8)
        for i in range(n_outgroups):
            # get the genotypes
            genotypes = data[i + 1].str.split(',', expand=True).astype(np.int8).to_numpy()

            # determine whether the site has an outgroup
            has_outgroup = genotypes.sum(axis=1) > 0

            # determine the outgroup allele indices provided the site has an outgroup
            outgroup_data[has_outgroup, i] = genotypes[has_outgroup].argmax(axis=1)

        # assign the outgroup data, convert to tuples for hashing
        data['outgroup_bases'] = [tuple(row) for row in outgroup_data]

        # return new columns
        return data.drop(range(n_outgroups + 1), axis=1)

    @classmethod
    def from_est_sfs(
            cls,
            file: str,
            use_prior: bool = True,
            n_runs_rate: int = 10,
            n_runs_polarization: int = 1,
            model: SubstitutionModel = K2SubstitutionModel(),
            parallelize: bool = True,
            seed: int = 0,
            chunk_size: int = 100000
    ) -> 'MaximumLikelihoodAncestralAnnotation':
        """
        Create instance from EST-SFS input file.

        :param file: File containing EST-SFS-formatted input data.
        :param use_prior: Whether to use the polarization prior (see :meth:`__init__`).
        :param n_runs_rate: Number of runs for rate estimation (see :meth:`__init__`).
        :param n_runs_polarization: Number of runs for polarization (see :meth:`__init__`).
        :param model: The substitution model (see :meth:`__init__`).
        :param parallelize: Whether to parallelize the runs (see :meth:`__init__`).
        :param seed: The seed to use for the random number generator.
        :param chunk_size: The chunk size for reading the file.
        :return: The instance.
        """
        # define an empty dataframe to accumulate the data
        data = None
        n_ingroups = 0

        # iterate over the file in chunks
        for i, chunk in enumerate(pd.read_csv(file, sep=r"\s+", header=None, dtype=str, chunksize=chunk_size)):

            # determine the number of ingroups
            n_ingroups = np.max(np.array(chunk.iloc[0, 0].split(','), dtype=int))

            # parse the data
            parsed = cls._parse_est_sfs(chunk)

            if data is None:
                # parse the data
                data = parsed
            else:
                # concatenate with previous data if available
                data = pd.concat([data, parsed])

            data = data.groupby(cls.group_cols, as_index=False, dropna=False).sum()

        # check if there is data
        if data is None:
            raise ValueError("No data found.")

        # determine the multiplicity
        data['multiplicity'] = data['sites'].apply(lambda x: len(x))

        # create from dataframe
        return cls.from_dataframe(
            data=data,
            n_runs_rate=n_runs_rate,
            n_runs_polarization=n_runs_polarization,
            model=model,
            parallelize=parallelize,
            use_prior=use_prior,
            n_ingroups=n_ingroups,
            grouped=True,
            seed=seed
        )

    @classmethod
    def from_data(
            cls,
            n_major: Iterable[int],
            major_bases: Iterable[str | int],
            minor_bases: Iterable[str | int],
            outgroup_bases: Iterable[Iterable[str | int]],
            n_ingroups: int,
            n_runs_rate: int = 10,
            n_runs_polarization: int = 1,
            model: SubstitutionModel = K2SubstitutionModel(),
            parallelize: bool = True,
            use_prior: bool = True,
            seed: int = 0,
            pass_indices: bool = False
    ) -> 'MaximumLikelihoodAncestralAnnotation':
        """
        Create an instance by passing the data directly.

        :param n_major: The number of major alleles per site. Note that this number has to be lower than ``n_ingroups``,
            as we consider the number of major alleles of subsamples of size ``n_ingroups``.
        :param major_bases: The major allele per site. A string representation of the base or the base index according
            to ``['A', 'C', 'G', 'T']`` if ``pass_indices`` is True.
        :param minor_bases: The minor allele per site. A string representation of the base or the base index according
            to ``['A', 'C', 'G', 'T']`` if ``pass_indices`` is True. If the site is mono-allelic, then the value should
            be ``None``.
        :param outgroup_bases: The outgroup alleles per site. A string representation of the base or the base index
            if ``pass_indices`` is True. This should be a list of lists, where the outer list corresponds to the sites
            and the inner list to the outgroups per site.
        :param n_ingroups: The number of ingroups samples (see :meth:`__init__`).
        :param n_runs_rate: The number of runs for rate estimation (see :meth:`__init__`).
        :param n_runs_polarization: The number of runs for polarization (see :meth:`__init__`).
        :param model: The substitution model (see :meth:`__init__`).
        :param parallelize: Whether to parallelize the runs.
        :param use_prior: Whether to use the prior (see :meth:`__init__`).
        :param seed: The seed for the random number generator.
        :param pass_indices: Whether to pass the base indices instead of the bases.
        :return: The instance.
        """
        # convert to numpy arrays
        n_major = np.array(list(n_major), dtype=np.int8)

        # make sure that the number of major alleles is larger than half the number of ingroups
        if not np.all(n_major >= (n_ingroups + 1) // 2):
            raise ValueError("Major allele counts cannot be larger than half the number of ingroups.")

        # convert to base indices
        if not pass_indices:
            major_bases = [base_indices[b] if b is not None else np.nan for b in major_bases]
            minor_bases = [base_indices[b] if b is not None else np.nan for b in minor_bases]
            outgroup_bases = [[base_indices[b] for b in c] for c in outgroup_bases]

        # create data frame
        data = pd.DataFrame({
            'n_major': n_major,
            'major_base': major_bases,
            'minor_base': minor_bases,
            'outgroup_bases': outgroup_bases
        })

        # create from dataframe
        return cls.from_dataframe(
            data=data,
            n_runs_rate=n_runs_rate,
            n_runs_polarization=n_runs_polarization,
            model=model,
            parallelize=parallelize,
            use_prior=use_prior,
            n_ingroups=n_ingroups,
            seed=seed
        )

    @classmethod
    def from_dataframe(
            cls,
            data: pd.DataFrame,
            n_ingroups: int,
            n_runs_rate: int = 10,
            n_runs_polarization: int = 1,
            model: SubstitutionModel = K2SubstitutionModel(),
            parallelize: bool = True,
            use_prior: bool = True,
            seed: int = 0,
            grouped: bool = False
    ) -> 'MaximumLikelihoodAncestralAnnotation':
        """
        Create an instance from a dataframe.

        :param data: Dataframe with the columns: ``major_base``, ``minor_base``, ``outgroup_bases``, ``n_major``.
        :param n_ingroups: The number of ingroups (see :meth:`__init__`).
        :param n_runs_rate: Number of runs for rate estimation (see :meth:`__init__`).
        :param n_runs_polarization: Number of runs for polarization (see :meth:`__init__`).
        :param model: The substitution model (see :meth:`__init__`).
        :param parallelize: Whether to parallelize computations.
        :param use_prior: Whether to use the prior (see :meth:`__init__`).
        :param seed: The seed for the random number generator.
        :param grouped: Whether the dataframe is already grouped by all columns (used for internal purposes).
        :return: The instance.
        """
        # check if dataframe is empty
        if data.empty:
            raise ValueError("Empty dataframe.")

        if not grouped:
            # only keep the columns that are needed
            data = data[cls.group_cols]

            # retain site index
            data['sites'] = data.index

            # convert outgroup bases to tuples
            data['outgroup_bases'] = data['outgroup_bases'].apply(tuple)

            # group by all columns in the chunk and keep track of the site indices
            data = data.groupby(cls.group_cols, as_index=False, dropna=False).agg(list).reset_index(drop=True)

        # determine the multiplicity
        data['multiplicity'] = data['sites'].apply(lambda x: len(x))

        # add missing columns with NaN as default value
        for col in cls.dtypes:
            if col not in data.columns:
                data[col] = np.nan

        # convert to the correct dtypes
        data = data.astype(cls.dtypes)

        # determine the number of outgroups
        n_outgroups = data.outgroup_bases.apply(len).max()

        anc = MaximumLikelihoodAncestralAnnotation(
            n_runs_rate=n_runs_rate,
            n_runs_polarization=n_runs_polarization,
            model=model,
            parallelize=parallelize,
            use_prior=use_prior,
            outgroups=[str(i) for i in range(n_outgroups)],  # pseudo names for outgroups
            n_outgroups=n_outgroups,
            ingroups=[str(i) for i in range(n_ingroups)],  # pseudo names for ingroups
            n_ingroups=n_ingroups
        )

        # assign data frame
        anc.configs = data

        # assign number of sites
        anc.n_sites = data.multiplicity.sum()

        # notify about the number of sites
        anc.logger.info(f"Loaded {anc.n_sites} sites for the inference.")

        # assign random number generator
        anc.rng = np.random.default_rng(seed)

        return anc

    def load_variants(self):
        """
        Load the variants from the VCF reader and parse them.
        """
        # initialize data frame
        self.configs = pd.DataFrame(columns=list(self.dtypes.keys()))
        self.configs.astype(self.dtypes)

        # columns to use as index
        index_cols = ['major_base', 'minor_base', 'outgroup_bases', 'n_major']

        # set index to initial site configuration
        self.configs.set_index(keys=index_cols, inplace=True)

        # create progress bar
        with self.annotator.get_pbar(desc="Loading sites") as pbar:

            # iterate over sites
            for i, variant in enumerate(self.reader):

                # parse the site
                site = self.parse_variant(variant)

                # check if site is not None
                if site is not None:

                    site_index = (
                        site.major_base,
                        site.minor_base,
                        tuple(site.outgroup_bases),
                        site.n_major
                    )

                    if site_index in self.configs.index:
                        # get the site data
                        site_data = self.configs.loc[site_index].to_dict()

                        # update the site data
                        site_data['multiplicity'] += 1
                        site_data['sites'] += [i]

                        # update the site data
                        # Note that there were problems updating the data frame directly
                        self.configs.loc[site_index] = site_data
                    else:
                        self.configs.loc[site_index] = site.to_dict() | {'sites': [i]}

                pbar.update()

                # explicitly stopping after ``n``sites fixes a bug with cyvcf2:
                # 'error parsing variant with `htslib::bcf_read` error-code: 0 and ret: -2'
                if i + 1 == self.annotator.n_sites or i + 1 == self.annotator.max_sites:
                    break

        # reset the index
        self.configs.reset_index(inplace=True, names=index_cols)

        # set the number of sites
        self.n_sites = self.configs.multiplicity.sum()

        # notify about the number of sites
        self.logger.info(f"Loaded {self.n_sites} sites for the inference.")

    def infer(self):
        """
        Infer the ancestral allele probabilities for the data provided. This consists of two steps:
        First, the rates are inferred using the likelihood function. Second, the polarization probabilities
        are inferred using the inferred rates if ``use_prior`` is ``True``.
        """
        # get the bounds
        bounds = self.model.get_bounds(self)

        # set the parameter names
        self.param_names = list(bounds.keys())

        # get the likelihood function
        fun = self.get_likelihood_rates()

        def optimize_rates(x0: Dict[str, float]) -> OptimizeResult:
            """
            Optimize the likelihood function for a single run.

            :param x0: The initial values.
            :return: The optimization results.
            """
            # optimize using scipy
            return minimize(
                fun=fun,
                x0=np.array(list(x0.values())),
                bounds=list(bounds.values()),
                method="L-BFGS-B"
            )

        # run the optimization in parallel
        results = parallelize_func(
            func=optimize_rates,
            data=[self.model.get_x0(bounds, self.rng) for _ in range(self.n_runs_rate)],
            parallelize=self.parallelize,
            pbar=True,
            desc="Optimizing rates",
            dtype=object
        )

        # get the likelihoods for each run
        self.likelihoods = -np.array([result.fun for result in results])

        # get the best likelihood
        self.likelihood = np.max(self.likelihoods)

        # get the best result
        self.result = results[np.argmax(self.likelihoods)]

        # check if the optimization was successful
        if not self.result.success:
            raise RuntimeError(f"Optimization failed with message: {self.result.message}")

        # get dictionary of MLE parameters
        self.params_mle = dict(zip(self.param_names, self.result.x))

        # check if the MLE parameters are near the bounds
        near_lower, near_upper = check_bounds(
            params=self.params_mle,
            bounds=bounds,
            scale='log',
            fixed_params=self.model.fixed_params
        )

        # warn if the MLE parameters are near the bounds
        if len(near_lower | near_upper) > 0:
            self.logger.warning(f'The MLE estimate for the rates is near the upper bound for '
                                f'{near_upper} and lower bound for {near_lower}. ')

        # warn if the MLE parameters are near the lower bounds
        if len(near_lower) > 0:
            self.logger.warning("Branch rates near lower bounds are typical for small sample sets. "
                                "Consider using a larger sample set or pooling the branch rates by setting "
                                "the ``SubstitutionModel.pool_branch_rates`` to ``True``.")

        # cache the branch probabilities for the MLE parameters
        self.model.cache(self.params_mle, 2 * self.n_outgroups - 1)

        # obtain the probability for each site and allele type (major/minor) under the MLE rate parameters
        self.configs.p_minor = self.get_p_configs(self.configs, self.model, BaseType.MINOR, self.params_mle)
        self.configs.p_major = self.get_p_configs(self.configs, self.model, BaseType.MAJOR, self.params_mle)
        self.configs.p_ancestral = self.calculate_p_ancestral(
            p_minor=self.configs['p_minor'].values,
            p_major=self.configs['p_major'].values,
            n_major=self.configs['n_major'].values
        )

    @cached_property
    def p_polarization(self) -> np.ndarray[float, (...,)]:
        """
        Get the polarization probabilities. This property is cached so that the polarization probabilities
        are only optimized once.

        :return: The polarization probabilities.
        """
        # folded frequency bin indices
        freq_indices = range(1, self.n_ingroups // 2 + 2)

        # get the likelihood functions
        funcs = dict((i, self.get_likelihood_polarization(i)) for i in freq_indices)

        def optimize_polarization(args: List[Any]) -> OptimizeResult:
            """
            Optimize the likelihood function for a single run.

            :param args: The arguments.
            :return: The optimization results.
            """
            # unpack arguments
            i, _, x0 = args

            # optimize using scipy
            return minimize(
                fun=funcs[int(i)],
                x0=np.array([x0]),
                bounds=[(0, 1)],
                method="L-BFGS-B"
            )

        # prepare arguments
        data = np.array(list(itertools.product(freq_indices, range(self.n_runs_polarization))))

        # get initial values
        initial_values = np.array([self.rng.uniform() for _ in range(data.shape[0])])

        # add initial values
        data = np.hstack((data, initial_values.reshape((-1, 1))))

        # run the optimization in parallel for each frequency bin over n_runs
        results = parallelize_func(
            func=optimize_polarization,
            data=data,
            parallelize=self.parallelize,
            pbar=True,
            desc="Optimizing polarization",
            dtype=object
        ).reshape(self.n_ingroups // 2 + 1, self.n_runs_polarization)

        # get the likelihoods for each run and frequency bin
        likelihoods = np.vectorize(lambda r: r.fun)(results)

        # choose run with the best likelihood for each frequency bin
        i_best = likelihoods.argmin(axis=1)

        # check for successful optimization
        if not np.all(np.vectorize(lambda r: r.success)(results[:, i_best])):
            # get the failure messages
            failures = [r.message for r in results[:, i_best] if not r.success]

            # raise an error
            raise RuntimeError("Polarization probability optimizations failed with messages: " + ", ".join(failures))

        # get the probabilities for each frequency bin
        probs = np.array([results[i, j].x[0] for i, j in enumerate(i_best)])

        # check for zeros or ones
        if np.any(probs == 0) or np.any(probs == 1):
            # get the number of bad frequency bins
            n_bad = np.sum((probs == 0) | (probs == 1))

            self.logger.fatal(f"Polarization probabilities are 0 for {n_bad} frequency bins which "
                              f"can be a real problem as it means that there are no sites "
                              f"for those bins. This may be due to ``n_ingroups`` "
                              f"being too large, or the number of provided sites being very "
                              f"small. You may also want to consider setting ``use_prior`` to ``False`` "
                              f"to avoid unreliable ancestral state inference and undefined ancestral "
                              f"state probabilities.")

        # return the probabilities
        return probs

    @staticmethod
    def get_p_tree(
            base: int,
            n_outgroups: int,
            internal_nodes: List[int] | np.ndarray[int],
            outgroup_bases: List[int] | np.ndarray[int],
            params: Dict[str, float],
            model: SubstitutionModel
    ) -> float:
        """
        Get the probability of a tree.

        :param base: An observed ingroup base index.
        :param n_outgroups: The number of outgroups.
        :param internal_nodes: The internal nodes of the tree. We have ``n_outgroups - 1`` internal nodes.
        :param outgroup_bases: The observed base indices for the outgroups.
        :param params: The parameters of the model.
        :param model: The model to use. Either 'K2' or 'JC'.
        """
        # get the number of branches
        n_branches = 2 * n_outgroups - 1

        # the probability for each branch
        p_branches = np.zeros(n_branches, dtype=float)

        # iterate over the branches
        for i in range(n_branches):

            # if we are on the first branch
            if i == 0:
                # combine ingroup base either with only outgroup or with first internal node
                b1 = base
                b2 = outgroup_bases[0] if n_outgroups == 1 else internal_nodes[0]

            # if we are on intermediate branches
            elif i < n_branches - 1:
                # every internal node that is not the last one combines either
                # with the next internal node or with an outgroup
                i_internal = (i + 1) // 2 - 1

                # get internal base
                b1 = internal_nodes[i_internal]

                # either connect to outgroup or next internal node
                b2 = outgroup_bases[i_internal] if i % 2 == 1 else internal_nodes[i_internal + 1]
            else:
                # last branch connects to last outgroup
                b1 = internal_nodes[-1]
                b2 = outgroup_bases[-1]

            # get the probability of the branch
            p_branches[i] = model.get_prob(b1, b2, i, params)

        # take product of all branch probabilities
        prod = p_branches.prod()

        return prod

    @staticmethod
    def is_na(x: Any) -> bool:
        """
        Check if a value is NaN or Na

        :param x: The value.
        :return: Whether the value is NaN.
        """
        return pd.isna(x) or np.isnan(x)

    @classmethod
    def get_p_config(
            cls,
            config: SiteConfig,
            base_type: BaseType,
            params: Dict[str, float],
            model: SubstitutionModel = K2SubstitutionModel()
    ) -> float:
        """
        Get the probability for a site configuration.

        :param config: The site configuration.
        :param base_type: The base type.
        :param params: The parameters for the substitution model.
        :param model: The substitution model to use.
        :return: The probability for a site.
        """
        n_outgroups = int(np.sum(np.array(config.outgroup_bases) >= 0))

        # if there are no outgroups, return 1
        if n_outgroups == 0:
            return 1

        # probability for each tree
        p_trees = np.zeros(4 ** (n_outgroups - 1), dtype=float)

        # get the focal base
        base = config.major_base if base_type == BaseType.MAJOR else config.minor_base

        # if the focal base is missing, return a probability of 0
        if cls.is_na(base):
            return 0

        # iterator over all possible internal node combinations
        for i, internal_nodes in enumerate(itertools.product(range(4), repeat=n_outgroups - 1)):
            # get the probability of the tree
            p_trees[i] = cls.get_p_tree(
                base=base,
                n_outgroups=n_outgroups,
                internal_nodes=np.array(internal_nodes),
                outgroup_bases=config.outgroup_bases,
                params=params,
                model=model
            )

        return p_trees.sum()

    @classmethod
    def get_p_configs(
            cls,
            configs: pd.DataFrame,
            model: SubstitutionModel,
            base_type: BaseType,
            params: Dict[str, float]
    ) -> np.ndarray[float, (...,)]:
        """
        Get the probabilities for each site configuration.

        :param configs: The site configurations.
        :param model: The substitution model.
        :param base_type: The base type.
        :param params: A dictionary of the rate parameters.
        :return: The probability for each site.
        """
        # the probabilities for each site
        p_configs = np.zeros(shape=(configs.shape[0]), dtype=float)

        # iterate over the sites
        for i, config in configs.iterrows():
            # get the log likelihood of the site
            p_configs[i] = cls.get_p_config(
                config=cast(SiteConfig, config),
                base_type=base_type,
                params=params,
                model=model
            )

        return p_configs

    def evaluate_likelihood_rates(self, params: Dict[str, float]) -> float:
        """
        Evaluate the likelihood function given a dictionary of parameters.

        :param params: A dictionary of parameters.
        :return: The log likelihood.
        """
        # cache the branch probabilities
        self.model.cache(params, 2 * self.n_outgroups - 1)

        # compute the likelihood
        ll = -self.get_likelihood_rates()([params[name] for name in self.param_names])

        # restore cached branch probabilities if necessary
        if self.params_mle is not None:
            self.model.cache(self.params_mle, 2 * self.n_outgroups - 1)

        return ll

    def get_likelihood_rates(self) -> Callable[[List[float]], float]:
        """
        Get the likelihood function.

        :return: The likelihood function.
        """
        # make variables available in the closure
        configs = self.configs
        model = self.model
        param_names = self.param_names
        n_outgroups = self.n_outgroups

        def compute_likelihood(params: List[float]) -> float:
            """
            Compute the negative log likelihood of the parameters.

            :param params: A list of rate parameters.
            :return: The negative log likelihood.
            """
            # unpack the parameters
            params = dict(zip(param_names, params))

            # cache the branch probabilities
            model.cache(params, 2 * n_outgroups - 1)

            # the likelihood for each site
            p_sites = np.zeros(shape=(configs.shape[0], 2), dtype=float)

            # get the probability for each site
            p_sites[:, 0] = MaximumLikelihoodAncestralAnnotation.get_p_configs(configs, model, BaseType.MAJOR, params)
            p_sites[:, 1] = MaximumLikelihoodAncestralAnnotation.get_p_configs(configs, model, BaseType.MINOR, params)

            # return the negative log likelihood and take average over major and minor bases
            # also multiply by the multiplicity of each site
            # the final likelihood is the product of the likelihoods for each site
            return -np.log((p_sites * configs.multiplicity.values[:, None]).mean(axis=1)).sum()

        return compute_likelihood

    def get_likelihood_polarization(self, i: int) -> Callable[[List[float]], float]:
        """
        Get the likelihood function.

        :param i: The ith frequency bin.
        The likelihood function evaluated for the ith frequency bin.
        """
        # make variables available in the closure
        configs = self.configs
        n_ingroups = self.n_ingroups

        def compute_likelihood(params: List[float]) -> float:
            """
            Compute the negative log likelihood of the parameters.

            :param params: The probability of polarization.
            :return: The negative log likelihood.
            """
            # get the probability of polarization for the ith frequency bin
            pi = params[0]

            # mask for sites that have i minor alleles
            i_minor = n_ingroups - configs.n_major == i

            # weight the sites by the probability of polarization
            # TODO is this the right way round?
            p_configs = pi * configs.p_major[i_minor] + (1 - pi) * configs.p_minor[i_minor]

            # return the negative log likelihood
            return -(np.log(p_configs) * configs.multiplicity[i_minor]).sum()

        return compute_likelihood

    def get_probs(self) -> np.ndarray[float, (...,)]:
        """
        Get the probabilities for the ancestral allele being the major allele for the sites used to estimate the
        parameters.

        :return: The probabilities
        """
        # get config indices for each site
        indices = self._get_site_indices()

        return self.calculate_p_ancestral(
            p_minor=self.configs.p_minor[indices].values,
            p_major=self.configs.p_major[indices].values,
            n_major=self.configs.n_major[indices].values
        )

    def _get_site_indices(self) -> np.ma.MaskedArray:
        """
        Get the list of config indices for each site.

        :return: The list of config indices as a masked array.
        """
        # Initialize indices as a masked array with all entries masked
        indices = np.ma.array(np.zeros(self.n_sites, dtype=int), mask=True)

        for i, config in self.configs.iterrows():
            for j in config.sites:
                indices[j] = i

                # unmask the updated index
                indices.mask[j] = False

        return indices

    def get_sfs(self) -> Spectrum:
        """
        Get the site-frequency spectrum for the sites used to estimate the parameters.

        :return: Spectrum object.
        """
        sfs = np.zeros(self.n_ingroups + 1, dtype=float)

        # get config indices for each site
        indices = self._get_site_indices()

        # iterate over the sites
        # TODO vectorize this
        for i, i_config in enumerate(indices):
            if self.configs.p_ancestral[i_config] >= 0.5:
                sfs[self.configs.n_major[i_config]] += 1
            else:
                sfs[self.n_ingroups - self.configs.n_major[i_config]] += 1

        return Spectrum(sfs)

    def _get_ancestral_base(
            self,
            config: SiteConfig
    ) -> (int, (float, float, float, float)):
        """
        Get the ancestral allele for each site.

        :param config: The site configuration.
        :return: The ancestral allele and a tuple of probability for the major being ancestral, the first base being
        ancestral, the second base being ancestral, and the polarization probability if using a prior.
        """
        # get the probability for the major allele
        p_minor = self.get_p_config(
            config=config,
            base_type=BaseType.MINOR,
            params=self.params_mle,
            model=self.model
        )

        # get the probability for the minor allele
        p_major = self.get_p_config(
            config=config,
            base_type=BaseType.MAJOR,
            params=self.params_mle,
            model=self.model
        )

        # get the probability of the major allele being ancestral
        p_ancestral = self.calculate_p_ancestral(p_minor=p_minor, p_major=p_major, n_major=config.n_major)

        # determine the ancestral allele
        ancestral_base = self._get_ancestral_from_prob(
            p_ancestral=p_ancestral,
            major_base=config.major_base,
            minor_base=config.minor_base
        )

        # check for NaNs
        if np.isnan(p_ancestral):
            self.logger.warning(f'p_ancestral is NaN for config {config.to_dict()}')

        if self.use_prior:
            # polarization prior for the major allele for the ith frequency bin
            pi = self.p_polarization[self.n_ingroups - config.n_major]
        else:
            pi = np.nan

        return ancestral_base, (p_ancestral, p_minor, p_major, pi)

    def _get_ancestral_from_prob(
            self,
            p_ancestral: np.ndarray[float] | float,
            major_base: np.ndarray[str] | str,
            minor_base: np.ndarray[str] | str
    ) -> np.ma.MaskedArray[float] | float:
        """
        Get the ancestral allele from the probability of the major allele being ancestral.

        :param p_ancestral: The probabilities of the major allele being ancestral.
        :param major_base: The major bases.
        :param minor_base: The minor bases.
        :return: Masked array of ancestral alleles. Missing values are masked.
        """
        # make function accept scalars
        if isinstance(p_ancestral, float):
            return self._get_ancestral_from_prob(
                np.array([p_ancestral]),
                np.array([major_base]),
                np.array([minor_base])
            )[0]

        # initialize masked array
        ancestral_bases = np.ma.masked_array(np.full(p_ancestral.shape, -1, dtype=np.int8))
        ancestral_bases.mask = np.isnan(p_ancestral)

        ancestral_bases[p_ancestral >= 0.5] = major_base[p_ancestral >= 0.5]
        ancestral_bases[p_ancestral < 0.5] = minor_base[p_ancestral < 0.5]

        return ancestral_bases

    def get_ancestral_base(
            self,
            n_major: int,
            base_minor: str,
            base_major: str,
            outgroup_bases: List[str]
    ) -> (int, (float, float, float, float)):
        """
        Get the ancestral allele for the given site information.

        TODO test this function

        :param n_major: The number of major alleles.
        :param base_minor: The minor allele.
        :param base_major: The major allele.
        :param outgroup_bases: The outgroup bases.
        :return: The ancestral allele and a tuple of probability for the major being ancestral, the first base being
            ancestral, the second base being ancestral, and the polarization probability if using a prior.
        """
        return self._get_ancestral_base(SiteConfig(dict(
            n_major=n_major,
            minor_base=base_indices[base_minor],
            major_base=base_indices[base_major],
            outgroup_bases=[base_indices[b] for b in outgroup_bases],
            mulitplicity=1
        )))

    def get_ancestral_bases(self) -> np.ma.MaskedArray[str]:
        """
        Get the ancestral allele for each site used to estimate the parameters.

        :return: Masked array of ancestral alleles. Missing values are masked.
        """
        # get config indices for each site
        indices = self._get_site_indices()

        # get the sites
        sites = self.configs.iloc[indices]

        # get the ancestral alleles using the ancestral probability from each site
        ancestral_bases = self._get_ancestral_from_prob(
            p_ancestral=sites.p_ancestral.values,
            major_base=sites.major_base.values,
            minor_base=sites.minor_base.values
        )

        # convert to base strings
        base_strings = np.ma.array(bases[ancestral_bases], mask=ancestral_bases.mask)

        # set missing values to N
        base_strings[base_strings.mask] = 'N'

        return base_strings

    def calculate_p_ancestral(
            self,
            p_minor: float | np.ndarray[float],
            p_major: float | np.ndarray[float],
            n_major: int | np.ndarray[int]
    ) -> float | np.ndarray[float]:
        """
        Calculate the probability that the ancestral allele is the major allele.

        :param p_minor: The probability or probabilities of the minor allele.
        :param p_major: The probability or probabilities of the major allele.
        :param n_major: The number or numbers of major alleles.
        :return: The probability or probabilities that the ancestral allele is the major allele.
        """
        if self.use_prior:
            # polarization prior for the major allele for the ith frequency bin
            pi = self.p_polarization[self.n_ingroups - n_major]

            # get the probability that the major allele is ancestral
            return pi * p_major / (pi * p_minor + (1 - pi) * p_major)

        # get the probability that the major allele is ancestral
        return p_major / (p_minor + p_major)

    def annotate_site(self, variant: Variant):
        """
        Annotate a single variant.

        :param variant: The variant to annotate.
        :return: The annotated variant.
        """
        # parse the site
        config = self.parse_variant(variant)

        if config is not None:
            i_ancestral, (p_ancestral, p_minor, p_major, pi) = self._get_ancestral_base(config)

            # only proceed if the ancestral allele is known
            if i_ancestral is not None:
                ancestral_base = bases[i_ancestral]

                if self.logger.level <= logging.DEBUG:
                    self.logger.debug(
                        f"ancestral base: {ancestral_base}, "
                        f"p_ancestral={p_ancestral:.4f}, "
                        f"p_minor={p_minor:.4f}, p_major={p_major:.4f}, "
                        f"pi={pi:.4f}, "
                        f"outgroup_bases={[bases[b] for b in config.outgroup_bases]}, "
                        f"n_major={config.n_major}, major_base={bases[config.major_base]}, "
                        f"minor_base={bases[config.minor_base] if not np.isnan(config.minor_base) else np.nan}, "
                        f"ref_base={variant.REF[0]}"
                    )

                # set the ancestral allele
                variant.INFO[self.annotator.info_ancestral] = ancestral_base

                # set info field for the probability of the ancestral allele
                variant.INFO[self.annotator.info_ancestral + "_info"] = (
                    f"p_ancestral={p_ancestral:.4f}, "
                    f"p_major={p_minor:.4f}, "
                    f"p_minor={p_major:.4f}"
                )

        # increase the number of annotated sites
        self.n_annotated += 1

    def plot_likelihoods(
            self,
            file: str = None,
            show: bool = True,
            title: str = 'rate likelihoods',
            scale: Literal['lin', 'log'] = 'lin',
            ax: plt.Axes = None,
            ylabel: str = 'lnl'
    ) -> plt.Axes:
        """
        Visualize the likelihoods of the rate optimization runs using a scatter plot.

        :param scale: y-scale of the plot.
        :param title: Plot title.
        :param file: File to save plot to.
        :param show: Whether to show plot.
        :param ax: Axes object to plot on.
        :param ylabel: Label for y-axis.
        :return: Axes object
        """
        return Visualization.plot_likelihoods(
            likelihoods=self.likelihoods,
            file=file,
            show=show,
            title=title,
            scale=scale,
            ax=ax,
            ylabel=ylabel,
        )

    def plot_polarisation(
            self,
            file: str = None,
            show: bool = True,
            title: str = 'ancestral allele probabilities',
            scale: Literal['lin', 'log'] = 'lin',
            ax: plt.Axes = None,
            ylabel: str = 'p'
    ) -> plt.Axes:
        """
        Visualize the polarization probabilities using a scatter plot.

        :param scale: y-scale of the plot.
        :param title: Plot title.
        :param file: File to save plot to.
        :param show: Whether to show plot.
        :param ax: Axes object to plot on.
        :param ylabel: y-axis label.
        :return: Axes object
        """
        return Visualization.plot_likelihoods(
            likelihoods=self.p_polarization,
            file=file,
            show=show,
            title=title,
            scale=scale,
            ax=ax,
            ylabel=ylabel
        )


class Annotator(VCFHandler):
    """
    Annotate a VCF file with the given annotations.
    """

    def __init__(
            self,
            vcf: str,
            output: str,
            annotations: List[Annotation],
            info_ancestral: str = 'AA',
            max_sites: int = np.inf,
            seed: int | None = 0,
            cache: bool = True
    ):
        """
        Create a new annotator instance.

        :param vcf: The path to the VCF file, can be gzipped, urls are also supported
        :param output: The path to the output file
        :param annotations: The annotations to apply.
        :param info_ancestral: The tag in the INFO field that contains the ancestral allele
        :param max_sites: Maximum number of sites to consider
        :param seed: Seed for the random number generator. Use ``None`` for no seed.
        :param cache: Whether to cache files downloaded from urls
        """
        super().__init__(
            vcf=vcf,
            info_ancestral=info_ancestral,
            max_sites=max_sites,
            seed=seed,
            cache=cache
        )

        self.output: str = output

        self.annotations: List[Annotation] = annotations

    def annotate(self):
        """
        Annotate the VCF file.
        """
        # count the number of sites
        self.n_sites = self.count_sites()

        # create the reader
        reader = VCF(self.download_if_url(self.vcf))

        # provide annotator to annotations and add info fields
        for annotation in self.annotations:
            annotation._setup(self, reader)

        # create the writer
        writer = Writer(self.output, reader)

        # get progress bar
        with self.get_pbar() as pbar:

            # iterate over the sites
            for i, variant in enumerate(reader):

                # apply annotations
                for annotation in self.annotations:
                    annotation.annotate_site(variant)

                # write the variant
                writer.write_record(variant)

                pbar.update()

                # explicitly stopping after ``n``sites fixes a bug with cyvcf2:
                # 'error parsing variant with `htslib::bcf_read` error-code: 0 and ret: -2'
                if i + 1 == self.n_sites or i + 1 == self.max_sites:
                    break

        # finalize annotations
        for annotation in self.annotations:
            annotation._teardown()

        # close the writer and reader
        writer.close()
        reader.close()
