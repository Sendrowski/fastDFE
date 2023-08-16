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
from typing import List, Optional, Dict, Tuple, Callable, Literal, Iterable, cast

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

bases = np.array(['G', 'A', 'T', 'C'])
codon_table = Bio.Data.CodonTable.standard_dna_table.forward_table
stop_codons = Bio.Data.CodonTable.standard_dna_table.stop_codons
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


class MaximumParsimonyAnnotation(AncestralAlleleAnnotation):
    """
    Annotation of ancestral alleles using maximum parsimony. The info field ``AA`` is added to the VCF file,
    which holds the ancestral allele. To be used with :class:`~fastdfe.parser.AncestralBaseStratification`.

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

    def __init__(
            self,
            bounds: Dict[str, Tuple[float, float]] = {}
    ):
        """
        Create a new substitution model instance.

        :param bounds: The bounds for the parameters.
        """
        # validate bounds
        self.validate_bounds(bounds)

        self.bounds = bounds

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

    def get_bounds(self, anc: 'OutgroupAncestralAlleleAnnotation') -> Dict[str, Tuple[float, float]]:
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

    @staticmethod
    @abstractmethod
    def get_prob(b1: int, b2: int, i: int, params: Dict[str, float]) -> float:
        """
        Get the probability of a branch using the substitution model.

        :param b1: First nucleotide state.
        :param b2: Second nucleotide state.
        :param i: The index of the branch.
        :param params: The parameters for the model.
        :return: The probability of the branch.
        """
        pass


class JCSubstitutionModel(SubstitutionModel):
    """
    Jukes-Cantor substitution model.
    """

    def __init__(
            self,
            bounds: Dict[str, Tuple[float, float]] = {'K': (1e-5, 10)}
    ):
        """
        Create a new substitution model instance.

        :param bounds: The bounds for the parameters.
        """
        super().__init__(bounds)

    def get_bounds(self, anc: 'OutgroupAncestralAlleleAnnotation') -> Dict[str, Tuple[float, float]]:
        """
        Get the bounds for the parameters.

        :param anc: The ancestral allele annotation instance.
        :return: The lower and upper bounds.
        """
        # get the bounds for the branch lengths
        return {f"K{i}": self.bounds["K"] for i in range(2 * anc.n_outgroups - 1)}

    @staticmethod
    def get_prob(b1: int, b2: int, i: int, params: Dict[str, float]) -> float:
        """
        Get the probability of a branch using the substitution model.

        :param b1: First nucleotide state.
        :param b2: Second nucleotide state.
        :param i: The index of the branch.
        :param params: The parameters for the model.
        :return: The probability of the branch.
        """
        # evolutionary rate parameter for the branch
        K = params["K" + str(i)]

        if b1 == b2:
            return np.exp(-K) + (1 / 6) * K ** 2 * np.exp(-K)

        return (1 / 3) * K * np.exp(-K) + (1 / 9) * K ** 2 * np.exp(-K)


class K2SubstitutionModel(SubstitutionModel):
    """
    Kimura 2-parameter substitution model.
    """

    def __init__(
            self,
            bounds: Dict[str, Tuple[float, float]] = {'K': (1e-5, 10), 'k': (0.02, 0.2)}
    ):
        """
        Create a new substitution model instance.

        :param bounds: The bounds for the parameters.
        """
        super().__init__(bounds)

    def get_bounds(self, anc: 'OutgroupAncestralAlleleAnnotation') -> Dict[str, Tuple[float, float]]:
        """
        Get the bounds for the parameters.

        :param anc: The ancestral allele annotation instance.
        :return: The lower and upper bounds.
        """
        # get the bounds for the branch lengths
        bounds = {f"K{i}": self.bounds["K"] for i in range(2 * anc.n_outgroups - 1)}

        # get the bounds for the K parameter
        bounds["k"] = self.bounds["k"]

        return bounds

    @staticmethod
    def get_prob(b1: int, b2: int, i: int, params: Dict[str, float]) -> float:
        """
        Get the probability of a branch using the K2 model.

        :param b1: First nucleotide state.
        :param b2: Second nucleotide state.
        :param i: The index of the branch.
        :param params: The parameters for the model.
        :return: The probability of the branch.
        """
        # evolutionary rate parameter for the branch
        K = params["K" + str(i)]

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

    #: The major allele.
    major_base: np.int8

    #: The minor allele.
    minor_base: np.int8

    #: The outgroup bases.
    outgroup_bases: np.ndarray[np.int8]

    # The probability of the major allele being ancestral.
    p_ancestral: np.float64

    # The probability of the minor allele.
    p_minor: np.float64

    # The probability of the major allele.
    p_major: np.float64


class BaseType(Enum):
    """
    The base type.
    """
    MINOR = 0
    MAJOR = 1


class OutgroupAncestralAlleleAnnotation(AncestralAlleleAnnotation):
    """
    TODO coming soon, currently being implemented
    TODO solve pickling issue with multiprocessing
    TODO vectorize computations

    Annotation of ancestral alleles using a sophisticated method similar to EST-SFS. The info field ``AA``
    is added to the VCF file, which holds the ancestral allele. To be used with
    :class:`~fastdfe.parser.AncestralBaseStratification`.
    """

    #: Bases
    bases = ['A', 'C', 'G', 'T']

    #: Base indices
    base_indices = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

    # the data types for the data frame
    dtypes = dict(
        n_major=np.int8,
        multiplicity=np.int16,
        sites=object,
        major_base=np.int8,
        minor_base=np.int8,
        outgroup_bases=object,
        p_ancestral=np.float64,
        p_minor=np.float64,
        p_major=np.float64
    )

    def __init__(
            self,
            outgroups: List[str],
            n_ingroups: int,
            ingroups: List[str] = None,
            n_outgroups: int = 2,
            n_runs_rate: int = 10,
            n_runs_polarization: int = 1,
            model: SubstitutionModel = K2SubstitutionModel(),
            parallelize: bool = False,
            use_prior: bool = True
    ):
        """
        Create a new ancestral allele annotation instance.

        :param outgroups: The outgroup samples to consider when determining the ancestral allele. A list of
            sample names as they appear in the VCF file.
        :param n_ingroups: The minimum number of ingroups that must be present at a site for it to be considered
            for ancestral allele inference. Note that a larger number of ingroups does not necessarily improve
            the accuracy of the ancestral allele inference as we infer the probability of the ancestral allele
            being the major allele for each minor allele count. A larger number of ingroups can thus lead to
            a large variance in the ancestral allele probability across minor allele counts, so it should only
            be increased if the number of sites used for the inference is large.
        :param ingroups: The ingroup samples to consider when determining the ancestral allele. If ``None``,
            all (non-outgroup) samples are considered. A list of sample names as they appear in the VCF file.
        :param n_outgroups: The minimum number of outgroups that must be present at a site for it to be considered
            for ancestral allele inference. More outgroups lead to a more accurate inference of the ancestral
            allele, but also increase the computational cost considerably.
        :param n_runs_rate: The number of runs to perform when determining the rate parameters.
        :param n_runs_polarization: The number of runs to perform when determining the polarization parameters. One
            run should be sufficient.
        :param parallelize: Whether to parallelize the computation.
        :param use_prior: Whether to incorporate information about the general probability of the major allele
            being the ancestral allele across all sites with the same minor allele count. This is useful in general
            as it provides more information about the ancestral allele, but it can lead to a bias if the number of sites
            is small.
        """
        super().__init__()

        #: Whether to parallelize the computation.
        self.parallelize: bool = parallelize

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

        #: The likelihoods for the different runs.
        self.likelihoods: np.ndarray[float, (...,)] | None = None

        #: The best (negative log) likelihood.
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

        # create mask for outgroups
        self.outgroup_mask = np.isin(samples, self.outgroups)

    def subsample(self, bases: np.ndarray, size: int) -> np.ndarray:
        """
        Subsample a set of bases.

        :param bases: A list of bases.
        :param size: The size of the subsample.
        :return: A subsample of the bases.
        """
        return bases[self.annotator.rng.choice(bases.shape[0], size=size, replace=False)]

    def parse_variant(self, variant: Variant) -> SiteConfig | None:
        """
        Parse a site.

        :param variant: The variant.
        :return: The site configuration.
        """
        # get the called ingroup and outgroup bases
        ingroups = get_called_bases(variant.gt_bases[self.ingroup_mask])
        outgroups = get_called_bases(variant.gt_bases[self.outgroup_mask])

        # get the numer of called ingroup and outgroup bases
        n_ingroups = len(ingroups)
        n_outgroups = len(outgroups)

        # only consider sites with enough ingroups and outgroups
        if n_ingroups >= self.n_ingroups and n_outgroups >= self.n_outgroups:

            # subsample ingroups and outgroups
            subsample_ingroups = self.subsample(ingroups, size=self.n_ingroups)
            subsample_outgroups = self.subsample(outgroups, size=self.n_outgroups)

            # get the counts of ingroups and outgroups
            counts_ingroups = Counter(subsample_ingroups)

            # only consider sites where the ingroups are at most bi-allelic
            if len(counts_ingroups) <= 2:

                # create site configuration
                site = SiteConfig()

                # get the major and minor allele
                most_common = counts_ingroups.most_common()

                # take the most common allele as the major allele
                site['major_base'] = self.base_indices[most_common[0][0]]

                # get the minor allele and its count
                site['n_major'] = most_common[0][1]

                # get the bases
                bases: List[str] = list(counts_ingroups.keys())

                # take the other allele as the minor allele
                if len(counts_ingroups) == 2:
                    site['minor_base'] = self.base_indices[bases[0] if bases[0] != most_common[0][0] else bases[1]]
                else:
                    site['minor_base'] = -1

                # get the outgroup alleles
                site['outgroup_bases'] = [self.base_indices[base] for base in subsample_outgroups]

                # set initial multiplicity
                site['multiplicity'] = 1

                return site

        return None

    @classmethod
    def from_est_sfs(
            cls,
            file: str,
            use_prior: bool = True,
            n_runs_rate: int = 10,
            n_runs_polarization: int = 1,
            model: SubstitutionModel = K2SubstitutionModel(),
            parallelize: bool = False,
            seed: int = 0,
            chunk_size: int = 100000
    ) -> 'OutgroupAncestralAlleleAnnotation':
        """
        Create from EST-SFS input file.

        :param file: File containing input data.
        :param use_prior: Whether to use the prior.
        :param n_runs_rate: Number of runs for rate estimation.
        :param n_runs_polarization: Number of runs for polarization.
        :param model: The substitution model.
        :param parallelize: Whether to parallelize.
        :param seed: The seed.
        :param chunk_size: The chunk size for reading the file.
        :return: The instance.
        """
        # define an empty dataframe to accumulate the data
        data = None
        n_ingroups = 0
        group_cols = None

        # iterate over the file in chunks
        for chunk in pd.read_csv(file, sep=r"\s+", header=None, dtype=str, chunksize=chunk_size):

            # extract the number of outgroups
            n_outgroups = chunk.shape[1] - 1

            # column indices that are grouped
            # we group by all columns
            group_cols = list(range(chunk.shape[1]))

            # retain site index
            chunk['sites'] = chunk.index

            # group by all columns in the chunk and keep track of the site indices
            grouped = chunk.groupby(group_cols, as_index=False).agg(list)

            # the first column contains the ingroup counts, split them
            ingroup_data = grouped[0].str.split(',', expand=True).astype(np.int8).to_numpy()

            # extract the number of ingroups samples
            n_ingroups = ingroup_data[0].sum()

            # determine the number of major alleles per site
            grouped['n_major'] = ingroup_data.max(axis=1)

            # sort by the number of alleles
            data_sorted = ingroup_data.argsort(axis=1)

            # determine the number of major alleles per site
            grouped['major_base'] = data_sorted[:, -1]

            # determine the mono-allelic sites
            mono_allelic = (ingroup_data > 1).sum(axis=1) == 1

            # determine the minor alleles
            minor_bases = np.zeros(grouped.shape[0], dtype=np.int8)
            minor_bases[mono_allelic] = -1
            minor_bases[~mono_allelic] = data_sorted[:, -2][~mono_allelic]

            # assign the minor alleles
            grouped['minor_base'] = minor_bases

            # extract outgroup data
            outgroup_data = np.full((grouped.shape[0], n_outgroups), -1, dtype=np.int8)
            for i in range(n_outgroups):
                # get the genotypes
                genotypes = grouped[i + 1].str.split(',', expand=True).astype(np.int8).to_numpy()

                # determine whether the site has an outgroup
                has_outgroup = genotypes.sum(axis=1) > 0

                # determine the outgroup allele indices provided the site has an outgroup
                outgroup_data[has_outgroup, i] = genotypes[has_outgroup].argmax(axis=1)

            # assign the outgroup data, convert to tuples for hashing
            grouped['outgroup_bases'] = [tuple(row) for row in outgroup_data]

            # drop unnecessary columns
            grouped.drop(columns=group_cols, inplace=True)

            if data is None:
                data = grouped
            else:
                # the new columns to gr
                group_new = ['major_base', 'minor_base', 'outgroup_bases', 'n_major']

                # If accumulator already has data, then merge with the grouped data from the current chunk
                data = pd.concat([data, grouped]).groupby(group_new, as_index=False).sum()

        # check if there is data
        if data is None:
            raise ValueError("No data found.")

        # determine the multiplicity
        data['multiplicity'] = data['sites'].apply(lambda x: len(x))

        # create from dataframe
        return OutgroupAncestralAlleleAnnotation.from_dataframe(
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
            parallelize: bool = False,
            use_prior: bool = True,
            seed: int = 0,
            pass_indices: bool = False
    ) -> 'OutgroupAncestralAlleleAnnotation':
        """
        Create an instance from data.

        :param n_major: The number of major alleles per site.
        :param major_bases: The major allele per site. A string representation of the base or the base index if
            ``pass_indices`` is True.
        :param minor_bases: The minor allele per site. A string representation of the base or the base index if
            ``pass_indices`` is True.
        :param outgroup_bases: The outgroup alleles per site. Note that the outgroup alleles all have to be the same
            length. A string representation of the base or the base index if ``pass_indices`` is True.
        :param n_ingroups: The number of ingroups samples.
        :param n_runs_rate: The number of runs for rate estimation.
        :param n_runs_polarization: The number of runs for polarization.
        :param model: The substitution model.
        :param parallelize: Whether to parallelize the runs.
        :param use_prior: Whether to use the prior.
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
            major_bases = [cls.base_indices[b] for b in major_bases]
            minor_bases = [cls.base_indices[b] for b in minor_bases]
            outgroup_bases = [[cls.base_indices[b] for b in c] for c in outgroup_bases]

        # create data frame
        data = pd.DataFrame({
            'n_major': n_major,
            'major_base': major_bases,
            'minor_base': minor_bases,
            'outgroup_bases': outgroup_bases
        })

        # create from dataframe
        return OutgroupAncestralAlleleAnnotation.from_dataframe(
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
            parallelize: bool = False,
            use_prior: bool = True,
            seed: int = 0,
            grouped: bool = False
    ) -> 'OutgroupAncestralAlleleAnnotation':
        """
        Create an instance from a dataframe.

        :param data: Dataframe with the columns: major_base, minor_base, outgroup_bases, n_major, possibly grouped
            by all columns.
        :param n_ingroups: The number of ingroups.
        :param n_runs_rate: Number of runs for rate estimation.
        :param n_runs_polarization: Number of runs for polarization.
        :param model: The substitution model.
        :param parallelize: Whether to parallelize.
        :param use_prior: Whether to use the prior.
        :param seed: The seed.
        :param grouped: Whether the dataframe is already grouped by all columns.
        :return: The instance.
        """
        # check if dataframe is empty
        if data.empty:
            raise ValueError("Empty dataframe.")

        if not grouped:
            # the new columns to gr
            group_cols = ['major_base', 'minor_base', 'outgroup_bases', 'n_major']

            # only keep the columns that are needed
            data = data[group_cols]

            # retain site index
            data['sites'] = data.index

            # convert outgroup bases to tuples
            data['outgroup_bases'] = data['outgroup_bases'].apply(tuple)

            # group by all columns in the chunk and keep track of the site indices
            data = data.groupby(group_cols, as_index=False).agg(list)

        # determine the multiplicity
        data['multiplicity'] = data['sites'].apply(lambda x: len(x))

        # add missing columns with NaN as default value
        for col in cls.dtypes:
            if col not in data.columns:
                data[col] = np.nan

        # convert to the correct dtypes
        data.astype(cls.dtypes, copy=False)

        anc = OutgroupAncestralAlleleAnnotation(
            n_runs_rate=n_runs_rate,
            n_runs_polarization=n_runs_polarization,
            model=model,
            parallelize=parallelize,
            use_prior=use_prior,
            outgroups=[],
            n_outgroups=len(data.outgroup_bases[0]),
            ingroups=[],
            n_ingroups=n_ingroups
        )

        # notify about the number of sites
        anc.logger.info(f"Loaded {data.shape[0]} sites.")

        # warn if few sites
        if data.shape[0] < 1000:
            anc.logger.warning(f"In order to obtain reliable results, at least 1000 sites are recommended. "
                               f"Consider increasing the number of sites to be included in the analysis.")

        # assign data frame
        anc.configs = data

        # assign number of sites
        anc.n_sites = data.multiplicity.sum()

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

        # set index to all initial site information
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
                        int(site.major_base),
                        int(site.minor_base),
                        tuple(site.outgroup_bases),
                        int(site.n_major)
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
        self.logger.info(f"Included {self.n_sites} sites for the inference.")

        # warn if few sites
        if self.n_sites < 1000:
            self.logger.warning(f"In order to obtain reliable results, at least 1000 sites are recommended. "
                                f"Consider increasing the number of sites to be included in the analysis. "
                                f"Alternative, consider decreasing min_ingroups and min_outgroups to "
                                f"include more sites.")

    def infer(self):
        """
        Infer the ancestral allele.
        """
        # get the bounds
        bounds = self.model.get_bounds(self)

        # set the parameter names
        self.param_names = list(bounds.keys())

        def optimize_rates(x0: Dict[str, float]) -> OptimizeResult:
            """
            Optimize the likelihood function for a single run.

            :param x0: The initial values.
            :return: The optimization results.
            """
            # optimize using scipy
            return minimize(
                fun=self.get_likelihood_rates(),
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
        self.likelihood = -np.min(self.likelihoods)

        # get the best result
        self.result = results[np.argmin(self.likelihoods)]

        # check if the optimization was successful
        if not self.result.success:
            raise RuntimeError(f"Optimization failed with message: {self.result.message}")

        # get dictionary of MLE parameters
        self.params_mle = dict(zip(self.param_names, self.result.x))

        # check bounds
        self.check_bounds(params=self.params_mle, bounds=bounds)

        # obtain the probability for each site and allele type (major/minor) under the MLE rate parameters
        self.configs.p_minor = self.get_p_sites(BaseType.MINOR, self.params_mle)
        self.configs.p_major = self.get_p_sites(BaseType.MAJOR, self.params_mle)
        self.configs.p_ancestral = self.calculate_p_ancestral(
            self.configs.p_minor,
            self.configs.p_major,
            self.configs.n_major
        )

    @cached_property
    def p_polarization(self) -> np.ndarray[float, (...,)]:
        """
        Get the polarization probabilities.
        """

        def optimize_polarization(args) -> OptimizeResult:
            """
            Optimize the likelihood function for a single run.

            :param args: The arguments.
            :return: The optimization results.
            """
            # unpack arguments
            i, _, x0 = args

            # optimize using scipy
            return minimize(
                fun=self.get_likelihood_polarization(i),
                x0=np.array([x0]),
                bounds=[(0, 1)],
                method="L-BFGS-B"
            )

        # prepare arguments
        data = np.array(list(itertools.product(range(1, self.n_ingroups // 2 + 2), range(self.n_runs_polarization))))

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
            raise RuntimeError("Polarization optimizations failed with messages: " + ", ".join(failures))

            # get the likelihoods for each frequency bin
        return np.array([results[i, j].x[0] for i, j in enumerate(i_best)])

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
    def get_p_site(
            site: SiteConfig,
            base_type: BaseType,
            params: Dict[str, float],
            model: SubstitutionModel = K2SubstitutionModel()
    ) -> float:
        """
        Get the probability for a site.

        TODO vectorize from here

        :param site: The site information.
        :param base_type: The base type.
        :param params: The parameters for the substitution model.
        :param model: The substitution model to use.
        :return: The probability for a site.
        """
        n_outgroups = int(np.sum(np.array(site.outgroup_bases) >= 0))

        # if there are no outgroups, return 1
        if n_outgroups == 0:
            return 1

        # probability for each tree
        p_trees = np.zeros(4 ** (n_outgroups - 1), dtype=float)

        # iterator over all possible internal node combinations
        for i, internal_nodes in enumerate(itertools.product(range(4), repeat=n_outgroups - 1)):
            # get the probability of the tree
            p_trees[i] = OutgroupAncestralAlleleAnnotation.get_p_tree(
                base=site.major_base if base_type == BaseType.MAJOR else site.minor_base,
                n_outgroups=n_outgroups,
                internal_nodes=internal_nodes,
                outgroup_bases=site.outgroup_bases,
                params=params,
                model=model
            )

        return p_trees.sum()

    def get_p_sites(
            self,
            base_type: BaseType,
            params: Dict[str, float]
    ) -> np.ndarray[float, (...,)]:
        """
        Get the probabilities for each site.

        :param base_type: The base type.
        :param params: A dictionary of the rate parameters.
        :return: The probability for each site.
        """
        # the probabilities for each site
        p_sites = np.zeros(shape=(self.configs.shape[0]), dtype=float)

        # iterate over the sites
        for i, config in self.configs.iterrows():
            # get the log likelihood of the site
            p_sites[i] = OutgroupAncestralAlleleAnnotation.get_p_site(
                site=cast(SiteConfig, config),
                base_type=base_type,
                params=params,
                model=self.model
            ) * config.multiplicity

        return p_sites

    def evaluate_likelihood_rates(self, params: Dict[str, float]) -> float:
        """
        Evaluate the likelihood function given a dictionary of parameters.

        :param params: A dictionary of parameters.
        :return: The log likelihood.
        """
        return -self.get_likelihood_rates()([params[name] for name in self.param_names])

    def get_likelihood_rates(self) -> Callable[[List[float]], float]:
        """
        Get the likelihood function.

        :return: The likelihood function.
        """

        def compute_likelihood(params: List[float]) -> float:
            """
            Compute the negative log likelihood of the parameters.

            :param params: A list of rate parameters.
            :return: The negative log likelihood.
            """
            params = dict(zip(self.param_names, params))

            # the likelihood for each site
            p_sites = np.zeros(shape=(self.configs.shape[0], 2), dtype=float)

            # get the probability for each site
            p_sites[:, 0] = self.get_p_sites(BaseType.MAJOR, params)
            p_sites[:, 1] = self.get_p_sites(BaseType.MINOR, params)

            # return the negative log likelihood and take average over major and minor bases
            return -np.log(p_sites.mean(axis=1)).sum()

        return compute_likelihood

    def get_likelihood_polarization(self, i: int) -> Callable[[List[float]], float]:
        """
        Get the likelihood function.

        :param i: The ith frequency bin.
        The likelihood function evaluated for the ith frequency bin.
        """

        def compute_likelihood(params: List[float]) -> float:
            """
            Compute the negative log likelihood of the parameters.

            :param params: The probability of polarization.
            :return: The negative log likelihood.
            """
            # get the probability of polarization for the ith frequency bin
            pi = params[0]

            # mask for sites that have i minor alleles
            i_minor = self.n_ingroups - self.configs.n_major == i

            # weight the sites by the probability of polarization
            p_sites = pi * self.configs.p_major[i_minor] + (1 - pi) * self.configs.p_minor[i_minor]

            # return the negative log likelihood
            return -np.log(p_sites).sum()

        return compute_likelihood

    def get_probs(self) -> np.ndarray[float, (...,)]:
        """
        Get the probabilities for the ancestral allele being the major allele for each site.

        :return: The probabilities
        """
        p1 = self.configs.p_major.to_numpy()
        p2 = self.configs.p_minor.to_numpy()

        if self.use_prior:
            # polarization prior for the major allele for the ith frequency bin
            pi = self.p_polarization[self.n_ingroups - self.configs.n_major]

            # get the probability that the major allele is ancestral
            return pi * p1 / (pi * p1 + (1 - pi) * p2)

        # get the probability that the major allele is ancestral
        return p1 / (p1 + p2)

    def get_site_indices(self) -> np.ndarray[int]:
        """
        Get the list of config indices for each site.

        :return: The list of config indices.
        """
        indices = np.full(self.n_sites, -1, dtype=int)

        for i, config in self.configs.iterrows():
            for j in config.sites:
                indices[j] = i

        return indices

    def get_sfs(self) -> Spectrum:
        """
        Get the site frequency spectrum for the sites used to estimate the parameters.

        :return: Spectrum object.
        """
        sfs = np.zeros(self.n_ingroups + 1, dtype=float)

        # get config indices for each site
        indices = self.get_site_indices()

        # iterate over the sites
        for i, i_config in enumerate(indices):
            if self.configs.p_ancestral[i_config] > 0.5:
                sfs[self.configs.n_major[i_config]] += 1
            else:
                sfs[self.n_ingroups - self.configs.n_major[i_config]] += 1

        return Spectrum(sfs)

    def get_ancestral_allele(
            self,
            site: SiteConfig
    ) -> (int, (float, float, float, float)):
        """
        Get the ancestral allele for each site.

        :param site: The site information.
        :return: The ancestral allele and a tuple of probability for the major being ancestral, the first base being
        ancestral, the second base being ancestral, and the polarization probability if using a prior.
        """
        # get the probability for the major allele
        p_minor = self.get_p_site(
            site=site,
            base_type=BaseType.MINOR,
            params=self.params_mle,
            model=self.model
        )

        # get the probability for the minor allele
        p_major = self.get_p_site(
            site=site,
            base_type=BaseType.MAJOR,
            params=self.params_mle,
            model=self.model
        )

        # get the probability of the major allele being ancestral
        p_ancestral = self.calculate_p_ancestral(p_minor, p_major, site.n_major)

        # determine the ancestral allele
        ancestral_base = site.major_base if p_ancestral > 0.5 else site.minor_base

        # polarization prior for the major allele for the ith frequency bin
        pi = self.p_polarization[self.n_ingroups - site.n_major]

        return ancestral_base, (p_ancestral, p_minor, p_major, pi)

    def calculate_p_ancestral(
            self,
            p_minor: float | np.ndarray[float],
            p_major: float | np.ndarray[float],
            n_major: int | np.ndarray[int]
    ) -> float:
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
            return pi * p_minor / (pi * p_minor + (1 - pi) * p_major)

        # get the probability that the major allele is ancestral
        return p_minor / (p_minor + p_major)

    def annotate_site(self, variant: Variant):
        """
        Annotate a single site.

        :param variant: The variant to annotate.
        :return: The annotated variant.
        """
        # parse the site
        site = self.parse_variant(variant)

        if site is not None:
            i_ancestral, (p_ancestral, p1, p2, pi) = self.get_ancestral_allele(site)

            ancestral_allele = self.bases[i_ancestral]

            self.logger.debug(
                f"ancestral allele: {ancestral_allele}, "
                f"p_ancestral={p_ancestral:.4f}, "
                f"p_major={p1:.4f}, p_minor={p2:.4f}, "
                f"pi={pi:.4f}, "
                f"outgroup_bases={[self.bases[b] for b in site.outgroup_bases]}, "
                f"n_major={site.n_major}, major_base={self.bases[site.major_base]}, "
                f"minor_base={self.bases[site.minor_base]}, "
                f"ref_base={variant.REF[0]}"
            )

            # set the ancestral allele
            variant.INFO[self.annotator.info_ancestral] = ancestral_allele

            # set info field for the probability of the ancestral allele
            variant.INFO[self.annotator.info_ancestral + "_info"] = (f"p_ancestral={p_ancestral:.4f}, "
                                                                     f"p_major={p1:.4f}, p_minor={p2:.4f}")

        # increase the number of annotated sites
        self.n_annotated += 1

    def plot_likelihoods(
            self,
            file: str = None,
            show: bool = True,
            title: str = 'rate likelihoods',
            scale: Literal['lin', 'log'] = 'lin',
            ax: plt.Axes = None,
            ylabel: str = 'lnl',
            **kwargs
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

    def plot_p_polarisation(
            self,
            file: str = None,
            show: bool = True,
            title: str = 'ancestral allele probabilities',
            scale: Literal['lin', 'log'] = 'lin',
            ax: plt.Axes = None,
            ylabel: str = 'p',
            **kwargs
    ) -> plt.Axes:
        """
        Visualize the probability of the major allele being ancestral for each frequency bin.

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

    def check_bounds(
            self,
            params: Dict[str, float],
            bounds: Dict[str, Tuple[float, float]],
            percentile: float = 1
    ) -> None:
        """
        Check if the given parameters are within the bounds.

        :param params: Parameters
        :param bounds: Bounds
        :param percentile: Percentile of the bounds to check
        :return: Whether the parameters are within the bounds
        """
        near_lower, near_upper = check_bounds(
            params=params,
            bounds=bounds,
            percentile=percentile
        )

        if len(near_lower | near_upper) > 0:
            self.logger.warning(f'The MLE estimate for the rates is near the upper bound for '
                                f'{near_upper} and lower bound for {near_lower}. Consider '
                                f'increasing the bounds or review the data if the problem persists.')


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
