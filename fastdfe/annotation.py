"""
VCF annotations and an annotator to apply them.
"""

__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"
__date__ = "2023-05-09"

import itertools
import logging
import re
import subprocess
import tempfile
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from functools import cached_property
from io import StringIO
from itertools import product
from typing import List, Optional, Dict, Tuple, Callable, Literal, Iterable, cast, Any, Generator

import Bio.Data.CodonTable
import numpy as np
import pandas as pd
from Bio import Phylo
from Bio.Phylo.BaseTree import Clade, Tree
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from cyvcf2 import Variant, Writer, VCF
from matplotlib import pyplot as plt
from scipy.optimize import minimize, OptimizeResult
from scipy.stats import hypergeom
from tqdm import tqdm

from .io_handlers import DummyVariant, MultiHandler, FASTAHandler
from .io_handlers import GFFHandler, get_major_base, get_called_bases
from .optimization import parallelize as parallelize_func, check_bounds
from .settings import Settings
from .spectrum import Spectra
from .visualization import Visualization

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

# include stop codons in codon table
for c in stop_codons:
    codon_table[c] = 'Σ'

# The degeneracy of the site according to how many unique amino acids
# are coding for when changing the site within the codon.
# We count the third position of the isoleucine codon as 2-fold degenerate.
# This is the only site that would normally have 3-fold degeneracy
# (https://en.wikipedia.org/wiki/Codon_degeneracy)
unique_to_degeneracy = {0: 0, 1: 2, 2: 2, 3: 4}


class Annotation(ABC):

    def __init__(self):
        """
        Create a new annotation instance.
        """
        #: The logger.
        self._logger = logger.getChild(self.__class__.__name__)

        #: The annotator.
        self._handler: Annotator | None = None

        #: The number of annotated sites.
        self.n_annotated: int = 0

    def _setup(self, handler: MultiHandler):
        """
        Provide context by passing the annotator. This should be called before the annotation starts.

        :param handler: The handler.
        """
        self._handler = handler

    def _rewind(self):
        """
        Rewind the annotation.
        """
        self.n_annotated = 0

    def _teardown(self):
        """
        Finalize the annotation. Called after all sites have been annotated.
        """
        self._logger.info(f'Annotated {self.n_annotated} sites.')

    @abstractmethod
    def annotate_site(self, variant: Variant | DummyVariant):
        """
        Annotate a single site.

        :param variant: The variant to annotate.
        :return: The annotated variant.
        """
        pass

    @staticmethod
    def count_target_sites(file: str, remove_overlaps: bool = False, contigs: List[str] = None) -> Dict[str, int]:
        """
        Count the number of target sites in a GFF file.

        :param file: The path to The GFF file path, possibly gzipped or a URL
        :param remove_overlaps: Whether to remove overlapping target sites.
        :param contigs: The contigs to count the target sites for.
        :return: The number of target sites per chromosome/contig.
        """
        return GFFHandler(file)._count_target_sites(
            remove_overlaps=remove_overlaps,
            contigs=contigs
        )


class DegeneracyAnnotation(Annotation):
    """
    Degeneracy annotation. We annotate the degeneracy by looking at each codon for coding variants.
    This also annotates mono-allelic sites.

    This annotation adds the info fields ``Degeneracy`` and ``Degeneracy_Info``, which hold the degeneracy
    of a site (0, 2, 4) and extra information about the degeneracy, respectively. To be used with
    :class:`~fastdfe.parser.DegeneracyStratification`.

    For this annotation to work, we require a FASTA and GFF file (passed to :class:`~fastdfe.parser.Parser` or
    :class:`~fastdfe.annotation.Annotator`).

    Example usage:

    ::

        import fastdfe as fd

        ann = fd.Annotator(
            vcf="http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/"
                "1000_genomes_project/release/20181203_biallelic_SNV/"
                "ALL.chr21.shapeit2_integrated_v1a.GRCh38.20181129.phased.vcf.gz",
            fasta="http://ftp.ensembl.org/pub/release-109/fasta/homo_sapiens/"
                  "dna/Homo_sapiens.GRCh38.dna.chromosome.21.fa.gz",
            gff="http://ftp.ensembl.org/pub/release-109/gff3/homo_sapiens/"
                "Homo_sapiens.GRCh38.109.chromosome.21.gff3.gz",
            output='sapiens.chr21.degeneracy.vcf.gz',
            annotations=[fd.DegeneracyAnnotation()],
            aliases=dict(chr21=['21'])
        )

        ann.annotate()

    """

    #: The genomic positions for coding sequences that are mocked.
    _pos_mock: int = 1e100

    def __init__(self):
        """
        Create a new annotation instance.
        """
        Annotation.__init__(self)

        #: The current coding sequence or the closest coding sequence downstream.
        self._cd: Optional[pd.Series] = None

        #: The coding sequence following the current coding sequence.
        self._cd_next: Optional[pd.Series] = None

        #: The coding sequence preceding the current coding sequence.
        self._cd_prev: Optional[pd.Series] = None

        #: The current contig.
        self._contig: Optional[SeqRecord] = None

        #: The variants that could not be annotated correctly.
        self.mismatches: List[Variant] = []

        #: The variant that were skipped because they were not in coding regions.
        self.n_skipped = 0

        #: The variants for which the codon could not be determined.
        self.errors: List[Variant] = []

    def _setup(self, handler: MultiHandler):
        """
        Provide context to the annotator.

        :param handler: The handler.
        """
        # require FASTA and GFF files
        handler._require_fasta(self.__class__.__name__)
        handler._require_gff(self.__class__.__name__)

        # call super
        super()._setup(handler)

        # touch the cached properties to make for a nicer logging experience
        # noinspection PyStatementEffect
        self._handler._cds

        # noinspection PyStatementEffect
        self._handler._ref

        handler._reader.add_info_to_header({
            'ID': 'Degeneracy',
            'Number': '.',
            'Type': 'Integer',
            'Description': 'n-fold degeneracy'
        })

        handler._reader.add_info_to_header({
            'ID': 'Degeneracy_Info',
            'Number': '.',
            'Type': 'Integer',
            'Description': 'Additional information about degeneracy annotation'
        })

    def _rewind(self):
        """
        Rewind the annotation.
        """
        Annotation._rewind(self)

        self._cd = None
        self._cd_next = None
        self._cd_prev = None
        self._contig = None

    def _parse_codon_forward(self, variant: Variant | DummyVariant):
        """
        Parse the codon in forward direction.

        :param variant: The variant.
        :return: Codon, Codon position, Codon start position, Position within codon, and relative position.
        """
        # position relative to start of coding sequence
        pos_rel = variant.POS - (self._cd.start + int(self._cd.phase))

        # position relative to codon
        pos_codon = pos_rel % 3

        # inclusive codon start, 1-based
        codon_start = variant.POS - pos_codon

        # the codon positions
        codon_pos = [codon_start, codon_start + 1, codon_start + 2]

        if (self._cd_prev is None or self._cd_next.start == self._pos_mock) and codon_pos[0] < self._cd.start:
            raise IndexError(f'Codon at site {variant.CHROM}:{variant.POS} overlaps with '
                             f'start position of current CDS and no previous CDS was given.')

        # Use final positions from previous coding sequence if current codon
        # starts before start position of current coding sequence
        if codon_pos[1] == self._cd.start:
            codon_pos[0] = self._cd_prev.end if self._cd_prev.strand == '+' else self._cd_prev.start
        elif codon_pos[2] == self._cd.start:
            codon_pos[1] = self._cd_prev.end if self._cd_prev.strand == '+' else self._cd_prev.start
            codon_pos[0] = self._cd_prev.end - 1 if self._cd_prev.strand == '+' else self._cd_prev.start + 1

        if (self._cd_next is None or self._cd_next.start == self._pos_mock) and codon_pos[2] > self._cd.end:
            raise IndexError(f'Codon at site {variant.CHROM}:{variant.POS} overlaps with '
                             f'end position of current CDS and no subsequent CDS was given.')

        # use initial positions from subsequent coding sequence if current codon
        # ends before end position of current coding sequence
        if codon_pos[2] == self._cd.end + 1:
            codon_pos[2] = self._cd_next.start if self._cd_next.strand == '+' else self._cd_next.end
        elif codon_pos[1] == self._cd.end + 1:
            codon_pos[1] = self._cd_next.start if self._cd_next.strand == '+' else self._cd_next.end
            codon_pos[2] = self._cd_next.start + 1 if self._cd_next.strand == '+' else self._cd_next.end - 1

        # seq uses 0-based positions
        codon = ''.join([str(self._contig[int(pos - 1)]) for pos in codon_pos]).upper()

        return codon, codon_pos, codon_start, pos_codon, pos_rel

    def _parse_codon_backward(self, variant: Variant | DummyVariant):
        """
        Parse the codon in reverse direction.

        :param variant: The variant.
        :return: Codon, Codon position, Codon start position, Position within codon, and relative position.
        """
        # position relative to end of coding sequence
        pos_rel = (self._cd.end - int(self._cd.phase)) - variant.POS

        # position relative to codon end
        pos_codon = pos_rel % 3

        # inclusive codon start, 1-based
        codon_start = variant.POS + pos_codon

        # the codon positions
        codon_pos = [codon_start, codon_start - 1, codon_start - 2]

        if (self._cd_prev is None or self._cd_next.start == self._pos_mock) and codon_pos[2] < self._cd.start:
            raise IndexError(f'Codon at site {variant.CHROM}:{variant.POS} overlaps with '
                             f'start position of current CDS and no previous CDS was given.')

        # Use final positions from previous coding sequence if current codon
        # ends before start position of current coding sequence.
        if codon_pos[1] == self._cd.start:
            codon_pos[2] = self._cd_prev.end if self._cd_prev.strand == '-' else self._cd_prev.start
        elif codon_pos[0] == self._cd.start:
            codon_pos[1] = self._cd_prev.end if self._cd_prev.strand == '-' else self._cd_prev.start
            codon_pos[2] = self._cd_prev.end - 1 if self._cd_prev.strand == '-' else self._cd_prev.start + 1

        if (self._cd_next is None or self._cd_next.start == self._pos_mock) and codon_pos[0] > self._cd.end:
            raise IndexError(f'Codon at site {variant.CHROM}:{variant.POS} overlaps with '
                             f'end position of current CDS and no subsequent CDS was given.')

        # use initial positions from subsequent coding sequence if current codon
        # starts before end position of current coding sequence
        if codon_pos[0] == self._cd.end + 1:
            codon_pos[0] = self._cd_next.start if self._cd_next.strand == '-' else self._cd_next.end
        elif codon_pos[1] == self._cd.end + 1:
            codon_pos[1] = self._cd_next.start if self._cd_next.strand == '-' else self._cd_next.end
            codon_pos[0] = self._cd_next.start + 1 if self._cd_next.strand == '-' else self._cd_next.end - 1

        # we use 0-based positions here
        codon = ''.join(str(self._contig[int(pos - 1)]) for pos in codon_pos)

        # take complement and convert to uppercase ('n' might be lowercase)
        codon = str(Seq(codon).complement()).upper()

        return codon, codon_pos, codon_start, pos_codon, pos_rel

    def _parse_codon(self, variant: Variant | DummyVariant):
        """
        Parse the codon for the given variant.

        :param variant: The variant to parse the codon for.
        :return: Codon, Codon position, Codon start position, Position within codon, and relative position.
        """

        if self._cd.strand == '+':
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

    def _fetch_cds(self, v: Variant | DummyVariant):
        """
        Fetch the coding sequence for the given variant.

        :param v: The variant to fetch the coding sequence for.
        :raises LookupError: If no coding sequence was found.
        """
        # get the aliases for the current chromosome
        aliases = self._handler.get_aliases(v.CHROM)

        # only fetch coding sequence if we are on a new chromosome or the
        # variant is not within the current coding sequence
        if self._cd is None or self._cd.seqid not in aliases or v.POS > self._cd.end:

            # reset coding sequences to mocking positions
            self._cd_prev = None
            self._cd = pd.Series({'seqid': v.CHROM, 'start': self._pos_mock, 'end': self._pos_mock})
            self._cd_next = pd.Series({'seqid': v.CHROM, 'start': self._pos_mock, 'end': self._pos_mock})

            # filter for the current chromosome
            on_contig = self._handler._cds[(self._handler._cds.seqid.isin(aliases))]

            # filter for positions ending after the variant
            cds = on_contig[(on_contig.end >= v.POS)]

            if not cds.empty:
                # take the first coding sequence
                self._cd = cds.iloc[0]

                self._logger.debug(f'Found coding sequence: {self._cd.seqid}:{self._cd.start}-{self._cd.end}, '
                                   f'reminder: {(self._cd.end - self._cd.start + 1) % 3}, '
                                   f'phase: {int(self._cd.phase)}, orientation: {self._cd.strand}, '
                                   f'current position: {v.CHROM}:{v.POS}')

                # filter for positions ending after the current coding sequence
                cds = on_contig[(on_contig.start > self._cd.end)]

                if not cds.empty:
                    # take the first coding sequence
                    self._cd_next = cds.iloc[0]

                # filter for positions starting before the current coding sequence
                cds = on_contig[(on_contig.end < self._cd.start)]

                if not cds.empty:
                    # take the last coding sequence
                    self._cd_prev = cds.iloc[-1]

            if self._cd.start == self._pos_mock and self.n_annotated == 0:
                self._logger.warning(f"No coding sequence found on all of contig '{v.CHROM}' and no previous "
                                     f'sites were annotated. Are you sure that this is the correct GFF file '
                                     f'and that the contig names match the chromosome names in the VCF file? '
                                     f'Note that you can also specify aliases for contig names in the VCF file.')

        # check if variant is located within coding sequence
        if self._cd is None or not (self._cd.start <= v.POS <= self._cd.end):
            raise LookupError(f"No coding sequence found, skipping record {v.CHROM}:{v.POS}")

    def _fetch_contig(self, v: Variant | DummyVariant):
        """
        Fetch the contig for the given variant.

        :param v: The variant to fetch the contig for.
        """
        aliases = self._handler.get_aliases(v.CHROM)

        # check if contig is up-to-date
        if self._contig is None or self._contig.id not in aliases:
            self._logger.debug(f"Fetching contig '{v.CHROM}'.")

            # fetch contig
            self._contig = self._handler.get_contig(aliases)

    def _fetch(self, variant: Variant | DummyVariant):
        """
        Fetch all required data for the given variant.

        :param variant: The variant to fetch the data for.
        :raises LookupError: if some data could not be found.
        """
        self._fetch_cds(variant)

        try:
            self._fetch_contig(variant)
        except LookupError:
            # log error as this should not happen
            self._logger.warning(f"Could not fetch contig '{variant.CHROM}'.")
            raise

    def annotate_site(self, v: Variant | DummyVariant):
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
        if self._cd.seqid in self._handler.get_aliases(v.CHROM) and self._cd.start <= v.POS <= self._cd.end:

            try:
                # parse codon
                codon, codon_pos, codon_start, pos_codon, pos_rel = self._parse_codon(v)

            except IndexError as e:

                # skip site on IndexError
                self._logger.warning(e)
                self.errors.append(v)
                return

            # make sure the reference allele matches with the position on the reference genome
            if str(self._contig[v.POS - 1]).upper() != v.REF.upper():
                self._logger.warning(f"Reference allele does not match with reference genome at {v.CHROM}:{v.POS}.")
                self.mismatches.append(v)
                return

            degeneracy = '.'
            if 'N' not in codon:
                degeneracy = self._get_degeneracy(codon, pos_codon)

                # increment counter of annotated sites
                self.n_annotated += 1

            v.INFO['Degeneracy'] = degeneracy
            v.INFO['Degeneracy_Info'] = f"{pos_codon},{self._cd.strand},{codon}"

            self._logger.debug(f'pos codon: {pos_codon}, pos abs: {v.POS}, '
                               f'codon start: {codon_start}, codon: {codon}, '
                               f'strand: {self._cd.strand}, ref allele: {self._contig[v.POS - 1]}, '
                               f'degeneracy: {degeneracy}, codon pos: {str(codon_pos)}, '
                               f'ref allele: {v.REF}')


class SynonymyAnnotation(DegeneracyAnnotation):
    """
    Synonymy annotation. This class annotates a variant with the synonymous/non-synonymous status.

    This annotation adds the info fields ``Synonymous`` and ``Synonymous_Info``, which hold
    the synonymy (Synonymous [0] or non-synonymous [1]) and the codon information, respectively.
    To be used with :class:`~fastdfe.parser.SynonymyStratification`.

    For this annotation to work, we require a FASTA and GFF file (passed to :class:`~fastdfe.parser.Parser` or
    :class:`~fastdfe.annotation.Annotator`).

    This class was tested against `VEP <VEP_>`_ and `SnpEff <SnpEff_>`_ and provides the same annotations in almost
    all cases.

    .. _VEP: https://www.ensembl.org/info/docs/tools/vep/index.html
    .. _SnpEff: https://pcingola.github.io/SnpEff/

    .. warning::
        Not recommended for use with :class:`~fastdfe.parser.Parser` as we also need to annotate mono-allelic sites.
        Consider using :class:`~fastdfe.annotation.DegeneracyAnnotation` and
        :class:`~fastdfe.parser.DegeneracyStratification` instead.
    """

    def __init__(self):
        """
        Create a new annotation instance.
        """
        super().__init__()

        #: The number of sites that did not match with VEP.
        self.vep_mismatches: List[Variant] = []

        #: The number of sites that did not math with the annotation provided by SnpEff
        self.snpeff_mismatches: List[Variant] = []

        #: The number of sites that were concordant with VEP.
        self.n_vep_comparisons: int = 0

        #: The number of sites that were concordant with SnpEff.
        self.n_snpeff_comparisons: int = 0

    def _setup(self, handler: MultiHandler):
        """
        Provide context to the annotator.

        :param handler: The handler.
        """
        # require FASTA and GFF files
        handler._require_fasta(self.__class__.__name__)
        handler._require_gff(self.__class__.__name__)

        Annotation._setup(self, handler)

        # touch the cached properties to make for a nicer logging experience
        # noinspection PyStatementEffect
        self._handler._cds

        # noinspection PyStatementEffect
        self._handler._ref

        handler._reader.add_info_to_header({
            'ID': 'Synonymy',
            'Number': '.',
            'Type': 'Integer',
            'Description': 'Synonymous (0) or non-synonymous (1)'
        })

        handler._reader.add_info_to_header({
            'ID': 'Synonymy_Info',
            'Number': '.',
            'Type': 'String',
            'Description': 'Alt codon and extra information'
        })

    def _get_alt_allele(self, variant: Variant | DummyVariant) -> str | None:
        """
        Get the alternative allele.

        :param variant: The variant to get the alternative allele for.
        :return: The alternative allele or None if there is no alternative allele.
        """
        if len(variant.ALT) > 0:

            # assume there is at most one alternative allele
            if self._cd.strand == '-':
                return Seq(variant.ALT[0]).complement().__str__()

            return variant.ALT[0]

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

    def _parse_codons_vep(self, variant: Variant | DummyVariant) -> List[str]:
        """
        Parse the codons from the VEP annotation if present.

        :param variant: The variant.
        :return: The codons.
        """
        # match codons
        match = re.search("([actgACTG]{3})/([actgACTG]{3})", variant.INFO.get('CSQ'))

        if match is not None:
            if len(match.groups()) != 2:
                self._logger.info(f'VEP annotation has more than two codons: {variant.INFO.get("CSQ")}')

            return [m.upper() for m in [match[1], match[2]]]

        return []

    @staticmethod
    def _parse_synonymy_snpeff(variant: Variant | DummyVariant) -> int | None:
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

    def _teardown(self):
        """
        Finalize the annotation.
        """
        super()._teardown()

        if self.n_vep_comparisons != 0:
            self._logger.info(f'Number of mismatches with VEP: {len(self.vep_mismatches)}')

        if self.n_snpeff_comparisons != 0:
            self._logger.info(f'Number of mismatches with SnpEff: {len(self.snpeff_mismatches)}')

    def annotate_site(self, v: Variant | DummyVariant):
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
            if self._cd.start <= v.POS <= self._cd.end:

                try:
                    # parse codon
                    codon, codon_pos, codon_start, pos_codon, pos_rel = self._parse_codon(v)

                except IndexError as e:

                    # skip site on IndexError
                    self._logger.warning(e)
                    self.errors.append(v)
                    return

                # make sure the reference allele matches with the position in the reference genome
                if str(self._contig[v.POS - 1]).upper() != v.REF.upper():
                    self._logger.warning(f"Reference allele does not match with reference genome at {v.CHROM}:{v.POS}.")
                    self.mismatches.append(v)
                    return

                # fetch the alternative allele if present
                alt = self._get_alt_allele(v)

                info = ''
                synonymy = '.'
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
                                self._logger.warning(f'VEP codons do not match with codons determined by '
                                                     f'codon table for {v.CHROM}:{v.POS}')

                                self.vep_mismatches.append(v)
                                return

                if v.INFO.get('ANN') is not None:
                    synonymy_snpeff = self._parse_synonymy_snpeff(v)

                    self.n_snpeff_comparisons += 1

                    if synonymy_snpeff is not None:
                        if synonymy_snpeff != synonymy:
                            self._logger.warning(f'SnpEff annotation does not match with custom '
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

    def _setup(self, handler: MultiHandler):
        """
        Add info fields to the header.

        :param handler: The handler.
        """
        super()._setup(handler)

        handler._reader.add_info_to_header({
            'ID': self._handler.info_ancestral,
            'Number': '.',
            'Type': 'Character',
            'Description': 'Ancestral Allele'
        })


class MaximumParsimonyAncestralAnnotation(AncestralAlleleAnnotation):
    """
    Annotation of ancestral alleles using maximum parsimony. To be used with
    :class:`~fastdfe.parser.AncestralBaseStratification` and :class:`Annotator` or :class:`~fastdfe.parser.Parser`.

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

    def _setup(self, handler: MultiHandler):
        """
        Add info fields to the header.

        :param handler: The handler.
        """
        super()._setup(handler)

        # create mask for ingroups
        if self.samples is None:
            self.samples_mask = np.ones(len(handler._reader.samples)).astype(bool)
        else:
            self.samples_mask = np.isin(handler._reader.samples, self.samples)

    def annotate_site(self, variant: Variant | DummyVariant):
        """
        Annotate a single site.

        :param variant: The variant to annotate.
        :return: The annotated variant.
        """
        # assign the ancestral allele
        variant.INFO[self._handler.info_ancestral] = self._get_ancestral(variant, self.samples_mask)

        if variant.INFO[self._handler.info_ancestral] != '.':
            # increase the number of annotated sites
            self.n_annotated += 1

    @staticmethod
    def _get_ancestral(
            variant: Variant | DummyVariant,
            mask: np.ndarray,
    ) -> str:
        """
        Get the ancestral allele for the given variant using maximum parsimony.

        :param variant: The variant to annotate.
        :param mask: The mask for the ingroups.
        :return: The ancestral allele or '.' if it could not be determined.
        """
        # take reference allele as ancestral if dummy variant
        if isinstance(variant, DummyVariant):
            return variant.REF

        # take only base to be ancestral if we have a monomorphic snv
        if not variant.is_snp and variant.REF in bases:
            b = list(set(get_called_bases(variant.gt_bases[mask])))

            if len(b) == 1 and b[0] in bases:
                return b[0]

        # take major base to be ancestral if we have an SNP
        if variant.is_snp:
            return get_major_base(variant.gt_bases[mask]) or '.'

        return '.'


class SubstitutionModel(ABC):
    """
    Base class for substitution models.
    """

    #: The possible transitions
    _transitions: np.ndarray[int, (..., ...)] = np.array([
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
        #: The logger.
        self._logger = logging.getLogger(self.__class__.__name__)

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

        for (b1, b2, i) in itertools.product(range(0, 4), range(0, 4), range(n_branches)):
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

    def get_bounds(self, n_outgroups: int) -> Dict[str, Tuple[float, float]]:
        """
        Get the bounds for the parameters.

        :param n_outgroups: The number of outgroups.
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
            if lower <= 0:
                raise ValueError(f'All lower bounds must be positive, got {lower} for {param}.')

            if lower > upper:
                raise ValueError(f'Lower bounds must be smaller than upper bounds, got {lower} > {upper} for {param}.')

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

    def _get_cached_prob(self, b1: int, b2: int, i: int, params: Dict[str, float]) -> float:
        """
        Get the probability of a branch using the substitution model with caching.

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

    def get_bounds(self, n_outgroups: int) -> Dict[str, Tuple[float, float]]:
        """
        Get the bounds for the parameters.

        :param n_outgroups: The number of outgroups.
        :return: The lower and upper bounds.
        """
        if self.pool_branch_rates:
            # pool the branch rates
            return {'K': self.get_bound('K')}

        # get the bounds for the branch lengths
        return {f"K{i}": self.get_bound(f"K{i}") for i in range(2 * n_outgroups - 1)}

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
            bounds: Dict[str, Tuple[float, float]] = {'K': (1e-5, 10), 'k': (0.1, 10)},
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

    def get_bounds(self, n_outgroups: int) -> Dict[str, Tuple[float, float]]:
        """
        Get the bounds for the parameters.

        :param n_outgroups: The number of outgroups.
        :return: The lower and upper bounds.
        """
        bounds = super().get_bounds(n_outgroups)

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


@dataclass
class SiteConfig:
    """
    Ancestral allele site configuration for a single subsample.
    """

    #: The number of major alleles.
    n_major: int

    #: The major allele base index.
    major_base: int

    #: The minor base index.
    minor_base: int

    #: The outgroup base indices.
    outgroup_bases: np.ndarray[int]

    #: The multiplicity of the site.
    multiplicity: float = 1.0

    #: The site indices.
    sites: np.ndarray[int] = field(default_factory=lambda: np.array([]))

    # The probability of the minor allele.
    p_minor: np.float64 = np.nan

    # The probability of the major allele.
    p_major: np.float64 = np.nan


@dataclass
class SiteInfo:
    """
    Ancestral allele information on a single site.
    """

    #: Dictionary mapping number of major alleles to its probability of observation.
    n_major: Dict[int, float]

    #: The major allele base.
    major_base: str

    #: The minor base index.
    minor_base: str

    #: The outgroup base indices.
    outgroup_bases: List[str]

    #: The probability of the minor allele being the ancestral allele (without prior).
    p_minor: float = np.nan

    #: The probability of the major allele being the ancestral allele (without prior).
    p_major: float = np.nan

    #: The probability of the major allele being the ancestral allele rather than the minor allele
    #: (possibly with prior if specified).
    p_major_ancestral: float = np.nan

    #: The ancestral base based on comparing major and minor allele.
    major_ancestral: str = '.'

    #: The probability of each base being the ancestral base for the first internal node.
    p_bases_first_node: Dict[str, float] = field(default_factory=dict)

    #: The probability that the mostly likely base for the first internal node is the ancestral base.
    p_first_node_ancestral: float = np.nan

    #: The ancestral base index for the first internal node.
    first_node_ancestral: str = '.'

    #: The branch rates.
    rate_params: Dict[str, float] = field(default_factory=dict)

    def plot_tree(
            self,
            ax: plt.Axes = None,
            show: bool = True,
    ):
        """
        Plot the tree for a site. Only Python visualization is supported.

        :param self: The site information.
        :param ax: Axes to plot on.
        :param show: Whether to show the plot.
        """
        if ax is None:
            ax = plt.gca()

        if 'K' in self.rate_params:
            branch_lengths = {f'K{i}': self.rate_params['K'] for i in range(len(self.outgroup_bases) * 2 - 1)}
        else:
            branch_lengths = self.rate_params

        n_outgroups = len(self.outgroup_bases)

        # Create major, minor, and ingroup clades
        major_clade = Clade(name=self.major_base, branch_length=0)
        minor_clade = Clade(name=self.minor_base, branch_length=0)
        ingroup = Clade(
            name="ingroup",
            clades=[major_clade, minor_clade],
            branch_length=branch_lengths['K0'] if n_outgroups > 0 else 0
        )

        current = ingroup

        # Create and attach outgroup clades to major and minor clades
        for i in range(n_outgroups):

            # last outgroup has half the branch length to the root
            # as we have no internal node
            if i < n_outgroups - 1:
                outgroup_length = branch_lengths[f"K{2 * i + 1}"]
            else:
                outgroup_length = branch_lengths[f"K{2 * i}"] / 2

            # create outgroup clade
            outgroup = Clade(
                name=self.outgroup_bases[i],
                branch_length=outgroup_length
            )

            # determine the branch length to the next node
            if i < n_outgroups - 2:
                node_length = branch_lengths[f"K{2 * i + 2}"]
            elif i == n_outgroups - 2:
                node_length = branch_lengths[f"K{2 * i + 2}"] / 2
            else:
                node_length = 0

            # create internal node / root
            current = Clade(
                name=f"internal {i + 1}" if i < n_outgroups - 1 else None,
                clades=[outgroup, current],
                branch_length=node_length
            )

        # create a tree object and visualize
        tree = Tree(root=current)
        Phylo.draw(tree, axes=ax, do_show=False)

        # remove Y-axis
        ax.axes.get_yaxis().set_visible(False)

        # remove frame
        for pos in ['top', 'right', 'left']:
            ax.spines[pos].set_visible(False)

        if show:
            plt.show()


class BaseType(Enum):
    """
    The base type, either major or minor.
    """
    MINOR = 0
    MAJOR = 1


class PolarizationPrior(ABC):
    """
    Base class for priors to be used with the :class:`MaximumLikelihoodAncestralAnnotation`.
    Using a prior, we incorporate information about the general probability of the major allele being the ancestral 
    allele across all sites with the same minor allele count, is useful in general as it provides more information 
    about the ancestral allele probabilities For example, when we have no outgroup information for a particular site, 
    it is good to know how likely it is that the major allele is ancestral in general and incorporate this information 
    into the estimates. 
    """

    def __init__(self, allow_divergence: bool = False):
        """
        Create a new instance.

        :param allow_divergence: Whether to allow divergence. If ``True``, the probability of the minor allele
            being ancestral, which is not contained in the ingroup subsample but rather in all specified ingroups
            or among the outgroup, is taken to be the same as if it was present in the ingroup subsample with
            frequency 1. This is a hack, but allows us to consider alleles that are not present in the ingroup
            subsample.

            .. warning:: Setting this to ``True`` greatly increases the probability of high-frequency derived alleles
                which introduces a strong bias in the distribution of frequency counts, e.g., the SFS. Only use this
                if you're interested in the most accurate ancestral state per site.
        """
        #: The logger.
        self._logger = logger.getChild(self.__class__.__name__)

        #: Whether to allow divergence.
        self.allow_divergence: bool = allow_divergence

        #: The polarization probabilities.
        self.probabilities: np.ndarray[float, (...,)] | None = None

    def _add_divergence(self):
        """
        Add divergence to the polarization probabilities.
        """
        # take divergence probabilities to be the same as alleles of frequency 1
        if self.allow_divergence:
            self.probabilities[0] = self.probabilities[1]
            self.probabilities[-1] = self.probabilities[-2]

        else:
            # set divergence probabilities to 0
            self.probabilities[0] = 1
            self.probabilities[-1] = 0

    @abstractmethod
    def _get_prior(self, configs: pd.DataFrame, n_ingroups: int) -> np.ndarray:
        """
        Get the polarization probabilities.
        
        :param configs: The site configurations.
        :param n_ingroups: The number of ingroups.
        """
        pass

    def plot(
            self,
            file: str = None,
            show: bool = True,
            title: str = 'polarization probabilities',
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
        :param ax: Axes to plot on. Only for Python visualization backend.
        :param ylabel: y-axis label.
        :return: Axes object
        """
        if self.probabilities is None:
            raise ValueError('Polarization probabilities have not been calculated yet.')

        return Visualization.plot_scatter(
            values=self.probabilities,
            file=file,
            show=show,
            title=title,
            scale=scale,
            ax=ax,
            ylabel=ylabel
        )


class KingmanPolarizationPrior(PolarizationPrior):
    """
    Prior based on the standard Kingman coalescent. To be used with 
    :class:`MaximumLikelihoodAncestralAnnotation`.
    """

    def _get_prior(self, configs: pd.DataFrame, n_ingroups: int) -> np.ndarray:
        """
        Get the polarization probabilities.

        :param configs: The site configurations.
        :param n_ingroups: The number of ingroups.
        """
        self.probabilities = np.zeros(n_ingroups + 1)

        # calculate polarization probabilities
        for i in range(1, n_ingroups):
            self.probabilities[i] = 1 / i / (1 / i + 1 / (n_ingroups - i))

        # add divergence probabilities
        self._add_divergence()

        return self.probabilities


class AdaptivePolarizationPrior(PolarizationPrior):
    """
    Adaptive prior. To be used with :class:`MaximumLikelihoodAncestralAnnotation`. This is the
    same prior as used in the EST-SFS paper. This prior is adaptive in the sense that the most likely polarization
    probabilities given the site configurations are found. This is the most accurate prior, but requires a lot of
    sites in order to work properly. You can check that the polarization probabilities are smooth enough across
    frequency counts by calling :meth:`~fastdfe.annotation.PolarizationPrior.plot_polarization`. If they are not
    smooth enough, you can increase the number of sites, decrease the number of ingroups, or
    use :class:`~fastdfe.annotation.KingmanPolarizationPrior` instead.
    """

    def __init__(
            self,
            n_runs: int = 1,
            parallelize: bool = True,
            allow_divergence: bool = False,
            seed: int | None = 0
    ):
        """
        Create a new adaptive prior instance.
        
        :param n_runs: The number of runs to perform when determining the polarization parameters. One
            run should be sufficient as only one parameter is optimized.
        :param parallelize: Whether to parallelize the optimization.
        :param allow_divergence: Whether to allow divergence. See :class:`PolarizationPrior` for details.
        :param seed: The seed for the random number generator.
        """
        super().__init__(allow_divergence=allow_divergence)

        #: The number of runs to use for the adaptive prior.
        self.n_runs: int = n_runs

        #: Whether to parallelize the optimization.
        self.parallelize: bool = parallelize

        #: The seed for the random number generator.
        self.seed: int | None = seed

        #: The random number generator.
        self.rng: np.random.Generator = np.random.default_rng(seed=self.seed)

    def _get_prior(
            self,
            configs: pd.DataFrame,
            n_ingroups: int
    ) -> np.ndarray[float, (...,)]:
        """
        Get the polarization probabilities.

        :param configs: The site configurations.
        :param n_ingroups: The number of ingroups.
        :return: The polarization probabilities.
        """

        # folded frequency bin indices
        # if the number of polymorphic bins is odd, the middle bin is fixed
        freq_indices = range(1, (n_ingroups + 1) // 2)

        # get the likelihood functions
        funcs = dict((i, self._get_likelihood(i, configs, n_ingroups)) for i in freq_indices)

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
        data = np.array(list(itertools.product(freq_indices, range(self.n_runs))))

        # get initial values
        initial_values = np.array([self.rng.uniform() for _ in range(data.shape[0])])

        # add initial values
        data = np.hstack((data, initial_values.reshape((-1, 1))))

        # run the optimization in parallel for each frequency bin over n_runs
        results: np.ndarray[OptimizeResult] = parallelize_func(
            func=optimize_polarization,
            data=data,
            parallelize=self.parallelize,
            pbar=True,
            desc=f"{self.__class__.__name__}>Optimizing polarization priors",
            dtype=object
        ).reshape(len(freq_indices), self.n_runs)

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
        # noinspection all
        probs = np.array([results[i, j].x[0] for i, j in enumerate(i_best)])

        # check for zeros or ones
        if np.any(probs == 0) or np.any(probs == 1):
            # get the number of bad frequency bins
            n_bad = np.sum((probs == 0) | (probs == 1))

            self._logger.fatal(f"Polarization probabilities are 0 for {n_bad} frequency bins which "
                               f"can be a real problem as it means that there are no sites "
                               f"for those bins. This may be due to ``n_ingroups`` "
                               f"being too large, or the number of provided sites being very "
                               f"small. If you can't increase the number of sites or decrease "
                               f"``n_ingroups``, consider using a the Kingman prior instead.")

        # if the number of ingroups is even
        if n_ingroups % 2 == 0:
            # noinspection all
            self.probabilities = np.concatenate(([1], probs, [0.5], 1 - probs[::-1], [0]))
        else:
            # if the number of ingroups is odd
            # noinspection all
            self.probabilities = np.concatenate(([1], probs, 1 - probs[::-1], [0]))

        # add divergence probabilities
        self._add_divergence()

        return self.probabilities

    @staticmethod
    def _get_likelihood(
            i: int,
            configs: pd.DataFrame,
            n_ingroups: int
    ) -> Callable[[List[float]], float]:
        """
        Get the likelihood function.

        :param i: The ith frequency bin.
        :param configs: The site configurations.
        :param n_ingroups: The number of ingroups.
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
            i_minor = n_ingroups - configs.n_major == i

            # weight the sites by the probability of polarization
            p_configs = pi * configs.p_major[i_minor] + (1 - pi) * configs.p_minor[i_minor]

            # return the negative log likelihood
            return -(np.log(p_configs) * configs.multiplicity[i_minor]).sum()

        return compute_likelihood


class _OutgroupAncestralAlleleAnnotation(AncestralAlleleAnnotation, ABC):
    """
    Abstract class for annotation of ancestral alleles using outgroup information.
    """
    #: Subsample mode.
    subsample_mode: Literal['random', 'probabilistic'] = 'random'

    def __init__(
            self,
            outgroups: List[str],
            n_ingroups: int,
            ingroups: List[str] | None = None,
            exclude: List[str] = [],
            seed: int | None = 0,
    ):
        """
        Create a new ancestral allele annotation instance.

        :param outgroups: The outgroup samples to consider when determining the ancestral allele. A list of
            sample names as they appear in the VCF file.
        :param n_ingroups:  The minimum number of ingroups that must be present at a site for it to be considered
            for ancestral allele inference.
        :param ingroups: The ingroup samples to consider when determining the ancestral allele. A list of
            sample names as they appear in the VCF file. If ``None``, all samples except the outgroups are
            considered.
        :param exclude: Samples to exclude from the ingroup. A list of sample names as they appear in the VCF file.
        :param seed: The seed for the random number generator.
        """
        # make sure the number of ingroups is at least 2
        if n_ingroups < 2:
            raise ValueError("The number of ingroups must be at least 2.")

        super().__init__()

        #: The ingroup samples to consider when determining the ancestral allele.
        self.ingroups: List[str] | None = ingroups

        #: The samples excluded from the ingroup.
        self.exclude: List[str] = exclude

        #: The outgroup samples to consider when determining the ancestral allele.
        self.outgroups: List[str] = outgroups

        #: The number of ingroups.
        self.n_ingroups: int = int(n_ingroups)

        #: The number of outgroups.
        self.n_outgroups: int = len(outgroups)

        #: The seed for the random number generator.
        self.seed: int | None = seed

        #: The random number generator.
        self.rng: np.random.Generator = np.random.default_rng(seed=self.seed)

        #: The outgroup mask.
        self._outgroup_mask: np.ndarray[bool] | None = None

        #: The outgroup indices.
        self._outgroup_indices: np.ndarray[int] | None = None

        #: The ingroup mask.
        self._ingroup_mask: np.ndarray[bool] | None = None

        #: 1-based positions of lowest and highest site position per contig (only when target_site_counter is used)
        # noinspection PyTypeChecker
        self._contig_bounds: Dict[str, Tuple[int, int]] = defaultdict(lambda: (np.inf, -np.inf))

    def _prepare_masks(self, samples: List[str]):
        """
        Prepare the masks for ingroups and outgroups.

        :param samples: All samples.
        """
        # create mask for ingroups
        if self.ingroups is None:
            self._ingroup_mask = ~ np.isin(samples, self.outgroups) & ~ np.isin(samples, self.exclude)
        else:
            self._ingroup_mask = np.isin(samples, self.ingroups) & ~ np.isin(samples, self.exclude)

        # create mask for outgroups
        self._outgroup_mask = np.isin(samples, self.outgroups)

        # make sure all specified outgroups are present
        if np.sum(self._outgroup_mask) != len(self.outgroups):
            # get missing outgroups
            missing = np.array(self.outgroups)[~np.isin(self.outgroups, samples)]

            raise ValueError(f"The specified outgroups ({', '.join(missing)}) are not present in the VCF file.")

        # outgroup indices
        # we ignore the order when using the mask
        self._outgroup_indices = np.array([samples.index(outgroup) for outgroup in self.outgroups])

        # inform of the number of ingroups
        self._logger.info(f"Subsampling {self.n_ingroups} ingroup haplotypes " +
                          ("randomly " if self.subsample_mode == "random" else "probabilistically ") +
                          f"from {np.sum(self._ingroup_mask)} individuals in total.")

        # inform on outgroup samples
        self._logger.info(f"Using {np.sum(self._outgroup_mask)} outgroup samples ({', '.join(self.outgroups)}).")

    def _setup(self, handler: MultiHandler):
        """
        Add info fields to the header.

        :param handler: The handler.
        """
        super()._setup(handler)

        # add info field
        handler._reader.add_info_to_header({
            'ID': self._handler.info_ancestral + '_info',
            'Number': '.',
            'Type': 'String',
            'Description': 'Additional information about the ancestral allele.'
        })

        # set reader
        self._reader = self._handler.load_vcf()

        # prepare masks
        self._prepare_masks(handler._reader.samples)

    @staticmethod
    def _subsample(
            genotypes: np.ndarray[Any],
            size: int,
            rng: np.random.Generator
    ) -> np.ndarray[str]:
        """
        Subsample a set of bases.

        :param genotypes: A list of bases.
        :param size: The size of the subsample.
        :return: A subsample of the bases.
        """
        if genotypes.shape[0] == 0:
            return np.array([])

        subsamples = rng.choice(
            a=genotypes.shape[0],
            size=min(size, genotypes.shape[0]),
            replace=False
        )

        return genotypes[subsamples]

    @staticmethod
    def _get_outgroup_bases(
            genotypes: np.ndarray[str],
            n_outgroups: int
    ) -> np.ndarray[str]:
        """
        Get the outgroup bases for a variant.

        :param genotypes: The VCF genotype strings.
        :param n_outgroups: The number of outgroups.
        :return: The outgroup bases.
        """
        outgroup_bases = np.full(n_outgroups, '.')

        for i, genotype in enumerate(genotypes):
            called_bases = get_called_bases([genotype])

            if len(called_bases) > 0:
                outgroup_bases[i] = called_bases[0]

        return outgroup_bases

    @classmethod
    def _subsample_site(
            cls,
            mode: Literal['random', 'probabilistic'],
            n: int,
            samples: np.ndarray[str],
            rng: np.random.Generator
    ) -> Tuple[np.ndarray[str], np.ndarray[int], np.ndarray[float]]:
        """
        Subsample a site, either randomly or probabilistically.

        :param mode: The subsampling mode.
        :param n: The number of ingroups to subsample to.
        :param samples: The samples.
        :return: Major alleles, major allele counts, and multiplicities,
            possibly including zero multiplicities.
        """
        if mode == 'random':
            # subsample ingroups
            samples = cls._subsample(samples, size=n, rng=rng)

        # get the major allele count
        most_common = Counter(samples).most_common()

        if mode == 'random':

            major_alleles = [most_common[0][0]]
            n_majors = [most_common[0][1]]
            m = [1]

        # if there is only one allele, probabilistic
        # subsampling is trivial
        elif len(most_common) < 2:

            major_alleles = [most_common[0][0]]
            n_majors = [n]
            m = [1]

        else:
            ref_allele = most_common[0][0]
            n_ref = most_common[0][1]
            alt_allele = most_common[1][0]

            # get the major allele counts
            n_majors = np.arange(n + 1)
            major_alleles = np.full(n + 1, ref_allele)
            m = hypergeom.pmf(k=n_majors, M=len(samples), n=n_ref, N=n)

            # flip alleles where the ref allele is not the major allele
            flip = n_majors < (n + 1) // 2
            n_majors[flip] = n - n_majors[flip]
            major_alleles[flip] = alt_allele

        return major_alleles, n_majors, m

    def _parse_variant(self, variant: Variant | DummyVariant) -> List[SiteConfig] | None:
        """
        Parse a VCF variant. We only consider sites that are at most bi-allelic in the in- and outgroups.

        :param variant: The variant.
        :return: list of site configurations containing a single element if subsample_mode is `random` or
            multiple elements if subsample_mode is `probabilistic` or ``None`` if the site is not valid.
        """
        # get the called ingroup bases
        ingroups = get_called_bases(variant.gt_bases[self._ingroup_mask])

        # get the numer of called ingroup and outgroup bases
        n_ingroups = len(ingroups)

        # only consider sites with enough ingroups
        if n_ingroups >= self.n_ingroups:

            # get the called outgroup bases
            # the order does not matter here
            outgroups = get_called_bases(variant.gt_bases[self._outgroup_mask])

            # get total base counts
            counts = Counter(np.concatenate((ingroups, outgroups)))

            # only consider sites where the in- and outgroups are at most bi-allelic
            if len(counts) <= 2:

                # get the bases
                b: List[str] = list(counts.keys())

                # subsample ingroups either randomly or probabilistically
                major_alleles, n_majors, multiplicities = self._subsample_site(
                    mode=self.subsample_mode,
                    n=self.n_ingroups,
                    samples=ingroups,
                    rng=self.rng
                )

                # Get the outgroup bases.
                # The outgroup order is important, so we can't use the mask here.
                outgroup_bases = self.get_base_index(self._get_outgroup_bases(
                    genotypes=np.array([variant.gt_bases[i] for i in self._outgroup_indices]),
                    n_outgroups=self.n_outgroups
                ))

                # create site configurations
                sites = []
                for i, (major_allele, n_major, multiplicity) in enumerate(zip(major_alleles, n_majors, multiplicities)):

                    if multiplicity > 0:
                        if len(counts) == 2:
                            # Take the other allele as the minor allele. We keep track of the minor allele
                            # even if it wasn't contained in the ingroup subsample.
                            minor_base: str = b[0] if b[0] != major_allele else b[1]
                        else:
                            minor_base: str = '.'

                        # create site configuration
                        site = SiteConfig(
                            major_base=base_indices[major_allele],
                            n_major=n_major,
                            minor_base=self.get_base_index(minor_base),
                            outgroup_bases=outgroup_bases,
                            multiplicity=multiplicity
                        )

                        sites.append(site)

                return sites

    @staticmethod
    def get_base_string(indices: int | np.ndarray[int]) -> str | np.ndarray[str]:
        """
        Get base string(s) from base index/indices.

        :param indices: The base index/indices.
        :return: Base string(s).
        """
        if isinstance(indices, np.ndarray):

            if len(indices) == 0:
                return np.array([])

            is_valid = indices != -1

            base_strings = np.full(indices.shape, '.', dtype=str)
            base_strings[is_valid] = bases[indices[is_valid]]

            return base_strings

        # assume integer
        if indices != -1:
            return bases[indices]

        return '.'

    @classmethod
    def get_base_index(cls, base_string: str | np.ndarray[str]) -> int | np.ndarray[int]:
        """
        Get base index/indices from base string(s).

        :param base_string: The base string(s).
        :return: Base index/indices.
        """
        if isinstance(base_string, np.ndarray):
            return np.array([cls.get_base_index(b) for b in base_string], dtype=int)

        # assume string
        if base_string in bases:
            return base_indices[base_string]

        return -1


class MaximumLikelihoodAncestralAnnotation(_OutgroupAncestralAlleleAnnotation):
    """
    Annotation of ancestral alleles following the probabilistic model of EST-SFS
    (https://doi.org/10.1534/genetics.118.301120). By default, the info field ``AA``
    (see :attr:`Annotator.info_ancestral`) is added to the VCF file, which holds the ancestral allele. To be used with
    :class:`Annotator` or :class:`~fastdfe.parser.Parser`. This class can also be used
    independently, see the :meth:`from_dataframe`, :meth:`from_data` and :meth:`from_est_sfs` methods.

    Initially, the branch rates are determined using MLE. Similar to :class:`Parser`, we can also specify the number of
    mutational target sites (see the `n_target_sites` argument) in case our VCF file does not contain the full set of
    monomorphic sites. This is necessary to obtain realistic branch rate estimates. You can also choose a prior for the
    polarization probabilities (see :class:`PolarizationPrior`). Eventually, for every site, the probability that the
    major allele is ancestral is calculated.

    When annotating the variants of a VCF file, we check the most likely ancestral allele against a naive
    ad-hoc ancestral allele annotation, and record the sites for which we have disagreement. You might want to
    sanity-check the mismatches to make sure the model has been properly specified (see :attr:`mismatches`).

    .. note::

        * The polarization prior corresponds to the Kingman coalescent probability by default. Using an adaptive prior,
          as in the EST-SFS paper, is also possible, but this is only recommended if the number of sites used for the
          inference is large (see :attr:`prior`).

        * The model can only handle sites that have at most 2 alleles across the in- and outgroups, so sites with more
          than 2 alleles are ignored. Only variants that are at most bi-allelic in the provided in- and outgroups are
          annotated.

        * The model determines the probability of the major allele being ancestral opposed to the minor allele. This can
          be problematic if the actual ancestral allele is not contained in the ingroup (possibly due to subsampling).
          To avoid this issue, we also keep track of potential minor alleles at frequency 0. If we were to ignore this,
          it would be impossible to infer divergence, i.e. fixed derived allele that are no longer observed in the
          ingroups (see :attr:`PolarizationPrior.allow_divergence`). That said, divergence counts are not informative
          on DFE inference with fastDFE and allow_divergence should not be set to ``True`` if interested in the SFS.

        * The model assumes a single coalescent topology for all sites, in which all outgroups coalesce first with the
          ingroup and not with each other. It is important to specify the outgroups in order of increasing divergence
          and not to select outgroups that are not much more closely related to each other than to the ingroup (as this
          would give rise to a different coalescent topology than the one assumed). You can call
          :meth:`get_outgroup_divergence` after the inference to check the estimated branch rates for each outgroup.
          The assumption of a single fixed topology should be good enough provided that in- and outgroups are
          sufficiently diverged.

    Example usage:

    ::

        import fastdfe as fd

        ann = fd.Annotator(
            vcf="https://github.com/Sendrowski/fastDFE/"
                "blob/dev/resources/genome/betula/all."
                "with_outgroups.subset.10000.vcf.gz?raw=true",
            annotations=[fd.MaximumLikelihoodAncestralAnnotation(
                outgroups=["ERR2103730"],
                n_ingroups=15
            )],
            output="genome.polarized.vcf.gz"
        )

        ann.annotate()

    """

    #: The data types for the data frame
    _dtypes = dict(
        n_major=np.int8,
        multiplicity=np.float64,
        sites=object,
        major_base=np.int8,
        minor_base=np.int8,
        outgroup_bases=object,
        p_major_ancestral=np.float64,
        p_minor=np.float64,
        p_major=np.float64
    )

    #: The columns to group by.
    _group_cols = ['major_base', 'minor_base', 'outgroup_bases', 'n_major']

    def __init__(
            self,
            outgroups: List[str],
            n_ingroups: int = 11,
            ingroups: List[str] | None = None,
            exclude: List[str] | None = None,
            n_runs: int = 10,
            model: SubstitutionModel = K2SubstitutionModel(),
            parallelize: bool = True,
            prior: PolarizationPrior | None = KingmanPolarizationPrior(),
            max_sites: int = 10000,
            seed: int | None = 0,
            confidence_threshold: float = 0,
            n_target_sites: int | None = None,
            subsample_mode: Literal['random', 'probabilistic'] = 'probabilistic'
    ):
        """
        Create a new ancestral allele annotation instance.

        :param outgroups: The outgroup samples to consider when determining the ancestral allele in the order of
            increasing divergence. A list of sample names as they appear in the VCF file. The order of the outgroups
            is important as it determines the order of the branches in the tree, whose rates are optimized, and whose
            topology is predetermined. The first outgroup is the closest outgroup to the ingroups, and the last
            outgroup is the most distant outgroup. More outgroups lead to a more accurate inference of the ancestral
            allele, but also increase the computational cost. Using more than 1 outgroup is recommended, but more than
            3 is likely not necessary. Sites where these outgroups are not present are not included when optimizing
            the rate parameters. Due to assumptions on the tree topology connecting the in- and outgroups, it is
            important that the outgroups are not much more closely related to each other than to the ingroups. Ideally,
            the optimized branch rates are show markedly different values, and in any case, they should be monotonically
            increasing with the outgroups (see :meth:`get_outgroup_divergence`).
        :param n_ingroups: The minimum number of ingroups that must be present at a site for it to be considered
            for ancestral allele inference. The ingroup subsampling is necessary since our model requires an equal
            number of ingroups for all sites. Note that a larger number of ingroups does not necessarily improve
            the accuracy of the ancestral allele inference (see ``prior``). A larger number of ingroups can lead
            to a large variance in the polarization probabilities, across different frequency counts. ``n_ingroups``
            should thus only be large if the number of sites used for the inference is also large. A sensible value
            for a reasonably large number of sites (a few thousand) is 10 or perhaps 20 for a larger numbers of sites.
            Very small values can lead to the ingroup subsamples not being representative of the actual allele
            frequencies at a site, especially when not using probabilistic subsampling (see ``subsample_mode``).
            This value also influences the number of frequency bins used for the polarization probabilities, and should
            thus not be too small. Note that if ``ingroups`` is an even number, the major allele is chosen arbitrarily
            if the number of major alleles is equal to the number of minor alleles. To avoid this, you can use an odd
            number of ingroups.
        :param ingroups: The ingroup samples to consider when determining the ancestral allele. If ``None``,
            all (non-outgroup) samples are considered. A list of sample names as they appear in the VCF file.
            Has to be at least as large as ``n_ingroups``.
        :param exclude: Samples to exclude from the ingroup. A list of sample names as they appear in the VCF file.
        :param n_runs: The number of optimization runs to perform when determining the branch rates. You can
            check that the likelihoods of the different runs are similar by calling :meth:`plot_likelihoods`.
        :param parallelize: Whether to parallelize the computation across multiple cores.
        :param prior: The prior to use for the polarization probabilities. See :class:`PolarizationPrior`, 
            :class:`KingmanPolarizationPrior` and :class:`AdaptivePolarizationPrior` for more information.
        :param max_sites: The maximum number of sites to consider. This is useful if the number of sites is very large.
            Choosing a reasonably large subset of sites (on the order of a few thousand bi-allelic sites) can speed up
            the computation considerably as parsing can be slow. This subset is then used to calibrate the rate
            parameters, and possibly the polarization priors.
        :param seed: The seed for the random number generator. If ``None``, a random seed is chosen.
        :param confidence_threshold: The confidence threshold for the ancestral allele annotation.
            Only if the probability of the major allele being ancestral as opposed to
            the minor allele is not within ``((1 - confidence_threshold) / 2, 1 - (1 - confidence_threshold) / 2)``,
            the ancestral allele is annotated. This is useful to avoid annotating sites where the ancestral allele
            state is not clear. Use values close to ``0`` to annotate as many sites as possible, and values close to
            ``1`` to annotate only sites where the ancestral allele state is very clear.

            .. warning:: This threshold introduces a bias by excluding more sites with high-frequency derived alleles
                and should thus be kept at ``0`` if the distribution of frequency counts is important, e.g., if the SFS
                is to be determined.
        :param n_target_sites: The total number of target sites if this class is used in conjunction with
            :class:`Parser` or :class:`Annotator`. This is useful if the provided set of sites only
            consists of bi-allelic sites. Specify here the total number of sites underlying the given dataset, i.e.,
            both mono- and bi-allelic sites. Ignoring mono-allelic sites will lead to overestimation of the rate
            parameters. For this to work, a FASTA file must be provided from which the mono-allelic sites can be
            sampled. Sampling takes place between the variants of the last and first site on every contig considered
            in the VCF file. Use `None` to disable this feature. Note that the number of target sites is automatically
            carried over if not specified and this class is used together with :class:`Parser`. In order to use this
            feature, you must also specify a FASTA file to :class:`Parser` or :class:`Annotator`. Also note that we
            extrapolate the number of mono-allelic sites to be sampled from the FASTA file based on the ratio of
            sites with called outgroup bases parsed from the VCF file. This is done to obtain branch rates that are
            comparable to the ones obtained when using a VCF file that contains both mono- and bi-allelic sites.
        :param subsample_mode: The subsampling mode. For ``random``, we draw once without replacement from the set of
            all available ingroup genotypes per site. For ``probabilistic``, we integrate over the hypergeometric
            distribution when parsing and computing the ancestral probabilities. Probabilistic subsampling requires a
            bit more time, but produces much more stable results, while requiring far fewer sites, so it is highly
            recommended.
        """
        super().__init__(
            ingroups=ingroups,
            exclude=exclude,
            outgroups=outgroups,
            n_ingroups=n_ingroups,
            seed=seed
        )

        # check that we have at least one outgroup
        if len(outgroups) < 1:
            raise ValueError("Must specify at least one outgroup. If you do not have any outgroup "
                             "information, consider using MaximumParsimonyAncestralAnnotation instead.")

        # check that we have enough ingroups specified if specified at all
        if ingroups is not None and len(ingroups) * 2 < n_ingroups:
            self._logger.warning("The number of specified ingroup samples is smaller than the "
                                 "number of ingroups (assumed diploidy). Please make sure to "
                                 "provide sufficiently many ingroups.")

        # raise warning on bias
        if confidence_threshold > 0:
            self._logger.warning("Please be aware that a confidence threshold of greater than 0 biases the SFS "
                                 "towards fewer high-frequency derived alleles.")

        # check subsample mode
        if subsample_mode not in ['random', 'probabilistic']:
            raise ValueError(f"Invalid subsample mode: {subsample_mode}")

        #: Whether to parallelize the computation.
        self.parallelize: bool = parallelize

        #: Maximum number of sites to consider
        self.max_sites: int = max_sites

        #: The confidence threshold for the ancestral allele annotation.
        self.confidence_threshold: float = confidence_threshold

        #: The prior to use for the polarization probabilities.
        self.prior: PolarizationPrior | None = prior

        #: Number of random ML starts when determining the rate parameters
        self.n_runs: int = int(n_runs)

        #: The substitution model.
        self.model: SubstitutionModel = model

        #: The VCF reader.
        self._reader: VCF | None = None

        #: The data frame holding all site configurations.
        self.configs: pd.DataFrame | None = None

        #: The probability of all sites per frequency bin.
        self.p_bins: Dict[str, np.ndarray[float, (n_ingroups - 1,)]] | None = None

        #: The total number of valid sites parsed (including sites not considered for ancestral allele inference).
        self.n_sites: int | None = None

        #: The parameter names in the order they are passed to the optimizer.
        self.param_names: List[str] = list(self.model.get_bounds(self.n_outgroups).keys())

        #: The log likelihoods for the different runs when optimizing the rate parameters.
        self.likelihoods: np.ndarray[float, (...,)] | None = None

        #: The best log likelihood when optimizing the rate parameters.
        self.likelihood: float | None = None

        #: Optimization result of the best run.
        self.result: OptimizeResult | None = None

        #: The MLE parameters.
        self.params_mle: Dict[str, float] | None = None

        #: Mismatches between the most likely ancestral allele and the ad-hoc ancestral allele.
        # This is only computed when annotating a VCF file, and only contains the mismatches
        # for sites that were actually annotated.
        self.mismatches: List[SiteInfo] = []

        #: The total number of target sites.
        self.n_target_sites: int | None = n_target_sites

        #: The subsampling mode.
        self.subsample_mode: Literal['random', 'probabilistic'] = subsample_mode

    def _setup(self, handler: MultiHandler):
        """
        Parse the VCF file and perform the optimization.

        :param handler: The handler.
        """
        from .parser import Parser, TargetSiteCounter

        super()._setup(handler)

        # try to carry over n_target_sites and fasta file from Parser
        if isinstance(handler, Parser) and isinstance(handler.target_site_counter, TargetSiteCounter):

            if self.n_target_sites is None:
                self.n_target_sites = handler.target_site_counter.n_target_sites

                self._logger.debug(f"Using n_target_sites={self.n_target_sites} from Parser.")

        if self.n_target_sites is not None:
            # check that we have a fasta file if we sample mono-allelic sites
            handler._require_fasta(self.__class__.__name__)

        # load data
        self._parse_vcf()

        # sample mono-allelic sites if necessary
        if self.n_target_sites is not None:
            self._sample_mono_allelic_sites()

        # notify about the number of sites
        self._logger.info(f"Included {int(self._get_mle_configs().multiplicity.sum())} sites for the inference.")

        # infer ancestral alleles
        self.infer()

    def _get_n_samples_fasta(self) -> int:
        """
        Get the number of sites to be sampled from the FASTA file. This assumed that the sites have not
        been sampled yet.
        """
        # ratio of parsed sites to sites used for MLE
        ratio_mle = self._get_mle_configs().multiplicity.sum() / self.n_sites

        return int(ratio_mle * (self.n_target_sites - self.n_sites))

    def _get_n_sites(self) -> int:
        """
        Get the number of sites to consider.
        """
        return int(self.configs.multiplicity.sum())

    def _sample_mono_allelic_sites(self):
        """
        Sample mono-allelic sites from the FASTA file.
        """
        # inform
        self._logger.info(f"Sampling mono-allelic sites.")

        if self.n_target_sites < self.n_sites:
            raise ValueError(f"The number of target sites ({self.n_target_sites}) must be at least "
                             f"as large as the number of sites parsed ({self.n_sites}).")

        # number of mono-allelic sites to sample
        n_samples = self._get_n_samples_fasta()

        # initialize progress bar
        pbar = tqdm(
            total=n_samples,
            desc=f'{self.__class__.__name__}>Sampling mono-allelic sites',
            disable=Settings.disable_pbar
        )

        # get array of ranges per contig of parsed variants
        ranges = np.array(list(self._contig_bounds.values()))

        # get range sizes
        range_sizes = ranges[:, 1] - ranges[:, 0]

        # determine sampling probabilities
        probs = range_sizes / np.sum(range_sizes)

        # sample number of sites per contig
        sample_counts = self.rng.multinomial(n_samples, probs)

        # sampled bases
        samples = dict(A=0, C=0, G=0, T=0)

        # iterate over contigs
        for contig, bounds, n in zip(self._contig_bounds.keys(), ranges, sample_counts):
            # get aliases
            aliases = self._handler.get_aliases(contig)

            # make sure we have a valid range
            if bounds[1] > bounds[0] and n > 0:
                self._logger.debug(f"Sampling {n} sites from contig '{contig}'.")

                # fetch contig
                record = self._handler.get_contig(aliases, notify=False)

                # sample sites
                i = 0
                while i < n:
                    pos = self.rng.integers(*bounds)

                    base = record.seq[pos - 1]

                    if base in bases:
                        # increase counters
                        samples[base] += 1
                        i += 1
                        pbar.update()

        # close progress bar
        pbar.close()

        # rewind fasta iterator
        FASTAHandler._rewind(self._handler)

        # add site counts to data frame
        self.configs = self._add_monomorphic_sites(samples)

        # update number of sites
        self.n_sites = self._get_n_sites()

    def _add_monomorphic_sites(self, samples: Dict[str, int]):
        """
        Add monomorphic sites to the data frame holding the site configurations.

        :param samples: The samples.
        :return: The data frame.
        """

        # get indices for new sites
        sites = np.concatenate(([self.n_sites], self.n_sites + np.cumsum(list(samples.values()))))

        # construct data frame of new sites
        df = pd.DataFrame(dict(
            n_major=self.n_ingroups,
            major_base=base_indices[base],
            minor_base=-1,
            outgroup_bases=(base_indices[base],) * self.n_outgroups,
            multiplicity=count,
            sites=list(range(sites[i], sites[i + 1])),
            n_outgroups=self.n_outgroups
        ) for i, (base, count) in enumerate(samples.items()))

        # add to data frame
        configs = pd.concat((self.configs, df))

        # aggregate
        return configs.groupby(self._group_cols + ['n_outgroups'], as_index=False, dropna=False).sum()

    def _teardown(self):
        """
        Teardown the annotation.
        """
        super()._teardown()

        # inform on mismatches
        self._logger.info(f"There were {len(self.mismatches)} mismatches between the most likely "
                          f"ancestral allele and the ad-hoc ancestral allele annotation.")

    @classmethod
    def _parse_est_sfs(cls, data: pd.DataFrame) -> pd.DataFrame:
        """
        Parse EST-SFS data.

        :param data: The data frame.
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
        data['major_base'] = data.major_base.astype(cls._dtypes['major_base'])

        # determine the mono-allelic sites
        poly_allelic = (ingroup_data > 0).sum(axis=1) > 1

        # determine the minor alleles
        minor_bases = np.full(data.shape[0], -1, dtype=np.int8)
        minor_bases[poly_allelic] = data_sorted[:, -2][poly_allelic]

        # assign the minor alleles
        data['minor_base'] = minor_bases

        # extract outgroup data
        outgroup_data = np.full((data.shape[0], n_outgroups), -1, dtype=np.int8)
        for i in range(n_outgroups):
            # get the genotypes
            genotypes = data[i + 1].str.split(',', expand=True).astype(np.int8).to_numpy()

            # determine whether the site has an outgroup
            has_outgroup = genotypes.sum(axis=1) > 0

            # determine the outgroup allele indices provided the site has an outgroup
            outgroup_data[has_outgroup, i] = genotypes[has_outgroup].argmax(axis=1)

        # assign the outgroup data, convert to tuples for hashing
        data['outgroup_bases'] = [tuple(row) for row in outgroup_data]

        # return new columns only
        return data.drop(range(n_outgroups + 1), axis=1)

    @classmethod
    def from_est_sfs(
            cls,
            file: str,
            prior: PolarizationPrior | None = KingmanPolarizationPrior(),
            n_runs: int = 10,
            model: SubstitutionModel = K2SubstitutionModel(),
            parallelize: bool = True,
            seed: int = 0,
            chunk_size: int = 100000
    ) -> 'MaximumLikelihoodAncestralAnnotation':
        """
        Create instance from EST-SFS input file.

        :param file: File containing EST-SFS-formatted input data.
        :param prior: The prior to use for the polarization probabilities (see :meth:`__init__`).
        :param n_runs: Number of runs for rate estimation (see :meth:`__init__`).
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

            data = data.groupby(cls._group_cols, as_index=False, dropna=False).sum()

        # check if there is data
        if data is None:
            raise ValueError("No data found.")

        # determine the multiplicity
        data['multiplicity'] = data['sites'].apply(lambda x: len(x))

        # create from dataframe
        return cls.from_dataframe(
            data=data,
            n_runs=n_runs,
            model=model,
            parallelize=parallelize,
            prior=prior,
            n_ingroups=n_ingroups,
            grouped=True,
            seed=seed
        )

    def to_est_sfs(self, file: str):
        """
        Write the object's state to an EST-SFS formatted file.

        :param file: The output file name.
        """
        # get config indices for each site
        indices = self._get_site_indices()

        # remove sites that are not included
        indices = indices[indices != -1]

        # get the sites
        sites = self.configs.iloc[indices]

        with open(file, 'w') as f:

            # iterate over rows
            for i, site in sites.iterrows():

                # ingroup counts
                ingroups = np.zeros(4, dtype=int)

                # major allele count
                ingroups[site['major_base']] = site['n_major']

                # minor allele count if not mono-allelic
                if site['minor_base'] != -1:
                    ingroups[site['minor_base']] = self.n_ingroups - site['n_major']

                # write ingroup counts
                outgroups = np.zeros((self.n_outgroups, 4), dtype=int)

                # fill outgroup counts
                for j, base in enumerate(site['outgroup_bases']):
                    if base != -1:
                        outgroups[j, base] = 1

                # write line
                f.write(
                    ','.join(ingroups.astype(str)) +
                    '\t' +
                    '\t'.join([','.join(o) for o in outgroups.astype(str)]) +
                    '\n'
                )

                # break if we reached the maximum number of sites
                if i + 1 >= self.max_sites:
                    break

    @classmethod
    def from_data(
            cls,
            n_major: Iterable[int],
            major_base: Iterable[str | int],
            minor_base: Iterable[str | int],
            outgroup_bases: Iterable[Iterable[str | int]],
            n_ingroups: int,
            n_runs: int = 10,
            model: SubstitutionModel = K2SubstitutionModel(),
            parallelize: bool = True,
            prior: PolarizationPrior | None = KingmanPolarizationPrior(),
            seed: int = 0,
            pass_indices: bool = False,
            confidence_threshold: float = 0
    ) -> 'MaximumLikelihoodAncestralAnnotation':
        """
        Create an instance by passing the data directly.

        :param n_major: The number of major alleles per site. Note that this number has to be lower than ``n_ingroups``,
            as we consider the number of major alleles of subsamples of size ``n_ingroups``.
        :param major_base: The major allele per site. A string representation of the base or the base index according
            to ``['A', 'C', 'G', 'T']`` if ``pass_indices`` is ``True``. Use ``None`` if the base is not defined when
            ``pass_indices`` is ``False`` and ``-1`` when ``pass_indices`` is ``True``.
        :param minor_base: The minor allele per site. A string representation of the base or the base index according
            to ``['A', 'C', 'G', 'T']`` if ``pass_indices`` is ``True``. Use ``None`` if the base is not defined when
            ``pass_indices`` is ``False`` and ``-1`` when ``pass_indices`` is ``True``.
        :param outgroup_bases: The outgroup alleles per site. A string representation of the base or the base index
            if ``pass_indices`` is ``True``. This should be a list of lists, where the outer list corresponds to the
            sites and the inner list to the outgroups per site. All sites are required to have the same number of
            outgroups. Use ``None`` if the base is not defined when ``pass_indices`` is ``False`` and ``-1`` when
            ``pass_indices`` is ``True``.
        :param n_ingroups: The number of ingroup samples (see :meth:`__init__`).
        :param n_runs: The number of runs for rate estimation (see :meth:`__init__`).
        :param model: The substitution model (see :meth:`__init__`).
        :param parallelize: Whether to parallelize the runs.
        :param prior: The prior to use for the polarization probabilities (see :meth:`__init__`).
        :param seed: The seed for the random number generator.
        :param pass_indices: Whether to pass the base indices instead of the bases.
        :param confidence_threshold: The confidence threshold for the ancestral allele annotation 
            (see :meth:`__init__`).
        :return: The instance.
        """
        # convert to numpy arrays
        n_major = np.array(list(n_major), dtype=np.int8)

        # make sure that the number of major alleles is not larger than the number of ingroups
        if np.any(n_major > n_ingroups):
            raise ValueError("Major allele counts cannot be larger than the number of ingroups.")

        # convert to base indices
        if not pass_indices:
            major_base = cls.get_base_index(np.array(list(major_base)))
            minor_base = cls.get_base_index(np.array(list(minor_base)))
            outgroup_bases = cls.get_base_index(np.array(list(outgroup_bases))).reshape(len(major_base), -1)

        # create data frame
        data = pd.DataFrame({
            'n_major': n_major,
            'major_base': major_base,
            'minor_base': minor_base,
            'outgroup_bases': list(outgroup_bases)
        })

        # create from dataframe
        return cls.from_dataframe(
            data=data,
            n_runs=n_runs,
            model=model,
            parallelize=parallelize,
            prior=prior,
            n_ingroups=n_ingroups,
            seed=seed,
            confidence_threshold=confidence_threshold
        )

    @classmethod
    def _from_vcf(
            cls,
            file: str,
            outgroups: List[str],
            n_ingroups: int,
            ingroups: List[str] = None,
            exclude: List[str] = None,
            n_runs: int = 10,
            model: SubstitutionModel = K2SubstitutionModel(),
            parallelize: bool = True,
            prior: PolarizationPrior | None = KingmanPolarizationPrior(),
            max_sites: int = np.inf,
            seed: int | None = 0,
            confidence_threshold: float = 0,
            subsample_mode: Literal['random', 'probabilistic'] = 'probabilistic'
    ) -> 'MaximumLikelihoodAncestralAnnotation':
        """
        Create an instance from a VCF file. In most cases, it is recommended to use the :class:`Annotator` or
        :class:`~fastdfe.parser.Parser` classes instead.

        :param file: The VCF file.
        :param outgroups: Same as in :meth:`__init__`.
        :param n_ingroups: Same as in :meth:`__init__`.
        :param ingroups: Same as in :meth:`__init__`.
        :param exclude: Same as in :meth:`__init__`.
        :param n_runs: Same as in :meth:`__init__`.
        :param model: Same as in :meth:`__init__`.
        :param parallelize: Same as in :meth:`__init__`.
        :param prior: Same as in :meth:`__init__`.
        :param max_sites: Same as in :meth:`__init__`.
        :param seed: Same as in :meth:`__init__`.
        :param confidence_threshold: Same as in :meth:`__init__`.
        :param subsample_mode: Same as in :meth:`__init__`.
        :return: The instance.
        """
        # create instance
        anc = MaximumLikelihoodAncestralAnnotation(
            outgroups=outgroups,
            n_ingroups=n_ingroups,
            ingroups=ingroups,
            exclude=exclude,
            n_runs=n_runs,
            model=model,
            parallelize=parallelize,
            prior=prior,
            max_sites=max_sites,
            seed=seed,
            confidence_threshold=confidence_threshold,
            subsample_mode=subsample_mode
        )

        # set up the handler
        super(cls, anc)._setup(MultiHandler(
            vcf=file,
            max_sites=max_sites,
            seed=seed
        ))

        # parse the variants
        anc._parse_vcf()

        return anc

    @classmethod
    def from_dataframe(
            cls,
            data: pd.DataFrame,
            n_ingroups: int,
            n_runs: int = 10,
            model: SubstitutionModel = K2SubstitutionModel(),
            parallelize: bool = True,
            prior: PolarizationPrior | None = KingmanPolarizationPrior(),
            seed: int = 0,
            grouped: bool = False,
            confidence_threshold: float = 0
    ) -> 'MaximumLikelihoodAncestralAnnotation':
        """
        Create an instance from a dataframe.

        :param data: Dataframe with the columns: ``major_base``, ``minor_base``, ``outgroup_bases``, ``n_major`` of
            type ``int``, ``int``, ``list`` and ``int``, respectively. The outgroup bases should have the same length
            for every site.
        :param n_ingroups: The number of ingroups (see :meth:`__init__`).
        :param n_runs: Number of runs for rate estimation (see :meth:`__init__`).
        :param model: The substitution model (see :meth:`__init__`).
        :param parallelize: Whether to parallelize computations.
        :param prior: The prior to use for the polarization probabilities (see :meth:`__init__`).
        :param seed: The seed for the random number generator. If ``None``, a random seed is chosen.
        :param grouped: Whether the dataframe is already grouped by all columns (used for internal purposes).
        :param confidence_threshold: The confidence threshold for the ancestral allele annotation 
            (see :meth:`__init__`).
        :return: The instance.
        """
        # check if dataframe is empty
        if data.empty:
            raise ValueError("Empty dataframe.")

        if not grouped:
            # only keep the columns that are needed
            data = data[cls._group_cols]

            # disable chained assignment warning
            with pd.option_context('mode.chained_assignment', None):

                # retain site index
                data['sites'] = data.index

                # convert outgroup bases to tuples
                data['outgroup_bases'] = data['outgroup_bases'].apply(tuple)

            # group by all columns in the chunk and keep track of the site indices
            data = data.groupby(cls._group_cols, as_index=False, dropna=False).agg(list).reset_index(drop=True)

        # determine the multiplicity
        data['multiplicity'] = data['sites'].apply(lambda x: len(x))

        # add missing columns with NA as default value
        for col in cls._dtypes:
            if col not in data.columns:
                data[col] = None

        # convert to the correct dtypes
        data = data.astype(cls._dtypes)

        # determine the number of outgroups
        data['n_outgroups'] = np.sum(np.array(data.outgroup_bases.to_list()) != -1, axis=1)

        # determine the number of outgroups
        n_outgroups = data.n_outgroups.max()

        anc = MaximumLikelihoodAncestralAnnotation(
            n_runs=n_runs,
            model=model,
            parallelize=parallelize,
            prior=prior,
            outgroups=[str(i) for i in range(n_outgroups)],  # pseudo names for outgroups
            ingroups=[str(i) for i in range(n_ingroups)],  # pseudo names for ingroups
            n_ingroups=n_ingroups,
            seed=seed,
            confidence_threshold=confidence_threshold,
            subsample_mode='random'
        )

        # assign data frame
        anc.configs = data

        # set the number of sites (which coincides with number of sites parsed)
        anc.n_sites = anc._get_n_sites()

        # notify about the number of sites
        anc._logger.info(f"Included {int(anc._get_mle_configs().multiplicity.sum())} sites for the inference.")

        return anc

    def _parse_vcf(self):
        """
        Parse variants from VCF file.
        """
        # initialize data frame
        self.configs = pd.DataFrame(columns=list(self._dtypes.keys()))
        self.configs.astype(self._dtypes)

        # columns to use as index
        index_cols = ['major_base', 'minor_base', 'outgroup_bases', 'n_major']

        # set index to initial site configuration
        self.configs.set_index(keys=index_cols, inplace=True)

        # trigger the site counter as we use it soon anyway
        _ = self._handler.n_sites

        # determine the total number of sites to be parsed
        total = self._handler.n_sites if self.max_sites == np.inf else self.max_sites

        # initialize counter in case we do not parse any sites
        i = -1

        # create progress bar
        with self._handler.get_pbar(desc=f"{self.__class__.__name__}>Parsing sites", total=total) as pbar:

            # iterate over sites
            for i, variant in enumerate(self._reader):

                # parse the site
                configs = self._parse_variant(variant)

                # check if site is not None
                if configs is not None:

                    if self.n_target_sites is not None:
                        # update bounds
                        low, high = self._contig_bounds[variant.CHROM]
                        self._contig_bounds[variant.CHROM] = (min(low, variant.POS), max(high, variant.POS))

                    for config in configs:

                        index = (
                            config.major_base,
                            config.minor_base,
                            tuple(config.outgroup_bases),
                            config.n_major
                        )

                        if index in self.configs.index:
                            # get the site data
                            site_data = self.configs.loc[index].to_dict()

                            # update the site data
                            site_data['multiplicity'] += config.multiplicity
                            site_data['sites'] += [i]

                            # update the site data
                            # Note that there were problems updating the data frame directly
                            self.configs.loc[index] = site_data
                        else:
                            self.configs.loc[index] = config.__dict__ | {'sites': [i]}

                pbar.update()

                # explicitly stopping after ``n`` sites fixes a bug with cyvcf2:
                # 'error parsing variant with `htslib::bcf_read` error-code: 0 and ret: -2'
                if i + 1 == self._handler.n_sites or i + 1 == self._handler.max_sites or i + 1 == self.max_sites:
                    break

        # reset the index
        self.configs.reset_index(inplace=True, names=index_cols)

        # create column for number of outgroups
        self.configs['n_outgroups'] = None

        if len(self.configs) > 0:
            # determine number of outgroups
            self.configs['n_outgroups'] = np.sum(np.array(self.configs.outgroup_bases.to_list()) != -1, axis=1)

        # total number of sites considered
        self.n_sites = i + 1

    def infer(self):
        """
        Infer the ancestral allele probabilities for the data provided. This method is only supposed to be called
        manually if the data is provided directly, e.g. using :meth:`from_data`, :meth:`from_dataframe` or
        :meth:`from_est_sfs`. If the data is provided using a VCF file, this method is called automatically.
        """
        # get the bounds
        bounds = self.model.get_bounds(self.n_outgroups)

        # get the likelihood function
        # this will raise an error if no data is available
        fun = self._get_likelihood()

        # log warning if unusually low number of monomorphic sites
        if self.configs[self.configs.minor_base == -1].multiplicity.sum() / self.n_sites < 0.75:
            self._logger.warning("The number of monomorphic sites is unusually low. Please note that "
                                 "including monomorphic sites is necessary to obtain realistic "
                                 "branch rate estimates.")

            if self.n_target_sites is None:
                self._logger.warning("If your dataset does not contain any monomorphic sites, consider "
                                     "using the `n_target_sites` argument.")

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
            data=[self.model.get_x0(bounds, self.rng) for _ in range(self.n_runs)],
            parallelize=self.parallelize,
            pbar=True,
            desc=f"{self.__class__.__name__}>Optimizing rates",
            dtype=object
        )

        # get the likelihoods for each run
        self.likelihoods = -np.array([result.fun for result in results])

        # get the best likelihood
        self.likelihood = np.max(self.likelihoods)

        # get the best result
        self.result: OptimizeResult = cast(OptimizeResult, results[np.argmax(self.likelihoods)])

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
            self._logger.warning(f'The MLE estimate for the rates is near the upper bound for '
                                 f'{near_upper} and lower bound for {near_lower}. (The tuples denote '
                                 f'(lower, value, upper) for every parameter.)')

        # check if the outgroup divergence is monotonically increasing
        if not self.is_monotonic():
            self._logger.warning("The outgroup rates are not monotonically increasing. This might indicate "
                                 "that the outgroups were not specified in the order of increasing divergence. "
                                 f"rates: {dict(zip(self.outgroups, self.get_outgroup_divergence()))}")

        # cache the branch probabilities for the MLE parameters
        self._renew_cache()

        # renew site configuration cache
        self._update_configs()

    def _update_configs(self):
        """
        Renew site configuration cache.
        """

        # obtain the probability for each site and minor allele under the MLE rate parameters
        self.configs.p_minor = self.get_p_configs(
            configs=self.configs,
            model=self.model,
            base_type=BaseType.MINOR,
            params=self.params_mle
        )

        # obtain the probability for each site and major allele under the MLE rate parameters
        self.configs.p_major = self.get_p_configs(
            configs=self.configs,
            model=self.model,
            base_type=BaseType.MAJOR,
            params=self.params_mle
        )

        # calculate the ancestral probabilities, i.e. probability of the major allele being ancestral
        # opposed to the minor allele
        self.configs.p_major_ancestral = self._calculate_p_major_ancestral(
            p_minor=self.configs['p_minor'].values,
            p_major=self.configs['p_major'].values,
            n_major=self.configs['n_major'].values
        )

    def set_mle_params(self, params: Dict[str, float]):
        """
        Set the MLE parameters and update the cache and site configurations. Use this method if you want to
        use different parameters for the annotation.

        :param params: The new parameters.
        """
        # set the parameters
        self.params_mle = params

        # renew cache
        self._renew_cache()

        # renew site configuration cache
        self._update_configs()

    def is_monotonic(self) -> bool:
        """
        Whether the outgroups are monotonically increasing in divergence.

        :return: Whether the outgroups are monotonically increasing in divergence.
        """
        # get the outgroup divergence
        div = self.get_outgroup_divergence()

        # check if the outgroup divergence is monotonically increasing
        return all(div[i] <= div[i + 1] for i in range(len(div) - 1))

    @cached_property
    def p_polarization(self) -> np.ndarray[float, (...,)] | None:
        """
        Get the polarization probabilities or ``None`` if ``prior`` is ``no``.
        """
        if isinstance(self.prior, PolarizationPrior):
            return self.prior._get_prior(
                configs=self.configs,
                n_ingroups=self.n_ingroups
            )

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
        if n_outgroups < 1:
            return 0.0

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
                i_internal = (i - 1) // 2

                # get internal base
                b1 = internal_nodes[i_internal]

                # either connect to outgroup or next internal node
                b2 = outgroup_bases[i_internal] if i % 2 == 1 else internal_nodes[i_internal + 1]
            else:
                # last branch connects to last internal node and last outgroup
                b1 = internal_nodes[-1]
                b2 = outgroup_bases[-1]

            # get the probability of the branch
            p_branches[i] = model._get_cached_prob(b1, b2, i, params)

        # take product of all branch probabilities
        prod = p_branches.prod()

        return prod

    @classmethod
    def get_p_config(
            cls,
            config: SiteConfig,
            base_type: BaseType,
            params: Dict[str, float],
            model: SubstitutionModel = K2SubstitutionModel(),
            internal: np.ndarray[int] | None = None
    ) -> float:
        """
        Get the probability for a site configuration.

        :param config: The site configuration.
        :param base_type: The base type.
        :param params: The parameters for the substitution model.
        :param model: The substitution model to use.
        :param internal: Base indices of internal nodes of the tree if fixed. If ``None``, the internal nodes
            are considered as free parameters. -1 also indicates a free parameter. The number of internal nodes
            is the number of outgroups minus one.
        :return: The probability for a site.
        """
        n_outgroups = len(config.outgroup_bases)

        # get the focal base
        base = config.major_base if base_type == BaseType.MAJOR else config.minor_base

        # if the focal base is missing we return a probability of 0
        if base == -1:
            return 0.0

        # number of free nodes
        n_free = 0

        # get internal node possibilities
        combs_internal = []
        for i in range(n_outgroups - 1):
            if internal is not None and internal[i] != -1:
                combs_internal.append([internal[i]])
            else:
                combs_internal.append([0, 1, 2, 3])
                n_free += 1

        # get outgroup possibilities
        combs_outgroup = []
        for i in range(n_outgroups):
            if config.outgroup_bases[i] != -1:
                combs_outgroup.append([config.outgroup_bases[i]])
            else:
                combs_outgroup.append([0, 1, 2, 3])
                n_free += 1

        # initialize the probability for each tree
        p_trees = np.zeros(4 ** n_free, dtype=float)

        # iterator over all possible internal node combinations
        for i, nodes in enumerate(itertools.product(*(combs_internal + combs_outgroup))):
            # get the probability of the tree
            p_trees[i] = cls.get_p_tree(
                base=base,
                n_outgroups=n_outgroups,
                internal_nodes=np.array(nodes[:n_outgroups - 1]),
                outgroup_bases=np.array(nodes[n_outgroups - 1:]),
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
        for i, config in enumerate(configs.itertuples()):
            # get the log likelihood of the site
            p_configs[i] = cls.get_p_config(
                config=cast(SiteConfig, config),
                base_type=base_type,
                params=params,
                model=model
            )

        return p_configs

    def evaluate_likelihood(self, params: Dict[str, float]) -> float:
        """
        Evaluate the likelihood function for the rate parameters.

        :param params: A dictionary of parameters.
        :return: The log likelihood.
        """
        # cache the branch probabilities
        self._renew_cache(params)

        # compute the likelihood
        ll = -self._get_likelihood()([params[name] for name in self.param_names])

        # restore cached branch probabilities if necessary
        if self.params_mle is not None:
            self._renew_cache()

        return ll

    def _renew_cache(self, params: Dict[str, float] = None):
        """
        Renew the cache of branch probabilities.

        :param params: The model parameters to use for caching. If ``None``, the MLE parameters are used.
        """
        # cache the branch probabilities
        self.model.cache(params if params is not None else self.params_mle, 2 * self.n_outgroups - 1)

    def _get_mle_configs(self) -> pd.DataFrame:
        """
        Get the site configurations used for the MLE with only included sites with
        the correct number of outgroups.
        """
        # only consider sites with the full number of outgroups
        return self.configs[self.configs.n_outgroups == self.n_outgroups]

    def _get_likelihood(self) -> Callable[[List[float]], float]:
        """
        Get the likelihood function for the rate parameters.

        :return: The likelihood function.
        """
        if self.configs is None:
            raise RuntimeError("No sites available. Note that you can't call infer() yourself "
                               "when using this class with Parser or Annotator.")

        # only consider sites with the correct number of outgroups
        configs = self._get_mle_configs()

        # Set the minor base to -1 if the major allele is fixed.
        # We don't want to consider minor allele not present in the subsample
        # when optimizing the branch rates.
        configs.loc[configs.n_major == self.n_ingroups, 'minor_base'] = -1

        # make variables available in the inner function
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

            # get the probability for each site and major allele
            p_sites[:, 0] = MaximumLikelihoodAncestralAnnotation.get_p_configs(
                configs=configs,
                model=model,
                base_type=BaseType.MAJOR,
                params=params
            )

            # get the probability for each site and minor allele
            p_sites[:, 1] = MaximumLikelihoodAncestralAnnotation.get_p_configs(
                configs=configs,
                model=model,
                base_type=BaseType.MINOR,
                params=params
            )

            # Return the negative log likelihood and take average over major and minor bases
            # Also multiply by the multiplicity of each site.
            # The final likelihood is the product of the likelihoods for each site.
            return -(np.log(p_sites.mean(axis=1)) * configs.multiplicity.values).sum()

        return compute_likelihood

    def _get_site_indices(self) -> np.ndarray:
        """
        Get the list of config indices for each site.

        :return: The list of config indices, use -1 for sites that are not included.
        """
        indices = np.full(self.n_sites, -1, dtype=int)

        for i, config in self.configs.iterrows():
            for j in config.sites:
                indices[j] = i

        return indices

    def _get_ancestral_from_prob(
            self,
            p_major_ancestral: np.ndarray[float] | float,
            major_base: np.ndarray[str] | str,
            minor_base: np.ndarray[str] | str
    ) -> np.ndarray[float] | float:
        """
        Get the ancestral allele from the probability of the major allele being ancestral.

        :param p_major_ancestral: The probabilities of the major allele being ancestral.
        :param major_base: The major bases.
        :param minor_base: The minor bases.
        :return: Array of ancestral alleles.
        """
        # make function accept scalars
        if isinstance(p_major_ancestral, float):
            return self._get_ancestral_from_prob(
                np.array([p_major_ancestral]),
                np.array([major_base]),
                np.array([minor_base])
            )[0]

        # initialize array
        ancestral_bases = np.full(p_major_ancestral.shape, -1, dtype=np.int8)

        ancestral_bases[p_major_ancestral >= 0.5] = major_base[p_major_ancestral >= 0.5]
        ancestral_bases[p_major_ancestral < 0.5] = minor_base[p_major_ancestral < 0.5]

        return ancestral_bases

    def _get_internal_prob(
            self,
            site: SiteConfig,
            internal: np.ndarray[int] | None = None
    ) -> float:
        """
        Get the ancestral allele for each site.

        :param site: The site configuration.
        :param internal: Base indices of internal nodes of the tree if fixed. If ``None``, the internal nodes
            are considered as free parameters. -1 also indicates a free parameter.
        :return: The ancestral allele, probability for the major being ancestral, the first base being
            ancestral, the second base being ancestral.
        """
        # get the probability for the major allele
        p_minor = self.get_p_config(
            config=site,
            base_type=BaseType.MINOR,
            params=self.params_mle,
            model=self.model,
            internal=internal
        )

        # get the probability for the minor allele
        p_major = self.get_p_config(
            config=site,
            base_type=BaseType.MAJOR,
            params=self.params_mle,
            model=self.model,
            internal=internal
        )

        return p_minor + p_major

    def _get_internal_probs(
            self,
            site: SiteConfig,
            i_internal: int,
    ) -> np.ndarray[float, (...,)]:
        """
        Get the internal probabilities for the sites used to estimate the parameters.

        :param site: The site configuration.
        :param i_internal: The index of the internal node.
        :return: The probabilities for each base and site.
        """
        # number of outgroups considered
        n_outgroups = len(site.outgroup_bases)

        # no internal nodes if there are fewer than two outgroups
        if n_outgroups < 2:
            return np.full(4, self._get_internal_prob(site))

        # initialize internal nodes
        internal = np.full(len(site.outgroup_bases), fill_value=-1, dtype=int)

        # initialize probabilities
        probs = np.zeros(shape=4, dtype=float)

        # get the internal node probabilities
        for j in range(4):
            internal[i_internal] = j
            probs[j] = self._get_internal_prob(site, internal=internal)

        return probs

    def get_inferred_site_info(self) -> Generator[SiteInfo, None, None]:
        """
        Get the site information for the sites included in the parsing process. The sites are in the same order as
        parsed. You can use :meth:`get_site_info` to get the site information for a specific site.

        :return: A generator yielding a dictionary with the site information (see :meth:`get_site_info`).
        :raises RuntimeError: If the subsample mode is ``probabilistic``.
        """
        # check if data is provided using a VCF file
        if self.subsample_mode == 'probabilistic':
            raise RuntimeError("get_inferred_site_info() not implemented with probabilistic subsampling.")

        # get config indices for each site
        indices = self._get_site_indices()

        # remove sites that are not included
        indices = indices[indices != -1]

        # get the sites
        sites = self.configs.iloc[indices]

        # iterate over the sites
        for site in sites.itertuples():
            yield self.get_site_info(
                n_major=site.n_major,
                major_base=site.major_base,
                minor_base=site.minor_base,
                outgroup_bases=site.outgroup_bases,
                pass_indices=True
            )

    def _get_site_info(self, configs: List[SiteConfig]) -> SiteInfo:
        """
        Get information on the specified sites using the inferred parameters.

        :param configs: The site configurations with differing numbers of major alleles with their multiplicities
            summing up to 1.
        :return: The site information.
        """
        if self.params_mle is None:
            raise RuntimeError("No maximum likelihood parameters available.")

        # use most likely configuration as reference
        i_max = np.argmax([c.multiplicity for c in configs])
        ref = configs[i_max]

        # get the probability for the minor allele
        p_minor = self.get_p_config(
            config=ref,  # use first config as representative
            base_type=BaseType.MINOR,
            params=self.params_mle,
            model=self.model
        )

        # get the probability for the major allele
        p_major = self.get_p_config(
            config=ref,  # use first config as representative
            base_type=BaseType.MAJOR,
            params=self.params_mle,
            model=self.model
        )

        # get the probability that the major allele is ancestral rather than the minor allele
        p_major_ancestral_probs = self._calculate_p_major_ancestral(
            p_minor=np.array([p_minor if c.minor_base == ref.minor_base else p_major for c in configs]),
            p_major=np.array([p_major if c.major_base == ref.major_base else p_minor for c in configs]),
            n_major=np.array([c.n_major for c in configs])
        )

        # configs for which the minor allele turned out to be the major allele in the subsample
        alt_config = np.array([c.minor_base != ref.minor_base for c in configs])

        p_major_ancestral_probs[alt_config] = 1 - p_major_ancestral_probs[alt_config]

        # take the weighted average
        weights = np.array([c.multiplicity for c in configs])
        p_major_ancestral = (p_major_ancestral_probs * weights).sum()

        # get the ancestral alleles using p_major_ancestral
        major_ancestral = self.get_base_string(self._get_ancestral_from_prob(
            p_major_ancestral=p_major_ancestral,
            major_base=ref.major_base,
            minor_base=ref.minor_base
        ))

        # ancestral base probabilities for the first node
        p_bases_first_node = self._get_internal_probs(site=ref, i_internal=0)

        # get the base probabilities for the first node
        total = p_bases_first_node.sum()
        i_max = np.argmax(p_bases_first_node)
        p_first_node_ancestral = p_bases_first_node[i_max] / total if total > 0 else 0
        first_node_ancestral = self.get_base_string(i_max)

        return SiteInfo(
            n_major={config.n_major: config.multiplicity for config in configs},
            major_base=self.get_base_string(ref.major_base),
            minor_base=self.get_base_string(ref.minor_base),
            outgroup_bases=list(self.get_base_string(np.array(ref.outgroup_bases))),
            p_minor=p_minor,
            p_major=p_major,
            p_major_ancestral=p_major_ancestral,
            major_ancestral=major_ancestral,
            p_bases_first_node=dict(zip(bases, p_bases_first_node)),
            p_first_node_ancestral=p_first_node_ancestral,
            first_node_ancestral=first_node_ancestral,
            rate_params=self.params_mle
        )

    def get_site_info(
            self,
            n_major: int,
            major_base: int | str,
            minor_base: int | str,
            outgroup_bases: List[int | str] | np.ndarray[int | str],
            pass_indices: bool = False
    ) -> SiteInfo:
        """
        Get information on the specified sites using the inferred parameters.

        :param n_major: The number of copies of the major allele.
        :param major_base: The major bases indices or strings.
        :param minor_base: The minor bases indices or strings.
        :param outgroup_bases: The outgroup base indices or strings.
        :param pass_indices: Whether to pass the indices as strings or convert them to integers.
        :return: The site information.
        """
        if not pass_indices:
            major_base = self.get_base_index(major_base)
            minor_base = self.get_base_index(minor_base)
            outgroup_bases = self.get_base_index(np.array(outgroup_bases))

        # initialize site configuration
        config = SiteConfig(
            n_major=n_major,
            major_base=major_base,
            minor_base=minor_base,
            outgroup_bases=outgroup_bases
        )

        return self._get_site_info([config])

    def _calculate_p_major_ancestral(
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
        # return empty array if p_minor is empty
        if isinstance(p_minor, np.ndarray) and len(p_minor) == 0:
            return np.array([])

        try:
            if self.prior is not None:
                # polarization prior for the major allele
                pi = self.p_polarization[self.n_ingroups - n_major]

                # get the probability that the major allele is ancestral
                return pi * p_major / (pi * p_major + (1 - pi) * p_minor)

            # get the probability that the major allele is ancestral
            return p_major / (p_major + p_minor)

        # only occurs when we deal with scalars
        except ZeroDivisionError:
            return np.nan

    @staticmethod
    def _is_confident(threshold: float, p: float) -> bool:
        """
        Whether we are confident enough about the ancestral allele state.

        :param threshold: Confidence threshold.
        :param p: Probability of the major allele being ancestral as opposed to the minor allele.
        :return: Whether we are confident enough.
        """
        return not (1 - threshold) / 2 < p < 1 - (1 - threshold) / 2

    def annotate_site(self, variant: Variant | DummyVariant):
        """
        Annotate a single site.

        :param variant: The variant to annotate.
        :return: The annotated variant.
        """
        # set default values
        ancestral_base = '.'
        ancestral_info = '.'

        # use maximum parsimony if we don't have an SNP
        if isinstance(variant, DummyVariant) or not variant.is_snp:
            ancestral_base = MaximumParsimonyAncestralAnnotation._get_ancestral(variant, self._ingroup_mask)
            ancestral_info = 'monomorphic'

            # increase the number of annotated sites
            self.n_annotated += 1

        else:

            configs = self._parse_variant(variant)

            if configs is not None:
                site = self._get_site_info(configs)

                # only proceed if the ancestral allele is known
                if site.major_ancestral in bases:

                    # get site information dictionary
                    site_dict = site.__dict__

                    # update info
                    ancestral_info = str(site_dict)

                    # only proceed with annotation if the confidence is high enough
                    if self._is_confident(self.confidence_threshold, site.p_major_ancestral):

                        # we take most likely configuration as reference
                        ref = configs[np.argmax([c.multiplicity for c in configs])]

                        # obtain ad hoc annotation for sanity checking
                        site_info_ad_hoc = _AdHocAncestralAnnotation._get_site_info(ref)

                        # log warning if ad hoc and maximum likelihood annotation disagree
                        if site_info_ad_hoc['ancestral_base'] != site.major_ancestral:
                            self._logger.debug(
                                "Mismatch with ad hoc ancestral allele annotation: " +
                                str(dict(
                                    site=f"{variant.CHROM}:{variant.POS}",
                                    ancestral_base_ad_hoc=site_info_ad_hoc['ancestral_base'],
                                ) | site_dict)
                            )

                            # append site to mismatches
                            self.mismatches.append(site)

                        # update ancestral base
                        ancestral_base = site.major_ancestral

                        # increase the number of annotated sites
                        self.n_annotated += 1

        # set the ancestral allele
        variant.INFO[self._handler.info_ancestral] = ancestral_base

        # set info field
        variant.INFO[self._handler.info_ancestral + "_info"] = ancestral_info

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
        :param ax: Axes to plot on. Only for Python visualization backend.
        :param ylabel: Label for y-axis.
        :return: Axes object
        """
        return Visualization.plot_scatter(
            values=self.likelihoods,
            file=file,
            show=show,
            title=title,
            scale=scale,
            ax=ax,
            ylabel=ylabel,
        )

    def get_folded_spectra(
            self,
            groups: List[Literal['major_base', 'minor_base', 'outgroup_bases']] = ['major_base'],
    ) -> Spectra:
        """
        Get the folded spectra for the parsed sites (used to estimate the parameters).

        :param groups: The groups to group the spectra by.
        :return: Spectra object
        """
        configs = self._get_mle_configs()

        # group by n_major and groups
        grouped = configs.groupby(['n_major'] + groups).sum()

        if len(groups) == 0:
            index = np.arange(self.n_ingroups + 1)
        else:
            # new index to include all possible values for n_major
            index = pd.MultiIndex.from_product(
                [np.arange(self.n_ingroups + 1).tolist()] + grouped.index.levels[1:],
                names=['n_major'] + groups
            )

        # reindex
        grouped = grouped.reindex(index, fill_value=0)

        # if we only group by n_major
        if len(groups) == 0:
            return Spectra.from_dict(dict(all=grouped.multiplicity[::-1].tolist()))

        # iterate over groups
        spectra = {}
        for i, group in grouped.groupby(level=groups):

            if not isinstance(i, tuple):
                name = f"{groups[0]}={self.get_base_string(i)}"
            else:
                name = ", ".join([f"{a}={self.get_base_string(b)}" for a, b in zip(groups, i)])
            spectra[name] = group.multiplicity[::-1].tolist()

        return Spectra.from_dict(spectra)

    @staticmethod
    def _get_branch(params: Dict[str, float], i: int) -> float:
        """
        Get the branch rate for the given index.

        :param params: The parameters.
        :param i: The index.
        :return: The branch rate.
        """
        return params['K'] if 'K' in params else params[f'K{i}']

    def get_outgroup_divergence(self) -> np.ndarray[float]:
        """
        Get the inferred branch rates between the ingroup and outgroups by combining the inferred branch rates.

        :return: One rate for each outgroup.
        """
        if self.params_mle is None:
            raise RuntimeError("No maximum likelihood parameters available.")

        # initialize array
        rates = np.zeros(self.n_outgroups, dtype=float)

        for i in range(self.n_outgroups):
            # if it's not the last outgroup
            if i < self.n_outgroups - 1:
                ingroup = [self._get_branch(self.params_mle, 2 * j) for j in range(i + 1)]
                outgroup = self._get_branch(self.params_mle, 2 * i + 1)
            else:
                ingroup = [self._get_branch(self.params_mle, 2 * j) for j in range(i)]
                outgroup = self._get_branch(self.params_mle, 2 * i)

            rates[i] = np.sum(ingroup + [outgroup])

        return rates


class _AdHocAncestralAnnotation(_OutgroupAncestralAlleleAnnotation):
    """
    Ad-hoc ancestral allele annotation using simple rules. Used for testing and sanity checking.
    """

    @staticmethod
    def _get_site_info(config: SiteConfig) -> dict:
        """
        Get site information from the site configuration.

        :param config: The site configuration.
        :return: Dictionary of with the key 'ancestral_base', denoting the ancestral base string.
        """
        # get ingroup and outgroup bases
        # noinspection PyTypeChecker
        bases_combined = np.concatenate(([config.major_base], [config.minor_base], config.outgroup_bases))

        # get scores for each base
        # noinspection PyTypeChecker
        scores = np.concatenate(([1.2], [1], [1 for _ in range(1, len(config.outgroup_bases) + 1)]))

        # get valid bases
        is_valid = bases_combined != -1

        # remove missing bases
        valid_bases = bases_combined[is_valid]

        # get valid scores
        valid_scores = scores[is_valid]

        # return missing if no valid bases
        if len(valid_bases) == 0:
            return dict(
                ancestral_base='.'
            )

        # get sum for each base
        score = np.array([np.sum(valid_scores[valid_bases == i]) for i in range(4)])

        # take most common base as ancestral
        ancestral_base = bases[score.argmax()]

        return dict(
            ancestral_base=ancestral_base
        )

    def annotate_site(self, variant: Variant | DummyVariant):
        """
        Annotate a single sites. Mono-allelic sites are assigned the major allele as ancestral. Sites with
        more than two alleles are ignored.

        :param variant: The variant to annotate.
        :return: The annotated variant.
        """
        ancestral_base = '.'
        ancestral_info = '.'

        # use maximum parsimony if we have a mono-allelic site
        if isinstance(variant, DummyVariant) or not variant.is_snp:
            ancestral_base = MaximumParsimonyAncestralAnnotation._get_ancestral(variant, self._ingroup_mask)
            ancestral_info = 'monomorphic'

        # parse the site
        configs = self._parse_variant(variant)

        if configs is not None:

            # use most likely configuration as reference
            ref = configs[np.argmax([c.multiplicity for c in configs])]

            # get site information dictionary
            site = self._get_site_info(ref)

            # only proceed if the ancestral allele is known
            if site['ancestral_base'] in bases:

                if site['major_base'] != site['ancestral_base']:
                    self._logger.debug(dict(site=f"{variant.CHROM}:{variant.POS}") | site)

                ancestral_base = site['ancestral_base']
                ancestral_info = str(site)

        # set the ancestral allele
        variant.INFO[self._handler.info_ancestral] = ancestral_base

        # set info field
        variant.INFO[self._handler.info_ancestral + "_info"] = ancestral_info

        # increase the number of annotated sites
        self.n_annotated += 1


class _ESTSFSAncestralAnnotation(AncestralAlleleAnnotation):
    """
    A wrapper around EST-SFS. Used for testing.
    """

    def __init__(
            self,
            anc: MaximumLikelihoodAncestralAnnotation
    ):
        """
        Create a new ESTSFSAncestralAnnotation instance.

        :param anc:
        """
        super().__init__()

        #: The ancestral annotation.
        self.anc = anc

        #: The likelihoods for each run.
        self.likelihoods: np.ndarray[float] | None = None

        #: The minimum likelihood.
        self.likelihood: float | None = None

        #: The MLE parameters.
        self.params_mle: Dict[str, float] | None = None

        #: The probabilities for each site.
        self.probs: pd.DataFrame | None = None

    def create_seed_file(self, seed_file: str):
        """
        Create the seed file.

        :param seed_file: Path to the seed file.
        """
        with open(seed_file, 'w') as f:
            f.write(str(self.anc.seed))

    def create_config_file(self, config_file: str):
        """
        Create the config file.

        :param config_file: Path to the config file.
        """
        models = dict(
            JCSubstitutionModel=0,
            K2SubstitutionModel=1
        )

        with open(config_file, 'w') as f:
            f.write(f"n_outgroup {self.anc.n_outgroups}\n")
            f.write(f"model {models[self.anc.model.__class__.__name__]}\n")
            f.write(f"nrandom {self.anc.n_runs}\n")

    def infer(
            self,
            binary: str = 'EST_SFS',
            wd: str = None,
            execute: Callable = None,
    ):
        """
        Infer the ancestral allele using EST-SFS.

        :param binary: The path to the EST-SFS binary.
        :param wd: The working directory.
        :param execute: The function to execute the bash command.
        """
        # define default function for executing command
        if execute is None:
            def shell(command: str):
                """
                Execute shell command.

                :param command: Command string
                """
                return subprocess.run(command, check=True, cwd=wd, shell=True)

            execute = shell

        with tempfile.NamedTemporaryFile('w') as sites_file, \
                tempfile.NamedTemporaryFile('w') as seed_file, \
                tempfile.NamedTemporaryFile('w') as config_file, \
                tempfile.NamedTemporaryFile('w') as out_sfs, \
                tempfile.NamedTemporaryFile('w') as out_p:
            # create the sites file
            self.anc.to_est_sfs(sites_file.name)

            # create the seed file
            self.create_seed_file(seed_file.name)

            # create the config file
            self.create_config_file(config_file.name)

            # construct command string
            command = (f"{binary} "
                       f"{config_file.name} "
                       f"{sites_file.name} "
                       f"{seed_file.name} "
                       f"{out_sfs.name} "
                       f"{out_p.name} ")

            # log command signature
            self._logger.info(f"Running: '{command}'")

            # execute command
            execute(command)

            self.parse_est_sfs_output(out_p.name)

    def parse_est_sfs_output(self, file: str):
        """
        Parse the output of the EST-SFS program containing the site probabilities.

        :param file: The file name.
        :return: The data frame.
        """
        # filter out lines starting with 0
        filtered_lines = []
        with open(file, 'r') as f:
            for i, line in enumerate(f):

                # strip line
                line = line.strip()

                if line.startswith('0'):
                    if i == 4:
                        # parse likelihoods
                        self.likelihoods = np.array(line.split()[2:], dtype=float)
                        self.likelihood = np.min(self.likelihoods)

                    if i == 5:
                        # parse MLE parameters
                        data = np.array(line.split()[2:])
                        self.params_mle = dict(zip([d.upper() for d in data[::2]], data[1::2].astype(float)))

                    if i == 6 and isinstance(self.anc.model, K2SubstitutionModel):
                        # parse kappa
                        self.params_mle['k'] = float(line.split()[2])
                else:
                    filtered_lines.append(line.strip())

        # read into dataframe
        self.probs = pd.read_csv(StringIO('\n'.join(filtered_lines)), sep=" ", header=None)

        # drop the first column
        self.probs.drop(self.probs.columns[0], axis=1, inplace=True)

        # rename columns
        self.probs.rename(columns={1: 'config', 2: 'prob'}, inplace=True)

    def annotate_site(self, variant: Variant | DummyVariant):
        """
        Not implemented.

        :param variant: The variant to annotate.
        :raises: NotImplementedError
        """
        raise NotImplementedError


class Annotator(MultiHandler):
    """
    Annotate a VCF file with the given annotations.

    Example usage:

    ::

        import fastdfe as fd

        ann = fd.Annotator(
            vcf="http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/"
                "1000_genomes_project/release/20181203_biallelic_SNV/"
                "ALL.chr21.shapeit2_integrated_v1a.GRCh38.20181129.phased.vcf.gz",
            fasta="http://ftp.ensembl.org/pub/release-109/fasta/homo_sapiens/"
                  "dna/Homo_sapiens.GRCh38.dna.chromosome.21.fa.gz",
            gff="http://ftp.ensembl.org/pub/release-109/gff3/homo_sapiens/"
                "Homo_sapiens.GRCh38.109.chromosome.21.gff3.gz",
            output='sapiens.chr21.degeneracy.vcf.gz',
            annotations=[fd.DegeneracyAnnotation()],
            aliases=dict(chr21=['21'])
        )

        ann.annotate()

    """

    def __init__(
            self,
            vcf: str,
            output: str,
            annotations: List[Annotation],
            gff: str | None = None,
            fasta: str | None = None,
            info_ancestral: str = 'AA',
            max_sites: int = np.inf,
            seed: int | None = 0,
            cache: bool = True,
            aliases: Dict[str, List[str]] = {},
    ):
        """
        Create a new annotator instance.

        :param vcf: The path to the VCF file, can be gzipped, urls are also supported
        :param output: The path to the output file
        :param annotations: The annotations to apply.
        :param gff: The path to the GFF file, can be gzipped, urls are also supported. Required for
            annotations that require a GFF file.
        :param fasta: The path to the FASTA file, can be gzipped, urls are also supported. Required for
            annotations that require a FASTA file.
        :param info_ancestral: The tag in the INFO field that contains the ancestral allele
        :param max_sites: Maximum number of sites to consider
        :param seed: Seed for the random number generator. Use ``None`` for no seed.
        :param cache: Whether to cache files downloaded from urls
        :param aliases: Dictionary of aliases for the contigs in the VCF file, e.g. ``{'chr1': ['1']}``.
            This is used to match the contig names in the VCF file with the contig names in the FASTA file and GFF file.

        """
        super().__init__(
            vcf=vcf,
            gff=gff,
            fasta=fasta,
            info_ancestral=info_ancestral,
            max_sites=max_sites,
            seed=seed,
            cache=cache,
            aliases=aliases
        )

        #: The path to the output file.
        self.output: str = output

        #: The annotations to apply.
        self.annotations: List[Annotation] = annotations

        #: The VCF writer.
        self._writer: Writer | None = None

    def _setup(self):
        """
        Set up the annotator.
        """
        for annotation in self.annotations:
            annotation._setup(self)

        # create the writer
        self._writer = Writer(self.output, self._reader)

    def _teardown(self):
        """
        Tear down the annotator.
        """
        for annotation in self.annotations:
            annotation._teardown()

        # close the writer and reader
        self._writer.close()
        self._reader.close()

    def annotate(self):
        """
        Annotate the VCF file.
        """
        self._logger.info('Start annotating')

        # set up the annotator
        self._setup()

        # get progress bar
        with self.get_pbar(desc=f"{self.__class__.__name__}>Processing sites") as pbar:

            # iterate over the sites
            for i, variant in enumerate(self._reader):

                # apply annotations
                for annotation in self.annotations:
                    annotation.annotate_site(variant)

                # write the variant
                self._writer.write_record(variant)

                # update the progress bar
                pbar.update()

                # explicitly stopping after ``n`` sites fixes a bug with cyvcf2:
                # 'error parsing variant with `htslib::bcf_read` error-code: 0 and ret: -2'
                if i + 1 == self.n_sites or i + 1 == self.max_sites:
                    break

        # tear down the annotator
        self._teardown()
