"""
VCF annotators.
"""

__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"
__date__ = "2023-05-09"

import logging
import re
from abc import ABC
from collections import Counter
from functools import cached_property
from itertools import product
from typing import List, Optional, Dict

import Bio.Data.CodonTable
import numpy as np
import pandas as pd
from Bio.Seq import Seq
from cyvcf2 import Variant, Writer, VCF
from pyfaidx import Fasta, FastaRecord

from .vcf import VCFHandler, get_called_bases

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
        #: The annotator.
        self.annotator: Annotator | None = None

        #: The number of annotated sites.
        self.n_annotated: int = 0

    def _setup(self, annotator: 'Annotator'):
        """
        Provide context by passing the annotator. This should be called before the annotation starts.

        :param annotator: The annotator.
        """
        self.annotator = annotator

    def _add_info(self, reader: VCF):
        """
        Add info fields to the header.

        :param reader: The reader.
        """
        pass

    def _teardown(self):
        """
        Finalize the annotation. Called after all sites have been annotated.
        """
        logger.info(type(self).__name__ + f': Annotated {self.n_annotated} sites.')

    def annotate_site(self, variant: Variant):
        """
        Annotate a single site.

        :param variant: The variant to annotate.
        :return: The annotated variant.
        """
        pass

    @staticmethod
    def _load_cds(file: str) -> pd.DataFrame:
        """
        Load coding sequences from a GFF file.

        :param file: The path to The GFF file path, possibly gzipped or a URL
        :return: The DataFrame.
        """
        # download and unzip if necessary
        local_file = VCFHandler.unzip_if_zipped(VCFHandler.download_if_url(file))

        # column labels for GFF file
        col_labels = ['seqid', 'source', 'type', 'start', 'end', 'score', 'strand', 'phase', 'attributes']

        dtypes = dict(
            seqid='category',
            type='category',
            start=int,
            end=int,
            strand='category',
            phase='category'
        )

        logger.info(f'Loading GFF file.')

        # load GFF file
        df = pd.read_csv(
            local_file,
            sep='\t',
            comment='#',
            names=col_labels,
            dtype=dtypes,
            usecols=['seqid', 'type', 'start', 'end', 'strand', 'phase']
        )

        # filter for coding sequences
        df = df[df['type'] == 'CDS']

        # drop type column
        df.drop(columns=['type'], inplace=True)

        # sort by seqid and start
        df.sort_values(by=['seqid', 'start'], inplace=True)

        return df


class AncestralAlleleAnnotation(Annotation, ABC):
    """
    Base class for ancestral allele annotation.
    """

    def _add_info(self, reader: VCF):
        """
        Add info fields to the header.

        :param reader: The reader.
        """
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
    """

    def annotate_site(self, variant: Variant):
        """
        Annotate a single site.

        :param variant: The variant to annotate.
        :return: The annotated variant.
        """
        # get the called bases
        b = get_called_bases(variant)

        if len(b) != 0:
            # get the major allele
            major_allele = Counter(b).most_common(1)[0][0]
        else:
            major_allele = '.'

        # set the ancestral allele
        variant.INFO[self.annotator.info_ancestral] = major_allele

        # increase the number of annotated sites
        self.n_annotated += 1


class DegeneracyAnnotation(Annotation):
    """
    Degeneracy annotation. We annotate the degeneracy by looking at each codon for coding variants.
    This also annotates mono-allelic sites.

    This annotation adds the info fields ``Degeneracy`` and ``Degeneracy_Info``, which hold the degeneracy
    of a site (0, 2, 4) and extra information about the degeneracy, respectively. To be used with
    :class:`~fastdfe.parser.DegeneracyStratification`.
    """

    #: The genomic positions for coding sequences that are mocked.
    _pos_mock: int = 1e100

    def __init__(self, gff_file: str, fasta_file: str, aliases: Dict[str, List[str]] = {}):
        """
        Create a new annotation instance.

        :param gff_file: The GFF file path, possibly gzipped or a URL
        :param fasta_file: The FASTA file path, possibly gzipped or a URL
        :param aliases: Dictionary of aliases for the contigs in the VCF file, e.g. ``{'chr1': ['1']}``.
            This is used to match the contig names in the VCF file with the contig names in the FASTA file and GFF file.
        """
        super().__init__()

        #: The GFF file.
        self.gff_file: str = gff_file

        #: The FASTA file.
        self.fasta_file: str = fasta_file

        #: Dictionary of aliases for the contigs in the VCF file
        self.aliases: Dict[str, List[str]] = aliases

        #: The current coding sequence or the closest coding sequence downstream.
        self.cd: Optional[pd.Series] = None

        #: The coding sequence following the current coding sequence.
        self.cd_next: Optional[pd.Series] = None

        #: The coding sequence preceding the current coding sequence.
        self.cd_prev: Optional[pd.Series] = None

        #: The current contig.
        self.contig: Optional[FastaRecord] = None

        #: The variants that could not be annotated correctly.
        self.mismatches: List[Variant] = []

        #: The variant that were skipped because they were not in coding regions.
        self.n_skipped = 0

        #: The variants for which the codon could not be determined.
        self.errors: List[Variant] = []

    @cached_property
    def _ref(self) -> Fasta:
        """
        Get the reference reader.

        :return: The reference reader.
        """
        return VCFHandler.load_fasta(self.fasta_file)

    @cached_property
    def _cds(self) -> pd.DataFrame:
        """
        The coding sequences.

        :return: The coding sequences.
        """
        return Annotation._load_cds(self.gff_file)

    def _setup(self, annotator: 'Annotator'):
        """
        Provide context to the annotator.

        :param annotator: The annotator.
        """
        super()._setup(annotator)

        # touch the cached properties to make for a nicer logging experience
        # noinspection PyStatementEffect
        self._cds

        # noinspection PyStatementEffect
        self._ref

    def _add_info(self, reader: VCF):
        """
        Add info fields to the header.

        :param reader: The reader.
        """
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

        # We assume here that cd_prev and cd_next have the same orientation
        # Use final positions from previous coding sequence if current codon
        # starts before start position of current coding sequence
        if codon_pos[1] == self.cd.start:
            codon_pos[0] = self.cd_prev.end
        elif codon_pos[2] == self.cd.start:
            codon_pos[1] = self.cd_prev.end
            codon_pos[0] = self.cd_prev.end - 1

        if self.cd_next is None and codon_pos[2] > self.cd.end:
            raise IndexError(f'Codon at site {variant.CHROM}:{variant.POS} overlaps with '
                             f'end position of current CDS and no subsequent CDS was given.')

        # use initial positions from subsequent coding sequence if current codon
        # ends before end position of current coding sequence
        if codon_pos[2] == self.cd.end + 1:
            codon_pos[2] = self.cd_next.start
        elif codon_pos[1] == self.cd.end + 1:
            codon_pos[1] = self.cd_next.start
            codon_pos[2] = self.cd_next.start + 1

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

        # We assume here that cd_prev and cd_next have the same orientation.
        # Use final positions from previous coding sequence if current codon
        # ends before start position of current coding sequence.
        if codon_pos[1] == self.cd.start:
            codon_pos[2] = self.cd_prev.end
        elif codon_pos[0] == self.cd.start:
            codon_pos[1] = self.cd_prev.end
            codon_pos[2] = self.cd_prev.end - 1

        if self.cd_next is None and codon_pos[0] > self.cd.end:
            raise IndexError(f'Codon at site {variant.CHROM}:{variant.POS} overlaps with '
                             f'end position of current CDS and no subsequent CDS was given.')

        # use initial positions from subsequent coding sequence if current codon
        # starts before end position of current coding sequence
        if codon_pos[0] == self.cd.end + 1:
            codon_pos[0] = self.cd_next.start
        elif codon_pos[1] == self.cd.end + 1:
            codon_pos[0] = self.cd_next.start + 1
            codon_pos[1] = self.cd_next.start

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
        aliases = VCFHandler.get_aliases(v.CHROM, self.aliases)

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

                logger.debug(f'Found coding sequence: {self.cd.seqid}:{self.cd.start}-{self.cd.end}, '
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
                logger.warning(f"No coding sequence found on all of contig '{v.CHROM}' and no previous "
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
        aliases = VCFHandler.get_aliases(v.CHROM, self.aliases)

        # fetch contig if not up to date
        if self.contig is None or self.contig.name not in aliases:
            self.contig = VCFHandler.get_contig(self._ref, aliases)

            logger.debug(f"Fetching contig '{self.contig.name}'.")

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
        if self.cd.seqid in VCFHandler.get_aliases(v.CHROM, self.aliases) and self.cd.start <= v.POS <= self.cd.end:

            try:
                # parse codon
                codon, codon_pos, codon_start, pos_codon, pos_rel = self._parse_codon(v)

            except IndexError as e:

                # skip site on IndexError
                logger.warning(e)
                self.errors.append(v)
                return

            # make sure the reference allele matches with the position on the reference genome
            if str(self.contig[v.POS - 1]).upper() != v.REF.upper():
                logger.warning(f"Reference allele does not match with reference genome at {v.CHROM}:{v.POS}.")
                self.mismatches.append(v)
                return

            degeneracy = '.'
            if 'N' not in codon:
                degeneracy = self._get_degeneracy(codon, pos_codon)

                # increment counter of annotated sites
                self.n_annotated += 1

            v.INFO['Degeneracy'] = degeneracy
            v.INFO['Degeneracy_Info'] = f"{pos_codon},{self.cd.strand},{codon}"

            logger.debug(f'pos codon: {pos_codon}, pos abs: {v.POS}, '
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

    def _add_info(self, reader: VCF):
        """
        Add the INFO field to the VCF header.

        :param reader: The VCF reader.
        """
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

    @staticmethod
    def _parse_codons_vep(variant: Variant) -> List[str]:
        """
        Parse the codons from the VEP annotation if present.

        :param variant: The variant.
        :return: The codons.
        """
        # match codons
        match = re.search("([actgACTG]{3})/([actgACTG]{3})", variant.INFO.get('CSQ'))

        if match is not None:
            if len(match.groups()) != 2:
                logger.info(f'VEP annotation has more than two codons: {variant.INFO.get("CSQ")}')

            return [m.upper() for m in [match[1], match[2]]]

        return []

    @staticmethod
    def _parse_synonymy_snpeff(variant: Variant) -> int | None:
        """
        Parse the synonymy from the annotation provided by SnpEff

        :param variant: The variant.
        :return: The codons.
        """
        if 'synonymous_variant' in variant.INFO.get('ANN'):
            return 1

        if 'missense_variant' in variant.INFO.get('ANN'):
            return 0

        return None

    def _teardown(self):
        """
        Finalize the annotation.
        """
        super()._teardown()

        logger.info(f'Number of mismatches with VEP: {len(self.vep_mismatches)}')

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
                    logger.warning(e)
                    self.errors.append(v)
                    return

                # make sure the reference allele matches with the position in the reference genome
                if str(self.contig[v.POS - 1]).upper() != v.REF.upper():
                    logger.warning(f"Reference allele does not match with reference genome at {v.CHROM}:{v.POS}.")
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
                                logger.warning(f'VEP codons do not match with codons determined by '
                                               f'codon table for {v.CHROM}:{v.POS}')

                                self.vep_mismatches.append(v)
                                return

                if v.INFO.get('ANN') is not None:
                    synonymy_snpeff = self._parse_synonymy_snpeff(v)

                    self.n_snpeff_comparisons += 1

                    if synonymy_snpeff is not None:
                        if synonymy_snpeff != synonymy:
                            logger.warning(f'SnpEff annotation does not match with custom '
                                           f'annotation for {v.CHROM}:{v.POS}')

                            self.snpeff_mismatches.append(v)
                            return

                # increase number of annotated sites
                self.n_annotated += 1

                # add to info field
                v.INFO['Synonymy'] = synonymy
                v.INFO['Synonymy_Info'] = info


class Annotator(VCFHandler):
    """
    Annotator base class.
    """

    def __init__(
            self,
            vcf: str,
            output: str,
            annotations: List[Annotation],
            info_ancestral: str = 'AA',
            max_sites: int = np.inf,
            seed: int | None = 0
    ):
        """
        Create a new annotator instance.

        :param vcf: The path to the VCF file, can be gzipped, urls are also supported
        :param output: The path to the output file
        :param annotations: The annotations to apply.
        :param info_ancestral: The tag in the INFO field that contains the ancestral allele
        :param max_sites: Maximum number of sites to consider
        :param seed: Seed for the random number generator
        """
        super().__init__(
            vcf=vcf,
            info_ancestral=info_ancestral,
            max_sites=max_sites,
            seed=seed
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
        reader = VCF(self.vcf)

        # provide annotator to annotations and add info fields
        for annotation in self.annotations:
            annotation._setup(self)
            annotation._add_info(reader)

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
