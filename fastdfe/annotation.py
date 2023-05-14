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
from itertools import product
from typing import List, Optional, Dict

import Bio.Data.CodonTable
import numpy as np
import pandas as pd
from Bio import SeqIO, SeqRecord
from Bio.Seq import Seq
from Bio.SeqIO.FastaIO import FastaIterator
from cyvcf2 import Variant, Writer, VCF

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

    def provide_context(self, annotator: 'Annotator'):
        """
        Provide context by passing the annotator. This should be called before the annotation starts.

        :param annotator: The annotator.
        """
        self.annotator = annotator

    def add_info(self, reader: VCF):
        """
        Add info fields to the header.

        :param reader: The reader.
        """
        pass

    def finalize(self):
        """
        Finalize the annotation. Called after all sites have been annotated.
        """
        logger.info(f'Annotated {self.n_annotated} sites.')

    def annotate_site(self, variant: Variant):
        """
        Annotate a single site.

        :param variant: The variant to annotate.
        :return: The annotated variant.
        """
        pass

    @staticmethod
    def load_gff(file: str):
        """
        Load a GFF file into a DataFrame.

        :param file: The path to The GFF file path, possibly gzipped and possibly a URL starting with ``https://``
        :return: The DataFrame.
        """
        cols = ['seqname', 'source', 'feature', 'start', 'end', 'score', 'strand', 'phase', 'attribute']

        # download and unzip if necessary
        local_file = VCFHandler.unzip_if_zipped(VCFHandler.download_if_url(file))

        return pd.read_csv(local_file, sep='\t', header=None, comment='#', names=cols)

    @staticmethod
    def load_cds(file: str):
        """
        Load the coding sequences from a GFF file into a DataFrame.

        :param file: The path to The GFF file path, possibly gzipped and possibly a URL starting with ``https://``
        :return: The DataFrame.
        """
        # create data frame
        df = Annotation.load_gff(file)

        # prepare list of coding sequences
        df = df[df.feature == 'CDS']

        # sort by start position
        df = df.sort_values('start')

        # convert to integer
        df['phase'] = pd.to_numeric(df['phase'], downcast='integer')

        # remove duplicates
        df = df.drop_duplicates(subset=['seqname', 'start'])

        # reset index
        df = df.reset_index(drop=True)

        return df

    @staticmethod
    def load_fasta(file: str) -> FastaIterator:
        """
        Load a FASTA file into a dictionary.

        :param file: The path to The FASTA file path, possibly gzipped and possibly a URL starting with ``https://``
        :return: Iterator over the sequences.
        """
        # download and unzip if necessary
        local_file = VCFHandler.unzip_if_zipped(VCFHandler.download_if_url(file))

        return SeqIO.parse(local_file, 'fasta')


class AncestralAlleleAnnotation(Annotation, ABC):
    """
    Base class for ancestral allele annotation.
    """

    def add_info(self, reader: VCF):
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
    Annotation of ancestral alleles using maximum parsimony.
    """

    def annotate_site(self, variant: Variant):
        """
        Annotate a single site.

        :param variant: The variant to annotate.
        :return: The annotated variant.
        """
        # get the called bases
        b = get_called_bases(variant)

        # get the major allele
        major_allele = Counter(b).most_common(1)[0][0]

        # set the ancestral allele
        variant.INFO[self.annotator.info_ancestral] = major_allele

        # increase the number of annotated sites
        self.n_annotated += 1


class DegeneracyAnnotation(Annotation):
    """
    Degeneracy annotation. We annotate the degeneracy by looking at each codon for coding variants.
    """

    def __init__(self, gff_file: str, fasta_file: str):
        """
        Create a new annotation instance.

        :param gff_file: The GFF file path, possibly gzipped and possibly a URL starting with ``https://``
        :param fasta_file: The FASTA file path, possibly gzipped and possibly a URL starting with ``https://``
        """
        super().__init__()

        #: The GFF file.
        self.gff_file: str = gff_file

        #: The FASTA file.
        self.fasta_file: str = fasta_file

        #: The reference reader.
        self.ref: FastaIterator = self.load_fasta(fasta_file)

        #: The coding sequences.
        self.cds: pd.DataFrame = self.load_cds(gff_file)

        #: The current coding sequence.
        self.cd: Optional[pd.Series] = None

        #: The previous coding sequence.
        self.cd_prev: Optional[pd.Series] = None

        #: The next coding sequence.
        self.cd_next: Optional[pd.Series] = None

        #: The current contig.
        self.contig: SeqRecord = None

        #: The current sequence.
        self.seq: Optional[str] = None

        #: The variants that could not be annotated correctly.
        self.mismatches: List[Variant] = []

        #: The variant that were skipped because they were not in coding regions.
        self.n_skipped = 0

        #: The variants for which the codon could not be determined.
        self.errors: List[Variant] = []

    def add_info(self, reader: VCF):
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

    def get_cd(self, v: Variant) -> pd.Series:
        """
        Get the coding sequence among the given coding sequences that encloses the given position.

        :param v: The variant.
        :return: the coding sequence
        :raises LookupError: if no coding sequence could be found
        """
        rows = self.cds[(self.cds.seqname == v.CHROM) & (self.cds.start <= v.POS) & (v.POS <= self.cds.end)]

        if len(rows) == 0:
            raise LookupError('No coding sequence found')

        return rows.iloc[0]

    def get_prev_cd(self) -> pd.Series:
        """
        Get the coding sequence on same gene preceding the current coding sequence.

        .. note::
            Here we don't make sure adjacent coding sequences are on the same gen,
            which should be fine in most cases.

        :return: Coding sequence or None if no previous coding sequence could be found.
        """
        rows = self.cds[(self.cds.seqname == self.cd.seqname) & (self.cds.end < self.cd.start)]

        # return the last row or None
        return rows.tail(1).iloc[0] if len(rows) else None

    def get_next_cd(self) -> pd.Series:
        """
        Get the coding sequence on same gene after the current coding sequence.

        .. note::
            Here we don't make sure adjacent coding sequences are on the same gen,
            which should be fine in most cases.

        :return: Coding sequence or None if no next coding sequence could be found.
        """
        rows = self.cds[(self.cds.seqname == self.cd.seqname) & (self.cds.start > self.cd.end)]

        # return the first row or None
        return rows.head(1).iloc[0] if len(rows) else None

    def get_contig(self, variant: Variant):
        """
        Fetch the contig the record is on. We assume that the records are passed
        in ascending order as we use a stream for accessing the contig sequences.

        :param variant: The record.
        :return: The contig and the sequence.
        :raises RuntimeError: if the contig could not be found.
        """
        try:

            contig = next(self.ref)
            seq = str(contig.seq)

            while contig.id != variant.CHROM:
                contig = next(self.ref)
                seq = str(contig.seq)

        except StopIteration:
            raise RuntimeError(f"Contig '{variant.CHROM}' not found in FASTA file.")

        return contig, seq

    def parse_codon_forward(self, variant: Variant):
        """
        Parse the codon in forward direction.

        :param variant: The variant.
        :return: Codon, Codon position, Codon start position, Position within codon, and relative position.
        """
        # position relative to start of coding sequence
        pos_rel = variant.POS - (self.cd.start + self.cd.phase)

        # position relative to codon
        pos_codon = pos_rel % 3

        # inclusive codon start, 1-based
        codon_start = variant.POS - pos_codon

        # the codon positions
        codon_pos = [codon_start, codon_start + 1, codon_start + 2]

        if self.cd_prev is None and codon_pos[0] < self.cd.start:
            raise IndexError(f'Codon at site {variant.CHROM}:{variant.POS} '
                             f'starts before current CDS and no previous CDS was given.')

        # we assume here that cd_prev and cd_next have the same orientation
        # use final positions from previous coding sequence if current codon
        # starts before start position of current coding sequence
        if codon_pos[1] == self.cd.start:
            codon_pos[0] = self.cd_prev.end
        elif codon_pos[2] == self.cd.start:
            codon_pos[1] = self.cd_prev.end
            codon_pos[0] = self.cd_prev.end - 1

        if self.cd_next is None and codon_pos[2] > self.cd.end:
            raise IndexError(f'Codon at site {variant.CHROM}:{variant.POS} '
                             f'ends after current CDS and no subsequent CDS was given.')

        # use initial positions from subsequent coding sequence if current codon
        # ends before end position of current coding sequence
        if codon_pos[2] == self.cd.end + 1:
            codon_pos[2] = self.cd_next.start
        elif codon_pos[1] == self.cd.end + 1:
            codon_pos[1] = self.cd_next.start
            codon_pos[2] = self.cd_next.start + 1

        # seq uses 0-based positions
        codon = ''.join(self.seq[pos - 1] for pos in codon_pos).upper()

        return codon, codon_pos, codon_start, pos_codon, pos_rel

    def parse_codon_backward(self, variant: Variant):
        """
        Parse the codon in reverse direction.

        :param variant: The variant.
        :return: Codon, Codon position, Codon start position, Position within codon, and relative position.
        """
        # position relative to end of coding sequence
        pos_rel = (self.cd.end - self.cd.phase) - variant.POS

        # position relative to codon end
        pos_codon = pos_rel % 3

        # inclusive codon start, 1-based
        codon_start = variant.POS + pos_codon

        # the codon positions
        codon_pos = [codon_start, codon_start - 1, codon_start - 2]

        if self.cd_prev is None and codon_pos[2] < self.cd.start:
            raise IndexError(f'Codon at site {variant.CHROM}:{variant.POS} '
                             f'starts before current CDS and no previous CDS was given.')

        # we assume here that cd_prev and cd_next have the same orientation
        # use final positions from previous coding sequence if current codon
        # ends before start position of current coding sequence
        if codon_pos[1] == self.cd.start:
            codon_pos[2] = self.cd_prev.end
        elif codon_pos[0] == self.cd.start:
            codon_pos[1] = self.cd_prev.end
            codon_pos[2] = self.cd_prev.end - 1

        if self.cd_next is None and codon_pos[0] > self.cd.end:
            raise IndexError(f'Codon at site {variant.CHROM}:{variant.POS} '
                             f'ends after current CDS and no subsequent CDS was given.')

        # use initial positions from subsequent coding sequence if current codon
        # starts before end position of current coding sequence
        if codon_pos[0] == self.cd.end + 1:
            codon_pos[0] = self.cd_next.start
        elif codon_pos[1] == self.cd.end + 1:
            codon_pos[0] = self.cd_next.start + 1
            codon_pos[1] = self.cd_next.start

        # we use 0-based positions here
        codon = ''.join(self.seq[pos - 1] for pos in codon_pos)

        # take complement and convert to uppercase ('n' might be lowercase)
        codon = str(Seq(codon).complement()).upper()

        return codon, codon_pos, codon_start, pos_codon, pos_rel

    def parse_codon(self, variant: Variant):
        """
        Parse the codon for the given variant.

        :param variant: The variant to parse the codon for.
        :return: Codon, Codon position, Codon start position, Position within codon, and relative position.
        """

        if self.cd.strand == '+':
            return self.parse_codon_forward(variant)

        return self.parse_codon_backward(variant)

    @staticmethod
    def get_degeneracy(codon: str, pos: int) -> int:
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
    def get_degeneracy_table() -> Dict[str, str]:
        """
        Create codon degeneracy table.

        :return: dictionary mapping codons to degeneracy
        """
        codon_degeneracy = {}
        for codon in product(bases, repeat=3):
            codon = ''.join(codon)
            codon_degeneracy[codon] = ''.join(
                [str(DegeneracyAnnotation.get_degeneracy(codon, pos)) for pos in range(0, 3)]
            )

        return codon_degeneracy

    def fetch_cds(self, v: Variant):
        """
        Fetch the coding sequence for the given variant.

        :param v: The variant to fetch the coding sequence for.
        :raises LookupError: If no coding sequence was found.
        """
        # fetch coding sequence if not up to date
        if self.cd is None or self.cd.seqname != v.CHROM or not (self.cd.start <= v.POS <= self.cd.end):

            try:
                self.cd = self.get_cd(v)
                self.cd_prev, self.cd_next = self.get_prev_cd(), self.get_next_cd()
            except LookupError:
                raise LookupError(f"No coding sequence found, skipping record {v.CHROM}:{v.POS}")

            logger.debug(f'Found coding sequence: {self.cd.seqname}:{self.cd.start}-{self.cd.end}, '
                         f'reminder: {(self.cd.end - self.cd.start + 1) % 3}, '
                         f'phase: {self.cd.phase}, orientation: {self.cd.strand}, '
                         f'current position: {v.CHROM}:{v.POS}')

    def fetch_contig(self, v: Variant):
        """
        Fetch the contig for the given variant.

        :param v: The variant to fetch the contig for.
        """
        # fetch contig if not up to date
        if self.contig is None or self.contig.id != v.CHROM:
            self.contig, self.seq = self.get_contig(v)

            logger.debug(f"Fetching contig '{self.contig.id}'.")

    def fetch(self, variant: Variant):
        """
        Fetch all required data for the given variant.

        :param variant:
        :raises LookupError: if some data could not be found.
        """
        self.fetch_cds(variant)
        self.fetch_contig(variant)

    def annotate_site(self, v: Variant):
        """
        Annotate a single site.

        :param v: The variant to annotate.
        """
        v.INFO['Degeneracy'] = '.'

        try:
            self.fetch(v)
        except LookupError:
            self.n_skipped += 1
            return

        # annotate if record is in coding sequence
        if self.cd.start <= v.POS <= self.cd.end:

            try:
                # parse codon
                codon, codon_pos, codon_start, pos_codon, pos_rel = self.parse_codon(v)

            except IndexError as e:

                # skip site on IndexError
                logger.warning(e)
                self.errors.append(v)
                return

            # make sure the reference allele matches with the position on the reference genome
            if self.contig[v.POS - 1] != v.REF:
                logger.warning("Reference allele does not match with reference genome.")
                self.mismatches.append(v)
                return

            degeneracy = None
            if 'N' not in codon:
                degeneracy = self.get_degeneracy(codon, pos_codon)

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
    However, as we also need to annotate monomorphic sites, this class is of limited use and mainly
    used for testing purposes.

    We also check for concordance with the prediction by VEP if present.
    """

    def __init__(self, *args, **kwargs):
        """
        Create a new SynonymyAnnotation object.

        :param args: The arguments.
        :param kwargs: The keyword arguments.
        """
        super().__init__(*args, **kwargs)

        self.vep_mismatches: List[Variant] = []

    def get_alt_allele(self, variant: Variant) -> str | None:
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
    def parse_codons_vep(variant: Variant) -> List[str]:
        """
        Parse the codons from the VEP annotation if present.

        :param variant: The variant.
        :return: The codons.
        """
        # match codons
        match = re.search("([actgACTG]{3})/([actgACTG]{3})", variant.INFO['CSQ'])

        if match is not None:
            return [m.upper() for m in [match[1], match[2]]]

        return []

    def finalize(self):
        """
        Finalize the annotation.
        """
        super().finalize()

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
                self.fetch(v)
            except LookupError:
                self.n_skipped += 1
                return

            # annotate if record is in coding sequence
            if self.cd.start <= v.POS <= self.cd.end:

                try:
                    # parse codon
                    codon, codon_pos, codon_start, pos_codon, pos_rel = self.parse_codon(v)

                except IndexError as e:

                    # skip site on IndexError
                    logger.warning(e)
                    self.errors.append(v)
                    return

                # make sure the reference allele matches with the position in the reference genome
                if self.contig[v.POS - 1] != v.REF:
                    logger.warning("Reference allele does not match with reference genome for: {v}.")
                    self.mismatches.append(v)
                    return

                # fetch the alternative allele if present
                alt = self.get_alt_allele(v)

                info = ''
                synonymy, alt_codon, codons_vep = None, None, None
                if alt is not None:
                    # alternative codon, 'n' might not be uppercase
                    alt_codon = self.mutate(codon, alt, pos_codon).upper()

                    # whether the alternative codon is synonymous
                    if 'N' not in codon and 'N' not in alt_codon:
                        synonymy = int(self.is_synonymous(codon, alt_codon))

                    # append alternative codon to info field
                    info += f'{alt_codon}'

                    # check if the alternative codon is a start codon
                    if alt_codon in start_codons:
                        info += ',start_gained'

                    # check if the alternative codon is a stop codon
                    if alt_codon in stop_codons:
                        info += ',stop_gained'

                    if v.INFO['CSQ'] is not None:

                        # fetch the codons from the VEP annotation
                        codons_vep = self.parse_codons_vep(v)

                        # Make sure the codons determined by VEP are the same as our codons.
                        # We can only do the comparison for variant sites.
                        if not np.array_equal(codons_vep, [codon, alt_codon]):
                            logger.warning(f'VEP codons do not match with codons determined by '
                                           f'codon table for: {v}.')

                            self.vep_mismatches.append(v)
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
            but have to start with ``https://``
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
            annotation.provide_context(self)
            annotation.add_info(reader)

        # create the writer
        writer = Writer(self.output, reader)

        # iterate over the sites
        for i, variant in enumerate(self.get_sites(reader)):

            # stop if max_sites was reached
            if i >= self.max_sites:
                break

            # apply annotations
            for annotation in self.annotations:
                annotation.annotate_site(variant)

            # write the variant
            writer.write_record(variant)

        # finalize annotations
        for annotation in self.annotations:
            annotation.finalize()
