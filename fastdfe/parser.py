"""
Parser module.
"""

__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"
__date__ = "2023-03-26"

import gzip
import itertools
import logging
from abc import ABC, abstractmethod
from typing import List, Callable, Literal, Optional, TextIO

import numpy as np
from cyvcf2 import VCF, Variant
from pyfaidx import Fasta
from tqdm import tqdm

from .spectrum import Spectra

# get logger
logger = logging.getLogger('fastdfe')

#: the DNA bases
bases = ["A", "C", "G", "T"]


class NoTypeException(BaseException):
    """
    Exception thrown when no type can be determined.
    """
    pass


class Stratification(ABC):
    """
    Abstract class for Stratifying the SFS by determining
    a site's type based on its properties.
    """

    @abstractmethod
    def get_type(self, variant: Variant) -> Optional[str]:
        """
        Get type of given Variant. Only the types
        given by :meth:`get_types()` are valid, or ``None`` if
        no type could be determined.

        :param variant: The vcf site
        :return: Type of the variant
        """
        pass

    @abstractmethod
    def get_types(self) -> List[str]:
        """
        Get all possible types.

        :return: List of types
        """
        pass


class BaseContextStratification(Stratification):
    """
    Stratify the SFS by the base context of the mutation. The number of flanking bases
    can be configured.
    """

    def __init__(self, fasta_file: str, n_flanking: int = 1):
        self.fasta_file = fasta_file
        self.n_flanking = n_flanking

        # load reference genome
        self.reference = Fasta(self.fasta_file)

    def get_type(self, variant: Variant) -> str:
        """
        Get the base context for a given mutation with k flanking bases.

        :param variant: The vcf site
        :return: Base context of the mutation with k flanking bases
        """
        pos = variant.POS - 1
        ref = variant.REF

        # retrieve the sequence of the chromosome from the reference genome
        sequence = self.reference[variant.CHROM][:].seq

        if pos < 0 or pos >= len(sequence):
            raise NoTypeException("Invalid position: Position must be within the bounds of the sequence.")

        upstream_start = max(0, pos - self.n_flanking)
        upstream_bases = sequence[upstream_start:pos]

        downstream_end = min(len(sequence), pos + self.n_flanking + 1)
        downstream_bases = sequence[pos + 1:downstream_end]

        return f"{upstream_bases}{ref}{downstream_bases}"

    def get_types(self) -> List[str]:
        """
        Create all possible base context with for self.k flanking bases.
        If self.use_transitions is True, include the transition to the alternate base.

        :return: List of contexts
        """
        return [''.join(c) for c in itertools.product(bases, repeat=2 * self.n_flanking + 1)]


class BaseTransitionStratification(Stratification):
    """
    Stratify the SFS by the base transition of the mutation, i.e. A>T.
    """

    def get_type(self, variant: Variant) -> str:
        """
        Get the base context for a given mutation with k flanking bases.

        :param variant: The vcf site
        :return: Base context of the mutation with k flanking bases
        """
        # assume there is at most one alternate allele
        alt = variant.ALT[0] if len(variant.ALT) != 0 else variant.REF

        return f"{variant.REF}>{alt}"

    def get_types(self) -> List[str]:
        """
        Create all possible base context with for self.k flanking bases.

        :return: List of contexts
        """
        return ['>'.join(c) for c in itertools.product(bases, repeat=2)]


class ReferenceBaseStratification(Stratification):
    """
    Stratify the SFS by the base context of the mutation: the reference base.
    """

    def get_type(self, variant: Variant) -> str:
        """
        Get the base of the reference allele.

        :param variant: The vcf site
        :return: Base context of the mutation with k flanking bases
        """
        return variant.REF

    def get_types(self) -> List[str]:
        """
        Create all possible base context with for self.k flanking bases.

        :return: List of contexts
        """
        return bases


class TransitionTransversionStratification(BaseTransitionStratification):
    """
    Stratify the SFS by whether we have a transition or transversion.
    """

    def get_type(self, variant: Variant) -> str:
        """
        Get the mutation type (transition or transversion) for a given mutation.

        :param variant: The vcf site
        :return: Mutation type ("transition" or "transversion")
        """
        # assume there is at most one alternate allele
        alt = variant.ALT[0] if len(variant.ALT) != 0 else variant.REF

        ref = variant.REF
        if (ref, alt) in [('A', 'G'), ('G', 'A'), ('C', 'T'), ('T', 'C')]:
            return "transition"
        else:
            return "transversion"

    def get_types(self) -> List[str]:
        """
        Create all possible mutation types (transition and transversion).

        :return: List of mutation types
        """
        return ["transition", "transversion"]


class DegeneracyStratification(Stratification):
    """
    Stratify SFS by degeneracy, i.e. whether a site is 4-fold degenerate (neutral) or 0-fold degenerate (selected).
    """

    def __init__(self, custom_callback: Callable[[Variant], str] = None):
        """
        Initialize the stratification.

        :param custom_callback: Custom callback to determine the type of mutation
        """
        self.get_degeneracy = custom_callback if custom_callback is not None else self.get_degeneracy_default

    @staticmethod
    def get_degeneracy_default(variant: Variant) -> Optional[Literal['neutral', 'selected']]:
        """
        Get degeneracy based on 'Degeneracy' tag.

        :param variant: The vcf site
        :return: Type of the mutation
        """
        degeneracy = variant.INFO.get('Degeneracy')

        if degeneracy is None:
            raise NoTypeException("Degeneracy tag not found.")
        else:
            if degeneracy == 4:
                return 'neutral'

            if degeneracy == 0:
                return 'selected'

            raise NoTypeException(f"Degeneracy tag has invalid value: {degeneracy}")

    def get_type(self, variant: Variant) -> Literal['neutral', 'selected']:
        """
        Get the base context for a given mutation with k flanking bases.

        :param variant: The vcf site
        :return: Type of the mutation
        :raises NoTypeException: If the mutation is not synonymous or non-synonymous
        """
        return self.get_degeneracy(variant)

    def get_types(self) -> List[str]:
        """
        Return possible types
        :return: List of contexts
        """
        return ['neutral', 'selected']


class Parser:
    def __init__(
            self,
            vcf_file: str,
            n: int,
            info_ancestral: str = 'AA',
            stratifications: List[Stratification] = [
                DegeneracyStratification()
            ],
            max_sites=np.inf,
            seed: int = 0
    ):

        self.n = n
        self.vcf_file = vcf_file

        self.info_ancestral = info_ancestral

        self.stratifications = stratifications

        self.max_sites = max_sites
        self.seed = seed

        # get a random generator instance
        self.rng = np.random.default_rng(seed=seed)

        self.sfs = self.create_sfs_dictionary()

    @staticmethod
    def open_file(file: str) -> TextIO:
        """
        Open a file, either gzipped or not.
        :param file:
        :return: stream
        """
        if file.endswith('.gz'):
            return gzip.open(file, "rt")

        return open(file, 'r')

    def count_lines_vcf(self):
        """
        Count the number of sites in the VCF file.
        """
        from . import disable_pbar

        i = 0

        logger.info('Counting number of sites.')

        with self.open_file(self.vcf_file) as f:

            with tqdm(disable=disable_pbar) as pbar:
                for line in f:
                    if not line.startswith('#'):
                        i += 1
                        pbar.update()

                    # stop counting if max_sites was reached
                    if i >= self.max_sites:
                        break

        return i

    def create_sfs_dictionary(self) -> dict:
        """
        Create an SFS dictionary initialized with all possible base contexts.

        :return: SFS dictionary
        """
        types = [s.get_types() for s in self.stratifications]
        # define the DNA bases
        contexts = ['.'.join(t) for t in itertools.product(*types)]

        # create dict
        sfs = {}
        for context in contexts:
            sfs[context] = np.zeros(self.n + 1)

        return sfs

    def parse(self) -> Spectra:
        """
        Parse the VCF file and return the SFS.
        :return: the spectra for the different stratifications
        """

        from . import disable_pbar

        sfs = self.create_sfs_dictionary()

        representation = '.'.join(['[' + ','.join(s.get_types()) + ']' for s in self.stratifications])
        logger.info(f'Using stratification: {representation}.')

        n_sites = self.count_lines_vcf()

        # parse VCF file
        vcf = VCF(self.vcf_file)

        logger.info(f'Starting to parse.')

        with tqdm(total=n_sites, disable=disable_pbar) as pbar:
            for i, variant in enumerate(vcf):

                # stop if max_sites was reached
                if i >= self.max_sites:
                    break

                # just update beforehand
                pbar.update()

                n_samples = variant.ploidy * variant.num_called

                if n_samples < self.n:
                    continue

                # determine reference allele count
                # this doesn't work for polyploids
                n_ref = variant.ploidy * variant.num_hom_ref + variant.num_het

                # swap reference and alternative allele if the AA info field
                # is defined and indicates so
                aa = variant.INFO.get(self.info_ancestral)

                if aa is None:
                    logger.warning(f'No ancestral allele defined for {variant.CHROM}:{variant.POS}.')
                else:
                    # adjust orientation if different
                    if aa != variant.REF:
                        variant.REF = aa

                        if len(variant.ALT) != 0:
                            variant.ALT[0] = variant.REF

                        n_ref = n_samples - n_ref

                # determine down-projected allele count
                k = self.rng.hypergeometric(ngood=n_samples - n_ref, nbad=n_ref, nsample=self.n)

                # try to obtain type
                try:
                    t = '.'.join([s.get_type(variant) for s in self.stratifications])
                except NoTypeException:
                    continue

                # add count by 1 if context is defined
                if t in sfs:
                    sfs[t][k] += 1

        return Spectra(sfs)
