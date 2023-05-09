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
from typing import List, Callable, Literal, Optional, TextIO, Iterable

import numpy as np
from cyvcf2 import VCF, Variant
from numpy.random import Generator
from pyfaidx import Fasta
from tqdm import tqdm

from .spectrum import Spectra

#: Logger
logger = logging.getLogger('fastdfe')

#: The DNA bases
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

    #: Parser instance
    parser: Optional['Parser'] = None

    def provide_context(self, parser: 'Parser'):
        """
        Provide the stratification with some context by specifying the parser.
        This should be done before called ``get_type``.

        :param parser: The parser
        """
        self.parser = parser

    def get_ancestral(self, variant: Variant) -> str:
        """
        Determine the ancestral allele.

        :param variant: The vcf site
        :return: Ancestral allele
        :raises ValueError: If the ancestral allele could not be determined
        """
        if variant.is_snp:
            # obtain ancestral allele
            aa = variant.INFO.get(self.parser.info_ancestral)

            # return the ancestral allele if it is a valid base
            if aa in bases:
                return aa

            # if we don't skip non-polarized sites, we raise an error
            if self.parser.ignore_not_polarized:
                raise ValueError("No valid AA tag found.")

        # if we don't skip non-polarized sites, or if the site is not a SNP
        # we return the reference allele
        return variant.REF

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
    can be configured. Note that we attempt to take the ancestral allele as the
    middle base. If ``ignore_not_polarized`` is set to ``True``, we skip sites
    that are not polarized, otherwise we use the reference allele as the middle base.
    """

    def __init__(self, fasta_file: str, n_flanking: int = 1):
        """
        Create instance. Note that we require a fasta file to be specified
        for base context to be able to be inferred

        :param fasta_file: The fasta file path
        :param n_flanking: The number of flanking bases
        """
        self.fasta_file = fasta_file
        self.n_flanking = n_flanking

        # load reference genome
        self.reference = Fasta(self.fasta_file)

    def get_type(self, variant: Variant) -> str:
        """
        Get the base context for a given mutation

        :param variant: The vcf site
        :return: Base context of the mutation
        """
        pos = variant.POS - 1

        try:
            ref = self.get_ancestral(variant)
        except ValueError:
            raise NoTypeException("Invalid ancestral allele: Ancestral allele must be a valid base.")

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
        Create all possible base contexts.

        :return: List of contexts
        """
        return [''.join(c) for c in itertools.product(bases, repeat=2 * self.n_flanking + 1)]


class BaseTransitionStratification(Stratification):
    """
    Stratify the SFS by the base transition of the mutation, i.e., ``A>T``.
    We parse the VCF file twice, once to determine the base transition probabilities
    used for the monomorphic counts, and once to determine the base transitions.
    Note that we assume sites here to be at most bi-allelic.
    """

    #: Base-transition probabilities
    probabilities = dict()

    def provide_context(self, parser: 'Parser'):
        """
        Determine base transition probabilities for polymorphic sites.
        We do this to calibrate the number of monomorphic sites.

        :param parser: The parser
        """
        self.parser = parser

        logger.info(f'Determining base transition probabilities.')

        # initialize counts
        counts = {base_transition: 0 for base_transition in self.get_types()}

        # count base transitions
        for i, variant in enumerate(parser.get_sites()):
            if variant.is_snp:
                try:
                    counts[self.get_type(variant)] += 1
                except NoTypeException:
                    pass

        # normalize counts to probabilities
        self.probabilities = {k: v / sum(counts.values()) for k, v in counts.items()}

    def get_type(self, variant: Variant) -> str:
        """
        Get the base transition for the given variant.

        :param variant: The vcf site
        :return: Base transition
        :raises NoTypeException: if not type could be determined
        """
        if variant.is_snp:
            ref = variant.REF

            # assume there is at most one alternate allele
            alt = variant.ALT[0]

            # obtain ancestral allele
            aa = variant.INFO.get(self.parser.info_ancestral)

            # swap reference and alternative allele
            # note that here we assume again the site is bi-allelic
            if aa in bases and aa != ref:
                ref, alt = alt, ref

            # report type if aa tag is valid or if we don't skip non-polarized sites
            if aa in bases or not self.parser.ignore_not_polarized:

                if ref == alt:
                    raise NoTypeException("Site marked as polymorphic, but reference "
                                          "and alternative allele are the same.")

                return f"{ref}>{alt}"

            raise NoTypeException("No valid AA tag found.")

        # for mono-allelic sites, we sample from the base-transition probabilities
        return self.parser.rng.choice(list(self.probabilities.keys()), p=list(self.probabilities.values()))

    def get_types(self) -> List[str]:
        """
        Get all possible base transitions.

        :return: List of contexts
        """
        return ['>'.join([a, b]) for a in bases for b in bases if a != b]


class AncestralBaseStratification(Stratification):
    """
    Stratify the SFS by the base context of the mutation: the reference base.
    If ``ignore_not_polarized`` is set to ``True``, we skip sites
    that are not polarized, otherwise we use the reference allele as ancestral base.
    """

    def get_type(self, variant: Variant) -> str:
        """
        Get the type which is the reference allele.

        :param variant: The vcf site
        :return: reference allele
        """
        try:
            return self.get_ancestral(variant)
        except ValueError:
            raise NoTypeException("Invalid ancestral allele: Ancestral allele must be a valid base.")

    def get_types(self) -> List[str]:
        """
        The possible base types.

        :return: List of contexts
        """
        return bases


class TransitionTransversionStratification(BaseTransitionStratification):
    """
    Stratify the SFS by whether we have a transition or transversion.
    We parse the VCF file twice here, once to determine the transition-transversion
    probabilities used for the monomorphic counts, and once to stratify the SFS.
    Note that we assume sites here to be at most bi-allelic.
    """

    #: Transition-transversion probabilities
    probabilities = dict(
        transition=0.5,
        transversion=0.5
    )

    def provide_context(self, parser: 'Parser'):
        """
        Determine transition-transversion probabilities for polymorphic sites.
        We do this to calibrate the number of monomorphic sites.

        :param parser: The parser
        """
        self.parser = parser

        logger.info(f'Determining transition-transversion probabilities.')

        # initialize counts
        counts = dict(
            transition=0,
            transversion=0
        )

        # count transitions and transversions
        for i, variant in enumerate(parser.get_sites()):
            if variant.is_snp:
                counts[self.get_type(variant)] += 1

        # normalize counts to probabilities
        self.probabilities = {k: v / sum(counts.values()) for k, v in counts.items()}

    def get_type(self, variant: Variant) -> str:
        """
        Get the mutation type (transition or transversion) for a given mutation.

        :param variant: The vcf site
        :return: Mutation type
        """
        if variant.is_snp:
            if (variant.REF, variant.ALT[0]) in [('A', 'G'), ('G', 'A'), ('C', 'T'), ('T', 'C')]:
                return "transition"
            else:
                return "transversion"

        # for mono-allelic sites, we sample from the transition-transversion probabilities
        return self.parser.rng.choice(list(self.probabilities.keys()), p=list(self.probabilities.values()))

    def get_types(self) -> List[str]:
        """
        All possible mutation types (transition and transversion).

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
        #: Custom callback to determine the degeneracy of mutation
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
        Get the degeneracy.

        :param variant: The vcf site
        :return: Type of the mutation
        :raises NoTypeException: If the mutation is not synonymous or non-synonymous
        """
        return self.get_degeneracy(variant)

    def get_types(self) -> List[str]:
        """
        Get all possible degeneracy type (``neutral`` and ``selected``).

        :return: List of contexts
        """
        return ['neutral', 'selected']


class Parser:
    """
    Parse site-frequency spectra from VCF files.

    By default, the parser looks at the ``AA`` tag in the VCF file's info field to retrieve
    the correct polarization. Sites for which this tag is not well-defined are by default
    included. Note that non-polarized frequency spectra provide little information on the
    distribution of beneficial mutations.

    :warning: Not tested for polyploids.
    """

    def __init__(
            self,
            vcf: str | Iterable[Variant],
            n: int,
            info_ancestral: str = 'AA',
            ignore_not_polarized: bool = False,
            stratifications: List[Stratification] = [
                DegeneracyStratification()
            ],
            max_sites: int = np.inf,
            seed: int | None = 0
    ):
        """
        Initialize the parser.

        :param vcf: The path to the VCF file or an iterable of variants
        :param n: The number of individuals in the sample. We down-sample to this number by drawing without replacement.
            Sites with fewer than ``n`` individuals are skipped.
        :param info_ancestral: The tag in the INFO field that contains the ancestral allele
        :param ignore_not_polarized: Whether to ignore sites that are not polarized, i.e., without a valid info tag
            providing the ancestral allele
        :param stratifications: List of stratifications to use
        :param max_sites: Maximum number of sites to parse
        :param seed: Seed for the random number generator
        """

        #: The number of individuals in the sample
        self.n = n

        #: The path to the VCF file or an iterable of variants
        self.vcf = vcf

        #: The tag in the INFO field that contains the ancestral allele
        self.info_ancestral: str = info_ancestral

        #: Whether to ignore sites that are not polarized, i.e., without a valid info tag providing the ancestral allele
        self.ignore_not_polarized: bool = ignore_not_polarized

        #: List of stratifications to use
        self.stratifications: List[Stratification] = stratifications

        #: Maximum number of sites to parse
        self.max_sites: int = max_sites

        #: Seed for the random number generator
        self.seed: Optional[int] = seed

        #: Random generator instance
        self.rng = np.random.default_rng(seed=seed)

        #: Number of sites to be parsed
        self.n_sites: Optional[int] = None

        #: The number of sites that were skipped when parsing
        self.n_skipped: int = 0

    @staticmethod
    def open_file(file: str) -> TextIO:
        """
        Open a file, either gzipped or not.

        :param file: File to open
        :return: stream
        """
        if file.endswith('.gz'):
            return gzip.open(file, "rt")

        return open(file, 'r')

    def count_lines_vcf(self) -> int:
        """
        Count the number of sites in the VCF.

        :return: Number of sites
        """
        from . import disable_pbar

        logger.info('Counting number of sites.')

        # if we don't have a file path, we can just count the number of variants
        if not isinstance(self.vcf, str):
            return len(list(self.vcf))

        i = 0
        with self.open_file(self.vcf) as f:

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

    def get_sites(self) -> Iterable[Variant]:
        """
        Return an iterable object over the VCF file's sites.

        :return: iterable
        """
        from . import disable_pbar

        vcf = self.vcf

        if isinstance(vcf, str):
            vcf = VCF(vcf)

        return tqdm(vcf, total=self.n_sites, disable=disable_pbar)

    def parse(self) -> Spectra:
        """
        Parse the VCF file.

        :return: The spectra for the different stratifications
        """
        sfs = self.create_sfs_dictionary()

        # create a string representation of the stratifications
        representation = '.'.join(['[' + ','.join(s.get_types()) + ']' for s in self.stratifications])

        # log the stratifications
        logger.info(f'Using stratification: {representation}.')

        # count the number of sites
        self.n_sites = self.count_lines_vcf()

        # make parser available to stratifications
        for s in self.stratifications:
            s.provide_context(self)

        # reset the number of skipped sites
        self.n_skipped = 0

        logger.info(f'Starting to parse.')

        for i, variant in enumerate(self.get_sites()):

            # stop if max_sites was reached
            if i >= self.max_sites:
                break

            # number of samples
            n_samples = variant.ploidy * variant.num_called

            # skip if not enough samples
            if n_samples < self.n:
                logger.debug(f'Skipping site due to too few samples {str(variant).strip()}.')
                self.n_skipped += 1
                continue

            # determine reference allele count
            # this doesn't work for polyploids
            n_ref = variant.ploidy * variant.num_hom_ref + variant.num_het

            # swap reference and alternative allele if the AA info field
            # is defined and indicates so
            aa = variant.INFO.get(self.info_ancestral)

            if aa not in bases:

                # log a warning
                logger.debug(f'No valid ancestral allele defined for {str(variant).strip()}.')

                if self.ignore_not_polarized:
                    self.n_skipped += 1
                    continue
            else:
                # adjust orientation if the ancestral allele is not the reference
                if aa != variant.REF:

                    # change alternative allele if defined
                    if len(variant.ALT) != 0:
                        # we only consider the first alternative allele
                        variant.ALT[0] = variant.REF

                    # change reference allele
                    variant.REF = aa

                    # change reference count
                    n_ref = n_samples - n_ref

            # determine down-projected allele count
            k = self.rng.hypergeometric(ngood=n_samples - n_ref, nbad=n_ref, nsample=self.n)

            # try to obtain type
            try:
                t = '.'.join([s.get_type(variant) for s in self.stratifications])

                # add count by 1 if context is defined
                if t in sfs:
                    sfs[t][k] += 1

            except NoTypeException as e:
                self.n_skipped += 1
                logger.debug(str(e) + ' ' + str(variant).strip())

        return Spectra(sfs)
