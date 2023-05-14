"""
Parser module.
"""

__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"
__date__ = "2023-03-26"

import itertools
import logging
from abc import ABC, abstractmethod
from typing import List, Callable, Literal, Optional, Iterable, Dict

import numpy as np
from cyvcf2 import Variant, VCF
from pyfaidx import Fasta

from .annotation import Annotation, Annotator
from .filtration import Filtration, NoPolyAllelicFiltration
from .spectrum import Spectra
from .vcf import VCFHandler

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

        :param fasta_file: The fasta file path, possibly gzipped and possibly a URL starting with ``https://``
        :param n_flanking: The number of flanking bases
        """
        self.fasta_file = fasta_file
        self.n_flanking = n_flanking

        # load reference genome
        self.reference = Fasta(VCFHandler.download_if_url(self.fasta_file))

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
    By default, we use the ``AA`` tag to determine the ancestral allele.

    Any subclass of ``AncestralAnnotation`` can be used to annotate the ancestral allele.
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
    Stratify SFS by degeneracy. We only consider sides which 4-fold degenerate (neutral) or
    0-fold degenerate (selected) which facilitates counting.

    ``DegeneracyAnnotation`` can be used to annotate the degeneracy of a site.
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
            raise NoTypeException("No degeneracy tag found.")
        else:
            if degeneracy == 4:
                return 'neutral'

            if degeneracy == 0:
                return 'selected'

            raise NoTypeException(f"Degeneracy tag has invalid value: '{degeneracy}' at {variant.CHROM}:{variant.POS}")

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


class Parser(VCFHandler):
    """
    Parse site-frequency spectra from VCF files.

    By default, the parser looks at the ``AA`` tag in the VCF file's info field to retrieve
    the correct polarization. Sites for which this tag is not well-defined are by default
    included. Note that non-polarized frequency spectra provide little information on the
    distribution of beneficial mutations.

    We can also annotate the SFS with additional information, such as the degeneracy of the
    sites and their ancestral alleles. This is done by providing a list of annotations to
    the parser. The annotations are applied in the order they are provided.

    The parser also allows to filter sites based on their annotations. This is done by
    providing a list of filtrations to the parser. By default, we filter out poly-allelic
    sites which is highly recommended as some stratifications assume sites to be at most bi-allelic.

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
            annotations: List[Annotation] = [],
            filtrations: List[Filtration] = [NoPolyAllelicFiltration()],
            max_sites: int = np.inf,
            seed: int | None = 0
    ):
        """
        Initialize the parser.

        :param vcf: The path to the VCF file or an iterable of variants, can be gzipped, urls are also supported
            but have to start with ``https://``
        :param n: The number of individuals in the sample. We down-sample to this number by drawing without replacement.
            Sites with fewer than ``n`` individuals are skipped.
        :param info_ancestral: The tag in the INFO field that contains the ancestral allele
        :param ignore_not_polarized: Whether to ignore sites that are not polarized, i.e., without a valid info tag
            providing the ancestral allele
        :param stratifications: List of stratifications to use
        :param annotations: List of annotations to use
        :param filtrations: List of filtrations to use.
        :param max_sites: Maximum number of sites to parse
        :param seed: Seed for the random number generator
        """
        super().__init__(
            vcf=vcf,
            max_sites=max_sites,
            seed=seed,
            info_ancestral=info_ancestral
        )

        #: The number of individuals in the sample
        self.n = n

        #: Whether to ignore sites that are not polarized, i.e., without a valid info tag providing the ancestral allele
        self.ignore_not_polarized: bool = ignore_not_polarized

        #: List of stratifications to use
        self.stratifications: List[Stratification] = stratifications

        #: List of annotations to use
        self.annotations: List[Annotation] = annotations

        #: List of filtrations to use
        self.filtrations: List[Filtration] = filtrations

        #: The number of sites that were skipped when parsing
        self.n_skipped: int = 0

        #: Dictionary of SFS indexed by type
        self.sfs: Dict[str, np.ndarray] = {}

    def create_sfs_dictionary(self):
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

        self.sfs = sfs

    def parse_site(self, variant: Variant):
        """
        Parse a single site.

        :param variant: The variant
        """

        # number of samples
        n_samples = variant.ploidy * variant.num_called

        # skip if not enough samples
        if n_samples < self.n:
            logger.debug(f'Skipping site due to too few samples at {variant.CHROM}:{variant.POS}.')
            self.n_skipped += 1
            return

        # determine reference allele count
        # this doesn't work for polyploids
        n_ref = variant.ploidy * variant.num_hom_ref + variant.num_het

        # swap reference and alternative allele if the AA info field
        # is defined and indicates so
        aa = variant.INFO.get(self.info_ancestral)

        if aa not in bases:

            # log a warning
            logger.debug(f'No valid ancestral allele defined at {variant.CHROM}:{variant.POS}.')

            if self.ignore_not_polarized:
                self.n_skipped += 1
                return
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
            if t in self.sfs:
                self.sfs[t][k] += 1

        except NoTypeException as e:
            self.n_skipped += 1
            logger.debug(e)

    def handle_site(self, variant: Variant):
        """
        Handle a single site.

        :param variant: The variant
        """
        # filter the variant
        for filtration in self.filtrations:
            if not filtration.filter_site(variant):
                return

        # apply annotations
        for annotation in self.annotations:
            annotation.annotate_site(variant)

        # parse site
        self.parse_site(variant)

    def parse(self) -> Spectra:
        """
        Parse the VCF file.

        :return: The spectra for the different stratifications
        """
        self.create_sfs_dictionary()

        # create a string representation of the stratifications
        representation = '.'.join(['[' + ','.join(s.get_types()) + ']' for s in self.stratifications])

        # log the stratifications
        logger.info(f'Using stratification: {representation}.')

        # count the number of sites
        self.n_sites = self.count_sites()

        # make parser available to stratifications
        for s in self.stratifications:
            s.provide_context(self)

        # instantiate annotator
        ann = Annotator(
            vcf=self.vcf,
            max_sites=self.max_sites,
            seed=self.seed,
            info_ancestral=self.info_ancestral,
            annotations=[],
            output=''
        )

        # create reader
        reader = VCF(self.vcf_local_path)

        # provide annotator to annotations and add info fields
        for annotation in self.annotations:
            annotation.provide_context(ann)
            annotation.add_info(reader)

        # reset the number of skipped sites
        self.n_skipped = 0

        logger.info(f'Starting to parse.')

        for i, variant in enumerate(self.get_sites(reader)):

            # stop if max_sites was reached
            if i >= self.max_sites:
                break

            # handle site
            self.handle_site(variant)

        logger.info(f'Finished parsing.')


        return Spectra(self.sfs)
