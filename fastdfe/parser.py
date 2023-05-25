"""
Parser module.
"""

__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"
__date__ = "2023-03-26"

import itertools
import logging
from abc import ABC, abstractmethod
from functools import cached_property
from typing import List, Callable, Literal, Optional, Iterable, Dict

import numpy as np
from cyvcf2 import Variant, VCF
from pyfaidx import Fasta

from .annotation import Annotation, Annotator
from .filtration import Filtration, PolyAllelicFiltration
from .spectrum import Spectra
from .vcf import VCFHandler

#: Logger
logger = logging.getLogger('fastdfe')

#: The DNA bases
bases = ["A", "C", "G", "T"]


def count_no_type(func: Callable) -> Callable:
    """
    Decorator that increases ``self.n_no_type`` by 1 if the decorated function raises a ``NoTypeException``.
    """

    def wrapper(self, variant):
        try:
            return func(self, variant)
        except NoTypeException as e:
            self.n_no_type += 1
            raise e

    return wrapper


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

    #: The number of sites that didn't have a type.
    n_no_type: int = 0

    def _setup(self, parser: 'Parser'):
        """
        Provide the stratification with some context by specifying the parser.
        This should be done before calling :meth:`get_type`.

        :param parser: The parser
        """
        self.parser = parser

    def _teardown(self):
        """
        Perform any necessary post-processing. This method is called after the actual stratification.
        """
        n_total = self.parser.n_sites - self.parser.n_skipped + self.n_no_type
        n_valid = n_total - self.n_no_type

        logger.info(f":{type(self).__name__} Number of sites with valid type: {n_valid} / {n_total}")

    def _get_ancestral(self, variant: Variant) -> str:
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

        # if we don't skip non-polarized sites, or if the site is not an SNP
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

    def __init__(self, fasta_file: str, n_flanking: int = 1, aliases: Dict[str, List[str]] = {}):
        """
        Create instance. Note that we require a fasta file to be specified
        for base context to be able to be inferred

        :param fasta_file: The fasta file path, possibly gzipped or a URL
        :param n_flanking: The number of flanking bases
        :param aliases: Dictionary of aliases for the contigs in the VCF file, e.g. ``{'chr1': ['1']}``.
            This is used to match the contig names in the VCF file with the contig names in the FASTA file and GFF file.
        """
        #: The fasta file
        self.fasta_file: str = fasta_file

        #: The number of flanking bases
        self.n_flanking: int = n_flanking

        #: Aliases for the contigs in the VCF file
        self.aliases: Dict[str, List[str]] = aliases

    @cached_property
    def _ref(self) -> Fasta:
        """
        Get the reference reader.

        :return: The reference reader.
        """
        return VCFHandler.load_fasta(self.fasta_file)

    @count_no_type
    def get_type(self, variant: Variant) -> str:
        """
        Get the base context for a given mutation

        :param variant: The vcf site
        :return: Base context of the mutation
        """
        pos = variant.POS - 1

        try:
            ref = self._get_ancestral(variant)
        except ValueError:
            raise NoTypeException("Invalid ancestral allele: Ancestral allele must be a valid base.")

        # retrieve the sequence of the chromosome from the reference genome
        sequence = VCFHandler.get_contig(self._ref, VCFHandler.get_aliases(variant.CHROM, self.aliases))

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

    def _setup(self, parser: 'Parser'):
        """
        Determine base transition probabilities for polymorphic sites.
        We do this to calibrate the number of monomorphic sites.

        :param parser: The parser
        """
        self.parser = parser

        logger.info(f'Determining base transition probabilities.')

        # initialize counts
        counts = {base_transition: 0 for base_transition in self.get_types()}

        with parser.get_pbar() as pbar:

            # count base transitions
            for i, variant in enumerate(VCF(parser.vcf)):
                if variant.is_snp:
                    try:
                        counts[self.get_type(variant)] += 1
                    except NoTypeException:
                        pass

                pbar.update()

        # normalize counts to probabilities
        self.probabilities = {k: v / sum(counts.values()) for k, v in counts.items()}

    @count_no_type
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

    Any subclass of :class:`~fastdfe.parser.AncestralAnnotation` can be used to annotate the ancestral allele.
    """

    @count_no_type
    def get_type(self, variant: Variant) -> str:
        """
        Get the type which is the reference allele.

        :param variant: The vcf site
        :return: reference allele
        """
        try:
            return self._get_ancestral(variant)
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

    def _setup(self, parser: 'Parser'):
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

        with parser.get_pbar() as pbar:

            # count transitions and transversions
            for i, variant in enumerate(VCF(parser.vcf)):
                if variant.is_snp:
                    counts[self.get_type(variant)] += 1

                pbar.update()

        # normalize counts to probabilities
        self.probabilities = {k: v / sum(counts.values()) for k, v in counts.items()}

    @count_no_type
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

    :class:`~fastdfe.annotation.DegeneracyAnnotation` can be used to annotate the degeneracy of a site.
    """

    def __init__(
            self,
            custom_callback: Callable[[Variant], str] = None,
    ):
        """
        Initialize the stratification.

        :param custom_callback: Custom callback to determine the type of mutation
        """
        #: Custom callback to determine the degeneracy of mutation
        self.get_degeneracy = custom_callback if custom_callback is not None else self._get_degeneracy_default

    @staticmethod
    def _get_degeneracy_default(variant: Variant) -> Optional[Literal['neutral', 'selected']]:
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

    @count_no_type
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


class SynonymyStratification(Stratification):
    """
    Stratify SFS by synonymy (neutral or selected). Note that we extrapolate the number of monomorphic sites
    for each type from the relative number of types for polymorphic sites.

    :class:`~fastdfe.annotation.SynonymyAnnotation` can be used to annotate the synonymy of a site.
    """

    def get_types(self) -> List[str]:
        """
        Get all possible synonymy types (``neutral`` and ``selected``).

        :return: List of contexts
        """
        return ['neutral', 'selected']

    @count_no_type
    def get_type(self, variant: Variant) -> Literal['neutral', 'selected']:
        """
        Get the synonymy using the custom synonymy annotation.

        :param variant: The vcf site
        :return: Type of the mutation, either ``neutral`` or ``selected``
        """
        synonymy = variant.INFO.get('Synonymy')

        if synonymy is None:
            raise NoTypeException("No synonymy tag found.")
        else:
            if synonymy == 1:
                return 'neutral'

            if synonymy == 0:
                return 'selected'

            raise NoTypeException(f"Synonymy tag has invalid value: '{synonymy}' at {variant.CHROM}:{variant.POS}")


class VEPStratification(SynonymyStratification):
    """
    Stratify SFS by synonymy (neutral or selected) based on annotation provided by VEP.
    """

    def get_types(self) -> List[str]:
        """
        Get all possible synonymy types (``neutral`` and ``selected``).

        :return: List of contexts
        """
        return ['neutral', 'selected']

    @count_no_type
    def get_type(self, variant: Variant) -> Literal['neutral', 'selected']:
        """
        Get the synonymy of a site.

        :param variant: The vcf site
        :return: Type of the mutation, either ``neutral`` or ``selected``
        """
        synonymy = variant.INFO.get('CSQ')

        if 'synonymous_variant' in synonymy:
            return 'neutral'

        if 'missense_variant' in synonymy:
            return 'selected'

        raise NoTypeException(f"Synonymy tag has invalid value: '{synonymy}' at {variant.CHROM}:{variant.POS}")


class SnpEffStratification(SynonymyStratification):
    """
    Stratify SFS by synonymy (neutral or selected) based on annotation provided by SnpEff.
    """

    def get_types(self) -> List[str]:
        """
        Get all possible synonymy types (``neutral`` and ``selected``).

        :return: List of contexts
        """
        return ['neutral', 'selected']

    @count_no_type
    def get_type(self, variant: Variant) -> Literal['neutral', 'selected']:
        """
        Get the synonymy of a site.

        :param variant: The vcf site
        :return: Type of the mutation, either ``neutral`` or ``selected``
        """
        synonymy = variant.INFO.get('CSQ')

        if 'synonymous_variant' in synonymy:
            return 'neutral'

        if 'missense_variant' in synonymy:
            return 'selected'

        raise NoTypeException(f"Synonymy tag has invalid value: '{synonymy}' at {variant.CHROM}:{variant.POS}")


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
            filtrations: List[Filtration] = [PolyAllelicFiltration()],
            max_sites: int = np.inf,
            n_target_sites: int = None,
            seed: int | None = 0
    ):
        """
        Initialize the parser.

        :param vcf: The path to the VCF file or an iterable of variants, can be gzipped, urls are also supported
        :param n: The number of individuals in the sample. We down-sample to this number by drawing without replacement.
            Sites with fewer than ``n`` individuals are skipped.
        :param info_ancestral: The tag in the INFO field that contains the ancestral allele
        :param ignore_not_polarized: Whether to ignore sites that are not polarized, i.e., without a valid info tag
            providing the ancestral allele
        :param stratifications: List of stratifications to use
        :param annotations: List of annotations to use
        :param filtrations: List of filtrations to use.
        :param max_sites: Maximum number of sites to parse
        :param n_target_sites: The number of mutational target sites.
            Allows to adjust the number of monomorphic site count. Ideally, we obtain the SFS by
            parsing VCF files that contain both mono- and polymorphic sites. This is because for DFE inference, we
            require the number of monomorphic sites to calibrate the mutation rate. However, often, only polymorphic
            sites are available. In this case, we can use ``n_target_sites`` to extrapolate the number of monomorphic
            sites by looking at the relative number of polymorphic sites for each type. Note that the total number of
            mono- and polymorphic sites should be specified here. This often corresponds to the number of sites in
            coding regions over the sequence considered.
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

        #: The number of mutational target sites
        self.n_target_sites: int | None = n_target_sites

        #: Whether to ignore sites that are not polarized, i.e., without a valid info tag providing the ancestral allele
        self.ignore_not_polarized: bool = ignore_not_polarized

        #: List of stratifications to use
        self.stratifications: List[Stratification] = stratifications

        #: List of annotations to use
        self.annotations: List[Annotation] = annotations

        #: List of filtrations to use
        self.filtrations: List[Filtration] = filtrations

        #: The number of sites that were skipped for various reasons
        self.n_skipped: int = 0

        #: Dictionary of SFS indexed by type
        self.sfs: Dict[str, np.ndarray] = {}

        #: The VCF reader
        self.reader: Optional[VCF] = None

    def _create_sfs_dictionary(self):
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

    def _parse_site(self, variant: Variant):
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

    def _handle_site(self, variant: Variant):
        """
        Handle a single site.

        :param variant: The variant
        """
        # filter the variant
        for filtration in self.filtrations:
            if not filtration.filter_site(variant):
                self.n_skipped += 1
                return

        # apply annotations
        for annotation in self.annotations:
            annotation.annotate_site(variant)

        # parse site
        self._parse_site(variant)

    def _teardown(self):
        """
        Teardown the parser.
        """
        for f in self.filtrations:
            f._teardown()

        for s in self.stratifications:
            s._teardown()

        for a in self.annotations:
            a._teardown()

    def _infer_monomorphic_counts(self):
        """
        Infer the number of monomorphic sites from the number of polymorphic sites.
        """
        # total number of polymorphic sites across all types
        n_polymorphic = np.sum([np.sum(c[1:-1]) for c in self.sfs.values()])

        for t, counts in self.sfs.items():
            self.sfs[t][0] = np.sum(counts[1:-1]) / n_polymorphic * (self.n_target_sites - n_polymorphic)
            self.sfs[t][-1] = 0

    def parse(self) -> Spectra:
        """
        Parse the VCF file.

        :return: The spectra for the different stratifications
        """
        self._create_sfs_dictionary()

        # create a string representation of the stratifications
        representation = '.'.join(['[' + ', '.join(s.get_types()) + ']' for s in self.stratifications])

        # log the stratifications
        logger.info(f'Using stratification: {representation}.')

        # count the number of sites
        self.n_sites = self.count_sites()

        # make parser available to stratifications
        for s in self.stratifications:
            s._setup(self)

        # touch all filtrations
        for f in self.filtrations:
            f._setup()

        # instantiate annotator to provide context to annotations
        ann = Annotator(
            vcf=self.vcf,
            max_sites=self.max_sites,
            seed=self.seed,
            info_ancestral=self.info_ancestral,
            annotations=[],
            output=''
        )

        # create reader
        reader = VCF(self.download_if_url(self.vcf))

        # provide annotator to annotations and add info fields
        for annotation in self.annotations:
            annotation._setup(ann)
            annotation._add_info(reader)

        logger.info(f'Starting to parse.')

        # create progress bar
        with self.get_pbar() as pbar:

            for i, variant in enumerate(reader):

                # handle site
                self._handle_site(variant)

                pbar.update()

                # explicitly stopping after ``n``sites fixes a bug with cyvcf2:
                # 'error parsing variant with `htslib::bcf_read` error-code: 0 and ret: -2'
                if i + 1 == self.n_sites or i + 1 == self.max_sites:
                    break

        # tear down all objects
        self._teardown()

        # close reader
        reader.close()

        # correct monomorphic counts if number of target sites is defined
        if self.n_target_sites is not None:
            self._infer_monomorphic_counts()

        logger.info(f'Parsed {self.n_sites - self.n_skipped} out of {self.n_sites} sites in total.')

        return Spectra(self.sfs)
