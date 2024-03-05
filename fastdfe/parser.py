"""
A VCF parser that can be used to extract the site frequency spectrum (SFS) from a VCF file.
Stratifying the SFS is supported by providing a list of :class:`Stratification` instances.
"""

__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"
__date__ = "2023-03-26"

import functools
import itertools
import logging
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from scipy.stats import hypergeom
from typing import List, Callable, Literal, Optional, Dict, Tuple

import numpy as np
from Bio.SeqRecord import SeqRecord
from cyvcf2 import Variant
from tqdm import tqdm

from .annotation import Annotation, SynonymyAnnotation, DegeneracyAnnotation, AncestralAlleleAnnotation
from .filtration import Filtration, PolyAllelicFiltration, SNPFiltration
from .io_handlers import bases, get_called_bases, FASTAHandler, NoTypeException, \
    DummyVariant, MultiHandler, VCFHandler, is_monomorphic_snp
from .settings import Settings
from .spectrum import Spectra

# logger
logger = logging.getLogger('fastdfe')


def _count_valid_type(func: Callable) -> Callable:
    """
    Decorator for counting the number of sites that had a valid type.
    """

    @functools.wraps(func)
    def wrapper(self, variant: Variant | DummyVariant):
        """
        Wrapper function.

        :param self: Class instance
        :param variant: The vcf site
        :return: The result of the decorated function
        """
        res = func(self, variant)
        self.n_valid += 1
        return res

    return wrapper


class Stratification(ABC):
    """
    Abstract class for Stratifying the SFS by determining a site's type based on its properties.
    """

    def __init__(self):
        """
        Create instance.
        """
        self._logger = logger.getChild(self.__class__.__name__)

        #: Parser instance
        self.parser: Optional['Parser'] = None

        #: The number of sites that didn't have a type.
        self.n_valid: int = 0

    def _setup(self, parser: 'Parser'):
        """
        Provide the stratification with some context by specifying the parser.
        This should be done before calling :meth:`get_type`.

        :param parser: The parser
        """
        self.parser = parser

    def _rewind(self):
        """
        Rewind the stratification.
        """
        self.n_valid = 0

    def _teardown(self):
        """
        Perform any necessary post-processing.
        """
        self._logger.info(f"Number of sites with valid type: {self.n_valid}")

    @abstractmethod
    def get_type(self, variant: Variant | DummyVariant) -> Optional[str]:
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


class SNPStratification(Stratification, ABC):
    """
    Abstract class for stratifications that can only handle SNPs. We need to issue a warning in this case.
    """

    def _setup(self, parser: 'Parser'):
        """
        Set up the stratification.

        :param parser: The parser
        """
        super()._setup(parser)

        # issue warning if we have an SNP stratification
        self._logger.warning(f"{self.__class__.__name__} can only handle SNPs and not mono-allelic sites. "
                             "This means you have to update the number of mono-allelic sites manually.")


class BaseContextStratification(Stratification, FASTAHandler):
    """
    Stratify the SFS by the base context of the mutation. The number of flanking bases
    can be configured. Note that we attempt to take the ancestral allele as the
    middle base. If ``skip_non_polarized`` is set to ``False``, we use the reference
    allele as the middle base.
    """

    def __init__(
            self,
            fasta: str,
            n_flanking: int = 1,
            aliases: Dict[str, List[str]] = {},
            cache: bool = True
    ):
        """
        Create instance. Note that we require a fasta file to be specified
        for base context to be able to be inferred

        :param fasta: The fasta file path, possibly gzipped or a URL
        :param n_flanking: The number of flanking bases
        :param aliases: Dictionary of aliases for the contigs in the VCF file, e.g. ``{'chr1': ['1']}``.
            This is used to match the contig names in the VCF file with the contig names in the FASTA file and GFF file.
        :param cache: Whether to cache files that are downloaded from URLs
        """
        Stratification.__init__(self)

        FASTAHandler.__init__(self, fasta, cache=cache, aliases=aliases)

        #: The number of flanking bases
        self.n_flanking: int = n_flanking

        #: The current contig
        self.contig: Optional[SeqRecord] = None

    def _rewind(self):
        """
        Rewind the stratification.
        """
        Stratification._rewind(self)
        FASTAHandler._rewind(self)

        self.contig = None

    @_count_valid_type
    def get_type(self, variant: Variant | DummyVariant) -> str:
        """
        Get the base context for a given mutation

        :param variant: The vcf site
        :return: Base context of the mutation
        """
        pos = variant.POS - 1

        # get the ancestral allele
        aa = self.parser._get_ancestral(variant)

        # get aliases
        aliases = self.get_aliases(variant.CHROM)

        # check if contig is up-to-date
        if self.contig is None or self.contig.id not in aliases:
            self._logger.debug(f"Fetching contig '{variant.CHROM}'.")

            # fetch contig
            self.contig = self.get_contig(aliases)

        # check if position is valid
        if pos < 0 or pos >= len(self.contig):
            raise NoTypeException("Invalid position: Position must be within the bounds of the sequence.")

        # get upstream bases
        upstream_start = max(0, pos - self.n_flanking)
        upstream_bases = str(self.contig.seq[upstream_start:pos])

        # get downstream bases
        downstream_end = min(len(self.contig), pos + self.n_flanking + 1)
        downstream_bases = str(self.contig.seq[pos + 1:downstream_end])

        return f"{upstream_bases}{aa}{downstream_bases}"

    def get_types(self) -> List[str]:
        """
        Create all possible base contexts.

        :return: List of contexts
        """
        return [''.join(c) for c in itertools.product(bases, repeat=2 * self.n_flanking + 1)]


class BaseTransitionStratification(SNPStratification):
    """
    Stratify the SFS by the base transition of the mutation, i.e., ``A>T``.

    .. warning::
        This stratification only works for SNPs. You thus need to update the number of mono-allelic sites manually.
    """

    @_count_valid_type
    def get_type(self, variant: Variant | DummyVariant) -> str:
        """
        Get the base transition for the given variant.

        :param variant: The vcf site
        :return: Base transition
        :raises NoTypeException: if not type could be determined
        """
        if variant.is_snp:
            ancestral = self.parser._get_ancestral(variant)

            derived = variant.REF if variant.REF != ancestral else variant.ALT[0]

            if ancestral in bases and derived in bases and ancestral != derived:
                return f"{ancestral}>{derived}"

            raise NoTypeException("Not a valid base transition.")

        raise NoTypeException("Site is not a SNP.")

    def get_types(self) -> List[str]:
        """
        Get all possible base transitions.

        :return: List of contexts
        """
        return ['>'.join([a, b]) for a in bases for b in bases if a != b]


class TransitionTransversionStratification(BaseTransitionStratification):
    """
    Stratify the SFS by whether we have a transition or transversion.

    .. warning::
        This stratification only works for SNPs. You thus need to update the number of mono-allelic sites manually.
    """

    @_count_valid_type
    def get_type(self, variant: Variant | DummyVariant) -> str:
        """
        Get the mutation type (transition or transversion) for a given mutation.

        :param variant: The vcf site
        :return: Mutation type
        """
        if variant.is_snp:

            if variant.ALT[0] not in bases:
                raise NoTypeException("Invalid alternate allele: Alternate allele must be a valid base.")

            if (variant.REF, variant.ALT[0]) in [('A', 'G'), ('G', 'A'), ('C', 'T'), ('T', 'C')]:
                return "transition"
            else:
                return "transversion"

        raise NoTypeException("Site is not a SNP.")

    def get_types(self) -> List[str]:
        """
        All possible mutation types (transition and transversion).

        :return: List of mutation types
        """
        return ["transition", "transversion"]


class AncestralBaseStratification(Stratification):
    """
    Stratify the SFS by the base context of the mutation: the reference base.
    If ``skip_non_polarized`` is set to ``False``, we use the reference allele as
    ancestral base. By default, we use the ``AA`` tag to determine the ancestral allele.

    Any subclass of :class:`~fastdfe.parser.AncestralAnnotation` can be used to annotate the ancestral allele.
    """

    @_count_valid_type
    def get_type(self, variant: Variant | DummyVariant) -> str:
        """
        Get the type which is the reference allele.

        :param variant: The vcf site
        :return: reference allele
        """
        return self.parser._get_ancestral(variant)

    def get_types(self) -> List[str]:
        """
        The possible base types.

        :return: List of contexts
        """
        return bases


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
        super().__init__()

        #: Custom callback to determine the degeneracy of mutation
        self.get_degeneracy = custom_callback if custom_callback is not None else self._get_degeneracy_default

    @staticmethod
    def _get_degeneracy_default(variant: Variant | DummyVariant) -> Optional[Literal['neutral', 'selected']]:
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

    @_count_valid_type
    def get_type(self, variant: Variant | DummyVariant) -> Literal['neutral', 'selected']:
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


class SynonymyStratification(SNPStratification):
    """
    Stratify SFS by synonymy (neutral or selected).

    :class:`~fastdfe.annotation.SynonymyAnnotation` can be used to annotate the synonymy of a site.

    .. warning::
        This stratification only works for SNPs. You thus need to update the number of mono-allelic sites manually.
    """

    def get_types(self) -> List[str]:
        """
        Get all possible synonymy types (``neutral`` and ``selected``).

        :return: List of contexts
        """
        return ['neutral', 'selected']

    @_count_valid_type
    def get_type(self, variant: Variant | DummyVariant) -> Literal['neutral', 'selected']:
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

    .. warning::
        This stratification only works for SNPs. You thus need to update the number of mono-allelic sites manually.
    """

    #: The tag used by VEP to annotate the synonymy
    info_tag = 'CSQ'

    def get_types(self) -> List[str]:
        """
        Get all possible synonymy types (``neutral`` and ``selected``).

        :return: List of contexts
        """
        return ['neutral', 'selected']

    @_count_valid_type
    def get_type(self, variant: Variant | DummyVariant) -> Literal['neutral', 'selected']:
        """
        Get the synonymy of a site.

        :param variant: The vcf site
        :return: Type of the mutation, either ``neutral`` or ``selected``
        """
        synonymy = variant.INFO.get(self.info_tag, '')

        if 'synonymous_variant' in synonymy:
            return 'neutral'

        if 'missense_variant' in synonymy:
            return 'selected'

        raise NoTypeException(f"Synonymy tag has invalid value: '{synonymy}' at {variant.CHROM}:{variant.POS}")


class SnpEffStratification(VEPStratification):
    """
    Stratify SFS by synonymy (neutral or selected) based on annotation provided by SnpEff.

    .. warning::
        This stratification only works for SNPs. You thus need to update the number of mono-allelic sites manually.
    """

    #: The tag used by SnpEff to annotate the synonymy
    info_tag = 'ANN'


class GenomePositionDependentStratification(Stratification, ABC):
    pass


class ContigStratification(GenomePositionDependentStratification):
    """
    Stratify SFS by contig.
    """

    @_count_valid_type
    def get_type(self, variant: Variant | DummyVariant) -> str:
        """
        Get the contig.

        :param variant: The vcf site
        :return: The contig name
        """
        return variant.CHROM

    def get_types(self) -> List[str]:
        """
        Get all possible contig type.

        :return: List of contexts
        """
        return list(self.parser._reader.seqnames)


class ChunkedStratification(GenomePositionDependentStratification):
    """
    Stratify SFS by creating ``n`` contiguous chunks of roughly equal size.

    .. note::
        Since the total number of sites is not known in advance, we cannot create contiguous
        chunks of exactly equal size.
    """

    def __init__(self, n_chunks: int):
        """
        Initialize the stratification.

        :param n_chunks: Number of sites per window
        """
        super().__init__()

        #: Number of chunks
        self.n_chunks: int = int(n_chunks)

        #: List of chunk sizes
        self.chunk_sizes: Optional[List[int]] = None

        #: Number of sites seen so far
        self.counter: int = 0

    def _setup(self, parser: 'Parser'):
        """
        Set up the stratification.

        :param parser: The parser
        """
        super()._setup(parser)

        # compute base chunk size and remainder
        base_chunk_size, remainder = divmod(parser.n_sites, self.n_chunks)

        # create list of chunk sizes
        self.chunk_sizes = [base_chunk_size + (i < remainder) for i in range(self.n_chunks)]

    def get_types(self) -> List[str]:
        """
        Get all possible window types.

        :return: List of contexts
        """
        return [f'chunk{i}' for i in range(self.n_chunks)]

    @_count_valid_type
    def get_type(self, variant: Variant | DummyVariant) -> str:
        """
        Get the type.

        :param variant: The vcf site
        :return: The type
        """
        # find the index of the chunk to which the current site belongs
        chunk_index = next(i for i, size in enumerate(self.chunk_sizes) if self.counter < sum(self.chunk_sizes[:i + 1]))

        # get the type
        t = f'chunk{chunk_index}'

        # update the counter
        self.counter += 1

        return t


class TargetSiteCounter:
    """
    Class for counting the number of target sites when parsing a VCF file that does not contain monomorphic sites.
    This class is used in conjunction with :class:`~fastdfe.parser.Parser` and samples sites from the given fasta
    file that are found in between variants on the same contig that were parsed in the VCF.
    Ideally, we obtain the SFS by parsing VCF files that contain both mono- and polymorphic sites. This is because
    we need to know about the number of mutational opportunities for synonymous and non-synonymous sites which
    contain plenty of information on the strength of selection. It is recommended to use a SNPFiltration when
    using this class to avoid biasing the result by monomorphic sites present in the VCF file.

    .. warning::
        This class is not compatible with stratifications based on info tags that are pre-defined in the VCF file, as
        opposed to those added dynamically using the ``annotations`` argument of the parser. We also need to
        stratify mono-allelic sites which, in this case, won't be present in the VCF file so that they have no
        info tags when sampling from the FASTA file, and are thus ignored by the stratifications. However, using the
        ``annotations`` argument of the parser, the info tags the stratifications are based on are added on-the-fly,
        also for monomorphic sites sampled from the FASTA file.
    """

    def __init__(
            self,
            n_target_sites: int,
            n_samples: int = int(1e5),
    ):
        """
        Initialize counter.

        :param n_target_sites: The total number of sites (mono- and polymorphic) that would be present in the VCF file
            if it contained monomorphic sites. This number should be considerably larger than the number of polymorphic
            sites in the VCF file. This value is not extremely important for the DFE inference, the ratio of synonymous
            to non-synonymous sites being more informative, but the order of magnitude should be correct, in any case.
        :param n_samples: The number of sites to sample from the fasta file. Many sampled sites will not be valid as
            they are non-coding. To obtain good estimates, a few thousand sites should be sampled per type of site
            (depending on the stratifications used).
        """
        #: The logger
        self._logger = logger.getChild(self.__class__.__name__)

        #: The total number of sites considered when parsing the VCF
        self.n_target_sites: int | None = int(n_target_sites)

        #: Number of samples
        self.n_samples: int = int(n_samples)

        #: The spectra before inferring the number of target sites
        self._sfs_polymorphic: Spectra | None = None

    def _setup(self, parser: 'Parser'):
        """
        Set up the counter.

        :param parser: The parser
        """
        self.parser = parser

        # check if we have a SNPFiltration
        if not any([isinstance(f, SNPFiltration) for f in self.parser.filtrations]):
            self._logger.warning("Recommended to use a SNPFiltration when using target site "
                                 "counter to avoid biasing the result by monomorphic sites.")

        # check if have degeneracy stratification but no degeneracy annotation
        if any([isinstance(s, DegeneracyStratification) for s in self.parser.stratifications]) \
                and not any([isinstance(a, DegeneracyAnnotation) for a in self.parser.annotations]):
            self._logger.warning("When using TargetSiteCounter with DegeneracyStratification, "
                                 "make sure to provide DegeneracyAnnotation to make sure the "
                                 "sites sampled from the FASTA file also have a degeneracy tag.")

    def _teardown(self):
        """
        Perform any necessary post-processing.
        """
        # tear down parser
        self.parser._teardown()

    def _suspend_snp_filtration(self):
        """
        Suspend SNP filtration to make sure we sample can actually sample monomorphic sites.
        """
        # store original filtrations
        self._filtrations = self.parser.filtrations

        # remove SNPFiltration
        self.parser.filtrations = [f for f in self.parser.filtrations if not isinstance(f, SNPFiltration)]

    def _resume_snp_filtration(self):
        """
        Resume SNP filtration.
        """
        # restore original filtrations
        self.parser.filtrations = self._filtrations

    def count(self):
        """
        Count the number of target sites.

        :return: The number of target sites
        """
        # rewind parser components
        self.parser._rewind()

        # suspend SNP filtration
        self._suspend_snp_filtration()

        # rewind fasta iterator
        FASTAHandler._rewind(self.parser)

        # initialize random number generator
        rng = np.random.default_rng(self.parser.seed)

        # initialize progress bar
        pbar = tqdm(
            total=self.n_samples,
            desc=f'{self.__class__.__name__}>Sampling target sites',
            disable=Settings.disable_pbar
        )

        # get array of ranges per contig of parsed variants
        ranges = np.array(list(self.parser._contig_bounds.values()))

        # get range sizes
        range_sizes = ranges[:, 1] - ranges[:, 0]

        # determine sampling probabilities
        probs = range_sizes / np.sum(range_sizes)

        # sample number of sites per contig
        samples = rng.multinomial(self.n_samples, probs)

        # keep track of SFS before update
        self._sfs_polymorphic = Spectra(self.parser.sfs)

        # initialize counter
        i = 0

        # iterate over contigs
        for contig, bounds, n in zip(self.parser._contig_bounds.keys(), ranges, samples):

            # get aliases
            aliases = self.parser.get_aliases(contig)

            # make sure we have a valid range
            if bounds[1] > bounds[0] and n > 0:

                self._logger.debug(f"Sampling {n} sites from contig '{contig}'.")

                # fetch contig
                record = self.parser.get_contig(aliases, notify=False)

                # get positions
                # we sort in ascending order as the parser expects the positions to be sorted
                positions = np.sort(rng.integers(*bounds, size=n))

                # sample sites
                for pos in positions:

                    # create dummy variant
                    variant = DummyVariant(
                        ref=record.seq[pos - 1],  # fasta is 0-based
                        pos=pos,  # VCF is 1-based
                        chrom=contig
                    )

                    # check if site was included in the SFS
                    if self.parser._process_site(variant):
                        i += 1

                    # update progress bar
                    pbar.update()

        # close progress bar
        pbar.close()

        # resume SNP filtration
        self._resume_snp_filtration()

        # tear down
        self._teardown()

        # notify on number of sites included in the SFS
        self._logger.info(f"{i} out of {self.n_samples} sampled sites were valid.")

    def _update_target_sites(self, spectra: Spectra) -> Spectra:
        """
        Update the target sites of the spectra.

        :param spectra: The spectra, including the sampled monomorphic sites.
        :return: The updated spectra.
        """
        # copy spectra
        spectra = spectra.copy()

        # cast to float to avoid implicit type conversion later on
        spectra.data = spectra.data.astype(float)

        # subtract by monomorphic counts of original spectra
        # we only want to consider the monomorphic sites sampled from the FASTA file
        spectra.data.iloc[[0, -1], :] -= self._sfs_polymorphic.data.iloc[[0, -1], :]

        # get number of monomorphic and polymorphic sites sampled from the FASTA and VCF file, respectively
        n_monomorphic = spectra.data.iloc[0, :].sum()
        n_polymorphic = spectra.data.iloc[1:, :].sum().sum()

        # check if we have enough target sites
        if self.n_target_sites < n_polymorphic:
            self._logger.warning(f"Number of polymorphic sites ({n_polymorphic}) exceeds the "
                                 f"number of target sites ({self.n_target_sites}) which does not make sense. "
                                 f"We leave the number of target sites unchanged. "
                                 f"Please remember to modify the number of target sites accordingly "
                                 f"if your VCF file contains only contains polymorphic sites.")
        elif n_monomorphic == 0:
            self._logger.warning(f"Number of monomorphic sites is zero which should only happen "
                                 f"if there are very few sites considered. Failed to update "
                                 f"the number of target sites.")
        else:

            # compute multiplicative factor to scale the total number of sites
            # to the number of target sites plus the number of polymorphic sites
            x = (self.n_target_sites + self._sfs_polymorphic.n_polymorphic.sum() - n_polymorphic) / n_monomorphic

            # extrapolate monomorphic counts using scaling factor
            spectra.data.iloc[0, :] *= x

            # subtract polymorphic counts from original spectra,
            # so that the total number of sites is equal to the number of target sites
            # we do this to correct for the fact that, for a type, we have relatively
            # fewer monomorphic sites if we have more polymorphic sites
            # TODO include monomorphic sites here from VCF?
            spectra.data.iloc[0, :] -= self._sfs_polymorphic.n_polymorphic

        return spectra


class Parser(MultiHandler):
    """
    Parse site-frequency spectra from VCF files.

    By default, the parser looks at the ``AA`` tag in the VCF file's info field to retrieve
    the correct polarization. Polymorphic sites for which this tag is not well-defined are by default
    ignored (see ``skip_non_polarized``).

    This class also offers on-the-fly annotation of the VCF sites such as site degeneracy and
    ancestral allele state. This is done by providing a list of annotations to the parser which are
    applied in the order they are provided.

    The parser also allows to filter sites based on site properties which is done by
    passing a list of filtrations. By default, we filter out poly-allelic sites as sites are assumed to be
    at most bi-allelic.

    In addition, the parser allows to stratify the SFS by providing a list of stratifications. This is useful
    to obtain the SFS for different types of sites for which we can jointly infer the DFEs using
    :class:`~fastdfe.joint_inference.JointInference`.

    To correctly determine the number of target sites when parsing a VCF file that does not contain monomorphic sites,
    we can use a :class:`~fastdfe.parser.TargetSiteCounter`. This class is used in conjunction with the parser and
    samples sites from the given FASTA file that are found in between variants on the same contig that were parsed
    in the VCF.

    Note that we assume the sites in the VCF file to be sorted by position in ascending order (per contig).

    Example usage:

    ::

        import fastdfe as fd

        # Parse selected and neutral SFS from human chromosome 1.
        p = fd.Parser(
            vcf="https://ngs.sanger.ac.uk/production/hgdp/hgdp_wgs.20190516/"
                "hgdp_wgs.20190516.full.chr21.vcf.gz",
            fasta="http://ftp.ensembl.org/pub/release-109/fasta/homo_sapiens/"
                  "dna/Homo_sapiens.GRCh38.dna.chromosome.21.fa.gz",
            gff="http://ftp.ensembl.org/pub/release-109/gff3/homo_sapiens/"
                "Homo_sapiens.GRCh38.109.chromosome.21.gff3.gz",
            aliases=dict(chr21=['21']),  # mapping for contig names
            n=10,  # SFS sample size
            # we use a target site counter to infer the number of target sites.
            target_site_counter=fd.TargetSiteCounter(
                n_samples=1000000,
                # determine number of target sites by looking at total length of coding sequences
                n_target_sites=fd.Annotation.count_target_sites(
                    "http://ftp.ensembl.org/pub/release-109/gff3/homo_sapiens/"
                    "Homo_sapiens.GRCh38.109.chromosome.21.gff3.gz"
                )['21']
            ),
            # add degeneracy annotation for sites
            annotations=[
                fd.DegeneracyAnnotation()
            ],
            filtrations=[
                # exclude non-SNPs as we infer monomorphic sites with target site counter
                fd.SNPFiltration(),
                # filter out sites not in coding sequences
                fd.CodingSequenceFiltration()
            ],
            # stratify by 4-fold/0-fold degeneracy
            stratifications=[fd.DegeneracyStratification()],
            info_ancestral='AA_ensembl'
        )

        sfs = p.parse()

        sfs.plot()

    """

    def __init__(
            self,
            vcf: str,
            n: int,
            gff: str | None = None,
            fasta: str | None = None,
            info_ancestral: str = 'AA',
            skip_non_polarized: bool = True,
            stratifications: List[Stratification] = [],
            annotations: List[Annotation] = [],
            filtrations: List[Filtration] = [PolyAllelicFiltration()],
            include_samples: List[str] = None,
            exclude_samples: List[str] = None,
            max_sites: int = np.inf,
            seed: int | None = 0,
            cache: bool = True,
            aliases: Dict[str, List[str]] = {},
            target_site_counter: TargetSiteCounter = None,
            subsample_mode: Literal['random', 'probabilistic'] = 'probabilistic',
    ):
        """
        Initialize the parser.

        :param vcf: The path to the VCF file, can be gzipped or a URL.
        :param gff: The path to the GFF file, possibly gzipped or a URL. This file is optional and depends on
            the stratifications, annotations and filtrations that are used.
        :param fasta: The path to the FASTA file, possibly gzipped or a URL. This file is optional and depends on
            the annotations and filtrations that are used.
        :param n: The size of the resulting SFS. We down-sample to this number by drawing without replacement from
            the set of all available genotypes per site. Sites with fewer than ``n`` genotypes are skipped.
        :param info_ancestral: The tag in the INFO field that contains ancestral allele information. Consider using
            an ancestral allele annotation if this information is not available yet.
        :param skip_non_polarized: Whether to skip poly-morphic sites that are not polarized, i.e., without a valid
            info tag providing the ancestral allele. If ``False``, we use the reference allele as ancestral allele (
            only recommended if working with folded spectra).
        :param stratifications: List of stratifications to use.
        :param annotations: List of annotations to use.
        :param filtrations: List of filtrations to use.
        :param include_samples: List of sample names to consider when determining the SFS. If ``None``, all samples
            are used. Note that this restriction does not apply to the annotations and filtrations.
        :param exclude_samples: List of sample names to exclude when determining the SFS. If ``None``, no samples
            are excluded. Note that this restriction does not apply to the annotations and filtrations.
        :param max_sites: Maximum number of sites to parse from the VCF file.
        :param seed: Seed for the random number generator. Use ``None`` for no seed.
        :param cache: Whether to cache files downloaded from URLs.
        :param aliases: Dictionary of aliases for the contigs in the VCF file, e.g. ``{'chr1': ['1']}``.
            This is used to match the contig names in the VCF file with the contig names in the FASTA file and GFF file.
        :param target_site_counter: The target site counter. If ``None``, we do not sample target sites.
        :param subsample_mode: The subsampling mode. For ``random``, we draw once without replacement from the set of
            all available genotypes per site. For ``probabilistic``, we add up the hypergeometric distribution for all
            sites. This will produce a smoother SFS, especially when a small number of sites is considered.
        """
        MultiHandler.__init__(
            self,
            vcf=vcf,
            gff=gff,
            fasta=fasta,
            info_ancestral=info_ancestral,
            max_sites=max_sites,
            seed=seed,
            cache=cache,
            aliases=aliases
        )

        # warn if SynonymyAnnotation is used
        if any(isinstance(a, SynonymyAnnotation) for a in annotations):
            logger.warning("SynonymyAnnotation is not recommended to be used with the parser as "
                           "it is not possible to determine the synonymy of monomorphic sites. "
                           "Consider using DegeneracyAnnotation instead.")

        #: The target site counter
        self.target_site_counter: TargetSiteCounter | None = target_site_counter

        #: The number of individuals in the sample
        self.n: int = int(n)

        #: The list of samples to include
        self.include_samples: List[str] | None = include_samples

        #: The list of samples to exclude
        self.exclude_samples: List[str] | None = exclude_samples

        #: The mask of samples to use
        self._samples_mask: np.ndarray | None = None

        #: Whether to skip sites that are not polarized, i.e., without a valid info tag providing the ancestral allele
        self.skip_non_polarized: bool = skip_non_polarized

        #: List of stratifications to use
        self.stratifications: List[Stratification] = stratifications

        #: List of annotations to use
        self.annotations: List[Annotation] = annotations

        #: List of filtrations to use
        self.filtrations: List[Filtration] = filtrations

        #: The number of sites that were skipped for various reasons
        self.n_skipped: int = 0

        #: The number of sites that were skipped because they had no valid ancestral allele
        self.n_no_ancestral: int = 0

        #: Dictionary of SFS indexed by joint type
        self.sfs: Dict[str, np.ndarray] = defaultdict(lambda: np.zeros(self.n + 1))

        #: 1-based positions of lowest and highest site position per contig (only when target_site_counter is used)
        # noinspection PyTypeChecker
        self._contig_bounds: Dict[str, Tuple[int, int]] = defaultdict(lambda: (np.inf, -np.inf))

        if subsample_mode not in ['random', 'probabilistic']:
            raise ValueError(f"Subsampling mode '{subsample_mode}' is not valid. "
                             f"Valid modes are 'random' and 'probabilistic'.")

        #: The subsampling mode
        self.subsample_mode: Literal['random', 'probabilistic'] = subsample_mode

    def _get_ancestral(self, variant: Variant | DummyVariant) -> str:
        """
        Determine the ancestral allele.

        :param variant: The vcf site
        :return: Ancestral allele
        :raises NoTypeException: If the site is not polarized and ``skip_non_polarized`` is ``True`` or if
            the ancestral allele or reference allele (in case of monomorphic sites) is not a valid base.
        """
        if variant.is_snp:
            # obtain ancestral allele
            aa = variant.INFO.get(self.info_ancestral)

            # return the ancestral allele if it is a valid base
            if aa in bases:
                return aa

            # if we skip non-polarized sites, we raise an error here
            if self.skip_non_polarized:
                raise NoTypeException("No valid AA tag found")

        # if we don't skip non-polarized sites, or if the site is not an SNP
        # we return the reference allele if valid
        if variant.REF in bases:
            return variant.REF

        # if the reference allele is not a valid base, we raise an error
        raise NoTypeException("Reference allele is not a valid base")

    def _create_sfs_dictionary(self) -> Dict[str, np.ndarray]:
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

    def _parse_site(self, variant: Variant | DummyVariant) -> bool:
        """
        Parse a single site.

        :param variant: The variant.
        :return: Whether the site was included in the SFS.
        """
        if variant.is_snp:

            # obtain called bases
            genotypes = get_called_bases(variant.gt_bases[self._samples_mask])

            # number of samples
            n_samples = len(genotypes)

            # skip if not enough samples
            if n_samples < self.n:
                self._logger.debug(f'Skipping site due to too few samples at {variant.CHROM}:{variant.POS}.')
                return False

            try:
                # determine ancestral allele
                aa = self._get_ancestral(variant)
            except NoTypeException:
                self.n_no_ancestral += 1
                return False

            # count called bases
            counter = Counter(genotypes)

            # determine ancestral allele count
            n_aa = counter[aa]

            # Determine down-projected allele count.
            # Here we implicitly assume that the site is bi-allelic.
            if self.subsample_mode == 'random':
                m = np.zeros(self.n + 1)
                k = hypergeom.rvs(M=n_samples, n=n_samples - n_aa, N=self.n, random_state=self.rng)
                m[k] = 1
            else:
                m = hypergeom.pmf(k=range(self.n + 1), M=n_samples, n=n_samples - n_aa, N=self.n)

        # if we have a mono-allelic SNPs
        elif is_monomorphic_snp(variant):
            # if we don't have an SNP, we assume the reference allele to be the ancestral allele,
            # so the derived allele count is 0
            # The polarization of monomorphic sites is not important for DFE inference with fastDFE, in any case
            m = np.zeros(self.n + 1)
            m[0] = 1
        else:
            # skip other types of sites
            self._logger.debug(f'Site is not a valid single nucleotide site at {variant.CHROM}:{variant.POS}.')
            return False

        # try to obtain type
        try:
            # create joint type
            t = '.'.join([s.get_type(variant) for s in self.stratifications]) or 'all'

            # add mass
            self.sfs[t] += m

        except NoTypeException as e:
            self._logger.debug(e)
            return False

        return True

    def _process_site(self, variant: Variant | DummyVariant) -> bool:
        """
        Handle a single variant.

        :param variant: The variant
        :return: Whether the site was included in the SFS.
        """
        # filter the variant
        for filtration in self.filtrations:
            if not filtration.filter_site(variant):
                return False

        # apply annotations
        for annotation in self.annotations:
            annotation.annotate_site(variant)

        # parse site
        return self._parse_site(variant)

    def _rewind(self):
        """
        Rewind the filtrations, annotations and stratifications, and fasta handler.
        """
        FASTAHandler._rewind(self)

        for f in self.filtrations:
            f._rewind()

        for a in self.annotations:
            a._rewind()

        for s in self.stratifications:
            s._rewind()

    def _setup(self):
        """
        Set up the parser.
        """
        # set up target site counter
        if self.target_site_counter is not None:
            self.target_site_counter._setup(self)

        # make parser available to stratifications
        for s in self.stratifications:
            s._setup(self)

        # create a string representation of the stratifications
        representation = '.'.join(['[' + ', '.join(s.get_types()) + ']' for s in self.stratifications]) or "[all]"

        # log the stratifications
        self._logger.info(f'Using stratification: {representation}.')

        # prepare samples mask
        self._prepare_samples_mask()

        # setup annotations
        for annotation in self.annotations:
            annotation._setup(self)

        # setup filtrations
        for f in self.filtrations:
            f._setup(self)

    def _prepare_samples_mask(self):
        """
        Prepare the samples mask.
        """
        # determine samples to include
        if self.include_samples is None:
            mask = np.ones(len(self._reader.samples)).astype(bool)
        else:
            mask = np.isin(self._reader.samples, self.include_samples)

        # determine samples to exclude
        if self.exclude_samples is not None:
            mask &= ~np.isin(self._reader.samples, self.exclude_samples)

        self._samples_mask = mask

    def _teardown(self):
        """
        Tear down parser components.
        """
        # tear down all objects
        for f in self.filtrations:
            f._teardown()

        for s in self.stratifications:
            s._teardown()

        for a in self.annotations:
            a._teardown()

    def parse(self) -> Spectra:
        """
        Parse the VCF file.

        :return: The spectra for the different stratifications
        """
        # set up parser
        self._setup()

        pbar = self.get_pbar(
            total=self.n_sites,
            desc=f"{self.__class__.__name__}>Processing sites"
        )

        # iterate over variants
        for i, variant in enumerate(self._reader):

            # handle site
            if self._process_site(variant):

                if self.target_site_counter is not None:
                    # update bounds
                    low, high = self._contig_bounds[variant.CHROM]
                    self._contig_bounds[variant.CHROM] = (min(low, variant.POS), max(high, variant.POS))
            else:
                self.n_skipped += 1

            # update progress bar
            pbar.update()

            # explicitly stopping after ``n`` sites fixes a bug with cyvcf2:
            # 'error parsing variant with `htslib::bcf_read` error-code: 0 and ret: -2'
            if i + 1 == self.n_sites or i + 1 == self.max_sites:
                break

        # close progress bar
        pbar.close()

        # tear down components
        self._teardown()

        # inform about number of sites without ancestral tag
        if self.n_no_ancestral > 0:
            self._logger.info(f'Skipped {self.n_no_ancestral} sites without ancestral allele information.')

        if len(self.sfs) == 0:
            self._logger.warning(f"No sites were included in the spectra. If this is not expected, "
                                 "please check that all components work as expected. You can also "
                                 "set the log level to DEBUG.")

            # warn that sites might not be polarized
            if self.skip_non_polarized and not any(isinstance(a, AncestralAlleleAnnotation) for a in self.annotations):
                self._logger.warning("Your variants might not be polarized and are thus not included in the spectra. "
                                     "If this is the case, consider using an ancestral allele annotation or setting "
                                     "'skip_non_polarized' to False.")
        else:
            n_included = self.n_sites - self.n_skipped

            self._logger.info(f'Included {n_included} out of {self.n_sites} sites in total from the VCF file.')

        # close VCF reader
        VCFHandler._rewind(self)

        # count target sites
        if self.target_site_counter is not None and self.n_skipped < self.n_sites:
            # count target sites
            self.target_site_counter.count()

            # update target sites
            self.sfs = self.target_site_counter._update_target_sites(Spectra(dict(self.sfs))).to_dict()

        return Spectra(dict(self.sfs)).sort_types()
