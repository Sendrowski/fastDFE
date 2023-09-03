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
from typing import List, Callable, Literal, Optional, Iterable, Dict

import numpy as np
from Bio.SeqRecord import SeqRecord
from cyvcf2 import Variant
from tqdm import tqdm

from .annotation import Annotation, SynonymyAnnotation
from .filtration import Filtration, PolyAllelicFiltration
from .io_handlers import bases, get_called_bases, FASTAHandler, NoTypeException, \
    DummyVariant, MultiHandler
from .spectrum import Spectra

# logger
logger = logging.getLogger('fastdfe')


def _count_valid_type(func: Callable) -> Callable:
    """
    Decorator for counting the number of sites that had a valid type.
    """

    @functools.wraps(func)
    def wrapper(self, variant: Variant):
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
        self.logger = logger.getChild(self.__class__.__name__)

        #: MultiHandler instance
        self.handler: Optional['Parser'] = None

        #: The number of sites that didn't have a type.
        self.n_valid: int = 0

    def _setup(self, handler: MultiHandler):
        """
        Provide the stratification with some context by specifying the handler.
        This should be done before calling :meth:`get_type`.

        :param handler: The handler
        """
        self.handler = handler

    def _rewind(self):
        """
        Rewind the stratification.
        """
        self.n_valid = 0

    def _teardown(self):
        """
        Perform any necessary post-processing.
        """
        self.logger.info(f"Number of sites with valid type: {self.n_valid}")

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


class BaseContextStratification(Stratification, FASTAHandler):
    """
    Stratify the SFS by the base context of the mutation. The number of flanking bases
    can be configured. Note that we attempt to take the ancestral allele as the
    middle base. If ``skip_non_polarized`` is set to ``True``, we skip sites
    that are not polarized, otherwise we use the reference allele as the middle base.
    """

    def __init__(
            self,
            fasta_file: str,
            n_flanking: int = 1,
            aliases: Dict[str, List[str]] = {},
            cache: bool = True
    ):
        """
        Create instance. Note that we require a fasta file to be specified
        for base context to be able to be inferred

        :param fasta_file: The fasta file path, possibly gzipped or a URL
        :param n_flanking: The number of flanking bases
        :param aliases: Dictionary of aliases for the contigs in the VCF file, e.g. ``{'chr1': ['1']}``.
            This is used to match the contig names in the VCF file with the contig names in the FASTA file and GFF file.
        :param cache: Whether to cache files that are downloaded from URLs
        """
        Stratification.__init__(self)

        FASTAHandler.__init__(self, fasta_file, cache=cache, aliases=aliases)

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
    def get_type(self, variant: Variant) -> str:
        """
        Get the base context for a given mutation

        :param variant: The vcf site
        :return: Base context of the mutation
        """
        pos = variant.POS - 1

        # get the ancestral allele
        aa = self.handler._get_ancestral(variant)

        # get aliases
        aliases = self.get_aliases(variant.CHROM)

        # check if contig is up-to-date
        if self.contig is None or self.contig.id not in aliases:
            self.logger.debug(f"Fetching contig '{variant.CHROM}'.")

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


class BaseTransitionStratification(Stratification):
    """
    Stratify the SFS by the base transition of the mutation, i.e., ``A>T``.

    TODO handle mono-allelic sites
    """

    @_count_valid_type
    def get_type(self, variant: Variant) -> str:
        """
        Get the base transition for the given variant.

        :param variant: The vcf site
        :return: Base transition
        :raises NoTypeException: if not type could be determined
        """
        if variant.is_snp:
            ancestral = self.handler._get_ancestral(variant)

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

    TODO handle mono-allelic sites
    """

    @_count_valid_type
    def get_type(self, variant: Variant) -> str:
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
    If ``skip_non_polarized`` is set to ``True``, we skip sites
    that are not polarized, otherwise we use the reference allele as ancestral base.
    By default, we use the ``AA`` tag to determine the ancestral allele.

    Any subclass of :class:`~fastdfe.parser.AncestralAnnotation` can be used to annotate the ancestral allele.
    """

    @_count_valid_type
    def get_type(self, variant: Variant) -> str:
        """
        Get the type which is the reference allele.

        :param variant: The vcf site
        :return: reference allele
        """
        return self.handler._get_ancestral(variant)

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

    @_count_valid_type
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
    Stratify SFS by synonymy (neutral or selected).

    :class:`~fastdfe.annotation.SynonymyAnnotation` can be used to annotate the synonymy of a site.
    """

    def get_types(self) -> List[str]:
        """
        Get all possible synonymy types (``neutral`` and ``selected``).

        :return: List of contexts
        """
        return ['neutral', 'selected']

    @_count_valid_type
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

    #: The tag used by VEP to annotate the synonymy
    info_tag = 'CSQ'

    def get_types(self) -> List[str]:
        """
        Get all possible synonymy types (``neutral`` and ``selected``).

        :return: List of contexts
        """
        return ['neutral', 'selected']

    @_count_valid_type
    def get_type(self, variant: Variant) -> Literal['neutral', 'selected']:
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
    """

    #: The tag used by SnpEff to annotate the synonymy
    info_tag = 'ANN'


class GenomePositionDependentStratification(Stratification, ABC):
    pass


class ContigStratification(GenomePositionDependentStratification):
    """
    Stratify SFS by contig.
    """

    def get_type(self, variant: Variant) -> str:
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
        return list(self.handler._reader.seqnames)


class ChunkedStratification(GenomePositionDependentStratification):
    """
    Stratify SFS by creating ``n`` chunks of roughly equal size.
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

    def _setup(self, handler: MultiHandler):
        """
        Set up the stratification.

        :param handler: The handler
        """
        super()._setup(handler)

        # compute base chunk size and remainder
        base_chunk_size, remainder = divmod(handler.n_sites, self.n_chunks)

        # create list of chunk sizes
        self.chunk_sizes = [base_chunk_size + (i < remainder) for i in range(self.n_chunks)]

    def get_types(self) -> List[str]:
        """
        Get all possible window types.

        :return: List of contexts
        """
        return [f'chunk{i}' for i in range(self.n_chunks)]

    def get_type(self, variant: Variant) -> str:
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
    contain plenty of information on the strength of selection.

    Note that we sample sites randomly to make sure we cover the entire genome when only a subset of the genome is
    considered. This provides more speed as we don't have to parse the entire FASTA file. The actual number of
    monomorphic sites is then extrapolated from ``n_target_sites``.
    """

    def __init__(
            self,
            n_target_sites: int,
            n_samples: int = int(1e6),
    ):
        """
        Initialize counter.

        :param n_target_sites: The total number of sites (mono- and polymorphic) that would be present in the VCF file
        if it contained monomorphic sites. This number should be considerably larger than the number of polymorphic
            sites in the VCF file. This value is not extremely important for the DFE inference, but the order of
            magnitude should be correct in any case.
        :param n_samples: The number of sites to sample from the fasta file.
        """
        #: The logger
        self.logger = logger.getChild(self.__class__.__name__)

        #: The total number of sites considered when parsing the VCF
        self.n_target_sites: int | None = int(n_target_sites)

        #: Number of samples
        self.n_samples: int = int(n_samples)

        #: The size of the contigs
        self._contig_sizes: Dict[str, int] = {}

    def _setup(self, parser: 'Parser'):
        """
        Set up the counter.

        :param parser: The parser
        """
        self.parser = parser

    def _get_bounds(self, aliases: List[str]) -> tuple[int, int]:
        """
        Get the bounds for the site positions on the given contig.

        :param aliases: The contig aliases
        :return: The bounds
        """
        for alias in aliases:
            if alias in self.parser._positions:
                return np.min(self.parser._positions[alias]), np.max(self.parser._positions[alias])

        raise ValueError(f"Contig '{aliases}' not found in VCF file, even though it was previously parsed. "
                         f"This should not happen.")

    def count(self):
        """
        Count the number of target sites.

        :return: The number of target sites
        """
        # rewind parser components
        self.parser._rewind()

        # count the number of sites per contig
        self.count_contig_sizes()

        # rewind fasta iterator
        FASTAHandler._rewind(self.parser)

        # initialize random number generator
        rng = np.random.default_rng(self.parser.seed)

        from . import disable_pbar

        # initialize progress bar
        pbar = tqdm(total=self.n_samples, desc='Sampling target sites', disable=disable_pbar)

        # determine sampling probabilities
        probs = np.array(list(self._contig_sizes.values())) / sum(self._contig_sizes.values())

        # sample number of sites per contig
        samples = rng.multinomial(self.n_samples, probs)

        # initialize counter
        i = 0

        # iterate over contigs
        for contig, n in zip(self._contig_sizes.keys(), samples):

            # get aliases
            aliases = self.parser.get_aliases(contig)

            # get bounds for site positions
            bounds = self._get_bounds(aliases)

            # make sure we have a valid range
            if bounds[1] > bounds[0]:

                self.logger.debug(f"Sampling {n} sites from contig '{contig}'.")

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

        # tear down parser components
        self.parser._teardown()

        # notify on number of sites included in the SFS
        self.logger.info(f"{i} out of {self.n_samples} sampled sites were valid.")

    def _update_target_sites(self, spectra: Spectra) -> Spectra:
        """
        Update the target sites of the spectra.

        :param spectra: The spectra.
        :return: The updated spectra.
        """
        # copy spectra
        spectra = spectra.copy()

        # get number of monomorphic and polymorphic sites
        n_monomorphic = spectra.data.iloc[0, :].sum()
        n_polymorphic = spectra.data.iloc[1:, :].sum().sum()

        # check if we have enough target sites
        if self.n_target_sites < n_polymorphic:
            self.logger.warning(f"Number of polymorphic sites ({n_polymorphic}) exceeds the "
                                f"number of target sites ({self.n_target_sites}) which does not make sense. "
                                f"We leave the number of target sites unchanged. "
                                f"Please remember to modify the number of target sites accordingly "
                                f"if your VCF file contains only contains polymorphic sites.")
        elif n_monomorphic == 0:
            self.logger.warning(f"Number of monomorphic sites is zero which should only happen "
                                f"if there are very few sites considered. Failed to update "
                                f"the number of target sites.")
        else:

            # compute multiplicative factor
            x = (self.n_target_sites - n_polymorphic) / n_monomorphic

            spectra.data.iloc[0, :] *= x

        return spectra

    def _get_records(self) -> Iterable[SeqRecord]:
        """
        Get a generator for the contigs to consider.
        """
        # iterate over contigs
        for contig in self.parser._positions.keys():
            aliases = self.parser.get_aliases(contig)

            yield self.parser.get_contig(aliases)

    def count_contig_sizes(self) -> Dict[str, int]:
        """
        Count the total number of sites per contig.

        :return: A dictionary with the number of sites per contig.
        """
        from . import disable_pbar

        # initialize progress bar
        pbar = tqdm(
            desc="Determining contig sizes",
            total=len(self.parser._positions),
            disable=disable_pbar
        )

        # iterate over contigs
        for record in self._get_records():
            self._contig_sizes[record.id] = len(record)
            pbar.update()

        # close progress bar
        pbar.close()

        return self._contig_sizes


class Parser(MultiHandler):
    """
    Parse site-frequency spectra from VCF files.

    By default, the parser looks at the ``AA`` tag in the VCF file's info field to retrieve
    the correct polarization. Sites for which this tag is not well-defined are by default
    included (see ``skip_non_polarized``). Note that non-polarized frequency spectra provide much less
    information on the DFE than polarized spectra.

    This class also offers on-the-fly annotation of the VCF sites such as the degeneracy of the sites
    and their ancestral alleles. This is done by providing a list of annotations to the parser which are
    applied in the order they are provided.

    The parser also allows to filter sites based on site properties. This is done by
    providing a list of filtrations to the parser. By default, we filter out poly-allelic
    sites.

    In addition, the parser allows to stratify the SFS by providing a list of stratifications. This is useful
    to obtain the SFS for different types of sites for which we can jointly infer the DFEs using
    :class:`~fastdfe.joint_inference.JointInference`.

    Note that we assume the sites in the VCF file to be sorted by position in ascending order (per contig).

    Example usage:

    ::

        import fastdfe as fd

        # parse selected and neutral SFS from human chromosome 1
        p = fd.Parser(
            vcf="https://ngs.sanger.ac.uk//production/hgdp/hgdp_wgs.20190516/"
                "hgdp_wgs.20190516.full.chr1.vcf.gz",
            fasta_file="http://ftp.ensembl.org/pub/release-109/fasta/homo_sapiens/"
                       "dna/Homo_sapiens.GRCh38.dna.chromosome.1.fa.gz",
            gff_file="http://ftp.ensembl.org/pub/release-109/gff3/homo_sapiens/"
                     "Homo_sapiens.GRCh38.109.chromosome.1.gff3.gz",
            aliases=dict(chr1=['1']),
            n=10,
            target_site_counter=fd.TargetSiteCounter(
                n_samples=1000000,
                n_target_sites=fd.Annotation.count_target_sites(
                    "http://ftp.ensembl.org/pub/release-109/gff3/homo_sapiens/"
                    "Homo_sapiens.GRCh38.109.chromosome.1.gff3.gz"
                )['1']
            ),
            annotations=[
                fd.DegeneracyAnnotation()
            ],
            filtrations=[
                fd.CodingSequenceFiltration()
            ],
            stratifications=[fd.DegeneracyStratification()],
            info_ancestral='AA_ensembl',
            skip_non_polarized=True
        )

        sfs = p.parse()

        sfs.plot()

    """

    def __init__(
            self,
            vcf: str | Iterable[Variant],
            n: int,
            gff_file: str | None = None,
            fasta_file: str | None = None,
            info_ancestral: str = 'AA',
            skip_non_polarized: bool = False,
            stratifications: List[Stratification] = [],
            annotations: List[Annotation] = [],
            filtrations: List[Filtration] = [PolyAllelicFiltration()],
            samples: List[str] = None,
            max_sites: int = np.inf,
            seed: int | None = 0,
            cache: bool = True,
            aliases: Dict[str, List[str]] = {},
            target_site_counter: TargetSiteCounter = None
    ):
        """
        Initialize the parser.

        :param vcf: The path to the VCF file or an iterable of variants, can be gzipped or a URL.
        :param gff_file: The path to the GFF file, possibly gzipped or a URL. This file is optional and depends on
            the stratifications, annotations and filtrations that are used.
        :param fasta_file: The path to the FASTA file, possibly gzipped or a URL. This file is optional and depends on
            the annotations and filtrations that are used.
        :param n: The size of the resulting SFS. We down-sample to this number by drawing without replacement from
            the set of all available genotypes per site. Sites with fewer than ``n`` genotypes are skipped.
        :param info_ancestral: The tag in the INFO field that contains ancestral allele information. Consider using
            an ancestral allele annotation if this information is not available yet.
        :param skip_non_polarized: Whether to skip poly-morphic sites that are not polarized, i.e., without a valid
            info tag providing the ancestral allele. Default is ``False`` so that we use the reference allele as the
            ancestral allele in such cases.
        :param stratifications: List of stratifications to use.
        :param annotations: List of annotations to use.
        :param filtrations: List of filtrations to use.
        :param samples: List of sample names to consider. If ``None``, all samples are considered.
        :param max_sites: Maximum number of sites to parse from the VCF file.
        :param seed: Seed for the random number generator. Use ``None`` for no seed.
        :param cache: Whether to cache files downloaded from URLs.
        :param aliases: Dictionary of aliases for the contigs in the VCF file, e.g. ``{'chr1': ['1']}``.
            This is used to match the contig names in the VCF file with the contig names in the FASTA file and GFF file.
        :param target_site_counter: The target site counter. If ``None``, we do not sample target sites.
        """
        MultiHandler.__init__(
            self,
            vcf=vcf,
            gff_file=gff_file,
            fasta_file=fasta_file,
            info_ancestral=info_ancestral,
            max_sites=max_sites,
            seed=seed,
            cache=cache,
            aliases=aliases
        )

        # warn if SynonymyAnnotation is used
        if any(isinstance(a, SynonymyAnnotation) for a in annotations):
            logger.warning("SynonymyAnnotation is not recommended to be used with the parser as "
                           "it is not possible to determine the synonymy of monomorphic sites."
                           "Consider using DegeneracyAnnotation instead.")

        #: The target site counter
        self.target_site_counter: TargetSiteCounter | None = target_site_counter

        #: The number of individuals in the sample
        self.n: int = int(n)

        #: The list of samples to use
        self.samples: List[str] | None = samples

        #: The mask of samples to use
        self.samples_mask: np.ndarray | None = None

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

        #: Dictionary of SFS indexed by joint type
        self.sfs: Dict[str, np.ndarray] = defaultdict(lambda: np.zeros(self.n + 1))

        #: 1-based positions of included sites per contig (only when target_site_counter is used)
        self._positions: Dict[str, List[int]] = defaultdict(list)

    def _get_ancestral(self, variant: Variant) -> str:
        """
        Determine the ancestral allele.

        :param variant: The vcf site
        :return: Ancestral allele
        :raises NoTypeException: If the site is not polarized and ``skip_non_polarized`` is ``True``
        """
        if variant.is_snp:
            # obtain ancestral allele
            aa = variant.INFO.get(self.info_ancestral)

            # return the ancestral allele if it is a valid base
            if aa in bases:
                return aa

            # if we skip non-polarized sites, we raise an error here
            if self.skip_non_polarized:
                raise NoTypeException("No valid AA tag found so we skip the site")

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

    def _parse_site(self, variant: Variant) -> bool:
        """
        Parse a single site.

        :param variant: The variant.
        :return: Whether the site was included in the SFS.
        """
        if variant.is_snp:

            # obtain called bases
            genotypes = get_called_bases(variant.gt_bases[self.samples_mask])

            # number of samples
            n_samples = len(genotypes)

            # skip if not enough samples
            if n_samples < self.n:
                self.logger.debug(f'Skipping site due to too few samples at {variant.CHROM}:{variant.POS}.')
                return False

            try:
                # determine ancestral allele
                aa = self._get_ancestral(variant)
            except NoTypeException:
                return False

            # count called bases
            counter = Counter(genotypes)

            # determine ancestral allele count
            n_aa = counter[aa]

            # determine down-projected allele count
            k = self.rng.hypergeometric(ngood=n_samples - n_aa, nbad=n_aa, nsample=self.n)

        # if we have a mono-allelic SNPs
        elif not (variant.is_mnp or variant.is_indel or variant.is_deletion or variant.is_sv) and len(variant.REF) == 1:
            # if we don't have an SNP, we assume the reference allele to be the ancestral allele,
            # so the derived allele count is 0
            # The polarization of monomorphic sites is not important for DFE inference, in any case
            k = 0
        else:
            # skip other types of sites
            self.logger.debug(f'Site is not a valid single nucleotide site at {variant.CHROM}:{variant.POS}.')
            return False

        # try to obtain type
        try:
            # create joint type
            t = '.'.join([s.get_type(variant) for s in self.stratifications]) or 'all'

            # add count by 1
            self.sfs[t][k] += 1

        except NoTypeException as e:
            self.logger.debug(e)
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
        if not self._parse_site(variant):
            return False

        return True

    def _rewind(self):
        """
        Rewind the filtrations, annotations and stratifications.
        """
        super()._rewind()

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
        self.logger.info(f'Using stratification: {representation}.')

        # create samples mask
        if self.samples is None:
            self.samples_mask = np.ones(len(self._reader.samples)).astype(bool)
        else:
            self.samples_mask = np.isin(self._reader.samples, self.samples)

        # setup annotations
        for annotation in self.annotations:
            annotation._setup(self)

        # setup filtrations
        for f in self.filtrations:
            f._setup(self)

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

        pbar = self.get_pbar(total=self.n_sites)

        # iterate over variants
        for i, variant in enumerate(self._reader):

            # handle site
            if self._process_site(variant):

                if self.target_site_counter is not None:
                    # save position
                    self._positions[variant.CHROM] += [variant.POS]
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

        # close VCF reader
        self._reader.close()

        if len(self.sfs) == 0:
            self.logger.warning(f"No sites were included in the spectra. If this is not expected, "
                                "please check that all components work as expected. You can also "
                                "set the log level to DEBUG.")
        else:
            n_included = self.n_sites - self.n_skipped

            self.logger.info(f'Included {n_included} out of {self.n_sites} sites in total from the VCF file.')

        # count target sites
        if self.target_site_counter is not None and self.n_skipped < self.n_sites:
            # count target sites
            self.target_site_counter.count()

            # update target sites
            self.sfs = self.target_site_counter._update_target_sites(Spectra(dict(self.sfs))).to_dict()

        return Spectra(dict(self.sfs)).sort_types()
