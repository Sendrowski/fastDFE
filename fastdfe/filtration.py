"""
VCF filtrations and a filterer to apply them.
"""

__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"
__date__ = "2023-05-11"

import functools
import logging
from abc import ABC, abstractmethod
from typing import Iterable, List, Optional, Callable, Dict

import numpy as np
import pandas as pd
from cyvcf2 import Variant, Writer

from .annotation import DegeneracyAnnotation
from .io_handlers import get_major_base, MultiHandler, get_called_bases, DummyVariant

# get logger
logger = logging.getLogger('fastdfe')


def _count_filtered(func: Callable) -> Callable:
    """
    Decorator that increases ``self.n_filtered`` by 1 if the decorated function returns False.
    """

    @functools.wraps(func)
    def wrapper(self, variant):
        result = func(self, variant)
        if not result:
            self.n_filtered += 1
        return result

    return wrapper


class Filtration(ABC):
    """
    Base class for filtering sites based on certain criteria.
    """

    #: The number of sites that didn't pass the filter.
    n_filtered: int = 0

    def __init__(self):
        """
        Initialize filtration.
        """
        #: The logger.
        self._logger = logger.getChild(self.__class__.__name__)

        #: The handler.
        self._handler: MultiHandler | None = None

    @abstractmethod
    @_count_filtered
    def filter_site(self, variant: Variant | DummyVariant) -> bool:
        """
        Filter site.

        :param variant: The variant to filter.
        :return: ``True`` if the variant should be kept, ``False`` otherwise.
        """
        pass

    def _setup(self, handler: MultiHandler):
        """
        Perform any necessary pre-processing. This method is called before the actual filtration.

        :param handler: The handler.
        """
        self._handler = handler

    def _rewind(self):
        """
        Rewind the filtration.
        """
        self.n_filtered = 0

    def _teardown(self):
        """
        Perform any necessary post-processing. This method is called after the actual filtration.
        """
        self._logger.info(f"Filtered out {self.n_filtered} sites.")


class MaskedFiltration(Filtration, ABC):
    """
    Filter sites based on a samples mask.
    """

    def __init__(
            self,
            use_parser: bool = True,
            include_samples: List[str] | None = None,
            exclude_samples: List[str] | None = None
    ):
        """
        Create a new filtration instance.

        :param use_parser: Whether to use the samples mask from the parser, if used together with parser.
        :param include_samples: The samples to include, defaults to all samples.
        :param exclude_samples: The samples to exclude, defaults to no samples.
        """
        super().__init__()

        #: Whether to use the samples mask from the parser, if used together with parser.
        self.use_parser: bool = use_parser

        #: The samples to include.
        self.include_samples: List[str] | None = include_samples

        #: The samples to exclude.
        self.exclude_samples: List[str] | None = exclude_samples

        #: The samples mask.
        self._samples_mask: np.ndarray | None = None

    def _prepare_samples_mask(self) -> np.ndarray | None:
        """
        Prepare the samples mask.

        :return: The samples mask.
        """
        from .parser import Parser

        if self.use_parser and isinstance(self._handler, Parser):

            # use samples mask from parser
            self._samples_mask = self._handler._samples_mask

        else:

            # determine samples to include
            if self.include_samples is None:

                mask = np.ones(len(self._handler._reader.samples)).astype(bool)
            else:
                mask = np.isin(self._handler._reader.samples, self.include_samples)

            # determine samples to exclude
            if self.exclude_samples is not None:
                mask &= ~np.isin(self._handler._reader.samples, self.exclude_samples)

            # set samples mask only if not all samples are included
            if not np.all(mask):
                self._samples_mask = mask

    def _setup(self, handler: MultiHandler):
        """
        Prepare the samples mask.

        :param handler: The handler.
        """
        super()._setup(handler)

        # prepare samples mask
        self._prepare_samples_mask()


class SNPFiltration(MaskedFiltration):
    """
    Only keep SNPs. Note that this entails discarding mono-morphic sites.
    """

    @_count_filtered
    def filter_site(self, variant: Variant | DummyVariant) -> bool:
        """
        Filter site.

        :param variant: The variant to filter.
        :return: ``True`` if the variant is an SNP, ``False`` otherwise.
        """
        # simply check whether the variant is an SNP if we don't have a samples mask
        if self._samples_mask is None or isinstance(variant, DummyVariant):
            return variant.is_snp

        # otherwise check whether the variant is an SNP among the included samples
        return len(np.unique(get_called_bases(variant.gt_bases[self._samples_mask]))) > 1


class SNVFiltration(Filtration):
    """
    Only keep single site variants (discard indels and MNPs but keep monomorphic sites).
    """

    @_count_filtered
    def filter_site(self, variant: Variant | DummyVariant) -> bool:
        """
        Filter site.

        :param variant: The variant to filter.
        :return: ``True`` if the variant is kept, ``False`` otherwise.
        """
        return np.all([alt in ['A', 'C', 'G', 'T'] for alt in [variant.REF] + variant.ALT])


class PolyAllelicFiltration(MaskedFiltration):
    """
    Filter out poly-allelic sites.
    """

    @_count_filtered
    def filter_site(self, variant: Variant | DummyVariant) -> bool:
        """
        Filter site. Note that we don't check explicitly all alleles, but rather
        rely on ``ALT`` field.

        :param variant: The variant to filter.
        :return: ``True`` if the variant is not poly-allelic, ``False`` otherwise.
        """
        # if we don't have a samples mask, simply check whether the variant is poly-allelic
        if self._samples_mask is None or isinstance(variant, DummyVariant):
            return len(variant.ALT) < 2

        # otherwise check whether the variant is poly-allelic among the included samples
        return len(np.unique(get_called_bases(variant.gt_bases[self._samples_mask]))) < 3


class AllFiltration(Filtration):
    """
    Filter out all sites. Only useful for testing purposes.
    """

    @_count_filtered
    def filter_site(self, variant: Variant | DummyVariant) -> bool:
        """
        Filter site.

        :param variant: The variant to filter.
        :return: ``False``.
        """
        return False


class NoFiltration(Filtration):
    """
    Do not filter out any sites. Only useful for testing purposes.
    """

    @_count_filtered
    def filter_site(self, variant: Variant | DummyVariant) -> bool:
        """
        Filter site.

        :param variant: The variant to filter.
        :return: ``True``.
        """
        return True


class CodingSequenceFiltration(Filtration):
    """
    Filter out sites that are not in coding sequences. This filter should find frequent use when parsing
    spectra for DFE inference as we only consider sites in coding sequences for this purpose.
    By using it, the annotation and parsing of unnecessary sites can be avoided which increases the speed.
    Note that we assume here that within contigs, sites in the GFF file are sorted by position in ascending order.

    For this filtration to work, we require a GFF file (passed to :class:`~fastdfe.parser.Parser` or
    :class:`~fastdfe.filtration.Filterer`).
    """

    def __init__(self):
        """
        Create a new filtration instance.
        """
        Filtration.__init__(self)

        #: The coding sequence enclosing the current variant or the closest one downstream.
        self.cd: Optional[pd.Series] = None

        #: The number of processed sites.
        self.n_processed: int = 0

    def _setup(self, handler: MultiHandler):
        """
        Touch the GFF file to load it.

        :param handler: The handler.
        """
        # require GFF file
        handler._require_gff(self.__class__.__name__)

        # setup GFF handler
        super()._setup(handler)

        # load coding sequences
        _ = handler._cds

    def _rewind(self):
        """
        Rewind the filtration.
        """
        super()._rewind()

        # reset coding sequence
        self.cd = None

    @_count_filtered
    def filter_site(self, v: Variant | DummyVariant) -> bool:
        """
        Filter site by whether it is in a coding sequence.

        :param v: The variant to filter.
        :return: ``True`` if the variant is in a coding sequence, ``False`` otherwise.
        """
        aliases = self._handler.get_aliases(v.CHROM)

        # if self.cd is None or not on the same chromosome or ends before the variant
        if self.cd is None or self.cd.seqid not in aliases or v.POS > self.cd.end:

            # initialize mock coding sequence
            self.cd = pd.Series({
                'seqid': v.CHROM,
                'start': DegeneracyAnnotation._pos_mock,
                'end': DegeneracyAnnotation._pos_mock
            })

            # find coding sequences downstream
            cds = self._handler._cds[self._handler._cds['seqid'].isin(aliases) & (self._handler._cds['end'] >= v.POS)]

            if not cds.empty:
                # take the first coding sequence
                self.cd = cds.iloc[0]

                if self.cd.start == v.POS:
                    self._logger.debug(f'Found coding sequence for {v.CHROM}:{v.POS}.')
                else:
                    self._logger.debug(f'Found coding sequence downstream of {v.CHROM}:{v.POS}.')

            if self.n_processed == 0 and self.cd.start == DegeneracyAnnotation._pos_mock:
                self._logger.warning(f'No subsequent coding sequence found on the same contig as the first variant. '
                                     f'Please make sure this is the correct GFF file with contig names matching '
                                     f'the VCF file. You can use the aliases parameter to match contig names.')

        self.n_processed += 1

        # check whether the variant is in the current coding sequence
        if self.cd is not None and self.cd.seqid in aliases and self.cd.start <= v.POS <= self.cd.end:
            return True

        return False


class DeviantOutgroupFiltration(Filtration):
    """
    Filter out sites where the major allele of the specified outgroup samples differs from the major
    allele of the ingroup samples.
    """

    def __init__(
            self,
            outgroups: List[str],
            ingroups: List[str] = None,
            strict_mode: bool = True,
            retain_monomorphic: bool = True
    ):
        """
        Construct DeviantOutgroupFiltration.

        :param outgroups: The name of the outgroup samples to consider.
        :param ingroups: The name of the ingroup samples to consider, defaults to all samples but the outgroups.
        :param strict_mode: Whether to filter out sites where no outgroup sample is present, defaults to ``True``.
        :param retain_monomorphic: Whether to retain monomorphic sites, defaults to ``True``, which is faster.
        """
        super().__init__()

        #: The ingroup samples.
        self.ingroups: List[str] | None = ingroups

        #: The outgroup samples.
        self.outgroups: List[str] = outgroups

        #: Whether to filter out sites where no outgroup sample is present.
        self.strict_mode: bool = strict_mode

        #: Whether to retain monomorphic sites.
        self.retain_monomorphic: bool = retain_monomorphic

        #: The samples found in the VCF file.
        self.samples: Optional[np.ndarray] = None

        #: The ingroup mask.
        self.ingroup_mask: Optional[np.ndarray] = None

        #: The outgroup mask.
        self.outgroup_mask: Optional[np.ndarray] = None

    def _setup(self, handler: MultiHandler):
        """
        Touch the reader to load the samples.

        :param handler: The handler.
        """
        super()._setup(handler)

        # create samples array
        self.samples: np.ndarray = np.array(handler._reader.samples)

        # create ingroup and outgroup masks
        self._create_masks()

    def _create_masks(self):
        """
        Create ingroup and outgroup masks based on the samples.
        """

        # create outgroup masks
        self.outgroup_mask: np.ndarray = np.isin(self.samples, self.outgroups)

        # make sure all outgroups are present
        if self.outgroup_mask.sum() != len(self.outgroups):
            raise ValueError(f'Not all outgroup samples are present in the VCF file: {self.outgroups}')

        # create ingroup mask
        if self.ingroups is None:
            self.ingroup_mask = ~self.outgroup_mask
        else:
            self.ingroup_mask = np.isin(self.samples, self.ingroups)

    @_count_filtered
    def filter_site(self, variant: Variant | DummyVariant) -> bool:
        """
        Filter site.

        :param variant: The variant to filter.
        :return: ``True`` if the variant should be kept, ``False`` otherwise.
        """
        # keep monomorphic sites if requested
        if not variant.is_snp and self.retain_monomorphic:
            return True

        # filter out dummies if retain_monomorphic is false
        if isinstance(variant, DummyVariant):
            return False

        # get major base among ingroup samples
        ingroup_base = get_major_base(variant.gt_bases[self.ingroup_mask])

        # get major base among outgroup samples
        outgroup_base = get_major_base(variant.gt_bases[self.outgroup_mask])

        # filter out if no outgroup base is present and strict mode is enabled
        if outgroup_base is None:
            return not self.strict_mode

        # filter out if outgroup base is different from ingroup base
        return ingroup_base == outgroup_base


class ExistingOutgroupFiltration(Filtration):
    """
    Filter out sites for which at least one of the specified outgroup samples has no called base.
    """

    def __init__(self, outgroups: List[str]):
        """
        Construct ExistingOutgroupFiltration.

        :param outgroups: The name of the outgroup samples that need to be present to pass the filter.
        """
        super().__init__()

        #: The outgroup samples.
        self.outgroups: List[str] = outgroups

        #: The samples found in the VCF file.
        self.samples: Optional[np.ndarray] = None

        #: The outgroup mask.
        self.outgroup_mask: Optional[np.ndarray] = None

    def _setup(self, handler: MultiHandler):
        """
        Touch the reader to load the samples.

        :param handler: The handler.
        """
        super()._setup(handler)

        # create samples array
        self.samples: np.ndarray = np.array(handler._reader.samples)

        # create outgroup mask
        self._create_mask()

    def _create_mask(self):
        """
        Create outgroup mask based on the samples.
        """
        self.outgroup_mask: np.ndarray = np.isin(self.samples, self.outgroups)

    @_count_filtered
    def filter_site(self, variant: Variant | DummyVariant) -> bool:
        """
        Filter site.

        :param variant: The variant to filter.
        :return: ``True`` if the variant should be kept, ``False`` otherwise.
        """
        # keep dummy variants
        if isinstance(variant, DummyVariant):
            return True

        # get outgroup genotypes
        outgroups = variant.gt_bases[self.outgroup_mask]

        # filter out if at least one outgroup has no called base
        for outgroup in outgroups:
            if len(get_called_bases(outgroup)) == 0:
                return False

        return True


class BiasedGCConversionFiltration(Filtration):
    """
    Only retain A<->T and G<->C substitutions (which are unaffected
    by biased gene conversion, see [CITGB]_).

    Mono-allelic sites are always retained, and we assume sites are at most bi-allelic. Note that the number of
    mutational target sites is reduced by this filtration.

    .. [CITGB] Pouyet et al., 'Background selection and biased
        gene conversion affect more than 95% of the human genome and bias demographic inferences.',
        Elife, 7:e36317, 2018
    """

    @_count_filtered
    def filter_site(self, variant: Variant | DummyVariant) -> bool:
        """
        Remove bi-allelic sites that are not A<->T or G<->C substitutions.

        :param variant: The variant to filter.
        :return: ``True`` if the variant should be kept, ``False`` otherwise.
        """
        if variant.is_snp and len(variant.ALT) > 0:
            return (variant.REF, variant.ALT[0]) in [('A', 'T'), ('T', 'A'), ('G', 'C'), ('C', 'G')]

        return True


class Filterer(MultiHandler):
    """
    Filter a VCF file using a list of filtrations.

    Example usage:

    ::

        import fastdfe as fd

        # only keep variants in coding sequences
        f = fd.Filterer(
            vcf="http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/"
                "1000_genomes_project/release/20181203_biallelic_SNV/"
                "ALL.chr21.shapeit2_integrated_v1a.GRCh38.20181129.phased.vcf.gz",
            gff="http://ftp.ensembl.org/pub/release-109/gff3/homo_sapiens/"
                "Homo_sapiens.GRCh38.109.chromosome.21.gff3.gz",
            output='sapiens.chr21.coding.vcf.gz',
            filtrations=[fd.CodingSequenceFiltration()],
            aliases=dict(chr21=['21'])
        )

        f.filter()

    """

    def __init__(
            self,
            vcf: str | Iterable[Variant],
            output: str,
            gff: str | None = None,
            filtrations: List[Filtration] = [],
            info_ancestral: str = 'AA',
            max_sites: int = np.inf,
            seed: int | None = 0,
            cache: bool = True,
            aliases: Dict[str, List[str]] = {}
    ):
        """
        Create a new filter instance.

        :param vcf: The VCF file, possibly gzipped or a URL.
        :param output: The output file.
        :param gff: The GFF file, possibly gzipped or a URL. This argument is required for some filtrations.
        :param filtrations: The filtrations.
        :param info_ancestral: The info field for the ancestral allele.
        :param max_sites: The maximum number of sites to process.
        :param seed: The seed for the random number generator. Use ``None`` for no seed.
        :param cache: Whether to cache files downloaded from urls.
        :param aliases: Dictionary of aliases for the contigs in the VCF file, e.g. ``{'chr1': ['1']}``.
        """
        super().__init__(
            vcf=vcf,
            gff=gff,
            info_ancestral=info_ancestral,
            max_sites=max_sites,
            seed=seed,
            cache=cache,
            aliases=aliases
        )

        #: The filtrations.
        self.filtrations: List[Filtration] = filtrations

        #: The output file.
        self.output: str = output

        #: The number of sites that did not pass the filters.
        self.n_filtered: int = 0

        #: The VCF writer.
        self._writer: Writer | None = None

    def is_filtered(self, variant: Variant | DummyVariant) -> bool:
        """
        Whether the given variant is kept.

        :param variant: The variant to check.
        :return: ``True`` if the variant is kept, ``False`` otherwise.
        """
        # filter the variant
        for filtration in self.filtrations:
            if not filtration.filter_site(variant):
                self.n_filtered += 1
                return False

        return True

    def _setup(self):
        """
        Set up the filtrations.
        """
        # setup filtrations
        for f in self.filtrations:
            f._setup(self)

        # create the writer
        self._writer = Writer(self.output, self._reader)

    def _teardown(self):
        """
        Tear down the filtrations.
        """
        for f in self.filtrations:
            f._teardown()

        # close the writer and reader
        self._writer.close()
        self._reader.close()

    def filter(self):
        """
        Filter the VCF.
        """
        self._logger.info('Start filtering')

        # setup filtrations
        self._setup()

        # get progress bar
        with self.get_pbar(desc=f"{self.__class__.__name__}>Processing sites") as pbar:

            # iterate over the sites
            for i, variant in enumerate(self._reader):

                if self.is_filtered(variant):
                    # write the variant
                    self._writer.write_record(variant)

                pbar.update()

                # explicitly stopping after ``n`` sites fixes a bug with cyvcf2:
                # 'error parsing variant with `htslib::bcf_read` error-code: 0 and ret: -2'
                if i + 1 == self.n_sites or i + 1 == self.max_sites:
                    break

        # teardown filtrations
        self._teardown()

        self._logger.info(f'Filtered out {self.n_filtered} of {self.n_sites} sites in total.')
