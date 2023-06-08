"""
VCF filtrations and a filterer to apply them.
"""

__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"
__date__ = "2023-05-11"

import functools
import logging
from typing import Iterable, List, Optional, Callable, Dict

import numpy as np
import pandas as pd
from cyvcf2 import Variant, Writer, VCF

from .annotation import DegeneracyAnnotation
from .bio_handlers import get_major_base, GFFHandler, VCFHandler

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


class Filtration:
    """
    Base class for filtering sites based on certain criteria.
    """

    #: The number of sites that didn't pass the filter.
    n_filtered: int = 0

    def __init__(self):
        """
        Initialize filtration.
        """
        self.logger = logger.getChild(self.__class__.__name__)

    @_count_filtered
    def filter_site(self, variant: Variant) -> bool:
        """
        Filter site.

        :param variant: The variant to filter.
        :return: ``True`` if the variant should be kept, ``False`` otherwise.
        """
        pass

    def _setup(self, reader: VCF):
        """
        Perform any necessary pre-processing. This method is called before the actual filtration.

        :param reader: The VCF reader.
        """
        pass

    def _teardown(self):
        """
        Perform any necessary post-processing. This method is called after the actual filtration.
        """
        self.logger.info(f"Filtered out {self.n_filtered} sites.")


class SNPFiltration(Filtration):
    """
    Only keep SNPs. Note that this includes discarding mono-morphic sites.
    """

    @_count_filtered
    def filter_site(self, variant: Variant) -> bool:
        """
        Filter site.

        :param variant: The variant to filter.
        :return: ``True`` if the variant is an SNP, ``False`` otherwise.
        """
        return variant.is_snp


class SNVFiltration(Filtration):
    """
    Only keep single site variants (discard indels and MNPs but keep monomorphic sites).
    """

    @_count_filtered
    def filter_site(self, variant: Variant) -> bool:
        """
        Filter site.

        :param variant: The variant to filter.
        :return: ``True`` if the variant is kept, ``False`` otherwise.
        """
        return np.all([alt in ['A', 'C', 'G', 'T'] for alt in [variant.REF] + variant.ALT])


class PolyAllelicFiltration(Filtration):
    """
    Filter out poly-allelic sites.
    """

    @_count_filtered
    def filter_site(self, variant: Variant) -> bool:
        """
        Filter site. Note that we don't check explicitly all alleles, but rather
        rely on ``ALT`` field.

        :param variant: The variant to filter.
        :return: ``True`` if the variant is not poly-allelic, ``False`` otherwise.
        """
        return len(variant.ALT) < 2


class AllFiltration(Filtration):
    """
    Filter out all sites. This is useful for testing purposes.
    """

    @_count_filtered
    def filter_site(self, variant: Variant) -> bool:
        """
        Filter site.

        :param variant: The variant to filter.
        :return: ``False``.
        """
        return False


class NoFiltration(Filtration):
    """
    Do not filter out any sites. This is useful for testing purposes.
    """

    @_count_filtered
    def filter_site(self, variant: Variant) -> bool:
        """
        Filter site.

        :param variant: The variant to filter.
        :return: ``True``.
        """
        return True


class CodingSequenceFiltration(Filtration, GFFHandler):
    """
    Filter out sites that are not in coding sequences. This filter should find frequent use when parsing
    spectra for DFE inference as we only consider sites in coding sequences for this purpose.
    By using it, the annotation and parsing of unnecessary sites can be avoided which increases the speed.
    Note that we assume here that within contigs, sites in the GFF file are sorted by position in ascending order.
    """

    def __init__(self, gff_file: str, aliases: Dict[str, List[str]] = {}, cache: bool = True):
        """
        Create a new filtration instance.

        :param gff_file: The GFF file.
        :param aliases: Dictionary of aliases for the contigs in the VCF file, e.g. ``{'chr1': ['1']}``.
            This is used to match the contig names in the VCF file with the contig names in the FASTA file.
        :param cache: Whether to cache files that are downloaded from URLs
        """
        Filtration.__init__(self)

        GFFHandler.__init__(self, gff_file, cache=cache)

        #: The coding sequence enclosing the current variant or the closest one downstream.
        self.cd: Optional[pd.Series] = None

        #: The number of processed sites.
        self.n_processed: int = 0

        #: The contig aliases.
        self.aliases: Dict[str, List[str]] = aliases

    def _setup(self, reader: VCF):
        """
        Touch the GFF file to load it.

        :param reader: The VCF reader.
        """
        super()._setup(reader)

        # load coding sequences
        # noinspection PyStatementEffect
        self._cds

    @_count_filtered
    def filter_site(self, v: Variant) -> bool:
        """
        Filter site by whether it is in a coding sequence.

        :param v: The variant to filter.
        :return: ``True`` if the variant is in a coding sequence, ``False`` otherwise.
        """
        aliases = self.get_aliases(v.CHROM, self.aliases)

        # if self.cd is None or not on the same chromosome or ends before the variant
        if self.cd is None or self.cd.seqid not in aliases or v.POS > self.cd.end:

            # initialize mock coding sequence
            self.cd = pd.Series({
                'seqid': v.CHROM,
                'start': DegeneracyAnnotation._pos_mock,
                'end': DegeneracyAnnotation._pos_mock
            })

            # find coding sequences downstream
            cds = self._cds[self._cds['seqid'].isin(aliases) & (self._cds['end'] >= v.POS)]

            if not cds.empty:
                # take the first coding sequence
                self.cd = cds.iloc[0]

                if self.cd.start == v.POS:
                    self.logger.debug(f'Found coding sequence for {v.CHROM}:{v.POS}.')
                else:
                    self.logger.debug(f'Found coding sequence downstream of {v.CHROM}:{v.POS}.')

            if self.n_processed == 0 and self.cd.start == DegeneracyAnnotation._pos_mock:
                self.logger.warning(f'No subsequent coding sequence found on the same contig as the first variant. '
                                    f'Please make sure this is the correct GFF file with contig names matching '
                                    f'the VCF file. You can use the aliases parameter to match contig names.')

        self.n_processed += 1

        # check whether the variant is in the current CDS
        if self.cd is not None and self.cd.seqid in aliases and self.cd.start <= v.POS <= self.cd.end:
            return True

        return False


class DeviantOutgroupFiltration(Filtration):
    """
    Filter out sites where the major allele of the specified outgroup samples differs from the major
    allele of the ingroup samples.
    """

    def __init__(self, outgroups: List[str], ingroups: List[str] = None, strict_mode: bool = True):
        """
        Construct DeviantOutgroupFiltration.

        :param outgroups: The name of the outgroup samples to consider.
        :param ingroups: The name of the ingroup samples to consider, defaults to all samples but the outgroups.
        :param strict_mode: Whether to filter out sites where no outgroup sample is present, defaults to ``True``
        which is recommended.
        """
        super().__init__()

        #: The ingroup samples.
        self.ingroups: List[str] | None = ingroups

        #: The outgroup samples.
        self.outgroups: List[str] = outgroups

        #: Whether to filter out sites where no outgroup sample is present.
        self.strict_mode: bool = strict_mode

        #: The samples found in the VCF file.
        self.samples: Optional[np.ndarray] = None

        #: The ingroup mask.
        self.ingroup_mask: Optional[np.ndarray] = None

        #: The outgroup mask.
        self.outgroup_mask: Optional[np.ndarray] = None

    def _setup(self, reader: VCF):
        """
        Touch the reader to load the samples.

        :param reader: The VCF reader.
        """
        super()._setup(reader)

        # create samples array
        self.samples: np.ndarray = np.array(reader.samples)

        # create ingroup and outgroup masks
        self.create_masks()

    def create_masks(self):
        """
        Create ingroup and outgroup masks based on the samples.
        """

        # create outgroup masks
        self.outgroup_mask: np.ndarray = np.isin(self.samples, self.outgroups)

        # create ingroup mask
        if self.ingroups is None:
            self.ingroup_mask = ~self.outgroup_mask
        else:
            self.ingroup_mask = np.isin(self.samples, self.ingroups)

    @_count_filtered
    def filter_site(self, variant: Variant) -> bool:
        """
        Filter site.

        :param variant: The variant to filter.
        :return: ``True`` if the variant should be kept, ``False`` otherwise.
        """
        if variant.is_snp:
            # get major base among ingroup samples
            ingroup_base = get_major_base(variant.gt_bases[self.ingroup_mask])

            # get major base among outgroup samples
            outgroup_base = get_major_base(variant.gt_bases[self.outgroup_mask])

            # filter out if no outgroup base is present and strict mode is enabled
            if outgroup_base is None:
                return not self.strict_mode

            # filter out if outgroup base is different from ingroup base
            return ingroup_base == outgroup_base

        return True


class BiasedGCConversionFiltration(Filtration):
    """
    Only retain A<->T and G<->C substitutions (which are unaffected
    by biased gene conversion, see [CITGB]_).

    Mono-allelic sites are always retained, and we assume sites are at most bi-allelic.

    .. [CITGB] See Pouyet et al., 'Background selection and biased
        gene conversion affect more than 95% of the human genome and bias demographic inferences.',
        Elife, 7:e36317, 2018
    """

    @_count_filtered
    def filter_site(self, variant: Variant) -> bool:
        """
        Remove bi-allelic sites that are not A<->T or G<->C substitutions.

        :param variant: The variant to filter.
        :return: ``True`` if the variant should be kept, ``False`` otherwise.
        """
        if variant.is_snp and len(variant.ALT) > 0:
            return variant.REF in 'AT' and variant.ALT[0] in 'AT' or variant.REF in 'GC' and variant.ALT[0] in 'GC'

        return True


class Filterer(VCFHandler):
    """
    Filter a VCF file using a list of filtrations.
    """

    def __init__(
            self,
            vcf: str | Iterable[Variant],
            output: str,
            filtrations: List[Filtration] = [],
            info_ancestral: str = 'AA',
            max_sites: int = np.inf,
            seed: int | None = 0,
            cache: bool = True
    ):
        """
        Create a new filter instance.

        :param vcf: The VCF file.
        :param output: The output file.
        :param filtrations: The filtrations.
        :param info_ancestral: The info field for the ancestral allele.
        :param max_sites: The maximum number of sites to process.
        :param seed: The seed for the random number generator. Use ``None`` for no seed.
        :param cache: Whether to cache files downloaded from urls.
        """
        super().__init__(
            vcf=vcf,
            info_ancestral=info_ancestral,
            max_sites=max_sites,
            seed=seed,
            cache=cache
        )

        #: The filtrations.
        self.filtrations: List[Filtration] = filtrations

        #: The output file.
        self.output: str = output

        #: The number of sites that did not pass the filters.
        self.n_filtered: int = 0

    def is_filtered(self, variant: Variant) -> bool:
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

    def filter(self):
        """
        Filter the VCF.
        """
        self.logger.info('Start filtering')

        # count the number of sites
        self.n_sites = self.count_sites()

        # create the reader
        reader = VCF(self.download_if_url(self.vcf))

        # create the writer
        writer = Writer(self.output, reader)

        # setup filtrations
        for f in self.filtrations:
            f._setup(reader)

        # get progress bar
        with self.get_pbar() as pbar:

            # iterate over the sites
            for i, variant in enumerate(reader):

                if self.is_filtered(variant):
                    # write the variant
                    writer.write_record(variant)

                pbar.update()

                # explicitly stopping after ``n``sites fixes a bug with cyvcf2:
                # 'error parsing variant with `htslib::bcf_read` error-code: 0 and ret: -2'
                if i + 1 == self.n_sites or i + 1 == self.max_sites:
                    break

        # teardown filtrations
        for f in self.filtrations:
            f._teardown()

        # close the writer and reader
        writer.close()
        reader.close()

        self.logger.info(f'Filtered out {self.n_filtered} out of {self.n_filtered} sites.')
