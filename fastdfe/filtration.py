"""
VCF filters.
"""

__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"
__date__ = "2023-05-11"

import logging
from functools import cached_property
from typing import Iterable, List, Optional, Callable, Dict

import numpy as np
import pandas as pd
from cyvcf2 import Variant, Writer, VCF

from .annotation import Annotation, DegeneracyAnnotation
from .vcf import VCFHandler

# get logger
logger = logging.getLogger('fastdfe')


def count_filtered(func: Callable) -> Callable:
    """
    Decorator that increases ``self.n_filtered`` by 1 if the decorated function returns False.
    """

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

    @count_filtered
    def filter_site(self, variant: Variant) -> bool:
        """
        Filter site.

        :param variant: The variant to filter.
        :return: ``True`` if the variant should be kept, ``False`` otherwise.
        """
        pass

    def _setup(self):
        """
        Perform any necessary pre-processing. This method is called before the actual filtration.
        """
        pass

    def _teardown(self):
        """
        Perform any necessary post-processing. This method is called after the actual filtration.
        """
        logger.info(type(self).__name__ + f": Filtered out {self.n_filtered} sites.")


class SNPFiltration(Filtration):
    """
    Only keep SNPs (discard monomorphic sites).
    """

    @count_filtered
    def filter_site(self, variant: Variant) -> bool:
        """
        Filter site.

        :param variant: The variant to filter.
        :return: ``True`` if the variant is an SNP, ``False`` otherwise.
        """
        return variant.is_snp


class SNVFiltration(Filtration):
    """
    Only keep single site variants (discardi indels and MNPs but keep monomorphic sites).
    """

    @count_filtered
    def filter_site(self, variant: Variant) -> bool:
        """
        Filter site.

        :param variant: The variant to filter.
        :return: ``True`` if the variant is kept, ``False`` otherwise.
        """
        return len(variant.REF) == 1


class PolyAllelicFiltration(Filtration):
    """
    Filter out poly-allelic sites.
    """

    @count_filtered
    def filter_site(self, variant: Variant) -> bool:
        """
        Filter site. Note that we don't check explicitly all alleles, but rather
        rely on ``ALT`` field.

        :param variant: The variant to filter.
        :return: ``True`` if the variant is not poly-allelic, ``False`` otherwise.
        """
        return len(variant.ALT) < 2


class CodingSequenceFiltration(Filtration):
    """
    Filter out sites that are not in coding sequences. This filter should find frequent use when parsing
    spectra for DFE inference as we only consider sites in coding sequences for this purpose.
    By using it, the annotation and parsing of unnecessary sites can be avoided which increases the speed.
    Note that we assume here that within contigs, sites in the GFF file are sorted by position in ascending order.
    """

    def __init__(self, gff_file: str, aliases: Dict[str, List[str]] = {}):
        """
        Create a new filtration instance.

        :param gff_file: The GFF file.
        :param aliases: Dictionary of aliases for the contigs in the VCF file, e.g. ``{'chr1': ['1']}``.
            This is used to match the contig names in the VCF file with the contig names in the FASTA file.
        """
        #: The GFF file.
        self.gff_file: str = gff_file

        #: The coding sequence enclosing the current variant or the closest one downstream.
        self.cd: Optional[pd.Series] = None

        #: The number of processed sites.
        self.n_processed: int = 0

        #: The contig aliases.
        self.aliases: Dict[str, List[str]] = aliases

    def _setup(self):
        """
        Touch the GFF file to load it.
        """
        # noinspection PyStatementEffect
        self._cds

    @cached_property
    def _cds(self) -> pd.DataFrame:
        """
        The coding sequences.

        :return: Dataframe with coding sequences.
        """
        return Annotation._load_cds(self.gff_file)

    @count_filtered
    def filter_site(self, v: Variant) -> bool:
        """
        Filter site by whether it is in a coding sequence.

        :param v: The variant to filter.
        :return: ``True`` if the variant is in a coding sequence, ``False`` otherwise.
        """
        aliases = VCFHandler.get_aliases(v.CHROM, self.aliases)

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
                # take the last coding sequence
                self.cd = cds.iloc[0]

                if self.cd.start == v.POS:
                    logger.debug(f'Found coding sequence for {v.CHROM}:{v.POS}.')
                else:
                    logger.debug(f'Found coding sequence downstream of {v.CHROM}:{v.POS}.')

            if self.n_processed == 0 and self.cd.start == DegeneracyAnnotation._pos_mock:
                logger.warning(f'No subsequent coding sequence found on the same contig as the first variant. '
                               f'Please make sure this is the correct GFF file with contig names matching '
                               f'the VCF file. You can use the aliases parameter to match contig names.')

        self.n_processed += 1

        # check whether the variant is in the current CDS
        if self.cd is not None and self.cd.seqid in aliases and self.cd.start <= v.POS <= self.cd.end:
            return True

        return False


class Filterer(VCFHandler):
    """
    Base class for filters.
    """

    def __init__(
            self,
            vcf: str | Iterable[Variant],
            output: str,
            filtrations: List[Filtration] = [],
            info_ancestral: str = 'AA',
            max_sites: int = np.inf,
            seed: int | None = 0
    ):
        """
        Create a new filter instance.

        :param vcf: The VCF file.
        :param output: The output file.
        :param filtrations: The filtrations.
        :param info_ancestral: The info field for the ancestral allele.
        :param max_sites: The maximum number of sites to process.
        :param seed: The seed for the random number generator.
        """
        super().__init__(
            vcf=vcf,
            info_ancestral=info_ancestral,
            max_sites=max_sites,
            seed=seed
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
        logger.info('Start filtering')

        # count the number of sites
        self.n_sites = self.count_sites()

        # create the reader
        reader = VCF(self.vcf)

        # create the writer
        writer = Writer(self.output, reader)

        # setup filtrations
        for f in self.filtrations:
            f._setup()

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

        logger.info(f'Filtered out {self.n_filtered} out of {self.n_filtered} sites.')
