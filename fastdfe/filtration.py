"""
VCF filters.
"""

__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"
__date__ = "2023-05-11"

import logging
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
from cyvcf2 import Variant, Writer, VCF

from .annotation import Annotation
from .vcf import VCFHandler

# get logger
logger = logging.getLogger('fastdfe')


class Filtration:
    """
    Base class for filtering sites based on certain criteria.
    """

    def filter_site(self, variant: Variant) -> bool:
        """
        Filter site.

        :param variant: The variant to filter.
        :return: ``True`` if the variant should be kept, ``False`` otherwise.
        """
        pass


class SNPFiltration(Filtration):
    """
    Only keep SNPs. Not that this also includes monomorphic sites.
    """

    def filter_site(self, variant: Variant) -> bool:
        """
        Filter site.

        :param variant: The variant to filter.
        :return: ``True`` if the variant is an SNP, ``False`` otherwise.
        """
        return variant.is_snp


class NoPolyAllelicFiltration(Filtration):
    """
    Filter out poly-allelic sites.
    """

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
    Filter out sites that are not in coding sequences.
    """

    def __init__(self, gff_file: str):
        """
        Create a new filtration instance.

        :param gff_file: The GFF file.
        """
        #: The GFF file.
        self.gff_file: str = gff_file

        #: The CDS annotation.
        self.cds: pd.DataFrame = Annotation.load_cds(gff_file)

        #: The current coding sequence.
        self.cd: Optional[pd.Series] = None

    def filter_site(self, v: Variant) -> bool:
        """
        Filter site by whether it is in a coding sequence.

        :param v: The variant to filter.
        :return: ``True`` if the variant is not in a coding sequence, ``False`` otherwise.
        """
        # fetch coding sequence if not up to date
        if self.cd is None or self.cd.seqname != v.CHROM or not (self.cd.start <= v.POS <= self.cd.end):
            rows = self.cds[(self.cds.seqname == v.CHROM) & (self.cds.start <= v.POS) & (v.POS <= self.cds.end)]

            return len(rows) != 0

        return True


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

        #: The number of filtered sites.
        self.n_filtered = 0

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

        # iterate over the sites
        for i, variant in enumerate(self.get_sites(reader)):

            # stop if max_sites was reached
            if i >= self.max_sites:
                break

            # filter the variant
            for filtration in self.filtrations:
                if not filtration.filter_site(variant):
                    self.n_filtered += 1
                    break

            # write the variant
            writer.write_record(variant)

        logger.info(f'Filtered {self.n_filtered} sites.')
