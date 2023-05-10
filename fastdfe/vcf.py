"""
VCF module.
"""

__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"
__date__ = "2023-05-09"

import gzip
import logging
from typing import Optional, TextIO, Iterable

import numpy as np
from cyvcf2 import VCF, Variant
from numpy.random import Generator
from tqdm import tqdm

#: Logger
logger = logging.getLogger('fastdfe')


def get_called_bases(variant: Variant) -> np.ndarray:
    """
    Get the called bases from a list of calls.

    :param variant: The variant to get the called bases from.
    :return: Array of called bases.
    """
    return np.array([b for b in '/'.join(variant.gt_bases).replace('|', '/') if b in 'ACGT'])


class VCFHandler:
    """
    Base class for VCF handling.
    """

    def __init__(
            self,
            vcf: str | Iterable[Variant],
            info_ancestral: str = 'AA',
            max_sites: int = np.inf,
            seed: int | None = 0
    ):
        """
        Create a new VCF instance.

        :param vcf: The path to the VCF file or an iterable of variants
        :param info_ancestral: The tag in the INFO field that contains the ancestral allele
        :param max_sites: Maximum number of sites to consider
        :param seed: Seed for the random number generator
        """

        #: The path to the VCF file or an iterable of variants
        self.vcf = vcf

        #: The tag in the INFO field that contains the ancestral allele
        self.info_ancestral: str = info_ancestral

        #: Maximum number of sites to consider
        self.max_sites: int = max_sites

        #: Seed for the random number generator
        self.seed: Optional[int] = seed

        #: Random generator instance
        self.rng = np.random.default_rng(seed=seed)

        #: Number of sites to consider
        self.n_sites: Optional[int] = None

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
