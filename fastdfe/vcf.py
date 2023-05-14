"""
VCF module.
"""

__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"
__date__ = "2023-05-09"

import gzip
import logging
import os
import shutil
import tempfile
from functools import cached_property
from typing import Optional, TextIO, Iterable
from urllib.parse import urlparse

import numpy as np
import requests
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


def count_sites(vcf: str | Iterable[Variant], max_sites: int = np.inf, disable_pbar: bool = False) -> int:
    """
    Count the number of sites in the VCF.

    :param vcf: The path to the VCF file or an iterable of variants
    :param max_sites: Maximum number of sites to consider
    :param disable_pbar: Disable the progress bar
    :return: Number of sites
    """

    # if we don't have a file path, we can just count the number of variants
    if not isinstance(vcf, str):
        return len(list(vcf))

    i = 0
    with open_file(vcf) as f:

        with tqdm(disable=disable_pbar) as pbar:
            for line in f:
                if not line.startswith('#'):
                    i += 1
                    pbar.update()

                # stop counting if max_sites was reached
                if i >= max_sites:
                    break

    return i


def open_file(file: str) -> TextIO:
    """
    Open a file, either gzipped or not.

    :param file: File to open
    :return: stream
    """
    if file.endswith('.gz'):
        return gzip.open(file, "rt")

    return open(file, 'r')


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

        :param vcf: The path to the VCF file or an iterable of variants, can be gzipped, urls are also supported
            but have to start with ``https://``
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

    def count_sites(self) -> int:
        """
        Count the number of sites in the VCF.

        :return: Number of sites
        """
        from . import disable_pbar

        logger.info('Counting number of sites.')

        return count_sites(self.vcf_local_path, max_sites=self.max_sites, disable_pbar=disable_pbar)

    @cached_property
    def vcf_local_path(self) -> Optional[str]:
        """
        Return the path to the given file if it is a local file, otherwise download it.

        :return: Path to the file.
        """
        if isinstance(self.vcf, str):
            return self.download_if_url(self.vcf)

        # return None if we don't have a file path

    @staticmethod
    def download_if_url(path: str) -> str:
        """
        Download the VCF file if it is a URL.

        :param path: The path to the VCF file.
        :return: The path to the downloaded file or the original path.
        """
        if path.startswith('https://'):
            # download the file and return path
            return VCFHandler.download_file(path)

        return path

    @staticmethod
    def unzip_if_zipped(file: str):
        """
        If the given file is gzipped, unzip it and return the path to the unzipped file.
        If the file is not gzipped, return the path to the original file.

        :param file: The path to the file.
        :return: The path to the unzipped file, or the original file if it was not gzipped.
        """
        # check if the file extension is .gz
        if file.endswith('.gz'):
            # create a new file path by removing the .gz extension
            unzipped = file[:-3]

            # unzip file
            with gzip.open(file, 'rb') as f_in:
                with open(unzipped, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)

            return unzipped

        return file

    @staticmethod
    def get_filename(url: str):
        """
        Return the file extension of a URL.

        :param url: The URL to get the file extension from.
        :return: The file extension.
        """
        parsed_url = urlparse(url)
        path = parsed_url.path
        filename = os.path.basename(path)

        return filename

    @staticmethod
    def download_file(url: str) -> str:
        """
        Download a file from a URL and return the path to the downloaded file.

        :param url: The URL to download the file from.
        :return: The path to the downloaded file.
        """
        logger.info(f'Downloading file from {url}')

        # start the stream
        response = requests.get(url, stream=True)

        # check if the request was successful
        response.raise_for_status()

        # extract the file extension from the URL
        filename = VCFHandler.get_filename(url)

        # create a temporary file with the original file extension
        with tempfile.NamedTemporaryFile(suffix='.' + filename, delete=False) as tmp:
            # download the response in chunks
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:  # filter out keep-alive chunks
                    tmp.write(chunk)

        return tmp.name

    def get_sites(self, vcf: Iterable[Variant] = None) -> Iterable[Variant]:
        """
        Return an iterable object over the VCF file's sites.

        :param vcf: The VCF file to iterate over.
        :return: iterable
        """
        from . import disable_pbar

        if vcf is None:
            vcf = VCF(self.vcf_local_path)

        return tqdm(vcf, total=self.n_sites, disable=disable_pbar, colour="black")
