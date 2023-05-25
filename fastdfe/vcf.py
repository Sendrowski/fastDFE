"""
VCF module.
"""

__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"
__date__ = "2023-05-09"

import gzip
import hashlib
import logging
import os
import shutil
import tempfile
from typing import Optional, TextIO, Iterable, Dict, List
from urllib.parse import urlparse

import numpy as np
import requests
from cyvcf2 import Variant
from numpy.random import Generator
from pyfaidx import Fasta, FastaRecord
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


def count_sites(vcf: str | Iterable[Variant], max_sites: int = np.inf) -> int:
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

    # whether to disable the progress bar
    from . import disable_pbar

    i = 0
    with open_file(vcf) as f:

        with tqdm(total=max_sites, disable=disable_pbar, desc='Counting sites') as pbar:

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
        return count_sites(self.download_if_url(self.vcf), max_sites=self.max_sites)

    @staticmethod
    def is_url(path: str) -> bool:
        """
        Check if the given path is a URL.

        :param path: The path to check.
        :return: ``True`` if the path is a URL, ``False`` otherwise.
        """
        try:
            result = urlparse(path)
            return all([result.scheme, result.netloc])
        except ValueError:
            return False

    @staticmethod
    def load_fasta(file: str) -> Fasta:
        """
        Load a FASTA file into a dictionary.

        :param file: The path to The FASTA file path, possibly gzipped or a URL
        :return: Iterator over the sequences.
        """
        # download and unzip if necessary
        local_file = VCFHandler.unzip_if_zipped(VCFHandler.download_if_url(file))

        return Fasta(local_file)

    @staticmethod
    def get_contig(fasta: Fasta, aliases) -> FastaRecord:
        """
        Get the contig from the FASTA file.

        :param fasta: The FASTA file.
        :param aliases: The contig aliases.
        :return: The contig.
        """
        for alias in aliases:
            if alias in fasta:
                return fasta[alias]

        raise LookupError(f'None of the contig aliases {aliases} were found in the FASTA file.')

    @staticmethod
    def get_aliases(contig: str, aliases: Dict[str, List[str]]) -> List[str]:
        """
        Get the alias for the given contig including the contig name itself.

        :return: The aliases.
        """
        if contig in aliases:
            return list(aliases[contig]) + [contig]

        return [contig]

    @staticmethod
    def download_if_url(path: str) -> str:
        """
        Download the VCF file if it is a URL.

        :param path: The path to the VCF file.
        :return: The path to the downloaded file or the original path.
        """
        if VCFHandler.is_url(path):
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
    def hash(s: str) -> str:
        """
        Return a truncated SHA1 hash of a string.

        :param s: The string to hash.
        :return: The SHA1 hash.
        """
        return hashlib.sha1(s.encode()).hexdigest()[:12]

    @staticmethod
    def download_file(url: str) -> str:
        """
        Download a file from a URL.

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

        from . import disable_pbar

        # create a temporary file with the original file extension
        with tempfile.NamedTemporaryFile(suffix='.' + filename, delete=False) as tmp:
            total_size = int(response.headers.get('content-length', 0))
            chunk_size = 8192

            # fix bug of missing attribute
            if not hasattr(response, '_content_consumed'):
                response._content_consumed = 0

            with tqdm(total=total_size,
                      unit='B',
                      unit_scale=True,
                      desc="Downloading file",
                      disable=disable_pbar) as pbar:

                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        tmp.write(chunk)
                        pbar.update(len(chunk))

        return tmp.name

    def get_pbar(self) -> tqdm:
        """
        Return a progress bar for the number of sites.

        :return: tqdm
        """
        from . import disable_pbar

        return tqdm(total=self.n_sites, disable=disable_pbar, desc="Processing sites")
