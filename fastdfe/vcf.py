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
from collections import Counter
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


def get_called_bases(genotypes: np.ndarray | List[str]) -> np.ndarray:
    """
    Get the called bases from a list of calls.

    :param genotypes: Array of genotypes in the form of strings.
    :return: Array of called bases.
    """
    return np.array([b for b in '/'.join(genotypes).replace('|', '/') if b in 'ACGT'])


def get_major_base(genotypes: np.ndarray | List[str]) -> str | None:
    """
    Get the major base from a list of calls.

    :param genotypes: Array of genotypes in the form of strings.
    :return: Major base.
    """
    # get the called bases
    bases = get_called_bases(genotypes)

    if len(bases) > 0:
        return Counter(bases).most_common()[0][0]

    return None


def count_sites(vcf: str | Iterable[Variant], max_sites: int = np.inf) -> int:
    """
    Count the number of sites in the VCF.

    :param vcf: The path to the VCF file or an iterable of variants
    :param max_sites: Maximum number of sites to consider
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


def download_if_url(path: str, cache: bool = True) -> str:
    """
    Download the VCF file if it is a URL.

    :param path: The path to the VCF file.
    :param cache: Whether to cache the file.
    :return: The path to the downloaded file or the original path.
    """
    if VCFHandler.is_url(path):
        # download the file and return path
        return VCFHandler.download_file(path, cache=cache)

    return path


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
            seed: int | None = 0,
            cache: bool = True
    ):
        """
        Create a new VCF instance.

        :param vcf: The path to the VCF file or an iterable of variants, can be gzipped, urls are also supported
        :param info_ancestral: The tag in the INFO field that contains the ancestral allele
        :param max_sites: Maximum number of sites to consider
        :param seed: Seed for the random number generator
        :param cache: Whether to cache files that are downloaded from URLs
        """
        self.logger = logger.getChild(self.__class__.__name__)

        #: The path to the VCF file or an iterable of variants
        self.vcf = vcf

        #: The tag in the INFO field that contains the ancestral allele
        self.info_ancestral: str = info_ancestral

        #: Maximum number of sites to consider
        self.max_sites: int = max_sites

        #: Seed for the random number generator
        self.seed: Optional[int] = seed

        #: Whether to cache files that are downloaded from URLs
        self.cache: bool = cache

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

    def load_fasta(self, file: str) -> Fasta:
        """
        Load a FASTA file into a dictionary.

        :param file: The path to The FASTA file path, possibly gzipped or a URL
        :return: Iterator over the sequences.
        """
        self.logger.info("Loading FASTA file")

        # download and unzip if necessary
        local_file = self.unzip_if_zipped(self.download_if_url(file))

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

    def download_if_url(self, path: str) -> str:
        """
        Download the VCF file if it is a URL.

        :param path: The path to the VCF file.
        :return: The path to the downloaded file or the original path.
        """
        return download_if_url(path, cache=self.cache)

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
        return os.path.basename(urlparse(url).path)

    @staticmethod
    def hash(s: str) -> str:
        """
        Return a truncated SHA1 hash of a string.

        :param s: The string to hash.
        :return: The SHA1 hash.
        """
        return hashlib.sha1(s.encode()).hexdigest()[:12]

    @staticmethod
    def download_file(url: str, cache: bool = True) -> str:
        """
        Download a file from a URL.

        :param cache: Whether to cache the file.
        :param url: The URL to download the file from.
        :return: The path to the downloaded file.
        """
        # start the stream
        response = requests.get(url, stream=True)

        # check if the request was successful
        response.raise_for_status()

        # extract the file extension from the URL
        filename = VCFHandler.get_filename(url)

        # create a temporary file path
        path = tempfile.gettempdir() + '/' + VCFHandler.hash(url) + '.' + filename

        # check if the file is already cached
        if cache and os.path.exists(path):
            logger.info(f'Using cached file at {path}')
            return path
        else:
            logger.info(f'Downloading file from {url}')

        from . import disable_pbar

        # create a temporary file with the original file extension
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            total_size = int(response.headers.get('content-length', 0))
            chunk_size = 8192

            with tqdm(total=total_size,
                      unit='B',
                      unit_scale=True,
                      desc="Downloading file",
                      disable=disable_pbar) as pbar:

                # write the file to disk
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        tmp.write(chunk)
                        pbar.update(len(chunk))

        # rename the file to the original file extension
        if cache:
            os.rename(tmp.name, path)

            logger.info(f'Cached file to {path}')

            return path

        return tmp.name

    def get_pbar(self) -> tqdm:
        """
        Return a progress bar for the number of sites.

        :return: tqdm
        """
        from . import disable_pbar

        return tqdm(total=self.n_sites, disable=disable_pbar, desc="Processing sites")
