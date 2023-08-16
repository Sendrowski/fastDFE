"""
Handlers the reading of VCF, GFF and FASTA files.
"""

__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"
__date__ = "2023-05-29"

import functools
import gzip
import hashlib
import logging
import os
import shutil
import tempfile
from collections import Counter
from functools import cached_property
from typing import List, Iterable, TextIO, Callable, Dict, Optional
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import requests
from Bio import SeqIO
from Bio.SeqIO.FastaIO import FastaIterator
from Bio.SeqRecord import SeqRecord
from cyvcf2 import Variant
from tqdm import tqdm

#: The DNA bases
bases = ["A", "C", "G", "T"]

# logger
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

        with tqdm(disable=disable_pbar, desc='Counting sites') as pbar:

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


def count_no_type(func: Callable) -> Callable:
    """
    Decorator that increases ``self.n_no_type`` by 1 if the decorated function raises a ``NoTypeException``.
    """

    @functools.wraps(func)
    def wrapper(self, variant):
        try:
            return func(self, variant)
        except NoTypeException as e:
            self.n_no_type += 1
            raise e

    return wrapper


class FileHandler:
    """
    Base class for file handling.
    """

    def __init__(self, cache: bool = True):
        """
        Create a new FileHandler instance.

        :param cache: Whether to cache files that are downloaded from URLs
        """
        #: The logger instance
        self.logger = logger.getChild(self.__class__.__name__)

        #: Whether to cache files that are downloaded from URLs
        self.cache: bool = cache

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

    @staticmethod
    def get_aliases(contig: str, aliases: Dict[str, List[str]]) -> List[str]:
        """
        Get the alias for the given contig including the contig name itself.

        :return: The aliases.
        """
        if contig in aliases:
            return list(aliases[contig]) + [contig]

        return [contig]


class FASTAHandler(FileHandler):

    def __init__(self, fasta_file: str, cache: bool = True):
        """
        Create a new FASTAHandler instance.

        :param fasta_file: The path to the FASTA file.
        :param cache: Whether to cache files that are downloaded from URLs
        """
        super().__init__(cache=cache)

        #: The path to the FASTA file.
        self.fasta_file: str = fasta_file

    @cached_property
    def _ref(self) -> FastaIterator:
        """
        Get the reference reader.

        :return: The reference reader.
        """
        return self.load_fasta(self.fasta_file)

    def load_fasta(self, file: str) -> FastaIterator:
        """
        Load a FASTA file into a dictionary.

        :param file: The path to The FASTA file path, possibly gzipped or a URL
        :return: Iterator over the sequences.
        """
        self.logger.info("Loading FASTA file")

        # download and unzip if necessary
        local_file = self.unzip_if_zipped(self.download_if_url(file))

        return SeqIO.parse(local_file, 'fasta')

    def get_contig(self, aliases, rewind: bool = True) -> SeqRecord:
        """
        Get the contig from the FASTA file.

        Note that ``pyfaidx`` would be more efficient here, but there were problems when running it in parallel.

        :param aliases: The contig aliases.
        :param rewind: Whether to allow for rewinding the iterator if the contig is not found.
        :return: The contig.
        """
        try:
            contig = next(self._ref)

            while contig.id not in aliases:
                contig = next(self._ref)

        except StopIteration:

            # if rewind is ``True``, we can rewind the iterator and try again
            # this might be necessary if the FASTA file and the VCF have a different order of contigs
            if rewind:
                self.logger.info("Rewinding FASTA iterator. The FASTA file and the "
                                 "VCF file might have a different order of contigs.")

                # renew fasta iterator
                # noinspection all
                del self._ref

                return self.get_contig(aliases, rewind=False)

            raise LookupError(f'None of the contig aliases {aliases} were found in the FASTA file.')

        return contig


class GFFHandler(FileHandler):
    """
    GFF handler.
    """

    def __init__(self, gff_file: str, cache: bool = True):
        """
        Constructor.

        :param gff_file: The path to the GFF file.
        :param cache: Whether to cache the file.
        """
        FileHandler.__init__(self, cache=cache)

        #: The logger
        self.logger = logger.getChild(self.__class__.__name__)

        #: The GFF file path
        self.gff_file = gff_file

    @cached_property
    def _cds(self) -> pd.DataFrame:
        """
        The coding sequences.

        :return: Dataframe with coding sequences.
        """
        return self._load_cds()

    def _load_cds(self) -> pd.DataFrame:
        """
        Load coding sequences from a GFF file.

        :return: The DataFrame.
        """
        # download and unzip if necessary
        local_file = self.unzip_if_zipped(self.download_if_url(self.gff_file))

        # column labels for GFF file
        col_labels = ['seqid', 'source', 'type', 'start', 'end', 'score', 'strand', 'phase', 'attributes']

        dtypes = dict(
            seqid='category',
            type='category',
            start=float,  # temporarily load as float to handle NA values
            end=float,  # temporarily load as float to handle NA values
            strand='category',
            phase='category'
        )

        self.logger.info(f'Loading GFF file.')

        # load GFF file
        df = pd.read_csv(
            local_file,
            sep='\t',
            comment='#',
            names=col_labels,
            dtype=dtypes,
            usecols=['seqid', 'type', 'start', 'end', 'strand', 'phase']
        )

        # filter for coding sequences
        df = df[df['type'] == 'CDS']

        # drop rows with NA values
        df = df.dropna()

        # convert start and end to int
        df['start'] = df['start'].astype(int)
        df['end'] = df['end'].astype(int)

        # drop type column
        df.drop(columns=['type'], inplace=True)

        # remove duplicates
        df = df.drop_duplicates(subset=['seqid', 'start', 'end'])

        # sort by seqid and start
        df.sort_values(by=['seqid', 'start'], inplace=True)

        return df

    def _count_target_sites(self) -> Dict[str, int]:
        """
        Count the number of target sites in a GFF file.

        :return: The number of target sites per chromosome/contig.
        """
        cds = self._compute_lengths(self._load_cds())

        # group by 'seqid' and calculate the sum of 'length'
        target_sites = cds.groupby('seqid')['length'].sum().to_dict()

        return target_sites

    @staticmethod
    def remove_overlaps(df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove overlapping coding sequences.

        :param df: The coding sequences.
        :return: The coding sequences without overlaps.
        """
        df['overlap'] = df['start'].shift(-1) <= df['end']

        df = df[~df['overlap']]

        return df.drop(columns=['overlap'])

    @staticmethod
    def _compute_lengths(cds: pd.DataFrame) -> pd.DataFrame:
        """
        Compute coding sequences lengths.

        :param cds: The coding sequences.
        :return: The coding sequences with lengths.
        """
        # remove duplicates
        cds = cds.drop_duplicates(subset=['seqid', 'start'])

        # remove overlaps
        # cds = GFFHandler.remove_overlaps(cds)

        # create a new column for the difference between 'end' and 'start'
        cds.loc[:, 'length'] = cds['end'] - cds['start'] + 1

        return cds


class VCFHandler(FileHandler):
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
        :param seed: Seed for the random number generator. Use ``None`` for no seed.
        :param cache: Whether to cache files that are downloaded from URLs
        """
        super().__init__(cache=cache)

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

    def get_pbar(self, desc: str = "Processing sites") -> tqdm:
        """
        Return a progress bar for the number of sites.

        :param desc: Description for the progress bar
        :return: tqdm
        """
        from . import disable_pbar

        return tqdm(total=self.n_sites, disable=disable_pbar, desc=desc)


class NoTypeException(BaseException):
    """
    Exception thrown when no type can be determined.
    """
    pass
