"""
Handlers the reading of VCF, GFF and FASTA files.
"""

__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"
__date__ = "2023-05-29"

import gzip
import hashlib
import logging
import os
import shutil
import tempfile
import warnings
from collections import Counter
from functools import cached_property
from typing import List, Iterable, TextIO, Dict, Optional, Tuple, Union
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import requests
from Bio import SeqIO
from Bio.SeqIO.FastaIO import FastaIterator
from Bio.SeqRecord import SeqRecord
from cyvcf2 import Variant, VCF
from pandas.errors import SettingWithCopyWarning
from tqdm import tqdm

from .settings import Settings

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
    # join genotypes
    joined_genotypes = ''.join(genotypes).replace('|', '/')

    # convert to numpy array of characters
    char_array = np.array(list(joined_genotypes))

    # return only characters that are in the bases list
    return char_array[np.isin(char_array, bases)]


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


def is_monomorphic_snp(variant: Union[Variant, 'DummyVariant']) -> bool:
    """
    Whether the given variant is a monomorphic SNP.

    :param variant: The vcf site
    :return: Whether the site is a monomorphic SNP
    """
    return (not (variant.is_snp or variant.is_mnp or variant.is_indel or variant.is_deletion or variant.is_sv)
            and variant.REF in bases)


def count_sites(
        vcf: str | Iterable[Variant],
        max_sites: int = np.inf,
        desc: str = 'Counting sites'
) -> int:
    """
    Count the number of sites in the VCF.

    :param vcf: The path to the VCF file or an iterable of variants
    :param max_sites: Maximum number of sites to consider
    :param desc: Description for the progress bar
    :return: Number of sites
    """

    # if we don't have a file path, we can just count the number of variants
    if not isinstance(vcf, str):
        return len(list(vcf))

    i = 0
    with open_file(vcf) as f:

        with tqdm(disable=Settings.disable_pbar, desc=desc) as pbar:

            for line in f:
                if not line.startswith('#'):
                    i += 1
                    pbar.update()

                # stop counting if max_sites was reached
                if i >= max_sites:
                    break

    return i


def download_if_url(path: str, cache: bool = True, desc: str = 'Downloading file') -> str:
    """
    Download the VCF file if it is a URL.

    :param path: The path to the VCF file.
    :param cache: Whether to cache the file.
    :param desc: Description for the progress bar
    :return: The path to the downloaded file or the original path.
    """
    if FileHandler.is_url(path):
        # download the file and return path
        return FileHandler.download_file(path, cache=cache, desc=desc)

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


class FileHandler:
    """
    Base class for file handling.
    """

    #: The logger instance
    _logger = logger.getChild(__qualname__)

    def __init__(self, cache: bool = True, aliases: Dict[str, List[str]] = {}):
        """
        Create a new FileHandler instance.

        :param cache: Whether to cache files that are downloaded from URLs
        :param aliases: The contig aliases.
        """
        #: Whether to cache files that are downloaded from URLs
        self.cache: bool = cache

        #: The contig mappings
        self._alias_mappings, self.aliases = self._expand_aliases(aliases)

    @staticmethod
    def _expand_aliases(alias_dict: Dict[str, List[str]]) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
        """
        Expand the contig aliases.
        """
        # map alias to primary alias
        mappings = {}

        # map primary alias to all aliases
        aliases = {}

        for contig, alias_list in alias_dict.items():
            all_aliases = alias_list + [contig]
            aliases[contig] = all_aliases

            for alias in all_aliases:
                mappings[alias] = contig

        return mappings, aliases

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
        return download_if_url(path, cache=self.cache, desc=f'{self.__class__.__name__}>Downloading file')

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

    @classmethod
    def download_file(cls, url: str, cache: bool = True, desc: str = 'Downloading file') -> str:
        """
        Download a file from a URL.

        :param cache: Whether to cache the file.
        :param url: The URL to download the file from.
        :param desc: Description for the progress bar
        :return: The path to the downloaded file.
        """
        # extract the file extension from the URL
        filename = FileHandler.get_filename(url)

        # create a temporary file path
        path = tempfile.gettempdir() + '/' + FileHandler.hash(url) + '.' + filename

        # check if the file is already cached
        if cache and os.path.exists(path):
            cls._logger.info(f'Using cached file at {path}')
            return path

        cls._logger.info(f'Downloading file from {url}')

        # start the stream
        response = requests.get(url, stream=True)

        # check if the request was successful
        response.raise_for_status()

        # create a temporary file with the original file extension
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            total_size = int(response.headers.get('content-length', 0))
            chunk_size = 8192

            with tqdm(total=total_size,
                      unit='B',
                      unit_scale=True,
                      desc=desc,
                      disable=Settings.disable_pbar) as pbar:

                # write the file to disk
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        tmp.write(chunk)
                        pbar.update(len(chunk))

        # rename the file to the original file extension
        if cache:
            os.rename(tmp.name, path)

            cls._logger.info(f'Cached file at {path}')

            return path

        return tmp.name

    def get_aliases(self, contig: str) -> List[str]:
        """
        Get all aliases for the given contig alias including the primary alias.

        :param contig: The contig.
        :return: The aliases.
        """
        if contig in self._alias_mappings:
            return self.aliases[self._alias_mappings[contig]]

        return [contig]


class FASTAHandler(FileHandler):

    def __init__(self, fasta: str | None, cache: bool = True, aliases: Dict[str, List[str]] = {}):
        """
        Create a new FASTAHandler instance.

        :param fasta: The path to the FASTA file.
        :param cache: Whether to cache files that are downloaded from URLs
        :param aliases: The contig aliases.
        """
        FileHandler.__init__(self, cache=cache, aliases=aliases)

        #: The path to the FASTA file.
        self.fasta: str = fasta

        #: The current contig.
        self._contig: SeqRecord | None = None

    @cached_property
    def _ref(self) -> FastaIterator | None:
        """
        Get the reference reader.

        :return: The reference reader.
        """
        if self.fasta is None:
            return

        return self.load_fasta(self.fasta)

    def load_fasta(self, file: str) -> FastaIterator:
        """
        Load a FASTA file into a dictionary.

        :param file: The path to The FASTA file path, possibly gzipped or a URL
        :return: Iterator over the sequences.
        """
        self._logger.info("Loading FASTA file")

        # download and unzip if necessary
        local_file = self.unzip_if_zipped(self.download_if_url(file))

        return SeqIO.parse(local_file, 'fasta')

    def get_contig(self, aliases, rewind: bool = True, notify: bool = True) -> SeqRecord:
        """
        Get the contig from the FASTA file.

        Note that ``pyfaidx`` would be more efficient here, but there were problems when running it in parallel.

        :param aliases: The contig aliases.
        :param rewind: Whether to allow for rewinding the iterator if the contig is not found.
        :param notify: Whether to notify the user when rewinding the iterator.
        :return: The contig.
        """
        # if the contig is already loaded, we can just return it
        if self._contig is not None and self._contig.id in aliases:
            return self._contig

        # if the contig is not loaded, we can try to load it
        try:
            self._contig = next(self._ref)

            # iterate until we find the contig
            while self._contig.id not in aliases:
                self._contig = next(self._ref)

        except StopIteration:

            # if rewind is ``True``, we can rewind the iterator and try again
            if rewind:
                if notify:
                    self._logger.info("Rewinding FASTA iterator.")

                # renew fasta iterator
                FASTAHandler._rewind(self)

                return self.get_contig(aliases, rewind=False)

            raise LookupError(f'None of the contig aliases {aliases} were found in the FASTA file.')

        return self._contig

    def get_contig_names(self) -> List[str]:
        """
        Get the names of the contigs in the FASTA file.

        :return: The contig names.
        """
        return [contig.id for contig in self._ref]

    def _rewind(self):
        """
        Rewind the fasta iterator.
        """
        if hasattr(self, '_ref'):
            # noinspection all
            del self._ref


class GFFHandler(FileHandler):
    """
    GFF handler.
    """

    def __init__(self, gff: str | None, cache: bool = True, aliases: Dict[str, List[str]] = {}):
        """
        Constructor.

        :param gff: The path to the GFF file.
        :param cache: Whether to cache the file.
        :param aliases: The contig aliases.
        """
        FileHandler.__init__(self, cache=cache, aliases=aliases)

        #: The logger
        self._logger = logger.getChild(self.__class__.__name__)

        #: The GFF file path
        self.gff = gff

    @cached_property
    def _cds(self) -> pd.DataFrame | None:
        """
        The coding sequences.

        :return: Dataframe with coding sequences.
        """
        if self.gff is None:
            return

        return self._load_cds()

    def _load_cds(self) -> pd.DataFrame:
        """
        Load coding sequences from a GFF file.

        :return: The DataFrame.
        """
        self._logger.info(f'Loading GFF file')

        # download and unzip if necessary
        local_file = self.unzip_if_zipped(self.download_if_url(self.gff))

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

    def _count_target_sites(self, remove_overlaps: bool = False, contigs: List[str] = None) -> Dict[str, int]:
        """
        Count the number of target sites in a GFF file.

        :param remove_overlaps: Whether to remove overlapping coding sequences.
        :param contigs: The contigs to consider.
        :return: The number of target sites per chromosome/contig.
        """
        cds = self._add_lengths(
            cds=self._load_cds(),
            remove_overlaps=remove_overlaps,
            contigs=contigs
        )

        # group by 'seqid' and calculate the sum of 'length'
        target_sites = cds.groupby('seqid', observed=False)['length'].sum().to_dict()

        # filter explicitly for contigs if necessary
        # as seqid is a categorical variable, groups were retained even if they were filtered out
        if contigs is not None:
            target_sites = {k: v for k, v in target_sites.items() if k in contigs}

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
    def _add_lengths(cds: pd.DataFrame, remove_overlaps: bool = False, contigs: List[str] = None) -> pd.DataFrame:
        """
        Compute coding sequences lengths.

        :param cds: The coding sequences.
        :param remove_overlaps: Whether to remove overlapping coding sequences.
        :param contigs: The contigs to consider.
        :return: The coding sequences with lengths.
        """
        # filter for contigs if necessary
        if contigs is not None:
            cds = cds[cds['seqid'].isin(contigs)]

        # remove duplicates
        cds = cds.drop_duplicates(subset=['seqid', 'start'])

        # remove overlaps
        if remove_overlaps:
            cds = GFFHandler.remove_overlaps(cds)

        # catch warning when adding a new column to a slice of a DataFrame
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=SettingWithCopyWarning)

            # create a new column for the difference between 'end' and 'start'
            cds['length'] = cds['end'] - cds['start'] + 1

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
            cache: bool = True,
            aliases: Dict[str, List[str]] = {}
    ):
        """
        Create a new VCF instance.

        :param vcf: The path to the VCF file or an iterable of variants, can be gzipped, urls are also supported
        :param info_ancestral: The tag in the INFO field that contains the ancestral allele
        :param max_sites: Maximum number of sites to consider
        :param seed: Seed for the random number generator. Use ``None`` for no seed.
        :param cache: Whether to cache files that are downloaded from URLs
        :param aliases: The contig aliases.
        """
        FileHandler.__init__(self, cache=cache, aliases=aliases)

        #: The path to the VCF file or an iterable of variants
        self.vcf = vcf

        #: The tag in the INFO field that contains the ancestral allele
        self.info_ancestral: str = info_ancestral

        #: Maximum number of sites to consider
        self.max_sites: int = int(max_sites) if not np.isinf(max_sites) else np.inf

        #: Seed for the random number generator
        self.seed: Optional[int] = int(seed) if seed is not None else None

        #: Random generator instance
        self.rng = np.random.default_rng(seed=seed)

    @cached_property
    def _reader(self) -> VCF:
        """
        Get the VCF reader.

        :return: The VCF reader.
        """
        return self.load_vcf()

    def _rewind(self):
        """
        Rewind the VCF iterator.
        """
        if hasattr(self, '_reader'):
            # noinspection all
            del self._reader

    def load_vcf(self) -> VCF:
        """
        Load a VCF file into a dictionary.

        :return: The VCF reader.
        """
        self._logger.info("Loading VCF file")

        return VCF(self.download_if_url(self.vcf))

    @cached_property
    def n_sites(self) -> int:
        """
        Get the number of sites in the VCF.

        :return: Number of sites
        """
        return self.count_sites()

    def count_sites(self) -> int:
        """
        Count the number of sites in the VCF.

        :return: Number of sites
        """
        return count_sites(
            vcf=self.download_if_url(self.vcf),
            max_sites=self.max_sites,
            desc=f'{self.__class__.__name__}>Counting sites'
        )

    def get_pbar(self, desc: str = "Processing sites", total: int | None = 0) -> tqdm:
        """
        Return a progress bar for the number of sites.

        :param desc: Description for the progress bar
        :param total: Total number of items
        :return: tqdm
        """
        return tqdm(
            total=self.n_sites if total == 0 else total,
            disable=Settings.disable_pbar,
            desc=desc
        )


class MultiHandler(VCFHandler, FASTAHandler, GFFHandler):
    """
    Handle VCF, FASTA and GFF files.
    """

    def __init__(
            self,
            vcf: str | Iterable[Variant],
            fasta: str | None = None,
            gff: str | None = None,
            info_ancestral: str = 'AA',
            max_sites: int = np.inf,
            seed: int | None = 0,
            cache: bool = True,
            aliases: Dict[str, List[str]] = {}
    ):
        """
        Create a new MultiHandler instance.

        :param vcf: The path to the VCF file or an iterable of variants, can be gzipped, urls are also supported
        :param fasta: The path to the FASTA file.
        :param gff: The path to the GFF file.
        :param info_ancestral: The tag in the INFO field that contains the ancestral allele
        :param max_sites: Maximum number of sites to consider
        :param seed: Seed for the random number generator. Use ``None`` for no seed.
        :param cache: Whether to cache files that are downloaded from URLs
        :param aliases: The contig aliases.
        """
        # initialize vcf handler
        VCFHandler.__init__(
            self,
            vcf=vcf,
            info_ancestral=info_ancestral,
            max_sites=max_sites,
            seed=seed,
            cache=cache,
            aliases=aliases
        )

        # initialize fasta handler
        FASTAHandler.__init__(
            self,
            fasta=fasta,
            cache=cache,
            aliases=aliases
        )

        # initialize gff handler
        GFFHandler.__init__(
            self,
            gff=gff,
            cache=cache,
            aliases=aliases
        )

    def _require_fasta(self, class_name: str):
        """
        Raise an exception if no FASTA file was provided.

        :param class_name: The name of the class that requires a FASTA file.
        """
        if self.fasta is None:
            raise ValueError(f'{class_name} requires a FASTA file to be specified.')

    def _require_gff(self, class_name: str):
        """
        Raise an exception if no GFF file was provided.

        :param class_name: The name of the class that requires a GFF file.
        """
        if self.gff is None:
            raise ValueError(f'{class_name} requires a GFF file to be specified.')

    def _rewind(self):
        """
        Rewind the fasta and gff handler.
        """
        FASTAHandler._rewind(self)
        VCFHandler._rewind(self)


class NoTypeException(BaseException):
    """
    Exception thrown when no type can be determined.
    """
    pass


class DummyVariant:
    """
    Dummy variant class to emulate a mono-allelic site.
    """

    #: Whether the variant is an SNP
    is_snp = False

    #: Whether the variant is an MNP
    is_mnp = False

    #: Whether the variant is an indel
    is_indel = False

    #: Whether the variant is a deletion
    is_deletion = False

    #: Whether the variant is a structural variant
    is_sv = False

    #: The alternate alleles
    ALT = []

    def __init__(self, ref: str, pos: int, chrom: str):
        """
        Initialize the dummy variant.

        :param ref: The reference allele
        :param pos: The position
        :param chrom: The contig
        """
        #: The reference allele
        self.REF = ref

        #: The position
        self.POS = pos

        #: The contig
        self.CHROM = chrom

        #: Info field
        self.INFO = {}
