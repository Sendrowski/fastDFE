"""
Handlers the reading of VCF, GFF and FASTA files.
"""

__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"
__date__ = "2023-05-29"

import gzip
import hashlib
import logging
import re
import os
import shutil
import tempfile
import warnings
from abc import ABC, abstractmethod
from collections import Counter
from functools import cached_property
from typing import List, Iterable, TextIO, Dict, Optional, Tuple, Union, Sequence, Type, Iterator
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import requests
from Bio import SeqIO
from Bio.SeqIO.FastaIO import FastaIterator
from Bio.SeqRecord import SeqRecord
from pandas.errors import SettingWithCopyWarning
from tqdm import tqdm

from .settings import Settings

#: The DNA bases
bases = ["A", "C", "G", "T"]

# logger
logger = logging.getLogger('fastdfe')


def get_called_bases(genotypes: Sequence[str]) -> np.ndarray:
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


def get_major_base(genotypes: Sequence[str]) -> str | None:
    """
    Get the major base from a list of calls.

    :param genotypes: Array of genotypes in the form of strings.
    :return: Major base.
    """
    # get the called bases
    bases = get_called_bases(genotypes)

    if len(bases) > 0:
        return Counter(bases).most_common()[0][0]


def is_monomorphic_snp(variant: Union['cyvcf2.Variant', 'DummyVariant']) -> bool:
    """
    Whether the given variant is a monomorphic SNP.

    :param variant: The vcf site
    :return: Whether the site is a monomorphic SNP
    """
    return (not (variant.is_snp or variant.is_mnp or variant.is_indel or variant.is_deletion or variant.is_sv)
            and variant.REF in bases)


def count_sites(
        vcf: str | Iterable['cyvcf2.Variant'],
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


class Variant:
    """
    Variant class to emulate a cyvcf2.Variant.
    """

    #: Whether the variant is an SNP
    is_snp: bool

    #: Whether the variant is an MNP
    is_mnp: bool

    #: Whether the variant is an indel
    is_indel: bool

    #: Whether the variant is a deletion
    is_deletion: bool

    #: Whether the variant is a structural variant
    is_sv: bool

    #: The alternate alleles
    ALT: List[str]

    #: The reference allele
    REF: str

    #: The variant position
    POS: int

    #: The contig/chromosome
    CHROM: str

    #: Info field
    INFO: Dict[str, any]

    #: The genotypes
    gt_bases: np.ndarray


class DummyVariant(Variant):
    """
    Dummy variant class to emulate mono-allelic sites sampled from a fasta file.
    """
    #: Whether the variant is an SNP
    is_snp: bool = False

    #: Whether the variant is an MNP
    is_mnp: bool = False

    #: Whether the variant is an indel
    is_indel: bool = False

    #: Whether the variant is a deletion
    is_deletion: bool = False

    #: Whether the variant is a structural variant
    is_sv: bool = False

    #: The alternate alleles
    ALT: List[str] = []

    #: Info field
    INFO: Dict[str, any] = {}

    #: The genotypes
    gt_bases: np.ndarray = np.array([])

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


class ZarrVariant(Variant):
    variant_contig: int

    def __init__(self, ref: str, pos: int, chrom: str,
                 gt_bases: np.ndarray, variant_contig: int = None,
                 is_snp: bool = False, is_mnp: bool = False,
                 info: Dict[str, any] = {}):
        """
        Initialize the Zarr variant.

        :param ref: The reference allele
        :param pos: The position
        :param chrom: The contig
        :param gt_bases: The genotype bases
        :param variant_contig: The contig id (number)
        :param is_snp: Whether the variant is an SNP
        :param is_mnp: Whether the variant is an MNP
        :param info: The INFO field
        """
        #: The reference allele
        self.REF = ref

        #: The position
        self.POS = pos

        #: The contig
        self.CHROM = chrom

        #: The genotypes
        self.gt_bases = gt_bases

        #: The contig id (number)
        self.variant_contig = variant_contig

        self.is_snp = is_snp

        self.is_mnp = is_mnp

        self.INFO = info


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
        os.rename(tmp.name, path)

        if cache:
            cls._logger.info(f'Cached file at {path}')

        return path

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
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=SettingWithCopyWarning)

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


class VariantReader(Iterable, ABC):
    """
    Base class for variant reading.
    """

    @abstractmethod
    def __iter__(self) -> Iterator[Variant]:
        pass

    @abstractmethod
    def add_info_to_header(self, data: dict):
        """
        Add an INFO field, `ID`, `Number`, `Type` and `Description`.

        :param data: The INFO field data.
        """
        pass

    @property
    @abstractmethod
    def samples(self) -> List[str]:
        """
        List of sample names.

        :return: The sample names.
        """
        pass

    @property
    @abstractmethod
    def seqnames(self) -> List[str]:
        """
        List of chromosome/contig names.

        :return: The sequence names.
        """
        pass

    @abstractmethod
    def close(self):
        """
        Close the reader.
        """
        pass


class VariantWriter(ABC):
    """
    Base class for variant writing.
    """

    @abstractmethod
    def write_record(self, variant: Variant):
        """
        Write a variant record.

        :param variant: The variant to write.
        """
        pass

    @abstractmethod
    def close(self):
        """
        Close the writer.
        """
        pass


class VariantHandler(FileHandler, ABC):
    """
    Base class for variant handling.
    """

    @abstractmethod
    def __init__(
            self,
            vcf: str | Iterable[Variant],
            output: str = None,
            info_ancestral: str = 'AA',
            max_sites: int = np.inf,
            seed: int | None = 0,
            cache: bool = True,
            aliases: Dict[str, List[str]] = {}
    ):
        """
        Create a new VCF instance.

        :param vcf: The path to the variant file or an iterable of variants, can be gzipped, urls are also supported
        :param output: The output variant file path.
        :param info_ancestral: The tag in the INFO field that contains the ancestral allele
        :param max_sites: Maximum number of sites to consider
        :param seed: Seed for the random number generator. Use ``None`` for no seed.
        :param cache: Whether to cache files that are downloaded from URLs
        :param aliases: The contig aliases.
        """
        super().__init__(cache=cache, aliases=aliases)

    @property
    @abstractmethod
    def _reader(self) -> VariantReader:
        """
        Get the variant reader.

        :return: The variant reader.
        """
        pass

    @property
    @abstractmethod
    def _writer(self) -> VariantWriter:
        """
        Get the variant writer.

        :return: The variant writer.
        """
        pass

    @property
    @abstractmethod
    def n_sites(self) -> int:
        """
        Get the number of sites in the variant file.

        :return: Number of sites
        """
        pass

    @abstractmethod
    def _rewind(self):
        """
        Rewind the variant iterator.
        """
        pass

    @abstractmethod
    def load_variants(self) -> VariantReader:
        """
        Load a variant file.

        :return: The variant reader.
        """
        pass

    @abstractmethod
    def count_sites(self) -> int:
        """
        Count the number of sites in the variant file.

        :return: Number of sites
        """
        pass

    @abstractmethod
    def get_pbar(self, desc: str = "Processing sites", total: int | None = 0) -> tqdm:
        """
        Return a progress bar for the number of sites.

        :param desc: Description for the progress bar
        :param total: Total number of items
        :return: tqdm
        """
        pass


class ZarrHandler(VariantHandler):
    """
    Zarr handler.
    """
    zarr_suffixes = [".vcz"]

    def __init__(
        self,
        vcf: str | Iterable['ZarrVariant'],
        output: str = None,
        info_ancestral: str = 'AA',
        max_sites: int = np.inf,
        seed: int | None = 0,
        cache: bool = True,
        aliases: Dict[str, List[str]] = {}
    ):
        """
        Create a new Zarr instance.

        :param vcf: The path to the Zarr store.
        :param output: The output store. Defaults to the input store.
        :param info_ancestral: The tag in the INFO field that contains the ancestral allele
        :param max_sites: Maximum number of sites to consider
        :param seed: Seed for the random number generator. Use ``None`` for no seed.
        :param cache: Whether to cache files that are downloaded from URLs
        :param aliases: The contig aliases.
        """
        FileHandler.__init__(self, cache=cache, aliases=aliases)

        #: The path to the VCF file or an iterable of variants
        self.vcf = vcf

        #: The output zarr store.
        self.output: str = self.vcf if output is None else output

        # The name of the ancestral allele Zarr array
        self.info_ancestral: str = info_ancestral

        #: Maximum number of sites to consider
        self.max_sites: int = int(max_sites) if not np.isinf(max_sites) else np.inf

        #: Seed for the random number generator
        self.seed: Optional[int] = int(seed) if seed is not None else None

        #: Random generator instance
        self.rng = np.random.default_rng(seed=seed)


    @cached_property
    def _reader(self) -> 'ZarrReader':
        return self.load_variants()


    @cached_property
    def _writer(self):
        """
        Get the Zarr writer.

        :return: The Zarr writer.
        """
        return ZarrWriter(self.output)


    def _rewind(self):
        """
        Rewind the Zarr iterator.
        """
        if hasattr(self, '_reader'):
            # noinspection all
            del self._reader

    def count_sites(self) -> int:
        """
        Count the number of sites in the Zarr store.

        :return: Number of sites
        """
        return self._reader.count_sites()


    @cached_property
    def n_sites(self) -> int:
        """
        Get the number of sites in the VCF.

        :return: Number of sites
        """
        return self.count_sites()


    def load_variants(self) -> 'ZarrReader':
        """
        Load a Zarr archive a dictionary.

        :return: The Zarr reader.
        """
        self._logger.info("Loading Zarr archive for reading")
        return ZarrReader(self.vcf)


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


    @staticmethod
    def is_zarr(path: str) -> bool:
        """
        Check if the given path is a Zarr file.

        :param path: The path to check.
        :return: ``True`` if the path is a Zarr file, ``False`` otherwise.
        """
        _, sfx = os.path.splitext(path)
        return os.path.isdir(path) and (sfx in ZarrHandler.zarr_suffixes)


class ZarrReader(VariantReader):
    """
    Zarr reader.
    """
    def __init__(
        self,
        zarrstore: str,
        samples: List[str] | None = None,
    ):
        """
        Create a new ZarrReader instance.
        :param zarrstore: The path to the Zarr store.
        """
        try:
            from vcztools import retrieval
        except ImportError:
            raise ImportError(
                "Zarr support in fastdfe requires the optional 'vcztools' package. "
                "Please install vcztools: pip install vcztools"
            )
        self._retrieval = retrieval
        self.zarr = zarrstore
        self._samples = self._parse_samples(samples)
        self.fields = None

    def __enter__(self):
        return self

    def __iter__(self) -> Iterator[ZarrVariant]:
        return self

    def __next__(self) -> ZarrVariant:
        try:
            v = next(self.iter)
        except IndexError:
            raise StopIteration
        except StopIteration:
            raise StopIteration
        alleles = v["variant_allele"]
        phased = v["call_genotype_phased"]
        gt_bases = np.array([
            ["/", "|"][np.int8(ph)].join(alleles[gt]) for ph, gt in zip(phased, v["call_genotype"])
        ])
        is_snp = v["variant_length"] == 1
        is_mnp = v["variant_length"] > 1

        # Set the info fields
        info = dict(
            (re.sub(r"^variant_", "", k), v[k]) for k in v.keys() if k.startswith("variant_")
        )

        return ZarrVariant(
            ref=alleles[0],
            chrom=self.seqnames[v["variant_contig"]],
            pos=v["variant_position"],
            gt_bases=gt_bases,
            variant_contig=v["variant_contig"],
            is_snp=is_snp, is_mnp=is_mnp,
            info=info
        )


    def _variant_iter(self) -> Iterator:
        """
        Get the Zarr variant iterator.
        :return: The Zarr variant iterator.
        """
        return self._retrieval.variant_iter(
            self.zarr, samples=self.samples,
            fields=self.fields
        )

    def _parse_samples(self, samples):
        all_samples = self.root['sample_id'][:]
        return list(self._retrieval.parse_samples(samples, all_samples)[0])


    @property
    def samples(self) -> List[str]:
        """
        List of sample names.

        :return: The sample names.
        """
        return self._samples


    @cached_property
    def iter(self) -> Iterator:
        """
        Get the Zarr variant iterator.

        :return: The Zarr variant iterator.
        """
        return self._variant_iter()


    def close(self):
        pass


    def count_sites(self) -> int:
        """
        Count the number of sites in the Zarr store.
        :return: Number of sites
        """
        return self.root["variant_id"].shape[0]

    def add_info_to_header(self, data: dict):
        pass

    @cached_property
    def root(self):
        """The Zarr root group."""
        return self._init_store()

    def _init_store(self):
        """Initialize the Zarr store."""
        try:
            import zarr
        except ImportError:
            raise ImportError(
                "Zarr support in fastdfe requires the optional 'zarr' package. "
                "Please install fastdfe with the 'zarr' extra: pip install fastdfe[zarr]"
            )
        return zarr.open(self.zarr, mode='r')

    @cached_property
    def seqnames(self):
        """
        List of chromosome/contig names.

        :return: The sequence names.
        """
        return list(self.root["contig_id"][:])


class ZarrWriter(VariantWriter):
    """
    Zarr writer.
    """

    def __init__(self, path: str):
        """
        Create a new ZarrWriter instance.
        :param path: The path to the Zarr store.
        """
        try:
            import zarr
        except ImportError:
            raise ImportError(
                "Zarr support in fastdfe requires the optional 'zarr' package. "
                "Please install fastdfe with the 'zarr' extra: pip install fastdfe[zarr]"
            )
        self.zarr = path
        self.store = zarr.open(self.zarr, mode='a')
        # FIXME: Initialize temporary arrays to hold fields that are
        # to be annotated

    def write_record(self, variant: ZarrVariant):
        # FIXME: write the modified fields to a temporary array and
        # then write these arrays in one go to the correct positions
        # in the Zarr store. Need to keep track of which fields are
        # being annotated.
        print(variant)

    def close(self):
        # FIXME: flush / write the temporary arrays to the Zarr store
        # here.
        pass



class VCFHandler(VariantHandler):
    """
    Base class for VCF handling.
    """

    def __init__(
            self,
            vcf: str | Iterable['cyvcf2.Variant'],
            output: str = None,
            info_ancestral: str = 'AA',
            max_sites: int = np.inf,
            seed: int | None = 0,
            cache: bool = True,
            aliases: Dict[str, List[str]] = {}
    ):
        """
        Create a new VCF instance.

        :param vcf: The path to the VCF file or an iterable of variants, can be gzipped, urls are also supported
        :param output: The output file.
        :param info_ancestral: The tag in the INFO field that contains the ancestral allele
        :param max_sites: Maximum number of sites to consider
        :param seed: Seed for the random number generator. Use ``None`` for no seed.
        :param cache: Whether to cache files that are downloaded from URLs
        :param aliases: The contig aliases.
        """
        FileHandler.__init__(self, cache=cache, aliases=aliases)

        #: The path to the VCF file or an iterable of variants
        self.vcf = vcf

        #: The output file.
        self.output: str = output

        #: The tag in the INFO field that contains the ancestral allele
        self.info_ancestral: str = info_ancestral

        #: Maximum number of sites to consider
        self.max_sites: int = int(max_sites) if not np.isinf(max_sites) else np.inf

        #: Seed for the random number generator
        self.seed: Optional[int] = int(seed) if seed is not None else None

        #: Random generator instance
        self.rng = np.random.default_rng(seed=seed)

    @cached_property
    def _reader(self) -> 'cyvcf2.VCF':
        """
        Get the VCF reader.

        :return: The VCF reader.
        """
        return self.load_variants()

    @cached_property
    def _writer(self) -> 'cyvcf2.VCF':
        """
        Get the VCF writer.

        :return: The VCF writer.
        """
        try:
            from cyvcf2 import Writer
        except ImportError:
            raise ImportError(
                "VCF support in fastdfe requires the optional 'cyvcf2' package. "
                "Please install fastdfe with the 'vcf' extra: pip install fastdfe[vcf]"
            )

        return Writer(self.output, self._reader)

    def _rewind(self):
        """
        Rewind the VCF iterator.
        """
        if hasattr(self, '_reader'):
            # noinspection all
            del self._reader

    def load_variants(self) -> 'cyvcf2.VCF':
        """
        Load a VCF file into a dictionary.

        :return: The VCF reader.
        """
        try:
            from cyvcf2 import VCF
        except ImportError:
            raise ImportError(
                "VCF support in fastdfe requires the optional 'cyvcf2' package. "
                "Please install fastdfe with the 'vcf' extra: pip install fastdfe[vcf]"
            )

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


class AutoVariantHandler:
    """
    VariantHandler wrapper that automatically delegates all behaviour to the
    correct backend, based on the input file.
    """

    def __init__(self, vcf: str, **kwargs):
        self.backend: VariantHandler = self.select_variant_handler(vcf)(vcf, **kwargs)

    @staticmethod
    def select_variant_handler(file: str) -> Type['VariantHandler']:
        """
        Select the appropriate variant handler.

        :return: The variant handler.
        """
        if ZarrHandler.is_zarr(file):
            return ZarrHandler

        return VCFHandler

    def __getattr__(self, name: str):
        """
        Forward attribute/method access to backend when not found on this wrapper.
        """
        try:
            return getattr(self.backend, name)
        except AttributeError:
            raise AttributeError(f"{self.__class__.__name__} has no attribute '{name}'")


class MultiHandler(AutoVariantHandler, FASTAHandler, GFFHandler):
    """
    Handle VCF, FASTA and GFF files.
    """

    def __init__(
            self,
            vcf: str | Iterable[Variant],
            fasta: str | None = None,
            gff: str | None = None,
            output: str = None,
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
        :param output: The output variant file path.
        :param info_ancestral: The tag in the INFO field that contains the ancestral allele
        :param max_sites: Maximum number of sites to consider
        :param seed: Seed for the random number generator. Use ``None`` for no seed.
        :param cache: Whether to cache files that are downloaded from URLs
        :param aliases: The contig aliases.
        """
        # initialize vcf handler
        AutoVariantHandler.__init__(
            self,
            vcf=vcf,
            output=output,
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
        Rewind the fasta and vcf handler.
        """
        FASTAHandler._rewind(self)
        VCFHandler._rewind(self)


class NoTypeException(BaseException):
    """
    Exception thrown when no type can be determined.
    """
    pass
