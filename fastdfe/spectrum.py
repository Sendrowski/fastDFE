"""
SFS utilities.
"""

__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"
__date__ = "2022-07-24"

import logging
from functools import cached_property
from typing import Dict, List, Union

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# get logger
logger = logging.getLogger('fastdfe')


def standard_kingman(n: int) -> 'Spectrum':
    """
    Get standard Kingman SFS.

    :param n: Standard Kingman SFS
    """
    return Spectrum(pad(1 / np.arange(1, n)))


def pad(counts: list | np.ndarray) -> np.ndarray:
    """
    Pad array with monomorphic counts.

    :param counts: SFS counts to pad
    :return: Padded array
    """
    return np.array([0] + list(counts) + [0])


class Spectrum:
    """
    Class for holding and manipulating a site-frequency spectrum.
    """

    def __init__(self, data: list | np.ndarray):
        """
        Initialize spectrum.

        :param data: SFS counts
        """
        self.data = np.array(data, dtype=float)

    @property
    def n(self) -> int:
        """
        The sample size.
        """
        return self.data.shape[0] - 1

    @property
    def n_sites(self) -> float:
        """
        The total number of sites.
        """
        return sum(self.data)

    @property
    def n_div(self) -> float:
        """
        Number of divergence counts.
        """
        return self.data[-1]

    @property
    def has_div(self) -> bool:
        """
        Whether n_div was specified.
        """
        return self.n_div != 0

    @property
    def n_monomorphic(self) -> float:
        """
        Number of monomorphic sites.
        """
        return self.data[0] + self.data[-1]

    @property
    def polymorphic(self) -> np.ndarray:
        """
        Get the polymorphic counts.
        """
        return self.data[1:-1]

    @property
    def n_polymorphic(self) -> np.ndarray:
        """
        Get the polymorphic counts.
        """
        return np.sum(self.polymorphic)

    def to_list(self) -> list:
        """
        Convert to list.

        :return: SFS counts
        """
        return list(self.data)

    def to_numpy(self) -> np.ndarray:
        """
        Convert to array.

        :return: SFS counts
        """
        return self.data

    @cached_property
    def theta(self) -> float:
        """
        Calculate site-wise theta using Watterson's estimator.

        :return: Site-wise theta
        """
        return self.n_polymorphic / np.sum(1 / np.arange(1, self.n)) / self.n_sites

    @staticmethod
    def from_polymorphic(data: list | np.ndarray) -> 'Spectrum':
        """
        Create Spectrum from polymorphic counts only.

        :param data: Polymorphic counts
        """
        return Spectrum([0] + list(data) + [0])

    @staticmethod
    def from_list(data: list | np.ndarray) -> 'Spectrum':
        """
        Create Spectrum from list.
        """
        return Spectrum(data)

    @staticmethod
    def from_polydfe(
            polymorphic: list | np.ndarray,
            n_sites: float,
            n_div: float
    ) -> 'Spectrum':
        """
        Create Spectra from polyDFE specification which treats the number
        of mutational target sites and the divergence counts separately.
        Note that the monomorphic counts are included here, although ignored.

        :param polymorphic: Polymorphic counts
        :param n_sites: Total number of sites
        :param n_div: Number of divergence counts
        """
        # determine number of monomorphic ancestral counts
        n_monomorphic = n_sites - np.sum(list(polymorphic) + [n_div])

        data = [n_monomorphic] + list(polymorphic) + [n_div]

        return Spectrum.from_list(data)

    def __mul__(self, other) -> 'Spectrum':
        """
        Multiply spectrum.

        :param other: Scalar
        """
        return Spectrum.from_list(self.data * other)

    __rmul__ = __mul__

    def __add__(self, other) -> 'Spectrum':
        """
        Add spectrum.

        :param other: Spectrum
        """
        return Spectrum.from_list(self.data * other)

    def __floordiv__(self, other) -> 'Spectrum':
        """
        Divide spectrum.

        :param other: Scalar
        """
        return Spectrum.from_list(self.data // other)

    def __truediv__(self, other) -> 'Spectrum':
        """
        Add spectrum.

        :param other: Scalar
        """
        return Spectrum.from_list(self.data / other)

    def plot(
            self,
            show: bool = True,
            file: str = None,
            title: str = None,
            show_monomorphic: bool = False,
            ax: plt.Axes = None
    ) -> plt.Axes:
        """
        Plot spectrum.

        :param show: Whether to show plot
        :param file: File to save plot to
        :param title: Title of plot
        :param show_monomorphic: Whether to show monomorphic counts
        :param ax: Axes to plot on
        :return: Axes
        """
        # import locally to avoid circular dependencies
        from fastdfe import Visualization

        return Visualization.plot_sfs_comparison(
            spectra=[self],
            file=file,
            show=show,
            title=title,
            show_monomorphic=show_monomorphic,
            ax=ax
        )

    @staticmethod
    def standard_kingman(n: int) -> 'Spectrum':
        """
        Get standard Kingman SFS.

        :param n: sample size
        :return: Standard Kingman SFS
        """
        return standard_kingman(n)


class Spectra:
    """
    Class for holding and manipulating spectra of multiple types.
    """

    def __init__(self, data: Dict[str, list | np.ndarray]):
        """
        Initialize spectra.

        :param data: Dictionary of SFS counts keyed by type
        """
        self.data = pd.DataFrame(data)

    @property
    def n(self) -> int:
        """
        The sample size.
        """
        return self.data.shape[0] - 1

    @property
    def k(self) -> int:
        """
        The number of types.
        """
        return self.data.shape[1]

    @property
    def n_monomorphic(self) -> float:
        """
        The number of monomorphic sites.
        """
        return self.data[0] + self.data[-1]

    @property
    def polymorphic(self) -> np.ndarray:
        """
        The polymorphic counts.
        """
        return self.data[1:-1]

    @property
    def n_polymorphic(self) -> np.ndarray:
        """
        The total number of polymorphic counts.
        """
        return np.sum(self.polymorphic)

    @staticmethod
    def from_list(data: list | np.ndarray, types: List) -> 'Spectra':
        """
        Create from array of spectra.
        Note that data.ndim needs to be 2.
        """
        return Spectra(dict((t, d) for t, d in zip(types, data)))

    @property
    def types(self) -> List[str]:
        """
        The types.
        """
        return self.data.columns.to_list()

    @property
    def n_sites(self) -> pd.Series:
        """
        The number of mutational target sites which is the sum of all SFS entries.
        """
        return self.data.sum()

    @property
    def n_div(self) -> pd.Series:
        """
        The number of divergence counts.
        """
        return self.data.iloc[-1]

    @property
    def has_div(self) -> pd.Series:
        """
        Whether n_div was specified.
        """
        return self.n_div != 0

    def normalize(self) -> 'Spectra':
        """
        Normalize spectra by sum of all entries.
        """
        return self / self.data.sum()

    def to_file(self, file: str):
        """
        Save object to file.

        :param file: File name
        """
        self.data.to_csv(file, index=False)

    def to_spectra(self) -> Dict[str, Spectrum]:
        """
        Convert to dictionary of spectrum objects.
        """
        return dict((t, self[t]) for t in self.types)

    def to_dataframe(self) -> pd.DataFrame:
        """
        Get representation as dataframe.
        """
        return self.data

    def to_numpy(self) -> np.ndarray:
        """
        Convert to numpy array.
        """
        return self.data.to_numpy().T

    def to_list(self) -> list:
        """
        Convert to nested list.
        """
        return list(list(d) for d in self.to_numpy())

    def to_dict(self) -> dict:
        """
        Convert to dictionary.
        """
        # return dictionary of lists
        return dict((k, list(v.values())) for k, v in self.data.to_dict().items())

    def __mul__(self, other) -> 'Spectra':
        """
        Multiply Spectra.

        :param other: Scalar
        """
        return Spectra.from_dataframe(self.data * other)

    __rmul__ = __mul__

    def __floordiv__(self, other) -> 'Spectra':
        """
        Divide Spectra.

        :param other: Scalar
        """
        return Spectra.from_dataframe(self.data // other)

    def __truediv__(self, other) -> 'Spectra':
        """
        Divide Spectra.

        :param other: Scalar
        """
        return Spectra.from_dataframe(self.data / other)

    def __len__(self) -> int:
        """
        Get number of spectra.
        """
        return self.k

    def __add__(self, other: 'Spectra') -> 'Spectra':
        """
        Merge types of two spectra objects by adding up their counts entry-wise.
        """
        return Spectra.from_dataframe(self.data.add(other.data, fill_value=0))

    def __getitem__(self, keys) -> Union['Spectrum', 'Spectra']:
        """
        Get item.

        :param keys: string or list of strings
        :return: Key or list of keys
        """
        # whether the input in an array
        is_array = isinstance(keys, (np.ndarray, list, tuple))

        # use regex to select columns
        subset = self.data.filter(regex=('|'.join(keys) if is_array else keys))

        # return spectrum object if only one column is left
        # and if not multiple keys were supplied
        if subset.shape[1] == 1 and not is_array:
            return Spectrum.from_list(list(subset.iloc[:, 0]))

        # wrap subset dataframe in spectra object
        return Spectra.from_dataframe(subset)

    def __setitem__(self, key: str, s: Spectrum):
        """
        Save new spectrum as type.

        :param key: Type
        :param s: Spectrum
        """
        self.data[key] = s.to_list()

    def __iter__(self):
        """
        Get iterator.
        """
        self.data.__iter__()

    def copy(self) -> 'Spectra':
        """
        Copy object.
        """
        return Spectra.from_dataframe(self.data.copy())

    def to_multi_index(self) -> 'Spectra':
        """
        Convert to Spectra object with multi-indexed columns.
        """
        other = self.copy()
        columns = [tuple(col.split('.')) for col in other.data.columns]
        other.data.columns = pd.MultiIndex.from_tuples(columns)

        return other

    def to_single_index(self) -> 'Spectra':
        """
        Convert to Spectra object with single-indexed columns (using dot notation).
        """
        other = self.copy()

        if other.data.columns.nlevels > 1:
            columns = other.data.columns.map('.'.join)
            other.data.columns = columns

        return other

    def get_empty(self) -> 'Spectra':
        """
        Get a Spectra object with zero counts but having the same shape and types as self.
        """
        return Spectra.from_dataframe(pd.DataFrame(0, index=self.data.index, columns=self.data.columns))

    def merge_groups(self, level: List[int] | int = 0) -> 'Spectra':
        """
        Group over given levels and sum up spectra so the spectra
        are summed over the levels that were not specified.
        """
        return Spectra.from_dataframe(self.to_multi_index().data.groupby(axis=1, level=level).sum()).to_single_index()

    @property
    def all(self) -> 'Spectrum':
        """
        The 'all' type equals the sum of all spectra.
        """
        return Spectrum.from_list(self.data.sum(axis=1).to_list())

    def combine(self, s: 'Spectra') -> 'Spectra':
        """
        Merge types of two Spectra objects.

        :param s: Other Spectra object
        """
        return Spectra(self.to_dict() | s.to_dict())

    @staticmethod
    def from_dict(data: dict) -> 'Spectra':
        """
        Load from dictionary.

        :param data: Dictionary of lists indexed by types
        """
        lists = [list(v.values() if isinstance(v, dict) else v) for v in data.values()]

        return Spectra.from_list(lists, types=list(data.keys()))

    @staticmethod
    def from_dataframe(data: pd.DataFrame) -> 'Spectra':
        """
        Load Spectra object from dataframe.

        :param data: Dataframe
        """
        return Spectra.from_dict(data.to_dict())

    @staticmethod
    def from_file(file: str) -> 'Spectra':
        """
        Save object to file.

        :param file: File name
        """
        return Spectra.from_dataframe(pd.read_csv(file))

    @staticmethod
    def from_spectra(spectra: Dict[str, Spectrum]) -> 'Spectra':
        """
        Create from dict of spectrum objects indexed by types.
        """
        return Spectra.from_list([sfs.to_list() for sfs in spectra.values()], types=list(spectra.keys()))

    @staticmethod
    def from_spectrum(sfs: Spectrum) -> 'Spectra':
        """
        Create from single spectrum object. The type of the spectrum is set to 'all'.

        :param sfs: Spectrum
        """
        return Spectra.from_spectra(dict(all=sfs))

    def plot(
            self,
            show: bool = True,
            file: str = None,
            title: str = None,
            use_subplots: bool = False,
            show_monomorphic: bool = False,
            ax: plt.Axes = None
    ) -> plt.Axes:
        """
        Visualize spectra.

        :param show: Whether to show the plot
        :param file: File name to save the plot to
        :param title: Plot title
        :param use_subplots: Whether to use subplots
        :param show_monomorphic: Whether to show monomorphic sites
        :param ax: Axes to plot on
        :return: Axes
        """
        # import locally to avoid circular dependencies
        from fastdfe import Visualization

        return Visualization.plot_sfs_comparison(
            spectra=list(self.to_spectra().values()),
            labels=self.types,
            file=file,
            show=show,
            title=title,
            use_subplots=use_subplots,
            show_monomorphic=show_monomorphic,
            ax=ax
        )

    def remove_empty(self) -> 'Spectra':
        """
        Remove types whose spectra have no counts.
        """
        return Spectra.from_dataframe(self.data.loc[:, self.data.any()])

    def remove_zero_entries(self) -> 'Spectra':
        """
        Remove types whose spectra have some zero entries.
        Note that we ignore zero counts in the last entry i.e. fixed derived alleles.
        """
        return Spectra.from_dataframe(self.data.loc[:, self.data[:-1].all()])

    def rename(self, names: List[str]) -> 'Spectra':
        """
        Rename types.

        :return: New names
        """
        other = self.copy()
        other.data.columns = names

        return other

    def prefix(self, prefix: str) -> 'Spectra':
        """
        Prefix types, i.e. 'type' -> 'prefix.type' for all types.

        :return: Prefix
        """
        return self.rename([prefix + '.' + col for col in self.types])

    def print(self):
        """
        Print spectra.
        """
        print(self.data.T)


def parse_polydfe_sfs_config(file: str) -> Spectra:
    """
    Parse frequency spectra and mutational target site from
    polyDFE configuration file.

    :return: File name
    """
    df = pd.read_csv(file, header=None, comment='#')

    # parse number of spectra and sample size
    n_neut, n_sel, n = np.array(df.iloc[0][0].split()).astype(int)

    # issue notice about number of spectra and sample size
    logger.info(f'Parsing {n_neut} neutral and {n_sel} selected SFS with '
                f'a sample size of {n}.')

    # issue notice that fastDFE does not support variable mutation rates
    if n_neut > 1 or n_sel > 1:
        logger.info('Note that fastDFE does not model variable mutation rates '
                    'as this did not turn out to change the inference result. '
                    'The parsed spectra are thus merged together.')

    def to_spectrum(data: np.array) -> Spectrum:
        """
        Parse spectrum and number of mutational target sites.
        We ignore the number of mutational target sites for divergence counts
        but include the divergence counts for completeness of the SFS.

        :return: Data
        """
        # iterate over spectra and merge them as we do not
        # support variable mutation rates
        data_merged = data.sum(axis=0)

        # polymorphic counts
        polymorphic = list(data_merged[:n - 1])

        # parse number mutational target sites for ingroup
        n_sites = float(data_merged[n - 1])

        # parse optional divergence counts
        n_div = float(data_merged[n]) if n < data_merged.shape[0] else 0

        return Spectrum.from_polydfe(polymorphic, n_sites=n_sites, n_div=n_div)

    # iterate over spectra and merge them as we do not
    # support variable mutation rates
    data_neut = np.array([df.iloc[i][0].split() for i in range(1, n_neut + 1)], dtype=float)
    sfs_neut = to_spectrum(data_neut)

    # iterate over spectra and merge them as we do not
    # support variable mutation rates
    data_sel = np.array([df.iloc[i][0].split() for i in range(n_neut + 1, n_neut + n_sel + 1)], dtype=float)
    sfs_sel = to_spectrum(data_sel)

    return Spectra.from_spectra(dict(
        sfs_neut=sfs_neut,
        sfs_sel=sfs_sel
    ))
