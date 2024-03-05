"""
SFS utilities.
"""

__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"
__date__ = "2022-07-24"

import logging
from functools import cached_property
from typing import Dict, List, Union, Iterable, Any, Literal

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import hypergeom

from .io_handlers import download_if_url
from .visualization import Visualization

# get logger
logger = logging.getLogger('fastdfe')


def standard_kingman(n: int) -> 'Spectrum':
    """
    Get standard Kingman SFS for theta = 1.

    :param n: Standard Kingman SFS
    :return: Spectrum
    """
    return Spectrum(pad(1 / np.arange(1, int(n))))


def pad(counts: list | np.ndarray) -> np.ndarray:
    """
    Pad array with monomorphic counts.

    :param counts: SFS counts to pad
    :return: Padded array
    """
    return np.array([0] + list(counts) + [0])


class Spectrum(Iterable):
    """
    Class for holding and manipulating a site-frequency spectrum.
    """

    def __init__(self, data: list | np.ndarray):
        """
        Initialize spectrum.

        :param data: SFS counts
        """
        self.data: np.ndarray = np.array(data, dtype=float)

    @property
    def n(self) -> int:
        """
        The sample size.

        :return: Sample size
        """
        return self.data.shape[0] - 1

    @property
    def n_sites(self) -> float:
        """
        The total number of sites.

        :return: Total number of sites
        """
        return sum(self.data)

    @property
    def n_div(self) -> float:
        """
        Number of divergence counts.

        :return: Number of divergence counts
        """
        return self.data[-1]

    @property
    def has_div(self) -> bool:
        """
        Whether n_div was specified.

        :return: Whether n_div was specified
        """
        return self.n_div != 0

    @property
    def n_monomorphic(self) -> float:
        """
        Number of monomorphic sites.

        :return: Number of monomorphic sites
        """
        return self.data[0] + self.data[-1]

    @property
    def polymorphic(self) -> np.ndarray:
        """
        Get the polymorphic counts.

        :return: Polymorphic counts
        """
        return self.data[1:-1]

    @property
    def n_polymorphic(self) -> np.ndarray:
        """
        Get the polymorphic counts.

        :return: Polymorphic counts
        """
        return np.sum(self.polymorphic)

    def to_list(self) -> list:
        """
        Convert to list.

        :return: SFS counts
        """
        return list(self.data)

    def to_spectra(self) -> 'Spectra':
        """
        Convert to Spectra object.

        :return: Spectra object
        """
        return Spectra.from_spectrum(self)

    def to_file(self, file: str):
        """
        Save object to file.

        :param file: File name
        """
        self.to_spectra().to_file(file)

    @staticmethod
    def from_file(file: str) -> 'Spectrum':
        """
        Load object from file.

        :param file: File name
        :return: Spectrum object
        """
        return Spectra.from_file(file).to_spectrum()

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

    def fold(self) -> 'Spectrum':
        """
        Fold the site-frequency spectrum.

        :return: Folded spectrum
        """
        mid = (self.n + 1) // 2
        data = self.data.copy()

        data[:mid] += data[-mid:][::-1]
        data[-mid:] = 0

        return Spectrum(data)

    def subsample(
            self,
            n: int,
            mode: Literal['random', 'probabilistic'] = 'probabilistic',
            seed: int | None = None
    ) -> 'Spectrum':
        """
        Subsample spectrum to a given sample size.

        .. warning::
            If using the 'random' mode, The SFS counts are cast to integers before subsampling so this will
            only provide sensible results if the SFS counts are integers or if they are large enough to be
            approximated well by integers. The 'probabilistic' mode does not have this limitation.

        :param n: Sample size
        :param mode: Subsampling mode. Either 'random' or 'probabilistic'.
        :param seed: Seed for random number generator. Only for 'random' mode.
        :return: Subsampled spectrum
        """
        if n >= self.n:
            raise ValueError(f'Subsampled sample size {n} must be smaller than original sample size {self.n}.')

        if mode not in ['random', 'probabilistic']:
            raise ValueError(f'Unknown subsampling mode {mode}.')

        subsample = np.zeros(n + 1, dtype=float)

        if mode == 'random':
            # add monomorphic counts
            subsample[0] = self.data[0]
            subsample[-1] = self.data[-1]

            # iterate over spectrum and subsample hypergeometrically
            for i, m in enumerate(self.polymorphic.astype(int)):
                # get subsampled counts
                samples = hypergeom.rvs(M=self.n, n=i + 1, N=n, size=m, random_state=seed)

                # add subsampled counts
                subsample += np.histogram(samples, bins=np.arange(n + 2))[0]
        else:
            for i, m in enumerate(self.data):
                probs = hypergeom.pmf(k=range(n + 1), M=self.n, n=i, N=n)

                # add subsampled counts
                subsample += m * probs

        return Spectrum(subsample)

    def resample(self, seed: int = None) -> 'Spectrum':
        """
        Resample SFS assuming independent Poisson counts.

        :param seed: Seed for random number generator.
        :return: Resampled spectrum.
        """
        return Spectrum.from_polydfe(
            # resample polymorphic sites only
            polymorphic=np.random.default_rng(seed=seed).poisson(lam=self.polymorphic),
            n_sites=self.n_sites,
            n_div=self.n_div
        )

    def is_folded(self) -> bool:
        """
        Check if the site-frequency spectrum is folded.

        :return: True if folded, False otherwise
        """
        mid = (self.n + 1) // 2

        return np.all(self.data[-mid:] == 0)

    def normalize(self) -> 'Spectrum':
        """
        Normalize SFS so that all non-monomorphic counts add up to 1.

        :return: Normalized spectrum
        """
        # copy array
        data = self.data.copy()

        # normalize counts
        data[1:-1] /= data[1:-1].sum()

        return Spectrum(data)

    def copy(self) -> 'Spectrum':
        """
        Copy the spectrum.

        :return: Copy of the spectrum
        """
        return Spectrum(self.data.copy())

    @staticmethod
    def from_polymorphic(data: list | np.ndarray) -> 'Spectrum':
        """
        Create Spectrum from polymorphic counts only.

        :param data: Polymorphic counts
        :return: Spectrum
        """
        return Spectrum([0] + list(data) + [0])

    @staticmethod
    def from_list(data: list | np.ndarray) -> 'Spectrum':
        """
        Create Spectrum from list.

        :param data: SFS counts
        :return: Spectrum
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

        :param polymorphic: Polymorphic counts
        :param n_sites: Total number of sites
        :param n_div: Number of divergence counts
        :return: Spectrum
        """
        # determine number of monomorphic ancestral counts
        n_monomorphic = n_sites - np.sum(list(polymorphic) + [n_div])

        data = [n_monomorphic] + list(polymorphic) + [n_div]

        return Spectrum(data)

    @staticmethod
    def _array_or_scalar(data: Iterable | float) -> np.ndarray | float:
        """
        Convert to array if iterable or return scalar otherwise.

        :param data: Iterable or scalar.
        :return: Array or scalar
        """
        if isinstance(data, Iterable):
            return np.array(list(data))

        return data

    def __mul__(self, other: Iterable | float) -> 'Spectrum':
        """
        Multiply spectrum.

        :param other: Iterable or scalar
        :return: Spectrum
        """
        return Spectrum(self.data * self._array_or_scalar(other))

    __rmul__ = __mul__

    def __add__(self, other: Iterable | float) -> 'Spectrum':
        """
        Add spectrum.

        :param other: Iterable or scalar
        :return: Spectrum
        """
        return Spectrum(self.data + self._array_or_scalar(other))

    def __sub__(self, other: Iterable | float) -> 'Spectrum':
        """
        Subtract spectrum.

        :param other: Iterable or scalar
        :return: Spectrum
        """
        return Spectrum(self.data - self._array_or_scalar(other))

    def __pow__(self, other: Iterable | float) -> 'Spectrum':
        """
        Power operator.

        :param other: Iterable or scalar
        :return: Spectrum
        """
        return Spectrum(self.data ** self._array_or_scalar(other))

    def __floordiv__(self, other: Iterable | float) -> 'Spectrum':
        """
        Divide spectrum.

        :param other: Iterable or scalar
        :return: Spectrum
        """
        return Spectrum(self.data // self._array_or_scalar(other))

    def __truediv__(self, other: Iterable | float) -> 'Spectrum':
        """
        Add spectrum.

        :param other: Iterable or scalar
        :return: Spectrum
        """
        return Spectrum(self.data / self._array_or_scalar(other))

    def __iter__(self):
        """
        Get iterator.

        :return: Iterator
        """
        return self.data.__iter__()

    def plot(
            self,
            show: bool = True,
            file: str = None,
            title: str = None,
            log_scale: bool = False,
            show_monomorphic: bool = False,
            kwargs_legend: dict = dict(prop=dict(size=8)),
            ax: plt.Axes = None
    ) -> plt.Axes:
        """
        Plot spectrum.

        :param show: Whether to show plot.
        :param file: File to save plot to.
        :param title: Title of plot.
        :param log_scale: Whether to use log scale on y-axis.
        :param show_monomorphic: Whether to show monomorphic counts.
        :param kwargs_legend: Keyword arguments passed to :meth:`plt.legend`. Only for Python visualization backend.
        :param ax: Axes to plot on. Only for Python visualization backend.
        :return: Axes
        """
        return Visualization.plot_spectra(
            spectra=[self.to_list()],
            file=file,
            show=show,
            title=title,
            log_scale=log_scale,
            show_monomorphic=show_monomorphic,
            ax=ax,
            kwargs_legend=kwargs_legend
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
    Class for holding and manipulating site-frequency spectra of multiple types.
    """

    def __init__(self, data: Dict[str, list | np.ndarray]):
        """
        Initialize spectra.

        :param data: Dictionary of SFS counts keyed by type
        """
        self.data: pd.DataFrame = pd.DataFrame(data)

    @property
    def n(self) -> int:
        """
        The sample size.

        :return: Sample size
        """
        return self.data.shape[0] - 1

    @property
    def k(self) -> int:
        """
        The number of types.

        :return: Number of types
        """
        return self.data.shape[1]

    @property
    def n_monomorphic(self) -> pd.Series:
        """
        The number of monomorphic sites.

        :return: Number of monomorphic sites
        """
        return self.data.iloc[0] + self.data.iloc[-1]

    @property
    def polymorphic(self) -> np.ndarray:
        """
        The polymorphic counts.

        :return: Polymorphic counts
        """
        return self.data[1:-1]

    @property
    def n_polymorphic(self) -> np.ndarray:
        """
        The total number of polymorphic counts.

        :return: Total number of polymorphic counts for each type
        """
        return self.polymorphic.sum()

    @staticmethod
    def from_list(data: list | np.ndarray, types: List) -> 'Spectra':
        """
        Create from array of spectra.
        Note that data.ndim needs to be 2.

        :param data: Array of spectra
        :param types: Types
        :return: Spectra
        """
        return Spectra(dict((t, d) for t, d in zip(types, data)))

    @property
    def types(self) -> List[str]:
        """
        The types.

        :return: Types
        """
        return self.data.columns.to_list()

    @property
    def n_sites(self) -> pd.Series:
        """
        The number of mutational target sites which is the sum of all SFS entries.

        :return: Number of mutational target sites for each type
        """
        return self.data.sum()

    @property
    def n_div(self) -> pd.Series:
        """
        The number of divergence counts.

        :return: Number of divergence counts for each type
        """
        return self.data.iloc[-1]

    @property
    def has_div(self) -> pd.Series:
        """
        Whether n_div was specified.

        :return: Whether n_div was specified for each type
        """
        # noinspection PyTypeChecker
        return self.n_div != 0

    def normalize(self) -> 'Spectra':
        """
        Normalize spectra by sum of all entries.

        :return: Normalized spectra
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

        :return: Dictionary of spectrum objects
        """
        return dict((t, self.select(t, use_regex=False)) for t in self.types)

    def to_dataframe(self) -> pd.DataFrame:
        """
        Get representation as dataframe.

        :return: Dataframe
        """
        return self.data

    def to_numpy(self) -> np.ndarray:
        """
        Convert to numpy array.

        :return: Numpy array
        """
        return self.data.to_numpy().T

    def to_list(self) -> list:
        """
        Convert to nested list.

        :return: Nested list
        """
        return list(list(d) for d in self.to_numpy())

    def to_dict(self) -> dict:
        """
        Convert to dictionary.

        :return: Dictionary of lists
        """
        # return dictionary of lists
        return dict((k, list(v.values())) for k, v in self.data.to_dict().items())

    def __mul__(self, other: Any) -> 'Spectra':
        """
        Multiply Spectra.

        :param other: Scalar
        :return: Spectra
        """
        return Spectra.from_dataframe(self.data * other)

    __rmul__ = __mul__

    def __floordiv__(self, other: Any) -> 'Spectra':
        """
        Divide Spectra.

        :param other: Scalar
        :return: Spectra
        """
        return Spectra.from_dataframe(self.data // other)

    def __truediv__(self, other: Any) -> 'Spectra':
        """
        Divide Spectra.

        :param other: Scalar
        :return: Spectra
        """
        return Spectra.from_dataframe(self.data / other)

    def __len__(self) -> int:
        """
        Get number of spectra.

        :return: Number of spectra
        """
        return self.k

    def __add__(self, other: 'Spectra') -> 'Spectra':
        """
        Merge types of two spectra objects by adding up their counts entry-wise.

        :param other: Spectra object
        :return: Spectra with merged types
        """
        return Spectra.from_dataframe(self.data.add(other.data, fill_value=0))

    def __getitem__(
            self,
            keys: str | List[str] | np.ndarray | tuple,
            use_regex: bool = True
    ) -> Union['Spectrum', 'Spectra']:
        """
        Get item.

        :param keys: String or list of strings, possibly regex to match type names
        :param use_regex: Whether to use regex to match type names
        :return: Spectrum or Spectra object depending on the number of matches
        """
        # whether the input in an array
        is_array = isinstance(keys, (np.ndarray, list, tuple))

        if use_regex:
            # subset dataframe using column names using regex
            subset = self.data.loc[:, self.data.columns.str.fullmatch('|'.join(keys) if is_array else keys)]
        else:
            # subset dataframe using column names
            subset = self.data.loc[:, keys]

        # return spectrum object if we have a series
        if isinstance(subset, pd.Series):
            return Spectrum(list(subset))

        # return spectrum object if only one column is left
        # and if not multiple keys were supplied
        if subset.shape[1] == 1 and not is_array:
            return Spectrum(list(subset.iloc[:, 0]))

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

        :return: Iterator
        """
        return self.data.__iter__()

    def select(
            self,
            keys: str | List[str] | np.ndarray | tuple,
            use_regex: bool = True
    ) -> 'Spectra':
        """
        Select types. Alias for __getitem__.

        :param keys: String or list of strings, possibly regex to match type names
        :param use_regex: Whether to use regex to match type names
        :return: Spectrum or Spectra depending on the number of matches
        """
        return self.__getitem__(keys, use_regex=use_regex)

    def copy(self) -> 'Spectra':
        """
        Copy object.

        :return: Copy of object
        """
        return Spectra.from_dataframe(self.data.copy())

    def _to_multi_index(self) -> 'Spectra':
        """
        Convert to Spectra object with multi-indexed columns.

        :return: Spectra object with multi-indexed columns
        """
        other = self.copy()
        columns = [tuple(col.split('.')) for col in other.data.columns]
        other.data.columns = pd.MultiIndex.from_tuples(columns)

        return other

    def _to_single_index(self) -> 'Spectra':
        """
        Convert to Spectra object with single-indexed columns (using dot notation).

        :return: Spectra object with single-indexed columns
        """
        other = self.copy()

        if other.data.columns.nlevels > 1:
            columns = other.data.columns.map('.'.join)
            other.data.columns = columns

        return other

    def get_empty(self) -> 'Spectra':
        """
        Get a Spectra object with zero counts but having the same shape and types as self.

        :return: Spectra object with zero counts
        """
        return Spectra.from_dataframe(pd.DataFrame(0, index=self.data.index, columns=self.data.columns))

    def merge_groups(self, level: List[int] | int = 0) -> 'Spectra':
        """
        Group over given levels and sum up spectra so the spectra
        are summed over the levels that were not specified.

        :param level: Level(s) to group over
        :return: Spectra object with merged groups
        """
        # cast to int
        level = [int(l) for l in level] if isinstance(level, Iterable) else int(level)

        return Spectra.from_dataframe(self._to_multi_index().data.T.groupby(level=level).sum().T)._to_single_index()

    def has_dots(self) -> bool:
        """
        Check whether column names contain dots.

        :return: True if column names contain dots, False otherwise
        """
        return any('.' in col for col in self.data.columns)

    def replace_dots(self, replacement: str = '_') -> 'Spectra':
        """
        Replace dots in column names with a given string.

        :param replacement: Replacement string
        :return: Spectra object with replaced dots
        """
        other = self.copy()
        other.data.columns = other.data.columns.str.replace('.', replacement)

        return other

    @property
    def all(self) -> 'Spectrum':
        """
        The 'all' type equals the sum of all spectra.

        :return: Spectrum object
        """
        return Spectrum(self.data.sum(axis=1).to_list())

    def combine(self, s: 'Spectra') -> 'Spectra':
        """
        Merge types of two Spectra objects.

        :param s: Other Spectra object
        :return: Merged Spectra object
        """
        return Spectra(self.to_dict() | s.to_dict())

    @staticmethod
    def from_dict(data: dict) -> 'Spectra':
        """
        Load from nested dictionary first indexed by types and then by samples.

        :param data: Dictionary of lists indexed by types
        :return: Spectra object
        """
        lists = [list(v.values() if isinstance(v, dict) else v) for v in data.values()]

        return Spectra.from_list(lists, types=list(data.keys()))

    @staticmethod
    def from_dataframe(data: pd.DataFrame) -> 'Spectra':
        """
        Load Spectra object from dataframe.

        :param data: Dataframe
        :return: Spectra object
        """
        return Spectra.from_dict(data.to_dict())

    @classmethod
    def from_file(cls, file: str) -> 'Spectra':
        """
        Save object to file.

        :param file: Path to file, possibly URL
        :return: Spectra object
        """
        return Spectra.from_dataframe(pd.read_csv(download_if_url(
            file,
            desc=f'{cls.__name__}>Downloading file'))
        )

    @staticmethod
    def from_spectra(spectra: Dict[str, Spectrum]) -> 'Spectra':
        """
        Create from dict of spectrum objects indexed by type.

        :param spectra: Dictionary of spectrum objects indexed by type
        :return: Spectra object
        """
        return Spectra.from_list([sfs.to_list() for sfs in spectra.values()], types=list(spectra.keys()))

    @staticmethod
    def from_spectrum(sfs: Spectrum) -> 'Spectra':
        """
        Create from single spectrum object. The type of the spectrum is set to 'all'.

        :param sfs: Spectrum
        :return: Spectra object
        """
        return Spectra.from_spectra(dict(all=sfs))

    def to_spectrum(self) -> Spectrum:
        """
        Convert to Spectrum object by summing over all types.

        :return: Spectrum object
        """
        return self.all

    def plot(
            self,
            show: bool = True,
            file: str = None,
            title: str = None,
            log_scale: bool = False,
            use_subplots: bool = False,
            show_monomorphic: bool = False,
            kwargs_legend: dict = dict(prop=dict(size=8)),
            ax: plt.Axes = None
    ) -> plt.Axes:
        """
        Visualize spectra.

        :param show: Whether to show the plot.
        :param file: File name to save the plot to.
        :param title: Plot title.
        :param log_scale: Whether to use log scale on y-axis.
        :param use_subplots: Whether to use subplots. Only for Python visualization backend.
        :param show_monomorphic: Whether to show monomorphic sites.
        :param kwargs_legend: Keyword arguments passed to :meth:`plt.legend`. Only for Python visualization backend.
        :param ax: Axes to plot on. Only for Python visualization backend and if ``use_subplots`` is ``False``.
        :return: Axes
        """
        return Visualization.plot_spectra(
            spectra=list(list(v) for v in self.to_spectra().values()),
            labels=self.types,
            file=file,
            show=show,
            title=title,
            log_scale=log_scale,
            use_subplots=use_subplots,
            show_monomorphic=show_monomorphic,
            kwargs_legend=kwargs_legend,
            ax=ax
        )

    def drop_empty(self) -> 'Spectra':
        """
        Remove types whose spectra have no counts.

        :return: Spectra with non-empty types
        """
        return Spectra.from_dataframe(self.data.loc[:, self.data.any()])

    def drop_zero_entries(self) -> 'Spectra':
        """
        Remove types whose spectra have some zero entries.
        Note that we ignore zero counts in the last entry i.e. fixed derived alleles.

        :return: Spectra with non-zero entries
        """
        return Spectra.from_dataframe(self.data.loc[:, self.data[:-1].all()])

    def drop_sparse(self, n_polymorphic: int) -> 'Spectra':
        """
        Remove types whose spectra have fewer than equal ``n_polymorphic`` polymorphic sites.

        :return: Spectra
        """
        return Spectra.from_dataframe(self.data.loc[:, self.data[1:-1].sum() > int(n_polymorphic)])

    def rename(self, names: List[str]) -> 'Spectra':
        """
        Rename types.

        :param names: New names
        :return: Spectra with renamed types
        """
        other = self.copy()
        other.data.columns = names

        return other

    def prefix(self, prefix: str) -> 'Spectra':
        """
        Prefix types, i.e. 'type' -> 'prefix.type' for all types.

        :param prefix: Prefix
        :return: Spectra with prefixed types
        """
        return self.rename([prefix + '.' + col for col in self.types])

    def reorder_levels(self, levels: List[int]) -> 'Spectra':
        """
        Reorder levels.

        :param levels: New order of levels
        :return: Spectra with reordered levels
        """
        s = self._to_multi_index()
        s.data.columns = s.data.columns.reorder_levels(levels)
        s = s._to_single_index()

        return s

    def print(self):
        """
        Print spectra.
        """
        print(self.data.T)

    def fold(self):
        """
        Fold spectra.

        :return: Folded spectra
        """
        return Spectra.from_spectra({t: s.fold() for t, s in self.to_spectra().items()})

    def subsample(
            self,
            n: int,
            mode: Literal['random', 'probabilistic'] = 'probabilistic'
    ) -> 'Spectra':
        """
        Subsample spectra to a given sample size.

        .. warning::
            If using the 'random' mode, The SFS counts are cast to integers before subsampling so this will
            only provide sensible results if the SFS counts are integers or if they are large enough to be
            approximated by integers. The 'probabilistic' mode does not have this limitation.

        :param n: Sample size
        :param mode: Subsampling mode. Either 'random' or 'probabilistic'.
        :return: Subsampled spectra
        """
        return Spectra.from_spectra({t: s.subsample(n, mode) for t, s in self.to_spectra().items()})

    def is_folded(self) -> Dict[str, bool]:
        """
        Check whether spectra are folded.

        :return: Dictionary of types and whether they are folded
        """
        return {t: s.is_folded() for t, s in self.to_spectra().items()}

    def sort_types(self) -> 'Spectra':
        """
        Sort types alphabetically.

        :return: Sorted spectra object
        """
        return Spectra.from_dataframe(self.data.sort_index(axis=1))


def parse_polydfe_sfs_config(file: str) -> Spectra:
    """
    Parse frequency spectra and mutational target site from
    polyDFE configuration file.

    :param file: File name
    :return: Spectra object
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

        :param data: Spectrum data
        :return: Spectrum object
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
