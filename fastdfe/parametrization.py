"""
DFE parametrizations.
"""

__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"
__date__ = "2023-02-26"

import logging
from abc import abstractmethod, ABC
from functools import wraps
from typing import Callable, List, Union, Dict, Tuple, Literal

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import gamma, expon

# get logger
logger = logging.getLogger('fastdfe')


def _from_string(model: Union['Parametrization', str]) -> 'Parametrization':
    """
    Return Parametrization from class name string.

    :param model: Class name string or parametrization object
    :return: Parametrization object
    """
    if isinstance(model, Parametrization):
        return model

    if isinstance(model, str):
        return globals()[model]()

    raise ValueError(f'Unknown parametrization: {model}')


def _to_string(model: Union['Parametrization', str]) -> str:
    """
    Return class name string from Parametrization.

    :param model: Class name string or Parametrization object
    :return: Class name string
    """
    if isinstance(model, Parametrization):
        return model.__class__.__name__

    return model


class Parametrization(ABC):
    """
    Base class for DFE parametrizations.

    Note that :func:`get_pdf` is not required to be implemented, provided that the
    linearized mode of fastDFE is used (which is highly recommended).
    """

    #: Default initial parameters
    x0: Dict[str, float] = {}

    #: Default parameter bounds
    bounds: Dict[str, Tuple[float, float]] = {}

    #: Scales over which to optimize the parameters, either 'log' or 'lin'
    scales: Dict[str, Literal['lin', 'log']] = {}

    #: The kind of submodels supported by holding some parameters fixed
    submodels: Dict[str, Dict[str, float]] = dict(
        full=dict(),
        dele=dict()
    )

    def __init__(self):
        """
        Initialize parametrization.
        """
        self._logger = logger.getChild(self.__class__.__name__)

        #: argument names
        self.param_names: List = list(self.x0.keys())

    @staticmethod
    def _accepts_scalars(func: Callable) -> Callable[[np.ndarray | float], np.ndarray | float]:
        """
        Make func accept scalar values.

        :return: Function that accepts scalars
        """

        @wraps(func)
        def wrapper(S: np.ndarray, *args, **kwargs) -> np.ndarray:
            """
            Wrapper function.

            :param S: Array of selection coefficients
            :param args: Positional arguments
            :param kwargs: Keyword arguments
            :return: Output of func
            """
            if isinstance(S, (int, float)):
                return func(np.array([S]), *args, **kwargs)[0]
            else:
                return func(S, *args, **kwargs)

        return wrapper

    @abstractmethod
    def get_pdf(self, **kwargs) -> Callable[[np.ndarray], np.ndarray]:
        """
        Get probability distribution function of DFE.

        :return: Function that accepts an array of selection coefficients and returns their probability density
        """
        pass

    @abstractmethod
    def get_cdf(self, **kwargs) -> Callable[[np.ndarray], np.ndarray]:
        """
        Get probability distribution function of DFE.

        :return: Function that accepts an array of selection coefficients and returns their cumulative probability
        """
        pass

    def _discretize(
            self,
            params,
            bins: np.ndarray,
            warn_mass: bool = True
    ) -> np.ndarray:
        """
        Discretize by using the CDF.

        :param params: Parameters of the parametrization
        :param bins: Bins to use for discretization
        :param warn_mass: Whether to warn if the total mass is not near 1.
        :return: Histogram
        """
        x = self.get_cdf(**params)(bins)

        # issue warning if values at bounds are outside [0, 1]
        if warn_mass and (not -1e-5 < x[0] < 1e-5 or not 1 - 1e-5 < x[-1] < 1 + 1e-5):
            self._logger.warning(f'CDF evaluates to {(x[0], x[-1])} at the lower and upper '
                                 f'bounds, which is a bit off from the expected (0, 1). '
                                 f'Used parameters: {params}.')

        # compute histogram
        hist = x[1:] - x[:-1]

        return hist

    def _normalize(self, params: dict) -> dict:
        """
        Normalize parameters.

        :param params: Parameters of the parametrization
        :return: Normalized parameters
        """
        # do nothing by default
        return params

    def plot(
            self,
            params: dict,
            intervals: list | np.ndarray = np.array([-np.inf, -100, -10, -1, 0, 1, np.inf]),
            file: str = None,
            show: bool = True,
            title: str = 'discretized DFE',
            ax: plt.Axes = None

    ) -> plt.Axes:
        """
        Plot the discretized DFE.

        :param params: Parameters of the parametrization
        :param intervals: Intervals to use for discretization.
        :param file: File to save the plot to.
        :param show: Whether to show the plot.
        :param title: Title of the plot.
        :param ax: Axes to use for the plot.
        :return: Axes
        """
        from .visualization import Visualization

        values = self._discretize(params, np.array(intervals), warn_mass=False)

        # check for nan values
        if np.isnan(values).any():
            self._logger.warning(f'NaN values in discretized DFE. Are the parameters valid?')

        return Visualization.plot_discretized(
            values=[values],
            file=file,
            show=show,
            intervals=np.array(intervals),
            title=title,
            ax=ax
        )


class GammaExpParametrization(Parametrization):
    r"""
    Parametrization for mixture of a gamma and exponential distribution. This corresponds to
    model C in polyDFE.

    We have the following probability density function:

    .. math::
        \phi(S; S_d, b, p_b, S_b) = (1 - p_b) f_\Gamma(-S; S_d, b) \cdot \mathbf{1}_{\{S < 0\}} +
        p_b f_e(S; S_b) \cdot \mathbf{1}_{\{S \geq 0\}}

    where:

    * :math:`S_d` is the mean of the DFE for :math:`S < 0`
    * :math:`b` is the shape of the gamma distribution
    * :math:`p_b` is the probability that :math:`S \geq 0`
    * :math:`S_b` is the mean of the DFE for :math:`S \geq 0`
    * :math:`f_\Gamma(x; m, b)` is the density of the gamma distribution with mean :math:`m` and shape :math:`b`
    * :math:`f_e(x; m)` is the density of the Exponential distribution with mean :math:`m`
    * :math:`\mathbf{1}_{\{A\}}` denotes the indicator function, which is 1 if :math:`A` is true, and 0 otherwise.

    The DFE has often been observed to be multi-modal for negative selection coefficients. A gamma distribution
    provides a good amount of flexibility to accommodate this.

    """

    #: Default initial parameters
    x0: Dict[str, float] = dict(
        S_d=-1000,
        b=0.4,
        p_b=0.05,
        S_b=1
    )

    #: Default parameter bounds, using non-zero lower bounds for S_d and S_b due to log-scaled scales
    bounds: Dict[str, Tuple[float, float]] = dict(
        S_d=(-1e5, -1e-2),
        b=(0.01, 10),
        p_b=(0, 0.5),
        S_b=(1e-4, 100)
    )

    #: Scales over which to optimize the parameters
    scales: Dict[str, Literal['lin', 'log']] = dict(
        S_d='log',
        b='log',
        p_b='lin',
        S_b='log'
    )

    #: The kind of submodels supported by holding some parameters fixed
    submodels: Dict[str, Dict[str, float]] = dict(
        full=dict(),
        dele=dict(
            p_b=0,
            S_b=1
        )
    )

    @staticmethod
    def get_pdf(S_d: float, b: float, p_b: float, S_b: float, **kwargs) -> Callable[[np.ndarray], np.ndarray]:
        """
        Get PDF.

        :param S_d: Mean selection coefficient for deleterious mutations
        :param b: Shape parameter for gamma distribution
        :param p_b: Probability of a beneficial mutation
        :param S_b: Mean selection coefficient for beneficial mutations
        :return: Function that accepts an array of selection coefficients and returns their probability density
        """

        @Parametrization._accepts_scalars
        def pdf(S: np.ndarray) -> np.ndarray:
            """
            The PDF.

            :param S: Selection coefficients
            :return: Probability density
            """
            x = np.zeros_like(S, dtype=float)

            # The gamma distribution may approach infinity as S -> 0
            # so in order to evaluate this function at s = 0,
            # we use--unlike polyDFE--the exponential mixture
            # distribution at this point.
            negative = S < 0

            # positive S
            x[negative] = (1 - p_b) * gamma.pdf(-S[negative], b, scale=-S_d / b)

            # Allow for S_b == 0 when p_b == 0 as well
            # which would produce an error otherwise.
            if S_b == 0 and p_b == 0:
                x[~negative] = 0
            else:
                # non-negative S
                x[~negative] = p_b * expon.pdf(S[~negative], scale=S_b)

            return x

        return pdf

    @staticmethod
    def get_cdf(S_d: float, b: float, p_b: float, S_b: float, **kwargs) -> Callable[[np.ndarray], np.ndarray]:
        """
        Get CDF.

        :param S_d: Mean selection coefficient for deleterious mutations
        :param b: Shape parameter for gamma distribution
        :param p_b: Probability of a beneficial mutation
        :param S_b: Mean selection coefficient for beneficial mutations
        :return: Function that accepts an array of selection coefficients and returns their cumulative probability
        """

        @Parametrization._accepts_scalars
        def cdf(S: np.ndarray) -> np.ndarray:
            """
            The CDF.

            :param S: Selection coefficients
            :return: Cumulative probability
            """
            x = np.zeros_like(S, dtype=float)

            negative = S < 0

            # positive S
            x[negative] = (1 - p_b) - ((1 - p_b) * gamma.cdf(-S[negative], b, scale=-S_d / b))

            # Allow for S_b = 0 when p_b = 0 as well
            # which would produce an error otherwise.
            if S_b == 0 and p_b == 0:
                x[~negative] = 1
            else:
                # non-negative S
                x[~negative] = (1 - p_b) + p_b * expon.cdf(S[~negative], scale=S_b)

            return x

        return cdf


class DisplacedGammaParametrization(Parametrization):
    r"""
    Parametrization for a reflected displaced gamma distribution.

    We have the following probability density function:

    .. math::
        \phi(S; \hat{S}, b, S_{max}) = f_\Gamma(S_{max} - S; S_{max} - \hat{S}, b) \cdot \mathbf{1}_{\{S \leq S_{max}\}}

    where:

    * :math:`\hat{S}` is the mean of the DFE
    * :math:`b` is the shape of the gamma distribution
    * :math:`S_{max}` is the maximum value that :math:`S` can take
    * :math:`f_\Gamma(x; m, b)` is the density of the gamma distribution with mean :math:`m` and shape :math:`b`
    * :math:`\mathbf{1}_{\{A\}}` denotes the indicator function, which is 1 if :math:`A` is true, and 0 otherwise.

    This parametrization uses a single gamma distribution for both positive and negative selection coefficients.
    This is a less flexible parametrization, which may produce results similar to the other models while requiring
    fewer parameters.

    .. warning::
        This model does not allow for a purely deleterious sub-parametrization, so
        :meth:`Inference.compare_nested_models` won't work as expected.

    """

    #: Default initial parameters
    x0: Dict[str, float] = dict(
        S_mean=-100,
        b=1,
        S_max=1
    )

    #: Default parameter bounds
    bounds: Dict[str, Tuple[float, float]] = dict(
        S_mean=(-100000, -0.01),
        b=(0.01, 10),
        S_max=(0.001, 100)
    )

    #: Scales over which to optimize the parameters
    scales: Dict[str, Literal['lin', 'log']] = dict(
        S_mean='log',
        b='lin',
        S_max='log'
    )

    #: The kind of submodels supported by holding some parameters fixed
    submodels: Dict[str, Dict[str, float]] = dict(
        full=dict(),
        dele=dict()
    )

    @staticmethod
    def get_pdf(S_mean: float, b: float, S_max: float, **kwargs) -> Callable[[np.ndarray], np.ndarray]:
        """
        Get PDF.

        :return: Function that accepts an array of selection coefficients and returns their probability density
        """

        @Parametrization._accepts_scalars
        def pdf(S: np.ndarray) -> np.ndarray:
            """
            The PDF.

            :param S: Selection coefficients
            :return: Probability density
            """
            x = np.zeros_like(S, dtype=float)

            # calculate the probability density for S <= S_max
            is_lower = S <= S_max
            x[is_lower] = gamma.pdf(S_max - S[is_lower], b, scale=max(S_max - S_mean, 1e-16) / b)

            return x

        return pdf

    @staticmethod
    def get_cdf(S_mean: float, b: float, S_max: float, **kwargs) -> Callable[[np.ndarray], np.ndarray]:
        """
        Get CDF.

        :return: Function that accepts an array of selection coefficients and returns their cumulative probability
        """

        @Parametrization._accepts_scalars
        def cdf(S: np.ndarray) -> np.ndarray:
            """
            The CDF.

            :param S: Selection coefficients
            :return: Cumulative probability
            """
            x = np.zeros_like(S, dtype=float)

            # calculate the cumulative probability for S <= S_max
            is_lower = S <= S_max
            x[is_lower] = 1 - gamma.cdf(S_max - S[is_lower], b, scale=max(S_max - S_mean, 1e-16) / b)

            # set the cumulative probability to 1 for S > S_max
            x[~is_lower] = 1

            return x

        return cdf


class GammaDiscreteParametrization(Parametrization):
    r"""
    Parametrization for a mixture of a gamma and discrete distribution. This corresponds to polyDFE's model B.

    We have the following probability density function:

    .. math::
        \phi(S; S_d, b, p_b, S_b) = (1 - p_b) f_{\Gamma}(S; S_d, b) \cdot \mathbf{1}_{\{S < 0\}} +
        p_b \cdot S_b \cdot \mathbf{1}_{\{0 \leq S \leq 1 / S_b\}}

    where:

    * :math:`S_d` is the mean of the DFE for :math:`S < 0`
    * :math:`b` is the shape of the gamma distribution
    * :math:`p_b` is the probability that :math:`S \geq 0`
    * :math:`S_b` is the shared selection coefficient of all positively selected mutations up to :math:`1/S_b`
    * :math:`f_\Gamma(x; m, b)` is the density of the gamma distribution with mean :math:`m` and shape :math:`b`

    This parametrization is similar to :class:`GammaExpParametrization`, but uses a constant mass for positive
    selection coefficients. The results should be rather similar in most cases.
    """

    #: Default initial parameters
    x0: Dict[str, float] = dict(
        S_d=-1000,
        b=0.4,
        p_b=0.05,
        S_b=1
    )

    #: default parameter bounds
    bounds: Dict[str, Tuple[float, float]] = dict(
        S_d=(-1e5, -1e-2),
        b=(0.01, 10),
        p_b=(0, 0.5),
        S_b=(1e-4, 100)
    )

    #: scales over which to optimize the parameters
    scales: Dict[str, Literal['lin', 'log']] = dict(
        S_d='log',
        b='log',
        p_b='lin',
        S_b='log'
    )

    #: The kind of submodels supported by holding some parameters fixed
    submodels: Dict[str, Dict[str, float]] = dict(
        full=dict(),
        dele=dict(
            p_b=0,
            S_b=1
        )
    )

    @staticmethod
    def get_pdf(S_d: float, b: float, p_b: float, S_b: float, **kwargs) -> Callable[[np.ndarray], np.ndarray]:
        """
        Get PDF.

        :param S_d: Mean of the DFE for S >= 0
        :param b: Shape of the gamma distribution
        :param p_b: Probability that S > 0
        :param S_b: Shared selection coefficient of all positively selected mutations up to S_b
        :return: Function that accepts an array of selection coefficients and returns their probability density
        """

        @Parametrization._accepts_scalars
        def pdf(S: np.ndarray) -> np.ndarray:
            """
            The PDF.

            :param S: Selection coefficients
            :return: Probability density
            """
            x = np.zeros_like(S, dtype=float)

            negative = S < 0
            x[negative] = (1 - p_b) * gamma.pdf(-S[negative], b, scale=-S_d / b)

            equal_to_S_b = (0 <= S) & (S <= 1 / S_b)
            x[equal_to_S_b] = p_b * S_b

            # the density is 0 for all other cases, no need to explicitly set it

            return x

        return pdf

    @staticmethod
    def get_cdf(S_d: float, b: float, p_b: float, S_b: float, **kwargs) -> Callable[[np.ndarray], np.ndarray]:
        """
        Get CDF.

        :param S_d: Mean of the DFE for S >= 0
        :param b: Shape of the gamma distribution
        :param p_b: Probability that S > 0
        :param S_b: Shared selection coefficient of all positively selected mutations up to S_b
        :return: Function that accepts an array of selection coefficients and returns their cumulative probability
        """

        @Parametrization._accepts_scalars
        def cdf(S: np.ndarray) -> np.ndarray:
            """
            The CDF.

            :param S: Selection coefficients
            :return: Cumulative probability
            """
            x = np.zeros_like(S, dtype=float)

            negative = S < 0
            x[negative] = (1 - p_b) - ((1 - p_b) * gamma.cdf(-S[negative], b, scale=-S_d / b))

            within_S_b = (0 <= S) & (S <= 1 / S_b)
            x[within_S_b] = (1 - p_b) + p_b * S_b * S[within_S_b]

            greater_than_S_b = S > 1 / S_b
            x[greater_than_S_b] = 1

            return x

        return cdf


class DiscreteParametrization(Parametrization):
    r"""
    Parametrization for a discrete distribution. This corresponds to polyDFE's model D.
    By default we use 6 bins, but this can be changed by passing a different array to the constructor.
    The resulting parameter names are :math:`S_1, S_2, \dots, S_k`, where :math:`k` is the number of bins.

    That is, the probability density function is given by:

    :math:`\phi(S; S_1, S_2, \dots, S_k) = \sum_{i=1}^{k} S_i/c_i \cdot \mathbf{1}_{\{S \in B_i\}}`
    such that :math:`\sum_{i=1}^{k} S_i = 1,`

    where :math:`B_i` and :math:`c_i` are interval :math:`i` and the width of interval :math:`i`, respectively.

    This parametrization has the advantage of not imposing a shape on the DFE. For a reasonably fine parametrization,
    the number of parameters is larger than those of the other models, however. We generally also observe larger
    confidence intervals for this parametrization, and the optimization procedure may well be less efficient as
    we have to re-normalize the parameters to make sure they sum up to 1.
    """

    def __init__(
            self,
            intervals: np.ndarray | list = np.array([-100000, -100, -10, -1, 0, 1, 1000])
    ):
        """
        Constructor.

        :param intervals: The intervals of the discrete distribution.
        """
        super().__init__()

        #: Intervals
        self.intervals = np.concatenate(([-np.inf], intervals, [np.inf]))

        #: Interval sizes
        self.interval_sizes = self.intervals[1:] - self.intervals[:-1]

        #: Number of intervals, including the two infinite ones
        self.k = self.intervals.shape[0] - 1

        #: All parameter names, including the fixed ones
        self.params = np.array([f"S{i}" for i in range(self.k)])

        #: Parameter names that are not fixed
        self.param_names: List[str] = self.params[1:-1].tolist()

        #: Fixed parameters
        self.fixed_params = {self.params[0]: 0, self.params[-1]: 0}

        #: Default initial parameters
        self.x0: Dict[str, float] = dict((p, 1 / (self.k - 2)) for p in self.param_names)

        #: Default parameter bounds
        self.bounds: Dict[str, Tuple[float, float]] = dict((p, (0, 1)) for p in self.param_names)

        #: Scales
        # noinspection all
        self.scales: Dict[str, Literal['lin', 'log']] = dict((p, 'lin') for p in self.param_names)

        #: Submodels
        self.submodels: Dict[str, Dict[str, float]] = dict(
            full=dict(),
            dele=dict((p, 0) for p in self.params[1:-1][self.intervals[1:-2] < 0])
        )

    def _normalize(self, params: dict) -> dict:
        """
        Add params for boundaries and normalize so that
        the parameters sum up to 1.

        :param params: Parameter values
        :return: Dict of normalized values plus remaining parameters
        """
        # make sure we only include parameter used for this parametrization
        filtered = dict((k, v) for k, v in params.items() if k in self.param_names)

        # sum up
        s = np.sum(list(filtered.values()))

        # normalize and include remaining parameters
        return params | dict((k, v / s) for k, v in filtered.items())

    def get_pdf(self, **kwargs) -> Callable[[np.ndarray], np.ndarray]:
        """
        Get PDF.

        :param kwargs: Parameter values S1, S2, ..., Sk
        :return: Function that computes the PDF for given S values
        """

        # normalize and add boundary parameters
        values = self._normalize(kwargs) | self.fixed_params

        @Parametrization._accepts_scalars
        def pdf(S: np.ndarray) -> np.ndarray:
            """
            The PDF.

            :param S: Selection coefficients
            :return: Probability density
            """
            x = np.zeros_like(S, dtype=float)

            # iterate over parameters and assign values
            for i, p in enumerate(self.params):
                x[(self.intervals[i] < S) & (S <= self.intervals[i + 1])] = values[p] / self.interval_sizes[i]

            return x

        return pdf

    def get_cdf(self, **kwargs) -> Callable[[np.ndarray], np.ndarray]:
        """
        Get CDF.

        :param kwargs: Parameter values S1, S2, ..., Sk
        :return: Function that computes the CDF for given S values
        """

        # normalize and add boundary parameters
        values = self._normalize(kwargs) | self.fixed_params

        # convert to numpy arrays
        intervals = np.array(self.intervals)
        interval_sizes = np.array(self.interval_sizes)

        # convert to ordered array values
        vals = np.array([values[self.params[i]] for i in range(self.k)])

        @Parametrization._accepts_scalars
        def cdf(S: np.ndarray) -> np.ndarray:
            """
            The CDF.

            :param S: Selection coefficients
            :return: Cumulative probability
            """
            # iterate over parameters and assign values
            cum = np.cumsum([values[self.params[i]] for i in range(self.k)])

            # obtain bin indices
            i = np.sum(self.intervals[:, None] <= S[None, :], axis=0) - 1

            # make sure we don't go out of bounds which can happen if S is np.inf
            i[i >= self.k] = self.k - 1

            # cumulative probability up to previous bin
            cum_prev = cum[np.maximum(i - 1, np.zeros_like(S, dtype=int))]

            # probability in current bin
            cum_within = np.abs(intervals[i] - S) / interval_sizes[i] * vals[i]
            cum_within[np.isnan(cum_within)] = 0

            # return cumulative probability
            return cum_prev + cum_within

        return cdf


class DiscreteFractionalParametrization(Parametrization):
    r"""
    Same model as :class:`DiscreteParametrization`, but re-parametrized by
    :math:`\hat{S}_1, \hat{S}_2, \dots, \hat{S}_{k-1}`, so that the mass in the ith interval :math:`S_i`
    is determined by the sum of masses to the left, i.e.

    :math:`S_i = \hat{S}_i \sum_{j<i} S_j, i = 1, \dots, k - 1`,

    :math:`S_k = 1 - \sum_{j=1}^{k-1} S_j`.

    This parametrization has the advantage of not imposing a shape on the DFE. For a reasonably fine parametrization,
    the number of parameters is larger than those of the other models, however. It is more easily optimized than
    :class:`DiscreteParametrization` as it has one parameter less but its parameters are more difficult to interpret.
    One disadvantage with discrete parametrizations is that there may be `gaps` in the estimated DFE.
    """

    def __init__(
            self,
            intervals: np.ndarray | list = np.array([-100000, -100, -10, -1, 0, 1, 1000])
    ):
        """
        Constructor.

        :param intervals: The intervals of the discrete distribution.
        """
        super().__init__()

        #: Intervals
        self.intervals = np.concatenate(([-np.inf], intervals, [np.inf]))

        #: Interval sizes
        self.interval_sizes = self.intervals[1:] - self.intervals[:-1]

        #: Number of intervals, including the two infinite ones
        self.k = self.intervals.shape[0] - 1

        #: All parameter names, including fixed parameters
        self.params = np.array([f"S{i}" for i in range(self.k)])

        #: Parameter names that are not fixed
        self.param_names = self.params[1:-2].tolist()

        #: Fixed parameters
        self.fixed_params = {self.params[0]: 0, self.params[-2]: 1, self.params[-1]: 0}

        #: Default initial parameters
        self.x0: Dict[str, float] = self.to_fractional(dict((p, 1 / (self.k - 2)) for p in self.param_names))

        #: Default parameter bounds
        self.bounds: Dict[str, Tuple[float, float]] = dict((p, (0, 1)) for p in self.param_names)

        #: Scales
        # noinspection all
        self.scales: Dict[str, Literal['lin', 'log']] = dict((p, 'lin') for p in self.param_names)

        #: Submodels
        self.submodels: Dict[str, Dict[str, float]] = dict(
            full=dict(),
            dele=dict((p, 0) for p in self.params[1:-1][self.intervals[1:-2] < 0])
        )

    def to_nominal(self, params: Dict[str, float]) -> Dict[str, float]:
        """
        Convert representation of fraction of total mass to the left
        to representation of fractions which sum to 1.

        :param params: Parameter values S0, S2, ..., Sk
        :return: Converted parameters
        """
        # converted parameters
        converted = params.copy()

        # cumulative sum up to the previous parameter
        mass = 0

        # take mass of current bin to be fraction of mass assigned to the previous bins
        for p in self.params[1:-2]:
            converted[p] = params[p] * (1 - mass)
            mass += converted[p]

        # assign remaining mass to last bin
        converted[self.params[-2]] = 1 - mass

        return converted

    def to_fractional(self, params: Dict[str, float]) -> Dict[str, float]:
        """
        Invert the to_nominal operation: Convert representation of fractions which sum to 1
        back to representation of fraction of total mass to the left.

        :param params: Converted parameters (nominal space)
        :return: Original parameters (parameter space)
        """
        # converted parameters
        converted = params.copy()

        # cumulative sum up to the previous parameter
        mass = 0

        # Iterate over parameters
        for p in self.params[1:-2]:
            # Calculate original parameter value
            converted[p] = params[p] / (1 - mass)

            # update cumulative sum
            mass += params[p]

        # last parameter is simply what's left
        converted[self.params[-2]] = 1 - mass

        return converted

    def get_pdf(self, **kwargs) -> Callable[[np.ndarray], np.ndarray]:
        """
        Get PDF.

        :param kwargs: Parameter values S1, S2, ..., Sk-1
        :return: Function that computes the PDF for given S values
        """

        # normalize and add boundary parameters
        values = self.to_nominal(kwargs | self.fixed_params)

        @Parametrization._accepts_scalars
        def pdf(S: np.ndarray) -> np.ndarray:
            """
            The PDF.

            :param S: Selection coefficients
            :return: Probability density
            """
            x = np.zeros_like(S, dtype=float)

            # iterate over parameters and assign values
            for i, p in enumerate(self.params):
                x[(self.intervals[i] < S) & (S <= self.intervals[i + 1])] = values[p] / self.interval_sizes[i]

            return x

        return pdf

    def get_cdf(self, **kwargs) -> Callable[[np.ndarray], np.ndarray]:
        """
        Get CDF.

        :param kwargs: Parameter values S1, S2, ..., Sk-1
        :return: Function that computes the CDF for given S values
        """

        # normalize and add boundary parameters
        values = self.to_nominal(kwargs | self.fixed_params)

        # convert to numpy arrays
        intervals = np.array(self.intervals)
        interval_sizes = np.array(self.interval_sizes)

        # convert to ordered array values
        vals = np.array([values[self.params[i]] for i in range(self.k)])

        @Parametrization._accepts_scalars
        def cdf(S: np.ndarray) -> np.ndarray:
            """
            The CDF.

            :param S: Selection coefficients
            :return: Cumulative probability
            """
            # iterate over parameters and assign values
            cum = np.cumsum([values[self.params[i]] for i in range(self.k)])

            # obtain bin indices
            i = np.sum(self.intervals[:, None] <= S[None, :], axis=0) - 1

            # make sure we don't go out of bounds which can happen if S is np.inf
            i[i >= self.k] = self.k - 1

            # cumulative probability up into previous bin
            cum_prev = cum[np.maximum(i - 1, np.zeros_like(S, dtype=int))]

            # probability in current bin
            cum_within = np.abs(intervals[i] - S) / interval_sizes[i] * vals[i]
            cum_within[np.isnan(cum_within)] = 0

            # return cumulative probability
            return cum_prev + cum_within

        return cdf
