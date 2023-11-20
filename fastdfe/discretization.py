"""
Discretization of DFE to SFS transformation.
"""

__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"
__date__ = "2023-02-26"

import logging
from functools import cached_property, wraps
from typing import Literal

import mpmath as mp
import numpy as np
from scipy.integrate import quad

from .optimization import parallelize as parallelize_func
from .parametrization import Parametrization

# configure logger
logger = logging.getLogger('fastdfe').getChild('Discretization')


def get_midpoints_and_spacing(bins: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Obtain midpoints and spacing for the given bins.

    :param bins: Array of bins
    :return: Midpoints, spacing
    """
    # obtain midpoints
    s = (bins[1:] + bins[:-1]) / 2
    interval_sizes = bins[1:] - bins[:-1]

    return s, interval_sizes


def get_bins(intervals_del: tuple, intervals_ben: tuple) -> np.ndarray:
    """
    Get bins using log-spaced distances for positive and negative
    values of S given by ``intervals_ben`` and ``intervals_del``, respectively.

    :param intervals_del: ``(min, max, n)`` for negative values of S where we use log-spaced distances
    :param intervals_ben: ``(min, max, n)`` for positive values of S where we use log-spaced distances
    :return: Bins
    """
    bins_del = -np.logspace(np.log10(np.abs(intervals_del[0])),
                            np.log10(np.abs(intervals_del[1])), intervals_del[2] + 1)

    bins_ben = np.logspace(np.log10(intervals_ben[0]), np.log10(intervals_ben[1]), intervals_ben[2] + 1)

    return np.concatenate([bins_del, bins_ben])


def H(x, S):
    """
    Allele frequency sojourn time. Note that this function is not used in practice.

    :param S: Selection coefficient
    :param x: Allele frequency
    :return: Allele frequency sojourn time
    """
    return ((1 - np.exp(-S * (1 - x), dtype=np.float128)) /
            (x * (1 - x) * (1 - np.exp(-S, dtype=np.float128)))).astype(float)


def H_regularized(x, S: float | int | np.ndarray):
    """
    As H(x, S) but replacing with the limits close to the limit points.
    Note that this function is not used in practice.

    :param x: Allele frequency
    :param S: Selection coefficient
    :return: Allele frequency sojourn time
    """
    # make it accept scalars
    if isinstance(S, (float, int)):
        return H_regularized(x, np.array([S]))[0]

    # else S is an array
    y = np.zeros_like(S, dtype=np.float128)

    # S close to zero
    close_to_zero = np.abs(S) < 1e-8

    # limit (1 - exp(-S * (1 - x))) / (x * (1 - x) * (1 - exp(-S))) as S -> 0
    # cf. https://www.wolframalpha.com/input?i=limit+%281+-+exp%28-S+*+%281+-+x%29%29%29+%2F+%
    # 28x+*+%281+-+x%29+*+%281+-+exp%28-S%29%29%29+as+S+-%3E+0
    y[close_to_zero] = 1 / x

    # evaluate function as usual
    y[~close_to_zero] = H(x, S[~close_to_zero])

    return y


def H_fixed(S: float | int | np.ndarray) -> float | int | np.ndarray:
    """
    The sojourn time as x -> 1. Note that this function is not used in practice.

    :param S: Selection coefficient
    :return: Sojourn time as x -> 1
    """
    return S / (1 - np.exp(-S))


def H_fixed_regularized(S: float | int | np.ndarray) -> float | int | np.ndarray:
    """
    As :func:`H_fixed` but replacing with the limits close to the limit points.
    Note that this function is not used in practice.

    :param S: Selection coefficient
    :return: Sojourn time as x -> 1
    """
    # make it accept scalars
    if isinstance(S, (float, int)):
        return H_fixed(np.array([S]))[0]

    # else S is an array
    y = np.zeros_like(S)

    # S close to zero
    close_to_zero = np.abs(S) < 1e-8

    # limit as S -> 0
    y[close_to_zero] = 1

    # avoid overflow for very low S
    very_negative = S < -500

    # for very low S we are close to 0
    y[very_negative] = 0

    # remaining values
    remaining = ~(close_to_zero | very_negative)

    # evaluate as usual
    y[remaining] = H_fixed(S[remaining])

    return y


def hyp1f1(a: float | np.ndarray, b: float | np.ndarray, z: float | np.ndarray) -> float | np.ndarray:
    """
    Vectorized version of mpmath's hyp1f1 function.

    :param a: First parameter
    :param b: Second parameter
    :param z: Third parameter
    :return: Result
    """
    return np.vectorize(mp.hyp1f1)(a, b, z)


def exp(x: float | np.ndarray) -> float | np.ndarray:
    """
    Vectorized version of mpmath's exp function.

    :param x: Exponent
    :return: Result
    """
    return np.vectorize(mp.exp)(x)


def to_float(func):
    """
    Decorator to convert the return value of a function to float.

    :param func: Function to be decorated
    :return: Wrapper function
    """

    @wraps(func)
    def wrapper(*args, **kwargs) -> float | np.ndarray:
        """
        Wrapper function.

        :param args: Positional arguments
        :param kwargs: Keyword arguments
        :return: Converted result
        """
        result = func(*args, **kwargs)

        if isinstance(result, np.ndarray):
            return result.astype(float)
        else:
            return float(result)

    return wrapper


class Discretization:
    """
    The integral mapping the DFE to allele counts can be discretized
    and computed once beforehand so that transforming the DFE to
    the expected SFS counts can be carried out by matrix multiplication.
    To evaluate the integral over the specified intervals, midpoint
    integration or Scipy's quad method can be used.
    The integral can also alternatively be evaluated numerically
    in each optimizing iteration.
    The loss function supports Poisson likelihoods and the L2 norm.
    """

    def __init__(
            self,
            n: int,
            intervals_del: (float, float, int) = (-1.0e+8, -1.0e-5, 1000),
            intervals_ben: (float, float, int) = (1.0e-5, 1.0e4, 1000),
            integration_mode: Literal['midpoint', 'quad'] = 'midpoint',
            linearized: bool = True,
            parallelize: bool = True,
    ):
        """
        Create Discretization instance.

        :return: Number of individuals
        :param intervals_del: ``(start, stop, n_interval)`` for deleterious population-scaled selection coefficients.
        :param intervals_ben: ``(start, stop, n_interval)`` for beneficial population-scaled selection coefficients.
        :param integration_mode : 'midpoint' or 'quad' for midpoint integration or Scipy's quad method.
        :return: Whether to linearize the integral
        :param parallelize: Whether to parallelize the computation of the discretization
        """
        self.n = n

        # make sure lower bounds are lower than upper bounds
        if not intervals_del[0] < intervals_del[1] or not intervals_ben[0] < intervals_ben[1]:
            raise Exception('Lower intervals bounds must be lower than upper bounds.')

        # make sure |S| is not too large
        if intervals_del[0] < -1e10 or intervals_ben[1] > 1e10:
            raise Exception('Bounds for S should within the interval [-1e10, 1e10] to avoid '
                            'unexpected numerical behavior.')

        # bounds for discretizing DFE
        self.intervals_del = intervals_del
        self.intervals_ben = intervals_ben

        self.integration_mode = integration_mode
        self.linearized = linearized
        self.parallelize = parallelize

        # iteration counter
        self.n_it = 0

        # define bins, midpoints and step size
        # these intervals are used when linearizing the integral
        # and when plotting the DFE
        self.bins = get_bins(intervals_del, intervals_ben)
        self.s, self.interval_sizes = get_midpoints_and_spacing(self.bins)

        # the number of intervals
        self.n_intervals = self.s.shape[0]

    def get_dfe_to_sfs_midpoint(self) -> np.ndarray:
        """
        Precompute linearized integral using midpoint integration.
        Consider parallelizing this.

        :return: Matrix of size (n_intervals, n)
        """
        logger.info('Precomputing linear DFE-SFS transformation using midpoint integration.')

        I = np.ones((self.n - 1, self.n_intervals + 1))

        # we use midpoint integration
        K = np.arange(1, self.n)[:, None] * I
        S = self.bins[None, :] * I

        def compute_slice(i: int) -> np.ndarray:
            """
            Compute allele counts of a given multiplicity.

            :param i: Multiplicity
            :return: Discretized counts
            """
            return self.get_allele_count_regularized(S[i], K[i])

        # retrieve allele counts
        P = parallelize_func(
            func=compute_slice,
            data=np.arange(self.n - 1),
            parallelize=self.parallelize,
            desc=f"{self.__class__.__name__}>Precomputing",
            dtype=float
        )

        # take midpoint and multiply by interval size
        return (P[:, :-1] + P[:, 1:]) / 2 * self.interval_sizes

    def get_dfe_to_sfs_quad(self) -> np.ndarray:
        """
        Precompute linearized integral using Scipy's quad method.

        :return: Matrix of size (n_intervals, n)
        """
        logger.info('Precomputing linear DFE-SFS transformation using scipy.integrate.quad.')

        # initialize matrix
        P = np.zeros((self.bins.shape[0] - 1, self.n - 1))

        # iterate over bins
        for i in range(self.bins.shape[0] - 1):

            logger.debug(f"Processing interval {(self.bins[i], self.bins[i + 1])}.")

            # iterate over allele count classes
            for j in range(self.n - 1):
                P[i, j] = quad(lambda s: (self.get_allele_count_regularized(
                    s, j + 1)), self.bins[i], self.bins[i + 1])[0]

        # For GammaExpParametrization, there is a discontinuity at 0
        # which can cause problems integrating it using quadrature.
        # We remove this discontinuity here by taking the average value
        # of the adjacent bins.
        # Find bin that contains 0.
        i_zero = np.where((self.bins[:-1] < 0) & (self.bins[1:] >= 0))[0][0]

        # take average of adjacent bins
        P[i_zero] = (P[i_zero - 1] + P[i_zero + 1]) / 2

        return P.T

    @cached_property
    def dfe_to_sfs(self) -> np.ndarray:
        """
        Linearized DFE to SFS transformation.

        :return: Matrix of size (n_intervals, n)
        """
        # precompute linearized transformation.
        if self.integration_mode == 'midpoint':
            return self.get_dfe_to_sfs_midpoint()

        if self.integration_mode == 'quad':
            return self.get_dfe_to_sfs_quad()

        raise NotImplementedError(f'Integration mode {self.integration_mode} not supported.')

    @to_float
    def get_allele_count_large_negative_S(self, S: float | np.ndarray, k: float | np.ndarray) -> np.ndarray | float:
        """
        Limit of get_allele_count for large negative S.

        :param S: Population-scaled selection coefficient
        :param k: Allele count
        :return: Number of counts in frequency class P(k)
        """
        return self.n / (k * (self.n - k)) * hyp1f1(k, self.n, S)

    def get_allele_count_large_positive_S(self, S: float | np.ndarray, k: float | np.ndarray) -> np.ndarray | float:
        """
        Limit of get_allele_count for large positive S.

        :param S: Population-scaled selection coefficient
        :param k: Allele count
        :return: Number of counts in frequency class P(k)
        """
        return self.n / (k * (self.n - k))

    @to_float
    def get_allele_count(self, S: float | np.ndarray, k: float | np.ndarray) -> np.ndarray | float:
        """
        The number of counts in frequency class P(k) with
        population-scaled selection coefficient S.
        Binomial sampling is included here.
        See appendix of https://pubmed.ncbi.nlm.nih.gov/31975166

        :param S: Population-scaled selection coefficient
        :param k: Allele count
        :return: Number of counts in frequency class P(k)
        """
        return (self.n / (k * (self.n - k))) * ((1 - exp(-S) * hyp1f1(k, self.n, S)) / (1 - exp(-S)))

    def get_allele_count_regularized(self, S: float | np.ndarray, k: float | np.ndarray) -> float | np.ndarray:
        """
        As :meth:`get_allele_count` but using the respective limits
        for ``S`` close to zero and ``S`` very negative.
        Note that ``S`` and ``k`` need to have the same shape.

        :param S: Population-scaled selection coefficient
        :param k: Allele count
        :return: Number of counts in frequency class ``P(k)`
        """
        # make it accept scalars for S
        if isinstance(S, (float, int)):
            return self.get_allele_count_regularized(np.array([S]), np.array([k]))[0]

        # else S is an array
        y = np.zeros_like(S, dtype=float)

        # S close to zero
        close_to_zero = np.abs(S) < 1e-8

        # simply take limit value as S -> 0
        # see https://www.wolframalpha.com/input?i=limit+%28n%2F%28k%28n-k%29%29%29+*+
        # 1%2F%281-exp%28-S%29%29+*+%281-exp%28-S%29*1F1%28k%2Cn%2CS%29%29+as+S+-%3E+0
        if close_to_zero.any():
            y[close_to_zero] = np.ones_like(S[close_to_zero]) / k[close_to_zero]

        # large negative S
        very_negative = S < -1e4

        # Evaluate using limit.
        # We check if there are any negative values here
        # as numpy's vectorize does not work with empty arrays.
        if very_negative.any():
            y[very_negative] = self.get_allele_count_large_negative_S(S[very_negative], k[very_negative])

        # remaining values of S
        remaining = np.array(~(close_to_zero | very_negative))

        # evaluate function as usual for remaining values
        if remaining.any():
            y[remaining] = self.get_allele_count(S[remaining], k[remaining])

        return y

    def model_selection_sfs(self, model: Parametrization, params: dict) -> np.ndarray:
        """
        Infer SFS from the given DFE PDF.
        Note that demography is not included here.

        :param params: Parameters of the DFE
        :param model: DFE parametrization
        :return: SFS counts
        """
        if self.linearized:

            # get discretized DFE
            dfe = model._discretize(params, self.bins) / self.interval_sizes

            # get SFS from DFE using linearization
            # the interval sizes are already included here
            counts_modelled = self.dfe_to_sfs @ dfe
        else:
            dfe_pdf = model.get_pdf(**params)

            def integrate(k: int) -> float:
                """
                Integrate over DFE using scipy's quad.
                Here we use the PDF.
                """

                def integrate_dfe(s: float) -> float:
                    """
                    Integrate over DFE for fixed k.
                    Note that this produces too small results for low s,
                    but we don't make use of it in production code anyway.
                    """
                    return self.get_allele_count_regularized(s, k) * dfe_pdf(s)

                return quad(integrate_dfe, self.bins[0], self.bins[-1])[0]

            counts_modelled = np.array([integrate(k) for k in range(1, self.n)])

        return counts_modelled

    def get_alpha(self, model: Parametrization, params: dict) -> float:
        """
        Get alpha, the proportion of beneficial non-synonymous substitutions.

        :param model: DFE parametrization
        :param params: Parameters of the DFE
        :return: Estimated for alpha
        """
        y = model._discretize(params, self.bins) * H_fixed_regularized(self.s)

        return np.sum(y[self.s > 0]) / np.sum(y)

    def get_interval_density(
            self,
            inter: list | np.ndarray = np.array([-np.inf, -100, -10, -1, 0, 1, np.inf])
    ) -> np.ndarray:
        """
        Get interval density.

        :param inter: Intervals
        :return: Interval density
        """
        return np.array([np.sum((inter[i - 1] < self.s) & (self.s < inter[i])) for i in range(1, len(inter))])
