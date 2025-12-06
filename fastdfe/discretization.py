"""
Discretization of DFE to SFS transformation.
"""

__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"
__date__ = "2023-02-26"

import logging
from functools import wraps
from itertools import product
from math import comb
from typing import Literal, Sequence, Tuple, Optional

import mpmath as mp
import numpy as np
from scipy.integrate import quad

from .optimization import parallelize as parallelize_func
from .parametrization import Parametrization

# configure logger
logger = logging.getLogger('fastdfe').getChild('Discretization')


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
    Class for discretizing the integral mapping the DFE to the expected SFS.
    """

    #: Dominance coefficient, static to ensure backward compatibility
    h: float = 0.5

    #: Grid of dominance coefficients, static to ensure backward compatibility
    grid_h: np.ndarray = np.array([0.5])

    def __init__(
            self,
            n: int,
            h: Optional[float] = 0.5,
            intervals_del: Tuple[float, float, int] = (-1.0e+8, -1.0e-5, 1000),
            intervals_ben: Tuple[float, float, int] = (1.0e-5, 1.0e4, 1000),
            intervals_h: Tuple[float, float, int] = (0.0, 1.0, 100),
            n_outer: int = 1000,
            n_inner: int = 200,
            s_chunk_size: int = 10,
            integration_mode: Literal['midpoint', 'quad'] = 'midpoint',
            linearized: bool = True,
            parallelize: bool = True,
    ):
        """
        Create Discretization instance.

        :param n: Number of individuals in the SFS sample.
        :param h: Dominance coefficient if not varying, else `None`. When `None`, `h` is precomputed across
            `intervals_h` and interpolated as needed.
        :param intervals_del: ``(start, stop, n_interval)`` for deleterious population-scaled selection coefficients.
        :param intervals_ben: ``(start, stop, n_interval)`` for beneficial population-scaled selection coefficients.
        :param intervals_h: ``(start, stop, n_interval)`` for dominance coefficients.
        :param n_outer: Number of grid points for outer integrals when computing allele counts for varying
            dominance coefficients.
        :param n_inner: Number of grid points for inner integrals when computing allele counts for varying
            dominance coefficients.
        :param s_chunk_size: Chunk size for S values when precomputing across dominance coefficients.
            This controls memory usage vs. speed trade-off.
        :param integration_mode : 'midpoint' or 'quad' for midpoint integration or Scipy's quad method.
        :param linearized: Whether to use linearized integral or compute integral numerically in each iteration.
        :param parallelize: Whether to parallelize the computation of the discretization.
        """
        # make sure lower bounds are lower than upper bounds
        if not intervals_del[0] < intervals_del[1] or not intervals_ben[0] < intervals_ben[1]:
            raise Exception('Lower intervals bounds must be lower than upper bounds.')

        # make sure |S| is not too large
        if intervals_del[0] < -1e10 or intervals_ben[1] > 1e10:
            raise Exception('Bounds for S should within the interval [-1e10, 1e10] to avoid '
                            'unexpected numerical behavior.')

        #: SFS sample size
        self.n: int = n

        #: Dominance coefficient
        self.h: float = h

        # interval bounds for discretizing DFE
        self.intervals_del: Tuple[float, float, int] = intervals_del
        self.intervals_ben: Tuple[float, float, int] = intervals_ben

        # interval bounds for discretizing dominance coefficients
        self.intervals_h: Tuple[float, float, int] = intervals_h

        # grid sizes for integrals when computing allele counts for varying dominance coefficients
        self.n_outer: int = n_outer
        self.n_inner: int = n_inner

        # chunk size for S values when precomputing across dominance coefficients
        self.s_chunk_size: int = s_chunk_size

        self.integration_mode: Literal['midpoint', 'quad'] = integration_mode
        self.linearized: bool = linearized
        self.parallelize: bool = parallelize

        # iteration counter
        self.n_it: int = 0

        # define bins, midpoints and step size
        # these intervals are used when linearizing the integral
        # and when plotting the DFE
        self.bins: np.ndarray = self.get_bins(intervals_del, intervals_ben)
        self.s, self.interval_sizes = self.get_midpoints_and_spacing(self.bins)

        # grid of dominance coefficients
        if h is None:
            self.grid_h = np.linspace(*self.intervals_h)
        else:
            self.grid_h = np.array([h])

        # the number of intervals
        self.n_intervals: int = self.s.shape[0]

        # cache for DFE to SFS transformations of shape (len(grid_h), n, n_intervals)
        self._cache: np.ndarray = None

    @staticmethod
    def exp(x: float | np.ndarray) -> float | np.ndarray:
        """
        Vectorized version of mpmath's exp function.

        :param x: Exponent
        :return: Result
        """
        return np.vectorize(mp.exp)(x)

    @staticmethod
    def hyp1f1(a: float | np.ndarray, b: float | np.ndarray, z: float | np.ndarray) -> float | np.ndarray:
        """
        Vectorized version of mpmath's hyp1f1 function.

        :param a: First parameter
        :param b: Second parameter
        :param z: Third parameter
        :return: Result
        """
        return np.vectorize(mp.hyp1f1)(a, b, z)

    @staticmethod
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

    @staticmethod
    def get_bins(intervals_del: tuple, intervals_ben: tuple) -> np.ndarray:
        """
        Get bins using log-spaced distances for positive and negative
        values of S given by ``intervals_ben`` and ``intervals_del``, respectively.

        :param intervals_del: ``(min, max, n)`` for negative values of S where we use log-spaced distances
        :param intervals_ben: ``(min, max, n)`` for positive values of S where we use log-spaced distances
        :return: Bins
        """
        bins_del = -np.logspace(
            np.log10(np.abs(intervals_del[0])),
            np.log10(np.abs(intervals_del[1])), intervals_del[2] + 1
        )

        bins_ben = np.logspace(
            np.log10(intervals_ben[0]),
            np.log10(intervals_ben[1]), intervals_ben[2] + 1
        )

        return np.concatenate([bins_del, bins_ben])

    @staticmethod
    def get_counts_high_precision(x, S):
        """
        Allele frequency sojourn times semidominance using high-precision arithmetic which is not supported
        on all platforms.

        :param S: Selection coefficient
        :param x: Allele frequency
        :return: Allele frequency sojourn time
        """
        return ((1 - np.exp(-S * (1 - x), dtype=np.float128)) /
                (x * (1 - x) * (1 - np.exp(-S, dtype=np.float128)))).astype(float)

    @staticmethod
    def get_counts_high_precision_regularized(x, S: float | int | np.ndarray):
        """
        As H(x, S) but replacing with the limits close to the limit points.
        Note that this function is not used in practice.

        :param x: Allele frequency
        :param S: Selection coefficient
        :return: Allele frequency sojourn time
        """
        # make it accept scalars
        if isinstance(S, (float, int)):
            return Discretization.get_counts_high_precision_regularized(x, np.array([S]))[0]

        # else S is an array
        y = np.zeros_like(S, dtype=np.float128)

        # S close to zero
        close_to_zero = np.abs(S) < 1e-8

        # limit (1 - exp(-S * (1 - x))) / (x * (1 - x) * (1 - exp(-S))) as S -> 0
        # cf. https://www.wolframalpha.com/input?i=limit+%281+-+exp%28-S+*+%281+-+x%29%29%29+%2F+%
        # 28x+*+%281+-+x%29+*+%281+-+exp%28-S%29%29%29+as+S+-%3E+0
        y[close_to_zero] = 1 / x

        # evaluate function as usual
        y[~close_to_zero] = Discretization.get_counts_high_precision(x, S[~close_to_zero])

        return y

    @staticmethod
    def get_counts_fixed(S: float | int | np.ndarray) -> float | int | np.ndarray:
        """
        The sojourn time as x -> 1  for semidominance.

        :param S: Selection coefficient
        :return: Sojourn time as x -> 1
        """
        return S / (1 - np.exp(-S))

    @staticmethod
    def get_counts_fixed_regularized(S: float | int | np.ndarray) -> float | int | np.ndarray:
        """
        As :func:`H_fixed` but replacing with the limits close to the limit points.

        :param S: Selection coefficient
        :return: Sojourn time as x -> 1
        """
        # make it accept scalars
        if isinstance(S, (float, int)):
            return Discretization.get_counts_fixed(np.array([S]))[0]

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
        y[remaining] = Discretization.get_counts_fixed(S[remaining])

        return y

    @staticmethod
    def _get_counts_dominant(
            n: int,
            k: int,
            h: np.ndarray,
            S: np.ndarray,
            n_outer: int = 1000,
            n_inner: int = 200,
            pow_cluster: int = 4
    ):
        """
        Allele frequency sojourn times under selection with specific dominance coefficients.

        :param n: Sample size.
        :param k: Allele count.
        :param h: Dominance coefficients.
        :param S: Selection coefficients.
        :param n_outer: Grid size for outer integral over allele frequencies for binomial sampling.
        :param n_inner: Grid size for inner integrals over allele frequencies.
        :param pow_cluster: Power for clustering grid points towards 0.

        :return: Array of shape (len(h), len(S)) with allele counts.
        """
        h = np.asarray(h)
        S = np.asarray(S)

        # x-grid (same as your scalar code)
        x = np.linspace(0, 1, n_outer) ** pow_cluster  # (n_outer,)

        # binomial term
        binom = comb(n, k) * x ** (k - 1) * (1 - x) ** (n - k - 1)  # (n_outer,)

        # result array
        y = np.empty((h.size, S.size), dtype=float)  # (H, P)

        is_deleterious = S <= 0
        if is_deleterious.any():
            S_neg = S[is_deleterious]  # (P1,)
            p1 = S_neg.size

            t = np.linspace(0, 1, n_inner)  # (n_inner,)

            # q-grid from x to 1
            xx = x[:, None, None, None] + (1 - x)[:, None, None, None] * t[None, None, None, :]
            xx = np.broadcast_to(xx, (n_outer, h.size, p1, n_inner))  # (n_outer, H, P1, n_inner)

            H4 = h[None, :, None, None]  # (1, H, 1, 1)
            SN4 = S_neg[None, None, :, None]  # (1, 1, P1, 1)

            exponent = -2 * SN4 * H4 * xx - SN4 * (1 - 2 * H4) * xx ** 2  # (n_outer, H, P1, n_inner)

            max_exp = exponent.max(axis=(0, 3), keepdims=True)  # (1, H, P1, 1)

            norm_q = np.trapz(np.exp(exponent - max_exp), xx, axis=3)  # (n_outer, H, P1)
            del exponent, xx
            norm0 = norm_q[0]  # (H, P1)

            H3 = h[None, :, None]  # (1, H, 1)
            SN3 = S_neg[None, None, :]  # (1, 1, P1)

            exponent_main = np.exp(
                2 * SN3 * H3 * x[:, None, None] +
                SN3 * (1 - 2 * H3) * x[:, None, None] ** 2
            )  # (n_outer, H, P1)

            ratio = norm_q / norm0  # (n_outer, H, P1)

            integrand = binom[:, None, None] * exponent_main * ratio  # (n_outer, H, P1)

            y[:, is_deleterious] = np.trapz(integrand, x, axis=0)  # (H, P1)

        if (~is_deleterious).any():
            S_pos = S[~is_deleterious]  # (P2,)
            p2 = S_pos.size

            xx0 = np.linspace(0, 1, n_inner) ** pow_cluster  # (n_inner,)

            Hn = h[:, None, None]  # (H, 1, 1)
            SPn = S_pos[None, :, None]  # (1, P2, 1)
            XX0 = xx0[None, None, :]  # (1, 1, n_inner)

            exponent0 = -2 * SPn * Hn * XX0 - SPn * (1 - 2 * Hn) * XX0 ** 2  # (H, P2, n_inner)
            norm0 = np.trapz(np.exp(exponent0), xx0, axis=2)  # (H, P2)

            t = np.linspace(0, 1, n_inner) ** 5  # (n_inner,)
            xx = x[:, None, None, None] + (1 - x)[:, None, None, None] * t[None, None, None, :]
            xx = np.broadcast_to(xx, (n_outer, h.size, p2, n_inner))  # (n_outer, H, P2, n_inner)

            X = x[:, None, None, None]  # (n_outer, 1, 1, 1)
            H4b = h[None, :, None, None]  # (1, H, 1, 1)
            SP4b = S_pos[None, None, :, None]  # (1, 1, P2, 1)

            term = SP4b * (X - xx) * ((1 - 2 * H4b) * (X + xx) + 2 * H4b)  # (n_outer, H, P2, n_inner)
            c = np.trapz(np.exp(term), xx, axis=3)  # (n_outer, H, P2)
            del term, xx

            integrand = binom[:, None, None] * c / norm0  # (n_outer, H, P2)

            y[:, ~is_deleterious] = np.trapz(integrand, x, axis=0)  # (H, P2)

        return y

    def precompute(self, force: bool = False):
        """
        Precompute DFE to SFS transformation, possibly across dominance coefficients.
        """
        if self._cache is not None and not force:
            return

        # compute special case h = 0.5
        if self.h == 0.5:
            self._cache = self.get_counts_semidominant()[:, None, :]
            return

        if self.h is None:
            logger.info('Precomputing DFE-SFS transformation across dominance coefficients.')
        else:
            logger.info(f'Precomputing DFE-SFS transformation for fixed h={self.h}.')

        self._cache = self.get_counts_dominant_grid(self.grid_h)

    def get_counts(self, h: float) -> np.ndarray:
        """
        Get DFE to SFS transformation for given dominance coefficient using precomputed values.
        Interpolates linearly between precomputed values.

        :param h: Dominance coefficient
        :return: Matrix of size (n_intervals, n)
        """
        # check for attribute to ensure backwards compatibility
        if not hasattr(self, "_cache") or self._cache is None:
            self.precompute()

        if len(self.grid_h) == 1:
            return self._cache[:, 0, :]

        # require h within precomputed range
        if not (self.grid_h[0] <= h <= self.grid_h[-1]):
            raise ValueError(f"h={h} is outside precomputed dominance range [{self.grid_h[0]}, {self.grid_h[-1]}].")

        # find surrounding indices and weights
        idx = np.searchsorted(self.grid_h, h)
        h0, h1 = self.grid_h[idx - 1], self.grid_h[idx]
        w = (h - h0) / (h1 - h0)

        # linear interpolation
        interpolated = (1.0 - w) * self._cache[:, idx - 1, :] + w * self._cache[:, idx, :]

        return interpolated

    def get_counts_semidominant_midpoint(self) -> np.ndarray:
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
            return self.get_counts_semidominant_regularized(S[i], K[i])

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

    def get_counts_semidominant_quad(self) -> np.ndarray:
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
                P[i, j] = quad(lambda s: (self.get_counts_semidominant_regularized(
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

    def get_counts_semidominant(self) -> np.ndarray:
        """
        Get uncached linearized DFE to SFS transformation for `h = 0.5` using special formulae.

        :return: Matrix of size (n_intervals, n)
        :raises NotImplementedError: If integration mode is not supported.
        """
        # precompute linearized transformation.
        if self.integration_mode == 'midpoint':
            return self.get_counts_semidominant_midpoint()

        if self.integration_mode == 'quad':
            return self.get_counts_semidominant_quad()

        raise NotImplementedError(f'Integration mode {self.integration_mode} not supported.')

    def get_counts_dominant(self, h: float) -> np.ndarray:
        """
        Get uncached DFE to SFS transformation for given dominance coefficient.

        :return: Matrix of size (n_intervals, n)
        """
        return self.get_counts_dominant_grid(np.array([h]))[:, 0, :]

    def get_counts_dominant_grid(self, h_grid: np.ndarray) -> np.ndarray:
        """
        Get DFE to SFS transformation for different dominance coefficients.

        :return: Matrix of size (n_intervals, n)
        """
        K = np.arange(1, self.n)

        # chunk S-bins
        S_chunks = np.array_split(self.bins, int(np.ceil(len(self.bins) / self.s_chunk_size)))

        def compute_slice(args):
            """
            Compute allele counts for given (i, j, c).

            :param args: Tuple of (i, j, c) where i is allele count index, j is dominance index, c is S-chunk index
            :return:
            """
            i, j, c = args
            return self._get_counts_dominant(
                n=self.n,
                k=K[i],
                S=S_chunks[c],
                h=[h_grid[j]],
                n_outer=self.n_outer,
                n_inner=self.n_inner
            )

        ijc = product(
            range(self.n - 1),
            range(len(h_grid)),
            range(len(S_chunks))
        )

        P = parallelize_func(
            func=compute_slice,
            data=list(ijc),
            parallelize=self.parallelize,
            desc=f"{self.__class__.__name__}>Precomputing",
            dtype=float,
            wrap_array=False
        )

        P = np.concatenate(P, axis=1)
        P = P.reshape(self.n - 1, len(h_grid), -1)

        return (P[:, :, :-1] + P[:, :, 1:]) / 2 * self.interval_sizes

    @to_float
    def get_counts_semidominant_large_negative_S(
            self,
            S: float | np.ndarray,
            k: float | np.ndarray
    ) -> np.ndarray | float:
        """
        Limit of get_allele_count for large negative S.

        :param S: Population-scaled selection coefficient
        :param k: Allele count
        :return: Number of counts in frequency class P(k)
        """
        return self.n / (k * (self.n - k)) * self.hyp1f1(k, self.n, S)

    def get_counts_semidominant_large_positive_S(
            self,
            S: float | np.ndarray,
            k: float | np.ndarray
    ) -> np.ndarray | float:
        """
        Limit of get_allele_count for large positive S.

        :param S: Population-scaled selection coefficient
        :param k: Allele count
        :return: Number of counts in frequency class P(k)
        """
        return self.n / (k * (self.n - k))

    @to_float
    def get_counts_semidominant_unregularized(
            self,
            S: float | np.ndarray,
            k: float | np.ndarray
    ) -> np.ndarray | float:
        """
        The number of counts in frequency class P(k) with
        population-scaled selection coefficient S and h = 0.5.
        Binomial sampling is included here.
        See appendix of https://pubmed.ncbi.nlm.nih.gov/31975166

        :param S: Population-scaled selection coefficient
        :param k: Allele count
        :return: Number of counts in frequency class P(k)
        """
        return (self.n / (k * (self.n - k))) * ((1 - self.exp(-S) * self.hyp1f1(k, self.n, S)) / (1 - self.exp(-S)))

    def get_counts_semidominant_regularized(self, S: float | np.ndarray, k: float | np.ndarray) -> float | np.ndarray:
        """
        As :meth:`get_counts_semidominant` but using the respective limits
        for ``S`` close to zero and ``S`` very negative.
        Note that ``S`` and ``k`` need to have the same shape.

        :param S: Population-scaled selection coefficient
        :param k: Allele count
        :return: Number of counts in frequency class ``P(k)`
        """
        # make it accept scalars for S
        if isinstance(S, (float, int)):
            return self.get_counts_semidominant_regularized(np.array([S]), np.array([k]))[0]

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
            y[very_negative] = self.get_counts_semidominant_large_negative_S(S[very_negative], k[very_negative])

        # remaining values of S
        remaining = np.array(~(close_to_zero | very_negative))

        # evaluate function as usual for remaining values
        if remaining.any():
            y[remaining] = self.get_counts_semidominant_unregularized(S[remaining], k[remaining])

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
            counts_modelled = self.get_counts(params.get('h', 0.5)) @ dfe
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
                    return self.get_counts_semidominant_regularized(s, k) * dfe_pdf(s)

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
        y = model._discretize(params, self.bins) * self.get_counts_fixed_regularized(self.s)

        return np.sum(y[self.s > 0]) / np.sum(y)

    def get_interval_density(
            self,
            inter: Sequence = np.array([-np.inf, -100, -10, -1, 0, 1, np.inf])
    ) -> np.ndarray:
        """
        Get interval density.

        :param inter: Intervals
        :return: Interval density
        """
        return np.array([np.sum((inter[i - 1] < self.s) & (self.s < inter[i])) for i in range(1, len(inter))])
