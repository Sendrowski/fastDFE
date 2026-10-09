"""
Discretization of DFE to SFS transformation.
"""

__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"
__date__ = "2023-02-26"

import logging
from itertools import product
from math import comb
from typing import Literal, Sequence, Tuple, Optional, Callable

import numpy as np
from scipy.integrate import quad, trapezoid
from scipy.special import hyp1f1

from .optimization import parallelize as parallelize_func
from .parametrization import Parametrization

# configure logger
logger = logging.getLogger('fastdfe').getChild('Discretization')


class Discretization:
    """
    Class for discretizing the integral mapping the DFE to the expected SFS.
    """

    #: Mapped dominance coefficient, static to ensure backward compatibility
    h_mapped: float = np.array([0.5])

    #: Parameter h mapping dominance coefficients if not varying, static to ensure backward compatibility
    h: Optional[float] = 0.5

    #: Dominance coefficient callback, static to ensure backward compatibility
    h_callback: Callable[[float, np.ndarray], np.ndarray] = None

    #: Interval bounds for discretizing dominance coefficients, static to ensure backward compatibility
    intervals_h: Tuple[float, float, int] = (0.0, 1.0, 21)

    #: Cache for :meth:`get_counts_fixed` of shape (len(grid_h), n_intervals), static for backward compatibility
    _counts_fixed: Optional[np.ndarray] = None

    def __init__(
            self,
            n: int,
            h: Optional[float] = 0.5,
            h_callback: Callable[[float, np.ndarray], np.ndarray] = lambda h, S: np.full_like(S, h),
            intervals_del: Tuple[float, float, int] = (-1.0e+8, -1.0e-5, 1000),
            intervals_ben: Tuple[float, float, int] = (1.0e-5, 1.0e4, 1000),
            intervals_h: Tuple[float, float, int] = (0.0, 1.0, 21),
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
        :param h: Parameter h mapping dominance coefficients if not varying, else ``None``. When ``None``, dominance
            coefficients are precomputed across ``intervals_h`` and interpolated as needed. Otherwise, only the fixed
            value is precomputed.
        :param h_callback: Callback ``(h, S)`` returning the dominance coefficient for each population-scaled
            selection coefficient ``S``. By default, the constant ``h`` is returned for every ``S``.
        :param intervals_del: ``(start, stop, n_interval)`` for deleterious population-scaled
            selection coefficients. The intervals will be log10-spaced. Decreasing the number of intervals to ``100``
            provides nearly identical results while increasing speed, especially when precomputing across dominance
            coefficients.
        :param intervals_ben: ``(start, stop, n_interval)`` for beneficial population-scaled
            selection coefficients. The intervals will be log10-spaced. Decreasing the number of intervals to ``100``
            provides nearly identical results while increasing speed, especially when precomputing across dominance
            coefficients.
        :param intervals_h: ``(start, stop, n_points)`` for the linearly spaced grid of dominance coefficients.
            This is only used when inferring dominance coefficients. Values of ``h`` between the grid points will be
            interpolated linearly.
        :param n_outer: Number of grid points for outer integrals when computing allele counts for varying
            dominance coefficients.
        :param n_inner: Number of grid points for inner integrals when computing allele counts for varying
            dominance coefficients.
        :param s_chunk_size: Chunk size for S values when precomputing across dominance coefficients.
            This controls memory usage vs. speed trade-off.
        :param integration_mode: ``'midpoint'`` for the trapezoidal rule over the bin edges or ``'quad'`` for
            :func:`scipy.integrate.quad`.
        :param linearized: Whether to discretize and cache the linearized integral mapping DFE to SFS or use
            :func:`scipy.integrate.quad` in each call. ``False`` is slow, intended for validation only and
            deprecated.
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
        self.n: int = int(n)

        #: Parameter h mapping dominance coefficients if not varying
        self.h: float = h

        #: Dominance coefficient callback
        self.h_callback: Callable[[float, np.ndarray], np.ndarray] = h_callback

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

        if not linearized:
            logger.warning(
                "The option linearized=False is slow and intended for validation only. "
                "It is deprecated and will be removed in a future version."
            )
        self.parallelize: bool = parallelize

        # define bins, midpoints and step size
        # these intervals are used when linearizing the integral
        # and when plotting the DFE
        self.bins: np.ndarray = self.get_bins(intervals_del, intervals_ben)
        self.s, self.interval_sizes = self.get_midpoints_and_spacing(self.bins)

        if h is None:
            self.h_mapped: Optional[np.ndarray] = None

            #: Grid over which to precompute dominance coefficients. Either fixed or varying.
            self.grid_h: Optional[np.ndarray] = np.linspace(intervals_h[0], intervals_h[1], int(intervals_h[2]))
        else:
            #: Mapped dominance coefficients if ``h`` is not ``None``.
            self.h_mapped: Optional[np.ndarray] = self.map_h(h, self.bins)

            self.grid_h: Optional[np.ndarray] = None

        # the number of intervals
        self.n_intervals: int = self.s.shape[0]

        # cache for DFE to SFS transformations of shape (n - 1, len(grid_h), n_intervals)
        self._cache: np.ndarray = None

        # cache for get_counts_fixed of shape (len(grid_h), n_intervals)
        self._counts_fixed: Optional[np.ndarray] = None

    @staticmethod
    def default_h_callback(h: float, S: np.ndarray) -> np.ndarray:
        """
        Default callback for mapping dominance coefficients.

        :param h: Dominance coefficient
        :param S: Population-scaled selection coefficients
        :return: Mapped dominance coefficients
        """
        return np.full_like(S, h)

    @staticmethod
    def get_midpoints_and_spacing(bins: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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
            np.log10(np.abs(intervals_del[1])),
            int(intervals_del[2]) + 1
        )

        bins_ben = np.logspace(
            np.log10(intervals_ben[0]),
            np.log10(intervals_ben[1]),
            int(intervals_ben[2]) + 1
        )

        return np.concatenate([bins_del, bins_ben])

    @staticmethod
    def get_counts_high_precision(x, S):
        """
        Allele frequency sojourn times semidominance using high-precision arithmetic which is not supported
        on all platforms.

        :param x: Allele frequency
        :param S: Selection coefficient
        :return: Allele frequency sojourn time
        """
        return ((1 - np.exp(-S * (1 - x), dtype=np.float128)) /
                (x * (1 - x) * (1 - np.exp(-S, dtype=np.float128)))).astype(float)

    @staticmethod
    def get_counts_high_precision_regularized(x, S: float | int | np.ndarray):
        """
        As :meth:`Discretization.get_counts_high_precision()
        <fastdfe.discretization.Discretization.get_counts_high_precision>` but using the limit ``1 / x`` for ``S``
        close to zero.
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
    def get_counts_fixed_semidominant(S: float | int | np.ndarray) -> float | int | np.ndarray:
        """
        The sojourn time as x -> 1  for semidominance.

        :param S: Selection coefficient
        :return: Sojourn time as x -> 1
        """
        return S / (1 - np.exp(-S))

    @staticmethod
    def get_counts_fixed_semidominant_regularized(S: float | int | np.ndarray) -> float | int | np.ndarray:
        """
        As :meth:`Discretization.get_counts_fixed_semidominant()
        <fastdfe.discretization.Discretization.get_counts_fixed_semidominant>` but replacing with the limits close
        to the limit points.

        :param S: Selection coefficient
        :return: Sojourn time as x -> 1
        """
        # make it accept scalars
        if isinstance(S, (float, int)):
            return Discretization.get_counts_fixed_semidominant_regularized(np.array([S], dtype=float))[0]

        # else S is an array
        S = np.asarray(S, dtype=float)
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
        y[remaining] = Discretization.get_counts_fixed_semidominant(S[remaining])

        return y

    @staticmethod
    def get_counts_fixed_dominant(
            S: np.ndarray,
            h: np.ndarray,
            n_bins: int = 1000,
            pow_cluster: int = 4
    ) -> np.ndarray:
        """
        Sojourn times as x -> 1 under selection with specific dominance coefficients.

        :param S: Selection coefficients.
        :param h: Dominance coefficients associated with selection coefficients or single dominance coefficient.
        :param n_bins: Number of grid points for integral over allele frequencies.
        :param pow_cluster: Power for clustering grid points towards 0.
        :return: Array of shape (len(S),) with sojourn times as x -> 1.
        """
        S = np.asarray(S, dtype=float)
        h = np.full_like(S, h) if np.isscalar(h) else np.asarray(h)

        x = np.linspace(0, 1, n_bins) ** pow_cluster  # (n_bins,)

        H2 = h[:, None]  # (P, 1)
        S2 = S[:, None]  # (P, 1)
        xx = x[None, :]  # (1, n_bins)

        exponent = -2 * S2 * H2 * xx - S2 * (1 - 2 * H2) * xx ** 2  # (P, n_bins)

        # stable exponentiation
        m = np.max(exponent, axis=1, keepdims=True)
        integrand = np.exp(exponent - m)  # (P, n_bins)

        Z = trapezoid(integrand, x, axis=1)

        return np.exp(-m[:, 0]) / Z

    def get_counts_fixed(self, h: float) -> np.ndarray:
        """
        Get sojourn times as x -> 1 for given dominance coefficient using precomputed values.
        Interpolates linearly between precomputed values.

        The result depends only on ``h`` and the fixed ``self.s`` bins (not on the DFE parameters),
        so it is precomputed once and reused on every objective evaluation.

        :param h: Parameter h mapping dominance coefficients
        :return: Sojourn times as x -> 1
        :raises ValueError: If ``h`` does not match the fixed ``self.h`` or is outside precomputed range.
        """
        # check for attribute to ensure backwards compatibility
        if not hasattr(self, "_counts_fixed") or self._counts_fixed is None:
            self.precompute_fixed()

        # return single precomputed value if h is fixed
        if self.h is not None:

            if not np.isclose(h, self.h):
                raise ValueError(f'Dominance coefficient {h} does not match precomputed value {self.h}.')

            return self._counts_fixed[0]

        # get surrounding indices and weights
        i, w = self.get_interpolation_weights(h)
        j = np.arange(self.n_intervals)

        # linear interpolation between h values
        return (1.0 - w) * self._counts_fixed[i - 1, j] + w * self._counts_fixed[i, j]

    def precompute_fixed(self, force: bool = False):
        """
        Precompute the sojourn times as x -> 1, possibly across dominance coefficients.

        Builds an array of shape ``(len(grid_h), n_intervals)`` (or ``(1, n_intervals)`` for fixed
        dominance) used by :meth:`get_counts_fixed`. The dominance dimension is sampled at constant
        ``grid_h`` values. Called explicitly (e.g. before pickling to optimization workers) only when
        the divergence flux is actually used, since :class:`Discretization` itself is agnostic to that.
        """
        if (hasattr(self, '_counts_fixed') and self._counts_fixed is not None) and not force:
            return

        if self.h is None:
            self._counts_fixed = np.stack(
                [self.get_counts_fixed_dominant(self.s, g) for g in self.grid_h],
                axis=0
            )
        else:
            self._counts_fixed = self.get_counts_fixed_dominant(self.s, self.map_h(self.h, self.s))[None, :]

    @staticmethod
    def _get_trapezoid_weights(x: np.ndarray) -> np.ndarray:
        """
        Get the weights of the trapezoidal rule along the last axis, so that the integral of ``y`` over ``x`` is
        the sum of ``weights * y`` along that axis.

        :param x: Grid points, increasing along the last axis.
        :return: Weights of the same shape as ``x``.
        """
        spacing = np.diff(x, axis=-1) / 2

        weights = np.zeros_like(x)
        weights[..., :-1] += spacing
        weights[..., 1:] += spacing

        return weights

    @staticmethod
    def _get_counts_dominant(
            n: int,
            k: int | np.ndarray,
            h: float | np.ndarray,
            S: np.ndarray,
            n_outer: int = 1000,
            n_inner: int = 200,
            pow_cluster: int = 4
    ):
        """
        Allele frequency sojourn times under selection with specific dominance coefficients.

        :param n: Sample size.
        :param k: Allele count or array of allele counts.
        :param h: Dominance coefficients associated with selection coefficients or single dominance coefficient.
        :param S: Selection coefficients.
        :param n_outer: Grid size for outer integral over allele frequencies for binomial sampling.
        :param n_inner: Grid size for inner integrals over allele frequencies.
        :param pow_cluster: Power for clustering grid points towards 0.

        :return: Array of allele counts of shape (len(S),) for a single allele count, else (len(k), len(S)).
        """
        S = np.asarray(S, dtype=float)
        h = np.full_like(S, h) if np.isscalar(h) else np.asarray(h)
        K = [int(i) for i in np.atleast_1d(k)]

        # x-grid
        x = np.linspace(0, 1, n_outer) ** pow_cluster  # (n_outer,)

        # binomial term, the only factor that depends on the allele count
        binom = [comb(n, i) * x ** (i - 1) * (1 - x) ** (n - i - 1) for i in K]  # (K, n_outer)

        # result array
        y = np.empty((len(K), S.size), dtype=float)  # (K, P)

        is_deleterious = S <= 0
        if is_deleterious.any():
            S_neg = S[is_deleterious]  # (P1,)
            h_neg = h[is_deleterious]  # (P1,)

            t = np.linspace(0, 1, n_inner)  # (n_inner,)

            # q-grid from x to 1
            xx = x[:, None, None] + (1 - x)[:, None, None] * t[None, None, :]  # (n_outer, 1, n_inner)

            H3 = h_neg[None, :, None]  # (1, P1, 1)
            SN3 = S_neg[None, :, None]  # (1, P1, 1)

            exponent = -2 * SN3 * H3 * xx - SN3 * (1 - 2 * H3) * xx ** 2  # (n_outer, P1, n_inner)

            exponent -= exponent.max(axis=2, keepdims=True)

            norm_q = np.exp(exponent, out=exponent) @ Discretization._get_trapezoid_weights(xx[:, 0, :])[:, :, None]
            norm_q = norm_q[:, :, 0]  # (n_outer, P1)
            del exponent, xx
            norm0 = norm_q[0]  # (P1,)

            H2 = h_neg[None, :]  # (1, P1)
            SN2 = S_neg[None, :]  # (1, P1)

            exponent_main = np.exp(
                2 * SN2 * H2 * x[:, None] +
                SN2 * (1 - 2 * H2) * x[:, None] ** 2
            )  # (n_outer, P1)

            ratio = norm_q / norm0  # (n_outer, P1)

            for i in range(len(K)):
                integrand = binom[i][:, None] * exponent_main * ratio  # (n_outer, P1)

                y[i, is_deleterious] = trapezoid(integrand, x, axis=0)  # (P1,)

        if (~is_deleterious).any():
            S_pos = S[~is_deleterious]  # (P2,)
            h_pos = h[~is_deleterious]  # (P2,)

            xx0 = np.linspace(0, 1, n_inner) ** pow_cluster  # (n_inner,)

            Hn = h_pos[:, None]  # (P2, 1)
            SPn = S_pos[:, None]  # (P2, 1)
            XX0 = xx0[None, :]  # (1, n_inner)

            exponent0 = -2 * SPn * Hn * XX0 - SPn * (1 - 2 * Hn) * XX0 ** 2  # (P2, n_inner)
            norm0 = trapezoid(np.exp(exponent0), xx0, axis=1)  # (P2,)

            t = np.linspace(0, 1, n_inner) ** 5  # (n_inner,)
            xx = x[:, None, None] + (1 - x)[:, None, None] * t[None, None, :]  # (n_outer, 1, n_inner)

            X = x[:, None, None]  # (n_outer, 1, 1)
            H3b = h_pos[None, :, None]  # (1, P2, 1)
            SP3b = S_pos[None, :, None]  # (1, P2, 1)

            term = SP3b * (X - xx) * ((1 - 2 * H3b) * (X + xx) + 2 * H3b)  # (n_outer, P2, n_inner)
            c = np.exp(term, out=term) @ Discretization._get_trapezoid_weights(xx[:, 0, :])[:, :, None]
            c = c[:, :, 0]  # (n_outer, P2)
            del term, xx

            for i in range(len(K)):
                integrand = binom[i][:, None] * c / norm0[None, :]  # (n_outer, P2)

                y[i, ~is_deleterious] = trapezoid(integrand, x, axis=0)  # (P2,)

        return y[0] if np.isscalar(k) else y

    def map_h(self, h: float, S: np.ndarray) -> np.ndarray:
        """
        Map dominance coefficients using ``h_callback``, or the constant default mapping if it is ``None``.

        :param h: Dominance coefficient
        :param S: Population-scaled selection coefficients
        :return: Mapped dominance coefficients
        """
        if self.h_callback is not None:
            return self.h_callback(h, S)

        return self.default_h_callback(h, S)

    def precompute(self, force: bool = False, fixed: bool = False):
        """
        Precompute DFE to SFS transformation, possibly across dominance coefficients.

        :param force: Recompute even if already cached.
        :param fixed: Also precompute the fixed counts (divergence sojourn times) used by
            :meth:`get_counts_fixed`. Caller-controlled because :class:`Discretization` is agnostic
            to whether the divergence flux is used; passing ``True`` caches them before the
            discretization is pickled to optimization workers.
        """
        # build fixed counts (divergence flux) when requested, independently of the _cache guard
        if fixed:
            self.precompute_fixed(force=force)

        if (hasattr(self, '_cache') and self._cache is not None) and not force:
            return

        h_mapped = np.unique(self.h_mapped)

        # compute special case h = 0.5
        if h_mapped.size == 1 and h_mapped[0] == 0.5:
            self._cache = self.get_counts_semidominant()[:, None, :]
            return

        if isinstance(self.h_mapped, np.ndarray) and np.unique(self.h_mapped).size == 1:
            logger.info(f'Precomputing DFE-SFS transformation for fixed h={self.h_mapped[0]}.')
        elif self.h is not None:
            logger.info('Precomputing DFE-SFS transformation for fixed dominance coefficients.')
        else:
            logger.info(
                f'Precomputing DFE-SFS transformation across dominance coefficients (grid size: {len(self.grid_h)}).'
            )

        self._cache = self.get_counts_dominant()

    def get_counts(self, h: float) -> np.ndarray:
        """
        Get DFE to SFS transformation for given dominance coefficient using precomputed values.
        Interpolates linearly between precomputed values.

        :param h: Parameter h mapping dominance coefficients
        :return: Matrix of shape (n - 1, n_intervals), rows indexed by allele count k = 1, ..., n - 1
        :raises ValueError: If ``h`` does not match the fixed ``self.h`` or is outside precomputed range.
        """
        # check for attribute to ensure backwards compatibility
        if not hasattr(self, "_cache") or self._cache is None:
            self.precompute()

        # return single precomputed value if h is fixed
        if self.h is not None:

            if not np.isclose(h, self.h):
                raise ValueError(f'Dominance coefficient {h} does not match precomputed value {self.h}.')

            return self._cache[:, 0, :]

        # get surrounding indices and weights
        i, w = self.get_interpolation_weights(h)
        j = np.arange(self.n_intervals)

        # linear interpolation between h values
        interpolated = (1.0 - w) * self._cache[:, i - 1, j] + w * self._cache[:, i, j]

        return interpolated

    def get_interpolation_weights(self, h: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get indices and weights for interpolating precomputed dominance coefficients across S.

        :param h: Parameter h mapping dominance coefficients
        :return: Indices and weights
        """
        hs = self.map_h(h, self.s)

        # require h within precomputed range
        if hs.min() < self.grid_h.min():
            raise ValueError(f'Dominance coefficient {hs.min()} smaller than precomputed minimum {self.grid_h.min()}.')

        if hs.max() > self.grid_h.max():
            raise ValueError(f'Dominance coefficient {hs.max()} larger than precomputed maximum {self.grid_h.max()}.')

        # find surrounding indices and weights
        i = np.searchsorted(self.grid_h, hs)
        h0, h1 = self.grid_h[i - 1], self.grid_h[i]
        w = (hs - h0) / (h1 - h0)

        return i, w

    def get_counts_semidominant_midpoint(self) -> np.ndarray:
        """
        Precompute linearized integral using the trapezoidal rule over the bin edges.

        :return: Matrix of shape (n - 1, n_intervals), rows indexed by allele count k = 1, ..., n - 1
        """
        logger.info('Precomputing semidominant DFE-SFS transformation using midpoint integration.')

        I = np.ones((self.n - 1, self.n_intervals + 1))

        # we use midpoint integration
        K = np.arange(1, self.n)[:, None] * I
        S = self.bins[None, :] * I

        # retrieve allele counts
        P = np.array([self.get_counts_semidominant_regularized(S[i], K[i]) for i in range(self.n - 1)])

        # take midpoint and multiply by interval size
        return (P[:, :-1] + P[:, 1:]) / 2 * self.interval_sizes

    def get_counts_semidominant_quad(self) -> np.ndarray:
        """
        Precompute linearized integral using :func:`scipy.integrate.quad`.

        :return: Matrix of shape (n - 1, n_intervals), rows indexed by allele count k = 1, ..., n - 1
        """
        logger.info('Precomputing semidominant DFE-SFS transformation using scipy.integrate.quad.')

        # initialize matrix
        P = np.zeros((self.bins.shape[0] - 1, self.n - 1))

        # iterate over bins
        for i in range(self.bins.shape[0] - 1):

            logger.debug(f"Processing interval {(self.bins[i], self.bins[i + 1])}.")

            # iterate over allele count classes
            for j in range(self.n - 1):
                P[i, j] = quad(lambda s: (self.get_counts_semidominant_regularized(
                    s, j + 1)), self.bins[i], self.bins[i + 1])[0]

        return P.T

    def get_counts_semidominant(self) -> np.ndarray:
        """
        Get uncached linearized DFE to SFS transformation for ``h = 0.5`` using special formulae.

        :return: Matrix of shape (n - 1, n_intervals), rows indexed by allele count k = 1, ..., n - 1
        :raises NotImplementedError: If integration mode is not supported.
        """
        # precompute linearized transformation.
        if self.integration_mode == 'midpoint':
            return self.get_counts_semidominant_midpoint()

        if self.integration_mode == 'quad':
            return self.get_counts_semidominant_quad()

        raise NotImplementedError(f'Integration mode {self.integration_mode} not supported.')

    def get_counts_dominant(self) -> np.ndarray:
        """
        Get DFE to SFS transformation for different dominance coefficients.

        :return: Array of shape (n - 1, len(grid_h), n_intervals), or (n - 1, 1, n_intervals) for fixed dominance
        """
        K = np.arange(1, self.n)

        # chunk S-bins
        n_chunks = int(np.ceil(len(self.bins) / self.s_chunk_size))
        S_chunks = np.array_split(self.bins, n_chunks)
        h_mapped_chunks = np.array_split(self.h_mapped, n_chunks) if self.h_mapped is not None else None

        # number of dominance entries
        n_h = self.grid_h.size if self.h is None else 1

        def compute_slice(args):
            """
            Compute allele counts of all frequency classes for given (j, c).

            :param args: Tuple of (j, c) where j is dominance index, c is S-chunk index
            """
            j, c = args

            return self._get_counts_dominant(
                n=self.n,
                k=K,
                S=S_chunks[c],
                h=np.full_like(S_chunks[c], self.grid_h[j]) if self.h is None else h_mapped_chunks[c],
                n_outer=self.n_outer,
                n_inner=self.n_inner
            )

        jc = list(product(
            range(n_h),
            range(n_chunks)
        ))

        P = parallelize_func(
            func=compute_slice,
            data=jc,
            parallelize=self.parallelize,
            desc=f"{self.__class__.__name__}>Precomputing",
            dtype=float,
            wrap_array=False
        )

        P = np.concatenate(P, axis=1)
        P = P.reshape(self.n - 1, n_h, -1)

        return (P[:, :, :-1] + P[:, :, 1:]) / 2 * self.interval_sizes

    def get_counts_semidominant_large_negative_S(
            self,
            S: float | np.ndarray,
            k: float | np.ndarray
    ) -> np.ndarray | float:
        """
        Limit of :meth:`Discretization.get_counts_semidominant_unregularized()
        <fastdfe.discretization.Discretization.get_counts_semidominant_unregularized>` for large negative S.

        :param S: Population-scaled selection coefficient
        :param k: Allele count
        :return: Number of counts in frequency class P(k)
        """
        return self.n / (k * (self.n - k)) * hyp1f1(k, self.n, S)

    def get_counts_semidominant_large_positive_S(
            self,
            S: float | np.ndarray,
            k: float | np.ndarray
    ) -> np.ndarray | float:
        """
        Limit of :meth:`Discretization.get_counts_semidominant_unregularized()
        <fastdfe.discretization.Discretization.get_counts_semidominant_unregularized>` for large positive S.

        :param S: Population-scaled selection coefficient
        :param k: Allele count
        :return: Number of counts in frequency class P(k)
        """
        return self.n / (k * (self.n - k))

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

        With sample size :math:`n` and Kummer's confluent hypergeometric function :math:`M(a, b, z)`,
        the number of counts is

        .. math::
            \\frac{n}{k (n - k)} \\frac{1 - e^{-S} M(k, n, S)}{1 - e^{-S}},

        where :math:`e^{-S} M(k, n, S) = M(n - k, n, -S)` by Kummer's transformation. The expression is evaluated
        in a form that does not overflow for large :math:`|S|` and does not cancel for small :math:`|S|`.

        :param S: Population-scaled selection coefficient
        :param k: Allele count
        :return: Number of counts in frequency class P(k)
        """
        scalar = np.ndim(S) == 0 and np.ndim(k) == 0
        S, k = np.broadcast_arrays(np.atleast_1d(np.asarray(S, dtype=float)), np.atleast_1d(np.asarray(k, dtype=float)))
        n = self.n

        numerator = np.empty(S.shape)
        denominator = np.empty(S.shape)

        # (1 - M(n - k, n, -S)) / (1 - e^{-S})
        positive = S >= 0.1
        numerator[positive] = 1 - hyp1f1(n - k[positive], n, -S[positive])
        denominator[positive] = -np.expm1(-S[positive])

        # (M(k, n, S) - e^S) / (1 - e^S)
        negative = S <= -0.1
        numerator[negative] = hyp1f1(k[negative], n, S[negative]) - np.exp(S[negative])
        denominator[negative] = -np.expm1(S[negative])

        # power series of 1 - M(n - k, n, -S), whose terms decay faster than 0.1^j / j!
        small = ~(positive | negative)
        term = np.ones(int(small.sum()))
        series = np.zeros_like(term)
        for j in range(1, 16):
            term = term * (n - k[small] + j - 1) / (n + j - 1) * -S[small] / j
            series += term

        numerator[small] = -series
        denominator[small] = -np.expm1(-S[small])

        y = n / (k * (n - k)) * numerator / denominator

        return float(y[0]) if scalar else y

    def get_counts_semidominant_regularized(self, S: float | np.ndarray, k: float | np.ndarray) -> float | np.ndarray:
        """
        As :meth:`Discretization.get_counts_semidominant_unregularized()
        <fastdfe.discretization.Discretization.get_counts_semidominant_unregularized>` but using the respective limits
        for ``S`` close to zero and ``S`` very negative.
        Note that ``S`` and ``k`` need to have the same shape.

        :param S: Population-scaled selection coefficient
        :param k: Allele count
        :return: Number of counts in frequency class ``P(k)``
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

            if params.get('h', 0.5) != 0.5:
                raise NotImplementedError('Non-linearized integral currently only implemented for h = 0.5.')

            dfe_pdf = model.get_pdf(**params)

            def integrate(k: int) -> float:
                """
                Integrate over DFE using :func:`scipy.integrate.quad`.
                Here we use the PDF.
                """

                def integrate_dfe(s: float) -> float:
                    """
                    Integrate over DFE for fixed k.
                    """
                    return self.get_counts_semidominant_regularized(s, k) * dfe_pdf(s)

                # integrate over each bin, as the integrand is concentrated around zero
                return sum(quad(integrate_dfe, a, b)[0] for a, b in zip(self.bins[:-1], self.bins[1:]))

            counts_modelled = np.array([integrate(k) for k in range(1, self.n)])

        return counts_modelled

    def get_alpha(self, model: Parametrization, params: dict) -> float:
        """
        Get alpha, the proportion of beneficial non-synonymous substitutions.

        This is the DFE-based (post-hoc) estimate that does not make use of observed divergence
        counts. It corresponds to polyDFE's ``estimateAlpha`` when no divergence data is supplied:
        ``alpha = divBen / (divDel + divBen)``, where ``divDel``/``divBen`` are the expected numbers
        of deleterious/beneficial substitutions integrated over the DFE.

        :param model: DFE parametrization
        :param params: Parameters of the DFE
        :return: Estimate for alpha
        """
        y = model._discretize(params, self.bins) * self.get_counts_fixed(params.get('h', 0.5))

        return np.sum(y[self.s > 0]) / np.sum(y)

    def get_divergence_flux(self, model: Parametrization, params: dict) -> float:
        """
        Get the expected number of selected substitutions per neutral substitution, integrated
        over the DFE. This is ``divDel + divBen`` and equals 1 for a purely neutral DFE. It mirrors
        polyDFE's per-site expected divergence relative to the neutral baseline, where the per
        selection coefficient contribution is ``S / (1 - exp(-S))`` (see :meth:`get_counts_fixed`).

        :param model: DFE parametrization
        :param params: Parameters of the DFE
        :return: Expected selected divergence relative to neutral divergence
        """
        return float(np.sum(model._discretize(params, self.bins) * self.get_counts_fixed(params.get('h', 0.5))))

    def get_alpha_divergence(
            self,
            model: Parametrization,
            params: dict,
            n_div_neut: float,
            n_div_sel: float,
            n_sites_div_neut: float,
            n_sites_div_sel: float
    ) -> float:
        """
        Get alpha from observed divergence counts, mirroring polyDFE's McDonald-Kreitman style
        ``estimateAlpha`` when divergence data is supplied::

            alpha = 1 - (L_sel / L_neut) * (D_neut / D_sel) * divDel

        where ``L`` are the mutational target sizes for divergence (``n_sites_div_neut`` / ``n_sites_div_sel``),
        ``D`` are the observed divergence counts (``n_div_neut`` / ``n_div_sel``), and ``divDel`` is the
        expected number of deleterious (and neutral) substitutions per neutral substitution,
        integrated over the DFE.

        :param model: DFE parametrization
        :param params: Parameters of the DFE
        :param n_div_neut: Observed neutral divergence counts
        :param n_div_sel: Observed selected divergence counts
        :param n_sites_div_neut: Neutral mutational target size for divergence
        :param n_sites_div_sel: Selected mutational target size for divergence
        :return: Estimate for alpha based on divergence counts
        """
        y = model._discretize(params, self.bins) * self.get_counts_fixed(params.get('h', 0.5))

        # expected deleterious (and neutral) divergence relative to the neutral baseline
        div_del = np.sum(y[self.s <= 0])

        return float(1 - (n_sites_div_sel / n_sites_div_neut) *
                     (n_div_neut / n_div_sel) * div_del)

    def get_omega(self, model: Parametrization, params: dict) -> float:
        """
        Get omega, the rate of non-synonymous over synonymous substitutions (dN/dS), integrated
        over the DFE. This is the DFE-based (post-hoc) estimate that does not make use of observed
        divergence counts. It equals the expected number of selected substitutions per neutral
        substitution, i.e. ``divDel + divBen`` (cf. :meth:`get_divergence_flux`), and is 1 for a
        purely neutral DFE.

        :param model: DFE parametrization
        :param params: Parameters of the DFE
        :return: Estimate for omega (dN/dS)
        """
        y = model._discretize(params, self.bins) * self.get_counts_fixed(params.get('h', 0.5))

        return float(np.sum(y))

    def get_omega_a(self, model: Parametrization, params: dict) -> float:
        """
        Get omega_a, the rate of adaptive non-synonymous over synonymous substitutions
        (adaptive dN/dS), integrated over the DFE. This is the DFE-based (post-hoc) estimate that
        does not make use of observed divergence counts and equals ``alpha * omega``, i.e. the
        expected number of beneficial substitutions per neutral substitution (``divBen``).

        :param model: DFE parametrization
        :param params: Parameters of the DFE
        :return: Estimate for omega_a (adaptive dN/dS)
        """
        y = model._discretize(params, self.bins) * self.get_counts_fixed(params.get('h', 0.5))

        return float(np.sum(y[self.s > 0]))

    def get_omega_divergence(
            self,
            model: Parametrization,
            params: dict,
            n_div_neut: float,
            n_div_sel: float,
            n_sites_div_neut: float,
            n_sites_div_sel: float
    ) -> float:
        """
        Get omega (dN/dS) from observed divergence counts, mirroring polyDFE's McDonald-Kreitman
        style estimate when divergence data is supplied::

            omega = (D_sel / L_sel) / (D_neut / L_neut)

        where ``L`` are the mutational target sizes for divergence (``n_sites_div_neut`` / ``n_sites_div_sel``)
        and ``D`` are the observed divergence counts (``n_div_neut`` / ``n_div_sel``).

        :param model: DFE parametrization
        :param params: Parameters of the DFE
        :param n_div_neut: Observed neutral divergence counts
        :param n_div_sel: Observed selected divergence counts
        :param n_sites_div_neut: Neutral mutational target size for divergence
        :param n_sites_div_sel: Selected mutational target size for divergence
        :return: Estimate for omega (dN/dS) based on divergence counts
        """
        return float((n_div_sel / n_sites_div_sel) /
                     (n_div_neut / n_sites_div_neut))

    def get_omega_a_divergence(
            self,
            model: Parametrization,
            params: dict,
            n_div_neut: float,
            n_div_sel: float,
            n_sites_div_neut: float,
            n_sites_div_sel: float
    ) -> float:
        """
        Get omega_a (adaptive dN/dS) from observed divergence counts, mirroring polyDFE's
        McDonald-Kreitman style estimate when divergence data is supplied::

            omega_a = alpha * omega

        where ``alpha`` is the MK-style estimate (:meth:`get_alpha_divergence`) and ``omega`` the
        observed dN/dS (:meth:`get_omega_divergence`).

        :param model: DFE parametrization
        :param params: Parameters of the DFE
        :param n_div_neut: Observed neutral divergence counts
        :param n_div_sel: Observed selected divergence counts
        :param n_sites_div_neut: Neutral mutational target size for divergence
        :param n_sites_div_sel: Selected mutational target size for divergence
        :return: Estimate for omega_a (adaptive dN/dS) based on divergence counts
        """
        alpha = self.get_alpha_divergence(model, params, n_div_neut, n_div_sel, n_sites_div_neut, n_sites_div_sel)
        omega = self.get_omega_divergence(model, params, n_div_neut, n_div_sel, n_sites_div_neut, n_sites_div_sel)

        return float(alpha * omega)

    def get_interval_density(
            self,
            inter: Sequence = np.array([-np.inf, -100, -10, -1, 0, 1, np.inf])
    ) -> np.ndarray:
        """
        Get the number of grid midpoints lying strictly inside each of the given intervals.

        :param inter: Interval edges
        :return: Number of grid midpoints per interval
        """
        return np.array([np.sum((inter[i - 1] < self.s) & (self.s < inter[i])) for i in range(1, len(inter))])

    def __eq__(self, other) -> bool:
        """
        Compare two instances for equality.

        :param other: Other Discretization instance
        :return: True if equal, False otherwise
        """
        if not isinstance(other, Discretization):
            return False

        return (
                self.n == other.n and
                self.h == other.h and
                np.array_equal(self.h_mapped, other.h_mapped) and
                self.intervals_del == other.intervals_del and
                self.intervals_ben == other.intervals_ben and
                self.intervals_h == other.intervals_h and
                self.n_outer == other.n_outer and
                self.n_inner == other.n_inner and
                self.s_chunk_size == other.s_chunk_size and
                self.integration_mode == other.integration_mode and
                self.linearized == other.linearized and
                self.parallelize == other.parallelize
        )

    def __hash__(self) -> int:
        """
        Hash instance.
        """
        return hash((
            self.n,
            self.h,
            tuple(self.intervals_del),
            tuple(self.intervals_ben),
            tuple(self.intervals_h),
            self.n_outer,
            self.n_inner,
            self.s_chunk_size,
            self.integration_mode,
            self.linearized,
            self.parallelize,
        ))
