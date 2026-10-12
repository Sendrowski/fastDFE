"""
Bootstrap utilities.
"""

__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"
__date__ = "2023-02-26"

import logging
import math
from typing import Literal, Sequence, Tuple

import numpy as np
from scipy.stats import norm as normal

# get logger
logger = logging.getLogger('fastdfe').getChild('Bootstrap')


class Bootstrap:
    """
    Bootstrap utilities.
    """

    @staticmethod
    def get_bounds_from_quantile(data: Sequence, a1: float, a2: float, n: int) -> Tuple[float, float]:
        """
        Get confidence interval bounds.

        :param data: Sorted data
        :param a1: Lower quantile
        :param a2: Upper quantile
        :param n: Number of data points
        :return: lower bound and upper bound, ``[None, None]`` if there are no data or ``a1`` or ``a2`` is NaN
        """
        if n == 0 or np.isnan(a1) or np.isnan(a2):
            return [None, None]

        # as many values lie below the lower bound as a1 * n, and above the upper bound as (1 - a2) * n. Halves are
        # rounded towards the centre in both tails, after removing the rounding error of the products
        i1 = math.floor(round(a1 * n, 9) + 0.5)
        i2 = math.ceil(round(a2 * n, 9) - 0.5) - 1

        i1, i2 = (min(max(i, 0), n - 1) for i in (i1, i2))

        return data[i1], data[max(i1, i2)]

    @staticmethod
    def get_ci_percentile(bootstraps: Sequence, a: float) -> Tuple[float, float]:
        """
        Get the confidence interval spanning the a to 1 - a quantiles using the percentile bootstrap.

        :param bootstraps: List of bootstraps
        :param a: Tail probability per side, so 0.05 yields a 90% interval.
        :return: lower bound and upper bound
        :raises ValueError: If ``a`` is outside ``[0, 0.5]``.
        """
        if not 0 <= a <= 0.5:
            raise ValueError('Tail probability must be between 0 and 0.5.')

        return Bootstrap.get_bounds_from_quantile(np.sort(bootstraps), a, 1 - a, len(bootstraps))

    @staticmethod
    def get_ci_bca(bootstraps: Sequence, original: float, a: float) -> Tuple[float, float]:
        """
        Get the bias-corrected confidence interval spanning the a to 1 - a quantiles, i.e. the BCa interval with
        the acceleration set to zero, cf. An Introduction to the Bootstrap, Bradley Efron, Robert J. Tibshirani,
        section 14.3. The bias correction is the standard normal quantile of the fraction of bootstraps below the
        original value, counting ties as one half.

        :param bootstraps: List of bootstraps
        :param original: Original value
        :param a: Tail probability per side, so 0.05 yields a 90% interval.
        :return: lower bound and upper bound, ``[None, None]`` if no bootstraps are given or the adjusted
            quantiles are NaN
        """
        n = len(bootstraps)

        if n == 0:
            return [None, None]

        data = np.sort(np.array(bootstraps))

        # keep the fraction within (0, 1), where the quantile is finite
        eps = np.finfo(float).eps
        below = (np.sum(data < original) + 0.5 * np.sum(data == original)) / n
        z0_hat = normal.ppf(np.clip(below, eps, 1 - eps))
        z_a = normal.ppf(a)

        a1 = normal.cdf(2 * z0_hat + z_a)
        a2 = normal.cdf(2 * z0_hat - z_a)

        return Bootstrap.get_bounds_from_quantile(data, a1, a2, n)

    @staticmethod
    def get_errors(
            values: Sequence,
            bs: np.ndarray,
            ci_level: float = 0.05,
            bootstrap_type: Literal['percentile', 'bca'] = 'percentile',
            point_estimate: Literal['original', 'mean', 'median'] = 'original',
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get error values and confidence intervals from the list of original
        values and its bootstraps for a list of parameters.

        :param values: The original values.
        :param bs: Array containing bootstraps as first and the values as second dimension.
        :param ci_level: Tail probability per side, so 0.05 yields a 90% interval.
        :param bootstrap_type: The bootstrap type.
        :param point_estimate: Whether to use 'original' MLE values, 'mean' or 'median' of bootstraps as point estimate.
        :return: Center values, errors around center, and confidence intervals.
        :raises NotImplementedError: If ``bootstrap_type`` is neither ``'percentile'`` nor ``'bca'``.
        """
        # number of values
        n_values = len(values)

        # use mean of bootstraps instead of original values
        if point_estimate == 'mean':
            center = np.mean(bs, axis=0)
        elif point_estimate == 'median':
            center = np.median(bs, axis=0)
        else:
            center = np.array(values)

        # determine confidence intervals and errors on y axis
        if bootstrap_type == 'percentile':
            cis = np.array([Bootstrap.get_ci_percentile(bs[:, i], ci_level) for i in range(n_values)]).T

        elif bootstrap_type == 'bca':
            cis = np.array([Bootstrap.get_ci_bca(bs[:, i], values[i], ci_level) for i in range(n_values)]).T
        else:
            raise NotImplementedError(f"Bootstrap type {bootstrap_type} not supported.")

        # raise warning if some confidence intervals could not be determined
        if np.sum(np.equal(cis, None)) > 0:
            logger.warning('Some confidence intervals could not be computed.')

            # set undefined confidence intervals to values
            mask = np.equal(cis, None)
            for j in range(cis.shape[1]):
                if mask[:, j].any():
                    cis[mask[:, j], j] = values[j]

        # determine error relative to center
        errors = np.array([center - cis[0], cis[1] - center])

        # issue notice if some errors are negative
        if np.sum(np.less(errors, 0)) > 0:
            logger.debug('Some computed errors were negative and were adjusted to 0.')

            # set negative errors to 0
            errors[errors < 0] = 0

        return center, errors, cis
