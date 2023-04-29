"""
Bootstrap utilities.
"""

__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"
__date__ = "2023-02-26"

import logging
from typing import Literal

import numpy as np
from scipy.stats import norm as normal

# get logger
logger = logging.getLogger('fastdfe')


class Bootstrap:
    """
    Bootstrap utilities.
    """

    @staticmethod
    def get_bounds_from_quantile(data: list | np.ndarray, a1: float, a2: float, n: int) -> (float, float):
        """
        Get confidence interval bounds.

        :param data: Sorted data
        :param a1: Lower quantile
        :param a2: Upper quantile
        :param n: Number of data points
        :return: lower bound and upper bound
        """
        if np.isnan(a1) or np.isnan(a2):
            return [None, None]

        return data[max(round(a1 * n), 0)], data[min(round(a2 * n), n - 1)]

    @staticmethod
    def get_ci_percentile(bootstraps: list | np.ndarray, a: float) -> (float, float):
        """
        Get the (1 - a)% confidence intervals using the percentile bootstrap.

        :param bootstraps: List of bootstraps
        :param a: Confidence level
        :return: lower bound and upper bound
        """
        return Bootstrap.get_bounds_from_quantile(np.sort(bootstraps), a, 1 - a, len(bootstraps))

    @staticmethod
    def get_ci_bca(bootstraps: list | np.ndarray, original: float, a: float) -> (float, float):
        """
        Get the (1 - a)% confidence intervals using the BCa method.
        cf. An Introduction to the Bootstrap, Bradley Efron, Robert J. Tibshirani, section 14.2.

        :param bootstraps: List of bootstraps
        :param original: Original value
        :param a: Confidence level
        :return: lower bound and upper bound
        """
        if sum(bootstraps) == 0:
            return [0, 0]

        n = len(bootstraps)
        data = np.sort(np.array(bootstraps))

        theta_hat_i = np.array([np.var(np.delete(data, i)) for i in range(n)])
        theta_hat = sum(theta_hat_i) / n

        a_hat = sum((theta_hat - theta_hat_i) ** 3) / (6 * (sum((theta_hat - theta_hat_i) ** 2)) ** (3 / 2))

        # we add epsilon here to avoid getting -inf when sum(data < original) / n is 0
        z0_hat = normal.ppf(sum(data < original) / n + np.finfo(float).eps)
        z_a = normal.ppf(a)

        a1 = normal.cdf(z0_hat + (z0_hat + z_a) / (1 - a_hat * (z0_hat + z_a)))
        a2 = normal.cdf(z0_hat + (z0_hat - z_a) / (1 - a_hat * (z0_hat - z_a)))

        return Bootstrap.get_bounds_from_quantile(data, a1, a2, n)

    @staticmethod
    def get_errors(
            values: list | np.ndarray,
            bs: np.ndarray,
            ci_level: float = 0.05,
            bootstrap_type: Literal['percentile', 'bca'] = 'percentile'
    ) -> (np.ndarray, np.ndarray):
        """
        Get error values and confidence intervals from the list of original
        values and its bootstraps for a list of parameters.

        :param values: The original values
        :param bs: The bootstraps
        :param ci_level: The confidence level
        :param bootstrap_type: The bootstrap type
        :return: Arrays of errors and confidence intervals
        """
        # number of values
        n_values = len(values)

        # replace value by mean
        means = np.mean(bs, axis=0)

        # determine confidence intervals and errors on y axis
        if bootstrap_type == 'percentile':
            cis = np.array([Bootstrap.get_ci_percentile(bs[:, i], ci_level) for i in range(n_values)]).T

            # Determine errors using mean values of the bootstraps.
            # Note that this sometimes causes the errors to be negative.
            errors = np.array([means - cis[0], cis[1] - means])
        elif bootstrap_type == 'bca':
            cis = np.array([Bootstrap.get_ci_bca(bs[:, i], values[i], ci_level) for i in range(n_values)]).T

            # raise warning if some confidence intervals could not be determined
            if np.sum(np.equal(cis, None)) > 0:
                logger.warning('Some confidence intervals could not be computed.')

                # set undefined confidence intervals to values
                cis[np.equal(cis, None)] = values[np.any(np.equal(cis, None), axis=0)]

            # determine error using original values
            errors = np.array([values - cis[0], cis[1] - values])
        else:
            raise NotImplementedError(f"Bootstrap type {bootstrap_type} not supported.")

        # issue notice if some errors are negative
        if np.sum(np.less(errors, 0)) > 0:
            logger.debug('Some computed errors were negative and were adjusted to 0.')

            # set negative errors to 0
            errors[errors < 0] = 0

        return errors, cis
