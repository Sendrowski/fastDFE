"""
MLE utilities.
"""

__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"
__date__ = "2023-02-26"

import numpy as np
from typing import Union
from scipy.special import factorial


class MLE:
    """
    MLE utilities.
    """

    @staticmethod
    def log_poisson(mu, k):
        """
        Compute log(Poisson(mu, k)).

        :param mu: Mean of Poisson distribution
        :param k: Number of events
        :return: log(Poisson(mu, k))
        """
        return k * np.log(mu) - mu - MLE.log_factorial(k)

    @staticmethod
    def log_factorial_stirling(n: Union[int, np.ndarray]):
        """
        Use Stirling's approximation for values larger than n_threshold.
        https://en.wikipedia.org/wiki/Stirling%27s_approximation

        :param n: n
        :return: log(n!)
        """
        # np.log(np.sqrt(2 * np.pi * n) * (n / np.e) ** n * (1 + 1 / (12 * n)))
        return 0.5 * np.log(2 * np.pi * n) + n * np.log(n / np.e) + np.log(1 + 1 / (12 * n))

    @staticmethod
    def log_factorial(n: np.ndarray, n_threshold: int = 100):
        """
        Compute log(n!).

        :param n: n
        :param n_threshold: Threshold for using Stirling's approximation
        :return: log(n!)
        """
        x = np.zeros_like(n, dtype=np.float64)

        low = n <= n_threshold

        # compute for low values of n
        x[low] = np.log(factorial(n[low]))

        # use Stirling's approximation for large value of n
        x[~low] = MLE.log_factorial_stirling(n[~low])

        return x
