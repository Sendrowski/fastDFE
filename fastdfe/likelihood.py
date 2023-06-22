"""
MLE utilities.
"""

__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"
__date__ = "2023-02-26"

import numpy as np
from scipy.special import factorial
from scipy.stats import poisson


class Likelihood:
    """
    Utilities for computing Poisson likelihoods.
    """

    #: Epsilon for numerical stability
    eps = 1e-50

    @staticmethod
    def add_epsilon(x: np.ndarray) -> np.ndarray:
        """
        Add epsilon to zero counts.

        :param x: Array to add epsilon to
        :return: Array with epsilon added to zero counts
        """
        # convert to float
        x = x.astype(float)

        # replace 0s with epsilon to avoid log(0)
        x[x == 0] = Likelihood.eps

        return x

    @staticmethod
    def poisson(mu: np.ndarray, k: np.ndarray) -> np.ndarray:
        """
        Compute Poisson(mu, k).

        :param mu: Mean of Poisson distribution
        :param k: Number of events
        :return: Poisson(mu, k)
        """
        # add epsilon to zero counts
        mu = Likelihood.add_epsilon(mu)

        return poisson.pmf(k, mu)

    @staticmethod
    def log_poisson(mu: np.ndarray, k: np.ndarray) -> np.ndarray:
        """
        Compute log(Poisson(mu, k)).

        :param mu: Mean of Poisson distribution
        :param k: Number of events
        :return: log(Poisson(mu, k))
        """
        # add epsilon to zero counts
        mu = Likelihood.add_epsilon(mu)

        return k * np.log(mu) - mu - Likelihood.log_factorial(k)

    @staticmethod
    def log_factorial_stirling(n: np.ndarray | float) -> np.ndarray | float:
        """
        Use Stirling's approximation for values larger than n_threshold.
        https://en.wikipedia.org/wiki/Stirling%27s_approximation

        :param n: n
        :return: log(n!)
        """
        # np.log(np.sqrt(2 * np.pi * n) * (n / np.e) ** n * (1 + 1 / (12 * n)))
        return 0.5 * np.log(2 * np.pi * n) + n * np.log(n / np.e) + np.log(1 + 1 / (12 * n))

    @staticmethod
    def log_factorial(n: np.ndarray, n_threshold: int = 100) -> np.ndarray:
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
        x[~low] = Likelihood.log_factorial_stirling(n[~low])

        return x
