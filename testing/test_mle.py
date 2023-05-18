from testing import prioritize_installed_packages

prioritize_installed_packages()

from unittest import TestCase

from scipy.special import factorial
from scipy.stats import poisson

import numpy as np
from fastdfe.mle import MLE


class MLETestCase(TestCase):

    def test_stirling_approximation(self):
        """
        Compare Stirling approximation of the factorial with its expected value.
        Note that we can only do this for moderately low exponents due to overflow.
        """
        x = np.arange(100, 150)

        expected = np.log(factorial(x))
        observed = MLE.log_factorial_stirling(x)

        self.assertTrue(np.linalg.norm(expected - observed, ord=1) > 1e-05)

    def test_log_poisson(self):
        """
        Test the log poisson function.
        """
        comparisons = [
            dict(mu=1, k=1, places=50),
            dict(mu=1, k=2, places=50),
            dict(mu=11, k=32, places=12),  # MLE.poisson becomes imprecise
            dict(mu=0, k=0, places=45),  # yields log likelihood very close to zero for approximation
            dict(mu=0, k=1, places=50),  # yields very low log likelihood
            dict(mu=1, k=0, places=50),  # yields log likelihood of -1
        ]

        for comp in comparisons:
            expected = np.log(MLE.poisson(mu=np.array([comp['mu']]), k=np.array([comp['k']])))[0]
            observed = MLE.log_poisson(mu=np.array([comp['mu']]), k=np.array([comp['k']]))[0]
            self.assertAlmostEqual(expected, observed, places=comp['places'])

    def test_log_poisson_zero_entries(self):
        """
        Make sure the log poisson function does not return inf or nan values.
        """
        res = MLE.log_poisson(mu=np.array([1, 0, 0]), k=np.array([0, 1, 0]))

        # check that there are no inf or nan values
        assert ((res == np.inf) | (np.isnan(res)) | (res == -np.inf)).sum() == 0
