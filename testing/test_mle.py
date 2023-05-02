from testing import prioritize_installed_packages

prioritize_installed_packages()

from unittest import TestCase

from scipy.special import factorial

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
