import numpy as np
from scipy import stats

from fastdfe.bootstrap import Bootstrap
from testing import TestCase


class BootstrapTestCase(TestCase):
    """
    Test the Bootstrap class.
    """

    def test_ci_percentile(self):
        x = np.arange(100)

        cis = Bootstrap.get_ci_percentile(x, 0.05)

        assert cis == (5, 95)

    def test_ci_bca(self):
        x = np.arange(100)

        cis = Bootstrap.get_ci_bca(x, 50, 0.05)

        assert cis == (5, 95)

    @staticmethod
    def get_cis(ci_level, values):
        """

        :param ci_level:
        :param values:
        :return:
        """
        # calculate the sample mean and standard error
        sample_mean = np.mean(values)
        standard_error = np.std(values)

        # determine the z-score for the chosen confidence level
        z_score = stats.norm.ppf(1 - ci_level)

        # calculate the margin of error
        margin_of_error = z_score * standard_error

        # calculate the confidence interval
        lower_bound = sample_mean - margin_of_error
        upper_bound = sample_mean + margin_of_error

        return lower_bound, upper_bound

    def compare_cis(self, values):
        """

        :param values:
        """
        # set the confidence level
        ci_level = 0.05

        # create synthetic data
        original = np.mean(values)

        lower_bound, upper_bound = self.get_cis(ci_level, values)

        # calculate errors and confidence intervals using the get_errors function
        errors_perc, cis_perc = Bootstrap.get_errors([original], np.array([values]).T, ci_level, 'percentile')
        errors_bca, cis_bca = Bootstrap.get_errors([original], np.array([values]).T, ci_level, 'bca')

        assert np.abs(lower_bound - cis_perc[0][0]) / lower_bound < 0.005
        assert np.abs(upper_bound - cis_perc[1][0]) / upper_bound < 0.005

        assert np.abs(lower_bound - cis_bca[0][0]) / lower_bound < 0.005
        assert np.abs(upper_bound - cis_bca[1][0]) / upper_bound < 0.005

    def test_get_errors_percentile_positive(self):
        """
        Test the get_errors function for positive values.
        """
        np.random.seed(42)
        self.compare_cis(np.random.normal(5, 2, size=100000))

    def test_get_errors_percentile_negative(self):
        """
        Test the get_errors function for negative values.
        """
        np.random.seed(42)
        self.compare_cis(-np.random.normal(5, 2, size=100000))

    def test_get_errors_percentile_positive_and_negative(self):
        """
        Test the get_errors function for positive and negative values.
        """
        np.random.seed(42)
        self.compare_cis(np.random.normal(0, 2, size=100000))

    def test_get_bounds_from_quantile_is_nan_return_nones(self):
        """
        Test get_bounds_from_quantile when data is nan.
        """
        bounds = Bootstrap.get_bounds_from_quantile([2, 2, 2], np.nan, np.nan, 3)

        self.assertEqual(tuple(bounds), (None, None))

    def test_get_ci_percentile_invalid_a_raises_value_error(self):
        """
        Test get_ci_percentile when a is invalid.
        """
        with self.assertRaises(ValueError):
            Bootstrap.get_ci_percentile([1, 2, 3], 1.1)

        with self.assertRaises(ValueError):
            Bootstrap.get_ci_percentile([1, 2, 3], -0.1)

    def test_get_errors_invalid_bootstrap_type_raises_not_implemented_error(self):
        """
        Test get_errors when bootstrap_type is invalid.
        """
        with self.assertRaises(NotImplementedError):
            Bootstrap.get_errors([1, 2, 3], np.array([[1, 2, 3]]), bootstrap_type='invalid')

    def test_get_ci_bca_no_data_return_zeros(self):
        """
        Test get_ci_bca when bootstraps is empty.
        """
        bounds = Bootstrap.get_ci_bca([], 1, 0.05)

        self.assertEqual(bounds, [0, 0])
