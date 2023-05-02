from testing import prioritize_installed_packages

prioritize_installed_packages()

from unittest import TestCase

import numpy as np
import pytest
from matplotlib import pyplot as plt

from fastdfe import GammaExpParametrization, BaseInference, Config
from fastdfe.discretization import Discretization, get_midpoints_and_spacing
from fastdfe.parametrization import DiscreteParametrization, GammaDiscreteParametrization, DisplacedGammaParametrization


class ParametrizationTestCase(TestCase):
    """
    TODO add integration tests for the different parametrizations
    """
    n = 20

    show_plots = True

    config_file = "testing/configs/pendula_C_full_anc/config.yaml"

    def test_compare_exact_vs_empirical_cdf_gamma_exp(self):
        """
        Compare exact vs empirical CDF for GammaExpParametrization.
        """
        p = GammaExpParametrization()
        params = GammaExpParametrization.x0
        d = Discretization(
            n=self.n,
            intervals_ben=(1e-15, 100, 1000),
            intervals_del=(-100000, -1e-15, 1000)
        )

        d1 = np.cumsum(p.get_pdf(**params)(d.s) * d.interval_sizes)
        d2 = p.get_cdf(**params)(d.s)

        plt.plot(d.s, d1, alpha=0.5, label='empirical CDF')
        plt.plot(d.s, d2, alpha=0.5, label='exact CDF')

        plt.legend()
        plt.show()

        plt.title('GammaExpParametrization')

        diff = np.max(np.abs(d1 - d2))

        assert diff < 0.005

    def test_compare_exact_vs_empirical_cdf_gamma_exp_low_shape(self):
        """
        Compare exact vs empirical CDF for GammaExpParametrization for low shape parameter.
        """
        p = GammaExpParametrization()
        d = Discretization(
            n=self.n,
            intervals_ben=(1e-15, 100, 1000),
            intervals_del=(-100000, -1e-15, 1000)
        )

        params = {
            'S_d': -37572.964129977896,
            'b': 3.8866779746119997,
            'p_b': 0.19204516636691807,
            'S_b': 0.0001
        }

        d1 = np.cumsum(p.get_pdf(**params)(d.s) * d.interval_sizes)
        d2 = p.get_cdf(**params)(d.s)

        plt.plot(d.s, d1, alpha=0.5, label='empirical CDF')
        plt.plot(d.s, d2, alpha=0.5, label='exact CDF')

        plt.title(p.__class__.__name__)
        plt.legend()
        plt.show()

        diff = np.max(np.abs(d1 - d2))

        assert diff < 0.009

    def test_compare_pdf_vs_discretized_cdf_gamma_exp(self):
        """
        Compare PDF vs discretized CDF for GammaExpParametrization.
        """
        p = GammaExpParametrization()
        d = Discretization(
            n=self.n,
            intervals_ben=(1e-15, 100, 1000),
            intervals_del=(-100000, -1e-15, 1000)
        )

        params = {
            'S_d': -37572.964129977896,
            'b': 3.8866779746119997,
            'p_b': 0.19204516636691807,
            'S_b': 0.0001
        }

        d1 = p.get_pdf(**params)(d.s) * d.interval_sizes
        d2 = p.discretize(params, d.bins)

        plt.plot(np.arange(d.n_intervals), d1, alpha=0.5, label='exact PDF')
        plt.plot(np.arange(d.n_intervals), d2, alpha=0.5, label='empirical PDF')

        plt.title(p.__class__.__name__)
        plt.legend()
        plt.show()

        diff = np.max(np.abs(d1 - d2))

        assert diff < 1e-4

    def test_compare_exact_vs_empirical_cdf_discretized_parametrization(self):
        """
        Compare exact vs empirical CDF for DiscreteParametrization
        """
        p = DiscreteParametrization()

        d = Discretization(
            n=self.n,
            intervals_ben=(1e-2, 1000, 1000),
            intervals_del=(-1000000, -1e-2, 1000)
        )

        """
        d.bins = p.intervals[1:-1]
        d.s, d.interval_sizes = get_midpoints_and_spacing(d.bins)
        d.n_intervals = len(d.interval_sizes)
        """

        params = {'S1': 0.25, 'S2': 0.35, 'S3': 0.2, 'S4': 0.1, 'S5': 0.05, 'S6': 0.05}

        d1 = np.cumsum(p.get_pdf(**params)(d.s) * d.interval_sizes)
        d2 = p.get_cdf(**params)(d.s)

        plt.plot(np.arange(d.n_intervals), d1, alpha=0.5, label='empirical CDF')
        plt.plot(np.arange(d.n_intervals), d2, alpha=0.5, label='exact CDF')

        plt.title(p.__class__.__name__)
        plt.legend()
        plt.show()

        assert np.max(np.abs(d1 - d2)) < 0.005

    def test_compare_pdf_vs_discretized_cdf_discretized_parametrization(self):
        """
        Compare PDF vs discretized CDF for DiscreteParametrization.
        """
        p = DiscreteParametrization()

        d = Discretization(
            n=self.n,
            intervals_ben=(1e-2, 1000, 1000),
            intervals_del=(-1000000, -1e-2, 1000)
        )

        """
        d.bins = p.intervals[1:-1]
        d.s, d.interval_sizes = get_midpoints_and_spacing(d.bins)
        d.n_intervals = len(d.interval_sizes)
        """

        params = {'S1': 0.25, 'S2': 0.35, 'S3': 0.2, 'S4': 0.1, 'S5': 0.05, 'S6': 0.05}

        d1 = p.get_pdf(**params)(d.s)
        d2 = p.discretize(params, d.bins) / d.interval_sizes

        plt.plot(np.arange(d.n_intervals), d1, alpha=0.5, label='exact PDF')
        plt.plot(np.arange(d.n_intervals), d2, alpha=0.5, label='empirical PDF')

        plt.title(p.__class__.__name__)
        plt.legend()
        plt.show()

        assert np.max(np.abs(d1 - d2)) < 0.03

    def test_compare_exact_vs_empirical_cdf_gamma_discrete(self):
        """
        Compare exact vs empirical CDF for GammaDiscreteParametrization.
        """
        p = GammaDiscreteParametrization()
        params = GammaDiscreteParametrization.x0
        d = Discretization(
            n=self.n,
            intervals_ben=(1e-15, 100, 1000),
            intervals_del=(-100000, -1e-15, 1000)
        )

        d1 = np.cumsum(p.get_pdf(**params)(d.s) * d.interval_sizes)
        d2 = p.get_cdf(**params)(d.s)

        plt.plot(d.s, d1, alpha=0.5, label='empirical CDF')
        plt.plot(d.s, d2, alpha=0.5, label='exact CDF')

        plt.title(p.__class__.__name__)
        plt.legend()
        plt.show()

        diff = np.max(np.abs(d1 - d2))

        assert diff < 0.05

    def test_compare_pdf_vs_discretized_cdf_gamma_discrete(self):
        """
        Compare PDF vs discretized CDF for GammaDiscreteParametrization.
        """
        p = GammaDiscreteParametrization()
        d = Discretization(
            n=self.n,
            intervals_ben=(1e-15, 100, 1000),
            intervals_del=(-100000, -1e-15, 1000)
        )

        params = GammaDiscreteParametrization.x0

        d1 = p.get_pdf(**params)(d.s) * d.interval_sizes
        d2 = p.discretize(params, d.bins)

        plt.plot(np.arange(d.n_intervals), d1, alpha=0.5, label='exact PDF')
        plt.plot(np.arange(d.n_intervals), d2, alpha=0.5, label='empirical PDF')

        plt.title(p.__class__.__name__)
        plt.legend()
        plt.show()

        diff = np.max(np.abs(d1 - d2))

        # discrete CDF produces a large point mass
        assert diff < 0.001

    def test_compare_exact_vs_empirical_cdf_displaced_gamma(self):
        """
        Compare exact vs empirical CDF for DisplacedGammaParametrization.
        """
        p = DisplacedGammaParametrization()
        params = DisplacedGammaParametrization.x0
        d = Discretization(
            n=self.n,
            intervals_ben=(1e-15, 100, 1000),
            intervals_del=(-100000, -1e-15, 1000)
        )

        d1 = np.cumsum(p.get_pdf(**params)(d.s) * d.interval_sizes)
        d2 = p.get_cdf(**params)(d.s)

        plt.plot(d.s, d1, alpha=0.5, label='empirical CDF')
        plt.plot(d.s, d2, alpha=0.5, label='exact CDF')

        plt.title(p.__class__.__name__)
        plt.legend()
        plt.show()

        diff = np.max(np.abs(d1 - d2))

        assert diff < 0.05

    def test_compare_pdf_vs_discretized_cdf_displaced_gamma(self):
        """
        Compare PDF vs discretized CDF for DisplacedGammaParametrization.
        """
        p = DisplacedGammaParametrization()
        d = Discretization(
            n=self.n,
            intervals_ben=(1e-15, 100, 1000),
            intervals_del=(-100000, -1e-15, 1000)
        )

        params = DisplacedGammaParametrization.x0

        d1 = p.get_pdf(**params)(d.s) * d.interval_sizes
        d2 = p.discretize(params, d.bins)

        plt.plot(np.arange(d.n_intervals), d1, alpha=0.5, label='exact PDF')
        plt.plot(np.arange(d.n_intervals), d2, alpha=0.5, label='empirical PDF')

        plt.title(p.__class__.__name__)
        plt.legend()
        plt.show()

        diff = np.max(np.abs(d1 - d2))

        # discrete CDF produces a large point mass
        assert diff < 2e-4

    def test_run_inference_discrete_parametrization(self):
        """
        Test that inference runs without error for DiscreteParametrization.
        """
        config = Config.from_file(self.config_file)

        config.update(
            model=DiscreteParametrization(),
            do_bootstrap=True,
            parallelize=False
        )

        inf = BaseInference.from_config(config)

        inf.run()

        inf.plot_continuous()
        inf.plot_discretized()

    def test_run_inference_gamma_exp_parametrization(self):
        """
        Test that inference runs without error for GammaExpParametrization.
        """
        config = Config.from_file(self.config_file)

        config.update(
            model=GammaExpParametrization(),
            do_bootstrap=True,
            parallelize=False
        )

        inf = BaseInference.from_config(config)

        inf.run()

        inf.plot_continuous()
        inf.plot_discretized()

    def test_run_inference_gamma_discrete_parametrization(self):
        """
        Test that inference runs without error for GammaDiscreteParametrization.
        """
        config = Config.from_file(self.config_file)

        config.update(
            model=GammaDiscreteParametrization(),
            do_bootstrap=True,
            parallelize=False
        )

        inf = BaseInference.from_config(config)

        inf.run()

        inf.plot_continuous()
        inf.plot_discretized()

    def test_run_inference_displaced_gamma_parametrization(self):
        """
        Test that inference runs without error for DisplacedGammaParametrization.
        """
        config = Config.from_file(self.config_file)

        config.update(
            model=DisplacedGammaParametrization(),
            do_bootstrap=True,
            parallelize=False
        )

        inf = BaseInference.from_config(config)

        inf.run()

        inf.plot_continuous()
        inf.plot_discretized()
