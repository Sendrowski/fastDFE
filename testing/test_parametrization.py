import numpy as np
import pytest
from matplotlib import pyplot as plt

import fastdfe as fd
from fastdfe import GammaExpParametrization, BaseInference, Config
from fastdfe.visualization import Visualization
from fastdfe.discretization import Discretization
from fastdfe.parametrization import DiscreteParametrization, GammaDiscreteParametrization, \
    DisplacedGammaParametrization, DiscreteFractionalParametrization
from testing import TestCase


@pytest.mark.inference
class ParametrizationTestCase(TestCase):
    """
    Test the parametrization classes.
    """
    n = 20

    config_file = "testing/cache/configs/pendula_C_full_anc/config.yaml"

    def test_cdf_and_pdf_accept_numpy_scalars_and_arrays_of_any_shape(self):
        """
        The CDF and PDF of every parametrization accept NumPy scalars and arrays of any shape. The discrete
        CDFs failed on everything but Python numbers and 1-d arrays.
        """
        from fastdfe import GammaExpParametrization, DiscreteFractionalParametrization

        grid = np.array([[-200.0, -5.0, -0.5], [0.5, 5.0, 2000.0]])

        for model in [GammaExpParametrization(), DiscreteFractionalParametrization(), DiscreteParametrization()]:
            for get in [model.get_cdf, model.get_pdf]:
                func = get(**model.x0)
                expected = np.array([func(float(S)) for S in grid.ravel()])

                np.testing.assert_array_equal(func(grid), expected.reshape(grid.shape))
                np.testing.assert_array_equal(func(grid.ravel()), expected)

                for scalar in [np.float32(-5), np.int64(-5), np.array(-5.0)]:
                    self.assertAlmostEqual(func(scalar), func(-5.0), places=6)

    def test_discrete_intervals_need_to_be_finite_and_increasing(self):
        """
        The discrete parametrizations reject interval boundaries that are infinite or not increasing. Infinite
        boundaries were accepted, which gave bins of infinite width holding mass.
        """
        from fastdfe import DiscreteFractionalParametrization

        for cls in [DiscreteFractionalParametrization, DiscreteParametrization]:
            for intervals in [[-np.inf, -10, 0, 10], [-10, 0, 10, np.inf], [-10, 0, 0, 10], [0, -10, 10]]:
                with self.assertRaises(ValueError):
                    cls(intervals=intervals)

            cls(intervals=[-10, 0, 10])

    def test_gamma_density_is_zero_at_minus_infinity(self):
        """
        The gamma-based densities are zero at ``S = -inf``. They were undefined there for a shape above 1.
        """
        from fastdfe import GammaExpParametrization, GammaDiscreteParametrization, DisplacedGammaParametrization

        S = np.array([-np.inf, -10.0, 0.5])

        for model in [GammaExpParametrization(), GammaDiscreteParametrization(), DisplacedGammaParametrization()]:
            for b in [0.5, 2.0]:
                density = model.get_pdf(**model.x0 | dict(b=b))(S)

                self.assertEqual(density[0], 0)
                self.assertTrue(np.isfinite(density).all())
                self.assertGreater(density[1], 0)

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
        d2 = p._discretize(params, d.bins)

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
        d2 = p._discretize(params, d.bins) / d.interval_sizes

        plt.plot(np.arange(d.n_intervals), d1, alpha=0.5, label='exact PDF')
        plt.plot(np.arange(d.n_intervals), d2, alpha=0.5, label='empirical PDF')

        plt.title(p.__class__.__name__)
        plt.legend()
        plt.show()

        assert np.max(np.abs(d1 - d2)[d.s != 0]) < 1e-12

    def test_compare_exact_vs_empirical_cdf_discretized_fractional_parametrization(self):
        """
        Compare exact vs empirical CDF for DiscreteParametrization
        """
        p = DiscreteFractionalParametrization()

        d = Discretization(
            n=self.n,
            intervals_ben=(1e-2, 1000, 1000),
            intervals_del=(-1000000, -1e-2, 1000)
        )

        params = {'S1': 0.5, 'S2': 0.5, 'S3': 0.5, 'S4': 0.9, 'S5': 0.3}

        d1 = np.cumsum(p.get_pdf(**params)(d.s) * d.interval_sizes)
        d2 = p.get_cdf(**params)(d.s)

        plt.plot(np.arange(d.n_intervals), d1, alpha=0.5, label='empirical CDF')
        plt.plot(np.arange(d.n_intervals), d2, alpha=0.5, label='exact CDF')

        plt.title(p.__class__.__name__)
        plt.legend()
        plt.show()

        assert np.max(np.abs(d1 - d2)) < 0.005

    def test_compare_pdf_vs_discretized_cdf_discretized_fractional_parametrization(self):
        """
        Compare PDF vs discretized CDF for DiscreteParametrization.
        """
        p = DiscreteFractionalParametrization()

        d = Discretization(
            n=self.n,
            intervals_ben=(1e-2, 1000, 1000),
            intervals_del=(-1000000, -1e-2, 1000)
        )

        params = {'S1': 0.5, 'S2': 0.5, 'S3': 0.5, 'S4': 0.9, 'S5': 0.3}

        d1 = p.get_pdf(**params)(d.s)
        d2 = p._discretize(params, d.bins) / d.interval_sizes

        plt.plot(np.arange(d.n_intervals), d1, alpha=0.5, label='exact PDF')
        plt.plot(np.arange(d.n_intervals), d2, alpha=0.5, label='empirical PDF')

        plt.title(p.__class__.__name__)
        plt.legend()
        plt.show()

        assert np.max(np.abs(d1 - d2)[d.s != 0]) < 1e-12

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
        d2 = p._discretize(params, d.bins)

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
        d2 = p._discretize(params, d.bins)

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

    def test_run_inference_discrete_fractional_parametrization(self):
        """
        Test that inference runs without error for DiscreteParametrization.
        """
        config = Config.from_file(self.config_file)

        config.update(
            model=DiscreteFractionalParametrization(),
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

    def test_plot_pdf(self):
        """
        Test that plotting the PDF works.
        """
        model = GammaExpParametrization()

        Visualization.plot_pdf(model, model.x0, s=np.linspace(-100, 100, 1000))

    def test_plot_cdf(self):
        """
        Test that plotting the CDF works.
        """
        model = GammaExpParametrization()

        Visualization.plot_cdf(model, model.x0, s=np.linspace(-100, 100, 1000))

    def test_to_fractional_with_all_mass_in_leading_bins(self):
        """
        ``to_fractional`` inverts ``to_nominal`` also when the leading bins hold all the mass, as for a purely
        deleterious DFE. The remaining fractions were computed as ``0 / 0`` and raised a ``ZeroDivisionError``.
        """
        model = DiscreteFractionalParametrization()

        nominal = dict(S1=0.25, S2=0.25, S3=0.25, S4=0.25, S5=0.0, S6=0.0)
        fractional = model.to_fractional(nominal)

        self.assertTrue(all(0 <= v <= 1 for v in fractional.values()))

        restored = model.to_nominal(fractional)

        for key, value in nominal.items():
            self.assertAlmostEqual(restored[key], value)

    def test_to_nominal_to_fractional_are_inverse_functions(self):
        """
        Test that to_nominal and to_fractional are inverse functions.
        """
        params = {'S1': 0.5, 'S2': 0.5, 'S3': 0.5, 'S4': 0.9, 'S5': 0.3}
        p = DiscreteFractionalParametrization()

        observed = p.to_fractional(p.to_nominal(params))
        observed.pop('S6')

        assert observed == params

    def test_plot_parametrization(self):
        """
        Test that plotting the parametrization works.
        """
        GammaExpParametrization().plot(
            params=GammaExpParametrization.x0
        )

    def test_deleterious_submodel_discrete_parametrization(self):
        """
        Test that deleterious submodel works for DiscreteParametrization.
        """
        for inv in [
            np.array([-100000, -100, -10, -1, 0, 1, 1000]),
            np.array([-100000, -1, 0, 1, 10, 1000])
        ]:
            p = DiscreteFractionalParametrization(intervals=inv)

            y = p.get_pdf(
                **(p.x0 | p.submodels['dele'])
            )(np.linspace(0.00001, 1000, 100))

            self.assertEqual(np.sum(y), 0)

            y = p.get_pdf(
                **(p.x0)
            )(np.linspace(0.00001, 1000, 100))

            self.assertGreater(np.sum(y), 0)

    def test_dfe_cdf_pdf(self):
        """
        Test that DFE CDF works for all parametrizations.
        """
        p = GammaExpParametrization()
        dfe = p.freeze(p.x0)

        s = np.linspace(*p.bounds['S_d'], 1000)

        y1 = p.get_cdf(**p.x0)(s)
        y2 = dfe.cdf(s)

        np.testing.assert_array_equal(y1, y2)

        y1 = p.get_pdf(**p.x0)(s)
        y2 = dfe.pdf(s)

        np.testing.assert_array_equal(y1, y2)

    def test_custom_bounds(self):
        """
        Test that custom bounds work as expected.
        """
        self.assertDictEqual(
            GammaExpParametrization(bounds=dict(p_b=(0.1, 0.9))).bounds,
            GammaExpParametrization.bounds | dict(p_b=(0.1, 0.9))
        )

        # make sure original class variable is not modified
        self.assertTupleEqual(
            GammaExpParametrization.bounds['p_b'],
            (0.0, 0.5)
        )

        self.assertTupleEqual(
            DiscreteParametrization(bounds=dict(S1=(0.1, 0.9))).bounds['S1'],
            (0.1, 0.9)
        )

        self.assertTupleEqual(
            DiscreteParametrization().bounds['S1'],
            (0.0, 1.0)
        )

    def test_dfe_serialization(self):
        """
        Test that DFE serialization works for all parametrizations.
        """
        dfe = GammaExpParametrization().freeze(GammaExpParametrization.x0)
        dfe.to_file("scratch/gamma_exp_dfe.json")
        dfe2 = fd.DFE.from_file("scratch/gamma_exp_dfe.json")

        self.assertDictEqual(dfe.params, dfe2.params)


class FastParametrizationTestCase(TestCase):
    """
    Fast-tier coverage of each DFE parametrization's math (pdf / cdf / discretize / normalize /
    freeze) evaluated directly, with no inference.
    """

    def test_gamma_exp_cdf_equals_scipy_stats_distributions(self):
        """
        The CDF of ``GammaExpParametrization``, which calls ``scipy.special`` directly, equals the mixture of
        the ``scipy.stats`` gamma and exponential distributions for parameters within the bounds.
        """
        from scipy.stats import gamma, expon

        rng = np.random.default_rng(0)
        S = np.concatenate([-np.logspace(8, -5, 200), [0], np.logspace(-5, 4, 200)])

        for _ in range(200):
            S_d, b, S_b = -10 ** rng.uniform(-2, 5), 10 ** rng.uniform(-2, 1), 10 ** rng.uniform(-4, 2)
            p_b = rng.uniform(0, 0.5)

            expected = np.where(
                S < 0,
                (1 - p_b) - (1 - p_b) * gamma.cdf(-S, b, scale=-S_d / b),
                (1 - p_b) + p_b * expon.cdf(S, scale=S_b)
            )

            np.testing.assert_array_equal(GammaExpParametrization.get_cdf(S_d=S_d, b=b, p_b=p_b, S_b=S_b)(S), expected)

    def test_dfe_discretize_normalizes_and_honours_warn_mass(self):
        """
        ``DFE.discretize`` returns a histogram summing to 1 and honours ``warn_mass`` with and without bootstraps.
        Without bootstraps the histogram was not normalized, and with bootstraps ``warn_mass=False`` was ignored.
        """
        import pandas as pd

        params = dict(S_d=-300.0, b=0.3, p_b=0.05, S_b=1.0)
        bootstraps = pd.DataFrame([params, dict(params, S_d=-400.0), dict(params, b=0.35)])

        # bins that miss the mass below -100 and above 0
        bins = np.array([-100.0, -10.0, -1.0, 0.0])

        for dfe in [fd.DFE(params=params, model=GammaExpParametrization()),
                    fd.DFE(params=params, model=GammaExpParametrization(), bootstraps=bootstraps)]:
            with self.assertNoLogs('fastdfe', level='WARNING'):
                values, _ = dfe.discretize(bins, warn_mass=False)

            self.assertAlmostEqual(values.sum(), 1)

            with self.assertLogs('fastdfe', level='WARNING'):
                dfe.discretize(bins, warn_mass=True)

    @staticmethod
    def _models():
        return [
            fd.GammaExpParametrization(),
            DisplacedGammaParametrization(),
            GammaDiscreteParametrization(),
            DiscreteParametrization(),
            DiscreteFractionalParametrization(),
        ]

    bins = np.array([-np.inf, -1e6, -1e3, -10, -1, 0, 1, 10, 1e3, 1e4, np.inf])

    def test_pdf_cdf_discretize(self):
        """Each model's pdf/cdf are finite and its discretized DFE integrates to ~1."""
        S = np.array([-1000., -10., -1., 0., 1., 10.])

        for model in self._models():
            with self.subTest(model=type(model).__name__):
                params = dict(model.x0)

                self.assertTrue(len(model.param_names) > 0)
                self.assertTrue(np.all(np.isfinite(model.get_pdf(**params)(S))))
                self.assertTrue(np.all(np.isfinite(model.get_cdf(**params)(S))))

                hist = model._discretize(params, self.bins, warn_mass=False)
                self.assertAlmostEqual(hist.sum(), 1.0, places=2)

                # parameter post-processing + DFE construction
                model._normalize(dict(params))
                self.assertIsInstance(model.freeze(params), fd.DFE)

    def test_plot_discretized(self):
        """The discretized-DFE plot renders for one model (covers the plotting path)."""
        model = fd.GammaExpParametrization()
        model.plot(params=dict(model.x0), file='scratch/fast_param_plot.png', show=False)
