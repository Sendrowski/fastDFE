import logging
from typing import Literal, List

import matplotlib.pyplot as plt
import numpy as np
from numpy import testing

from fastdfe.optimization import pack_params, unpack_params, flatten_dict, unflatten_dict, filter_dict, Optimization, \
    merge_dicts, correct_values, to_symlog, from_symlog, scale_bound, scale_value, unscale_value, unscale_bound, \
    check_bounds, collapse_fixed_to_mean, collapse_fixed, expand_shared, SharedParams
from testing import TestCase


class OptimizationTestCase(TestCase):
    """
    Test the Optimization class.
    """

    def test_pack_unpack_restores_dict(self):
        """
        Check whether the parameters can be restored from packed format.
        """
        params = {
            'all': {'S_d': -400.0, 'b': 0.4, 'p_b': 0.02, 'S_b': 4.0, 'eps': 0.05},
            'sub': {'S_d': -399.0, 'b': 0.39, 'p_b': 0.12, 'S_b': 4.1, 'eps': 0.02},
        }

        restored = unpack_params(pack_params(params), params)

        self.assertDictEqual(params, restored)

    def test_flatten_dict(self):
        """
        Test that flatten_dict works as expected.
        """
        d = {'a': 1, 'b': {'c': 2, 'd': {'e': 3}}}
        expected = {'a': 1, 'b.c': 2, 'b.d.e': 3}
        assert flatten_dict(d) == expected

        d = {'a': {'b': {'c': 1}}}
        expected = {'a.b.c': 1}
        assert flatten_dict(d) == expected

    def test_unflatten_dict(self):
        """
        Test that unflatten_dict works as expected.
        """
        d = {'a': 1, 'b.c': 2, 'b.d.e': 3}
        expected = {'a': 1, 'b': {'c': 2, 'd': {'e': 3}}}
        assert unflatten_dict(d) == expected

        d = {'a.b.c': 1}
        expected = {'a': {'b': {'c': 1}}}
        assert unflatten_dict(d) == expected

    def test_flatten_unflatten_dict(self):
        """
        Test that flatten_dict and unflatten_dict are inverse operations.
        """
        d = {'a': 1, 'b': {'c': 2, 'd': {'e': 3}}}
        expected = d
        flattened = flatten_dict(d)
        unflattened = unflatten_dict(flattened)
        assert unflattened == expected

    def test_filter_dict(self):
        """
        Test that filter_dict works as expected.
        """
        d = {'a': 1, 'b': {'c': 2, 'd': {'e': 3, 'f': {'g': 4}}}, 'c': {'g': 1, 'b': 2}}
        keys = ['a', 'c']
        expected = {'a': 1, 'b': {'c': 2}}
        assert filter_dict(d, keys) == expected

    def test_sample_x0_nested_dict(self):
        """
        Test that sample_x0 works as expected when the initial parameters are a nested dict.
        """
        x0 = {'a': 1, 'b': {'c': 2, 'd': 4}}

        o = Optimization(
            bounds={'a': (2, 2), 'b': (0, 0), 'c': (-1, -1), 'd': (4, 4)},
            scales={'a': 'lin', 'b': 'lin', 'c': 'lin', 'd': 'lin'},
            param_names=['a', 'b', 'c', 'd']
        )

        sample = o.sample_x0(x0)
        expected = {'a': 2.0, 'b': {'c': -1.0, 'd': 4.0}}

        assert sample == expected

    def test_merge_dicts(self):
        """
        Test that merge_dicts works as expected.
        """
        dict1 = {'a': 1, 'b': {'c': 2, 'd': 3}}
        dict2 = {'b': {'d': 4, 'e': 5}, 'f': 6}
        expected_result = {'a': 1, 'b': {'c': 2, 'd': 4, 'e': 5}, 'f': 6}

        assert merge_dicts(dict1, dict2) == expected_result

        dict3 = {'a': {'b': 1, 'c': {'d': 2, 'e': {'f': 3}}}}
        dict4 = {'a': {'b': 4, 'c': {'d': 5, 'e': {'g': 6}}}}
        expected_result2 = {'a': {'b': 4, 'c': {'d': 5, 'e': {'f': 3, 'g': 6}}}}

        assert merge_dicts(dict3, dict4) == expected_result2

        dict5 = {'a': 1, 'b': {'c': 2}}
        dict6 = {'b': {'c': 3}, 'd': 4}
        expected_result3 = {'a': 1, 'b': {'c': 3}, 'd': 4}

        assert merge_dicts(dict5, dict6) == expected_result3

        dict7 = {'a': 1}
        dict8 = {'b': 2}
        expected_result4 = {'a': 1, 'b': 2}

        assert merge_dicts(dict7, dict8) == expected_result4

    def test_positive_bounds(self):
        """
        Test that ValueError is raised when bounds are positive.
        """
        bounds = (1, 10)
        scale = 'lin'
        for _ in range(100):
            sample = Optimization.sample_value(bounds, scale)
            self.assertGreaterEqual(sample, bounds[0])
            self.assertLessEqual(sample, bounds[1])

    def test_negative_bounds(self):
        """
        Test that ValueError is raised when bounds are negative.
        """
        bounds = (-10, -1)
        scale = 'lin'
        for _ in range(100):
            sample = Optimization.sample_value(bounds, scale)
            self.assertGreaterEqual(sample, bounds[0])
            self.assertLessEqual(sample, bounds[1])

    def test_mixed_bounds(self):
        """
        Test that ValueError is raised when bounds are mixed.
        """
        bounds = (-5, 5)
        scale = 'lin'
        for _ in range(100):
            sample = Optimization.sample_value(bounds, scale)
            self.assertGreaterEqual(sample, bounds[0])
            self.assertLessEqual(sample, bounds[1])

    def test_positive_bounds_log_scale(self):
        """
        Test that ValueError is raised when bounds are positive and scale is log.
        """
        bounds = (1, 10)
        scale = 'log'
        for _ in range(100):
            sample = Optimization.sample_value(bounds, scale)
            self.assertGreaterEqual(sample, bounds[0])
            self.assertLessEqual(sample, bounds[1])

    def test_negative_bounds_log_scale(self):
        """
        Test that ValueError is raised when bounds are negative and scale is log.
        """
        bounds = (-10, -1)
        scale = 'log'
        for _ in range(100):
            sample = Optimization.sample_value(bounds, scale)
            self.assertGreaterEqual(sample, bounds[0])
            self.assertLessEqual(sample, bounds[1])

    def test_mixed_bounds_log_scale(self):
        """
        Test that ValueError is raised when bounds are mixed and scale is log.
        """
        bounds = (-5, 5)
        scale = 'log'
        with self.assertRaises(ValueError):
            Optimization.sample_value(bounds, scale)

    def test_expand_shared_all_follows_explicit_types(self):
        """
        An entry with ``types='all'`` expands to every type when it follows an entry naming explicit types.
        Reassigning the ``types`` argument inside the loop made it inherit the preceding entry's types.
        """
        expanded = expand_shared(
            [SharedParams(types=['a', 'b'], params=['p']), SharedParams()],
            types=['a', 'b', 'c'],
            names=['p', 'q']
        )

        self.assertEqual(expanded[0], SharedParams(types=['a', 'b'], params=['p']))
        self.assertEqual(expanded[1], SharedParams(types=['a', 'b', 'c'], params=['p', 'q']))

    def test_log_scales_cover_the_bounds(self):
        """
        Samples on the log and symlog scales reach the upper bound. Passing ``upper - lower`` as the upper limit
        confined bounds ``(1, 3)`` to ``[1, 2]`` and raised for bounds with ``upper < 2 * lower``.
        """
        rng = np.random.default_rng(0)

        for bounds in [(1, 3), (10, 15)]:
            x = np.array([Optimization.sample_value(bounds, 'log', rng) for _ in range(2000)])

            self.assertGreaterEqual(x.min(), bounds[0])
            self.assertLessEqual(x.max(), bounds[1])
            self.assertGreater(x.max(), bounds[1] - 0.05 * (bounds[1] - bounds[0]))

        x = np.abs([Optimization.sample_value((0.6, 1.0), 'symlog', rng) for _ in range(2000)])

        self.assertGreaterEqual(x.min(), 0.6)
        self.assertLessEqual(x.max(), 1.0)
        self.assertGreater(x.max(), 0.98)

    def test_correct_values_no_changes(self):
        """
        Test that values are not changed when they are within bounds.
        """
        params = {"a.a": 2, "a.b": 5, "a.c": 10}
        bounds = {"a": (1, 3), "b": (4, 6), "c": (8, 12)}
        scales = {"a": "lin", "b": "lin", "c": "lin"}

        expected = params.copy()
        actual = correct_values(params, bounds, scales)
        self.assertEqual(expected, actual)

    def test_correct_values_corrections(self):
        """
        Test that values are corrected when they are out of bounds.
        """
        params = {"a.a": 0, "a.b": 7, "a.c": 20}
        bounds = {"a": (1, 3), "b": (4, 6), "c": (8, 12)}
        scales = {"a": "lin", "b": "lin", "c": "lin"}

        expected = {"a.a": 1, "a.b": 6, "a.c": 12}
        actual = correct_values(params, bounds, scales)
        self.assertEqual(expected, actual)

    def test_correct_values_threshold_warning(self):
        """
        Test that a warning is raised when the threshold is exceeded.
        """
        params = {"a.a": 0, "a.b": 7, "a.c": 20}
        bounds = {"a": (1, 3), "b": (4, 6), "c": (8, 12)}
        scales = {"a": "lin", "b": "lin", "c": "lin"}

        with self.assertLogs(level="WARNING", logger=logging.getLogger('fastdfe')):
            correct_values(params, bounds, scales, warn=True, threshold=0.05)

    def test_symlog_scale(self):
        """
        Test the symlog scale.
        """
        testing.assert_almost_equal(4, from_symlog(to_symlog(4)))
        testing.assert_almost_equal(-4, from_symlog(to_symlog(-4)))

        y = np.array([to_symlog(x) for x in range(-100, 100)])

        # check that the sequence is strictly increasing
        self.assert_strictly_increasing(y)

        # plot the sequence
        plt.plot(y)
        plt.show()

    def test_symlog_scale_around_linear_threshold(self):
        """
        The symlog scale is continuous and strictly increasing across the linear threshold, and every value is
        recovered by the inverse. The transformation dropped from 1 to ``log10(2)`` at the threshold, so values
        in ``(linthresh, 9 * linthresh)`` were mapped back below the threshold.
        """
        linthresh = 1e-4
        x = np.concatenate([-np.logspace(2, -8, 201), [0], np.logspace(-8, 2, 201)])

        y = np.array([to_symlog(v, linthresh) for v in x])

        self.assert_strictly_increasing(y)
        testing.assert_allclose([from_symlog(v, linthresh) for v in y], x, rtol=1e-12)

        # continuous at the threshold, which maps to 1
        self.assertEqual(to_symlog(linthresh, linthresh), 1)
        self.assertAlmostEqual(to_symlog(linthresh * (1 + 1e-9), linthresh), 1, places=8)
        self.assertAlmostEqual(from_symlog(1 + 1e-9, linthresh) / linthresh, 1, places=8)

        testing.assert_allclose(unscale_value(scale_value(5e-4, (1e-4, 1e4), 'symlog'), (1e-4, 1e4), 'symlog'), 5e-4)

    def assert_strictly_increasing(self, y: np.ndarray):
        """
        Check that the sequence is strictly increasing.

        :raises AssertionError: if the sequence is not strictly increasing
        """
        if not np.all(np.diff(y) > 0):
            raise AssertionError("The sequence is not strictly increasing.")

    def test_convert_bound(self):
        """
        Test the conversion of bounds to the various scales.
        """
        testing.assert_almost_equal(
            scale_bound((-10, 10), 'lin'),
            (-10, 10)
        )

        testing.assert_almost_equal(
            scale_bound((0.1, 10), 'log'),
            (-1, 1)
        )

        # the linear threshold of 0.1 maps to 1 and the two decades above it add 2
        symlog_bounds = (-3, 3)
        bounds = (0.1, 10)
        testing.assert_almost_equal(
            scale_bound(bounds, 'symlog'),
            symlog_bounds
        )

        x = np.linspace(*symlog_bounds, 100)
        y = np.array([from_symlog(i, linthresh=bounds[0]) for i in x])

        # check that the sequence is strictly increasing
        self.assert_strictly_increasing(y)

        plt.plot(x, y)
        plt.show()

    def test_scale_value_inverse(self):
        """
        Make sure that scaling and then unscaling a value returns the original value.
        """
        # Test parameters
        test_values = np.array([-1e4, -1e2, -1e0, 1e0, 1e2, 1e4])

        # only used to define linthresh
        bounds = np.array([
            (-1e5, -1e-5),
            (-1e3, -1e-3),
            (-1e1, -1e-1),
            (1e-1, 1e1),
            (1e-3, 1e3),
            (1e-5, 1e5)
        ])

        scales: List[Literal['lin', 'log', 'symlog']] = ['lin', 'log', 'symlog']
        tolerance = 1e-10

        for scale in scales:
            for value, bound in zip(test_values, bounds):

                if scale == 'symlog' and bound[1] < 0:
                    # doesn't work for negative bounds
                    continue

                # Scale and then unscale the value
                scaled_value = scale_value(value, bound, scale)
                unscaled_value = unscale_value(scaled_value, bound, scale)

                # Check if the original value and unscaled_value are close within the tolerance
                assert np.isclose(value, unscaled_value, atol=tolerance, rtol=0)

    def test_scale_bounds_inverse(self):
        """
        Make sure that scaling and then unscaling bounds returns the original bounds.
        """
        # Test parameters
        test_bounds = [(1e-5, 1e6), (1e-1, 1e4), (1e-3, 1e1)]
        scales: List[Literal['lin', 'log', 'symlog']] = ['lin', 'log', 'symlog']

        for scale in scales:
            for bounds in test_bounds:
                # Scale and then unscale the bounds
                scaled_bounds = scale_bound(bounds, scale)
                unscaled_bounds = unscale_bound(scaled_bounds, scale, linthresh=bounds[0])

                # Check if the original bounds and unscaled_bounds are close within the tolerance
                assert np.allclose(bounds, unscaled_bounds, atol=1e-7, rtol=0)

    def test_scale_bounds_raises_error(self):
        """
        Test that scale_bounds raises an error when the bounds or the scale are not valid.
        """
        with self.assertRaises(ValueError):
            scale_bound((-1, 1), 'log')

        with self.assertRaises(ValueError):
            scale_bound((0, 1), 'log')

        with self.assertRaises(ValueError):
            scale_bound((-1, 1), 'symlog')

        with self.assertRaises(ValueError):
            scale_bound((1, 2), 'foo')

    def test_plot_scaled_values(self):
        """
        Test plotting the scaled values within the untransformed bounds for the different scales.
        """
        bounds = dict(
            lin=(-10, 10),
            log=(0.1, 10),
            symlog=(0.001, 10)
        )

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        scale: Literal['lin', 'log', 'symlog']
        for i, scale in enumerate(['lin', 'log', 'symlog']):
            bounds_scaled = scale_bound(bounds[scale], scale)
            x_scaled = [unscale_value(x, bounds=bounds[scale], scale=scale) for x in np.linspace(*bounds_scaled, 100)]
            y_scaled = np.array([scale_value(x, bounds=bounds[scale], scale=scale) for x in x_scaled])

            x_unscaled = np.linspace(*unscale_bound(bounds_scaled, scale=scale, linthresh=bounds[scale][0]), 100)
            y_unscaled = np.array([unscale_value(i, bounds=bounds[scale], scale=scale) for i in y_scaled])
            axes[i].plot(x_unscaled, y_unscaled, label=scale)
            axes[i].set_title(scale)

        plt.show()

    def test_sample_value_symlog_scale(self):
        """
        Sample a value from the symlog scale.
        """
        y = [Optimization.sample_value(scale='symlog', bounds=(0.1, 10)) for _ in range(10000)]

        plt.hist(y, bins=50)
        plt.show()

        # check that we both have positive and negative values
        assert np.any(np.array(y) > 0) and np.any(np.array(y) < 0)

    def test_check_bounds_linear(self):
        """
        Test that check_bounds works as expected for linear scale.
        """
        bounds = dict(
            param1=(1, 10),
            param2=(1, 10),
            param3=(-10, -1),
            param4=(-10, -1),
            param5=(-10, -1),
        )

        params = dict(
            param1=1.1,
            param2=1.005,
            param3=-9.9,
            param4=-9.995,
            param5=11,
        )

        fixed_params = {}
        percentile = 1

        lower, upper = check_bounds(bounds, params, fixed_params, percentile, scale='lin')

        self.assertEqual(['param2', 'param4'], list(lower.keys()))
        self.assertEqual(['param5'], list(upper.keys()))

    def test_check_bounds_log(self):
        """
        Test that check_bounds works as expected for log scale.
        """
        bounds = dict(
            param1=(1, 10),
            param2=(1, 10),
            param3=(-10, -1),
            param4=(-10, -1),
            param5=(-1, 1),
            param6=(-1, 1),
            param7=(-1, 1),
            param8=(-1, 1),
        )

        params = dict(
            param1=1.1,
            param2=1.005,
            param3=-9.9,
            param4=-9.995,
            param5=0,
            param6=0.99,
            param7=-0.99,
            param8=0.1,
        )

        fixed_params = {}
        percentile = 1

        lower, upper = check_bounds(bounds, params, fixed_params, percentile, scale='log')

        self.assertEqual(['param2'], list(lower.keys()))

        # param7 doesn't work for log scale
        self.assertEqual(['param6', 'param8'], list(upper.keys()))

    def test_bounds_admitting_a_single_value(self):
        """
        Equal lower and upper bounds give that value as a sample and are not reported as a bound being
        reached. Sampling failed on the log scale, and the report divided by zero.
        """
        from fastdfe.optimization import Optimization, check_bounds

        self.assertEqual(Optimization.sample_value((0.3, 0.3), 'log', random_state=0), 0.3)
        self.assertEqual(Optimization.sample_value((-2, -2), 'log', random_state=0), -2)
        self.assertEqual(Optimization.sample_value((0.3, 0.3), 'lin', random_state=0), 0.3)

        with np.errstate(all='raise'):
            near_lower, near_upper = check_bounds(
                params={'all.b': 0.3}, bounds=dict(b=(0.3, 0.3)), fixed_params={}, scale='lin'
            )

        self.assertEqual((near_lower, near_upper), ({}, {}))

    def test_expand_fixed_type_takes_precedence_over_all(self):
        """
        A fixed value given for a type takes precedence over the one given for 'all', in either order of
        the dictionary. The later entry won, so 'all' listed last replaced the value of the type.
        """
        from fastdfe.optimization import expand_fixed

        expected = dict(a=dict(b=0.3, h=0.5), c=dict(b=0.1, h=0.5))

        for fixed_params in [
            dict(all=dict(b=0.1, h=0.5), a=dict(b=0.3)),
            dict(a=dict(b=0.3), all=dict(b=0.1, h=0.5))
        ]:
            self.assertEqual(expand_fixed(fixed_params, ['a', 'c']), expected)

    def test_sample_x0_with_integer_seed_draws_parameters_independently(self):
        """
        Initial values sampled with an integer seed are independent draws. Each parameter was drawn from a
        generator newly created from the seed, so all took the same quantile of their bounds.
        """
        names = ['a', 'b', 'c']
        opt = Optimization(
            bounds={k: (1e-3, 1e3) for k in names}, scales={k: 'log' for k in names}, param_names=names
        )

        sample = opt.sample_x0(dict(all=dict(a=1, b=1, c=1)), random_state=5)['all']

        self.assertEqual(len(set(sample.values())), 3)
        self.assertEqual(sample, opt.sample_x0(dict(all=dict(a=1, b=1, c=1)), random_state=5)['all'])

    def test_collapse_fixed_to_mean(self):
        """
        Test the collapse_fixed_to_mean function.
        """
        self.assertDictEqual(
            collapse_fixed_to_mean(
                {
                    'pendula': {'eps': 0, 'h': 0.6, 'p_b': 0, 'S_b': 1},
                    'pubescens': {'eps': 0, 'h': 0.5, 'p_b': 0, 'S_b': 1}
                },
                types=['pendula', 'pubescens']
            ),
            {'all': {'eps': 0, 'h': 0.55, 'p_b': 0, 'S_b': 1}}
        )

        self.assertDictEqual(
            collapse_fixed_to_mean(
                {
                'all': {'b': 1},
                'pendula': {'eps': 0, 'h': 0.6, 'p_b': 0, 'S_b': 1},
                'pubescens': {'eps': 0, 'h': 0.5, 'p_b': 0, 'S_b': 1}
                },
                types=['pendula', 'pubescens']
            ),
            {'all': {'eps': 0, 'h': 0.55, 'p_b': 0, 'S_b': 1, 'b': 1}}
        )

    def test_collapse_fixed(self):
        """
        Test the collapse_fixed function.
        """
        self.assertDictEqual(
            collapse_fixed(dict(
                all=dict(h=0.5),
                pendula=dict(S_b=0.5, p_b=0, S_d=-100),
                pubescens=dict(S_b=0.5, p_b=0.1, b=0.3)),
                types=['pendula', 'pubescens']
            ),
            dict(all=dict(h=0.5, S_b=0.5), pendula=dict(S_d=-100, p_b=0), pubescens=dict(b=0.3, p_b=0.1))
        )

        self.assertDictEqual(
            collapse_fixed(dict(
                all=dict(h=0.5, S_b=0.5),
                pendula=dict(p_b=0, S_d=-100),
                pubescens=dict(p_b=0.1, b=0.3),
                foo=dict(bar=1)),
                types=['pendula', 'pubescens']
            ),
            dict(
                all=dict(h=0.5, S_b=0.5),
                pendula=dict(S_d=-100, p_b=0),
                pubescens=dict(b=0.3, p_b=0.1),
                foo=dict(bar=1)
            )
        )

        self.assertDictEqual(
            collapse_fixed(dict(
                all=dict(h=0.5, S_b=0.5),
                pendula=dict(p_b=0, S_d=-100),
                pubescens=dict(p_b=0.1, b=0.3),
                foo=dict(bar=1)),
                types=[]
            ),
            dict(
                all=dict(h=0.5, S_b=0.5),
                pendula=dict(p_b=0, S_d=-100),
                pubescens=dict(p_b=0.1, b=0.3),
                foo=dict(bar=1),
            )
        )
