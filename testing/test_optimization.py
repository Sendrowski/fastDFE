from testing import prioritize_installed_packages

prioritize_installed_packages()

from typing import Literal, List
from unittest import TestCase

import matplotlib.pyplot as plt
import numpy as np

from fastdfe.optimization import pack_params, unpack_params, flatten_dict, unflatten_dict, filter_dict, Optimization, \
    merge_dicts, correct_values, to_symlog, from_symlog, scale_bound, scale_value, unscale_value, unscale_bound

from numpy import testing


class OptimizationTestCase(TestCase):

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
        d = {'a': 1, 'b': {'c': 2, 'd': {'e': 3}}}
        expected = {'a': 1, 'b.c': 2, 'b.d.e': 3}
        assert flatten_dict(d) == expected

        d = {'a': {'b': {'c': 1}}}
        expected = {'a.b.c': 1}
        assert flatten_dict(d) == expected

    def test_unflatten_dict(self):
        d = {'a': 1, 'b.c': 2, 'b.d.e': 3}
        expected = {'a': 1, 'b': {'c': 2, 'd': {'e': 3}}}
        assert unflatten_dict(d) == expected

        d = {'a.b.c': 1}
        expected = {'a': {'b': {'c': 1}}}
        assert unflatten_dict(d) == expected

    def test_flatten_unflatten_dict(self):
        d = {'a': 1, 'b': {'c': 2, 'd': {'e': 3}}}
        expected = d
        flattened = flatten_dict(d)
        unflattened = unflatten_dict(flattened)
        assert unflattened == expected

    def test_filter_dict(self):
        d = {'a': 1, 'b': {'c': 2, 'd': {'e': 3, 'f': {'g': 4}}}, 'c': {'g': 1, 'b': 2}}
        keys = ['a', 'c']
        expected = {'a': 1, 'b': {'c': 2}}
        assert filter_dict(d, keys) == expected

    def test_sample_x0_nested_dict(self):
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
        bounds = (1, 10)
        scale = 'lin'
        for _ in range(100):
            sample = Optimization.sample_value(bounds, scale)
            self.assertGreaterEqual(sample, bounds[0])
            self.assertLessEqual(sample, bounds[1])

    def test_negative_bounds(self):
        bounds = (-10, -1)
        scale = 'lin'
        for _ in range(100):
            sample = Optimization.sample_value(bounds, scale)
            self.assertGreaterEqual(sample, bounds[0])
            self.assertLessEqual(sample, bounds[1])

    def test_mixed_bounds(self):
        bounds = (-5, 5)
        scale = 'lin'
        for _ in range(100):
            sample = Optimization.sample_value(bounds, scale)
            self.assertGreaterEqual(sample, bounds[0])
            self.assertLessEqual(sample, bounds[1])

    def test_positive_bounds_log_scale(self):
        bounds = (1, 10)
        scale = 'log'
        for _ in range(100):
            sample = Optimization.sample_value(bounds, scale)
            self.assertGreaterEqual(sample, bounds[0])
            self.assertLessEqual(sample, bounds[1])

    def test_negative_bounds_log_scale(self):
        bounds = (-10, -1)
        scale = 'log'
        for _ in range(100):
            sample = Optimization.sample_value(bounds, scale)
            self.assertGreaterEqual(sample, bounds[0])
            self.assertLessEqual(sample, bounds[1])

    def test_mixed_bounds_log_scale(self):
        bounds = (-5, 5)
        scale = 'log'
        with self.assertRaises(ValueError):
            Optimization.sample_value(bounds, scale)

    def test_correct_x0(self):
        bounds = {
            "a": (0, 10),
            "b": (-5, 5),
            "c": (1, 100)
        }

        x0 = {
            "some.prefix.a": -2,
            "another.prefix.b": 7,
            "yet.another.prefix.c": 50
        }

        expected_corrected = {
            "some.prefix.a": 0,
            "another.prefix.b": 5,
            "yet.another.prefix.c": 50
        }

        corrected = correct_values(x0, bounds)
        self.assertEqual(corrected, expected_corrected)

        x0_no_correction = {
            "some.prefix.a": 4,
            "another.prefix.b": -3,
            "yet.another.prefix.c": 30
        }

        corrected_no_correction = correct_values(x0_no_correction, bounds)
        self.assertEqual(corrected_no_correction, x0_no_correction)

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

        symlog_bounds = (-2.0043213737826426, 2.0043213737826426)
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
