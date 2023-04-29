from unittest import TestCase

from fastdfe.optimization import pack_params, unpack_params, flatten_dict, unflatten_dict, filter_dict, Optimization, \
    merge_dicts, correct_values


class OptimizationTestCase(TestCase):
    """
    TODO there might be less padding for np.float128 on windows machines
    cd. https://numpy.org/doc/stable/user/basics.types.html
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
