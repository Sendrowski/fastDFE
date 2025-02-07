import copy
import logging
from abc import ABC
from unittest import mock

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.random._generator import Generator
from pandas.testing import assert_frame_equal
from scipy.optimize import OptimizeResult

import fastdfe as fd
from fastdfe.discretization import Discretization
from fastdfe.polydfe import PolyDFE
from testing import TestCase


class InferenceTestCase(TestCase, ABC):
    """
    Test inference.
    """

    def assertEqualInference(self, obj1: object, obj2: object, ignore_keys=[]):
        """
        Compare Inference objects, recursively comparing their attributes.
        :param obj1: First object
        :param obj2: Second object
        :param ignore_keys: Keys to ignore
        """
        ignore_keys += ['_logger']

        if not hasattr(obj1, '__dict__'):
            self.assertEqual(obj1, obj2)
        else:

            d1 = obj1.__dict__
            d2 = obj2.__dict__

            for (key1, value1), (key2, value2) in zip(d1.items(), d2.items()):
                if key1 in ignore_keys:
                    pass

                elif isinstance(value1, OptimizeResult):
                    pass

                elif isinstance(value1, dict):
                    self.assertEqual(value1.keys(), value2.keys())

                    for k in value1.keys():
                        self.assertEqualInference(value1[k], value2[k], ignore_keys=ignore_keys)

                elif isinstance(value1, (int, float, str, bool, tuple, list, type(None), np.bool_)):
                    self.assertEqual(value1, value2)

                elif isinstance(value1, np.ndarray):
                    np.testing.assert_equal(value1, value2)

                elif isinstance(value1, pd.DataFrame):
                    pd.testing.assert_frame_equal(value1.reset_index(drop=True), value2.reset_index(drop=True))

                elif isinstance(value1, Generator):
                    pass

                elif isinstance(value1, fd.Spectrum):
                    np.testing.assert_equal(value1.to_list(), value2.to_list())

                elif isinstance(value1, object):
                    self.assertEqualInference(value1, value2, ignore_keys=ignore_keys)

                else:
                    raise AssertionError('Some objects were not compared.')


class PolyDFETestCase(InferenceTestCase):
    """
    Test results against cached PolyDFE results.
    """

    # configs for testing against cached polydfe results
    configs_polydfe = [
        'pendula_C_full_bootstrapped_100',
        'pendula_C_full_anc_bootstrapped_100',
        'pendula_C_deleterious_bootstrapped_100',
        'pendula_C_deleterious_anc_bootstrapped_100',
        'pubescens_C_full_bootstrapped_100',
        'pubescens_C_full_anc_bootstrapped_100',
        'pubescens_C_deleterious_bootstrapped_100',
        'pubescens_C_deleterious_anc_bootstrapped_100',
        'example_1_C_full_bootstrapped_100',
        'example_1_C_full_anc_bootstrapped_100',
        # 'example_1_C_deleterious_bootstrapped_100' # polydfe does not finish in this case
        'example_1_C_deleterious_anc_bootstrapped_100',
        'example_2_C_full_bootstrapped_100',
        'example_2_C_full_anc_bootstrapped_100',
        'example_2_C_deleterious_bootstrapped_100',
        'example_2_C_deleterious_anc_bootstrapped_100',
        'example_3_C_full_bootstrapped_100',
        'example_3_C_full_anc_bootstrapped_100',
        'example_3_C_deleterious_bootstrapped_100',
        'example_3_C_deleterious_anc_bootstrapped_100'
    ]

    def compare_with_polydfe(self, config: str):
        """
        Test whether the results are similar to those of PolyDFE.
        """
        conf = fd.Config.from_file(f"testing/cache/configs/{config}/config.yaml")
        inf_polydfe = PolyDFE.from_file(f"testing/cache/polydfe/{config}/serialized.json")

        inf_fastdfe = fd.BaseInference.from_config(conf)
        inf_fastdfe.run()

        # compare discretized DFE
        dfe_fastdfe = inf_fastdfe.get_discretized()
        dfe_polydfe = inf_polydfe.get_discretized()

        ci_fastdfe = np.array([dfe_fastdfe[0] - dfe_fastdfe[1][0], dfe_fastdfe[0] + dfe_fastdfe[1][1]])
        ci_polydfe = np.array([dfe_polydfe[0] - dfe_polydfe[1][0], dfe_polydfe[0] + dfe_polydfe[1][1]])

        plt.clf()
        _, axs = plt.subplots(1, 2, figsize=(10, 4))
        inf_fastdfe.plot_discretized(ax=axs[0], show=False, title='fastdfe')
        inf_polydfe.plot_discretized(ax=axs[1], show=True, title='polydfe')

        # assert that the confidence intervals almost overlap
        assert np.all(ci_fastdfe[0] <= ci_polydfe[1] + 0.05)
        assert np.all(ci_polydfe[0] <= ci_fastdfe[1] + 0.05)

    @staticmethod
    def generate_compare_with_polydfe(config: str) -> callable:
        """
        Generate test for comparing with PolyDFE.
        """

        def compare_with_polydfe(self):
            """
            Compare with PolyDFE.

            :param self: Self
            """
            self.compare_with_polydfe(config)

        return compare_with_polydfe

    # dynamically generate tests
    for config in configs_polydfe:
        locals()[f'test_compare_with_polydfe_{config}'] = generate_compare_with_polydfe(config)


class BaseInferenceTestCase(InferenceTestCase):
    """
    Test BaseInference.
    """
    config_file = "testing/cache/configs/pendula_C_full_anc/config.yaml"
    serialized = "testing/cache/fastdfe/pendula_C_full_anc/serialized.json"

    def test_run_inference_from_config_parallelized(self):
        """
        Run inference from config file.
        """
        config = fd.Config.from_file(self.config_file)
        config.update(parallelize=True)

        inference = fd.BaseInference.from_config(config)
        inference.run()

    def test_run_inference_from_config_not_parallelized(self):
        """
        Run inference from config file.
        """
        config = fd.Config.from_file(self.config_file)
        config.update(parallelize=False)

        inference = fd.BaseInference.from_config(config)
        inference.run()

    def test_seeded_inference_is_deterministic_non_parallelized(self):
        """
        Check that inference is deterministic when seeded and not parallelized.
        """
        config = fd.Config.from_file(self.config_file)

        config.update(
            seed=0,
            do_bootstrap=True,
            parallelize=False,
            n_bootstraps=2
        )

        inference = fd.BaseInference.from_config(config)
        inference.run()

        inference2 = fd.BaseInference.from_config(config)
        inference2.run()

        self.assertEqual(inference.params_mle, inference2.params_mle)
        assert_frame_equal(inference.bootstraps, inference2.bootstraps)

    def test_seeded_inference_is_deterministic_parallelized(self):
        """
        Check that seeded inference is deterministic when parallelized.
        """
        config = fd.Config.from_file(self.config_file)

        config.update(
            seed=0,
            do_bootstrap=True,
            parallelize=True,
            n_bootstraps=2
        )

        inference = fd.BaseInference.from_config(config)
        inference.run()

        inference2 = fd.BaseInference.from_config(config)
        inference2.run()

        self.assertEqual(inference.params_mle, inference2.params_mle)
        assert_frame_equal(inference.bootstraps, inference2.bootstraps)

    def test_compare_inference_with_log_scales_vs_lin_scales(self):
        """
        Compare inference with log scales vs linear scales.
        """
        config = fd.Config.from_file(self.config_file)

        model = fd.GammaExpParametrization()
        model.scales = dict(
            S_d='log',
            b='log',
            p_b='lin',
            S_b='log'
        )

        config.update(
            model=model,
            do_bootstrap=True
        )

        inference_log = fd.BaseInference.from_config(config)
        inference_log.run()

        model = copy.copy(model)
        model.scales = dict(
            S_d='lin',
            b='lin',
            p_b='lin',
            S_b='lin'
        )

        config.update(model=model)
        inference_lin = fd.BaseInference.from_config(config)
        inference_lin.run()

        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        inference_lin.plot_inferred_parameters(ax=axs[0], show=False)
        inference_log.plot_inferred_parameters(ax=axs[1], show=False)

        axs[0].set_title('Linear scales')
        axs[1].set_title('Log scales')

        plt.tight_layout()
        plt.show()

        # get confidence intervals
        cis_lin = inference_lin.get_errors_discretized_dfe()[1]
        cis_log = inference_log.get_errors_discretized_dfe()[1]

        # assert that the confidence intervals overlap
        assert np.all(cis_lin[0] < cis_log[1])
        assert np.all(cis_log[0] < cis_lin[1])

        # assert inference_log.likelihood > inference_lin.likelihood

        self.assertAlmostEqual(inference_log.likelihood, inference_lin.likelihood, places=0)

    def test_compare_inference_with_log_scales_vs_lin_scales_tutorial(self):
        """
        Compare inference with log scales vs linear scales.
        """
        model = fd.GammaExpParametrization()
        model.scales = dict(
            S_d='log',
            b='log',
            p_b='lin',
            S_b='log'
        )

        inference_log = fd.BaseInference(
            sfs_neut=fd.Spectrum([177130, 997, 441, 228, 156, 117, 114, 83, 105, 109, 652]),
            sfs_sel=fd.Spectrum([797939, 1329, 499, 265, 162, 104, 117, 90, 94, 119, 794]),
            model=model,
            do_bootstrap=True,
            n_runs=10
        )
        inference_log.run()

        model = fd.GammaExpParametrization()
        model.scales = dict(
            S_d='lin',
            b='lin',
            p_b='lin',
            S_b='lin'
        )

        inference_lin = fd.BaseInference(
            sfs_neut=fd.Spectrum([177130, 997, 441, 228, 156, 117, 114, 83, 105, 109, 652]),
            sfs_sel=fd.Spectrum([797939, 1329, 499, 265, 162, 104, 117, 90, 94, 119, 794]),
            model=model,
            do_bootstrap=True,
            n_runs=10
        )
        inference_lin.run()

        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        inference_lin.plot_discretized(ax=axs[0], show=False)
        inference_log.plot_discretized(ax=axs[1], show=False)

        axs[0].set_title('Linear scales')
        axs[1].set_title('Log scales')

        plt.tight_layout()
        plt.show()

        # get confidence intervals
        cis_lin = inference_lin.get_errors_discretized_dfe()[1]
        cis_log = inference_log.get_errors_discretized_dfe()[1]

        # assert that the confidence intervals overlap
        # assert np.all(cis_lin[0] < cis_log[1])
        # assert np.all(cis_log[0] < cis_lin[1])

        # not always true
        # assert inference_log.likelihood > inference_lin.likelihood

        # using log scales appears to provide a lower likelihood in general
        # but things become more equal when increasing ``n_runs``
        self.assertAlmostEqual(inference_log.likelihood, inference_lin.likelihood, places=0)

        pass

    def test_restore_serialized_inference(self):
        """
        Check whether Inference can properly be serialized and restored.
        """
        inference = fd.BaseInference.from_config_file(self.config_file)
        inference.run()

        serialized = 'scratch/test_serialize_inference.json'
        inference.to_file(serialized)
        inference_restored = fd.BaseInference.from_file(serialized)

        # self.assertEqual(inference.to_json(), inference_restored.to_json())
        self.assertEqualInference(inference, inference_restored)

    def test_inference_run_bootstrap(self):
        """
        Run bootstrap on inference.
        """
        # unserialize
        inference = fd.BaseInference.from_file(self.serialized)

        # make sure the likelihoods of the different runs are cached
        self.assertEqual(len(inference.likelihoods), inference.n_runs)

        self.assertIsNone(inference.bootstraps)

        inference.bootstrap(2, parallelize=False)

        self.assertIsNotNone(inference.bootstraps)

        # make sure the likelihoods of the different runs are not overwritten
        self.assertEqual(len(inference.likelihoods), inference.n_runs)

    def test_evaluate_likelihood_same_as_mle_results(self):
        """
        Check that evaluating the loss function at the MLE result
        yields the same likelihood as the one reported.
        """
        # unserialize
        inference = fd.BaseInference.from_file(self.serialized)

        self.assertAlmostEqual(inference.evaluate_likelihood(dict(all=inference.params_mle)), inference.likelihood)

    def test_visualize_inference_without_bootstraps(self):
        """
        Plot everything possible.
        """
        config = fd.Config.from_file(self.config_file)

        config.update(
            do_bootstrap=False
        )

        inference = fd.BaseInference.from_config(config)

        inference.plot_all()
        inference.plot_bucket_sizes()

    def test_visualize_inference_with_bootstraps(self):
        """
        Plot everything possible.
        """
        config = fd.Config.from_file(self.config_file)

        config.update(
            do_bootstrap=True
        )

        inference = fd.BaseInference.from_config(config)

        inference.plot_all()
        inference.plot_bucket_sizes()

    def test_plot_likelihoods(self):
        """
        Plot likelihoods.
        """
        # unserialize
        inference = fd.BaseInference.from_file(self.serialized)

        inference.plot_likelihoods()

    def test_plot_interval_density(self):
        """
        Plot likelihoods.
        """
        # unserialize
        inference = fd.BaseInference.from_file(self.serialized)

        inference.plot_interval_density()

    def test_plot_inferred_parameters(self):
        """
        Plot inferred parameters.
        """
        # unserialize
        inference = fd.BaseInference.from_file(self.serialized)

        inference.plot_inferred_parameters()

    def test_plot_inferred_parameters_boxplot(self):
        """
        Plot inferred parameters.
        """
        # unserialize
        inference = fd.BaseInference.from_file(self.serialized)

        inference.bootstrap(6)

        inference.plot_inferred_parameters_boxplot()

    def test_plot_observed_sfs(self):
        """
        Plot likelihoods.
        """
        # unserialize
        inference = fd.BaseInference.from_file(self.serialized)

        inference.plot_observed_sfs()

    def test_plot_provide_axes(self):
        """
        Plot likelihoods.
        """
        # unserialize
        inference = fd.BaseInference.from_file(self.serialized)

        _, axs = plt.subplots(ncols=2)

        inference.plot_interval_density(show=False, ax=axs[0])
        inference.plot_interval_density(ax=axs[1])

    def test_visualize_inference_bootstrap(self):
        """
        Plot everything possible.
        """
        # unserialize
        inference = fd.BaseInference.from_file(self.serialized)

        # bootstrap
        inference.bootstrap(2)

        inference.plot_all()

    def test_perform_bootstrap_without_running(self):
        """
        Perform bootstrap before having run the main inference.
        """
        # unserialize
        inference = fd.BaseInference.from_config_file(self.config_file)

        # bootstrap
        inference.bootstrap(2)

    def test_compare_nested_with_bootstrap(self):
        """
        Compare nested likelihoods.
        """
        # unserialize
        inference = fd.BaseInference.from_config_file(self.config_file)

        inference.compare_nested_models()

        inference.plot_nested_models()
        inference.plot_nested_models(remove_empty=True)
        inference.plot_nested_models(transpose=True)

    def test_compare_nested_without_bootstrap(self):
        """
        Compare nested likelihoods.
        """
        # unserialize
        inference = fd.BaseInference.from_config_file(self.config_file)

        inference.compare_nested_models(do_bootstrap=False)

        inference.plot_nested_models(do_bootstrap=False)

    def test_compare_nested_different_parametrizations_raises_error(self):
        """
        Make sure that comparing nested likelihoods with different parametrizations raises an error.
        """
        config1 = fd.Config.from_file(self.config_file)
        config1.update(model=fd.GammaExpParametrization())
        inf1 = fd.BaseInference.from_config(config1)

        config2 = fd.Config.from_file(self.config_file)
        config2.update(model=fd.DiscreteParametrization())
        inf2 = fd.BaseInference.from_config(config2)

        with self.assertRaises(ValueError):
            inf1.compare_nested(inf2)

    def test_compare_nested_different_fixed_params_return_none(self):
        """
        Make sure that comparing nested likelihoods with different fixed parameters returns None.
        """
        config1 = fd.Config.from_file(self.config_file)
        config1.update(fixed_params=dict(all=dict(S_b=1, p_b=0)))
        inf1 = fd.BaseInference.from_config(config1)

        config2 = fd.Config.from_file(self.config_file)
        config2.update(fixed_params=dict(all=dict(S_b=1, eps=0)))
        inf2 = fd.BaseInference.from_config(config2)

        self.assertIsNone(inf1.compare_nested(inf2))

    def test_compare_nested_valid_comparison_1df(self):
        """
        Make sure that comparing nested models with the same parametrization works as expected.
        """
        config1 = fd.Config.from_file(self.config_file)
        config1.update(fixed_params=dict(all=dict(S_b=1, p_b=0)), do_bootstrap=False)
        inf1 = fd.BaseInference.from_config(config1)
        inf1.run()

        config2 = fd.Config.from_file(self.config_file)
        config2.update(fixed_params=dict(all=dict(S_b=1)), do_bootstrap=False)
        inf2 = fd.BaseInference.from_config(config2)
        inf2.run()

        assert 0 < inf1.compare_nested(inf2) < 1

    def test_compare_nested_valid_automatic_run(self):
        """
        Make sure that comparing nested models with the same parametrization works as expected.
        """
        config1 = fd.Config.from_file(self.config_file)
        config1.update(fixed_params=dict(all=dict(S_b=1, p_b=0)), do_bootstrap=False)
        inf1 = fd.BaseInference.from_config(config1)

        config2 = fd.Config.from_file(self.config_file)
        config2.update(fixed_params=dict(all=dict(S_b=1)), do_bootstrap=False)
        inf2 = fd.BaseInference.from_config(config2)

        assert 0 < inf1.compare_nested(inf2) < 1

    def test_plot_nested_model_comparison_without_running(self):
        """
        Perform nested model comparison before having run the main inference.
        We just make sure this triggers a run.
        """
        # unserialize
        inference = fd.BaseInference.from_config_file(self.config_file)

        inference.plot_nested_models()

    def test_plot_nested_model_comparison_different_parametrizations(self):
        """
        Perform nested model comparison before having run the main inference.
        We just make sure this triggers a run.
        """
        parametrizations = [
            fd.GammaExpParametrization(),
            fd.DisplacedGammaParametrization(),
            fd.DiscreteParametrization(),
            fd.DiscreteFractionalParametrization(),
            fd.GammaDiscreteParametrization()
        ]

        for p in parametrizations:
            config = fd.Config.from_file(self.config_file).update(
                model=p,
                do_bootstrap=True,
                n_bootstraps=2
            )

            # unserialize
            inference = fd.BaseInference.from_config(config)

            inference.plot_nested_models(title=p.__class__.__name__)

    def test_recreate_from_config(self):
        """
        Test whether inference can be recreated from config.
        """
        inf = fd.BaseInference.from_config_file(self.config_file)

        inf2 = fd.BaseInference.from_config(inf.create_config())

        self.assertEqualInference(inf, inf2)

    def test_cached_result(self):
        """
        Test inference against cached result.
        """
        inference = fd.BaseInference.from_file(self.serialized)
        inference2 = fd.BaseInference.from_config(inference.create_config())

        inference2.run()

        inference2.to_file("scratch/test_cached_result.json")
        inference2.get_summary().to_file("scratch/test_cached_result_summary.json")

        # inference.get_summary().to_file("scratch/test_cached_result_summary_actual.json")

        def get_dict(inf: fd.BaseInference) -> dict:
            """

            :param inf:
            :return:
            """
            exclude = ['execution_time', 'result']

            return dict((k, v) for k, v in inference.get_summary().__dict__.items() if k not in exclude)

        self.assertDictEqual(get_dict(inference), get_dict(inference2))

    def test_run_if_necessary_wrapper_triggers_run(self):
        """
        Test whether the run_if_necessary wrapper triggers a run if the inference has not been run yet.
        """
        inference = fd.BaseInference.from_config_file(self.config_file)

        with mock.patch.object(fd.BaseInference, 'run', side_effect=Exception()) as mock_run:
            try:
                inference.plot_inferred_parameters(show=False)
            except Exception:
                pass

            mock_run.assert_called_once()

    def test_run_if_necessary_wrapper_not_triggers_run(self):
        """
        Test whether the run_if_necessary wrapper does not trigger a run if the inference has already been run.
        """
        inference = fd.BaseInference.from_file(self.serialized)

        with mock.patch.object(fd.BaseInference, 'run') as mock_run:
            inference.plot_inferred_parameters(show=False)
            mock_run.assert_not_called()

    def test_run_with_sfs_neut_zero_entry(self):
        """
        Test whether inference can be run with a neutral SFS that has a zero entry.
        """
        config = fd.Config.from_file(self.config_file)

        # set neutral SFS entry to zero
        sfs = config.data['sfs_neut']['all'].to_list()
        sfs[3] = 0
        sfs = fd.Spectrum(sfs)
        config.data['sfs_neut'] = fd.Spectra.from_spectrum(sfs)

        inference = fd.BaseInference.from_config(config)

        inference.run()

        # here both the nuisance parameters and epsilon cause the high number of high-frequency variants
        inference.plot_sfs_comparison()

        inference.plot_discretized()

    def test_run_with_sfs_sel_zero_entry(self):
        """
        Test whether inference can be run with a selected SFS that has a zero entry.
        """
        config = fd.Config.from_file(self.config_file)

        # set selected SFS entry to zero
        sfs = config.data['sfs_sel']['all'].to_list()
        sfs[3] = 0
        sfs = fd.Spectrum(sfs)
        config.data['sfs_sel'] = fd.Spectra.from_spectrum(sfs)

        inference = fd.BaseInference.from_config(config)

        inference.run()

        # this looks as expected, i.e. we reduce the overall distance between the modelled and observed SFS
        inference.plot_sfs_comparison()

        inference.plot_discretized()

    def test_run_with_both_sfs_zero_entry(self):
        """
        Test whether inference can be run with both SFS that have a zero entry.
        """
        config = fd.Config.from_file(self.config_file)

        # set neutral SFS entry to zero
        sfs = config.data['sfs_neut']['all'].to_list()
        sfs[3] = 0
        sfs = fd.Spectrum(sfs)
        config.data['sfs_neut'] = fd.Spectra.from_spectrum(sfs)

        # set selected SFS entry to zero
        sfs = config.data['sfs_sel']['all'].to_list()
        sfs[3] = 0
        sfs = fd.Spectrum(sfs)
        config.data['sfs_sel'] = fd.Spectra.from_spectrum(sfs)

        inference = fd.BaseInference.from_config(config)

        inference.run()

        # the nuisance parameters cause the zero entry in the modelled SFS
        inference.plot_sfs_comparison()

        inference.plot_discretized()

    def test_fixed_parameters(self):
        """
        Test whether fixed parameters are correctly set.
        """
        config = fd.Config.from_file(self.config_file)

        config.data['fixed_params'] = dict(all=dict(b=1.123, eps=0.123))

        inference = fd.BaseInference.from_config(config)

        inference.run()

        assert inference.params_mle['b'] == config.data['fixed_params']['all']['b']
        assert inference.params_mle['eps'] == config.data['fixed_params']['all']['eps']
        assert inference.params_mle['S_b'] != config.data['x0']['all']['S_b']
        assert inference.params_mle['S_d'] != config.data['x0']['all']['S_d']
        assert inference.params_mle['p_b'] != config.data['x0']['all']['p_b']

    def test_get_n_optimized(self):
        """
        Test get number of parameters to be optimized
        """
        config = fd.Config.from_file(self.config_file)

        assert fd.BaseInference.from_config(config).get_n_optimized() == 5

        config.data['fixed_params'] = dict(all=dict(S_b=1, p_b=0))

        assert fd.BaseInference.from_config(config).get_n_optimized() == 3

    def test_non_existing_fixed_param_raises_error(self):
        """
        Test that a non-existing fixed parameter raises an error.
        """
        config = fd.Config.from_file(self.config_file)

        config.data['fixed_params'] = dict(all=dict(S_b=1, p_b=0, foo=1))

        with self.assertRaises(ValueError):
            fd.BaseInference.from_config(config)

    def test_get_cis_params_mle_no_bootstraps(self):
        """
        Get the MLE parameters from the cis inference.
        """
        inference = fd.BaseInference.from_file(self.serialized)

        cis = inference.get_cis_params_mle()

        assert cis is None

    def test_get_cis_params_mle_with_bootstraps(self):
        """
        Get the MLE parameters from the cis inference.
        """
        inference = fd.BaseInference.from_file(self.serialized)

        inference.bootstrap()

        cis = inference.get_cis_params_mle()

    def test_get_discretized_errors_with_bootstraps(self):
        """
        Get the discretized DFE errors.
        """
        inference = fd.BaseInference.from_file(self.serialized)

        inference.bootstrap()

        res = inference.get_discretized()

    def test_get_discretized_errors_no_bootstraps(self):
        """
        Get the discretized DFE errors.
        """
        inference = fd.BaseInference.from_file(self.serialized)

        values, errors = inference.get_discretized()

        assert errors is None

    def test_spectrum_with_zero_monomorphic_counts_throws_error(self):
        """
        Test whether a spectrum with zero monomorphic counts throws an error.
        """
        sfs_neut = fd.Spectrum([0, 997, 441, 228, 156, 117, 114, 83, 105, 109, 652])
        sfs_sel = fd.Spectrum([797939, 1329, 499, 265, 162, 104, 117, 90, 94, 119, 794])

        with self.assertRaises(ValueError):
            fd.BaseInference(
                sfs_neut=sfs_neut,
                sfs_sel=sfs_sel,
            )

    def test_spectrum_with_few_monomorphic_counts_raises_warning(self):
        """
        Test whether a spectrum with zero monomorphic counts throws an error.
        """
        sfs_neut = fd.Spectrum([1000, 997, 441, 228, 156, 117, 114, 83, 105, 109, 652])
        sfs_sel = fd.Spectrum([797939, 1329, 499, 265, 162, 104, 117, 90, 94, 119, 794])

        with self.assertLogs(level="WARNING", logger=logging.getLogger('fastdfe')):
            fd.BaseInference(
                sfs_neut=sfs_neut,
                sfs_sel=sfs_sel,
            )

    @staticmethod
    def test_adjust_polarization():
        """
        Test whether polarization is adjusted correctly.
        """
        counts_list = [np.array([1, 2, 3, 4, 5]), np.array([2, 3, 5, 7, 11])]
        eps_list = [0, 0.1, 0.5, 1]

        expected_counts_list = [
            [
                np.array([1, 2, 3, 4, 5]),
                np.array([1.4, 2.2, 3.0, 3.8, 4.6]),
                np.array([3.0, 3.0, 3.0, 3.0, 3.0]),
                np.array([5, 4, 3, 2, 1])
            ],
            [
                np.array([2, 3, 5, 7, 11]),
                np.array([2.9, 3.4, 5, 6.6, 10.1]),
                np.array([6.5, 5.0, 5.0, 5.0, 6.5]),
                np.array([11, 7, 5, 3, 2])
            ]
        ]

        for i, counts in enumerate(counts_list):
            for j, eps in enumerate(eps_list):
                adjusted_counts = fd.BaseInference._adjust_polarization(counts, eps)
                assert np.isclose(np.sum(counts), np.sum(adjusted_counts))
                assert np.allclose(adjusted_counts, expected_counts_list[i][j])

    def test_estimating_full_dfe_with_folded_sfs_raises_warning(self):
        """
        Test whether estimating the full DFE with a folded SFS raises a warning.
        """
        sfs_neut = fd.Spectrum([1, 1, 0, 0])
        sfs_sel = fd.Spectrum([1, 1, 0, 0])

        with self.assertLogs(level="WARNING", logger=logging.getLogger('fastdfe')):
            fd.BaseInference(
                sfs_neut=sfs_neut,
                sfs_sel=sfs_sel,
            )

    @staticmethod
    def test_folded_inference_even_sample_size():
        """
        Test whether a spectrum with zero monomorphic counts throws an error.
        """
        sfs_neut = fd.Spectrum([177130, 997, 441, 228, 156, 117, 114, 83, 105, 109, 652]).fold()
        sfs_sel = fd.Spectrum([797939, 1329, 499, 265, 162, 104, 117, 90, 94, 119, 794]).fold()

        inf = fd.BaseInference(
            sfs_neut=sfs_neut,
            sfs_sel=sfs_sel,
            fixed_params=dict(all=dict(S_b=1, p_b=0))
        )

        inf.plot_discretized()

    @staticmethod
    def test_folded_inference_odd_sample_size():
        """
        Test whether a spectrum with zero monomorphic counts throws an error.
        """
        sfs_neut = fd.Spectrum([177130, 997, 441, 228, 156, 117, 114, 83, 105, 652]).fold()
        sfs_sel = fd.Spectrum([797939, 1329, 499, 265, 162, 104, 117, 90, 94, 794]).fold()

        inf = fd.BaseInference(
            sfs_neut=sfs_neut,
            sfs_sel=sfs_sel,
            fixed_params=dict(all=dict(S_b=1, p_b=0))
        )

        inf.plot_discretized()

        pass

    def test_sample_data_fixed_result(self):
        """
        Test whether a spectrum with zero monomorphic counts throws an error.
        """
        sfs_neut = fd.Spectrum([177130, 997, 441, 228, 156, 117, 114, 83, 105, 109, 652])
        sfs_sel = fd.Spectrum([797939, 1329, 499, 265, 162, 104, 117, 90, 94, 119, 794])

        inf = fd.BaseInference(
            sfs_neut=sfs_neut,
            sfs_sel=sfs_sel
        )

        inf.run()

        expected = {
            'S_b': 0.0004270,
            'S_d': -9868.141535,
            'b': 0.150810,
            'eps': 0.006854,
            'p_b': 0.0
        }

        self.assertAlmostEqual(inf.params_mle['S_b'], expected['S_b'], delta=1e-3)
        self.assertAlmostEqual(inf.params_mle['S_d'], expected['S_d'], delta=2e-1)
        self.assertAlmostEqual(inf.params_mle['b'], expected['b'], delta=1e-2)
        self.assertAlmostEqual(inf.params_mle['eps'], expected['eps'], delta=1e-3)
        self.assertAlmostEqual(inf.params_mle['p_b'], expected['p_b'], delta=1e-4)

    def test_base_inference_l2_loss_type(self):
        """
        Test whether the L2 loss type is correctly set.
        """
        sfs_neut = fd.Spectrum([177130, 997, 441, 228, 156, 117, 114, 83, 105, 109, 652])
        sfs_sel = fd.Spectrum([797939, 1329, 499, 265, 162, 104, 117, 90, 94, 119, 794])

        inf = fd.BaseInference(
            sfs_neut=sfs_neut,
            sfs_sel=sfs_sel,
            loss_type='L2',
            do_bootstrap=True,
            n_bootstraps=5
        )

        inf.run()

    @staticmethod
    def test_few_polymorphic_sites_raises_no_error():
        """
        Test whether a spectrum with few polymorphic sites raises no error.
        """
        inf = fd.BaseInference(
            sfs_neut=fd.Spectrum([1243, 0, 0, 1, 0, 0, 0]),
            sfs_sel=fd.Spectrum([12421, 0, 0, 0, 0, 1, 0]),
        )

        # There are sometimes problems with the optimization for spectra like these,
        # but this should be nothing to worry about
        inf.run()

        inf.plot_discretized()

        pass

    def test_no_polymorphic_sites_raises_error(self):
        """
        Test whether a spectrum with zero monomorphic counts throws an error.
        """
        with self.assertRaises(ValueError) as context:
            fd.BaseInference(
                sfs_neut=fd.Spectrum([1243, 0, 0, 0, 0, 0, 0]),
                sfs_sel=fd.Spectrum([12421, 0, 0, 0, 0, 0, 0]),
            )

        print(context.exception)

    def test_different_sfs_sample_sizes_raises_error(self):
        """
        Test whether a spectrum with zero monomorphic counts throws an error.
        """
        with self.assertRaises(ValueError) as context:
            fd.BaseInference(
                sfs_neut=fd.Spectrum([1000, 4, 2, 1]),
                sfs_sel=fd.Spectrum([1000, 4, 2, 1, 0])
            )

        print(context.exception)

    @staticmethod
    def test_spectra_with_fractional_counts():
        """
        Test whether a spectrum with fractional counts works as expected.
        """
        sfs_neut = fd.Spectrum([177130.4, 997.3, 441.2, 228.1, 156.45, 117.2, 114.9, 83.12, 105.453, 109.1, 652.05])
        sfs_sel = fd.Spectrum([797939.4, 1329.3, 499.2, 265.1, 162.12, 104.2, 117.9, 90.12, 94.453, 119.1, 794.05])

        inf_float = fd.BaseInference(
            sfs_neut=sfs_neut,
            sfs_sel=sfs_sel
        )

        inf_float.run()

        inf_int = fd.BaseInference(
            sfs_neut=fd.Spectrum(sfs_neut.data.astype(int)),
            sfs_sel=fd.Spectrum(sfs_sel.data.astype(int)),
        )

        inf_int.run()

        fd.Inference.plot_discretized([inf_float, inf_int], labels=['float', 'int'])

        assert np.abs((inf_float.params_mle['S_d'] - inf_int.params_mle['S_d']) / inf_float.params_mle['S_d']) < 1e-2
        assert np.abs((inf_float.params_mle['b'] - inf_int.params_mle['b']) / inf_float.params_mle['b']) < 1e-2

    @staticmethod
    def test_manuscript_example():
        """
        Test the example from the manuscript.
        """
        # create inference object
        inf = fd.BaseInference(
            sfs_neut=fd.Spectrum([66200, 410, 120, 60, 42, 43, 52, 65, 0]),
            sfs_sel=fd.Spectrum([281937, 600, 180, 87, 51, 43, 49, 61, 0]),
            model=fd.GammaExpParametrization(),  # the model to use
            n_runs=10,  # number of optimization runs
            n_bootstraps=100,  # number of bootstrap replicates
            do_bootstrap=True
        )

        # create subplots
        axs = plt.subplots(2, 2, figsize=(11, 7))[1].flatten()

        # plot results
        types = ['neutral', 'selected']
        inf.plot_sfs_comparison(ax=axs[0], show=False, sfs_types=types)
        inf.plot_sfs_comparison(ax=axs[1], show=False, colors=['C1', 'C5'])
        inf.plot_inferred_parameters(ax=axs[2], show=False)
        inf.plot_discretized(ax=axs[3], show=True)

    def test_infer_no_selection(self):
        """
        Test whether we infer a mostly neutral DFE when there is no selection.
        """
        sfs = fd.Spectrum.standard_kingman(10, n_monomorphic=100)

        # use different scalings which should not affect the result
        for m in [1, 10, 0.1, sfs]:
            inf = fd.BaseInference(
                sfs_neut=sfs * m,
                sfs_sel=sfs * m,
                seed=42,
                model=fd.DiscreteFractionalParametrization(
                    intervals=np.array([-100000, -0.1, 0.1, 10000])
                )
            )

            inf.run()

            # check that DFE is very neutral
            self.assertAlmostEqual(1, inf.get_discretized(intervals=np.array([-100000, -0.1, 0.1, 10000]))[0][1])

    def test_infer_strongly_deleterious_selection_target_sites(self):
        """
        Test whether we infer a strongly deleterious DFE when there is strong selection.
        """
        # use different mutational target sizes
        inf = fd.BaseInference(
            sfs_neut=fd.Spectrum.standard_kingman(10, n_monomorphic=100),
            sfs_sel=fd.Spectrum.standard_kingman(10, n_monomorphic=1000),
            seed=42
        )

        inf.run()

        # check that DFE is very strongly deleterious
        self.assertGreaterEqual(inf.get_discretized()[0][0], 0.8)

    def test_infer_deleterious_selection_sfs_shape(self):
        """
        Test whether we infer a deleterious DFE when there is selection.
        """
        n = 10

        sfs_neut = fd.Spectrum.standard_kingman(n, n_monomorphic=100)
        sfs_sel = fd.Spectrum.from_polymorphic(
            Discretization(n=n).get_allele_count_regularized(-10 * np.ones(n - 1), np.arange(1, n))
        )
        sfs_sel.data[0] = 100

        # use different mutational target sizes
        inf = fd.BaseInference(
            sfs_neut=sfs_neut,
            sfs_sel=sfs_sel,
            seed=42
        )

        inf.run()

        # check that DFE is very deleterious
        self.assertAlmostEqual(1, inf.get_discretized(np.array([-np.inf, -1, np.inf]))[0][0])

    def test_infer_beneficial_selection_target_sites(self):
        """
        Test whether we infer a weakly beneficial DFE when there is weak selection.
        """
        # use different mutational target sizes
        inf = fd.BaseInference(
            sfs_neut=fd.Spectrum.standard_kingman(10, n_monomorphic=100),
            sfs_sel=fd.Spectrum.standard_kingman(10, n_monomorphic=10),
            seed=42
        )

        inf.run()

        # check that DFE is very weakly beneficial
        self.assertGreaterEqual(inf.get_discretized()[0][-1], 0.4)
