import copy
from unittest import TestCase, mock

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.random._generator import Generator
from scipy.optimize import OptimizeResult

from fastdfe import Spectrum, BaseInference, Config, Spectra, GammaExpParametrization


class AbstractInferenceTestCase(TestCase):
    def assertEqualInference(self, obj1: object, obj2: object, ignore_keys=[]):
        """
        Compare Inference objects, recursively comparing their attributes.
        :param obj1: First object
        :param obj2: Second object
        :param ignore_keys: Keys to ignore
        """

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

                elif isinstance(value1, Spectrum):
                    np.testing.assert_equal(value1.to_list(), value2.to_list())

                elif isinstance(value1, object):
                    self.assertEqualInference(value1, value2, ignore_keys=ignore_keys)

                else:
                    raise AssertionError('Some objects were not compared.')


class BaseInferenceTestCase(AbstractInferenceTestCase):
    config_file = "testing/configs/pendula_C_full_anc/config.yaml"
    serialized = "testing/fastdfe/pendula_C_full_anc/serialized.json"

    show_plots = True

    maxDiff = None

    def test_run_inference_from_config_parallelized(self):
        """
        Successfully run inference from config file.
        """
        config = Config.from_file(self.config_file)
        config.update(parallelize=True)

        inference = BaseInference.from_config(config)
        inference.run()

    def test_run_inference_from_config_not_parallelized(self):
        """
        Successfully run inference from config file.
        """
        config = Config.from_file(self.config_file)
        config.update(parallelize=False)

        inference = BaseInference.from_config(config)
        inference.run()

    def test_compare_inference_with_log_scales_vs_lin_scales(self):
        """
        Compare inference with log scales vs linear scales.
        """
        config = Config.from_file(self.config_file)

        model = GammaExpParametrization()
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

        inference_log = BaseInference.from_config(config)
        inference_log.run()

        model = copy.copy(model)
        model.scales = dict(
            S_d='lin',
            b='lin',
            p_b='lin',
            S_b='lin'
        )

        config.update(model=model)
        inference_lin = BaseInference.from_config(config)
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

        assert inference_log.likelihood > inference_lin.likelihood

    def test_compare_inference_with_log_scales_vs_lin_scales_tutorial(self):
        """
        Compare inference with log scales vs linear scales.
        """
        model = GammaExpParametrization()
        model.scales = dict(
            S_d='log',
            b='log',
            p_b='lin',
            S_b='log'
        )

        inference_log = BaseInference(
            sfs_neut=Spectrum([177130, 997, 441, 228, 156, 117, 114, 83, 105, 109, 652]),
            sfs_sel=Spectrum([797939, 1329, 499, 265, 162, 104, 117, 90, 94, 119, 794]),
            model=model,
            do_bootstrap=True
        )
        inference_log.run()

        model = copy.copy(model)
        model.scales = dict(
            S_d='lin',
            b='lin',
            p_b='lin',
            S_b='lin'
        )

        inference_lin = BaseInference(
            sfs_neut=Spectrum([177130, 997, 441, 228, 156, 117, 114, 83, 105, 109, 652]),
            sfs_sel=Spectrum([797939, 1329, 499, 265, 162, 104, 117, 90, 94, 119, 794]),
            model=model,
            do_bootstrap=True,
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
        assert np.all(cis_lin[0] < cis_log[1])
        assert np.all(cis_log[0] < cis_lin[1])

        # the likelihood is indeed how when optimizing on the log sclae
        assert inference_log.likelihood > inference_lin.likelihood

    def test_restore_serialized_inference(self):
        """
        Check whether Inference can properly be serialized and restored.
        """
        inference = BaseInference.from_config_file(self.config_file)
        inference.run()

        serialized = 'scratch/test_serialize_inference.json'
        inference.to_file(serialized)
        inference_restored = BaseInference.from_file(serialized)

        # self.assertEqual(inference.to_json(), inference_restored.to_json())
        self.assertEqualInference(inference, inference_restored)

    def test_inference_run_bootstrap(self):
        """
        Run bootstrap on inference.
        """
        # unserialize
        inference = BaseInference.from_file(self.serialized)

        self.assertIsNone(inference.bootstraps)

        inference.bootstrap(100, parallelize=False)

        self.assertIsNotNone(inference.bootstraps)

    def test_evaluate_likelihood_same_as_mle_results(self):
        """
        Check that evaluating the loss function at the MLE result
        yields the same likelihood as the one reported.
        """
        # unserialize
        inference = BaseInference.from_file(self.serialized)

        assert inference.evaluate_likelihood(dict(all=inference.params_mle)) == inference.likelihood

    def test_visualize_inference_without_bootstraps(self):
        """
        Plot everything possible.
        """
        config = Config.from_file(self.config_file)

        config.update(
            do_bootstrap=False
        )

        inference = BaseInference.from_config(config)

        inference.plot_all(show=self.show_plots)
        inference.plot_bucket_sizes(show=self.show_plots)

    def test_visualize_inference_with_bootstraps(self):
        """
        Plot everything possible.
        """
        config = Config.from_file(self.config_file)

        config.update(
            do_bootstrap=True
        )

        inference = BaseInference.from_config(config)

        inference.plot_all(show=self.show_plots)
        inference.plot_bucket_sizes(show=self.show_plots)

    def test_plot_likelihoods(self):
        """
        Plot likelihoods.
        """
        # unserialize
        inference = BaseInference.from_file(self.serialized)

        inference.plot_likelihoods(show=self.show_plots, scale='log')

    def test_plot_interval_density(self):
        """
        Plot likelihoods.
        """
        # unserialize
        inference = BaseInference.from_file(self.serialized)

        inference.plot_interval_density(show=self.show_plots)

    def test_plot_provide_axes(self):
        """
        Plot likelihoods.
        """
        # unserialize
        inference = BaseInference.from_file(self.serialized)

        _, axs = plt.subplots(ncols=2)

        inference.plot_interval_density(show=False, ax=axs[0])
        inference.plot_interval_density(show=self.show_plots, ax=axs[1])

    def test_visualize_inference_bootstrap(self):
        """
        Plot everything possible.
        """
        # unserialize
        inference = BaseInference.from_file(self.serialized)

        # bootstrap
        inference.bootstrap(2)

        inference.plot_all(show=self.show_plots)

    def test_perform_bootstrap_without_running(self):
        """
        Perform bootstrap before having run the main inference.
        """
        # unserialize
        inference = BaseInference.from_config_file(self.config_file)

        # bootstrap
        inference.bootstrap(2)

    def test_compare_nested_likelihoods(self):
        """
        Compare nested likelihoods.
        """
        # unserialize
        inference = BaseInference.from_config_file(self.config_file)

        # bootstrap
        inference.compare_nested_models()

    def test_plot_nested_model_comparison_without_running(self):
        """
        Perform nested model comparison before having run the main inference.
        We just make sure this triggers a run.
        """
        # unserialize
        inference = BaseInference.from_config_file(self.config_file)

        # bootstrap
        inference.plot_nested_likelihoods()

    def test_cached_result(self):
        """
        Test inference against cached result.
        """
        inference = BaseInference.from_file(self.serialized)
        inference2 = BaseInference.from_config(inference.create_config())

        inference2.run()

        inference2.to_file("scratch/test_cached_result.json")
        inference2.get_summary().to_file("scratch/test_cached_result_summary.json")

        # inference.get_summary().to_file("scratch/test_cached_result_summary_actual.json")

        def get_dict(inf: BaseInference) -> dict:
            exclude = ['execution_time', 'result']

            return dict((k, v) for k, v in inference.get_summary().__dict__.items() if k not in exclude)

        self.assertDictEqual(get_dict(inference), get_dict(inference2))

    def test_run_if_necessary_wrapper_triggers_run(self):
        """
        Test whether the run_if_necessary wrapper triggers a run if the inference has not been run yet.
        """
        inference = BaseInference.from_config_file(self.config_file)

        with mock.patch.object(BaseInference, 'run', side_effect=Exception()) as mock_run:
            try:
                inference.plot_inferred_parameters(show=False)
            except Exception:
                pass

            mock_run.assert_called_once()

    def test_run_if_necessary_wrapper_not_triggers_run(self):
        """
        Test whether the run_if_necessary wrapper does not trigger a run if the inference has already been run.
        """
        inference = BaseInference.from_file(self.serialized)

        with mock.patch.object(BaseInference, 'run') as mock_run:
            inference.plot_inferred_parameters(show=False)
            mock_run.assert_not_called()

    def test_run_with_sfs_neut_zero_entry(self):
        """
        Test whether inference can be run with a neutral SFS that has a zero entry.
        """
        config = Config.from_file(self.config_file)

        sfs = config.data['sfs_neut']['all'].to_list()
        sfs[3] = 0
        sfs = Spectrum(sfs)
        config.data['sfs_neut'] = Spectra.from_spectrum(sfs)

        inference = BaseInference.from_config(config)

        inference.run()

        inference.plot_all(show=self.show_plots)

    def test_run_with_sfs_sel_zero_entry(self):
        """
        Test whether inference can be run with a selected SFS that has a zero entry.
        """
        config = Config.from_file(self.config_file)

        sfs = config.data['sfs_sel']['all'].to_list()
        sfs[3] = 0
        sfs = Spectrum(sfs)
        config.data['sfs_sel'] = Spectra.from_spectrum(sfs)

        inference = BaseInference.from_config(config)

        inference.run()

        inference.plot_all(show=self.show_plots)

    def test_fixed_parameters(self):
        """
        Test whether fixed parameters are correctly set.
        """
        config = Config.from_file(self.config_file)

        config.data['fixed_params'] = dict(all=dict(b=1.123, eps=0.123))

        inference = BaseInference.from_config(config)

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
        config = Config.from_file(self.config_file)

        assert BaseInference.from_config(config).get_n_optimized() == 5

        config.data['fixed_params'] = dict(all=dict(S_b=1, p_b=0))

        assert BaseInference.from_config(config).get_n_optimized() == 3

    def test_non_existing_fixed_param_raises_error(self):
        """
        Test that a non-existing fixed parameter raises an error.
        """
        config = Config.from_file(self.config_file)

        config.data['fixed_params'] = dict(all=dict(S_b=1, p_b=0, foo=1))

        with self.assertRaises(ValueError):
            BaseInference.from_config(config)
