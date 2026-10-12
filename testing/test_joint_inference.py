import functools
import json
import logging
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
from pandas._testing import assert_frame_equal
from scipy.stats import chi2

import fastdfe as fd
from fastdfe import JointInference, Config, SharedParams, Covariate, Spectra
from fastdfe.optimization import flatten_dict
from testing import TestCase
from testing.test_base_inference import InferenceTestCase


@pytest.mark.inference
class JointInferenceTestCase(InferenceTestCase):
    """
    Test the JointInference class.
    """
    config_file = "testing/cache/configs/pendula.pubescens.example_1.example_2.example_3_C_full_anc/config.yaml"

    maxDiff = None

    def test_run_and_restore_from_file(self):
        """
        Test that joint inference can be restored from file.
        """
        inference = JointInference.from_config_file(
            "resources/configs/shared/covariates_Sd_fixed_params/config.yaml"
        )

        inference.run()

        out = "scratch/test_restore_from_file.json"

        inference.to_file(out)

        inference2 = JointInference.from_file(out)

        self.assertEqualObjects(inference, inference2)

    def test_recreate_from_config(self):
        """
        Test whether inference can be recreated from config.
        """
        inf = JointInference.from_config_file(self.config_file)

        inf2 = JointInference.from_config(inf.create_config())

        self.assertEqualObjects(inf, inf2, ignore_keys=['fixed_params'])

    def test_perform_lrt_shared(self):
        """
        Test that the perform_lrt_shared method works.
        """
        inference = JointInference.from_file(
            "testing/cache/fastdfe/templates/shared/shared_example_2/serialized.json"
        )

        # inference.evaluate_likelihood(inference.params_mle)

        # inference.marginal_inferences['example_1'].evaluate_likelihood(
        # dict(all=inference.params_mle['example_1']))

        assert inference.perform_lrt_shared() < 1

        np.testing.assert_array_equal(inference.get_shared_param_names(), ['S_d'])

    def test_perform_lrt_covariates_no_covariates_raises_error(self):
        """
        Test that the perform_lrt_covariates method works.
        """
        inference = JointInference.from_file(
            "testing/cache/fastdfe/templates/shared/shared_example_1/serialized.json"
        )

        with pytest.raises(ValueError):
            inference.perform_lrt_covariates()

    @pytest.mark.slow
    def test_perform_lrt_covariates(self):
        """
        Test that the perform_lrt_covariates method works.
        """
        inf = JointInference.from_file(
            "testing/cache/fastdfe/templates/shared/covariates_Sd_example_1/serialized.json"
        )

        inf.n_bootstraps = 2

        # the model without covariates is rebuilt from the config, fix h so that it is not precomputed over a grid
        inf.fixed_params = dict(all=dict(h=0.5))

        assert inf.perform_lrt_covariates() <= 1

    def test_no_shared_params(self):
        """
        Test that the perform_lrt_covariates method works.
        """
        # create inference object
        inf = JointInference(
            sfs_neut=Spectra(dict(
                pendula=[177130, 997, 441, 228, 156, 117, 114, 83, 105, 109, 652],
                pubescens=[172528, 3612, 1359, 790, 584, 427, 325, 234, 166, 76, 31]
            )),
            sfs_sel=Spectra(dict(
                pendula=[797939, 1329, 499, 265, 162, 104, 117, 90, 94, 119, 794],
                pubescens=[791106, 5326, 1741, 1005, 756, 546, 416, 294, 177, 104, 41]
            )),
            fixed_params=dict(all=dict(h=0.5)),
            do_bootstrap=False,
            parallelize=False,
            n_bootstraps=2,
            n_runs=1
        )

        inf.run()
        inf.plot_discretized()

        self.assertEqual(inf.get_shared_param_names(), [])

    def test_perform_lrt_covariates_two_samples(self):
        """
        Test that the perform_lrt_covariates method works.
        """
        # create inference object
        inf = JointInference(
            sfs_neut=Spectra(dict(
                pendula=[177130, 997, 441, 228, 156, 117, 114, 83, 105, 109, 652],
                pubescens=[172528, 3612, 1359, 790, 584, 427, 325, 234, 166, 76, 31]
            )),
            sfs_sel=Spectra(dict(
                pendula=[797939, 1329, 499, 265, 162, 104, 117, 90, 94, 119, 794],
                pubescens=[791106, 5326, 1741, 1005, 756, 546, 416, 294, 177, 104, 41]
            )),
            fixed_params=dict(all=dict(eps=0, p_b=0, S_b=1, h=0.5)),
            do_bootstrap=True,
            n_bootstraps=2,
            n_runs=1,
            covariates=[Covariate(param='S_d', values=dict(pendula=0.3, pubescens=0.6))]
        )

        assert inf.perform_lrt_covariates() <= 1

    def test_plot_inferred_parameters_boxplot(self):
        """
        Test that the plot_inferred_parameters_boxplot method works.
        """
        inference = JointInference.from_file(
            "testing/cache/fastdfe/templates/shared/shared_example_2/serialized.json"
        )

        inference.bootstrap(6)

        inference.plot_inferred_parameters_boxplot()

    def test_plot_inferred_parameters(self):
        """
        Test that the plot_inferred_parameters_boxplot method works.
        """
        inference = JointInference.from_file(
            "testing/cache/fastdfe/templates/shared/shared_example_2/serialized.json"
        )

        inference.bootstrap(2)

        inference.plot_inferred_parameters()

    def test_plot_all_without_bootstraps(self):
        """
        Test that the plot_all method works.
        """
        inference = JointInference.from_file(
            "testing/cache/fastdfe/templates/shared/covariates_Sd_example_1/serialized.json"
        )

        inference.plot_all()

    def test_plot_all_with_bootstraps(self):
        """
        Test that the plot_all method works.
        """
        inference = JointInference.from_file(
            "testing/cache/fastdfe/templates/shared/covariates_Sd_example_1/serialized.json"
        )

        inference.bootstrap(2)

        inference.plot_all()

    @staticmethod
    def test_plot_covariate():
        """
        Test the plot_covariate method.
        """
        inference = JointInference.from_file(
            "testing/cache/fastdfe/templates/shared/covariates_Sd_example_1/serialized.json"
        )

        inference.plot_covariate()

        pass

    @staticmethod
    def test_plot_not_showing_types():
        """
        Test the plot_covariate method.
        """
        inference = JointInference.from_file(
            "testing/cache/fastdfe/templates/shared/covariates_Sd_example_1/serialized.json"
        )

        inference.plot_covariate(show_types=False)

        pass

    @staticmethod
    def test_plot_covariate_bootstrap():
        """
        Test the plot_covariate method.
        """
        inference = JointInference.from_file(
            "testing/cache/fastdfe/templates/shared/covariates_Sd_example_1/serialized.json"
        )

        inference.bootstrap(2)

        inference.plot_covariate()

        pass

    @staticmethod
    def test_plot_discretized_without_showing_marginals():
        """
        Test the plot_discretized method
        """
        inference = JointInference.from_file(
            "testing/cache/fastdfe/templates/shared/covariates_Sd_example_1/serialized.json"
        )

        inference.plot_discretized(show_marginals=False)

        pass

    def test_evaluate_likelihood_same_as_mle_results(self):
        """
        Check that evaluating the loss function at the MLE results
        yields the same likelihood as the one reported.
        """
        # rerun inference to make sure we don't have platform-specific differences
        inference = JointInference.from_file(
            "testing/cache/fastdfe/templates/shared/shared_example_1/serialized.json"
        )

        # make sure joint likelihood is almost the same
        self.assertAlmostEqual(inference.evaluate_likelihood(inference.params_mle), inference.likelihood)

        for t in inference.types:
            # make sure marginal likelihoods are the same
            self.assertAlmostEqual(inference.marginal_inferences[t].evaluate_likelihood(
                dict(all=inference.marginal_inferences[t].params_mle)), inference.marginal_inferences[t].likelihood)

            # make sure joint likelihoods are not the same as when evaluated individually
            assert inference.joint_inferences[t].evaluate_likelihood(
                dict(all=inference.joint_inferences[t].params_mle)) != inference.joint_inferences[t].likelihood

            # make sure all joint likelihoods coincide with likelihood of joint inference
            assert inference.joint_inferences[t].likelihood == inference.likelihood

    def test_fixed_parameters(self):
        """
        Test whether fixed parameters are correctly set.
        """
        config = Config.from_file(
            "testing/cache/configs/pendula.pubescens.example_1.example_2.example_3_C_full_anc/config.yaml"
        )

        # define fixed parameters
        config.data['fixed_params'] = dict(
            all=dict(eps=0.123, h=0.5),
            example_1=dict(b=1.234),
            example_2=dict(b=1.132),
            pendula=dict(S_b=10.1)
        )

        # define fixed parameters
        config.data['shared_params'] = [
            SharedParams(types='all', params=['S_d', 'p_b'])
        ]

        inf = JointInference.from_config(config)

        inf.run()

        params_mle = inf.params_mle

        fixed = {
            'all': {'eps': 0.123, 'h': 0.5},
            'example_1': {'eps': 0.123, 'h': 0.5, 'b': 1.234},
            'example_2': {'eps': 0.123, 'h': 0.5, 'b': 1.132},
            'example_3': {'eps': 0.123, 'h': 0.5},
            'pendula': {'eps': 0.123, 'h': 0.5, 'S_b': 10.1},
            'pubescens': {'eps': 0.123, 'h': 0.5},
        }

        assert inf.marginal_inferences['all'].optimization.fixed_params == flatten_dict(dict(all=fixed['all']))

        # check that eps is fixed for all types
        for t in inf.types:
            assert inf.marginal_inferences[t].optimization.fixed_params == flatten_dict(dict(all=fixed[t]))
            assert inf.joint_inferences[t].optimization.fixed_params == flatten_dict(dict(all=fixed[t]))
            assert params_mle[t]['eps'] == config.data['fixed_params']['all']['eps']

        # check that b is fixed for example_1 and example_2, but not for the others
        assert params_mle['example_1']['b'] == config.data['fixed_params']['example_1']['b']
        assert params_mle['example_2']['b'] == config.data['fixed_params']['example_2']['b']
        assert params_mle['example_3']['b'] != config.data['x0']['all']['b']
        assert params_mle['example_3']['b'] != config.data['fixed_params']['example_1']['b']
        assert params_mle['example_3']['b'] != config.data['fixed_params']['example_2']['b']
        assert params_mle['pendula']['b'] != config.data['x0']['all']['b']
        assert params_mle['pubescens']['b'] != config.data['x0']['all']['b']

        # check that S_b is fixed for pendula, but not for the others
        self.assertAlmostEqual(params_mle['pendula']['S_b'], config.data['fixed_params']['pendula']['S_b'])
        assert params_mle['pubescens']['S_b'] != config.data['x0']['all']['S_b']

    def test_shared_same_parameter_twice_among_different_types(self):
        """
        Test sharing a parameter twice among different types.
        """
        config = Config.from_file(
            "testing/cache/configs/pendula.pubescens.example_1.example_2.example_3_C_full_anc/config.yaml"
        )

        config.update(
            fixed_params={'all': dict(S_b=1, eps=0, p_b=0, h=0.5)},
            shared_params=[
                SharedParams(types=['example_1', 'example_2'], params=['b']),
                SharedParams(types=['pendula', 'pubescens'], params=['b'])
            ],
            n_bootstraps=2,
            do_bootstraps=True,
            n_runs=1
        )

        inf = JointInference.from_config(config)

        inf.run()

        # check that b is shared between example_1 and example_2
        assert inf.params_mle['example_1']['b'] == inf.params_mle['example_2']['b']

        # check that b is shared between pendula and pubescens
        assert inf.params_mle['pendula']['b'] == inf.params_mle['pubescens']['b']

        # check that b is not shared between example_1 and pendula
        assert inf.params_mle['example_1']['b'] != inf.params_mle['pendula']['b']

    def test_fixed_parameters_cannot_be_shared(self):
        """
        Test that trying to shared fixed parameters raises an error.
        """
        config = Config.from_file(
            "testing/cache/configs/pendula.pubescens.example_1.example_2.example_3_C_full_anc/config.yaml"
        )

        # define fixed parameters
        config.data['fixed_params'] = dict(
            example_1=dict(b=1.234),
        )

        # define fixed parameters
        config.data['shared_params'] = [
            SharedParams(types=['example_1'], params=['S_d', 'p_b', 'b'])
        ]

        with pytest.raises(ValueError):
            JointInference.from_config(config)

    @pytest.mark.slow
    def test_compare_nested_models(self):
        """
        Test nested model comparison.
        """
        config = fd.Config.from_file("resources/configs/shared/covariates_Sd_fixed_params/config.yaml")
        config.update(n_runs=2, do_bootstrap=False)

        inf = JointInference.from_config(config)

        P, inferences = inf.compare_nested_models()

        assert inferences['full.anc'].get_n_optimized() == 18
        assert inferences['full.anc'].marginal_inferences['all'].get_n_optimized() == 5

        assert inferences['full.no_anc'].get_n_optimized() == 13
        assert inferences['full.no_anc'].marginal_inferences['all'].get_n_optimized() == 4

        assert inferences['dele.anc'].get_n_optimized() == 8
        assert inferences['dele.anc'].marginal_inferences['all'].get_n_optimized() == 3

        assert inferences['dele.no_anc'].get_n_optimized() == 3
        assert inferences['dele.no_anc'].marginal_inferences['all'].get_n_optimized() == 2

    def test_get_n_optimized(self):
        """
        Test get number of parameters to be optimized.
        """
        inf = JointInference.from_config_file(
            "resources/configs/shared/covariates_Sd_fixed_params/config.yaml"
        )

        assert inf.get_n_optimized() == 11

        inf = JointInference.from_config_file(
            "resources/configs/shared/covariates_Sd_example_1/config.yaml"
        )

        assert inf.get_n_optimized() == 18

        inf = JointInference.from_config_file(
            "resources/configs/shared/covariates_dummy_example_1/config.yaml"
        )

        assert inf.get_n_optimized() == 19

    def test_non_existing_shared_param_raises_error(self):
        """
        Test that trying to share a non-existing parameter raises an error.
        """
        config = Config.from_file(
            "testing/cache/configs/pendula.pubescens.example_1.example_2.example_3_C_full_anc/config.yaml"
        )

        # define shared parameters
        config.data['shared_params'] = [
            SharedParams(types=['example_1'], params=['b', 'foo', 'bar'])
        ]

        with pytest.raises(ValueError):
            JointInference.from_config(config)

    def test_non_existing_covariate_raises_error(self):
        """
        Test that trying to share a non-existing parameter raises an error.
        """
        config = Config.from_file(
            "testing/cache/configs/pendula.pubescens.example_1.example_2.example_3_C_full_anc/config.yaml"
        )

        types = ['pendula', 'pubescens', 'example_1', 'example_2', 'example_3']

        # define shared parameters
        config.data['covariates'] = [
            Covariate(values={t: 1 for t in types}, param='foo')
        ]

        with pytest.raises(ValueError):
            JointInference.from_config(config)

    def test_covariate_random_covariates(self):
        """
        Check that random covariates are not significant.
        """
        # parse SFS
        spectra = Spectra.from_file("resources/SFS/betula/pendula/degeneracy_ref_base_tutorial.csv")

        # create inference object
        inf = JointInference(
            sfs_neut=spectra[['neutral.*']].merge_groups(1),
            sfs_sel=spectra[['selected.*']].merge_groups(1),
            covariates=[Covariate(param='S_d', values=dict(A=1, C=2, T=3, G=4))],
            fixed_params={'all': dict(S_b=1, eps=0, h=0.5)},
            parallelize=True,
            do_bootstrap=True,
            n_bootstraps=2
        )

        # run inference
        inf.run()

        inf_no_cov = inf.run_joint_without_covariates()

        p = inf.perform_lrt_covariates()

        inf.plot_inferred_parameters()

        # the test should be non-significant as we chose random covariates
        assert p > 0.05

    @pytest.mark.slow
    def test_covariates_strong_correlation(self):
        """
        Check that strong correlation between covariates is detected as significant. The covariate is
        optimized on the linear scale, as the default symlog scale misses the optimum for these data.
        """
        # parse SFS
        spectra = Spectra.from_file("resources/SFS/betula/pendula/degeneracy_ref_base_tutorial.csv")

        # create inference object
        inf = JointInference(
            sfs_neut=spectra[['neutral.*']].merge_groups(1),
            sfs_sel=spectra[['selected.*']].merge_groups(1),
            covariates=[Covariate(
                param='S_d',
                values=dict(A=-100000, C=-2045, T=-60504, G=-5024),
                bounds=(-1e4, 1e4),
                bounds_scale='lin'
            )],
            fixed_params={'all': dict(S_b=1, eps=0, p_b=0, h=0.5)},
            parallelize=True,
            do_bootstrap=True,
            n_bootstraps=10,
            n_runs=20
        )

        # run inference
        inf.run()

        inf_no_cov = inf.run_joint_without_covariates()

        p = inf.perform_lrt_covariates()

        inf.plot_inferred_parameters()

        # we expect 'c0' to be close to 1
        # self.assertAlmostEqual(inf.params_mle['T']['c0'], 1, places=2)

        # the test should be more or less significant as we chose good covariates
        assert p < 0.15

    def test_covariates_plot_inferred_parameters(self):
        """
        Check that covariates are plotted correctly.
        """
        # parse SFS
        spectra = Spectra.from_file("resources/SFS/betula/pendula/degeneracy_ref_base_tutorial.csv")

        # create inference object
        inf = JointInference(
            sfs_neut=spectra[['neutral.*']].merge_groups(1),
            sfs_sel=spectra[['selected.*']].merge_groups(1),
            covariates=[Covariate(param='S_d', values=dict(A=-100000, C=-2045, T=-60504, G=-5024))],
            fixed_params={'all': dict(S_b=1, eps=0, p_b=0, h=0.5)},
            parallelize=True,
            do_bootstrap=True,
            n_runs=10,
            n_bootstraps=2
        )

        inf.plot_inferred_parameters()

    def test_get_discretized_errors_with_bootstraps(self):
        """
        Get the discretized DFE errors.
        """
        inference = JointInference.from_file(
            "testing/cache/fastdfe/templates/shared/shared_example_2/serialized.json"
        )

        inference.bootstrap(2)

        res = inference.get_discretized()

    def test_get_discretized_errors_no_bootstraps(self):
        """
        Get the discretized DFE errors.
        """
        inference = JointInference.from_file(
            "testing/cache/fastdfe/templates/shared/shared_example_1/serialized.json"
        )

        values, errors = inference.get_discretized()

        # check that all errors are None
        for err in errors.values():
            assert err is None

    def test_joint_inference_with_folded_spectra(self):
        """
        Test that joint inference works with folded spectra.
        """
        # neutral SFS for two types
        sfs_neut = Spectra(dict(
            pendula=[177130, 997, 441, 228, 156, 117, 114, 83, 105, 109, 652],
            pubescens=[172528, 3612, 1359, 790, 584, 427, 325, 234, 166, 76, 31]
        )).fold()

        # selected SFS for two types
        sfs_sel = Spectra(dict(
            pendula=[797939, 1329, 499, 265, 162, 104, 117, 90, 94, 119, 794],
            pubescens=[791106, 5326, 1741, 1005, 756, 546, 416, 294, 177, 104, 41]
        )).fold()

        # create inference object
        inf = JointInference(
            sfs_neut=sfs_neut,
            sfs_sel=sfs_sel,
            shared_params=[SharedParams(types=["pendula", "pubescens"], params=["eps", "S_d"])],
            fixed_params={'all': dict(h=0.5)},
            do_bootstrap=True,
            n_bootstraps=2
        )

        # run inference
        inf.run()

        # assert that all inferences are folded
        assert inf.folded
        assert np.array([v.folded for v in inf.get_inferences().values()]).all()

        inf.plot_discretized()
        inf.plot_sfs_comparison()

    def test_joint_inference_types_with_dots_throws_error(self):
        """
        Test that joint inference with types containing dots throws an error.
        """
        # neutral SFS for two types
        sfs_neut = Spectra({
            'pendula.foo': [177130, 997, 441, 228, 156, 117, 114, 83, 105, 109, 652],
            'pubescens.bar.foo': [172528, 3612, 1359, 790, 584, 427, 325, 234, 166, 76, 31]
        })

        # selected SFS for two types
        sfs_sel = Spectra({
            'pendula.foo': [797939, 1329, 499, 265, 162, 104, 117, 90, 94, 119, 794],
            'pubescens.bar.foo': [791106, 5326, 1741, 1005, 756, 546, 416, 294, 177, 104, 41]
        })

        # create inference object
        with pytest.raises(ValueError):
            JointInference(
                sfs_neut=sfs_neut,
                sfs_sel=sfs_sel,
                fixed_params={'all': dict(eps=0, h=0.5)},
                shared_params=[SharedParams(types=["pendula.foo", "pubescens.bar.foo"], params=["S_d", "S_b", "p_b"])],
                do_bootstrap=True
            )

    def test_spectra_with_disparate_types_throws_error(self):
        """
        Test that joint inference with types containing dots throws an error.
        """
        # neutral SFS for two types
        sfs_neut = Spectra({
            'pendula': [177130, 997, 441, 228, 156, 117, 114, 83, 105, 109, 652],
            'pubescens': [172528, 3612, 1359, 790, 584, 427, 325, 234, 166, 76, 31]
        })

        # selected SFS for two types
        sfs_sel = Spectra({
            'pendula': [797939, 1329, 499, 265, 162, 104, 117, 90, 94, 119, 794],
            'pubescens.bar.foo': [791106, 5326, 1741, 1005, 756, 546, 416, 294, 177, 104, 41]
        })

        # create inference object
        with pytest.raises(ValueError):
            JointInference(
                sfs_neut=sfs_neut,
                sfs_sel=sfs_sel,
                fixed_params={'all': dict(eps=0, h=0.5)},
                shared_params=[SharedParams(types=["pendula", "pubescens.bar.foo"], params=["S_d", "S_b", "p_b"])],
                do_bootstrap=True
            )

    @staticmethod
    @pytest.mark.slow
    def test_manuscript_example():
        """
        Test the example in the manuscript.

        Note the warning on a large residual for the joint inference is not
        unusual as the fit to each component SFS naturally is worse when sharing parameters.
        """
        config = fd.Config.from_file("resources/configs/arabidopsis/covariates_example.yaml")
        config.update(n_bootstraps=10)

        inf = fd.JointInference.from_config(config)

        inf.run()

        # get p-value for covariate significance
        p = inf.perform_lrt_covariates()

        _, axs = plt.subplots(1, 2, figsize=(10.5, 3.5))
        p_str = f"p = {p:.1e}" if p >= 1e-100 else "p < 1e-100"

        inf.plot_covariate(ax=axs[0], xlabel='RSA', show=False)
        inf.plot_discretized(
            title=f"DFE comparison, " + p_str, ax=axs[1],
            show_marginals=False, show=True
        )

        pass

    @pytest.mark.skip("Only works with variable h")
    def test_alternative_optimizer(self):
        """
        Test for alternative optimizer.
        """
        config = fd.Config.from_file(
            "resources/configs/shared/covariates_Sd_fixed_params/config.yaml"
        ).update(
            do_bootstrap=True,
            n_bootstraps=1,
            method_mle='Powell'
        )

        inf = fd.JointInference.from_config(config)
        inf.run()

        self.assertTrue(hasattr(inf.result, 'direc'))
        self.assertTrue(hasattr(inf.bootstrap_results[0], 'direc'))

        self.assertTrue(hasattr(inf.marginal_inferences['all'].result, 'direc'))
        self.assertTrue(hasattr(inf.marginal_inferences['example_1'].bootstrap_results[0], 'direc'))

    def test_raise_error_n_above_400(self):
        """
        Test whether an error is raised when sample size is above 400.
        """
        with self.assertRaises(ValueError) as context:
            fd.JointInference(
                sfs_neut=fd.Spectrum.standard_kingman(401, n_monomorphic=10000).to_spectra(),
                sfs_sel=fd.Spectrum.standard_kingman(401, n_monomorphic=10000).to_spectra()
            )

        print(context.exception)

    def test_bootstrap_n_retries(self):
        """
        Test whether bootstrapping with retries works as expected.
        """
        config = fd.Config.from_file(
            "resources/configs/shared/covariates_Sd_fixed_params/config.yaml"
        ).update(
            n_bootstrap_retries=3,
            parallelize=True,
            n_bootstraps=3,
            do_bootstrap=True
        )
        inf = fd.JointInference.from_config(config)
        inf.run()

        for inf in list(inf.marginal_inferences.values()) + [inf]:
            self.assertEqual(3, len(inf.bootstraps.likelihoods_runs.iloc[0]))
            self.assertEqual(3, len(inf.bootstraps.x0_runs.iloc[0]))
            self.assertGreater(inf.bootstraps.likelihoods_std.mean(), 0)

    def test_get_alpha(self):
        """
        Test get_alpha method.
        """
        inf = JointInference.from_file(
            "testing/cache/fastdfe/templates/shared/shared_example_1/serialized.json"
        )

        alpha = inf.get_alpha()

        # assert keys are present
        for t in inf.types:
            assert t in alpha

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

        inf = fd.JointInference.from_config(config)
        inf.run()

        inf2 = fd.JointInference.from_config(config)
        inf2.run()

        self.assertEqual(inf.params_mle, inf2.params_mle)
        assert_frame_equal(inf.bootstraps, inf2.bootstraps)

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

        inference = fd.JointInference.from_config(config)
        inference.run()

        inference2 = fd.JointInference.from_config(config)
        inference2.run()

        self.assertEqual(inference.params_mle, inference2.params_mle)
        assert_frame_equal(inference.bootstraps, inference2.bootstraps)

    def test_joint_different_fixed_h(self):
        """
        Test joint inference with different fixed h values.
        """
        inf = JointInference(
            sfs_neut=Spectra(dict(
                pendula=[177130, 997, 441, 228, 156, 117, 114, 83, 105, 109, 652],
                pendula2=[177130, 997, 441, 228, 156, 117, 114, 83, 105, 109, 652],
            )),
            sfs_sel=Spectra(dict(
                pendula=[797939, 1329, 499, 265, 162, 104, 117, 90, 94, 119, 794],
                pendula2=[797939, 1329, 499, 265, 162, 104, 117, 90, 94, 119, 794],
            )),
            fixed_params=dict(all=dict(eps=0, S_b=1, p_b=0), pendula=dict(h=0.2), pendula2=dict(h=0.8)),
            shared_params=[SharedParams(types='all', params=['b'])],
            intervals_del=(-1.0e+8, -1.0e-5, 30),
            intervals_ben=(1.0e-5, 1.0e4, 30),
            intervals_h=(0, 1, 6),
            do_bootstrap=False,
            parallelize=True,
            n_runs=10
        )

        inf.run()

        # make sure that h values are correctly held fixed
        self.assertEqual(0.2, inf.params_mle['pendula']['h'])
        self.assertEqual(0.8, inf.params_mle['pendula2']['h'])

        self.assertEqual(0.2, inf.joint_inferences['pendula'].params_mle['h'])
        self.assertEqual(0.8, inf.joint_inferences['pendula2'].params_mle['h'])

        self.assertEqual(0.2, inf.marginal_inferences['pendula'].params_mle['h'])
        self.assertEqual(0.8, inf.marginal_inferences['pendula2'].params_mle['h'])

        inf.plot_discretized()

        # make sure that the DFE for h=0.2 is shifted towards more deleterious mutations for both marginals and joints
        discretized = inf.get_discretized()[0]
        self.assertGreater(discretized["marginal.pendula"][0], discretized["marginal.pendula2"][0])
        self.assertGreater(discretized["joint.pendula"][0], discretized["joint.pendula2"][0])

    def test_joint_optimized_different_h(self):
        """
        Test joint inference with different optimized h values.
        """
        inf = JointInference(
            sfs_neut=Spectra(dict(
                pendula=[177130, 997, 441, 228, 156, 117, 114, 83, 105, 109, 652],
                pubescens=[172528, 3612, 1359, 790, 584, 427, 325, 234, 166, 76, 31]
            )),
            sfs_sel=Spectra(dict(
                pendula=[797939, 1329, 499, 265, 162, 104, 117, 90, 94, 119, 794],
                pubescens=[791106, 5326, 1741, 1005, 756, 546, 416, 294, 177, 104, 41]
            )),
            fixed_params=dict(all=dict(eps=0, S_b=1, p_b=0)),
            shared_params=[SharedParams(types='all', params=['b', 'S_d'])],
            intervals_del=(-1.0e+8, -1.0e-5, 30),
            intervals_ben=(1.0e-5, 1.0e4, 30),
            intervals_h=(0, 1, 6),
            do_bootstrap=False,
            parallelize=True,
            n_runs=10
        )

        inf.run()

        self.assertNotEqual(inf.params_mle['pendula']['h'], inf.params_mle['pubescens']['h'])
        self.assertNotEqual(
            inf.joint_inferences['pendula'].params_mle['h'],
            inf.joint_inferences['pubescens'].params_mle['h']
        )



class FastJointInferenceTestCase(TestCase):
    """
    Fast-tier JointInference coverage: tiny two-type fits (coarse discretization, single run, no
    bootstrap) exercising shared parameters, a covariate, the aggregate getters and the
    serialization round-trip.
    """

    sfs_neut = fd.Spectrum([177130, 997, 441, 228, 156, 117, 114, 83, 105, 109, 652])
    sfs_sel = fd.Spectrum([797939, 1329, 499, 265, 162, 104, 117, 90, 94, 119, 794])

    def _make(self, **kwargs):
        kwargs.setdefault('do_bootstrap', False)
        kwargs.setdefault('n_runs', 1)
        kwargs.setdefault('parallelize', False)
        return fd.JointInference(
            sfs_neut=fd.Spectra(dict(a=list(self.sfs_neut), b=list(self.sfs_neut))),
            sfs_sel=fd.Spectra(dict(a=list(self.sfs_sel), b=list(self.sfs_sel))),
            intervals_del=(-1.0e+8, -1.0e-5, 20),
            intervals_ben=(1.0e-5, 1.0e4, 20),
            **kwargs
        )

    def test_nested_models_fix_shared_parameters(self):
        """
        The nested models of a joint inference hold shared parameters at their fixed values. Shared
        parameters stayed free, so every nested model was fitted as the full one, and a shared parameter
        was counted once per type in the degrees of freedom.
        """
        inf = self._make(
            fixed_params=dict(all=dict(h=0.5)),
            shared_params=[fd.SharedParams(params=['S_b', 'p_b', 'eps'], types='all')]
        )

        _, inferences = inf.compare_nested_models()

        for name, sub in inferences.items():
            for t in sub.types:
                if name.startswith('dele'):
                    self.assertEqual(sub.params_mle[t]['p_b'], 0)
                    self.assertEqual(sub.params_mle[t]['S_b'], 1)

                if name.endswith('no_anc'):
                    self.assertEqual(sub.params_mle[t]['eps'], 0)

        self.assertLess(inferences['dele.no_anc'].likelihood, inferences['full.anc'].likelihood - 1)

        # eps, and p_b and S_b, are one parameter each, of which eps = 0 and p_b = 0 lie on a bound
        self.assertEqual(inferences['full.no_anc']._get_df_nested(inferences['full.anc']), (1, 1))
        self.assertEqual(inferences['dele.anc']._get_df_nested(inferences['full.anc']), (2, 1))
        self.assertEqual(inferences['dele.no_anc']._get_df_nested(inferences['full.anc']), (3, 2))

    def test_nested_models_refit_marginal_inferences(self):
        """
        The marginal inferences of a nested joint model are fitted under its fixed parameters. They kept the
        fits of the parent model.
        """
        inf = self._make(fixed_params=dict(all=dict(h=0.5)), shared_params=[fd.SharedParams(params=['S_d'])])
        inf.run()

        _, inferences = inf.compare_nested_models()

        for marginal in inferences['dele.no_anc'].marginal_inferences.values():
            self.assertEqual(marginal.params_mle['p_b'], 0)
            self.assertEqual(marginal.params_mle['eps'], 0)

    def test_nested_models_keep_fixed_parameters_of_a_type(self):
        """The nested models keep a parameter that is fixed for one type only."""
        inf = self._make(fixed_params=dict(all=dict(h=0.5), a=dict(b=0.3)))

        _, inferences = inf.compare_nested_models()

        for sub in inferences.values():
            self.assertAlmostEqual(sub.params_mle['a']['b'], 0.3, places=10)
            self.assertNotAlmostEqual(sub.params_mle['b']['b'], 0.3, places=3)
            self.assertNotIn('b', sub.marginal_inferences['all'].fixed_params['all'])

    def test_fixed_params_of_unknown_type_raise(self):
        """Fixed parameters that are not keyed by 'all' or a type raise an error and are left unchanged."""
        with self.assertRaises(ValueError):
            self._make(fixed_params=dict(eps=0, h=0.5))

        with self.assertRaises(ValueError):
            self._make(fixed_params=dict(all=dict(h=0.5), c=dict(eps=0)))

        fixed_params = dict(a=dict(eps=0))
        self._make(fixed_params=fixed_params)

        self.assertEqual(fixed_params, dict(a=dict(eps=0)))

    def test_divergence_target_sizes_not_matching_types_raise(self):
        """
        Divergence target sizes that lack a type or name an unknown one raise an error. A missing type
        failed during the optimization, and the value of an unknown key was added to the pooled size.
        """
        for sizes in [dict(a=1e5), dict(a=1e5, b=1e5, c=1e5), {}]:
            with self.assertRaises(ValueError):
                self._make(n_sites_div_neut=sizes, n_sites_div_sel=dict(a=1e5, b=1e5))

            with self.assertRaises(ValueError):
                self._make(n_sites_div_neut=dict(a=1e5, b=1e5), n_sites_div_sel=sizes)

    def test_folded_inference_on_unfolded_spectra_compares_folded_spectra(self):
        """
        With ``folded=True`` on unfolded spectra, the residual and the SFS comparison plot use the folded
        observed SFS. They used the unfolded one against the folded modelled SFS, which inflated the residual.
        """
        inf = self._make(folded=True)
        inf.run()

        modelled = np.concatenate([i.sfs_mle.polymorphic for i in inf.joint_inferences.values()])
        observed = np.concatenate([i.sfs_sel.fold().polymorphic for i in inf.joint_inferences.values()])

        self.assertAlmostEqual(inf.get_residual(2), np.linalg.norm(modelled - observed), places=8)

        ax = inf.plot_sfs_comparison(show=False, sfs_types=['observed', 'modelled'], types=['a'])
        heights = np.array([bar.get_height() for bar in ax.patches])

        # observed and modelled SFS of the marginal and the joint inference, nine polymorphic classes each,
        # of which the upper four are empty
        self.assertEqual(len(heights), 36)
        self.assertEqual(np.count_nonzero(heights), 20)

    def test_run_without_progress_bar_shows_none(self):
        """``pbar=False`` reaches every optimization and bootstrap that a joint run performs."""
        from unittest import mock
        from fastdfe import optimization

        inf = self._make(shared_params=[fd.SharedParams(params=['S_d'])], do_bootstrap=True, n_bootstraps=2)

        with mock.patch.object(optimization, 'tqdm', wraps=optimization.tqdm) as tqdm:
            inf.run(pbar=False)

        self.assertEqual(tqdm.call_count, 0)

    def test_accessors_run_the_inference_if_required(self):
        """The accessors of results trigger the run of a joint inference that was not run yet."""
        for accessor in [
            lambda inf: inf.get_discretized(confidence_intervals=False),
            lambda inf: inf.get_alpha(),
            lambda inf: inf.get_omega(),
            lambda inf: inf.get_omega_a(),
        ]:
            inf = self._make()

            accessor(inf)

            self.assertIsNotNone(inf.params_mle)

    def test_type_named_all_raises(self):
        """
        A type named ``all`` raises an error, as the name denotes the pooled spectra. Such an inference was
        constructed and then failed during the run.
        """
        with self.assertRaises(ValueError):
            fd.JointInference(
                sfs_neut=fd.Spectra(dict(all=list(self.sfs_neut), b=list(self.sfs_neut))),
                sfs_sel=fd.Spectra(dict(all=list(self.sfs_sel), b=list(self.sfs_sel))),
                do_bootstrap=False
            )

    def test_type_name_with_colon_raises(self):
        """
        A type name containing a colon raises an error, as colons join the types that share a parameter. Such
        an inference failed with a ``KeyError`` during the joint optimization.
        """
        with self.assertRaises(ValueError):
            fd.JointInference(
                sfs_neut=fd.Spectra({'a:1': list(self.sfs_neut), 'b': list(self.sfs_neut)}),
                sfs_sel=fd.Spectra({'a:1': list(self.sfs_sel), 'b': list(self.sfs_sel)}),
                do_bootstrap=False
            )

    def test_rejected_number_of_bootstraps_is_not_stored(self):
        """
        A rejected number of bootstrap samples leaves the stored number unchanged. It was stored before the
        check, so every later bootstrap without an explicit number failed as well.
        """
        inf = self._make(n_bootstraps=2)
        inf.run()

        with self.assertRaises(ValueError):
            inf.bootstrap(n_samples=0)

        self.assertEqual(inf.n_bootstraps, 2)
        self.assertEqual(len(inf.bootstrap()), 2)

    def test_bootstrap_on_inference_not_run_runs_it_first(self):
        """
        Bootstrapping a joint inference that was not run yet runs it first. The marginal inferences were
        fitted by their own bootstrap instead, before a run had given them their initial values.
        """
        from unittest import mock

        inf = self._make(do_bootstrap=True, n_bootstraps=2, x0=dict(a=dict(b=0.3)))
        marginal = inf.marginal_inferences['a']

        run, started_from = fd.BaseInference.run, []

        def record(self, *args, **kwargs):
            if self is marginal:
                started_from.append(self.x0['all']['b'])

            return run(self, *args, **kwargs)

        with mock.patch.object(fd.BaseInference, 'run', record):
            inf.bootstrap()

        self.assertEqual(started_from, [0.3])
        self.assertEqual(len(inf.bootstraps), 2)

    def test_bounds_given_for_covariate_take_precedence(self):
        """
        Bounds given to the inference for a covariate replace those of the covariate, as the scales do. The
        bounds of the covariate took precedence, so the given ones were ignored.
        """
        inf = self._make(
            covariates=[fd.Covariate(param='S_d', values=dict(a=0.3, b=0.6))],
            bounds=dict(c0=(-5, 5)),
            scales=dict(c0='lin')
        )

        self.assertEqual(inf.bounds['c0'], (-5, 5))
        self.assertEqual(inf.scales['c0'], 'lin')

    def test_run_with_bounds_given_for_covariate(self):
        """
        A joint inference runs with bounds given for a covariate and no scale, the covariate staying within
        the range the symmetric log scale spans, from minus to plus the upper bound. The inferences of the
        types receive the bounds too but have no scale for a covariate, which raised a ``KeyError`` when their
        estimates were compared with the bounds.
        """
        inf = self._make(
            covariates=[fd.Covariate(param='S_d', values=dict(a=1, b=5))],
            bounds=dict(c0=(1e-3, 10))
        )
        inf.run()

        self.assertEqual(inf.optimization.bounds['c0'], (1e-3, 10))
        self.assertTrue(-10 <= inf.params_mle['a']['c0'] <= 10)

    def test_initial_values_before_run_keep_parameters_of_other_types(self):
        """
        Before the run, the initial values of a type keep the parameters that other types share among
        themselves. All types held the same dictionary, so sharing a parameter among some types removed it
        from the others, and the number of optimized parameters was too low.
        """
        types = ['a', 'b', 'c']

        inf = fd.JointInference(
            sfs_neut=fd.Spectra({t: list(self.sfs_neut) for t in types}),
            sfs_sel=fd.Spectra({t: list(self.sfs_sel) for t in types}),
            shared_params=[fd.SharedParams(params=['b'], types=['a', 'b'])],
            do_bootstrap=False
        )

        self.assertIn('b', inf.get_x0()['c'])
        self.assertEqual(
            sorted(inf.get_optimized_param_names()), ['a.S_d', 'a:b.b', 'b.S_d', 'c.S_d', 'c.b']
        )

    def test_covariate_on_parameter_shared_among_some_types(self):
        """
        A covariate on a parameter that is shared among some of the types shares it among all types only.
        The partial sharing was kept beside it, which left an optimized parameter without any effect.
        """
        types = ['a', 'b', 'c']

        inf = fd.JointInference(
            sfs_neut=fd.Spectra({t: list(self.sfs_neut) for t in types}),
            sfs_sel=fd.Spectra({t: list(self.sfs_sel) for t in types}),
            shared_params=[fd.SharedParams(params=['S_d', 'b'], types=['a', 'b'])],
            covariates=[fd.Covariate(param='S_d', values=dict(a=0.3, b=0.6, c=0.9))],
            do_bootstrap=False
        )

        self.assertEqual([(s.params, s.types) for s in inf.shared_params], [(['b'], ['a', 'b']), (['S_d'], types)])

        self.assertEqual(sorted(inf.get_optimized_param_names()), ['a:b.b', 'a:b:c.S_d', 'a:b:c.c0', 'c.b'])

    def test_sharing_all_parameters_shares_those_not_fixed(self):
        """
        ``SharedParams(params='all')`` shares the parameters that are not fixed. It included the fixed ones,
        which raised an error for the default fixed parameters.
        """
        inf = self._make(shared_params=[fd.SharedParams()])

        self.assertEqual([(s.params, s.types) for s in inf.shared_params], [(['S_d', 'b'], ['a', 'b'])])

        # a fixed parameter that is listed explicitly raises as before
        with self.assertRaises(ValueError):
            self._make(shared_params=[fd.SharedParams(params=['S_d', 'eps'])])

    def test_plot_inferred_parameters_applies_point_estimate(self):
        """
        The parameter plot of a joint inference shows the requested point estimate. The argument was not
        passed on, so the mean of the bootstraps was always shown.
        """
        inf = self._make(shared_params=[fd.SharedParams(params=['S_d'])], do_bootstrap=True, n_bootstraps=3)
        inf.run()

        heights = {
            point_estimate: [bar.get_height() for bar in inf.plot_inferred_parameters(
                show=False, point_estimate=point_estimate, scale='lin'
            ).patches]
            for point_estimate in ['original', 'mean']
        }

        self.assertEqual(len(heights['original']), len(heights['mean']))
        self.assertFalse(np.allclose(heights['original'], heights['mean']))

    def test_stats_discretized_and_bucket_sizes(self):
        """
        The statistics of the discretized DFE are returned per inference, and the plot of bucket sizes, which
        belongs to a single inference, raises a ``NotImplementedError``. Both failed with a ``TypeError``.
        """
        inf = self._make(shared_params=[fd.SharedParams(params=['S_d'])], do_bootstrap=True, n_bootstraps=3)
        inf.run()

        stats = inf.get_stats_discretized()

        self.assertEqual(set(stats), set(inf.get_inferences()))

        for center, errors, ci in stats.values():
            self.assertEqual(center.shape, (6,))
            self.assertAlmostEqual(center.sum(), 1, places=6)

        with self.assertRaises(NotImplementedError):
            inf.plot_bucket_sizes(show=False)

        inf.marginal_inferences['all'].plot_bucket_sizes(show=False)

    def test_serializing_lambda_covariate_callback_warns(self):
        """
        Serializing a joint inference whose covariate callback is a lambda logs a warning, as the callback is
        not restored. It was dropped without notice, so the restored inference used the default covariate.
        """
        callback = lambda covariate, type, params: fd.Covariate.apply_default(covariate, type, params)

        inf = self._make(covariates=[fd.Covariate(param='S_d', values=dict(a=0.3, b=0.6), callback=callback)])

        with self.assertLogs('fastdfe', level='WARNING') as logs:
            inf.to_json()

        self.assertTrue(any('covariate' in message for message in logs.output))

        with self.assertNoLogs('fastdfe', level='WARNING'):
            self._make(covariates=[fd.Covariate(param='S_d', values=dict(a=0.3, b=0.6))]).to_json()

    def test_serializing_partial_covariate_callback(self):
        """
        A joint inference whose covariate callback is a ``functools.partial`` of a module-level function is
        serialized and restored with it. Reading the qualified name of the callback raised an
        ``AttributeError``.
        """
        callback = functools.partial(fd.Covariate.apply_default)

        inf = self._make(covariates=[fd.Covariate(param='S_d', values=dict(a=0.3, b=0.6), callback=callback)])

        with self.assertNoLogs('fastdfe', level='WARNING'):
            restored = fd.JointInference.from_json(inf.to_json())

        self.assertIs(restored.covariates['c0'].callback.func, fd.Covariate.apply_default)

    def test_evaluate_likelihood_at_mle_with_covariate(self):
        """
        The likelihood evaluated at the MLE parameters of a joint inference with a covariate equals the
        likelihood of the fit. The parameters include the effect of the covariate, which was applied again.
        """
        inf = self._make(
            covariates=[fd.Covariate(param='S_d', values=dict(a=-1000, b=-5000), x0=1)],
            fixed_params=dict(all=dict(eps=0, p_b=0, S_b=1, h=0.5)),
            n_runs=3
        )
        inf.run()

        self.assertNotAlmostEqual(inf.params_mle['a']['c0'], 0, places=3)
        self.assertAlmostEqual(inf.evaluate_likelihood(inf.params_mle), inf.likelihood, places=6)
        self.assertAlmostEqual(inf.evaluate_likelihood(inf.params_mle_raw), inf.likelihood, places=6)

    def test_marginal_bootstraps_are_independent_across_types(self):
        """
        The marginal inferences of the types resample their spectra independently, also when the types hold the
        same data. They shared one seed, so their bootstraps were drawn from the same random numbers.
        """
        inf = self._make(do_bootstrap=True, n_bootstraps=3)
        inf.run()

        a, b = [inf.marginal_inferences[t] for t in ['a', 'b']]

        self.assertNotEqual(a.seed, b.seed)
        self.assertFalse(np.allclose(a.bootstraps.likelihood, b.bootstraps.likelihood))

        # the same seed gives the same result
        again = self._make(do_bootstrap=True, n_bootstraps=3)
        again.run()

        np.testing.assert_array_equal(again.marginal_inferences['a'].bootstraps.likelihood, a.bootstraps.likelihood)

    def test_nested_models_share_all_free_parameters(self):
        """
        ``SharedParams(params='all')`` shares every parameter a nested model leaves free. The parameters
        were resolved once against the fixed parameters of the constructor, so with the default fixed
        parameters the full nested models estimated ``p_b``, ``S_b`` and ``eps`` once per type.
        """
        inf = self._make(shared_params=[fd.SharedParams(types='all', params='all')])

        self.assertEqual(set(inf.shared_params[0].params), {'S_d', 'b'})

        _, inferences = inf.compare_nested_models()

        self.assertEqual(set(inferences['full.anc'].shared_params[0].params), {'S_d', 'b', 'p_b', 'S_b', 'eps'})
        self.assertEqual(inferences['full.anc'].get_n_optimized(), 5)
        self.assertEqual(inferences['full.no_anc'].get_n_optimized(), 4)
        self.assertEqual(inferences['dele.anc'].get_n_optimized(), 3)
        self.assertEqual(inferences['dele.no_anc'].get_n_optimized(), 2)

    def test_getters_run_joint_inference_if_required(self):
        """
        The discretized statistics and the bootstrap parameters of a joint inference that was not run yet
        trigger the fit. They went to the inferences of the types directly, which refuse to be run on their own.
        """
        self.assertEqual(
            set(self._make(do_bootstrap=True, n_bootstraps=2).get_stats_discretized()),
            {'marginal.all', 'marginal.a', 'marginal.b', 'joint.a', 'joint.b'}
        )
        self.assertIn('a.S_d', self._make().get_bootstrap_params())
        self.assertIn('a.S_d', self._make().get_bootstrap_param_names())

    def test_config_leaves_covariate_bounds_to_the_covariate(self):
        """
        An inference restored from the config of a joint inference takes the bounds and scale of a covariate
        from the covariate. The config stored them among the bounds and scales, where they overrode those of
        a covariate the config was updated with.
        """
        inf = self._make(covariates=[fd.Covariate(param='S_d', values=dict(a=1, b=2))])

        covariate = fd.Covariate(param='b', values=dict(a=1, b=2), bounds=(1e-2, 1), bounds_scale='lin', x0=0.5)

        restored = fd.JointInference.from_config(inf.create_config().update(covariates=[covariate]))

        self.assertEqual(restored.bounds['c0'], (1e-2, 1))
        self.assertEqual(restored.scales['c0'], 'lin')

    def test_base_inference_from_config_of_joint_inference(self):
        """
        A ``BaseInference`` can be created from the config of a joint inference, for the spectra pooled over
        the types. The config holds the fixed parameters and initial values by type, which raised a
        ``ValueError``.
        """
        config = self._make(x0=dict(a=dict(S_d=-777.0))).create_config()

        inf = fd.BaseInference.from_config(config)

        self.assertEqual(inf.fixed_params, dict(all=dict(eps=0, h=0.5, p_b=0, S_b=1)))
        np.testing.assert_array_equal(inf.sfs_sel.data, 2 * np.array(list(self.sfs_sel)))

    def test_shared_params_docstring_example(self):
        """
        The usage example of ``SharedParams`` constructs. It shared ``eps``, which is fixed by default, and
        raised a ``ValueError``.
        """
        source = textwrap.dedent(fd.SharedParams.__doc__.split('::')[1])

        exec(source.replace('inf.run()', ''), {})

    def test_folded_joint_inference_with_fixed_parameters_by_type(self):
        """
        A folded joint inference whose beneficial parameters are fixed for each type does not warn about
        estimating the full DFE. Only the parameters fixed for ``all`` were considered, so the warning was
        logged for an inference restored from its own config.
        """
        fixed = dict(eps=0, h=0.5, p_b=0, S_b=1)
        spectra = dict(
            sfs_neut=fd.Spectra(dict(a=list(self.sfs_neut.fold()), b=list(self.sfs_neut.fold()))),
            sfs_sel=fd.Spectra(dict(a=list(self.sfs_sel.fold()), b=list(self.sfs_sel.fold())))
        )

        def warnings(**kwargs) -> list:
            """
            Warnings on the full DFE logged when constructing the inference.
            """
            with self.assertLogs('fastdfe', level='WARNING') as logs:
                logging.getLogger('fastdfe').warning('sentinel')
                fd.JointInference(
                    intervals_del=(-1.0e+8, -1.0e-5, 20), intervals_ben=(1.0e-5, 1.0e4, 20), parallelize=False,
                    **spectra, **kwargs
                )

            return [m for m in logs.output if 'full DFE' in m]

        self.assertEqual(warnings(fixed_params=dict(a=fixed, b=fixed)), [])
        self.assertNotEqual(warnings(fixed_params=dict(a=fixed, b=dict(eps=0, h=0.5))), [])

    def test_joint_inferences_of_types_hold_likelihoods_of_runs(self):
        """
        The inferences of the types hold the likelihoods of the runs of the joint fit, so they can be plotted.
        They held none, and ``plot_likelihoods`` raised a ``TypeError``.
        """
        inf = self._make(shared_params=[fd.SharedParams(types='all', params=['S_d'])], n_runs=2)
        inf.run()

        for t in inf.types:
            self.assertEqual(inf.joint_inferences[t].likelihoods, inf.likelihoods)

            inf.joint_inferences[t].plot_likelihoods(show=False)

    def test_lrt_p_values_of_sharing_and_covariates(self):
        """
        The p-values of the tests of sharing and of covariates are the chi-square tail probabilities of
        twice the difference in log-likelihood, with one degree of freedom for one shared parameter over two
        types and for one covariate.
        """
        kwargs = dict(
            sfs_neut=fd.Spectra(dict(
                a=[177130, 997, 441, 228, 156, 117, 114, 83, 105, 109, 652],
                b=[172528, 3612, 1359, 790, 584, 427, 325, 234, 166, 76, 31]
            )),
            sfs_sel=fd.Spectra(dict(
                a=[797939, 1329, 499, 265, 162, 104, 117, 90, 94, 119, 794],
                b=[791106, 5326, 1741, 1005, 756, 546, 416, 294, 177, 104, 41]
            )),
            intervals_del=(-1.0e+8, -1.0e-5, 20),
            intervals_ben=(1.0e-5, 1.0e4, 20),
            fixed_params=dict(all=dict(eps=0, p_b=0, S_b=1, h=0.5)),
            do_bootstrap=False,
            parallelize=False,
            n_runs=3,
            seed=0
        )

        shared = fd.JointInference(shared_params=[fd.SharedParams(types='all', params=['S_d'])], **kwargs)
        shared.run()

        ll_marginal = sum(inf.likelihood for inf in shared.marginals_without_all().values())

        self.assertGreater(ll_marginal - shared.likelihood, 0.01)
        self.assertAlmostEqual(shared.perform_lrt_shared(), chi2.sf(2 * (ll_marginal - shared.likelihood), 1))

        covariate = fd.JointInference(covariates=[fd.Covariate(param='b', values=dict(a=1, b=5))], **kwargs)
        covariate.run()

        simple = covariate.run_joint_without_covariates(do_bootstrap=False)

        self.assertGreater(covariate.likelihood - simple.likelihood, 0.01)
        self.assertAlmostEqual(
            covariate.perform_lrt_covariates(), chi2.sf(2 * (covariate.likelihood - simple.likelihood), 1)
        )

    def test_plot_covariate_shows_given_label_as_text(self):
        """
        A y-axis label given to ``plot_covariate`` is shown as given, and the default is the parameter in
        math mode. Every label was set in math mode, which dropped the spaces of a given one.
        """
        inf = self._make(
            covariates=[fd.Covariate(param='S_d', values=dict(a=1, b=5))], do_bootstrap=True, n_bootstraps=2
        )

        self.assertEqual(inf.plot_covariate(show=False, ylabel='strength of selection').get_ylabel(),
                         'strength of selection')
        self.assertEqual(inf.plot_covariate(show=False).get_ylabel(), '$S_d$')

    def test_nested_models_reject_fixed_covariate_parameter(self):
        """A nested model cannot hold a parameter fixed that has a covariate."""
        inf = self._make(
            covariates=[fd.Covariate(param='p_b', values=dict(a=0.3, b=0.6))],
            fixed_params=dict(all=dict(h=0.5, eps=0))
        )

        with self.assertRaises(ValueError):
            inf.compare_nested_models()

    def test_covariate_lrt_and_plots(self):
        """Covariate LRT (tiny bootstrap) plus the joint plots."""
        inf = self._make(
            covariates=[fd.Covariate(param='S_d', values=dict(a=0.3, b=0.6))],
            fixed_params=dict(all=dict(eps=0, p_b=0, S_b=1, h=0.5)),
            do_bootstrap=True,
            n_bootstraps=2,
        )
        self.assertLessEqual(inf.perform_lrt_covariates(), 1)

        inf.plot_discretized(show=False)
        inf.plot_inferred_parameters(show=False)
        inf.plot_covariate(show=False)

    def test_create_config_preserves_divergence_target_sizes(self):
        """
        A joint inference recreated from its config, as done for the model without covariates, uses divergence
        with the same per-type target sizes.
        """
        nsd_neut = dict(a=self.sfs_neut.n_sites, b=self.sfs_neut.n_sites)
        nsd_sel = dict(a=self.sfs_sel.n_sites, b=self.sfs_sel.n_sites)
        inf = self._make(n_sites_div_neut=nsd_neut, n_sites_div_sel=nsd_sel)

        config = inf.create_config()

        self.assertEqual(config.data['n_sites_div_neut'], nsd_neut)
        self.assertEqual(config.data['n_sites_div_sel'], nsd_sel)

        inf2 = fd.JointInference.from_config(config)

        self.assertTrue(inf._use_divergence)
        self.assertTrue(inf2._use_divergence)
        self.assertTrue(all(i._use_divergence for i in inf2.joint_inferences.values()))

    def test_joint_likelihood_with_divergence_equals_sum_of_marginals(self):
        """
        With divergence counts in the likelihood and no shared parameters, the joint likelihood equals the sum
        of the marginal likelihoods, and the joint bootstrap runs. The joint likelihood did not receive the
        divergence target sizes and raised a TypeError.
        """
        inf = self._make(
            n_sites_div_neut=dict(a=self.sfs_neut.n_sites, b=self.sfs_neut.n_sites),
            n_sites_div_sel=dict(a=self.sfs_sel.n_sites, b=self.sfs_sel.n_sites),
            shared_params=[],
            do_bootstrap=True,
            n_bootstraps=2
        )
        inf.run()

        marginal = sum(m.likelihood for t, m in inf.marginal_inferences.items() if t != 'all')

        self.assertTrue(inf._use_divergence)
        self.assertAlmostEqual(inf.likelihood, marginal, places=3)
        self.assertEqual(len(inf.bootstraps), 2)

        # the bootstrapped divergence estimates of each type use the divergence counts of the sample
        for joint in inf.joint_inferences.values():
            bs = joint.bootstraps

            np.testing.assert_allclose(
                bs.omega, (bs.n_div_sel / joint.n_sites_div_sel) / (bs.n_div_neut / joint.n_sites_div_neut), rtol=1e-12
            )

    def test_type_name_with_regex_characters(self):
        """
        Type names are taken literally. A name such as ``a+b`` was used as a pattern when selecting the spectra
        of the type, matched nothing and raised an ``IndexError``.
        """
        inf = fd.JointInference(
            sfs_neut=fd.Spectra({'a+b': list(self.sfs_neut), 'c': list(self.sfs_neut)}),
            sfs_sel=fd.Spectra({'a+b': list(self.sfs_sel), 'c': list(self.sfs_sel)}),
            shared_params=[fd.SharedParams(types='all', params=['b'])],
            intervals_del=(-1.0e+8, -1.0e-5, 20),
            intervals_ben=(1.0e-5, 1.0e4, 20),
            do_bootstrap=True,
            n_bootstraps=2,
            n_runs=1,
            parallelize=False
        )
        inf.run()

        self.assertEqual(list(inf.joint_inferences), ['a+b', 'c'])
        np.testing.assert_array_equal(inf.joint_inferences['a+b'].sfs_sel.data, self.sfs_sel.data)
        self.assertEqual(len(inf.joint_inferences['a+b'].bootstraps), 2)

    def test_run_takes_do_bootstrap(self):
        """
        ``run`` applies its ``do_bootstrap`` argument, as ``BaseInference.run`` does. The argument was ignored,
        so an inference constructed with ``do_bootstrap=True`` was bootstrapped regardless.
        """
        inf = self._make(
            shared_params=[fd.SharedParams(types='all', params=['S_d'])], do_bootstrap=True, n_bootstraps=2
        )
        inf.run(do_bootstrap=False)

        self.assertIsNone(inf.bootstraps)

        inf.run(do_bootstrap=True)

        self.assertEqual(len(inf.bootstraps), 2)

    def test_lrt_shared_without_degrees_of_freedom(self):
        """
        With two types and a covariate on the shared parameter, the joint model has as many free parameters as
        the marginal models, so the test of sharing is not defined. It returned a p-value of 0, which reads as
        strong evidence against sharing.
        """
        inf = self._make(
            covariates=[fd.Covariate(param='S_d', values=dict(a=1, b=2))],
            fixed_params=dict(all=dict(eps=0, p_b=0, S_b=1, h=0.5))
        )
        inf.run()

        with self.assertLogs('fastdfe', level='WARNING') as logs:
            p = inf.perform_lrt_shared()

        self.assertTrue(np.isnan(p))
        self.assertIn('not defined', logs.output[0])

        # sharing without a covariate leaves one degree of freedom
        shared = self._make(
            shared_params=[fd.SharedParams(types='all', params=['S_d'])],
            fixed_params=dict(all=dict(eps=0, p_b=0, S_b=1, h=0.5))
        )
        shared.run()

        self.assertTrue(0 <= shared.perform_lrt_shared() <= 1)

    def test_fixed_dominance_overridden_for_one_type(self):
        """
        A dominance coefficient fixed for all types can be fixed at another value for one type. The
        discretization was precomputed for the value given under ``all`` alone, so the run raised.
        """
        inf = self._make(
            fixed_params=dict(all=dict(eps=0, h=0.5, S_b=1, p_b=0), a=dict(h=0.3)),
            shared_params=[fd.SharedParams(types='all', params=['S_d'])],
            intervals_h=(0, 1, 6)
        )
        inf.run()

        self.assertIsNone(inf.discretization.h)
        self.assertEqual(inf.params_mle['a']['h'], 0.3)
        self.assertEqual(inf.params_mle['b']['h'], 0.5)

        # a value common to all types is still precomputed alone
        self.assertEqual(self._make(fixed_params=dict(all=dict(eps=0, h=0.5, S_b=1, p_b=0))).discretization.h, 0.5)

    def test_dfe_spectra_and_summary_per_type(self):
        """
        ``get_dfe`` and ``get_summary`` return one entry per type and ``get_spectra`` the neutral, selected and
        modelled SFS of each type. The inherited methods assumed a single type and raised.
        """
        inf = self._make(shared_params=[fd.SharedParams(types='all', params=['b'])])
        inf.run()

        spectra = inf.get_spectra()

        self.assertEqual(
            spectra.types, [f'{t}.{name}' for t in ('a', 'b') for name in ('neutral', 'selected', 'modelled')]
        )
        np.testing.assert_array_equal(spectra['b.modelled'].data, inf.sfs_mle['b'].data)
        np.testing.assert_array_equal(spectra['a.selected'].data, inf.joint_inferences['a'].sfs_sel.data)

        dfes = inf.get_dfe()

        self.assertEqual(list(dfes), ['a', 'b'])
        self.assertEqual(dfes['a'].params, inf.joint_inferences['a'].params_mle)

        summaries = inf.get_summary()

        self.assertEqual(list(summaries), ['a', 'b'])
        self.assertEqual(summaries['b'].spectra.types, ['neutral', 'selected', 'modelled'])

        # the summaries can be serialized
        self.assertIn('modelled', summaries['a'].to_json())

    def test_model_without_covariates_keeps_folding(self):
        """
        The model without covariates used by the covariate LRT is fitted to folded spectra when the joint
        inference is. Its config lacked ``folded``, so with ``folded=True`` on unfolded spectra the two
        likelihoods were computed on different data. Both types hold the same spectra here, so the covariate
        cannot improve the fit.
        """
        inf = self._make(
            covariates=[fd.Covariate(param='S_d', values=dict(a=1, b=2))],
            fixed_params=dict(all=dict(eps=0, p_b=0, S_b=1, h=0.5)),
            folded=True,
            n_bootstrap_retries=5,
            n_runs=2
        )
        inf.run()

        simple = inf.run_joint_without_covariates(do_bootstrap=False)

        self.assertIs(simple.folded, True)
        self.assertEqual(simple.n_bootstrap_retries, 5)
        self.assertAlmostEqual(simple.likelihood, inf.likelihood, places=2)

    def test_bootstrap_type_name_contained_in_another(self):
        """
        The bootstraps of a type hold the columns of that type only, also when its name is contained in the name
        of another type. The columns were selected by an unanchored pattern, which assigned the columns of
        ``noncoding`` to ``coding`` as well and made the bootstrap fail.
        """
        sfs_sel = list(self.sfs_sel)

        inf = fd.JointInference(
            sfs_neut=fd.Spectra(dict(coding=list(self.sfs_neut), noncoding=list(self.sfs_neut))),
            sfs_sel=fd.Spectra(dict(coding=sfs_sel, noncoding=sfs_sel[:1] + [2 * sfs_sel[1]] + sfs_sel[2:])),
            shared_params=[fd.SharedParams(types='all', params=['b'])],
            fixed_params=dict(all=dict(eps=0, p_b=0, S_b=1, h=0.5)),
            intervals_del=(-1.0e+8, -1.0e-5, 20),
            intervals_ben=(1.0e-5, 1.0e4, 20),
            do_bootstrap=True,
            n_bootstraps=2,
            n_runs=1,
            parallelize=False
        )
        inf.run()

        for t, joint in inf.joint_inferences.items():
            self.assertFalse(joint.bootstraps.columns.duplicated().any())
            np.testing.assert_array_equal(joint.bootstraps['S_d'], inf.bootstraps[f'{t}.S_d'])

    def test_rerun_recomputes_model_without_covariates(self):
        """
        ``run_joint_without_covariates`` caches its result. After the joint inference is rerun, a further call
        recomputes the model without covariates instead of returning that of the previous run.
        """
        inf = self._make(
            covariates=[fd.Covariate(param='S_d', values=dict(a=0.3, b=0.6))],
            fixed_params=dict(all=dict(eps=0, p_b=0, S_b=1, h=0.5))
        )
        inf.run()

        first = inf.run_joint_without_covariates(do_bootstrap=False)

        self.assertIs(first, inf.run_joint_without_covariates(do_bootstrap=False))

        inf.n_runs = 2
        inf.run()

        second = inf.run_joint_without_covariates(do_bootstrap=False)

        self.assertIsNot(first, second)
        self.assertEqual(second.n_runs, 2)

    def test_rerun_discards_bootstraps(self):
        """A joint inference rerun without bootstrapping does not keep the bootstraps of the previous run."""
        inf = self._make(shared_params=[fd.SharedParams(types='all', params=['S_d'])], do_bootstrap=True, n_bootstraps=2)
        inf.run()

        self.assertEqual(len(inf.bootstraps), 2)

        inf.do_bootstrap = False
        inf.run()

        self.assertIsNone(inf.bootstraps)
        self.assertTrue(all(i.bootstraps is None for i in inf.joint_inferences.values()))

    def test_x0_per_type_seeds_marginal_inference(self):
        """
        Initial values given under a type seed the marginal inference of that type, the other types start from the
        pooled estimates. Keys other than ``all`` were ignored.
        """
        inf = self._make(x0=dict(a=dict(S_d=-777.0)))
        inf.run()

        pooled = inf.marginal_inferences['all'].params_mle

        self.assertEqual(inf.marginal_inferences['a'].x0['all']['S_d'], -777.0)
        self.assertEqual(inf.marginal_inferences['a'].x0['all']['b'], pooled['b'])
        self.assertEqual(inf.marginal_inferences['b'].x0['all'], pooled)

        # the per-type initial values survive the config round-trip
        self.assertEqual(inf.create_config().data['x0']['a'], dict(S_d=-777.0))
        self.assertEqual(fd.JointInference.from_config(inf.create_config())._x0_types, dict(a=dict(S_d=-777.0)))

    def test_x0_unknown_keys_raise(self):
        """Keys of ``x0`` other than ``all`` and the types are rejected."""
        with self.assertRaises(ValueError):
            self._make(x0=dict(c=dict(S_d=-777.0)))

        with self.assertRaises(ValueError):
            fd.BaseInference(sfs_neut=self.sfs_neut, sfs_sel=self.sfs_sel, x0=dict(a=dict(S_d=-777.0)))

    def test_bootstrap_sample_runner_does_not_capture_the_inference(self):
        """
        The function sent to the workers for each bootstrap sample holds no reference to the inference object.
        Capturing it serialized the whole inference, bootstrap tables included, once per sample.
        """
        inf = self._make()
        inf.run()

        for runner in [inf._get_run_bootstrap_sample(), inf.marginal_inferences['all']._get_run_bootstrap_sample()]:
            self.assertNotIn('self', runner.__code__.co_freevars)

    def test_run_returns_modelled_spectra(self):
        """run() returns the modelled SFS of the joint inference for each type."""
        inf = self._make()
        sfs = inf.run()

        self.assertIsInstance(sfs, fd.Spectra)
        self.assertIs(sfs, inf.sfs_mle)
        self.assertEqual(sorted(sfs.types), ['a', 'b'])

        for t in ['a', 'b']:
            np.testing.assert_array_equal(sfs[t].polymorphic, inf.joint_inferences[t].sfs_mle.polymorphic)

    def test_shared_lrt(self):
        """Shared-parameter LRT path and shared-param introspection."""
        inf = self._make(shared_params=[fd.SharedParams(types='all', params=['S_d'])])
        inf.run()

        self.assertEqual(inf.get_shared_param_names(), ['S_d'])
        self.assertLessEqual(inf.perform_lrt_shared(), 1)

    def test_no_covariates_lrt_raises(self):
        """perform_lrt_covariates without covariates raises."""
        inf = self._make()
        inf.run()

        with self.assertRaises(ValueError):
            inf.perform_lrt_covariates()

    def test_shared_params_and_covariate(self):
        """A joint fit with a shared parameter and a covariate yields finite per-type estimates."""
        inf = self._make(
            shared_params=[fd.SharedParams(types='all', params=['b'])],
            covariates=[fd.Covariate(param='S_d', values=dict(a=1.0, b=2.0))],
        )
        inf.run()

        for t in inf.types:
            self.assertTrue(np.isfinite(inf.marginal_inferences[t].get_alpha()))

        self.assertTrue(np.isfinite(inf.likelihood))

    def test_compare_nested_models_keeps_h_fixed(self):
        """
        Regression: with a covariate the joint fixed params are expanded per type, so the fixed
        dominance h must still be kept fixed across the nested models. Otherwise h is freed, the
        optimizer varies it, and the h=0.5 discretization raises.
        """
        inf = self._make(covariates=[fd.Covariate(param='S_d', values=dict(a=0.3, b=0.6))])

        # h is fixed to 0.5 by default, per type
        self.assertNotIn('all', inf.fixed_params)

        # the full nested-model comparison must run without freeing h (previously raised
        # "Dominance coefficient ... does not match precomputed value 0.5.")
        P, inferences = inf.compare_nested_models()

        # h stays fixed at 0.5 in every nested sub-inference and type
        for sub in inferences.values():
            for t in sub.types:
                self.assertEqual(sub.fixed_params[t].get('h'), 0.5)

    def test_getters_and_serialization(self):
        """Aggregate getters, spectra and a to_file/from_file round-trip."""
        inf = self._make()
        inf.run()

        for getter in (inf.get_alpha, inf.get_omega, inf.get_omega_a):
            self.assertEqual(set(getter()), set(inf.types))

        out = 'scratch/fast_joint_inference.json'
        inf.to_file(out)
        inf2 = fd.JointInference.from_file(out)
        self.assertEqual(set(inf2.types), set(inf.types))

    def test_sfs_comparison_plot_residual_and_aggregate_none(self):
        """SFS-comparison plot, the L2 residual, and the (undefined) aggregate omega/omega_a."""
        inf = self._make()
        inf.run()

        # observed-vs-modelled SFS comparison, both the single-axes and the subplot layout
        inf.plot_sfs_comparison(show=False)
        inf.plot_sfs_comparison(sfs_types=['neutral', 'selected'], use_subplots=True, show=False)

        self.assertTrue(np.isfinite(inf.get_residual(2)))

        # there is no single aggregate omega / omega_a for a joint inference
        self.assertIsNone(inf.omega)
        self.assertIsNone(inf.omega_a)

    def test_set_fixed_params_propagates_to_all_inferences(self):
        """``_set_fixed_params`` expands over types and propagates to every sub-inference."""
        inf = self._make()

        inf._set_fixed_params(dict(all=dict(eps=0.01)))

        # expanded to every concrete type
        self.assertEqual(set(inf.fixed_params), set(inf.types))
        for t in inf.types:
            self.assertEqual(dict(eps=0.01), inf.fixed_params[t])
            self.assertEqual(dict(all=dict(eps=0.01)), inf.marginal_inferences[t].fixed_params)
            self.assertEqual(dict(all=dict(eps=0.01)), inf.joint_inferences[t].fixed_params)

        self.assertEqual(dict(all=dict(eps=0.01)), inf.marginal_inferences['all'].fixed_params)
        self.assertEqual({'a.eps': 0.01, 'b.eps': 0.01}, inf.optimization.fixed_params)

    #: Script running a seeded covariate joint inference and printing its run table, bootstraps and MLE
    covariate_script = textwrap.dedent("""
        import json
        import fastdfe as fd

        fd.Settings.parallelize = False
        fd.Settings.disable_pbar = True

        sfs_neut = [177130, 997, 441, 228, 156, 117, 114, 83, 105, 109, 652]
        sfs_sel = [797939, 1329, 499, 265, 162, 104, 117, 90, 94, 119, 794]

        inf = fd.JointInference(
            sfs_neut=fd.Spectra(dict(a=sfs_neut, b=sfs_neut)),
            sfs_sel=fd.Spectra(dict(a=sfs_sel, b=sfs_sel)),
            intervals_del=(-1.0e+8, -1.0e-5, 20),
            intervals_ben=(1.0e-5, 1.0e4, 20),
            covariates=[fd.Covariate(param='S_d', values=dict(a=0.3, b=0.6))],
            n_runs=4,
            n_bootstraps=2,
            parallelize=False
        )
        inf.run()

        print(json.dumps(dict(
            runs=inf.runs.drop(columns='result').to_csv(),
            bootstraps=inf.bootstraps.to_csv(),
            params_mle=repr(inf.params_mle)
        )))
    """)

    def test_covariate_inference_independent_of_global_numpy_rng(self):
        """
        Regression: a seeded covariate joint inference must not depend on the global numpy RNG state.
        ``Optimization.sample_value`` drew the sign of a ``symlog``-scaled parameter, which covariates are by
        default, via ``uniform.rvs()`` without a random state, i.e. from the unseeded global numpy RNG. The
        best run was unaffected, but the initial values of the other runs, the std across runs and the
        bootstrap replicates changed between otherwise identical sessions.
        """
        results = []

        for global_seed in (0, 1):
            np.random.seed(global_seed)

            inf = self._make(
                covariates=[fd.Covariate(param='S_d', values=dict(a=0.3, b=0.6))],
                n_runs=4,
                do_bootstrap=True,
                n_bootstraps=2
            )
            inf.run()
            results.append(inf)

        assert_frame_equal(results[0].runs, results[1].runs)
        assert_frame_equal(results[0].bootstraps, results[1].bootstraps)
        self.assertEqual(results[0].params_mle, results[1].params_mle)

    @pytest.mark.inference
    def test_covariate_inference_identical_across_processes(self):
        """
        Regression: a seeded covariate joint inference must give identical run tables, bootstrap replicates
        and MLE parameters in fresh processes with different ``PYTHONHASHSEED`` values. Each fresh process
        starts the global numpy RNG from OS entropy, and set iteration over strings follows the hash seed,
        so either source leaking into initial values or parameter packing makes regenerated outputs differ.
        """
        outputs = []

        for hash_seed in ('0', '1'):
            result = subprocess.run(
                [sys.executable, '-c', self.covariate_script],
                cwd=Path(__file__).parents[1],
                env=os.environ | dict(PYTHONHASHSEED=hash_seed, MPLBACKEND='Agg'),
                capture_output=True,
                text=True,
                check=True
            )
            outputs.append(json.loads(result.stdout.strip().splitlines()[-1]))

        self.assertEqual(outputs[0], outputs[1])

    @pytest.mark.inference
    def test_hatch_styles_identical_across_hash_seeds(self):
        """
        Regression: ``Visualization.get_hatch`` indexed hatch styles by the position of a label prefix in a
        set of prefixes, so whether the marginal or the joint bars were drawn with ``/`` or ``\\`` depended
        on ``PYTHONHASHSEED`` and plots of joint inferences differed between otherwise identical sessions.
        """
        script = (
            "from fastdfe.visualization import Visualization\n"
            "labels = ['marginal.all', 'marginal.a', 'marginal.b', 'joint.a', 'joint.b', 'other.a']\n"
            "print([Visualization.get_hatch(i, labels) for i in range(len(labels))])\n"
        )

        outputs = set()

        for hash_seed in ('0', '1', '2', '3'):
            result = subprocess.run(
                [sys.executable, '-c', script],
                cwd=Path(__file__).parents[1],
                env=os.environ | dict(PYTHONHASHSEED=hash_seed, MPLBACKEND='Agg'),
                capture_output=True,
                text=True,
                check=True
            )
            outputs.add(result.stdout.strip().splitlines()[-1])

        self.assertEqual(1, len(outputs))
