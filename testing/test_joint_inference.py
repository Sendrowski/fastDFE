import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
from pandas._testing import assert_frame_equal

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
        dominance h must still be recovered (via _get_fixed_h) and kept fixed across the nested
        models. Otherwise h is freed, the optimizer varies it, and the h=0.5 discretization raises.
        """
        inf = self._make(covariates=[fd.Covariate(param='S_d', values=dict(a=0.3, b=0.6))])

        # h is fixed to 0.5 by default; the helper must recover it despite per-type expansion
        self.assertNotIn('all', inf.fixed_params)
        self.assertEqual(inf._get_fixed_h(), 0.5)

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

        inf._set_fixed_params(dict(all=dict(eps=0)))

        # expanded to every concrete type
        self.assertEqual(set(inf.fixed_params), set(inf.types))
        for t in inf.types:
            self.assertEqual(0, inf.fixed_params[t]['eps'])
            self.assertEqual(0, inf.marginal_inferences[t].fixed_params['all']['eps'])
            self.assertEqual(0, inf.joint_inferences[t].fixed_params['all']['eps'])

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
