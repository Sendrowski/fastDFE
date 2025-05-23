import matplotlib.pyplot as plt
import numpy as np
import pytest

import fastdfe as fd
from fastdfe import JointInference, Config, SharedParams, Covariate, Spectra
from fastdfe.optimization import flatten_dict
from testing.test_base_inference import InferenceTestCase


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

        self.assertEqualInference(inference, inference2)

    def test_recreate_from_config(self):
        """
        Test whether inference can be recreated from config.
        """
        inf = JointInference.from_config_file(self.config_file)

        inf2 = JointInference.from_config(inf.create_config())

        self.assertEqualInference(inf, inf2, ignore_keys=['fixed_params'])

    def test_bootstrap_joint_inference(self):
        """
        Test that the bootstrap method works.
        """
        inference = JointInference.from_file(
            "testing/cache/fastdfe/templates/shared/shared_example_1/serialized.json"
        )

        inference.bootstrap(2)

    def test_perform_lrt_shared(self):
        """
        Test that the perform_lrt_shared method works.
        """
        inference = JointInference.from_file(
            "testing/cache/fastdfe/templates/shared/shared_example_1/serialized.json"
        )

        # inference.evaluate_likelihood(inference.params_mle)

        # inference.marginal_inferences['example_1'].evaluate_likelihood(
        # dict(all=inference.params_mle['example_1']))

        assert inference.perform_lrt_shared() < 1

        np.testing.assert_array_equal(inference.get_shared_param_names(), ['S_b', 'S_d', 'b', 'eps', 'p_b'])

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

        assert inf.perform_lrt_covariates() < 1

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
            fixed_params=dict(all=dict(eps=0, p_b=0, S_b=1)),
            do_bootstrap=True,
            n_bootstraps=2,
            n_runs=1,
            covariates=[Covariate(param='S_d', values=dict(pendula=0.3, pubescens=0.6))]
        )

        assert inf.perform_lrt_covariates()

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
            all=dict(eps=0.123),
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
            'all': {'eps': 0.123},
            'example_1': {'eps': 0.123, 'b': 1.234},
            'example_2': {'eps': 0.123, 'b': 1.132},
            'example_3': {'eps': 0.123},
            'pendula': {'eps': 0.123, 'S_b': 10.1},
            'pubescens': {'eps': 0.123},
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
            fixed_params={'all': dict(S_b=1, eps=0, p_b=0)},
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
        inf = JointInference.from_config_file(
            "resources/configs/shared/covariates_Sd_fixed_params/config.yaml"
        )

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
            fixed_params={'all': dict(S_b=1, eps=0)},
            parallelize=True,
            do_bootstrap=True,
            n_bootstraps=2
        )

        # run inference
        inf.run()

        inf_no_cov = inf.run_joint_without_covariates(do_bootstrap=True)

        p = inf.perform_lrt_covariates(do_bootstrap=True)

        inf.plot_inferred_parameters()

        # the test should be non-significant as we chose random covariates
        assert p > 0.05

    @pytest.mark.slow
    def test_covariates_strong_correlation(self):
        """
        Check that strong correlation between covariates is detected as significant.
        """
        # parse SFS
        spectra = Spectra.from_file("resources/SFS/betula/pendula/degeneracy_ref_base_tutorial.csv")

        # create inference object
        inf = JointInference(
            sfs_neut=spectra[['neutral.*']].merge_groups(1),
            sfs_sel=spectra[['selected.*']].merge_groups(1),
            covariates=[Covariate(param='S_d', values=dict(A=-100000, C=-2045, T=-60504, G=-5024))],
            fixed_params={'all': dict(S_b=1, eps=0, p_b=0)},
            parallelize=True,
            do_bootstrap=True,
            n_bootstraps=100,
            n_runs=20
        )

        # run inference
        inf.run()

        inf_no_cov = inf.run_joint_without_covariates(do_bootstrap=True)

        p = inf.perform_lrt_covariates(do_bootstrap=True)

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
            fixed_params={'all': dict(S_b=1, eps=0, p_b=0)},
            parallelize=True,
            do_bootstrap=True,
            n_runs=10,
            n_bootstraps=2
        )

        inf.plot_inferred_parameters()

    def test_get_cis_params_mle_with_boostraps(self):
        """
        Get the MLE parameters from the cis inference.
        """
        inference = JointInference.from_file(
            "testing/cache/fastdfe/templates/shared/shared_example_1/serialized.json"
        )

        inference.bootstrap(2)

        cis = inference.get_cis_params_mle()

    def test_get_cis_params_mle_no_boostraps(self):
        """
        Get the MLE parameters from the cis inference.
        """
        inference = JointInference.from_file(
            "testing/cache/fastdfe/templates/shared/shared_example_1/serialized.json"
        )

        cis = inference.get_cis_params_mle()

        # check that all values are None
        for inf in cis.values():
            assert inf is None

    def test_get_discretized_errors_with_bootstraps(self):
        """
        Get the discretized DFE errors.
        """
        inference = JointInference.from_file(
            "testing/cache/fastdfe/templates/shared/shared_example_1/serialized.json"
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
                fixed_params={'all': dict(eps=0)},
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
                fixed_params={'all': dict(eps=0)},
                shared_params=[SharedParams(types=["pendula", "pubescens.bar.foo"], params=["S_d", "S_b", "p_b"])],
                do_bootstrap=True
            )

    @staticmethod
    def test_manuscript_example():
        """
        Test the example in the manuscript.

        Note the warning on a large residual for the joint inference is not
        unusual as the fit to each component SFS naturally is worse when sharing parameters.
        """
        inf = fd.JointInference.from_config_file(
            "https://github.com/Sendrowski/fastDFE/"
            "blob/master/resources/configs/arabidopsis/"
            "covariates_example.yaml?raw=true"
        )

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
