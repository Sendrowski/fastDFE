import numpy as np

from testing import prioritize_installed_packages

prioritize_installed_packages()

import pytest

from fastdfe import JointInference, Config, SharedParams, Covariate, Spectra
from fastdfe.optimization import flatten_dict
from testing.test_base_inference import AbstractInferenceTestCase


class JointInferenceTestCase(AbstractInferenceTestCase):
    config_file = "testing/configs/pendula.pubescens.example_1.example_2.example_3_C_full_anc/config.yaml"

    show_plots = False

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
            "testing/fastdfe/templates/shared/shared_example_1/serialized.json"
        )

        inference.bootstrap(2)

    def test_perform_lrt_shared(self):
        """
        Test that the perform_lrt_shared method works.
        """
        inference = JointInference.from_file(
            "testing/fastdfe/templates/shared/shared_example_1/serialized.json"
        )

        # inference.evaluate_likelihood(inference.params_mle)

        # inference.marginal_inferences['example_1'].evaluate_likelihood(
        # dict(all=inference.params_mle['example_1']))

        assert inference.perform_lrt_shared() < 1

    def test_perform_lrt_covariates_no_covariates_raises_error(self):
        """
        Test that the perform_lrt_covariates method works.
        """
        inference = JointInference.from_file(
            "testing/fastdfe/templates/shared/shared_example_1/serialized.json"
        )

        with pytest.raises(ValueError):
            inference.perform_lrt_covariates()

    def test_perform_lrt_covariates(self):
        """
        Test that the perform_lrt_covariates method works.
        """
        # inf = JointInference.from_config_file("resources/configs/shared/covariates_Sd_example_1/config.yaml")
        # inf.run()

        inf = JointInference.from_file(
            "testing/fastdfe/templates/shared/covariates_Sd_example_1/serialized.json"
        )

        assert inf.perform_lrt_covariates() < 1

    def test_plot_all_without_bootstraps(self):
        """
        Test that the plot_all method works.
        """
        inference = JointInference.from_file(
            "testing/fastdfe/templates/shared/covariates_Sd_example_1/serialized.json"
        )

        inference.plot_all()

    def test_plot_all_with_bootstraps(self):
        """
        Test that the plot_all method works.
        """
        inference = JointInference.from_file(
            "testing/fastdfe/templates/shared/covariates_Sd_example_1/serialized.json"
        )

        inference.bootstrap(20)

        inference.plot_all()

    def test_evaluate_likelihood_same_as_mle_results(self):
        """
        Check that evaluating the loss function at the MLE results
        yields the same likelihood as the one reported.
        """
        # rerun inference to make sure we don't have platform-specific differences
        inference = JointInference.from_file(
            "testing/fastdfe/templates/shared/shared_example_1/serialized.json"
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
            "testing/configs/pendula.pubescens.example_1.example_2.example_3_C_full_anc/config.yaml"
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
            "testing/configs/pendula.pubescens.example_1.example_2.example_3_C_full_anc/config.yaml"
        )

        # define shared parameters
        config.data['shared_params'] = [
            SharedParams(types=['example_1', 'example_2'], params=['b']),
            SharedParams(types=['pendula', 'pubescens'], params=['b'])
        ]

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
            "testing/configs/pendula.pubescens.example_1.example_2.example_3_C_full_anc/config.yaml"
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

    def test_compared_nested_models(self):
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
            "testing/configs/pendula.pubescens.example_1.example_2.example_3_C_full_anc/config.yaml"
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
            "testing/configs/pendula.pubescens.example_1.example_2.example_3_C_full_anc/config.yaml"
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
        spectra = Spectra.from_file("resources/SFS/spectra/pendula_degeneracy_ref_base_tutorial.csv")

        # create inference object
        inf = JointInference(
            sfs_neut=spectra[['neutral.*']].merge_groups(1),
            sfs_sel=spectra[['selected.*']].merge_groups(1),
            covariates=[Covariate(param='S_d', values=dict(A=1, C=2, T=3, G=4))],
            fixed_params={'all': dict(S_b=1, eps=0)},
            parallelize=True,
            do_bootstrap=True,
            n_bootstraps=10
        )

        # run inference
        inf.run()

        inf_no_cov = inf.run_joint_without_covariates(do_bootstrap=True)

        p = inf.perform_lrt_covariates(do_bootstrap=True)

        inf.plot_inferred_parameters()

        # the test should be non-significant as we chose random covariates
        assert p > 0.05

    def test_covariates_strong_correlation(self):
        """
        Check that strong correlation between covariates is detected as significant.
        """
        # parse SFS
        spectra = Spectra.from_file("resources/SFS/spectra/pendula_degeneracy_ref_base_tutorial.csv")

        # create inference object
        inf = JointInference(
            sfs_neut=spectra[['neutral.*']].merge_groups(1),
            sfs_sel=spectra[['selected.*']].merge_groups(1),
            covariates=[Covariate(param='S_d', values=dict(A=-100000, C=-2045, T=-60504, G=-5024))],
            fixed_params={'all': dict(S_b=1, eps=0, p_b=0)},
            parallelize=True,
            do_bootstrap=True,
            n_bootstraps=10,
            n_runs=10
        )

        # run inference
        inf.run()

        inf_no_cov = inf.run_joint_without_covariates(do_bootstrap=True)

        p = inf.perform_lrt_covariates(do_bootstrap=True)

        inf.plot_inferred_parameters()

        # we expect 'c0' to be close to 1
        self.assertAlmostEqual(inf.params_mle['T']['c0'], 1, places=2)

        # the test should be highly significant as we chose good covariates
        assert p < 0.05

    def test_covariates_plot_inferred_parameters(self):
        """
        Check that covariates are plotted correctly.
        """
        # parse SFS
        spectra = Spectra.from_file("resources/SFS/spectra/pendula_degeneracy_ref_base_tutorial.csv")

        # create inference object
        inf = JointInference(
            sfs_neut=spectra[['neutral.*']].merge_groups(1),
            sfs_sel=spectra[['selected.*']].merge_groups(1),
            covariates=[Covariate(param='S_d', values=dict(A=-100000, C=-2045, T=-60504, G=-5024))],
            fixed_params={'all': dict(S_b=1, eps=0, p_b=0)},
            parallelize=True,
            do_bootstrap=True,
            n_runs=10,
            n_bootstraps=10
        )

        inf.plot_inferred_parameters()

    def test_get_cis_params_mle_with_boostraps(self):
        """
        Get the MLE parameters from the cis inference.
        """
        inference = JointInference.from_file(
            "testing/fastdfe/templates/shared/shared_example_1/serialized.json"
        )

        inference.bootstrap(10)

        cis = inference.get_cis_params_mle()

    def test_get_cis_params_mle_no_boostraps(self):
        """
        Get the MLE parameters from the cis inference.
        """
        inference = JointInference.from_file(
            "testing/fastdfe/templates/shared/shared_example_1/serialized.json"
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
            "testing/fastdfe/templates/shared/shared_example_1/serialized.json"
        )

        inference.bootstrap(10)

        res = inference.get_discretized()

    def test_get_discretized_errors_no_bootstraps(self):
        """
        Get the discretized DFE errors.
        """
        inference = JointInference.from_file(
            "testing/fastdfe/templates/shared/shared_example_1/serialized.json"
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
            n_bootstraps=10
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
