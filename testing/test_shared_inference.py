import pytest

from fastdfe import SharedInference, Config, SharedParams, Covariate
from fastdfe.optimization import flatten_dict
from testing.test_base_inference import AbstractInferenceTestCase


class SharedInferenceTestCase(AbstractInferenceTestCase):
    config_file = "testing/configs/pendula.pubescens.example_1.example_2.example_3_C_full_anc/config.yaml"

    show_plots = False

    maxDiff = None

    def test_run_and_restore_from_file(self):
        """
        Test that shared inference can be restored from file.
        """
        inference = SharedInference.from_config_file(
            "resources/configs/shared/covariates_Sd_fixed_params/config.yaml"
        )

        inference.run()

        out = "scratch/test_restore_from_file.json"

        inference.to_file(out)

        inference2 = SharedInference.from_file(out)

        self.assertEqualInference(inference, inference2)

    def test_bootstrap_shared_inference(self):
        """
        Test that the bootstrap method works.
        """
        inference = SharedInference.from_file(
            "testing/fastdfe/templates/shared/shared_example_1/serialized.json"
        )

        inference.bootstrap(2)

    def test_perform_lrt_shared(self):
        """
        Test that the perform_lrt_shared method works.
        """
        inference = SharedInference.from_file(
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
        inference = SharedInference.from_file(
            "testing/fastdfe/templates/shared/shared_example_1/serialized.json"
        )

        with pytest.raises(ValueError):
            inference.perform_lrt_covariates()

    def test_perform_lrt_covariates(self):
        """
        Test that the perform_lrt_covariates method works.
        """
        # inf = SharedInference.from_config_file("resources/configs/shared/covariates_Sd_example_1/config.yaml")
        # inf.run()

        inf = SharedInference.from_file(
            "testing/fastdfe/templates/shared/covariates_Sd_example_1/serialized.json"
        )

        assert inf.perform_lrt_covariates() < 1

    def test_plot_covariates(self):
        """
        Test that the plot_covariates method works.
        """
        inference = SharedInference.from_file(
            "testing/fastdfe/templates/shared/covariates_Sd_example_1/serialized.json"
        )

        inference.plot_covariates()

    def test_plot_all(self):
        """
        Test that the plot_all method works.
        """
        inference = SharedInference.from_file(
            "testing/fastdfe/templates/shared/covariates_Sd_example_1/serialized.json"
        )

        inference.plot_all()

    def test_evaluate_likelihood_same_as_mle_results(self):
        """
        Check that evaluating the loss function at the MLE results
        yields the same likelihood as the one reported.
        """
        # unserialize
        inference = SharedInference.from_file(
            "testing/fastdfe/templates/shared/shared_example_1/serialized.json"
        )

        # make sure joint likelihood is the same
        assert inference.evaluate_likelihood(inference.params_mle) == inference.likelihood

        for t in inference.types:
            # make sure marginal likelihoods are the same
            assert inference.marginal_inferences[t].evaluate_likelihood(
                dict(all=inference.marginal_inferences[t].params_mle)) == inference.marginal_inferences[t].likelihood

            # make sure joint likelihoods are not the same as when evaluated individually
            assert inference.joint_inferences[t].evaluate_likelihood(
                dict(all=inference.joint_inferences[t].params_mle)) != inference.joint_inferences[t].likelihood

            # make sure all joint likelihoods coincide with likelihood of shared inference
            inference.joint_inferences[t].likelihood = inference.likelihood

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

        inf = SharedInference.from_config(config)

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
        assert params_mle['pendula']['S_b'] == config.data['fixed_params']['pendula']['S_b']
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

        inf = SharedInference.from_config(config)

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
            SharedInference.from_config(config)

    def test_compared_nested_models(self):
        """
        Test nested model comparison.
        """
        inf = SharedInference.from_config_file(
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
        inf = SharedInference.from_config_file(
            "resources/configs/shared/covariates_Sd_fixed_params/config.yaml"
        )

        assert inf.get_n_optimized() == 11

        inf = SharedInference.from_config_file(
            "resources/configs/shared/covariates_Sd_example_1/config.yaml"
        )

        assert inf.get_n_optimized() == 18

        inf = SharedInference.from_config_file(
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
            SharedInference.from_config(config)

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
            SharedInference.from_config(config)
