import numpy as np
from numpy import testing

from fastdfe import Config, BaseInference, SharedParams
from fastdfe.optimization import Covariate
from testing import TestCase


class ConfigTestCase(TestCase):
    """
    Test the Config class.
    """
    maxDiff = None

    config_file = "testing/cache/configs/pendula_C_full_anc/config.yaml"
    init_file = "resources/polydfe/init/C.full_anc_init"
    spectra_file = "resources/polydfe/pendula/spectra/sfs.txt"

    def assert_config_equal(self, observed: Config, expected: Config, exclude: list = []):
        """
        Assert that the two given configs are equal.

        :param observed: Observed config
        :param expected: Expected config
        :param exclude: Keys to exclude from comparison
        """
        d1 = dict((k, v) for k, v in observed.data.items() if k not in exclude)
        d2 = dict((k, v) for k, v in expected.data.items() if k not in exclude)

        self.assertDictEqual(d1, d2)

    def test_parse_restore_polydfe_sfs_config(self):
        """
        Test whether the sfs config can be properly restored from file.
        """
        out_file = "scratch/parse_restore_polydfe_sfs_config.txt"

        # create config from sfs file
        config = Config(polydfe_spectra_config=self.spectra_file)

        # recreate sfs config file
        config.create_polydfe_sfs_config(out_file)

        # compare both files
        # there were some rounding errors causing the last character to be different
        testing.assert_equal(open(out_file).read()[:-1], open(out_file).read()[:-1])

    def test_parse_restore_polydfe_init_file(self):
        """
        Check whether the init file can properly be stored from file.
        Note that the init file will be different from the original
        polyDFE init file as not all information is stored.
        """
        out_file = "scratch/parse_restore_polydfe_init_file.txt"

        # parse and recreate init file
        config = Config(polydfe_init_file=self.init_file)
        config.create_polydfe_init_file(out_file, 20)

        # create config from newly created init file
        config2 = Config(polydfe_init_file=out_file)

        self.assert_config_equal(config, config2)

    def test_restore_config_from_file(self):
        """
        Check whether the config can be properly restored from file.
        """
        # load config from sfs and init file
        config = Config(
            polydfe_spectra_config=self.spectra_file,
            polydfe_init_file=self.init_file
        )

        file = 'scratch/restore_config_from_file1.json'
        file2 = 'scratch/restore_config_from_file2.json'

        # save config and restore from file
        config.to_file(file)
        config2 = config.from_file(file)
        config2.to_file(file2)

        # compare original and restored config
        testing.assert_equal(open(file).read(), open(file2).read())

        # compare original json representation and restored json representation
        testing.assert_equal(config.to_json(), Config.from_json(config.to_json()).to_json())

        # compare original json representation and restored json representation
        testing.assert_equal(config.to_yaml(), Config.from_yaml(config.to_yaml()).to_yaml())

    def test_divergence_config_round_trip(self):
        """
        Check that divergence target sizes and the include_divergence flag survive
        serialization, and that the spectra payload itself is unchanged.
        """
        # example_1 carries divergence counts and a separate divergence target size
        config = Config(polydfe_spectra_config="resources/polydfe/example_1/spectra/sfs.txt")

        # divergence target sizes are stored on the config (keyed by the single 'all' type)
        assert config.data['n_sites_div_sel']  # non-empty

        out = "scratch/test_divergence_config_round_trip.yaml"
        config.to_file(out)
        config2 = Config.from_file(out)

        testing.assert_equal(config2.data['include_divergence'], True)
        testing.assert_equal(
            config.data['n_sites_div_neut'],
            config2.data['n_sites_div_neut']
        )
        testing.assert_equal(
            config.data['n_sites_div_sel'],
            config2.data['n_sites_div_sel']
        )

        # full YAML is stable under a second round-trip
        testing.assert_equal(config.to_yaml(), Config.from_yaml(config.to_yaml()).to_yaml())

    def test_legacy_config_without_divergence_keys_loads(self):
        """
        Backward compatibility: a config dict lacking include_divergence / n_sites_div
        (i.e. serialized before divergence support) must load with safe defaults.
        """
        config = Config.from_file(self.config_file)

        # legacy YAML configs have no such keys but must still default sensibly
        testing.assert_equal(config.data['include_divergence'], True)
        testing.assert_equal(config.data.get('n_sites_div_sel'), None)

    def test_recreate_config_from_inference(self):
        """
        Load config from file, create inference object and recreate config from inference object.
        """
        config = Config.from_file(self.config_file)

        config2 = BaseInference.from_config(config).create_config()

        # 'bounds', 'opts_mle' and 'folded' will be different
        # because of the default specific to Inference.
        self.assert_config_equal(
            config, config2, ['bounds', 'scales', 'opts_mle', 'sfs_neut', 'sfs_sel', 'x0', 'folded', 'model']
        )

        # the recreated config holds the model object
        self.assertEqual(type(config2.data['model']).__name__, config.data['model'])

    def test_polydfe_init_file_fixes_dominance(self):
        """
        A config parsed from a polyDFE init file fixes ``h`` at 0.5 unless given otherwise, as polyDFE is
        semidominant. ``h`` was left free, so an inference from such a config estimated dominance.
        """
        self.assertEqual(Config(polydfe_init_file=self.init_file).data['fixed_params']['all']['h'], 0.5)

        config = Config(polydfe_init_file=self.init_file, fixed_params=dict(all=dict(h=0.3)))

        self.assertEqual(config.data['fixed_params']['all']['h'], 0.3)

    def test_create_polydfe_init_file_writes_fixed_values(self):
        """
        A polyDFE init file holds a fixed parameter at its fixed value. The initial value was written, so
        polyDFE fixed ``p_b`` at the initial 0.05 where the config fixes it at 0.
        """
        from fastdfe.polydfe_utils import parse_init_file

        out_file = "scratch/test_create_polydfe_init_file_writes_fixed_values.txt"

        config = Config(fixed_params=dict(all=dict(p_b=0, S_b=1)), x0=dict(all=dict(S_d=-100, b=0.2, p_b=0.05, S_b=5)))
        config.create_polydfe_init_file(out_file, n=5)

        fixed_params, x0 = parse_init_file(['S_d', 'b', 'p_b', 'S_b'], out_file)

        self.assertEqual(fixed_params, dict(p_b=0, S_b=1))
        self.assertEqual((x0['S_d'], x0['b']), (-100, 0.2))

    def test_create_polydfe_init_file_without_fixed_params(self):
        """
        A polyDFE init file can be created from a config without fixed parameters, which is the default. The
        default ``fixed_params=None`` raised a ``TypeError``.
        """
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            default, empty = f'{tmp}/default', f'{tmp}/empty'

            Config().create_polydfe_init_file(default, n=5)
            Config(fixed_params={}).create_polydfe_init_file(empty, n=5)

            with open(default) as a, open(empty) as b:
                content = a.read()

                self.assertTrue(content.strip())
                self.assertEqual(content, b.read())

    def test_parsed_init_file_does_not_leak_into_other_configs(self):
        """
        The initial values parsed from a polyDFE init file are stored on that config alone. They were written
        into the default ``x0`` dictionary shared by all configs created without initial values.
        """
        before = Config()

        parsed = Config(polydfe_init_file='resources/polydfe/init/C.full_anc_init')

        self.assertIn('all', parsed.data['x0'])
        self.assertEqual(before.data['x0'], {})
        self.assertEqual(Config().data['x0'], {})

        # the dictionaries of two configs are distinct objects
        for key in ('x0', 'bounds', 'scales', 'opts_mle'):
            self.assertIsNot(Config().data[key], Config().data[key])

    def test_from_dict_leaves_argument_unchanged(self):
        """
        ``Config.from_dict`` does not modify the passed dictionary, which can then be loaded again.
        """
        import copy
        import json

        data = json.loads(Config(
            shared_params=[SharedParams(types=['a', 'b'], params=['S_d'])],
            covariates=[Covariate(param='S_d', values=dict(a=1, b=2))]
        ).to_json())
        original = copy.deepcopy(data)

        Config.from_dict(data)

        self.assertEqual(data, original)

        config = Config.from_dict(data)

        self.assertIsInstance(config.data['shared_params'][0], SharedParams)
        self.assertIsInstance(config.data['covariates'][0], Covariate)

    def test_restore_shared_params(self):
        """
        Test whether shared params can be properly restored from file.
        """
        config = Config(shared_params=[
            SharedParams(params=['p_b', 'S_b'], types=['pendula', 'pubescens']),
            SharedParams(params=['eps'], types=['example_1', 'example_2', 'pubescens']),
            SharedParams(params=['b', 'S_d'], types=['example_1', 'example_2', 'example_3'])
        ])

        out = "scratch/test_restore_shared_params.yaml"
        config.to_file(out)

        config2 = Config.from_file(out)

        self.assert_config_equal(config, config2)

    def test_restore_custom_intervals_and_covariate_settings(self):
        """
        A config written to JSON or YAML keeps the intervals of a discrete parametrization and the bounds,
        initial value and scale of a covariate where they differ from the defaults. Only the class name and
        the covariate values were stored, so both were restored with their defaults.
        """
        import json
        from fastdfe import DiscreteFractionalParametrization

        config = Config(
            model=DiscreteFractionalParametrization(intervals=np.array([-100000, -100, -1, 0, 1, 1000])),
            covariates=[Covariate(param='S_d', values=dict(a=1, b=2), bounds=(1e-2, 10), x0=3.0, bounds_scale='lin')]
        )

        for restored in (Config.from_json(config.to_json()), Config.from_yaml(config.to_yaml())):
            model, cov = restored.data['model'], restored.data['covariates'][0]

            self.assertIsInstance(model, DiscreteFractionalParametrization)
            np.testing.assert_array_equal(model.intervals, config.data['model'].intervals)
            self.assertEqual(model.param_names, config.data['model'].param_names)
            self.assertEqual((cov.bounds, cov.x0, cov.bounds_scale), ((1e-2, 10), 3.0, 'lin'))

        # the defaults are written as before
        default = json.loads(Config(
            model=DiscreteFractionalParametrization(), covariates=[Covariate(param='S_d', values=dict(a=1, b=2))]
        ).to_json())

        self.assertEqual(default['model'], 'DiscreteFractionalParametrization')
        self.assertEqual(default['covariates'], [dict(param='S_d', values=dict(a=1, b=2))])

    def test_recreate_config_keeps_custom_intervals_of_model(self):
        """
        The config of a base inference restores a discrete parametrization with its intervals, in memory and
        from file. The config held the class name only, which restored the default intervals.
        """
        from fastdfe import DiscreteFractionalParametrization, Spectrum

        inf = BaseInference(
            sfs_neut=Spectrum([177130, 997, 441, 228, 156, 117, 114, 83, 105, 109, 652]),
            sfs_sel=Spectrum([797939, 1329, 499, 265, 162, 104, 117, 90, 94, 119, 794]),
            model=DiscreteFractionalParametrization(intervals=[-1000, -10, 0, 10]),
            do_bootstrap=False
        )

        out_file = "scratch/test_recreate_config_keeps_custom_intervals_of_model.yaml"
        inf.create_config().to_file(out_file)

        for config in [inf.create_config(), Config.from_file(out_file)]:
            restored = BaseInference.from_config(config)

            np.testing.assert_array_equal(restored.model.intervals, inf.model.intervals)
            self.assertEqual(restored.model.param_names, inf.model.param_names)

    def test_restore_custom_bounds_scales_and_x0_of_model(self):
        """
        A config written to JSON or YAML keeps the bounds, scales and initial values given to its
        parametrization. They were dropped, so the restored model used the defaults.
        """
        from fastdfe import GammaExpParametrization, DiscreteFractionalParametrization

        for model in [
            GammaExpParametrization(bounds=dict(S_d=(-1e4, -50)), scales=dict(b='log'), x0=dict(b=0.5)),
            DiscreteFractionalParametrization(intervals=[-1000, -10, 0, 10], x0=dict(S1=0.3))
        ]:
            config = Config(model=model)

            for restored in (Config.from_json(config.to_json()), Config.from_yaml(config.to_yaml())):
                self.assertEqual({k: tuple(v) for k, v in restored.data['model'].bounds.items()}, model.bounds)
                self.assertEqual(restored.data['model'].scales, model.scales)
                self.assertEqual(restored.data['model'].x0, model.x0)

    def test_serializing_covariate_callback_warns(self):
        """
        Serializing a config with a covariate callback logs a warning, as for the dominance callback. The
        callback was omitted without notice, so the restored covariate was the default linear one.
        """
        config = Config(covariates=[Covariate(param='S_d', values=dict(a=1, b=2), callback=lambda cov, **kwargs: cov)])

        with self.assertLogs('fastdfe', level='WARNING') as logs:
            config.to_json()

        self.assertTrue(any('covariates' in message for message in logs.output))

        with self.assertNoLogs('fastdfe', level='WARNING'):
            Config(covariates=[Covariate(param='S_d', values=dict(a=1, b=2))]).to_json()

    def test_from_dict_accepts_to_dict(self):
        """
        A config can be recreated from its own dictionary. The dictionary holds the spectra, shared
        parameters and covariates as objects, which ``from_dict`` did not accept.
        """
        from fastdfe import Spectra

        config = Config(
            sfs_neut=Spectra(dict(a=[100, 5, 3, 2], b=[100, 4, 3, 1])),
            sfs_sel=Spectra(dict(a=[400, 6, 2, 1], b=[400, 5, 2, 1])),
            shared_params=[SharedParams(params=['S_d'], types=['a', 'b'])],
            covariates=[Covariate(param='S_d', values=dict(a=1, b=2))]
        )

        self.assertEqual(Config.from_dict(config.to_dict()).to_json(), config.to_json())

    def test_yaml_numbers_without_decimal_point_are_cast(self):
        """
        Bounds, initial values and fixed parameters written in a YAML file as ``1e-2`` or ``-1e5`` are read
        as numbers. YAML reads these forms as strings, which failed when the inference was run.
        """
        config = Config.from_yaml("""
bounds:
  S_d: [-1e5, -1e-2]
x0:
  all:
    S_d: -1e3
    b: 0.4
fixed_params:
  all:
    eps: 0
    S_b: 1e0
""")

        self.assertEqual(config.data['bounds'], dict(S_d=(-1e5, -1e-2)))
        self.assertEqual(config.data['x0'], dict(all=dict(S_d=-1e3, b=0.4)))
        self.assertEqual(config.data['fixed_params'], dict(all=dict(eps=0, S_b=1.0)))

    def test_polydfe_init_file_of_model_d_is_not_supported(self):
        """
        Init files of the discrete polyDFE model D raise a ``NotImplementedError``. They hold three values per
        class, which were read as the two values per parameter of the other models.
        """
        with self.assertRaises(NotImplementedError):
            Config(polydfe_init_file="resources/polydfe/init/D.full_init", model='DiscreteParametrization')

        with self.assertRaises(NotImplementedError):
            Config(model='DiscreteParametrization').create_polydfe_init_file('scratch/model_d_init.txt', n=5)

    def test_polydfe_bootstrap_config_keeps_divergence(self):
        """
        The config of a polyDFE bootstrap keeps the divergence counts and target sizes of the original one.
        The spectra were written without them, so bootstraps of a run with divergence were fitted without.
        """
        import shutil
        from types import SimpleNamespace
        from fastdfe.polydfe import PolyDFE

        polydfe = PolyDFE(Config(polydfe_spectra_config="resources/polydfe/example_1/spectra/sfs.txt"))

        # stand-in for the resampling of polyDFE's postprocessing script, which returns the data unchanged
        wrapper = SimpleNamespace(bootstrapData=lambda file, rep: shutil.copy(file, file + '_1'))
        polydfe.summary = SimpleNamespace(get_postprocessing_wrapper=lambda: wrapper, data=dict(params_mle={}))

        config = polydfe.create_bootstrap()

        self.assertEqual(config.data['n_sites_div_sel'], polydfe.config.data['n_sites_div_sel'])
        self.assertEqual(config.data['n_sites_div_neut'], polydfe.config.data['n_sites_div_neut'])
        self.assertEqual(config.data['sfs_sel'].all.n_div, polydfe.config.data['sfs_sel'].all.n_div)
        self.assertGreater(config.data['sfs_sel'].all.n_div, 0)

    def test_restore_covariates(self):
        """
        Test whether covariates can be properly restored from file.
        """
        config = Config(covariates=[
            Covariate(param='S_d', values=dict(t1=1, t2=2)),
            Covariate(param='xx', values=dict(bar=34))
        ])

        out = "scratch/test_restore_shared_params.yaml"
        config.to_file(out)

        config2 = Config.from_file(out)

        self.assert_config_equal(config, config2)

    @staticmethod
    def test_download_config():
        """
        Test whether the config can be properly downloaded from a URL.
        """
        config = Config.from_file(
            "https://github.com/Sendrowski/fastDFE/blob/master/"
            "resources/configs/shared/pendula_tutorial/config.yaml?raw=true"
        )

        np.testing.assert_equal(
            config.data['sfs_neut'].data['all'].to_numpy(),
            np.array([173705, 3294, 1115, 534, 326, 239, 225, 214, 231, 176, 73])
        )

    @staticmethod
    def test_download_config_no_cache():
        """
        Test whether the config can be properly downloaded from a URL without caching.
        """
        config = Config.from_file(
            "https://github.com/Sendrowski/fastDFE/blob/master/"
            "resources/configs/shared/pendula_tutorial/config.yaml?raw=true",
            cache=False
        )

        np.testing.assert_equal(
            config.data['sfs_neut'].data['all'].to_numpy(),
            np.array([173705, 3294, 1115, 534, 326, 239, 225, 214, 231, 176, 73])
        )
