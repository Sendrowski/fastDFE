from testing import prioritize_installed_packages

prioritize_installed_packages()

from unittest import TestCase

from numpy import testing

from fastdfe import Config, BaseInference, SharedParams
from fastdfe.optimization import Covariate


class ConfigTestCase(TestCase):
    maxDiff = None

    config_file = "testing/configs/pendula_C_full_anc/config.yaml"
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

    def test_recreate_config_from_inference(self):
        """
        Load config from file, create inference object and recreate config from inference object.
        """
        config = Config.from_file(self.config_file)

        config2 = BaseInference.from_config(config).create_config()

        # 'bounds' and 'opts_mle' will be different
        # because of the default specific to Inference.
        self.assert_config_equal(config, config2, ['bounds', 'opts_mle', 'sfs_neut', 'sfs_sel', 'x0'])

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
