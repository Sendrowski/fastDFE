import os
from unittest import TestCase

import pytest

import fastdfe


class PolyDFEWrapperTestCase(TestCase):
    polydfe_bin = 'resources/polydfe/bin/polyDFE-2.0-macOS-64-bit'
    postprocessing_source = os.getcwd() + '/' + fastdfe.PolyDFEResult.default_postprocessing_source

    config = "testing/configs/pendula_C_full_anc/config.yaml"
    serialized = "testing/polydfe/pendula_C_full_anc/serialized.json"

    def test_run_polydfe_from_config(self):
        """
        Run polyDFE from config.
        """
        # run polyDFE
        polydfe = fastdfe.PolyDFE.from_config_file(self.config)
        polydfe.run(f"scratch/test_run_polydfe_from_config.txt",
                    bin=self.polydfe_bin, postprocessing_source=self.postprocessing_source)

        polydfe.to_file("scratch/test_run_polydfe_from_config.json")

    def test_restore_serialized_wrapper(self):
        """
        Serialize polyDFE wrapper and restore.
        """
        # run polyDFE
        polydfe = fastdfe.PolyDFE.from_config_file(self.config)

        polydfe.run(
            output_file="scratch/test_restore_serialized_wrapper.json",
            bin=self.polydfe_bin,
            postprocessing_source=self.postprocessing_source
        )

        # serialize and restore from JSON
        polydfe_restored = polydfe.from_json(polydfe.to_json())

        # compare JSON representation
        self.assertEqual(polydfe.to_json(), polydfe_restored.to_json())

    @pytest.mark.skip(reason="takes too long")
    def test_run_bootstrap_sample(self):
        """
        Serialize polyDFE wrapper and restore.
        """
        # run polyDFE
        polydfe = fastdfe.PolyDFE.from_file(self.serialized)

        config = polydfe.create_bootstrap()

        bootstrap = fastdfe.PolyDFE(config)

        bootstrap.run(
            output_file="scratch/test_run_bootstrap_sample.json",
            bin=self.polydfe_bin,
            postprocessing_source=self.postprocessing_source
        )

    def test_visualize_inference(self):
        """
        Plot everything possible.
        """
        polydfe = fastdfe.PolyDFE.from_file(self.serialized)

        polydfe.plot_all()
