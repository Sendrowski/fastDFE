import os

import pytest

from fastdfe.polydfe import PolyDFEResult, PolyDFE
from testing import TestCase


class PolyDFEWrapperTestCase(TestCase):
    """
    Test polyDFE wrapper.
    """

    polydfe_bin = 'resources/polydfe/bin/polyDFE-2.0-macOS-64-bit'
    postprocessing_source = 'resources/polydfe/postprocessing/script.R'

    config = "testing/cache/configs/pendula_C_full_anc/config.yaml"
    serialized = "testing/cache/polydfe/pendula_C_full_anc/serialized.json"

    def test_run_polydfe_from_config(self):
        """
        Run polyDFE from config.
        """
        # run polyDFE
        polydfe = PolyDFE.from_config_file(self.config)
        polydfe.run(
            output_file=f"scratch/test_run_polydfe_from_config.txt",
            binary=self.polydfe_bin,
            postprocessing_source=self.postprocessing_source
        )

        polydfe.to_file("scratch/test_run_polydfe_from_config.json")

    def test_restore_serialized_wrapper(self):
        """
        Serialize polyDFE wrapper and restore.
        """
        # run polyDFE
        polydfe = PolyDFE.from_config_file(self.config)

        polydfe.run(
            output_file="scratch/test_restore_serialized_wrapper.json",
            binary=self.polydfe_bin,
            postprocessing_source=self.postprocessing_source
        )

        # serialize and restore from JSON
        polydfe_restored = polydfe.from_json(polydfe.to_json())

        # compare JSON representation
        self.assertEqual(polydfe.to_json(), polydfe_restored.to_json())

    @pytest.mark.slow
    def test_run_bootstrap_sample(self):
        """
        Serialize polyDFE wrapper and restore.
        """
        # run polyDFE
        polydfe = PolyDFE.from_file(self.serialized)

        config = polydfe.create_bootstrap()

        bootstrap = PolyDFE(config)

        bootstrap.run(
            output_file="scratch/test_run_bootstrap_sample.json",
            binary=self.polydfe_bin,
            postprocessing_source=self.postprocessing_source
        )

    def test_visualize_inference(self):
        """
        Plot everything possible.
        """
        polydfe = PolyDFE.from_file(self.serialized)

        polydfe.plot_all()
