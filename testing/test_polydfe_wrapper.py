import glob
import os
import subprocess

import numpy as np
import pytest

import fastdfe as fd
from fastdfe.polydfe import PolyDFEResult, PolyDFE
from fastdfe.spectrum import parse_polydfe_sfs_config
from testing import TestCase


def find_runnable_polydfe_binary():
    """
    Find a polyDFE binary under resources/polydfe/bin that actually executes on this machine
    (e.g. the natively built arm64 binary on Apple Silicon, where the bundled x86_64 binary needs
    Rosetta). Returns the path, or None if none runs.
    """
    for binary in sorted(glob.glob('resources/polydfe/bin/polyDFE*')):
        try:
            # polyDFE exits non-zero on -h but a runnable binary prints its usage
            out = subprocess.run([binary], capture_output=True, timeout=20)
            if b'polyDFE' in out.stdout + out.stderr:
                return binary
        except OSError:
            continue

    return None


@pytest.mark.slow
class PolyDFEWrapperTestCase(TestCase):
    """
    Test polyDFE wrapper. The whole case is nightly-only (``slow``): every test either runs the
    polyDFE binary live or loads a serialized polyDFE result.
    """

    polydfe_bin = 'resources/polydfe/bin/polyDFE-2.0-macOS-64-bit'
    postprocessing_source = 'resources/polydfe/postprocessing/script.R'

    config = "testing/cache/configs/pendula_C_full_anc/config.yaml"
    # the bare pendula_C_full_anc polyDFE run was never committed; use the bootstrapped variant that
    # is present in the cache (this test only loads + plots a serialized result)
    serialized = "testing/cache/polydfe/pendula_C_full_anc_bootstrapped_100/serialized.json"

    # polyDFE example datasets carrying divergence counts
    divergence_examples = ['example_1', 'example_2', 'example_3']

    def test_divergence_alpha_matches_fastdfe(self):
        """
        Validate fastDFE's divergence-based (McDonald-Kreitman style) alpha against polyDFE's own
        ``estimateAlpha`` run with divergence. polyDFE is run fresh with the divergence counts in
        the SFS, then ``estimateAlpha`` is called with the ``div`` argument; fastDFE infers from the
        same divergence-bearing spectra and reports its MK alpha. They must agree closely.
        """
        binary = find_runnable_polydfe_binary()

        if binary is None or not os.path.exists(self.postprocessing_source):
            pytest.skip('No runnable polyDFE binary / postprocessing script available.')

        for example in self.divergence_examples:
            # full-model config with the divergence-bearing spectra from the polyDFE resource
            config = fd.Config.from_file(f'testing/cache/configs/{example}_C_full_anc/config.yaml')
            config.data['fixed_params']['all']['h'] = 0.5
            config.data['do_bootstrap'] = False
            sp = parse_polydfe_sfs_config(f'resources/polydfe/{example}/spectra/sfs.txt')
            config.data['sfs_neut'] = fd.Spectra.from_spectrum(sp['sfs_neut'])
            config.data['sfs_sel'] = fd.Spectra.from_spectrum(sp['sfs_sel'])
            config.data['n_sites_div_neut'] = {'all': sp['n_sites_div_neut']}
            config.data['n_sites_div_sel'] = {'all': sp['n_sites_div_sel']}

            # run polyDFE with divergence and get its MK alpha
            polydfe = PolyDFE.from_config(config)
            polydfe.run(
                output_file=f'scratch/test_divergence_alpha_{example}.txt',
                binary=binary,
                postprocessing_source=self.postprocessing_source
            )
            alpha_polydfe = polydfe.get_alpha_divergence()

            # fastDFE MK alpha from the same data
            inf = fd.BaseInference.from_config(config)
            inf.run()
            alpha_fastdfe = inf.get_alpha(use_divergence=True)

            assert abs(alpha_polydfe - alpha_fastdfe) < 0.1, \
                f'{example}: polyDFE MK alpha {alpha_polydfe} vs fastDFE {alpha_fastdfe}'

    def _divergence_config(self, example: str, include_divergence: bool) -> 'fd.Config':
        """
        Build a full-model config for a polyDFE example. When ``include_divergence`` is False the
        divergence target size is stripped so that neither polyDFE nor fastDFE uses divergence.
        """
        from fastdfe.spectrum import Spectrum

        config = fd.Config.from_file(f'testing/cache/configs/{example}_C_full_anc/config.yaml')
        config.data['fixed_params']['all']['h'] = 0.5
        config.data['do_bootstrap'] = False
        config.data['include_divergence'] = include_divergence

        sp = parse_polydfe_sfs_config(f'resources/polydfe/{example}/spectra/sfs.txt')

        config.data['sfs_neut'] = fd.Spectra.from_spectrum(sp['sfs_neut'])
        config.data['sfs_sel'] = fd.Spectra.from_spectrum(sp['sfs_sel'])

        # drop the divergence target sizes when divergence is disabled -> the SFS writer omits the
        # divergence columns and fastDFE infers from polymorphism only
        if include_divergence:
            config.data['n_sites_div_neut'] = {'all': sp['n_sites_div_neut']}
            config.data['n_sites_div_sel'] = {'all': sp['n_sites_div_sel']}

        return config

    def test_divergence_dfe_shift_matches_polydfe(self):
        """
        Validate that including divergence reshapes fastDFE's DFE the same way it reshapes
        polyDFE's: the inferred (binned) DFEs must agree between the tools in both modes, and the
        per-bin shift caused by divergence must point in the same direction in both tools.
        """
        binary = find_runnable_polydfe_binary()

        if binary is None or not os.path.exists(self.postprocessing_source):
            pytest.skip('No runnable polyDFE binary / postprocessing script available.')

        def polydfe_dfe(config, tag):
            poly = PolyDFE.from_config(config)
            poly.run(output_file=f'scratch/test_dfe_shift_{tag}.txt', binary=binary,
                     postprocessing_source=self.postprocessing_source)
            return np.asarray(poly.get_discretized()[0])

        def fastdfe_dfe(config):
            inf = fd.BaseInference.from_config(config)
            inf.run()
            return np.asarray(inf.get_discretized()[0])

        for example in self.divergence_examples:
            poly_div = polydfe_dfe(self._divergence_config(example, True), f'{example}_div')
            poly_nodiv = polydfe_dfe(self._divergence_config(example, False), f'{example}_nodiv')
            fast_div = fastdfe_dfe(self._divergence_config(example, True))
            fast_nodiv = fastdfe_dfe(self._divergence_config(example, False))

            # the tools agree on the binned DFE in both modes
            assert np.abs(fast_div - poly_div).sum() < 0.25, f'{example}: div DFE disagree'
            assert np.abs(fast_nodiv - poly_nodiv).sum() < 0.25, f'{example}: nodiv DFE disagree'

            # divergence reshapes both DFEs in the same direction
            shift_fast = fast_div - fast_nodiv
            shift_poly = poly_div - poly_nodiv

            # only assert direction when the shift is non-trivial (avoid amplifying optimizer noise)
            if np.abs(shift_poly).sum() > 0.02:
                cosine = float(shift_fast @ shift_poly /
                               (np.linalg.norm(shift_fast) * np.linalg.norm(shift_poly)))
                sign_agreement = int(np.sum(np.sign(shift_fast) == np.sign(shift_poly)))

                assert cosine > 0.7, f'{example}: divergence shift direction misaligned (cos={cosine})'
                assert sign_agreement >= len(shift_fast) - 1, \
                    f'{example}: divergence shift sign agreement {sign_agreement}/{len(shift_fast)}'

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
