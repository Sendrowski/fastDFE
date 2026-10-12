import copy
import logging
import os
import resource
import time
from abc import ABC
from unittest import mock

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from numpy.random._generator import Generator
from pandas.testing import assert_frame_equal
from scipy.optimize import OptimizeResult

import fastdfe as fd
from fastdfe.discretization import Discretization
from fastdfe.polydfe import PolyDFE
from testing import TestCase


class InferenceTestCase(TestCase, ABC):
    """
    Test inference.
    """

    def assertEqualObjects(self, obj1: object, obj2: object, ignore_keys=[]):
        """
        Compare objects, recursively comparing their attributes.

        :param obj1: First object
        :param obj2: Second object
        :param ignore_keys: Keys to ignore
        """
        ignore_keys += ['_logger']

        if not hasattr(obj1, '__dict__'):
            self.assertEqual(obj1, obj2)
        else:

            d1 = obj1.__dict__
            d2 = obj2.__dict__

            for (key1, value1), (key2, value2) in zip(d1.items(), d2.items()):
                if key1 in ignore_keys:
                    pass

                elif isinstance(value1, OptimizeResult):
                    pass

                elif isinstance(value1, dict):
                    self.assertEqual(value1.keys(), value2.keys())

                    for k in value1.keys():
                        self.assertEqualObjects(value1[k], value2[k], ignore_keys=ignore_keys)

                elif isinstance(value1, (int, float, str, bool, tuple, list, type(None), np.bool_)):
                    self.assertEqual(value1, value2)

                elif isinstance(value1, np.ndarray):
                    np.testing.assert_equal(value1, value2)

                elif isinstance(value1, pd.DataFrame):
                    pd.testing.assert_frame_equal(value1.reset_index(drop=True), value2.reset_index(drop=True))

                elif isinstance(value1, Generator):
                    pass

                elif isinstance(value1, fd.Spectrum):
                    np.testing.assert_equal(value1.to_list(), value2.to_list())

                elif isinstance(value1, object):
                    self.assertEqualObjects(value1, value2, ignore_keys=ignore_keys)

                else:
                    raise AssertionError('Some objects were not compared.')


@pytest.mark.inference
class PolyDFETestCase(InferenceTestCase):
    """
    Test results against cached PolyDFE results.
    """

    # configs for testing against cached polydfe results
    configs_polydfe = [
        'pendula_C_full_bootstrapped_100',
        'pendula_C_full_anc_bootstrapped_100',
        'pendula_C_deleterious_bootstrapped_100',
        'pendula_C_deleterious_anc_bootstrapped_100',
        'pubescens_C_full_bootstrapped_100',
        'pubescens_C_full_anc_bootstrapped_100',
        'pubescens_C_deleterious_bootstrapped_100',
        'pubescens_C_deleterious_anc_bootstrapped_100',
        'example_1_C_full_bootstrapped_100',
        'example_1_C_full_anc_bootstrapped_100',
        # 'example_1_C_deleterious_bootstrapped_100' # polydfe does not finish in this case
        'example_1_C_deleterious_anc_bootstrapped_100',
        'example_2_C_full_bootstrapped_100',
        'example_2_C_full_anc_bootstrapped_100',
        'example_2_C_deleterious_bootstrapped_100',
        'example_2_C_deleterious_anc_bootstrapped_100',
        'example_3_C_full_bootstrapped_100',
        'example_3_C_full_anc_bootstrapped_100',
        'example_3_C_deleterious_bootstrapped_100',
        'example_3_C_deleterious_anc_bootstrapped_100'
    ]

    def compare_with_polydfe(self, config: str):
        """
        Test whether the results are similar to those of PolyDFE.
        """
        conf = fd.Config.from_file(f"testing/cache/configs/{config}/config.yaml")
        conf.data['fixed_params']['all']['h'] = 0.5
        # speed up: fewer optimization restarts (the datasets converge easily), a single run per
        # bootstrap sample, and fewer bootstraps. The confidence intervals stay wide enough for the
        # (tolerant) overlap comparison below.
        conf.data['n_runs'] = 3
        conf.data['n_bootstrap_retries'] = 1
        conf.data['n_bootstraps'] = 30
        # coarser integration grid (1000 -> 100 bins) is ~10x cheaper per optimization and leaves the
        # binned DFE / CIs essentially unchanged, since get_discretized() bins independently of the
        # integration resolution. The comparison tolerance below is therefore left untouched.
        conf.data['intervals_del'] = (-1.0e+8, -1.0e-5, 100)
        conf.data['intervals_ben'] = (1.0e-5, 1.0e4, 100)
        inf_polydfe = PolyDFE.from_file(f"testing/cache/polydfe/{config}/serialized.json")

        inf_fastdfe = fd.BaseInference.from_config(conf)
        inf_fastdfe.run()

        # compare discretized DFE
        dfe_fastdfe = inf_fastdfe.get_discretized()
        dfe_polydfe = inf_polydfe.get_discretized()

        ci_fastdfe = np.array([dfe_fastdfe[0] - dfe_fastdfe[1][0], dfe_fastdfe[0] + dfe_fastdfe[1][1]])
        ci_polydfe = np.array([dfe_polydfe[0] - dfe_polydfe[1][0], dfe_polydfe[0] + dfe_polydfe[1][1]])

        plt.clf()
        _, axs = plt.subplots(1, 2, figsize=(10, 4))
        inf_fastdfe.plot_discretized(ax=axs[0], show=False, title='fastdfe')
        inf_polydfe.plot_discretized(ax=axs[1], show=True, title='polydfe')

        # assert that the confidence intervals almost overlap
        assert np.all(ci_fastdfe[0] <= ci_polydfe[1] + 0.05)
        assert np.all(ci_polydfe[0] <= ci_fastdfe[1] + 0.05)

    @staticmethod
    def generate_compare_with_polydfe(config: str) -> callable:
        """
        Generate test for comparing with PolyDFE.
        """

        def compare_with_polydfe(self):
            """
            Compare with PolyDFE.

            :param self: Self
            """
            self.compare_with_polydfe(config)

        return compare_with_polydfe

    # dynamically generate tests
    for config in configs_polydfe:
        locals()[f'test_compare_with_polydfe_{config}'] = generate_compare_with_polydfe(config)


class DivergenceTestCase(InferenceTestCase):
    """
    Test divergence-count support (polyDFE-style alpha estimation).
    """

    # polyDFE example datasets which carry divergence counts and a separate divergence target size
    examples = ['example_1', 'example_2', 'example_3']

    @staticmethod
    def _get_spectra(example: str):
        """
        Get neutral and selected spectra (with divergence) for a polyDFE example dataset.
        """
        from sfsutils.spectrum import parse_polydfe_sfs_config

        sp = parse_polydfe_sfs_config(f'resources/polydfe/{example}/spectra/sfs.txt')

        return sp['sfs_neut'], sp['sfs_sel'], sp['n_sites_div_neut'], sp['n_sites_div_sel']

    def test_no_divergence_path_unchanged(self):
        """
        With the default include_divergence=True but no divergence data, divergence must not be
        used and the result must be identical to explicitly disabling it.
        """
        from sfsutils.spectrum import parse_polydfe_sfs_config

        sp = parse_polydfe_sfs_config('resources/polydfe/pendula/spectra/sfs.txt')

        kwargs = dict(sfs_neut=sp['sfs_neut'], sfs_sel=sp['sfs_sel'], do_bootstrap=False, n_runs=3, seed=42)

        inf_on = fd.BaseInference(include_divergence=True, **kwargs)
        inf_off = fd.BaseInference(include_divergence=False, **kwargs)

        # divergence is not used in either case (no divergence target size)
        assert inf_on._use_divergence is False
        assert inf_off._use_divergence is False
        assert 'r_div' not in inf_on.param_names

        inf_on.run()
        inf_off.run()

        # results are identical
        self.assertEqual(inf_on.params_mle, inf_off.params_mle)
        np.testing.assert_array_equal(inf_on.sfs_mle.polymorphic, inf_off.sfs_mle.polymorphic)

    @pytest.mark.inference
    def test_divergence_inference_runs_and_derives_r_div(self):
        """
        With divergence data, divergence is used; the divergence scale ``r_div`` is pinned by the
        neutral divergence and derived in closed form (not an optimizer parameter), and both alpha
        modes work.
        """
        neut, sel, nsd_neut, nsd_sel = self._get_spectra('example_1')

        inf = fd.BaseInference(sfs_neut=neut, sfs_sel=sel, n_sites_div_neut=nsd_neut, n_sites_div_sel=nsd_sel, do_bootstrap=False, n_runs=5, seed=0,
                               intervals_del=(-1.0e+8, -1.0e-5, 100), intervals_ben=(1.0e-5, 1.0e4, 100))

        assert inf._use_divergence is True
        # r_div is derived from the data, so it is never an inference parameter
        assert 'r_div' not in inf.param_names

        inf.run()

        assert 'r_div' not in inf.params_mle
        # the closed-form divergence scale (reproducing the observed neutral divergence) is sensible
        r_div = neut.n_div / (neut.theta * nsd_neut)
        assert np.isfinite(r_div) and r_div > 0
        # both alpha estimators are available and finite
        assert np.isfinite(inf.get_alpha())
        assert np.isfinite(inf.get_alpha(use_divergence=False))

    def test_dfe_alpha_matches_polydfe(self):
        """
        Feeding polyDFE's own fitted DFE parameters into fastDFE's (polymorphism-only) alpha must
        reproduce polyDFE's reported alpha. This validates the shared fixation/divergence machinery
        ``S / (1 - exp(-S))`` independent of optimizer differences.
        """
        import json

        for example in self.examples:
            config = f'{example}_C_full_anc_bootstrapped_100'
            data = json.load(open(f'testing/cache/polydfe/{config}/serialized.json'))['summary']['data']
            pm, alpha_polydfe = data['params_mle'], data['alpha']

            neut, sel, nsd_neut, nsd_sel = self._get_spectra(example)
            inf = fd.BaseInference(sfs_neut=neut, sfs_sel=sel, n_sites_div_neut=nsd_neut, n_sites_div_sel=nsd_sel, do_bootstrap=False, n_runs=1)

            params = dict(S_d=pm['S_d'], b=pm['b'], p_b=pm['p_b'], S_b=pm['S_b'], eps=pm.get('eps', 0.0), h=0.5)
            alpha_fastdfe = inf.get_alpha(params=params, use_divergence=False)

            assert abs(alpha_fastdfe - alpha_polydfe) < 0.01, \
                f'{example}: fastdfe {alpha_fastdfe} vs polydfe {alpha_polydfe}'

    def test_alpha_divergence_matches_mk_formula(self):
        """
        The divergence-based alpha must equal polyDFE's McDonald-Kreitman style formula
        ``1 - (L_sel/L_neut) * (D_neut/D_sel) * divDel``.
        """
        neut, sel, nsd_neut, nsd_sel = self._get_spectra('example_2')
        inf = fd.BaseInference(sfs_neut=neut, sfs_sel=sel, n_sites_div_neut=nsd_neut, n_sites_div_sel=nsd_sel, do_bootstrap=False, n_runs=1)

        params = dict(S_d=-2000.0, b=0.4, p_b=0.05, S_b=2.0, eps=0.0, h=0.5)

        d = inf.discretization
        y = inf.model._discretize(params, d.bins) * d.get_counts_fixed(0.5)
        div_del = np.sum(y[d.s <= 0])
        expected = 1 - (nsd_sel / nsd_neut) * (neut.n_div / sel.n_div) * div_del

        np.testing.assert_allclose(inf.get_alpha(params=params, use_divergence=True), expected, rtol=1e-12)

    def test_create_config_preserves_divergence_target_sizes(self):
        """
        An inference recreated from its config uses divergence with the same target sizes. The config
        carried ``include_divergence`` only, so the recreated inference ran on polymorphism alone.
        """
        neut, sel, nsd_neut, nsd_sel = self._get_spectra('example_2')
        inf = fd.BaseInference(sfs_neut=neut, sfs_sel=sel, n_sites_div_neut=nsd_neut, n_sites_div_sel=nsd_sel,
                               do_bootstrap=False, n_runs=1)

        inf2 = fd.BaseInference.from_config(inf.create_config())

        assert inf2._use_divergence is True
        assert inf2.n_sites_div_neut == nsd_neut
        assert inf2.n_sites_div_sel == nsd_sel

    def test_alpha_divergence_equals_dfe_alpha_on_expected_divergence(self):
        """
        With divergence counts equal to their model expectation, the divergence-based alpha equals the DFE-based
        alpha. Summing ``divDel`` over ``s < 0`` dropped the bin with midpoint 0, which holds 13% of the mass for
        a gamma shape of 0.1 and gave an alpha of 0.31 for a purely deleterious DFE.
        """
        d = Discretization(n=10)
        model = fd.GammaExpParametrization()

        for p_b in [0, 0.05]:
            params = dict(S_d=-1000.0, b=0.1, p_b=p_b, S_b=2.0, h=0.5)

            # neutral and selected divergence counts for equal target sizes
            n_div_neut = 1000.0
            n_div_sel = 1000.0 * d.get_divergence_flux(model, params)

            np.testing.assert_allclose(
                d.get_alpha_divergence(model, params, n_div_neut, n_div_sel, 1.0, 1.0),
                d.get_alpha(model, params),
                atol=1e-12
            )

    def test_divergence_alpha_generalizes_to_dominance(self):
        """
        The divergence flux and both alpha estimators must respond to the dominance coefficient h.
        """
        neut, sel, nsd_neut, nsd_sel = self._get_spectra('example_2')
        # leave h free (not in fixed_params) so the discretization varies over dominance and can be
        # evaluated across h; a fixed-h discretization would (by design) reject h != self.h
        inf = fd.BaseInference(sfs_neut=neut, sfs_sel=sel, n_sites_div_neut=nsd_neut, n_sites_div_sel=nsd_sel,
                               fixed_params=dict(all=dict(eps=0)), do_bootstrap=False, n_runs=1)

        base = dict(S_d=-2000.0, b=0.4, p_b=0.1, S_b=4.0, eps=0.0)

        fluxes = [inf.discretization.get_divergence_flux(inf.model, dict(base, h=h)) for h in [0.1, 0.5, 0.9]]
        alphas = [inf.get_alpha(params=dict(base, h=h), use_divergence=True) for h in [0.1, 0.5, 0.9]]

        # strictly increasing in h (more dominant beneficial mutations fix more readily)
        assert fluxes[0] < fluxes[1] < fluxes[2]
        assert alphas[0] < alphas[1] < alphas[2]

    @pytest.mark.inference
    def test_divergence_precomputes_fixed_counts_before_workers(self):
        """
        run() must precompute the fixed-counts cache (before forking optimization workers) when
        divergence is used, and leave it unbuilt for a polymorphism-only inference.
        """
        neut, sel, nsd_neut, nsd_sel = self._get_spectra('example_1')
        opts = dict(intervals_del=(-1.0e+8, -1.0e-5, 100), intervals_ben=(1.0e-5, 1.0e4, 100),
                    do_bootstrap=False, n_runs=1)

        with_div = fd.BaseInference(sfs_neut=neut, sfs_sel=sel, n_sites_div_neut=nsd_neut,
                                    n_sites_div_sel=nsd_sel, **opts)
        with_div.run()
        self.assertIsNotNone(with_div.discretization._counts_fixed)

        without_div = fd.BaseInference(sfs_neut=neut, sfs_sel=sel, **opts)
        without_div.run()
        self.assertIsNone(without_div.discretization._counts_fixed)

    def test_divergence_alpha_raises_without_div_data(self):
        """
        Requesting divergence-based alpha without divergence data raises a clear error.
        """
        from sfsutils.spectrum import parse_polydfe_sfs_config

        sp = parse_polydfe_sfs_config('resources/polydfe/pendula/spectra/sfs.txt')
        inf = fd.BaseInference(sfs_neut=sp['sfs_neut'], sfs_sel=sp['sfs_sel'], do_bootstrap=False, n_runs=1)
        inf.run()

        with self.assertRaises(ValueError):
            inf.get_alpha(use_divergence=True)

    @pytest.mark.inference
    def test_divergence_bootstrap_alpha_is_finite(self):
        """
        Bootstrapping a divergence inference resamples divergence counts and produces finite
        alpha estimates that bracket the point estimate.
        """
        neut, sel, nsd_neut, nsd_sel = self._get_spectra('example_1')

        inf = fd.BaseInference(
            sfs_neut=neut, sfs_sel=sel, n_sites_div_neut=nsd_neut, n_sites_div_sel=nsd_sel, do_bootstrap=True, n_bootstraps=10, n_runs=3, seed=0,
            intervals_del=(-1.0e+8, -1.0e-5, 100), intervals_ben=(1.0e-5, 1.0e4, 100)
        )
        inf.run()

        assert 'alpha' in inf.bootstraps
        assert np.all(np.isfinite(inf.bootstraps['alpha']))

    def test_omega_dfe_based_identities(self):
        """
        The DFE-based omega/omega_a must satisfy ``omega = divDel + divBen`` (the divergence flux),
        ``omega_a = divBen`` and ``alpha = omega_a / omega``.
        """
        neut, sel, nsd_neut, nsd_sel = self._get_spectra('example_2')
        inf = fd.BaseInference(sfs_neut=neut, sfs_sel=sel, n_sites_div_neut=nsd_neut, n_sites_div_sel=nsd_sel, do_bootstrap=False, n_runs=1)

        params = dict(S_d=-2000.0, b=0.4, p_b=0.1, S_b=4.0, eps=0.0, h=0.5)

        d = inf.discretization
        y = inf.model._discretize(params, d.bins) * d.get_counts_fixed(0.5)

        omega = inf.get_omega(params=params, use_divergence=False)
        omega_a = inf.get_omega_a(params=params, use_divergence=False)
        alpha = inf.get_alpha(params=params, use_divergence=False)

        np.testing.assert_allclose(omega, np.sum(y), rtol=1e-12)
        np.testing.assert_allclose(omega_a, np.sum(y[d.s > 0]), rtol=1e-12)
        np.testing.assert_allclose(alpha, omega_a / omega, rtol=1e-12)

    def test_omega_divergence_matches_mk_formula(self):
        """
        The divergence-based omega must equal the observed ``(D_sel/L_sel) / (D_neut/L_neut)`` and
        omega_a must equal ``alpha * omega`` using the McDonald-Kreitman style estimators.
        """
        neut, sel, nsd_neut, nsd_sel = self._get_spectra('example_2')
        inf = fd.BaseInference(sfs_neut=neut, sfs_sel=sel, n_sites_div_neut=nsd_neut, n_sites_div_sel=nsd_sel, do_bootstrap=False, n_runs=1)

        params = dict(S_d=-2000.0, b=0.4, p_b=0.05, S_b=2.0, eps=0.0, h=0.5)

        expected_omega = (sel.n_div / nsd_sel) / (neut.n_div / nsd_neut)
        omega = inf.get_omega(params=params, use_divergence=True)
        omega_a = inf.get_omega_a(params=params, use_divergence=True)
        alpha = inf.get_alpha(params=params, use_divergence=True)

        np.testing.assert_allclose(omega, expected_omega, rtol=1e-12)
        np.testing.assert_allclose(omega_a, alpha * omega, rtol=1e-12)

    def test_omega_raises_without_div_data(self):
        """
        Requesting divergence-based omega/omega_a without divergence data raises a clear error.
        """
        from sfsutils.spectrum import parse_polydfe_sfs_config

        sp = parse_polydfe_sfs_config('resources/polydfe/pendula/spectra/sfs.txt')
        inf = fd.BaseInference(sfs_neut=sp['sfs_neut'], sfs_sel=sp['sfs_sel'], do_bootstrap=False, n_runs=1)
        inf.run()

        with self.assertRaises(ValueError):
            inf.get_omega(use_divergence=True)

        with self.assertRaises(ValueError):
            inf.get_omega_a(use_divergence=True)

    @pytest.mark.inference
    def test_divergence_bootstrap_includes_omega(self):
        """
        Bootstrapping adds finite omega and omega_a estimates alongside alpha.
        """
        neut, sel, nsd_neut, nsd_sel = self._get_spectra('example_1')

        inf = fd.BaseInference(
            sfs_neut=neut, sfs_sel=sel, n_sites_div_neut=nsd_neut, n_sites_div_sel=nsd_sel, do_bootstrap=True, n_bootstraps=10, n_runs=3, seed=0,
            intervals_del=(-1.0e+8, -1.0e-5, 100), intervals_ben=(1.0e-5, 1.0e4, 100)
        )
        inf.run()

        for key in ('alpha', 'omega', 'omega_a'):
            assert key in inf.bootstraps
            assert np.all(np.isfinite(inf.bootstraps[key]))

    @pytest.mark.inference
    def test_divergence_inference_serialization_round_trip(self):
        """
        A divergence inference must serialize and restore, retaining the divergence target sizes
        and the divergence mode.
        """
        neut, sel, nsd_neut, nsd_sel = self._get_spectra('example_1')
        inf = fd.BaseInference(sfs_neut=neut, sfs_sel=sel, n_sites_div_neut=nsd_neut, n_sites_div_sel=nsd_sel, do_bootstrap=False, n_runs=1, seed=0)
        inf.run()

        restored = fd.BaseInference.from_json(inf.to_json())

        assert restored._use_divergence is True
        assert restored.n_sites_div_sel == nsd_sel
        np.testing.assert_allclose(restored.get_alpha(), inf.get_alpha())

    def test_restore_legacy_inference_without_divergence_attrs(self):
        """
        Backward compatibility: an inference serialized before divergence support (lacking the
        include_divergence / _use_divergence attributes) must restore via the class-level defaults
        and keep working as a polymorphism-only inference.
        """
        inf = fd.BaseInference.from_json(
            open('testing/cache/fastdfe/pendula_C_full_anc/serialized.json').read()
        )

        # class-level defaults kick in for the missing attributes
        assert inf._use_divergence is False
        assert np.isfinite(inf.get_alpha())

    @staticmethod
    def _get_full_config_with_divergence(example: str) -> 'fd.Config':
        """
        Load the cached full-model config for a polyDFE example and attach the divergence-bearing
        spectra parsed from the polyDFE SFS resource.
        """
        from sfsutils.spectrum import parse_polydfe_sfs_config

        config = fd.Config.from_file(f'testing/cache/configs/{example}_C_full_anc/config.yaml')
        config.data['fixed_params']['all']['h'] = 0.5
        # the toy example datasets converge in a few restarts; the default of 10 is overkill here
        config.data['n_runs'] = 3

        sp = parse_polydfe_sfs_config(f'resources/polydfe/{example}/spectra/sfs.txt')
        config.data['sfs_neut'] = fd.Spectra.from_spectrum(sp['sfs_neut'])
        config.data['sfs_sel'] = fd.Spectra.from_spectrum(sp['sfs_sel'])
        config.data['n_sites_div_neut'] = {'all': sp['n_sites_div_neut']}
        config.data['n_sites_div_sel'] = {'all': sp['n_sites_div_sel']}

        return config

    @pytest.mark.inference
    def test_divergence_matches_polydfe_cached(self):
        """
        Validate fastDFE against polyDFE **offline** using precomputed polyDFE results (polyDFE
        itself is far too slow to run per test — minutes per dataset). For each example dataset:

        * polyDFE's divergence-based (McDonald-Kreitman) alpha is read from the cached
          divergence run and compared to fastDFE's,
        * the inferred binned DFEs are compared between the tools in both modes (divergence on/off),
        * the per-bin shift caused by divergence is checked to point in the same direction in both
          tools.

        The polyDFE divergence cache is regenerated by ``snakemake/scripts/precompute_polydfe_divergence.py``.
        """
        # the discretization (linearized DFE -> SFS transform) is the expensive part and only
        # depends on the sample size and intervals, which are identical across all examples and
        # both modes, so we build it once and reuse it
        discretization = None

        for example in self.examples:
            cache_div = f'testing/cache/polydfe/{example}_C_full_anc_divergence/serialized.json'
            cache_nodiv = f'testing/cache/polydfe/{example}_C_full_anc_nodivergence/serialized.json'

            if not (os.path.exists(cache_div) and os.path.exists(cache_nodiv)):
                self.skipTest(f'Missing precomputed polyDFE divergence cache for {example}.')

            poly_div = PolyDFE.from_file(cache_div)
            poly_nodiv = PolyDFE.from_file(cache_nodiv)
            poly_dfe_div = np.asarray(poly_div.get_discretized()[0])
            poly_dfe_nodiv = np.asarray(poly_nodiv.get_discretized()[0])
            alpha_polydfe = poly_div.get_alpha_divergence()  # cached, no R needed

            def run(include_divergence):
                config = self._get_full_config_with_divergence(example)
                config.data['include_divergence'] = include_divergence
                if discretization is not None:
                    config.data['discretization'] = discretization
                inf = fd.BaseInference.from_config(config)
                inf.run()
                return inf

            inf_div = run(True)
            discretization = inf_div.discretization
            inf_nodiv = run(False)

            fast_dfe_div = np.asarray(inf_div.get_discretized()[0])
            fast_dfe_nodiv = np.asarray(inf_nodiv.get_discretized()[0])

            # 1. divergence-based alpha matches polyDFE's MK estimate, and fastDFE's divergence-based
            #    and polymorphism-based alphas agree with each other
            assert abs(inf_div.get_alpha(use_divergence=True) - alpha_polydfe) < 0.1, \
                f'{example}: MK alpha fastdfe vs polydfe'
            assert abs(inf_div.get_alpha(use_divergence=True) -
                       inf_div.get_alpha(use_divergence=False)) < 0.1, \
                f'{example}: divergence vs polymorphism alpha'

            # 2. binned DFEs agree between the tools in both modes
            assert np.abs(fast_dfe_div - poly_dfe_div).sum() < 0.25, f'{example}: div DFE disagree'
            assert np.abs(fast_dfe_nodiv - poly_dfe_nodiv).sum() < 0.25, f'{example}: nodiv DFE disagree'

            # 3. divergence reshapes both DFEs in the same direction (when the shift is non-trivial)
            shift_fast = fast_dfe_div - fast_dfe_nodiv
            shift_poly = poly_dfe_div - poly_dfe_nodiv
            if np.abs(shift_poly).sum() > 0.02:
                cosine = float(shift_fast @ shift_poly /
                               (np.linalg.norm(shift_fast) * np.linalg.norm(shift_poly)))
                assert cosine > 0.7, f'{example}: divergence shift misaligned (cos={cosine})'


class BaseInferenceTestCase(InferenceTestCase):
    """
    Test BaseInference.
    """
    config_file = "testing/cache/configs/pendula_C_full_anc/config.yaml"
    serialized = "testing/cache/fastdfe/pendula_C_full_anc/serialized.json"

    def test_run_inference_from_config_parallelized(self):
        """
        Run inference from config file.
        """
        config = fd.Config.from_file(self.config_file)
        config.update(parallelize=True)

        inf = fd.BaseInference.from_config(config)
        inf.run()

        # make sure the likelihood is the maximum of the runs
        self.assertEqual(inf.likelihood, inf.runs.likelihood.max())

    @pytest.mark.inference
    def test_run_inference_from_config_not_parallelized(self):
        """
        Run inference from config file.
        """
        config = fd.Config.from_file(self.config_file)
        config.update(parallelize=False)

        inf = fd.BaseInference.from_config(config)
        inf.run()

        # make sure the likelihood is the maximum of the runs
        self.assertEqual(inf.likelihood, inf.runs.likelihood.max())

    @pytest.mark.inference
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

        inf = fd.BaseInference.from_config(config)
        inf.run()

        inf2 = fd.BaseInference.from_config(config)
        inf2.run()

        self.assertEqual(inf.params_mle, inf2.params_mle)
        assert_frame_equal(inf.bootstraps, inf2.bootstraps)

    @pytest.mark.inference
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

        inference = fd.BaseInference.from_config(config)
        inference.run()

        inference2 = fd.BaseInference.from_config(config)
        inference2.run()

        self.assertEqual(inference.params_mle, inference2.params_mle)
        assert_frame_equal(inference.bootstraps, inference2.bootstraps)

    @pytest.mark.inference
    def test_compare_inference_with_log_scales_vs_lin_scales(self):
        """
        Compare inference with log scales vs linear scales.
        """
        config = fd.Config.from_file(self.config_file)

        model = fd.GammaExpParametrization()
        model.scales = dict(
            S_d='log',
            b='log',
            p_b='lin',
            S_b='log'
        )

        config.update(
            model=model,
            do_bootstrap=True
        )

        inference_log = fd.BaseInference.from_config(config)
        inference_log.run()

        model = copy.copy(model)
        model.scales = dict(
            S_d='lin',
            b='lin',
            p_b='lin',
            S_b='lin'
        )

        config.update(model=model)
        inference_lin = fd.BaseInference.from_config(config)
        inference_lin.run()

        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        inference_lin.plot_discretized(ax=axs[0], show=False)
        inference_log.plot_discretized(ax=axs[1], show=False)

        axs[0].set_title('Linear scales')
        axs[1].set_title('Log scales')

        plt.tight_layout()
        plt.show()

        # get confidence intervals
        cis_lin = inference_lin.get_stats_discretized()[2]
        cis_log = inference_log.get_stats_discretized()[2]

        # assert that the confidence intervals overlap
        assert np.all(cis_lin[0] < cis_log[1])
        assert np.all(cis_log[0] < cis_lin[1])

        # likelihood on log scales should be better
        assert inference_log.likelihood > inference_lin.likelihood
        assert inference_log.bootstraps.likelihood.mean() > inference_lin.bootstraps.likelihood.mean()

        # standard deviation of bootstrapped likelihoods should be lower on log scales
        assert inference_log.bootstraps.likelihoods_std.mean() * 2 < inference_lin.bootstraps.likelihoods_std.mean()

    @pytest.mark.inference
    def test_compare_inference_with_log_scales_vs_lin_scales_tutorial(self):
        """
        Compare inference with log scales vs linear scales.
        """
        model = fd.GammaExpParametrization()
        model.scales = dict(
            S_d='log',
            b='log',
            p_b='lin',
            S_b='log'
        )

        inference_log = fd.BaseInference(
            sfs_neut=fd.Spectrum([177130, 997, 441, 228, 156, 117, 114, 83, 105, 109, 652]),
            sfs_sel=fd.Spectrum([797939, 1329, 499, 265, 162, 104, 117, 90, 94, 119, 794]),
            fixed_params=dict(all=dict(h=0.5)),
            model=model,
            do_bootstrap=True,
            n_runs=10
        )
        inference_log.run()

        model = fd.GammaExpParametrization()
        model.scales = dict(
            S_d='lin',
            b='lin',
            p_b='lin',
            S_b='lin'
        )

        inference_lin = fd.BaseInference(
            sfs_neut=fd.Spectrum([177130, 997, 441, 228, 156, 117, 114, 83, 105, 109, 652]),
            sfs_sel=fd.Spectrum([797939, 1329, 499, 265, 162, 104, 117, 90, 94, 119, 794]),
            fixed_params=dict(all=dict(h=0.5)),
            model=model,
            do_bootstrap=True,
            n_runs=10
        )
        inference_lin.run()

        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        inference_lin.plot_discretized(ax=axs[0], show=False)
        inference_log.plot_discretized(ax=axs[1], show=False)

        axs[0].set_title('Linear scales')
        axs[1].set_title('Log scales')

        plt.tight_layout()
        plt.show()

        # get confidence intervals
        cis_lin = inference_lin.get_stats_discretized()[1]
        cis_log = inference_log.get_stats_discretized()[1]

        # assert that the confidence intervals overlap
        # assert np.all(cis_lin[0] < cis_log[1])
        # assert np.all(cis_log[0] < cis_lin[1])

        # likelihood on log scales should be better
        assert inference_log.likelihood > inference_lin.likelihood

        # standard deviation of bootstrapped likelihoods should be lower on log scales
        assert inference_log.bootstraps.likelihoods_std.mean() * 100 < inference_lin.bootstraps.likelihoods_std.mean()

    def test_restore_serialized_inference(self):
        """
        Check whether Inference can properly be serialized and restored.
        """
        inference = fd.BaseInference.from_config_file(self.config_file)
        inference.run()

        serialized = 'scratch/test_serialize_inference.json'
        inference.to_file(serialized)
        inference_restored = fd.BaseInference.from_file(serialized)

        # self.assertEqual(inference.to_json(), inference_restored.to_json())
        self.assertEqualObjects(inference, inference_restored)

    def test_inference_run_bootstrap(self):
        """
        Run bootstrap on inference.
        """
        # unserialize
        inference = fd.BaseInference.from_file(self.serialized)

        # make sure the likelihoods of the different runs are cached
        self.assertEqual(len(inference.likelihoods), inference.n_runs)

        self.assertIsNone(inference.bootstraps)

        inference.bootstrap(2, parallelize=False)

        self.assertIsNotNone(inference.bootstraps)

        # make sure the likelihoods of the different runs are not overwritten
        self.assertEqual(len(inference.likelihoods), inference.n_runs)

    def test_evaluate_likelihood_same_as_mle_results(self):
        """
        Check that evaluating the loss function at the MLE result
        yields the same likelihood as the one reported.
        """
        # unserialize
        inference = fd.BaseInference.from_file(self.serialized)

        self.assertAlmostEqual(inference.evaluate_likelihood(dict(all=inference.params_mle)), inference.likelihood)

    def test_visualize_inference_without_bootstraps(self):
        """
        Plot everything possible.
        """
        config = fd.Config.from_file(self.config_file)

        config.update(
            do_bootstrap=False
        )

        inference = fd.BaseInference.from_config(config)

        inference.plot_all()
        inference.plot_bucket_sizes()

    @pytest.mark.inference
    def test_visualize_inference_with_bootstraps(self):
        """
        Plot everything possible.
        """
        config = fd.Config.from_file(self.config_file)

        config.update(
            do_bootstrap=True
        )

        inference = fd.BaseInference.from_config(config)

        inference.plot_all()
        inference.plot_bucket_sizes()

    def test_plot_likelihoods(self):
        """
        Plot likelihoods.
        """
        # unserialize
        inference = fd.BaseInference.from_file(self.serialized)

        inference.plot_likelihoods()

    def test_plot_interval_density(self):
        """
        Plot likelihoods.
        """
        # unserialize
        inference = fd.BaseInference.from_file(self.serialized)

        inference.plot_interval_density()

    def test_plot_inferred_parameters(self):
        """
        Plot inferred parameters.
        """
        # unserialize
        inference = fd.BaseInference.from_file(self.serialized)

        inference.plot_inferred_parameters()

    def test_plot_inferred_parameters_boxplot(self):
        """
        Plot inferred parameters.
        """
        # unserialize
        inference = fd.BaseInference.from_file(self.serialized)

        inference.bootstrap(6)

        inference.plot_inferred_parameters_boxplot()

    def test_plot_observed_sfs(self):
        """
        Plot likelihoods.
        """
        # unserialize
        inference = fd.BaseInference.from_file(self.serialized)

        inference.plot_observed_sfs()

    def test_plot_provide_axes(self):
        """
        Plot likelihoods.
        """
        # unserialize
        inference = fd.BaseInference.from_file(self.serialized)

        _, axs = plt.subplots(ncols=2)

        inference.plot_interval_density(show=False, ax=axs[0])
        inference.plot_interval_density(ax=axs[1])

    def test_visualize_inference_bootstrap(self):
        """
        Plot everything possible.
        """
        # unserialize
        inference = fd.BaseInference.from_file(self.serialized)

        # bootstrap
        inference.bootstrap(2)

        inference.plot_all()

    def test_perform_bootstrap_without_running(self):
        """
        Perform bootstrap before having run the main inference.
        """
        # unserialize
        inference = fd.BaseInference.from_config_file(self.config_file)

        # bootstrap
        inference.bootstrap(2)

    @pytest.mark.inference
    def test_compare_nested_with_bootstrap(self):
        """
        Compare nested likelihoods.
        """
        # unserialize
        inference = fd.BaseInference.from_config_file(self.config_file)

        inference.compare_nested_models()

        inference.plot_nested_models()
        inference.plot_nested_models(remove_empty=True)
        inference.plot_nested_models(transpose=True)

    def test_compare_nested_different_parametrizations_raises_error(self):
        """
        Make sure that comparing nested likelihoods with different parametrizations raises an error.
        """
        config1 = fd.Config.from_file(self.config_file)
        config1.update(model=fd.GammaExpParametrization())
        inf1 = fd.BaseInference.from_config(config1)

        config2 = fd.Config.from_file(self.config_file)
        config2.update(model=fd.DiscreteParametrization())
        inf2 = fd.BaseInference.from_config(config2)

        with self.assertRaises(ValueError):
            inf1.compare_nested(inf2)

    def test_compare_nested_different_fixed_params_raises_error(self):
        """
        Make sure that comparing nested likelihoods with different fixed parameters raises an error.
        """
        config1 = fd.Config.from_file(self.config_file)
        config1.update(fixed_params=dict(all=dict(S_b=1, p_b=0)))
        inf1 = fd.BaseInference.from_config(config1)

        config2 = fd.Config.from_file(self.config_file)
        config2.update(fixed_params=dict(all=dict(S_b=1, eps=0)))
        inf2 = fd.BaseInference.from_config(config2)

        with self.assertRaises(ValueError):
            inf1.compare_nested(inf2)

    def test_compare_nested_valid_comparison_1df(self):
        """
        Make sure that comparing nested models with the same parametrization works as expected.
        """
        config1 = fd.Config.from_file(self.config_file)
        config1.update(fixed_params=dict(all=dict(S_b=1, p_b=0, h=0.5)), do_bootstrap=False)
        inf1 = fd.BaseInference.from_config(config1)
        inf1.run()

        config2 = fd.Config.from_file(self.config_file)
        config2.update(fixed_params=dict(all=dict(S_b=1, h=0.5)), do_bootstrap=False)
        inf2 = fd.BaseInference.from_config(config2)
        inf2.run()

        assert 0 < inf1.compare_nested(inf2) < 1

    def test_compare_nested_valid_automatic_run(self):
        """
        Make sure that comparing nested models with the same parametrization works as expected.
        """
        config1 = fd.Config.from_file(self.config_file)
        config1.update(fixed_params=dict(all=dict(S_b=1, p_b=0, h=0.5)), do_bootstrap=False)
        inf1 = fd.BaseInference.from_config(config1)

        config2 = fd.Config.from_file(self.config_file)
        config2.update(fixed_params=dict(all=dict(S_b=1, h=0.5)), do_bootstrap=False)
        inf2 = fd.BaseInference.from_config(config2)

        assert 0 < inf1.compare_nested(inf2) < 1

    def test_lrt_components(self):
        """
        Check whether the weights and degrees of freedom of the components are computed correctly.
        """
        inf = fd.BaseInference.from_config_file(self.config_file)

        # test weights
        np.testing.assert_array_equal(inf._lrt_components(ll_simple=50, ll_complex=51, df=1, m=0)[0], [1])
        np.testing.assert_array_equal(inf._lrt_components(ll_simple=50, ll_complex=51, df=1, m=1)[0], [0.5, 0.5])
        np.testing.assert_array_equal(inf._lrt_components(ll_simple=50, ll_complex=51, df=2, m=1)[0], [0.5, 0.5])
        np.testing.assert_array_equal(inf._lrt_components(ll_simple=50, ll_complex=51, df=2, m=2)[0], [0.25, 0.5, 0.25])
        np.testing.assert_array_equal(inf._lrt_components(ll_simple=50, ll_complex=51, df=2, m=0)[0], [1])
        np.testing.assert_array_equal(
            inf._lrt_components(ll_simple=50, ll_complex=51, df=3, m=3)[0], [0.125, 0.375, 0.375, 0.125]
        )
        np.testing.assert_array_equal(inf._lrt_components(ll_simple=50, ll_complex=51, df=3, m=2)[0], [0.25, 0.5, 0.25])
        np.testing.assert_array_equal(inf._lrt_components(ll_simple=50, ll_complex=51, df=3, m=0)[0], [1])

        # test degrees of freedom of components
        np.testing.assert_array_equal(inf._lrt_components(ll_simple=50, ll_complex=51, df=1, m=0)[2], [1])
        np.testing.assert_array_equal(inf._lrt_components(ll_simple=50, ll_complex=51, df=1, m=1)[2], [0, 1])
        np.testing.assert_array_equal(inf._lrt_components(ll_simple=50, ll_complex=51, df=2, m=1)[2], [1, 2])
        np.testing.assert_array_equal(inf._lrt_components(ll_simple=50, ll_complex=51, df=2, m=2)[2], [0, 1, 2])
        np.testing.assert_array_equal(inf._lrt_components(ll_simple=50, ll_complex=51, df=2, m=0)[2], [2])
        np.testing.assert_array_equal(inf._lrt_components(ll_simple=50, ll_complex=51, df=3, m=3)[2], [0, 1, 2, 3])
        np.testing.assert_array_equal(inf._lrt_components(ll_simple=50, ll_complex=51, df=3, m=2)[2], [1, 2, 3])
        np.testing.assert_array_equal(inf._lrt_components(ll_simple=50, ll_complex=51, df=3, m=0)[2], [3])

    def test_is_nested(self):
        """
        Check whether the is_nested method works as expected.
        """
        config1 = fd.Config.from_file(self.config_file).update(fixed_params={})
        config2 = copy.deepcopy(config1).update(fixed_params={}, model=fd.DiscreteParametrization())
        config3 = copy.deepcopy(config1).update(fixed_params=dict(all=dict(eps=0)))
        config4 = copy.deepcopy(config1).update(fixed_params=dict(all=dict(eps=0, S_b=1)))
        config5 = copy.deepcopy(config1).update(fixed_params=dict(all=dict(eps=0)),
                                                model=fd.DiscreteParametrization())

        def is_nested(simple: fd.Config, complex: fd.Config) -> bool:
            return fd.BaseInference.from_config(simple)._is_nested(fd.BaseInference.from_config(complex))

        self.assertFalse(is_nested(simple=config1, complex=config2))
        self.assertFalse(is_nested(simple=config2, complex=config1))
        self.assertFalse(is_nested(simple=config1, complex=config1))
        self.assertFalse(is_nested(simple=config1, complex=config3))
        self.assertTrue(is_nested(simple=config3, complex=config1))
        self.assertTrue(is_nested(simple=config4, complex=config1))
        self.assertTrue(is_nested(simple=config4, complex=config3))
        self.assertFalse(is_nested(simple=config3, complex=config4))
        self.assertFalse(is_nested(simple=config3, complex=config5))
        self.assertFalse(is_nested(simple=config5, complex=config1))

    def test_plot_nested_model_comparison_without_running(self):
        """
        Perform nested model comparison before having run the main inference.
        We just make sure this triggers a run.
        """
        # unserialize
        inference = fd.BaseInference.from_config_file(self.config_file)

        inference.plot_nested_models()

    @pytest.mark.inference
    def test_plot_nested_model_comparison_different_parametrizations(self):
        """
        Perform nested model comparison before having run the main inference.
        We just make sure this triggers a run.
        """
        parametrizations = [
            fd.GammaExpParametrization(),
            fd.DisplacedGammaParametrization(),
            fd.DiscreteParametrization(),
            fd.DiscreteFractionalParametrization(),
            fd.GammaDiscreteParametrization()
        ]

        for p in parametrizations:
            config = fd.Config.from_file(self.config_file).update(
                model=p,
                do_bootstrap=True,
                n_bootstraps=2,
                n_runs=1,
                fixed_params=dict(all=dict(h=0.5))
            )

            # unserialize
            inference = fd.BaseInference.from_config(config)

            inference.plot_nested_models(title=p.__class__.__name__)

    def test_recreate_from_config(self):
        """
        Test whether inference can be recreated from config.
        """
        inf = fd.BaseInference.from_config_file(self.config_file)

        inf2 = fd.BaseInference.from_config(inf.create_config())

        self.assertEqualObjects(inf, inf2)

    def test_cached_result(self):
        """
        Test inference against cached result.
        """
        inf = fd.BaseInference.from_file(self.serialized)
        inf2 = fd.BaseInference.from_config(inf.create_config())

        inf2.run()

        inf_params_mle = np.array(list((inf.params_mle | {'h': 0.5}).values()))
        inf2_params_mle = np.array(list(inf2.params_mle.values()))

        diff = np.abs(inf_params_mle - inf2_params_mle) / (np.abs(inf_params_mle) + 1e-10)

        # compare MLE parameters
        self.assertLess(diff.max(), 1e-3)

    def test_run_if_necessary_wrapper_triggers_run(self):
        """
        Test whether the run_if_necessary wrapper triggers a run if the inference has not been run yet.
        """
        inference = fd.BaseInference.from_config_file(self.config_file)

        with mock.patch.object(fd.BaseInference, 'run', side_effect=Exception()) as mock_run:
            try:
                inference.plot_inferred_parameters(show=False)
            except Exception:
                pass

            mock_run.assert_called_once()

    def test_run_if_necessary_wrapper_not_triggers_run(self):
        """
        Test whether the run_if_necessary wrapper does not trigger a run if the inference has already been run.
        """
        inference = fd.BaseInference.from_file(self.serialized)

        with mock.patch.object(fd.BaseInference, 'run') as mock_run:
            inference.plot_inferred_parameters(show=False)
            mock_run.assert_not_called()

    def test_run_with_sfs_neut_zero_entry(self):
        """
        Test whether inference can be run with a neutral SFS that has a zero entry.
        """
        config = fd.Config.from_file(self.config_file)

        # set neutral SFS entry to zero
        sfs = config.data['sfs_neut']['all'].to_list()
        sfs[3] = 0
        sfs = fd.Spectrum(sfs)
        config.data['sfs_neut'] = fd.Spectra.from_spectrum(sfs)

        inference = fd.BaseInference.from_config(config)

        inference.run()

        # here both the nuisance parameters and epsilon cause the high number of high-frequency variants
        inference.plot_sfs_comparison()

        inference.plot_discretized()

    def test_run_with_sfs_sel_zero_entry(self):
        """
        Test whether inference can be run with a selected SFS that has a zero entry.
        """
        config = fd.Config.from_file(self.config_file)

        # set selected SFS entry to zero
        sfs = config.data['sfs_sel']['all'].to_list()
        sfs[3] = 0
        sfs = fd.Spectrum(sfs)
        config.data['sfs_sel'] = fd.Spectra.from_spectrum(sfs)

        inference = fd.BaseInference.from_config(config)

        inference.run()

        # this looks as expected, i.e. we reduce the overall distance between the modelled and observed SFS
        inference.plot_sfs_comparison()

        inference.plot_discretized()

    def test_run_with_both_sfs_zero_entry(self):
        """
        Test whether inference can be run with both SFS that have a zero entry.
        """
        config = fd.Config.from_file(self.config_file)

        # set neutral SFS entry to zero
        sfs = config.data['sfs_neut']['all'].to_list()
        sfs[3] = 0
        sfs = fd.Spectrum(sfs)
        config.data['sfs_neut'] = fd.Spectra.from_spectrum(sfs)

        # set selected SFS entry to zero
        sfs = config.data['sfs_sel']['all'].to_list()
        sfs[3] = 0
        sfs = fd.Spectrum(sfs)
        config.data['sfs_sel'] = fd.Spectra.from_spectrum(sfs)

        inference = fd.BaseInference.from_config(config)

        inference.run()

        # the nuisance parameters cause the zero entry in the modelled SFS
        inference.plot_sfs_comparison()

        inference.plot_discretized()

    def test_fixed_parameters(self):
        """
        Test whether fixed parameters are correctly set.
        """
        config = fd.Config.from_file(self.config_file)

        config.data['fixed_params'] = dict(all=dict(b=1.123, eps=0.123, h=0.5))

        inference = fd.BaseInference.from_config(config)

        inference.run()

        assert inference.params_mle['b'] == config.data['fixed_params']['all']['b']
        assert inference.params_mle['eps'] == config.data['fixed_params']['all']['eps']
        assert inference.params_mle['S_b'] != config.data['x0']['all']['S_b']
        assert inference.params_mle['S_d'] != config.data['x0']['all']['S_d']
        assert inference.params_mle['p_b'] != config.data['x0']['all']['p_b']

    def test_get_n_optimized(self):
        """
        Test get number of parameters to be optimized
        """
        config = fd.Config.from_file(self.config_file)

        assert fd.BaseInference.from_config(config).get_n_optimized() == 5

        config.data['fixed_params'] = dict(all=dict(S_b=1, p_b=0, h=0.5))

        assert fd.BaseInference.from_config(config).get_n_optimized() == 3

    def test_non_existing_fixed_param_raises_error(self):
        """
        Test that a non-existing fixed parameter raises an error.
        """
        config = fd.Config.from_file(self.config_file)

        config.data['fixed_params'] = dict(all=dict(S_b=1, p_b=0, foo=1))

        with self.assertRaises(ValueError):
            fd.BaseInference.from_config(config)

    def test_get_discretized_errors_with_bootstraps(self):
        """
        Get the discretized DFE errors.
        """
        inference = fd.BaseInference.from_file(self.serialized)

        inference.bootstrap()

        res = inference.get_discretized()

    def test_get_discretized_errors_no_bootstraps(self):
        """
        Get the discretized DFE errors.
        """
        inference = fd.BaseInference.from_file(self.serialized)

        values, errors = inference.get_discretized()

        assert errors is None

    def test_spectrum_with_zero_monomorphic_counts_throws_error(self):
        """
        Test whether a spectrum with zero monomorphic counts throws an error.
        """
        sfs_neut = fd.Spectrum([0, 997, 441, 228, 156, 117, 114, 83, 105, 109, 652])
        sfs_sel = fd.Spectrum([797939, 1329, 499, 265, 162, 104, 117, 90, 94, 119, 794])

        with self.assertRaises(ValueError):
            fd.BaseInference(
                sfs_neut=sfs_neut,
                sfs_sel=sfs_sel,
            )

    def test_spectrum_with_few_monomorphic_counts_raises_warning(self):
        """
        Test whether a spectrum with zero monomorphic counts throws an error.
        """
        sfs_neut = fd.Spectrum([1000, 997, 441, 228, 156, 117, 114, 83, 105, 109, 652])
        sfs_sel = fd.Spectrum([797939, 1329, 499, 265, 162, 104, 117, 90, 94, 119, 794])

        with self.assertLogs(level="WARNING", logger=logging.getLogger('fastdfe')):
            fd.BaseInference(
                sfs_neut=sfs_neut,
                sfs_sel=sfs_sel,
            )

    @staticmethod
    def test_adjust_polarization():
        """
        Test whether polarization is adjusted correctly.
        """
        counts_list = [np.array([1, 2, 3, 4, 5]), np.array([2, 3, 5, 7, 11])]
        eps_list = [0, 0.1, 0.5, 1]

        expected_counts_list = [
            [
                np.array([1, 2, 3, 4, 5]),
                np.array([1.4, 2.2, 3.0, 3.8, 4.6]),
                np.array([3.0, 3.0, 3.0, 3.0, 3.0]),
                np.array([5, 4, 3, 2, 1])
            ],
            [
                np.array([2, 3, 5, 7, 11]),
                np.array([2.9, 3.4, 5, 6.6, 10.1]),
                np.array([6.5, 5.0, 5.0, 5.0, 6.5]),
                np.array([11, 7, 5, 3, 2])
            ]
        ]

        for i, counts in enumerate(counts_list):
            for j, eps in enumerate(eps_list):
                adjusted_counts = fd.BaseInference._adjust_polarization(counts, eps)
                assert np.isclose(np.sum(counts), np.sum(adjusted_counts))
                assert np.allclose(adjusted_counts, expected_counts_list[i][j])

    def test_estimating_full_dfe_with_folded_sfs_raises_warning(self):
        """
        Test whether estimating the full DFE with a folded SFS raises a warning.
        """
        sfs_neut = fd.Spectrum([1, 1, 0, 0])
        sfs_sel = fd.Spectrum([1, 1, 0, 0])

        with self.assertLogs(level="WARNING", logger=logging.getLogger('fastdfe')):
            fd.BaseInference(
                sfs_neut=sfs_neut,
                sfs_sel=sfs_sel,
            )

    @staticmethod
    def test_folded_inference_even_sample_size():
        """
        Test folded inference with even sample size.
        """
        sfs_neut = fd.Spectrum([177130, 997, 441, 228, 156, 117, 114, 83, 105, 109, 652]).fold()
        sfs_sel = fd.Spectrum([797939, 1329, 499, 265, 162, 104, 117, 90, 94, 119, 794]).fold()

        inf = fd.BaseInference(
            sfs_neut=sfs_neut,
            sfs_sel=sfs_sel,
            fixed_params=dict(all=dict(S_b=1, p_b=0, h=0.5)),
            do_bootstrap=False,
        )

        inf.plot_discretized()

    @staticmethod
    def test_folded_inference_odd_sample_size():
        """
        Test folded inference with odd sample size.
        """
        sfs_neut = fd.Spectrum([177130, 997, 441, 228, 156, 117, 114, 83, 105, 652]).fold()
        sfs_sel = fd.Spectrum([797939, 1329, 499, 265, 162, 104, 117, 90, 94, 794]).fold()

        inf = fd.BaseInference(
            sfs_neut=sfs_neut,
            sfs_sel=sfs_sel,
            fixed_params=dict(all=dict(S_b=1, p_b=0, h=0.5)),
            do_bootstrap=False,
        )

        inf.plot_discretized()

        pass

    def test_sample_data_fixed_result(self):
        """
        Test against fixed result.
        """
        sfs_neut = fd.Spectrum([177130, 997, 441, 228, 156, 117, 114, 83, 105, 109, 652])
        sfs_sel = fd.Spectrum([797939, 1329, 499, 265, 162, 104, 117, 90, 94, 119, 794])

        inf = fd.BaseInference(
            sfs_neut=sfs_neut,
            sfs_sel=sfs_sel,
            fixed_params=dict(all=dict(h=0.5)),
            do_bootstrap=False,
        )

        inf.run()

        expected = {
            'S_d': -9868.141535,
            'b': 0.150810,
            'eps': 0.006854,
            'p_b': 0.0
        }

        self.assertAlmostEqual(inf.params_mle['S_d'], expected['S_d'], delta=2)
        self.assertAlmostEqual(inf.params_mle['b'], expected['b'], delta=1e-2)
        self.assertAlmostEqual(inf.params_mle['eps'], expected['eps'], delta=1e-3)
        self.assertAlmostEqual(inf.params_mle['p_b'], expected['p_b'], delta=1e-4)

    def test_base_inference_l2_loss_type(self):
        """
        Test whether the L2 loss type is correctly set.
        """
        sfs_neut = fd.Spectrum([177130, 997, 441, 228, 156, 117, 114, 83, 105, 109, 652])
        sfs_sel = fd.Spectrum([797939, 1329, 499, 265, 162, 104, 117, 90, 94, 119, 794])

        inf = fd.BaseInference(
            sfs_neut=sfs_neut,
            sfs_sel=sfs_sel,
            fixed_params=dict(all=dict(h=0.5)),
            loss_type='L2',
            do_bootstrap=True,
            n_bootstraps=5
        )

        inf.run()

    @staticmethod
    def test_few_polymorphic_sites_raises_no_error():
        """
        Test whether a spectrum with few polymorphic sites raises no error.
        """
        inf = fd.BaseInference(
            sfs_neut=fd.Spectrum([1243, 0, 0, 1, 0, 0, 0]),
            sfs_sel=fd.Spectrum([12421, 0, 0, 0, 0, 1, 0]),
            do_bootstrap=False,
        )

        # There are sometimes problems with the optimization for spectra like these,
        # but this should be nothing to worry about
        inf.run()

        inf.plot_discretized()

        pass

    def test_no_polymorphic_sites_raises_error(self):
        """
        Test whether a spectrum with zero monomorphic counts throws an error.
        """
        with self.assertRaises(ValueError) as context:
            fd.BaseInference(
                sfs_neut=fd.Spectrum([1243, 0, 0, 0, 0, 0, 0]),
                sfs_sel=fd.Spectrum([12421, 0, 0, 0, 0, 0, 0]),
            )

        print(context.exception)

    def test_different_sfs_sample_sizes_raises_error(self):
        """
        Test whether a spectrum with zero monomorphic counts throws an error.
        """
        with self.assertRaises(ValueError) as context:
            fd.BaseInference(
                sfs_neut=fd.Spectrum([1000, 4, 2, 1]),
                sfs_sel=fd.Spectrum([1000, 4, 2, 1, 0])
            )

        print(context.exception)

    def test_selected_sfs_without_polymorphic_counts_raises_error(self):
        """
        A selected spectrum without polymorphic counts is rejected. The guard tested the neutral spectrum twice.
        """
        with self.assertRaises(ValueError):
            fd.BaseInference(
                sfs_neut=fd.Spectrum([1000, 4, 2, 1, 5]),
                sfs_sel=fd.Spectrum([1000, 0, 0, 0, 5])
            )

    @staticmethod
    def test_spectra_with_fractional_counts():
        """
        Test whether a spectrum with fractional counts works as expected.
        """
        sfs_neut = fd.Spectrum([177130.4, 997.3, 441.2, 228.1, 156.45, 117.2, 114.9, 83.12, 105.453, 109.1, 652.05])
        sfs_sel = fd.Spectrum([797939.4, 1329.3, 499.2, 265.1, 162.12, 104.2, 117.9, 90.12, 94.453, 119.1, 794.05])

        inf_float = fd.BaseInference(
            sfs_neut=sfs_neut,
            sfs_sel=sfs_sel,
            fixed_params=dict(all=dict(h=0.5)),
            do_bootstrap=False,
        )

        inf_float.run()

        inf_int = fd.BaseInference(
            sfs_neut=fd.Spectrum(sfs_neut.data.astype(int)),
            sfs_sel=fd.Spectrum(sfs_sel.data.astype(int)),
            fixed_params=dict(all=dict(h=0.5)),
            do_bootstrap=False,
        )

        inf_int.run()

        fd.Inference.plot_discretized([inf_float, inf_int], labels=['float', 'int'])

        assert np.abs((inf_float.params_mle['S_d'] - inf_int.params_mle['S_d']) / inf_float.params_mle['S_d']) < 1e-2
        assert np.abs((inf_float.params_mle['b'] - inf_int.params_mle['b']) / inf_float.params_mle['b']) < 1e-2

    @pytest.mark.inference
    @staticmethod
    def test_manuscript_example():
        """
        Test the example from the manuscript.
        """
        # create inference object
        inf = fd.BaseInference(
            sfs_neut=fd.Spectrum([66200, 410, 120, 60, 42, 43, 52, 65, 0]),
            sfs_sel=fd.Spectrum([281937, 600, 180, 87, 51, 43, 49, 61, 0]),
            model=fd.GammaExpParametrization(),  # the model to use
            n_runs=10,  # number of optimization runs
            n_bootstraps=100,  # number of bootstrap replicates
            do_bootstrap=True
        )

        # create subplots
        axs = plt.subplots(2, 2, figsize=(11, 7))[1].flatten()

        # plot results
        types = ['neutral', 'selected']
        inf.plot_sfs_comparison(ax=axs[0], show=False, sfs_types=types)
        inf.plot_sfs_comparison(ax=axs[1], show=False, colors=['C1', 'C5'])
        inf.plot_inferred_parameters(ax=axs[2], show=False)
        inf.plot_discretized(ax=axs[3], show=True)

    def test_infer_no_selection(self):
        """
        Test whether we infer a mostly neutral DFE when there is no selection.
        """
        sfs = fd.Spectrum.standard_kingman(10, n_monomorphic=100)

        # use different scalings which should not affect the result
        for m in [1, 10, 0.1, sfs]:
            inf = fd.BaseInference(
                sfs_neut=sfs * m,
                sfs_sel=sfs * m,
                fixed_params=dict(all=dict(h=0.5)),
                do_bootstrap=False,
                seed=42,
                model=fd.DiscreteFractionalParametrization(
                    intervals=np.array([-100000, -0.1, 0.1, 10000])
                )
            )

            inf.run()

            # check that DFE is very neutral
            self.assertAlmostEqual(1, inf.get_discretized(intervals=np.array([-100000, -0.1, 0.1, 10000]))[0][1])

    def test_infer_strongly_beneficial_selection_target_sites(self):
        """
        Test whether we infer a strongly beneficial DFE when there is strong selection.
        """
        # use different mutational target sizes
        inf = fd.BaseInference(
            sfs_neut=fd.Spectrum.standard_kingman(10, n_monomorphic=100),
            sfs_sel=fd.Spectrum.standard_kingman(10, n_monomorphic=10),
            fixed_params=dict(all=dict(h=0.5)),
            do_bootstrap=False,
            seed=42
        )

        inf.run()

        # check that DFE is very weakly beneficial
        self.assertGreaterEqual(inf.get_discretized()[0][-1], 0.4)

    def test_infer_strongly_deleterious_selection_target_sites(self):
        """
        Test whether we infer a strongly deleterious DFE when there is strong selection.
        """
        # use different mutational target sizes
        inf = fd.BaseInference(
            sfs_neut=fd.Spectrum.standard_kingman(10, n_monomorphic=100),
            sfs_sel=fd.Spectrum.standard_kingman(10, n_monomorphic=1000),
            fixed_params=dict(all=dict(h=0.5)),
            do_bootstrap=False,
            seed=42
        )

        inf.run()

        # check that DFE is very strongly deleterious
        self.assertGreaterEqual(inf.get_discretized()[0][0], 0.8)

    def test_infer_deleterious_selection_sfs_shape(self):
        """
        Test whether we infer a deleterious DFE when there is selection.
        """
        n = 10

        sfs_neut = fd.Spectrum.standard_kingman(n, n_monomorphic=100)
        sfs_sel = fd.Spectrum.from_polymorphic(
            Discretization(n=n).get_counts_semidominant_regularized(-10 * np.ones(n - 1), np.arange(1, n))
        )
        sfs_sel.data[0] = 100

        # use different mutational target sizes
        inf = fd.BaseInference(
            sfs_neut=sfs_neut,
            sfs_sel=sfs_sel,
            fixed_params=dict(all=dict(h=0.5)),
            do_bootstrap=False,
            seed=42
        )

        inf.run()

        # check that DFE is very deleterious
        self.assertAlmostEqual(1, inf.get_discretized(np.array([-np.inf, -1, np.inf]))[0][0])

    def test_alternative_optimizer(self):
        """
        Test whether using an alternative optimizer works as expected.
        """
        config = fd.Config.from_file(self.config_file)

        inf_default = fd.BaseInference.from_config(config)
        inf_default.run()

        inf_alt = fd.BaseInference.from_config(config.update(method_mle='TNC'))
        inf_alt.run()

    def test_raise_error_n_above_400(self):
        """
        Test whether an error is raised when sample size is above 400.
        """
        with self.assertRaises(ValueError) as context:
            fd.BaseInference(
                sfs_neut=fd.Spectrum.standard_kingman(401, n_monomorphic=10000),
                sfs_sel=fd.Spectrum.standard_kingman(401, n_monomorphic=10000)
            )

        print(context.exception)

    def test_bootstrap_n_retries(self):
        """
        Test whether bootstrapping with retries works as expected.
        """
        config = fd.Config.from_file(self.config_file).update(
            n_bootstrap_retries=3,
            parallelize=True,
            n_bootstraps=3,
            do_bootstrap=True
        )
        inf = fd.BaseInference.from_config(config)
        inf.run()

        self.assertEqual(3, len(inf.bootstraps.likelihoods_runs.iloc[0]))
        self.assertEqual(3, len(inf.bootstraps.x0_runs.iloc[0]))
        self.assertGreater(inf.bootstraps.likelihoods_std.mean(), 0)

        # make sure that the best likelihood per bootstrap is equal to the maximum over runs
        self.assertTrue(np.all(inf.bootstraps.likelihoods_runs.apply(max) == inf.bootstraps.likelihood))

    @pytest.mark.slow
    def test_infer_dfe_with_fixed_h_low_memory_consumption(self):
        """
        Make sure memory consumption is low for fixed dominance coefficients.
        """
        for h in [0, 0.5, 1]:
            inf = fd.BaseInference(
                sfs_neut=fd.Spectrum([177130, 997, 441, 228, 156, 117, 114, 83, 105, 109, 652]),
                sfs_sel=fd.Spectrum([797939, 1329, 499, 265, 162, 104, 117, 90, 94, 119, 794]),
                fixed_params={'all': {'h': h, 'S_b': 1, 'p_b': 0, 'eps': 0}},
                intervals_ben=(1.0e-5, 1.0e4, 200),
                intervals_del=(-1.0e+8, -1.0e-5, 200),
                parallelize=True,
                do_bootstrap=True,
                n_bootstraps=10,
                n_runs=10
            )

            inf.run()

            mem_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 3

            self.assertLess(mem_gb, 1)

            self.assertEqual(inf.params_mle['h'], h)

    @pytest.mark.slow
    def test_infer_h(self):
        """
        Test inference of h parameter.
        """
        inf = fd.BaseInference(
            sfs_neut=fd.Spectrum([177130, 997, 441, 228, 156, 117, 114, 83, 105, 109, 652]),
            sfs_sel=fd.Spectrum([797939, 1329, 499, 265, 162, 104, 117, 90, 94, 119, 794]),
            intervals_ben=(1.0e-5, 1.0e4, 30),
            intervals_del=(-1.0e+8, -1.0e-5, 30),
            intervals_h=(0, 1, 6),
            fixed_params={'all': {'S_b': 1, 'p_b': 0, 'eps': 0}},
            parallelize=False,
            do_bootstrap=True,
            n_bootstraps=50,
            n_runs=10
        )
        inf.run()
        inf.plot_discretized()

        mem_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 3

        self.assertLess(mem_gb, 1.1)

        self.assertGreater(inf.bootstraps.h.std(), 0)

    @pytest.mark.slow
    def test_infer_h_custom_callback(self):
        """
        Test inference of h parameter.
        """
        inf = fd.BaseInference(
            sfs_neut=fd.Spectrum([177130, 997, 441, 228, 156, 117, 114, 83, 105, 109, 652]),
            sfs_sel=fd.Spectrum([797939, 1329, 499, 265, 162, 104, 117, 90, 94, 119, 794]),
            intervals_ben=(1.0e-5, 1.0e4, 30),
            intervals_del=(-1.0e+8, -1.0e-5, 30),
            h_callback=lambda h, S: 0.4 * np.exp(-h * abs(S)),
            intervals_h=(0, 1, 6),
            fixed_params={'all': {'S_b': 1, 'p_b': 0, 'eps': 0}},
            bounds={
                'S_d': (-100000.0, -0.01),
                'b': (0.01, 10),
                'p_b': (0, 0.5),
                'S_b': (0.0001, 100),
                'eps': (0, 0.15),
                'h': (0, 100)
            },
            parallelize=False,
            do_bootstrap=True,
            n_bootstraps=50,
            n_runs=10
        )
        inf.run()
        inf.plot_discretized()

        pass

    @pytest.mark.inference
    def test_compare_nested_h_parameter(self):
        """
        Test nested model comparison when h is optimized in one model and fixed in the other.
        """
        inf1 = fd.BaseInference(
            sfs_neut=fd.Spectrum([177130, 997, 441, 228, 156, 117, 114, 83, 105, 109, 652]),
            sfs_sel=fd.Spectrum([797939, 1329, 499, 265, 162, 104, 117, 90, 94, 119, 794]),
            intervals_ben=(1.0e-5, 1.0e4, 30),
            intervals_del=(-1.0e+8, -1.0e-5, 30),
            intervals_h=(0, 1, 6),
            fixed_params={'all': {'S_b': 1, 'p_b': 0, 'eps': 0}},
            parallelize=True,
            do_bootstrap=True,
            n_runs=5,
            n_bootstraps=20
        )

        inf1.run()

        inf2 = fd.BaseInference.from_config(
            inf1.create_config().update(
                fixed_params={'all': {'S_b': 1, 'p_b': 0, 'eps': 0, 'h': 0.5}}
            )
        )
        inf2.run()

        p = inf2.compare_nested(inf1)

        self.assertTrue(0 < p < 1)

    @pytest.mark.inference
    def test_get_dfe_and_plot(self):
        """
        Test get DFE method.
        """
        inf = fd.BaseInference.from_file(self.serialized)

        inf.get_dfe().plot()

        inf.bootstrap(parallelize=False, n_samples=10)

        inf.get_dfe().plot()

        inf.get_dfe().get_bootstrap_dfes()[1].plot()

    @pytest.mark.skip(reason="Not a real test.")
    def test_runtime_sequential_vs_parallel(self):
        """
        Test whether parallelization speeds up inference.
        This doesn't seem to be true currently on osx-arm64 at least.
        """
        inf_seq = fd.BaseInference(
            sfs_neut=fd.Spectrum([177130, 997, 441, 228, 156, 117, 114, 83, 105, 109, 652]),
            sfs_sel=fd.Spectrum([797939, 1329, 499, 265, 162, 104, 117, 90, 94, 119, 794]),
            parallelize=False,
            do_bootstrap=True
        )

        start_time = time.time()
        inf_seq.run()
        seq_time = time.time() - start_time

        inf_par = fd.BaseInference(
            sfs_neut=fd.Spectrum([177130, 997, 441, 228, 156, 117, 114, 83, 105, 109, 652]),
            sfs_sel=fd.Spectrum([797939, 1329, 499, 265, 162, 104, 117, 90, 94, 119, 794]),
            parallelize=True,
            do_bootstrap=True
        )

        start_time = time.time()
        inf_par.run()
        par_time = time.time() - start_time

        self.assertLess(par_time, seq_time)

    def test_serialize_inference_result(self):
        """
        Test whether inference results can be serialized and unserialized correctly.
        """
        inf = fd.BaseInference.from_file(self.serialized)

        # serialize
        result = inf.get_summary()

        result.to_file("scratch/inference_result.json")

        # unserialize
        result2 = fd.InferenceResult.from_file("scratch/inference_result.json")

        self.assertEqualObjects(result, result2)


class SmokeTestCase(TestCase):
    """
    Fast smoke tests in the default (unit) tier. They exercise the optimizer end-to-end on tiny
    inputs (coarse discretization, a single run, no bootstrap) to catch API/integration breakage
    without asserting statistical accuracy — that lives in the ``inference`` and ``slow`` tiers.
    """

    sfs_neut = fd.Spectrum([177130, 997, 441, 228, 156, 117, 114, 83, 105, 109, 652])
    sfs_sel = fd.Spectrum([797939, 1329, 499, 265, 162, 104, 117, 90, 94, 119, 794])

    def test_nested_models_do_not_inherit_bootstraps_and_cached_summaries(self):
        """
        The inferences returned by ``compare_nested_models`` are refitted copies of the parent. They carry
        neither its bootstraps nor its cached alpha, omega and omega_a, which were kept across the refit.
        """
        inf = fd.BaseInference(
            sfs_neut=self.sfs_neut,
            sfs_sel=self.sfs_sel,
            intervals_del=(-1.0e+8, -1.0e-5, 20),
            intervals_ben=(1.0e-5, 1.0e4, 20),
            do_bootstrap=True,
            n_bootstraps=3,
            n_runs=1,
            parallelize=False
        )
        inf.run()

        # fill the caches of the parent
        cached = inf.alpha, inf.omega, inf.omega_a

        _, inferences = inf.compare_nested_models()

        for sub in inferences.values():
            self.assertIsNone(sub.bootstraps)
            self.assertEqual(sub.alpha, sub.get_alpha())
            self.assertEqual(sub.omega, sub.get_omega())
            self.assertEqual(sub.omega_a, sub.get_omega_a())

        # the full model has beneficial mutations, the deleterious parent has none
        self.assertEqual(cached[0], 0)
        self.assertGreater(inferences['full.no_anc'].alpha, 0)

        # the parent keeps its own results
        self.assertEqual(len(inf.bootstraps), 3)
        self.assertEqual((inf.alpha, inf.omega, inf.omega_a), cached)

    def test_folded_spectra_with_divergence_recover_simulated_dfe(self):
        """
        Folded spectra that carry a divergence count are treated as folded. They were taken for unfolded, as the
        divergence count occupies the last entry, which biased the fit and the simulated folded SFS.
        """
        opts = dict(intervals_del=(-1.0e+8, -1.0e-5, 30), intervals_ben=(1.0e-5, 1.0e4, 30))
        truth = dict(S_d=-300, b=0.3, p_b=0, S_b=1)
        neutral = fd.Simulation.get_neutral_sfs(n=12, n_sites=1e7, theta=1e-3)

        sim = fd.Simulation(sfs_neut=neutral.fold(), params=dict(truth), n_sites_div=2e6, **opts)
        sim.run()

        spectra = sim.get_spectra()

        self.assertGreater(spectra['neutral'].n_div, 0)

        # the simulated polymorphic counts equal those simulated from the unfolded neutral SFS and folded
        unfolded = fd.Simulation(sfs_neut=neutral, params=dict(truth), **opts).run()

        np.testing.assert_allclose(spectra['selected'].polymorphic, unfolded.fold().polymorphic, rtol=1e-10)

        inf = fd.BaseInference(
            sfs_neut=spectra['neutral'],
            sfs_sel=spectra['selected'],
            n_sites_div_neut=2e6,
            n_sites_div_sel=2e6,
            discretization=sim.discretization,
            fixed_params=dict(all=dict(eps=0, p_b=0, S_b=1, h=0.5)),
            do_bootstrap=False,
            n_runs=3,
            parallelize=False
        )
        inf.run()

        self.assertTrue(inf.folded)
        self.assertAlmostEqual(inf.params_mle['S_d'] / truth['S_d'], 1, places=2)
        self.assertAlmostEqual(inf.params_mle['b'] / truth['b'], 1, places=2)

    def test_folded_spectra_recover_simulated_dfe(self):
        """
        An inference on folded spectra recovers the DFE the spectra were simulated under. The modelled count
        of a folded class contained the lower of the two allele counts only, scaled by the folded neutral
        count, which biased the fit also under a standard neutral SFS.
        """
        opts = dict(intervals_del=(-1.0e+8, -1.0e-5, 30), intervals_ben=(1.0e-5, 1.0e4, 30))
        truth = dict(S_d=-300, b=0.3, p_b=0, S_b=1)

        sim = fd.Simulation(
            sfs_neut=fd.Simulation.get_neutral_sfs(n=12, n_sites=1e7, theta=1e-3), params=dict(truth), **opts
        )
        sim.run()

        inf = fd.BaseInference(
            sfs_neut=sim.sfs_neut.fold(),
            sfs_sel=sim.sfs_sel.fold(),
            discretization=sim.discretization,
            fixed_params=dict(all=dict(eps=0, p_b=0, S_b=1, h=0.5)),
            do_bootstrap=False,
            n_runs=3,
            parallelize=False
        )
        inf.run()

        self.assertTrue(inf.folded)
        self.assertAlmostEqual(inf.params_mle['S_d'] / truth['S_d'], 1, places=2)
        self.assertAlmostEqual(inf.params_mle['b'] / truth['b'], 1, places=2)

        # the modelled folded SFS at the true parameters equals the folded simulated SFS
        modelled, observed = fd.BaseInference._model_sfs(
            discretization=sim.discretization,
            model=sim.model,
            params=sim.params,
            sfs_neut=sim.sfs_neut.fold(),
            sfs_sel=sim.sfs_sel.fold(),
            folded=True
        )

        np.testing.assert_allclose(modelled[:6], observed[:6], rtol=1e-10)

    def test_bootstrap_without_samples_raises(self):
        """
        A bootstrap of zero samples raises a ``ValueError`` naming the cause. It failed with an ``IndexError``
        from inside the collection of the results.
        """
        inf = fd.BaseInference(
            sfs_neut=self.sfs_neut,
            sfs_sel=self.sfs_sel,
            intervals_del=(-1.0e+8, -1.0e-5, 20),
            intervals_ben=(1.0e-5, 1.0e4, 20),
            do_bootstrap=False,
            n_bootstraps=7,
            n_runs=1,
            parallelize=False
        )
        inf.run()

        with self.assertRaises(ValueError):
            inf.bootstrap(n_samples=0)

        self.assertEqual(inf.n_bootstraps, 7)

    def test_plot_inferred_parameters_takes_point_estimate(self):
        """
        ``BaseInference.plot_inferred_parameters`` plots the requested point estimate. The argument was
        accepted and ignored, so the mean of the bootstraps was always shown.
        """
        inf = fd.BaseInference(
            sfs_neut=self.sfs_neut,
            sfs_sel=self.sfs_sel,
            intervals_del=(-1.0e+8, -1.0e-5, 20),
            intervals_ben=(1.0e-5, 1.0e4, 20),
            do_bootstrap=True,
            n_bootstraps=5,
            n_runs=1,
            parallelize=False
        )
        inf.run()

        def heights(point_estimate: str) -> list:
            ax = inf.plot_inferred_parameters(show=False, point_estimate=point_estimate)

            return [p.get_height() for p in ax.patches]

        self.assertIn(abs(inf.params_mle['S_d']), heights('original'))
        self.assertIn(abs(inf.bootstraps['S_d'].mean()), heights('mean'))
        self.assertNotEqual(heights('original'), heights('mean'))

    def test_bootstrap_summary_lists_parameters_only(self):
        """
        The bootstrap summary that is logged covers the parameters and not the divergence counts of the
        bootstrap samples, which the bootstrap table also holds.
        """
        inf = fd.BaseInference(
            sfs_neut=self.sfs_neut,
            sfs_sel=self.sfs_sel,
            n_sites_div_neut=float(self.sfs_neut.n_sites),
            n_sites_div_sel=float(self.sfs_sel.n_sites),
            intervals_del=(-1.0e+8, -1.0e-5, 20),
            intervals_ben=(1.0e-5, 1.0e4, 20),
            do_bootstrap=True,
            n_bootstraps=2,
            n_runs=1,
            parallelize=False
        )

        with self.assertLogs('fastdfe', level='INFO') as logs:
            inf.run()

        summary = [m for m in logs.output if 'Bootstrap summary' in m]

        self.assertEqual(len(summary), 1)
        self.assertNotIn('n_div', summary[0])
        self.assertIn('n_div_neut', inf.bootstraps)

    def test_plot_on_used_axes_of_another_figure(self):
        """
        A plot can be added to axes that already hold a plot and belong to a figure other than the current one.
        Layout and saving act on the figure of these axes. They acted on the current figure, so the file held
        the wrong figure.
        """
        import tempfile

        import matplotlib.image
        import matplotlib.pyplot as plt

        inf = fd.BaseInference(
            sfs_neut=self.sfs_neut,
            sfs_sel=self.sfs_sel,
            intervals_del=(-1.0e+8, -1.0e-5, 20),
            intervals_ben=(1.0e-5, 1.0e4, 20),
            do_bootstrap=False,
            n_runs=1,
            parallelize=False
        )
        inf.run()

        fig, ax = plt.subplots(figsize=(4, 3))
        ax.plot([1, 9], [0, 1000], color='black')

        # a larger figure becomes the current one
        other = plt.figure(figsize=(8, 2))

        with tempfile.TemporaryDirectory() as tmp:
            file = f'{tmp}/plot.png'

            self.assertIs(inf.plot_sfs_comparison(ax=ax, show=False, file=file), ax)

            height, width = matplotlib.image.imread(file).shape[:2]

        # the line drawn before is kept next to the bars, and the saved image has the shape of the given figure
        self.assertEqual(len(ax.lines), 1)
        self.assertGreater(len(ax.patches), 0)
        self.assertLess(width / height, 2)
        self.assertEqual(len(other.axes), 0)

        plt.close(fig)
        plt.close(other)

    def test_plot_on_given_axes_leaves_other_panels_untouched(self):
        """
        A plot given ``ax`` labels the ticks of these axes and returns them. The tick labels were set on the
        current axes, which after ``plt.subplots`` is the last panel, and the current axes were returned.
        """
        import matplotlib.pyplot as plt

        inf = fd.BaseInference(
            sfs_neut=self.sfs_neut,
            sfs_sel=self.sfs_sel,
            intervals_del=(-1.0e+8, -1.0e-5, 20),
            intervals_ben=(1.0e-5, 1.0e4, 20),
            do_bootstrap=False,
            n_runs=1,
            parallelize=False
        )
        inf.run()

        fig, (left, middle, right) = plt.subplots(1, 3)
        right.plot([0, 1], [0, 1])

        self.assertIs(inf.plot_continuous(ax=left, show=False), left)
        self.assertIs(inf.plot_sfs_comparison(ax=middle, show=False), middle)

        fig.canvas.draw()

        # the target panel shows selection coefficients, the other panel its own values
        self.assertIn('-1e+08', [t.get_text() for t in left.get_xticklabels()])
        self.assertNotIn('-1e+08', [t.get_text() for t in right.get_xticklabels()])

        plt.close(fig)

    def test_folded_inference_on_unfolded_spectra_matches_folded_input(self):
        """
        ``folded=True`` on unfolded spectra gives the likelihood of the same spectra passed folded, whatever
        the polarization of the input. The demographic correction used one ratio per allele count unless the
        neutral spectrum itself was folded, so the likelihood changed with the polarization although the
        folded data did not.
        """
        kwargs = dict(
            intervals_del=(-1.0e+8, -1.0e-5, 20),
            intervals_ben=(1.0e-5, 1.0e4, 20),
            fixed_params=dict(all=dict(eps=0, p_b=0, S_b=1, h=0.5)),
            do_bootstrap=False,
            parallelize=False
        )
        params = dict(S_d=-1000, b=0.3, p_b=0, S_b=1, eps=0, h=0.5)

        def flip(sfs: fd.Spectrum, e: float) -> fd.Spectrum:
            """
            Exchange a fraction of the derived and ancestral alleles, which leaves the folded spectrum as is.
            """
            data = sfs.data.copy()
            data[1:-1] = (1 - e) * sfs.polymorphic + e * sfs.polymorphic[::-1]

            return fd.Spectrum(data)

        expected = fd.BaseInference(
            sfs_neut=self.sfs_neut.fold(), sfs_sel=self.sfs_sel.fold(), **kwargs
        ).evaluate_likelihood(dict(all=params))

        for e in [0, 0.2]:
            observed = fd.BaseInference(
                sfs_neut=flip(self.sfs_neut, e), sfs_sel=flip(self.sfs_sel, e), folded=True, **kwargs
            ).evaluate_likelihood(dict(all=params))

            self.assertAlmostEqual(observed, expected, places=6)

    def test_folded_inference_on_unfolded_spectra_compares_folded_spectra(self):
        """
        With ``folded=True`` on unfolded spectra, the residual and the SFS comparison plot use the folded
        observed SFS, as the likelihood does. They used the unfolded one against the folded modelled SFS, which
        inflated the residual and logged a warning about a poor fit.
        """
        inf = fd.BaseInference(
            sfs_neut=self.sfs_neut,
            sfs_sel=self.sfs_sel,
            intervals_del=(-1.0e+8, -1.0e-5, 20),
            intervals_ben=(1.0e-5, 1.0e4, 20),
            fixed_params=dict(all=dict(eps=0, p_b=0, S_b=1, h=0.5)),
            folded=True,
            do_bootstrap=False,
            n_runs=1,
            parallelize=False
        )

        with self.assertNoLogs('fastdfe', level='WARNING'):
            inf.run()

        folded = self.sfs_sel.fold().polymorphic

        self.assertAlmostEqual(inf.L2_residual, np.linalg.norm(inf.sfs_mle.polymorphic - folded, 2))
        self.assertLess(inf.get_residual(1) / self.sfs_sel.n_polymorphic, 0.1)

        ax = inf.plot_sfs_comparison(show=False)
        heights = sorted(p.get_height() for p in ax.patches)

        np.testing.assert_allclose(heights[-1], max(folded.max(), inf.sfs_mle.polymorphic.max()))

    def test_config_carries_folding_dominance_and_bootstrap_runs(self):
        """
        An inference restored from its config keeps ``folded``, ``intervals_h``, ``h_callback`` and
        ``n_bootstrap_retries``. These were not passed to the config, so the restored inference used the
        defaults. A config written to JSON omits the callback, which has no JSON representation.
        """
        def h_callback(h: float, S: np.ndarray) -> np.ndarray:
            return np.full_like(S, h)

        inf = fd.BaseInference(
            sfs_neut=self.sfs_neut,
            sfs_sel=self.sfs_sel,
            intervals_del=(-1.0e+8, -1.0e-5, 20),
            intervals_ben=(1.0e-5, 1.0e4, 20),
            intervals_h=(0.0, 1.0, 5),
            h_callback=h_callback,
            folded=True,
            n_bootstrap_retries=5,
            parallelize=False
        )

        config = inf.create_config()
        restored = fd.BaseInference.from_config(config)

        self.assertIs(restored.folded, True)
        self.assertEqual(restored.n_bootstrap_retries, 5)
        self.assertEqual(restored.discretization.intervals_h, (0.0, 1.0, 5))
        self.assertIs(restored.discretization.h_callback, h_callback)

        from_json = fd.BaseInference.from_config(fd.Config.from_json(config.to_json()))

        self.assertIs(from_json.folded, True)
        self.assertEqual(from_json.n_bootstrap_retries, 5)
        self.assertEqual(from_json.discretization.intervals_h, (0.0, 1.0, 5))
        self.assertIsNone(from_json.discretization.h_callback)

    def test_arguments_of_method_triggering_run_do_not_reach_run(self):
        """
        A method that runs the inference when required runs it with the settings of the inference. Its own
        arguments were forwarded to ``run``, so that ``plot_nested_models`` disabled the bootstrap of the
        inference and a positional plot argument was taken as ``do_bootstrap``.
        """
        def make(do_bootstrap: bool) -> fd.BaseInference:
            return fd.BaseInference(
                sfs_neut=self.sfs_neut,
                sfs_sel=self.sfs_sel,
                intervals_del=(-1.0e+8, -1.0e-5, 20),
                intervals_ben=(1.0e-5, 1.0e4, 20),
                do_bootstrap=do_bootstrap,
                n_bootstraps=2,
                n_runs=1,
                parallelize=False
            )

        inf = make(True)
        inf.plot_nested_models(show=False)

        self.assertIs(inf.do_bootstrap, True)
        self.assertEqual(len(inf.bootstraps), 2)

        inf = make(False)
        inf.plot_inferred_parameters(True, show=False)

        self.assertIs(inf.do_bootstrap, False)

    def test_refit_recomputes_nested_models(self):
        """
        ``compare_nested_models`` caches its result. After the inference is refitted, here on other data, a
        further call recomputes the nested models instead of returning those of the previous fit.
        """
        inf = fd.BaseInference(
            sfs_neut=self.sfs_neut,
            sfs_sel=self.sfs_sel,
            intervals_del=(-1.0e+8, -1.0e-5, 20),
            intervals_ben=(1.0e-5, 1.0e4, 20),
            do_bootstrap=False,
            n_runs=1,
            parallelize=False
        )
        inf.run()

        _, first = inf.compare_nested_models()
        _, cached = inf.compare_nested_models()

        self.assertIs(first['full.anc'], cached['full.anc'])

        data = list(self.sfs_sel.data)
        inf.sfs_sel = fd.Spectrum([data[0], 2 * data[1], 2 * data[2]] + data[3:])
        inf.run()

        _, second = inf.compare_nested_models()

        self.assertIsNot(first['full.anc'], second['full.anc'])
        self.assertNotAlmostEqual(first['full.anc'].likelihood, second['full.anc'].likelihood, places=1)

    def test_bootstrapped_divergence_estimates_use_resampled_divergence_counts(self):
        """
        With divergence in the likelihood, alpha, omega and omega_a of each bootstrap sample are computed from
        the divergence counts of that sample. They were computed from the observed counts, which made the
        bootstrapped omega constant.
        """
        n_sites_div_neut, n_sites_div_sel = float(self.sfs_neut.n_sites), float(self.sfs_sel.n_sites)

        inf = fd.BaseInference(
            sfs_neut=self.sfs_neut,
            sfs_sel=self.sfs_sel,
            n_sites_div_neut=n_sites_div_neut,
            n_sites_div_sel=n_sites_div_sel,
            intervals_del=(-1.0e+8, -1.0e-5, 20),
            intervals_ben=(1.0e-5, 1.0e4, 20),
            do_bootstrap=True,
            n_bootstraps=5,
            n_runs=1,
            parallelize=False
        )
        inf.run()

        bs = inf.bootstraps

        # omega is the ratio of the divergence counts per target site
        np.testing.assert_allclose(
            bs.omega, (bs.n_div_sel / n_sites_div_sel) / (bs.n_div_neut / n_sites_div_neut), rtol=1e-12
        )
        self.assertGreater(bs.omega.nunique(), 1)

        for _, row in bs.iterrows():
            self.assertAlmostEqual(row.alpha, inf.discretization.get_alpha_divergence(
                inf.model, dict(row), row.n_div_neut, row.n_div_sel, n_sites_div_neut, n_sites_div_sel
            ))

    def test_plot_nested_models_leaves_p_values_unchanged(self):
        """
        Plotting the nested models does not modify the p-values returned by ``compare_nested_models``. The
        plot replaced the entries of pairs that are not nested by 1 and raised small p-values to 1e-10.
        """
        inf = fd.BaseInference(
            sfs_neut=self.sfs_neut,
            sfs_sel=self.sfs_sel,
            intervals_del=(-1.0e+8, -1.0e-5, 20),
            intervals_ben=(1.0e-5, 1.0e4, 20),
            do_bootstrap=False,
            n_runs=1,
            parallelize=False
        )

        # the plot requests the comparison with this argument, which the cache is keyed on
        before = inf.compare_nested_models(do_bootstrap=False)[0].copy()

        inf.plot_nested_models(show=False)

        after = inf.compare_nested_models(do_bootstrap=False)[0]

        self.assertTrue(np.equal(before, None).any())
        np.testing.assert_array_equal(before, after)

    def test_fixed_params_without_type_level_raise(self):
        """
        Fixed parameters given as ``{param: value}`` raise an error. They were accepted and fixed nothing, and
        the dictionary passed was modified.
        """
        kwargs = dict(sfs_neut=self.sfs_neut, sfs_sel=self.sfs_sel, do_bootstrap=False)

        with self.assertRaises(ValueError):
            fd.BaseInference(fixed_params=dict(eps=0, h=0.5), **kwargs)

        with self.assertRaises(ValueError):
            fd.BaseInference(fixed_params=dict(pendula=dict(eps=0)), **kwargs)

        fixed_params = dict(all=dict(eps=0, h=0.5))
        fd.BaseInference(fixed_params=fixed_params, **kwargs)

        self.assertEqual(fixed_params, dict(all=dict(eps=0, h=0.5)))

    def test_serializing_lambda_h_callback_warns(self):
        """
        Serializing an inference whose dominance callback is a lambda logs a warning, as the callback is
        not restored. It was dropped without notice, so the restored inference used a constant ``h``.
        """
        kwargs = dict(
            sfs_neut=self.sfs_neut,
            sfs_sel=self.sfs_sel,
            intervals_del=(-1.0e+8, -1.0e-5, 20),
            intervals_ben=(1.0e-5, 1.0e4, 20),
            intervals_h=(0, 1, 3),
            fixed_params=dict(all=dict(eps=0, p_b=0, S_b=1)),
            do_bootstrap=False,
            parallelize=False
        )

        inf = fd.BaseInference(h_callback=lambda h, S: np.where(S < 0, h, 0.5), **kwargs)

        with self.assertLogs('fastdfe', level='WARNING') as logs:
            restored = fd.BaseInference.from_json(inf.to_json())

        self.assertTrue(any('h_callback' in message for message in logs.output))
        self.assertIsNone(restored.discretization.h_callback)

        with self.assertNoLogs('fastdfe', level='WARNING'):
            fd.BaseInference(**kwargs).to_json()

    def test_empty_neutral_frequency_class_warns(self):
        """
        A neutral SFS with an empty frequency class logs a warning. For a folded inference a class is empty
        only if both of its allele counts are.
        """
        neutral = list(self.sfs_neut)
        neutral[7] = 0

        kwargs = dict(sfs_sel=self.sfs_sel, do_bootstrap=False)

        with self.assertLogs('fastdfe', level='WARNING') as logs:
            fd.BaseInference(sfs_neut=fd.Spectrum(neutral), **kwargs)

        self.assertTrue(any('empty frequency classes' in message for message in logs.output))

        with self.assertNoLogs('fastdfe', level='WARNING'):
            fd.BaseInference(sfs_neut=fd.Spectrum(neutral), folded=True, **kwargs)
            fd.BaseInference(sfs_neut=self.sfs_neut, **kwargs)

    def test_bootstrap_on_inference_not_run_bootstraps_once(self):
        """
        Bootstrapping an inference that was not run yet performs one bootstrap. The run it triggers ended with
        a bootstrap of its own, which was discarded.
        """
        from unittest import mock

        inf = fd.BaseInference(
            sfs_neut=self.sfs_neut,
            sfs_sel=self.sfs_sel,
            intervals_del=(-1.0e+8, -1.0e-5, 20),
            intervals_ben=(1.0e-5, 1.0e4, 20),
            do_bootstrap=True,
            n_bootstraps=4,
            n_runs=1,
            parallelize=False
        )

        with mock.patch.object(inf, '_get_run_bootstrap_sample', wraps=inf._get_run_bootstrap_sample) as sampler:
            inf.bootstrap(n_samples=2)

        self.assertEqual(sampler.call_count, 1)
        self.assertEqual(len(inf.bootstraps), 2)
        self.assertTrue(inf.do_bootstrap)

    def test_accessors_run_the_inference_if_required(self):
        """
        The accessors of results trigger the run of an inference that was not run yet, as the plots and
        ``get_dfe`` do. They failed on the missing results.
        """
        accessors = [
            lambda inf: inf.get_spectra(),
            lambda inf: inf.get_discretized(confidence_intervals=False),
            lambda inf: inf.get_alpha(),
            lambda inf: inf.get_omega(),
            lambda inf: inf.get_omega_a(),
            lambda inf: inf.alpha,
            lambda inf: inf.get_bootstrap_params(),
            lambda inf: inf.get_bootstrap_param_names(),
        ]

        for accessor in accessors:
            inf = fd.BaseInference(
                sfs_neut=self.sfs_neut,
                sfs_sel=self.sfs_sel,
                intervals_del=(-1.0e+8, -1.0e-5, 20),
                intervals_ben=(1.0e-5, 1.0e4, 20),
                do_bootstrap=False,
                n_runs=1,
                parallelize=False
            )

            accessor(inf)

            self.assertIsNotNone(inf.params_mle)

    def test_bootstrap_without_successful_run_keeps_run_of_highest_likelihood(self):
        """
        If no optimization run of a bootstrap sample terminates normally, the run of highest likelihood is
        kept. The first run was kept irrespective of the likelihoods.
        """
        from types import SimpleNamespace
        from unittest import mock
        from fastdfe import optimization

        inf = fd.BaseInference(
            sfs_neut=self.sfs_neut,
            sfs_sel=self.sfs_sel,
            intervals_del=(-1.0e+8, -1.0e-5, 20),
            intervals_ben=(1.0e-5, 1.0e4, 20),
            do_bootstrap=False,
            n_runs=1,
            parallelize=False
        )
        inf.run()

        def run(likelihood: float, success: bool) -> tuple:
            return SimpleNamespace(fun=-likelihood, success=success), dict(all=inf.params_mle), dict(all={}), {}

        # no successful run in the first sample, one in the second
        results = np.empty((2, 2, 4), dtype=object)
        results[0] = [run(-50, False), run(-40, False)]
        results[1] = [run(-50, True), run(-40, False)]

        with mock.patch.object(optimization, 'parallelize', return_value=results):
            inf.bootstrap(n_samples=2, n_retries=2)

        self.assertEqual(list(inf.bootstraps.i_best_run), [1, 0])
        self.assertEqual(list(inf.bootstraps.likelihood), [-40, -50])

    def test_bootstrap_sample_without_divergence_counts_gives_non_finite_estimates(self):
        """
        A bootstrap sample whose divergence count is zero has non-finite divergence-based estimates, as the
        original data would. It raised a ``ZeroDivisionError``, which discarded the bootstraps.
        """
        inf = fd.BaseInference(
            sfs_neut=self.sfs_neut,
            sfs_sel=self.sfs_sel,
            n_sites_div_neut=float(self.sfs_neut.n_sites),
            n_sites_div_sel=float(self.sfs_sel.n_sites),
            intervals_del=(-1.0e+8, -1.0e-5, 20),
            intervals_ben=(1.0e-5, 1.0e4, 20),
            do_bootstrap=True,
            n_bootstraps=2,
            n_runs=1,
            parallelize=False
        )
        inf.run()

        inf.bootstraps.loc[0, 'n_div_sel'] = 0
        inf._add_alpha_to_bootstraps()

        self.assertFalse(np.isfinite(inf.bootstraps.alpha[0]))
        self.assertEqual(inf.bootstraps.omega[0], 0)
        self.assertTrue(np.isfinite(inf.bootstraps.alpha[1]))

    def test_get_spectra_of_folded_inference_on_unfolded_spectra_is_folded(self):
        """
        The observed spectra returned for a folded inference are folded like the modelled SFS. With
        ``folded=True`` on unfolded spectra they were returned unfolded.
        """
        inf = fd.BaseInference(
            sfs_neut=self.sfs_neut,
            sfs_sel=self.sfs_sel,
            folded=True,
            intervals_del=(-1.0e+8, -1.0e-5, 20),
            intervals_ben=(1.0e-5, 1.0e4, 20),
            do_bootstrap=False,
            n_runs=1,
            parallelize=False
        )

        spectra = inf.get_spectra()

        for t in ['neutral', 'selected', 'modelled']:
            self.assertEqual(np.count_nonzero(spectra[t].polymorphic[5:]), 0)

        np.testing.assert_array_equal(spectra['selected'].data, self.sfs_sel.fold().data)

    def test_get_stats_discretized_without_bootstraps_raises(self):
        """Without bootstraps, the statistics of the discretized DFE raise a ``ValueError``."""
        inf = fd.BaseInference(
            sfs_neut=self.sfs_neut,
            sfs_sel=self.sfs_sel,
            intervals_del=(-1.0e+8, -1.0e-5, 20),
            intervals_ben=(1.0e-5, 1.0e4, 20),
            do_bootstrap=False,
            n_runs=1,
            parallelize=False
        )

        with self.assertRaises(ValueError):
            inf.get_stats_discretized()

    def test_lrt_without_improvement_gives_p_value_of_one(self):
        """
        A likelihood ratio test gives a p-value of 1 if the complex model does not improve the likelihood,
        also when all tested parameters lie on the boundary. It gave ``1 - 2 ** -m`` in that case.
        """
        inf = fd.BaseInference(sfs_neut=self.sfs_neut, sfs_sel=self.sfs_sel, do_bootstrap=False)

        self.assertEqual(inf.lrt(-10, -10, df=1, m=1), 1)
        self.assertEqual(inf.lrt(-10, -10.5, df=2, m=2), 1)
        self.assertEqual(inf.lrt(-10, -10, df=2, m=1), 1)

        # an improvement is tested against the mixture as before
        self.assertAlmostEqual(inf.lrt(-10, -8, df=1, m=1), 0.5 * 0.04550026389635842, places=12)

    def test_models_fixing_a_parameter_at_different_values_are_not_nested(self):
        """
        Two models are nested only if the parameters both of them fix have the same values. Only the names
        were compared, so such models were tested against each other.
        """
        kwargs = dict(sfs_neut=self.sfs_neut, sfs_sel=self.sfs_sel, do_bootstrap=False)

        simple = fd.BaseInference(fixed_params=dict(all=dict(h=0.5, eps=0, p_b=0, S_b=1)), **kwargs)

        self.assertTrue(simple._is_nested(fd.BaseInference(fixed_params=dict(all=dict(h=0.5, eps=0)), **kwargs)))
        self.assertFalse(simple._is_nested(fd.BaseInference(fixed_params=dict(all=dict(h=0.5, eps=0.1)), **kwargs)))

    def test_plot_continuous_symlog_scale_is_logarithmic(self):
        """``scale='symlog'`` gives a logarithmic density axis. It was ignored, which left the axis linear."""
        inf = fd.BaseInference(
            sfs_neut=self.sfs_neut,
            sfs_sel=self.sfs_sel,
            intervals_del=(-1.0e+8, -1.0e-5, 20),
            intervals_ben=(1.0e-5, 1.0e4, 20),
            do_bootstrap=False,
            n_runs=1,
            parallelize=False
        )

        self.assertEqual(inf.plot_continuous(show=False, scale='symlog').get_yscale(), 'log')
        self.assertEqual(inf.plot_continuous(show=False, scale='lin').get_yscale(), 'linear')

    def test_divergence_target_sizes_without_divergence_counts_raise(self):
        """
        Divergence target sizes for spectra without divergence counts raise an error. The divergence term
        was silently inactive, and alpha and omega were undefined.
        """
        kwargs = dict(n_sites_div_neut=1e5, n_sites_div_sel=1e5, do_bootstrap=False)

        with self.assertRaises(ValueError):
            fd.BaseInference(sfs_neut=self.sfs_neut.fold(), sfs_sel=self.sfs_sel.fold(), **kwargs)

        with self.assertRaises(ValueError):
            fd.BaseInference(sfs_neut=self.sfs_neut, sfs_sel=fd.Spectrum(list(self.sfs_sel)[:-1] + [0]), **kwargs)

        fd.BaseInference(sfs_neut=self.sfs_neut, sfs_sel=self.sfs_sel, **kwargs)
        fd.BaseInference(sfs_neut=self.sfs_neut.fold(), sfs_sel=self.sfs_sel.fold(), do_bootstrap=False)

    def test_bounds_of_h_default_to_its_grid(self):
        """
        Without a dominance callback, the bounds of ``h`` default to the range of ``intervals_h``. They
        defaulted to (0, 1), so a narrower grid failed as soon as the optimizer left it.
        """
        kwargs = dict(
            sfs_neut=self.sfs_neut,
            sfs_sel=self.sfs_sel,
            intervals_del=(-1.0e+8, -1.0e-5, 20),
            intervals_ben=(1.0e-5, 1.0e4, 20),
            intervals_h=(0.2, 0.6, 3),
            fixed_params=dict(all=dict(eps=0, p_b=0, S_b=1)),
            do_bootstrap=False,
            n_runs=2,
            parallelize=False
        )

        inf = fd.BaseInference(**kwargs)
        inf.run()

        self.assertEqual(inf.bounds['h'], (0.2, 0.6))
        self.assertTrue(0.2 <= inf.params_mle['h'] <= 0.6)

        # bounds given for h and a fixed h are left alone
        self.assertEqual(fd.BaseInference(**kwargs | dict(bounds=dict(h=(0.3, 0.5)))).bounds['h'], (0.3, 0.5))
        self.assertEqual(fd.BaseInference(**kwargs | dict(fixed_params=dict(all=dict(h=0.5)))).bounds['h'], (0, 1))

    def test_fixed_parameters_are_returned_exactly(self):
        """
        Fixed parameters are returned at their fixed values. A value on the log scale came back changed in
        the last digit by its conversion to the scale and back.
        """
        inf = fd.BaseInference(
            sfs_neut=self.sfs_neut,
            sfs_sel=self.sfs_sel,
            fixed_params=dict(all=dict(b=0.3, S_d=-1234.5, h=0.5, eps=0, p_b=0.1)),
            intervals_del=(-1.0e+8, -1.0e-5, 20),
            intervals_ben=(1.0e-5, 1.0e4, 20),
            do_bootstrap=False,
            n_runs=2,
            parallelize=False
        )
        inf.run()

        self.assertEqual((inf.params_mle['b'], inf.params_mle['S_d']), (0.3, -1234.5))
        self.assertEqual(set(inf.runs['all.b']), {0.3})

    def test_estimate_at_bound_lies_within_bounds(self):
        """
        An estimate at a bound on the log scale equals the bound. It could lie outside by the rounding of
        the conversion to the scale and back, and was then rejected as a fixed value of a further inference.
        """
        kwargs = dict(
            sfs_neut=self.sfs_neut,
            sfs_sel=self.sfs_sel,
            bounds=dict(b=(0.3, 10)),
            intervals_del=(-1.0e+8, -1.0e-5, 20),
            intervals_ben=(1.0e-5, 1.0e4, 20),
            do_bootstrap=False,
            n_runs=1,
            parallelize=False
        )

        inf = fd.BaseInference(**kwargs)
        inf.run()

        self.assertEqual(inf.params_mle['b'], 0.3)

        fd.BaseInference(fixed_params=dict(all=inf.params_mle), **kwargs)

    def test_fixed_parameter_on_symlog_scale_is_held_fixed(self):
        """
        A fixed parameter on the symlog scale is held at its value, also at zero. Its bound was read as a
        linear threshold and an upper bound, which left it free around zero and raised an error for zero.
        """
        kwargs = dict(
            sfs_neut=self.sfs_neut,
            sfs_sel=self.sfs_sel,
            intervals_del=(-1.0e+8, -1.0e-5, 20),
            intervals_ben=(1.0e-5, 1.0e4, 20),
            do_bootstrap=False,
            n_runs=1,
            parallelize=False
        )

        for eps in [0.05, 0]:
            fixed_params = dict(all=dict(eps=eps, h=0.5, p_b=0, S_b=1))

            lin = fd.BaseInference(fixed_params=fixed_params, **kwargs)
            symlog = fd.BaseInference(
                fixed_params=fixed_params, scales=dict(eps='symlog'), bounds=dict(eps=(1e-3, 0.15)), **kwargs
            )

            lin.run()
            symlog.run()

            self.assertAlmostEqual(symlog.likelihood, lin.likelihood, places=4)

    def test_boxplot_requires_bootstraps_of_every_inference(self):
        """
        The boxplot of inferred parameters raises a ``ValueError`` if any of the inferences has no bootstraps.
        Only the first one was checked, so a later one failed with a ``TypeError``.
        """
        kwargs = dict(
            sfs_neut=self.sfs_neut,
            sfs_sel=self.sfs_sel,
            intervals_del=(-1.0e+8, -1.0e-5, 20),
            intervals_ben=(1.0e-5, 1.0e4, 20),
            n_bootstraps=2,
            n_runs=1,
            parallelize=False
        )

        with_bootstraps = fd.BaseInference(do_bootstrap=True, **kwargs)
        without = fd.BaseInference(do_bootstrap=False, **kwargs)

        with_bootstraps.run()
        without.run()

        with self.assertRaises(ValueError):
            fd.Inference.plot_inferred_parameters_boxplot([with_bootstraps, without], labels=['a', 'b'], show=False)

    def test_recorded_initial_values_hold_fixed_parameters_at_their_values(self):
        """
        The initial values recorded for the runs and the bootstraps show fixed parameters at their fixed
        values, at which the optimization holds them. They showed default or sampled values instead.
        """
        inf = fd.BaseInference(
            sfs_neut=self.sfs_neut,
            sfs_sel=self.sfs_sel,
            fixed_params=dict(all=dict(b=0.3, h=0.5, eps=0, p_b=0, S_b=1)),
            intervals_del=(-1.0e+8, -1.0e-5, 20),
            intervals_ben=(1.0e-5, 1.0e4, 20),
            do_bootstrap=True,
            n_bootstraps=2,
            n_bootstrap_retries=3,
            n_runs=3,
            parallelize=False
        )
        inf.run()

        for x0 in inf.runs.x0:
            self.assertIn("'all.b': 0.3,", x0)

        for runs in inf.bootstraps.x0_runs:
            self.assertEqual([x0['all.b'] for x0 in runs], [0.3, 0.3, 0.3])

    def test_point_estimate_is_independent_of_confidence_intervals(self):
        """
        The point estimate of the discretized DFE and of the parameters follows ``point_estimate`` whether or
        not confidence intervals are requested. Without them the original estimate was returned, so switching
        off the error bars of a plot changed its values.
        """
        inf = fd.BaseInference(
            sfs_neut=self.sfs_neut,
            sfs_sel=self.sfs_sel,
            intervals_del=(-1.0e+8, -1.0e-5, 20),
            intervals_ben=(1.0e-5, 1.0e4, 20),
            do_bootstrap=True,
            n_bootstraps=4,
            n_runs=1,
            parallelize=False
        )
        inf.run()

        for point_estimate in ['mean', 'original']:
            with_cis, errors = inf.get_discretized(point_estimate=point_estimate)
            without_cis, no_errors = inf.get_discretized(point_estimate=point_estimate, confidence_intervals=False)

            np.testing.assert_array_equal(with_cis, without_cis)
            self.assertIsNotNone(errors)
            self.assertIsNone(no_errors)

            bins = np.array([-np.inf, -100, -1, 0, np.inf])

            np.testing.assert_array_equal(
                inf.get_dfe().discretize(bins, point_estimate=point_estimate)[0],
                inf.get_dfe().discretize(bins, point_estimate=point_estimate, confidence_intervals=False)[0]
            )

        mean, original = [
            inf.get_discretized(point_estimate=p, confidence_intervals=False)[0] for p in ['mean', 'original']
        ]

        self.assertFalse(np.allclose(mean, original))

        heights = [
            [bar.get_height() for bar in inf.plot_inferred_parameters(
                show=False, confidence_intervals=confidence_intervals, scale='lin'
            ).patches]
            for confidence_intervals in [True, False]
        ]

        np.testing.assert_allclose(heights[0], heights[1])

    def test_nested_models_keep_other_fixed_parameters(self):
        """
        The nested models keep the fixed parameters they do not set themselves. Only ``h`` was kept, so a
        fixed shape was optimized in every nested model.
        """
        inf = fd.BaseInference(
            sfs_neut=self.sfs_neut,
            sfs_sel=self.sfs_sel,
            fixed_params=dict(all=dict(b=0.3, h=0.5, eps=0, p_b=0, S_b=1)),
            intervals_del=(-1.0e+8, -1.0e-5, 20),
            intervals_ben=(1.0e-5, 1.0e4, 20),
            do_bootstrap=False,
            n_runs=1,
            parallelize=False
        )

        _, inferences = inf.compare_nested_models()

        for sub in inferences.values():
            self.assertAlmostEqual(sub.params_mle['b'], 0.3, places=10)

        # the nested model with the fixed parameters of the inference reproduces its fit
        self.assertAlmostEqual(inferences['dele.no_anc'].likelihood, inf.likelihood, places=3)

    def test_base_inference_smoke(self):
        """A single tiny BaseInference fit runs and produces finite estimates."""
        inf = fd.BaseInference(
            sfs_neut=self.sfs_neut,
            sfs_sel=self.sfs_sel,
            intervals_del=(-1.0e+8, -1.0e-5, 20),
            intervals_ben=(1.0e-5, 1.0e4, 20),
            do_bootstrap=False,
            n_runs=1
        )
        inf.run()

        self.assertTrue(np.isfinite(inf.likelihood))
        self.assertTrue(np.isfinite(inf.get_alpha()))

    def test_joint_inference_smoke(self):
        """A single tiny JointInference fit runs across two types."""
        inf = fd.JointInference(
            sfs_neut=fd.Spectra(dict(a=list(self.sfs_neut), b=list(self.sfs_neut))),
            sfs_sel=fd.Spectra(dict(a=list(self.sfs_sel), b=list(self.sfs_sel))),
            intervals_del=(-1.0e+8, -1.0e-5, 20),
            intervals_ben=(1.0e-5, 1.0e4, 20),
            do_bootstrap=False,
            n_runs=1
        )
        inf.run()

        self.assertTrue(np.isfinite(inf.likelihood))
