"""
Test the Simulation class.
"""
import re
from itertools import product

import numpy as np
import pytest

import fastdfe as fd
from testing import TestCase


def expand(template, **wildcards):
    """
    Generate combinations of strings with placeholders.
    """
    keys, values = zip(*wildcards.items())
    combinations = product(*values)

    return [template.format(**dict(zip(keys, combo))) for combo in combinations]


@pytest.mark.inference
class SLiMTestCase(TestCase):
    """
    Test results against cached SLiM results.
    """

    # configs for testing against cached slim results
    configs_slim = expand(  # deleterious DFE
        "testing/cache/slim/n_replicate=1/n_chunks={n_chunks}/g=1e4/L={L}/mu={mu}/r=1e-7/N=1e3/s_b={s_b}/b={b}/s_d={s_d}/p_b={p_b}/n={n}/{folded}/sfs.csv",
        n_chunks=[100],
        L=["1e7"],
        mu=["1e-8"],
        s_b=["1e-9"],
        b=[3, 1, 0.3],
        s_d=["3e-3", "3e-2", "3e-1"],
        p_b=[0],
        folded=["folded", "unfolded"],
        n=[20, 100]
    ) + expand(  # full DFE
        "testing/cache/slim/n_replicate=1/n_chunks={n_chunks}/g=1e4/L={L}/mu={mu}/r=1e-7/N=1e3/s_b={s_b}/b={b}/s_d={s_d}/p_b={p_b}/n={n}/{folded}/sfs.csv",
        n_chunks=[40],
        L=["1e7"],
        mu=["1e-8"],
        s_b=["1e-4", "1e-3"],
        b=[0.4],
        s_d=["1e-1", "1e0"],
        p_b=[0.01, 0.05],
        folded=["folded", "unfolded"],
        n=[20, 100]
    ) + expand(
        "testing/cache/slim/n_replicate=1/n_chunks=100/g=1e4/L=1e7/mu=1e-8/r=1e-7/N=1e3/{params}/n={n}/dominance_{h}/unfolded/sfs.csv",
        h=np.round(np.linspace(0.1, 1, 10), 1),
        params=[
            "s_b=1e-3/b=0.3/s_d=3e-1/p_b=0.00",
            "s_b=1e-3/b=0.1/s_d=3e-2/p_b=0.00",
            "s_b=1e-2/b=0.1/s_d=3e-1/p_b=0.01",
            "s_b=1e-3/b=0.3/s_d=3e-2/p_b=0.05"
        ],
        n=[20, 100]
    ) + expand(
        "testing/cache/slim/n_replicate=1/n_chunks=100/g=1e4/L=1e7/mu=1e-8/r=1e-6/N=1e3/{params}/n={n}/dominance_0.0/unfolded/sfs.csv",
        params=[
            "s_b=1e-3/b=0.3/s_d=3e-1/p_b=0.00",
            "s_b=1e-3/b=0.1/s_d=3e-2/p_b=0.00",
            "s_b=1e-2/b=0.1/s_d=3e-1/p_b=0.01",
            "s_b=1e-3/b=0.3/s_d=3e-2/p_b=0.05"
        ],
        n=[20, 100]
    ) + expand(
        "testing/cache/slim/n_replicate=1/n_chunks=100/g=1e4/L=1e7/mu=1e-8/r=1e-6/N=1e3/{params}/n={n}/dominance_function_{k}/unfolded/sfs.csv",
        k=[0.1, 1, 10, 100],
        params=[
            "s_b=1e-3/b=0.3/s_d=3e-1/p_b=0.00",
            "s_b=1e-3/b=0.1/s_d=3e-2/p_b=0.00",
            "s_b=1e-2/b=0.1/s_d=3e-1/p_b=0.01",
            "s_b=1e-3/b=0.3/s_d=3e-2/p_b=0.05"
        ],
        n=[20]
    )

    @staticmethod
    def _dominance(file_path: str):
        """
        Dominance coefficient encoded in the path of a SLiM result, ``None`` for semidominant and
        dominance-function simulations.
        """
        match = re.search(r"dominance_([\d.]+)/", file_path)

        return float(match.group(1)) if match else None

    def test_compare_against_slim(self):
        """
        Test the modelled SFS against SLiM simulations with semidominant mutations, for unfolded and folded
        spectra.
        """
        self._compare_against_slim([f for f in self.configs_slim if 'dominance_' not in f], tol=0.1)

    def test_compare_against_slim_dominance_low(self):
        """
        Test the modelled SFS against SLiM simulations with dominance coefficients from 0.1 to 0.5 for a
        sample size of 20.
        """
        self._compare_against_slim(
            [f for f in self.configs_slim if 0 < (self._dominance(f) or 0) <= 0.5 and '/n=20/' in f]
        )

    def test_compare_against_slim_dominance_high(self):
        """
        Test the modelled SFS against SLiM simulations with dominance coefficients from 0.6 to 1 for a
        sample size of 20.
        """
        self._compare_against_slim(
            [f for f in self.configs_slim if (self._dominance(f) or 0) > 0.5 and '/n=20/' in f]
        )

    def _compare_against_slim_dominance_n100(self, h: float):
        """
        Test the modelled SFS against SLiM simulations with the given dominance coefficient for a sample size
        of 100.

        :param h: Dominance coefficient
        """
        self._compare_against_slim([f for f in self.configs_slim if self._dominance(f) == h and '/n=100/' in f])

    def test_compare_against_slim_recessive(self):
        """
        Test the modelled SFS against SLiM simulations with fully recessive mutations.
        """
        self._compare_against_slim([f for f in self.configs_slim if self._dominance(f) == 0])

    def test_compare_against_slim_dominance_function(self):
        """
        Test the modelled SFS against SLiM simulations with dominance depending on the selection coefficient,
        modelled by the same dominance function. The spectra were compared against the semidominant model,
        which the tolerance of 0.16 let pass.
        """
        self._compare_against_slim([f for f in self.configs_slim if 'dominance_function_' in f], tol=0.04)

    def _compare_against_slim(self, configs, tol: float = 0.16):
        """
        Compare the modelled selected SFS against the SLiM result of each of the given configurations.

        :param configs: Paths to the SLiM spectra
        :param tol: Largest accepted root mean squared relative difference
        """
        self.assertGreater(len(configs), 0)

        cached = {}

        for file_path in configs:
            spectra = fd.Spectra.from_file(file_path)

            params = {}
            for p in ['s_b', 's_d', 'b', 'p_b', 'mu', 'n']:
                # match parameter values
                match = re.search(rf"\b{p}=([\d.e+-]+)", file_path)
                if match:
                    params[p] = float(match.group(1))

            if match := re.search(r"dominance_([\d.]+)", file_path):
                params['h'] = float(match.group(1))

            Ne = spectra['neutral'].theta / (4 * params['mu'])
            params['S_b'] = 4 * Ne * params['s_b']
            params['S_d'] = -4 * Ne * params['s_d']

            # dominance function of the SLiM simulations
            h_callback = None
            if match := re.search(r"dominance_function_([\d.]+)/", file_path):
                h_callback = lambda h, S, k=float(match.group(1)), Ne=Ne: np.maximum(
                    0.4 * np.exp(-k * abs(S / (4 * Ne))), 0.1
                )

            model = fd.GammaExpParametrization()
            model.bounds['S_b'] = (1e-10, 100)

            sim = fd.Simulation(
                params=params,
                h_callback=h_callback,
                sfs_neut=spectra['neutral'],
                model=model,
                intervals_del=(-1.0e+8, -1.0e-5, 100),
                intervals_ben=(1.0e-5, 1.0e4, 100)
            )

            # cache discretization
            if sim.discretization in cached:
                sim.discretization = cached[sim.discretization]
            else:
                cached[sim.discretization] = sim.discretization

            def get_diff(sfs_sel: fd.Spectrum) -> float:
                """
                Root mean squared relative difference of a modelled SFS to the SLiM result.

                :param sfs_sel: Modelled selected SFS
                :return: Root mean squared relative difference
                """
                slim = spectra['selected'].data[:-1]
                diff_rel = (slim - sfs_sel.data[:-1]) / slim

                # exclude bins with few counts due to large variance
                diff_rel[slim < 20] = 0

                return np.sqrt((diff_rel ** 2).mean())

            diff = get_diff(sim.run())

            if diff < tol:
                fd.logger.info(f"{diff:.3f} < {tol} for {params}")
            else:
                fd.logger.fatal(f"{diff:.3f} >= {tol} for {params}")

            self.assertLess(diff, tol)

            # a dominance coefficient far from 0.5 fits better than semidominance, which the tolerance alone lets pass
            if abs(params.get('h', 0.5) - 0.5) >= 0.2:
                semidominant = fd.Simulation(
                    params=params | dict(h=0.5),
                    sfs_neut=spectra['neutral'],
                    model=model,
                    intervals_del=(-1.0e+8, -1.0e-5, 100),
                    intervals_ben=(1.0e-5, 1.0e4, 100)
                )

                self.assertLess(diff, get_diff(semidominant.run()))

    # SLiM configs carrying beneficial mutations (p_b > 0), whose fixed (n-)class therefore contains
    # a non-trivial amount of selected divergence
    configs_slim_divergence = expand(
        "testing/cache/slim/n_replicate=1/n_chunks=40/g=1e4/L=1e7/mu=1e-8/r=1e-7/N=1e3/"
        "s_b={s_b}/b=0.4/s_d={s_d}/p_b={p_b}/n=20/unfolded/sfs.csv",
        s_b=['1e-4', '1e-3'],
        s_d=['1e0', '1e-1'],
        p_b=['0.05', '0.01']
    )

    def test_divergence_alpha_against_slim(self):
        """
        Ground-truth test of the divergence machinery against SLiM, across all beneficial (p_b > 0)
        full-DFE scenarios. The fixed (n-)class of the SLiM SFS *is* the divergence (substitutions),
        and divergence shares the polymorphism target size here. We attach it as
        the inference's ``n_sites_div_neut`` / ``n_sites_div_sel`` and infer the full DFE both with and without divergence.

        We assert that (1) the divergence-based (McDonald-Kreitman) alpha recovers the alpha implied
        by the true DFE for every scenario, (2) using divergence is at least as good as polymorphism
        alone, and (3) for weak beneficial selection — where beneficial mutations barely segregate
        and polymorphism alone cannot detect them — divergence recovers alpha while polymorphism
        fails, i.e. divergence recovers the (beneficial) DFE substantially more precisely.
        """
        from sfsutils.spectrum import Spectrum

        errors_div, errors_poly_weak, errors_div_weak = [], [], []

        for file_path in self.configs_slim_divergence:
            spectra = fd.Spectra.from_file(file_path)

            params = {p: float(re.search(rf"\b{p}=([\d.e+-]+)", file_path).group(1))
                      for p in ['s_b', 's_d', 'b', 'p_b', 'mu', 'n']}
            Ne = spectra['neutral'].theta / (4 * params['mu'])
            params['S_b'] = 4 * Ne * params['s_b']
            params['S_d'] = -4 * Ne * params['s_d']

            # ground-truth alpha implied by the known DFE
            model = fd.GammaExpParametrization()
            model.bounds['S_b'] = (1e-10, 100)
            alpha_true = fd.Simulation(
                params=params, sfs_neut=spectra['neutral'], model=model,
                intervals_del=(-1.0e+8, -1.0e-5, 100), intervals_ben=(1.0e-5, 1.0e4, 100)
            ).get_alpha()

            kwargs = dict(
                model=fd.GammaExpParametrization(),
                fixed_params={'all': {'eps': 0, 'h': 0.5}},  # full DFE (infer p_b, S_b)
                intervals_del=(-1.0e+8, -1.0e-5, 100), intervals_ben=(1.0e-5, 1.0e4, 100),
                x0={'all': {'S_d': -1000, 'b': 0.4, 'p_b': 0.01, 'S_b': 1}},
                do_bootstrap=False, n_runs=30, seed=0
            )

            # inference with divergence (the fixed class shares the polymorphism target size)
            inf_div = fd.BaseInference(
                sfs_neut=Spectrum(spectra['neutral'].data),
                sfs_sel=Spectrum(spectra['selected'].data),
                n_sites_div_neut=spectra['neutral'].n_sites,
                n_sites_div_sel=spectra['selected'].n_sites,
                **kwargs
            )
            inf_div.run()
            alpha_div = inf_div.get_alpha(use_divergence=True)

            # inference from polymorphism alone (no divergence target size)
            inf_poly = fd.BaseInference(
                sfs_neut=Spectrum(spectra['neutral'].data),
                sfs_sel=Spectrum(spectra['selected'].data),
                **kwargs
            )
            inf_poly.run()
            alpha_poly = inf_poly.get_alpha(use_divergence=False)

            err_div, err_poly = abs(alpha_div - alpha_true), abs(alpha_poly - alpha_true)

            # (1) divergence recovers the true alpha
            self.assertLess(err_div, 0.12, f"{params}: divergence alpha {alpha_div:.3f} vs true {alpha_true:.3f}")

            # (2) using divergence is at least as good as polymorphism alone
            self.assertLessEqual(err_div, err_poly + 0.06, f"{params}: divergence worse than polymorphism")

            errors_div.append(err_div)
            if params['s_b'] <= 1e-4:  # weak beneficial selection
                errors_div_weak.append(err_div)
                errors_poly_weak.append(err_poly)

        # (3) for weak beneficial selection, polymorphism alone fails but divergence recovers alpha
        assert errors_div_weak, "expected weak-beneficial SLiM scenarios"
        self.assertLess(np.mean(errors_div_weak), 0.1)
        self.assertGreater(np.mean(errors_poly_weak) - np.mean(errors_div_weak), 0.1)

    def test_divergence_omega_against_slim(self):
        """
        Ground-truth test of omega (dN/dS) and omega_a (adaptive dN/dS) against SLiM, across all
        beneficial (p_b > 0) full-DFE scenarios. The fixed (n-)class of the SLiM SFS *is* the
        divergence (substitutions), sharing the polymorphism target size, so the observed dN/dS is
        directly available. We assert that both the McDonald-Kreitman (divergence) estimate and the
        DFE-integral estimate recover the omega/omega_a implied by the known true DFE.
        """
        from sfsutils.spectrum import Spectrum

        for file_path in self.configs_slim_divergence:
            spectra = fd.Spectra.from_file(file_path)

            params = {p: float(re.search(rf"\b{p}=([\d.e+-]+)", file_path).group(1))
                      for p in ['s_b', 's_d', 'b', 'p_b', 'mu', 'n']}
            Ne = spectra['neutral'].theta / (4 * params['mu'])
            params['S_b'] = 4 * Ne * params['s_b']
            params['S_d'] = -4 * Ne * params['s_d']

            # ground-truth omega/omega_a implied by the known DFE
            model = fd.GammaExpParametrization()
            model.bounds['S_b'] = (1e-10, 100)
            sim = fd.Simulation(
                params=params, sfs_neut=spectra['neutral'], model=model,
                intervals_del=(-1.0e+8, -1.0e-5, 100), intervals_ben=(1.0e-5, 1.0e4, 100)
            )
            omega_true, omega_a_true = sim.get_omega(), sim.get_omega_a()

            # infer the full DFE with divergence (the fixed class shares the polymorphism target size)
            inf = fd.BaseInference(
                sfs_neut=Spectrum(spectra['neutral'].data),
                sfs_sel=Spectrum(spectra['selected'].data),
                n_sites_div_neut=spectra['neutral'].n_sites,
                n_sites_div_sel=spectra['selected'].n_sites,
                model=fd.GammaExpParametrization(),
                fixed_params={'all': {'eps': 0, 'h': 0.5}},
                intervals_del=(-1.0e+8, -1.0e-5, 100), intervals_ben=(1.0e-5, 1.0e4, 100),
                do_bootstrap=False, n_runs=3, seed=0
            )
            inf.run()

            # both the observed (McDonald-Kreitman) and DFE-integral estimates recover the truth
            for use_divergence in (True, False):
                self.assertLess(
                    abs(inf.get_omega(use_divergence=use_divergence) - omega_true), 0.05,
                    f"{params}: omega (use_divergence={use_divergence}) "
                    f"{inf.get_omega(use_divergence=use_divergence):.3f} vs true {omega_true:.3f}"
                )
                self.assertLess(
                    abs(inf.get_omega_a(use_divergence=use_divergence) - omega_a_true), 0.05,
                    f"{params}: omega_a (use_divergence={use_divergence}) "
                    f"{inf.get_omega_a(use_divergence=use_divergence):.3f} vs true {omega_a_true:.3f}"
                )

    # empirical (substitution-count) alpha for the s_b=1e-3 dominance family, cached by
    # snakemake/scripts/precompute_empirical_alpha_slim.py
    empirical_alpha_cache = 'testing/cache/slim/empirical_alpha/s_b=1e-3_b=0.3_s_d=3e-2_p_b=0.05.json'

    def test_divergence_alpha_against_slim_dominance_low(self):
        """
        Validate the divergence-based alpha against the empirical SLiM ground truth for dominance
        coefficients from 0.1 to 0.5, see :meth:`_divergence_alpha_against_slim_dominance`.
        """
        self._divergence_alpha_against_slim_dominance(0, 0.5)

    def test_divergence_alpha_against_slim_dominance_high(self):
        """
        Validate the divergence-based alpha against the empirical SLiM ground truth for dominance
        coefficients from 0.6 to 1, see :meth:`_divergence_alpha_against_slim_dominance`.
        """
        self._divergence_alpha_against_slim_dominance(0.5, 1)

    def _divergence_alpha_against_slim_dominance(self, h_min: float, h_max: float):
        """
        Validate the divergence-based alpha against the *empirical* SLiM ground truth across
        dominance coefficients. The empirical alpha is the actual proportion of fixed beneficial
        substitutions counted directly from SLiM (see the cache above), rather than an analytical
        proxy. The fully recessive case (h=0) is excluded as a known hard regime (recessive
        beneficials are nearly invisible while rare, so the SFS/divergence carry little signal).

        :param h_min: Exclusive lower limit of the dominance coefficients to test
        :param h_max: Inclusive upper limit of the dominance coefficients to test
        """
        import json
        from sfsutils.spectrum import Spectrum

        cache = json.load(open(self.empirical_alpha_cache))
        base = ('testing/cache/slim/n_replicate=1/n_chunks=100/g=1e4/L=1e7/mu=1e-8/r=1e-7/N=1e3/'
                's_b=1e-3/b=0.3/s_d=3e-2/p_b=0.05/n=20')

        for rec in cache['alpha'].values():
            h, alpha_empirical = rec['h'], rec['alpha']
            if not h_min < h <= h_max:
                continue

            spectra = fd.Spectra.from_file(f'{base}/dominance_{h}/unfolded/sfs.csv')
            inf = fd.BaseInference(
                sfs_neut=Spectrum(spectra['neutral'].data),
                sfs_sel=Spectrum(spectra['selected'].data),
                n_sites_div_neut=spectra['neutral'].n_sites,
                n_sites_div_sel=spectra['selected'].n_sites,
                model=fd.GammaExpParametrization(),
                fixed_params={'all': {'eps': 0, 'h': h}},
                intervals_del=(-1.0e+8, -1.0e-5, 100), intervals_ben=(1.0e-5, 1.0e4, 100),
                do_bootstrap=False, n_runs=3, seed=0
            )
            inf.run()

            self.assertLess(
                abs(inf.get_alpha(use_divergence=True) - alpha_empirical), 0.1,
                f'h={h}: divergence alpha {inf.get_alpha(use_divergence=True):.3f} vs '
                f'empirical {alpha_empirical:.3f}'
            )


# one test per dominance coefficient for a sample size of 100, whose discretization is the expensive part
for _h in np.round(np.linspace(0.1, 1, 10), 1):
    setattr(
        SLiMTestCase,
        f"test_compare_against_slim_dominance_n100_h{_h}".replace('.', '_'),
        lambda self, h=float(_h): self._compare_against_slim_dominance_n100(h)
    )


@pytest.mark.slow
class SimulationTestCase(TestCase):
    """
    Test the Simulation class.
    """

    def test_get_neutral_sfs(self):
        """
        Test the get_neutral_sfs method.
        """
        theta = 1e-4
        n_sites = 1e8
        n = 10
        sfs = fd.Simulation.get_neutral_sfs(theta=theta, n_sites=n_sites, n=n)

        self.assertAlmostEqual(sfs.n_sites, n_sites)
        self.assertAlmostEqual(sfs.n, n)
        self.assertAlmostEqual(sfs.theta, theta)

        np.testing.assert_array_almost_equal(sfs.data[1:-1], [theta * n_sites / i for i in range(1, 10)])

        self.assertAlmostEqual(sfs.data[0], n_sites - sum(sfs.data[1:]))
        self.assertAlmostEqual(sfs.data[-1], 0)

    def test_get_neutral_sfs_with_demography(self):
        """
        Test the get_neutral_sfs method with demography.
        """
        theta = 1e-4
        n_sites = 1e8
        n = 10
        r = [0.98, 1.21, 0.87, 1.43, 0.92, 1.32, 0.99, 1.12, 0.95]
        sfs = fd.Simulation.get_neutral_sfs(theta=theta, n_sites=n_sites, n=n, r=r)

        self.assertAlmostEqual(sfs.n_sites, n_sites)
        self.assertAlmostEqual(sfs.n, n)

        # nuisance parameters affect mutation rate
        self.assertAlmostEqual(sfs.theta, theta, delta=1e-5)

        np.testing.assert_array_almost_equal(sfs.data[1:-1], [theta * n_sites / i * r[i - 1] for i in range(1, 10)])

        self.assertAlmostEqual(sfs.data[0], n_sites - sum(sfs.data[1:]))
        self.assertAlmostEqual(sfs.data[-1], 0)

    def test_get_neutral_sfs_wrong_length_r(self):
        """
        Test the get_neutral_sfs method with wrong length r.
        """
        theta = 1e-4
        n_sites = 1e8
        n = 10
        r = [1] * 10

        with self.assertRaises(ValueError) as e:
            fd.Simulation.get_neutral_sfs(theta=theta, n_sites=n_sites, n=n, r=r)

        print(e.exception)

    def test_recover_result_full_dfe_no_demography(self):
        """
        Test that the simulated result can be recovered by inference.
        """
        # very deleterious DFEs were difficult to recover
        sim = fd.Simulation(
            sfs_neut=fd.Simulation.get_neutral_sfs(n=20, n_sites=1e8, theta=1e-4),
            params=dict(S_d=-300, b=1, p_b=0.05, S_b=0.1)
        )

        sfs_sel = sim.run()

        inf = fd.BaseInference(
            sfs_sel=sfs_sel,
            sfs_neut=sim.sfs_neut,
            discretization=sim.discretization,
            fixed_params=dict(all=dict(eps=0, h=0.5)),
            do_bootstrap=True,
            n_bootstraps=100,
            parallelize=True
        )

        inf.run()

        self.assertAlmostEqual(sim.n_sites, inf.sfs_sel.n_sites, delta=1e-6)
        self.assertAlmostEqual(sim.n_sites, inf.sfs_neut.n_sites)

        self.assertAlmostEqual(sim.theta, inf.sfs_neut.theta)

        for key in sim.params:
            self.assertAlmostEqual(inf.params_mle[key], sim.params[key], delta=np.abs(sim.params[key]) / 500)

        mean = inf.bootstraps.select_dtypes("number").mean()

        self.assertAlmostEqual(mean['S_d'], sim.params['S_d'], delta=np.abs(sim.params['S_d']) / 10)
        self.assertAlmostEqual(mean['b'], sim.params['b'], delta=np.abs(sim.params['b']) / 10)
        self.assertAlmostEqual(mean['p_b'], sim.params['p_b'], delta=np.abs(sim.params['p_b']) / 8)
        self.assertAlmostEqual(mean['S_b'], sim.params['S_b'], delta=np.abs(sim.params['S_b']) * 3)

        self.assertAlmostEqual(mean['eps'], 0, delta=1e-3)

    def test_recover_result_full_dfe_with_demography(self):
        """
        Test that the simulated result can be recovered by inference.
        """
        sim = fd.Simulation(
            sfs_neut=fd.Simulation.get_neutral_sfs(
                n=20,
                n_sites=1e8,
                theta=1e-4,
                r=[0.98, 1.21, 0.87, 1.43, 0.92, 1.32, 0.99,
                   1.12, 0.95, 1.21, 0.87, 1.43, 0.92,
                   0.99, 1.12, 0.95, 1.21, 0.87, 1.43]
            ),
            params=dict(S_d=-300, b=1, p_b=0.05, S_b=0.1)
        )

        sfs_sel = sim.run()

        inf = fd.BaseInference(
            sfs_sel=sfs_sel,
            sfs_neut=sim.sfs_neut,
            discretization=sim.discretization,
            fixed_params=dict(all=dict(eps=0, h=0.5)),
            do_bootstrap=True,
            n_bootstraps=100,
            parallelize=True
        )

        inf.run()

        self.assertAlmostEqual(sim.n_sites, inf.sfs_sel.n_sites, delta=1e-6)
        self.assertAlmostEqual(sim.n_sites, inf.sfs_neut.n_sites)

        self.assertAlmostEqual(sim.theta, inf.sfs_neut.theta)

        for key in sim.params:
            self.assertAlmostEqual(inf.params_mle[key], sim.params[key], delta=np.abs(sim.params[key]) / 100)

        mean = inf.bootstraps.select_dtypes("number").mean()

        self.assertAlmostEqual(mean['S_d'], sim.params['S_d'], delta=np.abs(sim.params['S_d']) / 8)
        self.assertAlmostEqual(mean['b'], sim.params['b'], delta=np.abs(sim.params['b']) / 10)
        self.assertAlmostEqual(mean['p_b'], sim.params['p_b'], delta=np.abs(sim.params['p_b']) / 5)
        self.assertAlmostEqual(mean['S_b'], sim.params['S_b'], delta=np.abs(sim.params['S_b']) * 3)

        self.assertAlmostEqual(inf.bootstraps['eps'].mean(), 0, delta=1e-3)

    def test_recover_result_deleterious_dfe_no_demography(self):
        """
        Test that the simulated result can be recovered by inference.
        """
        sim = fd.Simulation(
            sfs_neut=fd.Simulation.get_neutral_sfs(n=20, n_sites=1e8, theta=1e-4),
            params=dict(S_d=-300, b=1, p_b=0, S_b=0.1)
        )

        sfs_sel = sim.run()

        inf = fd.BaseInference(
            sfs_sel=sfs_sel,
            sfs_neut=sim.sfs_neut,
            discretization=sim.discretization,
            fixed_params=dict(all=dict(p_b=0, S_b=0.1, h=0.5, eps=0)),
            do_bootstrap=True,
            n_bootstraps=100,
            parallelize=True
        )

        inf.run()

        self.assertAlmostEqual(sim.n_sites, inf.sfs_sel.n_sites)
        self.assertAlmostEqual(sim.n_sites, inf.sfs_neut.n_sites)

        self.assertAlmostEqual(sim.theta, inf.sfs_neut.theta)

        for key in sim.params:
            self.assertAlmostEqual(inf.params_mle[key], sim.params[key], delta=np.abs(sim.params[key]) / 1000)

        self.assertAlmostEqual(inf.bootstraps['S_d'].mean(), sim.params['S_d'], delta=10)
        self.assertAlmostEqual(inf.bootstraps['b'].mean(), sim.params['b'], delta=0.1)
        self.assertAlmostEqual(inf.bootstraps['eps'].mean(), 0, delta=1e-3)

        self.assertAlmostEqual(inf.bootstraps['p_b'].mean(), 0)
        self.assertAlmostEqual(inf.bootstraps['S_b'].mean(), 0.1)

    def test_usage_example_simulation_class(self):
        """
        Test the usage example of the Simulation class.
        """
        import fastdfe as fd

        # create simulation object by specifying neutral SFS and DFE
        sim = fd.Simulation(
            sfs_neut=fd.Simulation.get_neutral_sfs(n=20, n_sites=1e8, theta=1e-4),
            params=dict(S_d=-300, b=0.3, p_b=0.1, S_b=0.1),
            model=fd.GammaExpParametrization()
        )

        # perform the simulation
        sfs_sel = sim.run()

        # plot SFS
        sfs_sel.plot()

    def test_parameter_out_of_bounds(self):
        """
        Test that the simulation of a neutral DFE with the Wright-Fisher model is correct.
        """
        with self.assertRaises(ValueError) as e:
            fd.Simulation(
                sfs_neut=fd.Simulation.get_neutral_sfs(n=10, n_sites=1e8, theta=1e-4),
                params=dict(S_d=0, b=1, p_b=0, S_b=1),
            )

        print(e.exception)

    def test_get_alpha(self):
        """
        Test the get_alpha method.
        """
        alphas = []
        for h in np.linspace(0, 1, 11):
            sim = fd.Simulation(
                sfs_neut=fd.Simulation.get_neutral_sfs(n=20, n_sites=1e8, theta=1e-4),
                params=dict(S_d=-300, b=0.3, p_b=0.05, S_b=0.1, h=h)
            )

            alphas.append(sim.get_alpha())

        # make sure order is ascending
        self.assertTrue(np.all(np.diff(alphas) >= 0))

        self.assertEqual(fd.Simulation(
            sfs_neut=fd.Simulation.get_neutral_sfs(n=20, n_sites=1e8, theta=1e-4),
            params=dict(S_d=-300, b=0.3, p_b=0, S_b=0.1, h=h)
        ).get_alpha(), 0)

        self.assertAlmostEqual(fd.Simulation(
            model=fd.GammaExpParametrization(bounds=dict(p_b=(0, 1))),
            sfs_neut=fd.Simulation.get_neutral_sfs(n=20, n_sites=1e8, theta=1e-4),
            params=dict(S_d=-300, b=0.3, p_b=1, S_b=0.1, h=h)
        ).get_alpha(), 1, delta=1e-3)


class SimulationSmokeTestCase(TestCase):
    """
    Fast (fast-tier) coverage of the analytic ``Simulation`` class on a tiny SFS with a coarse
    discretization.
    """

    @staticmethod
    def _make(params=None, **kwargs):
        return fd.Simulation(
            sfs_neut=fd.Simulation.get_neutral_sfs(n=10, n_sites=1e6, theta=1e-3),
            params=params if params is not None else dict(S_d=-500, b=0.4, p_b=0.05, S_b=10),
            intervals_del=(-1.0e+8, -1.0e-5, 20),
            intervals_ben=(1.0e-5, 1.0e4, 20),
            **kwargs
        )

    def test_default_parameters_do_not_modify_parametrization(self):
        """
        A simulation adds ``eps`` and ``h`` to its own copy of the parameters. It added them to the dictionary
        it was given, which with the default parameters is the class-level ``x0`` of the parametrization, so
        every parametrization created afterwards listed ``eps`` and ``h`` among its parameter names.
        """
        sfs_neut = fd.Simulation.get_neutral_sfs(n=10, n_sites=1e6, theta=1e-3)
        param_names = fd.GammaExpParametrization().param_names
        params = dict(S_d=-300, b=0.3, p_b=0.1, S_b=2)

        sim = fd.Simulation(sfs_neut=sfs_neut)
        fd.Simulation(sfs_neut=sfs_neut, params=params)

        self.assertEqual(sim.params, fd.GammaExpParametrization.x0 | dict(eps=0, h=0.5))
        self.assertEqual(fd.GammaExpParametrization().param_names, param_names)
        self.assertNotIn('eps', fd.GammaExpParametrization.x0)
        self.assertEqual(params, dict(S_d=-300, b=0.3, p_b=0.1, S_b=2))

    def test_run_and_summaries(self):
        """run() plus the spectra and alpha/omega/omega_a summaries are finite."""
        sim = self._make()
        sfs_sel = sim.run()

        self.assertEqual(sfs_sel.n, 10)
        self.assertTrue(np.all(np.isfinite(sfs_sel.to_list())))

        spectra = sim.get_spectra()
        self.assertEqual(set(spectra.types), {'neutral', 'selected'})

        for value in (sim.get_alpha(), sim.get_omega(), sim.get_omega_a()):
            self.assertTrue(np.isfinite(value))

    def test_params_out_of_bounds_raises(self):
        """Out-of-range eps / dominance and out-of-bounds DFE params are rejected."""
        with self.assertRaises(ValueError):
            self._make(params=dict(S_d=-500, b=0.4, p_b=0.05, S_b=10, eps=2.0))

        with self.assertRaises(ValueError):
            self._make(params=dict(S_d=-500, b=0.4, p_b=0.05, S_b=10, h=1.5))

    def test_no_divergence_class_by_default(self):
        """Without n_sites_div the fixed-derived class stays zero (backward compatible)."""
        self.assertEqual(self._make().run().n_div, 0)

    def test_divergence_class_matches_flux(self):
        """
        With n_sites_div the fixed-derived class equals the expected number of substitutions
        theta * L_div * flux (selected) and theta * L_div (neutral, flux = 1).
        """
        L = 2e6
        sim = self._make(n_sites_div=L)
        sfs_sel = sim.run()

        flux = sim.discretization.get_divergence_flux(sim.model, sim.params)
        np.testing.assert_allclose(sfs_sel.n_div, sim.theta * L * flux, rtol=1e-9)
        # the neutral spectrum carries theta * L_div (the neutral substitution flux is 1)
        np.testing.assert_allclose(sim.get_spectra()['neutral'].n_div, sim.theta * L, rtol=1e-9)

    def test_divergence_class_is_part_of_number_of_sites(self):
        """
        With ``n_sites_div``, the simulated spectra keep the number of sites and the mutation rate of the
        simulation, so the expected selected SFS at the true parameters equals the simulated one. The
        divergence class was added on top of the sites, which lowered the neutral mutation rate and biased the
        inferred DFE for large ``n_sites_div``.
        """
        sim = self._make(n_sites_div=50 * 1e6)
        sim.run()
        spectra = sim.get_spectra()
        neut, sel = spectra['neutral'], spectra['selected']

        self.assertAlmostEqual(neut.n_sites / sim.n_sites, 1, places=12)
        self.assertAlmostEqual(sel.n_sites / sim.n_sites, 1, places=12)
        self.assertAlmostEqual(neut.theta / sim.theta, 1, places=12)
        self.assertGreater(neut.n_div / neut.n_sites, 0.01)

        modelled, observed = fd.BaseInference._model_sfs(
            discretization=sim.discretization,
            model=sim.model,
            params=sim.params,
            sfs_neut=neut,
            sfs_sel=sel,
            folded=False
        )

        np.testing.assert_allclose(modelled, observed, rtol=1e-10)

        # the monomorphic class is capped at zero when there are more substitutions than sites
        sim = self._make(n_sites_div=1e12)
        sim.run()

        spectra = sim.get_spectra()

        self.assertEqual([spectra[t].data[0] for t in ('neutral', 'selected')], [0, 0])

    def test_divergence_roundtrip_recovers_truth(self):
        """
        Feeding the simulated (clean, matched-model) divergence back into the divergence-based
        estimators recovers the truth. ``omega`` matches exactly (both equal the total substitution
        flux); ``alpha`` matches up to the near-neutral bin, where the MK and DFE estimators
        partition the s=0 mass differently (there is no finite-sample misattribution here).
        """
        L = 2e6
        sim = self._make(n_sites_div=L)
        sim.run()
        spectra = sim.get_spectra()

        inf = fd.BaseInference(
            sfs_neut=spectra['neutral'], sfs_sel=spectra['selected'],
            n_sites_div_neut=L, n_sites_div_sel=L,
            discretization=sim.discretization,                  # share -> exact agreement
            fixed_params={'all': {'eps': 0, 'h': 0.5}}
        )

        true = dict(sim.params)
        # the divergence-based omega equals the DFE omega exactly (both are the substitution flux)
        np.testing.assert_allclose(inf.get_omega(params=true, use_divergence=True),
                                   inf.get_omega(params=true, use_divergence=False), rtol=1e-9)
        # the divergence-based alpha recovers the true alpha (near-neutral bin aside)
        np.testing.assert_allclose(inf.get_alpha(params=true, use_divergence=True),
                                   sim.get_alpha(), atol=5e-3)
