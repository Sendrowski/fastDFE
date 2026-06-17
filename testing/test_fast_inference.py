"""
Fast end-to-end inference tests on a *simulated* SFS with a small discretization grid, a single
optimizer run and loose tolerance. These stay well under a second each but meaningfully exercise
the inference stack (discretization, optimization, likelihood, bootstrap, joint inference) that the
pure-unit tests cannot reach. Assertions check that inference ran *sensibly* (correct sign, valid
ranges, within bounds, finite), not exact recovery -- exact recovery is covered by the inference tier.
"""
import numpy as np
import pytest

import fastdfe as fd
from fastdfe.discretization import Discretization
from fastdfe.parametrization import GammaExpParametrization

# small grid + single run + loose tolerance keeps each inference fast
FAST = dict(
    intervals_del=(-1.0e8, -1.0e-5, 100),
    intervals_ben=(1.0e-5, 1.0e4, 100),
    n_runs=1,
    parallelize=False,
    seed=0,
    opts_mle=dict(gtol=1e-3, maxiter=100),
)


@pytest.fixture(scope='module')
def sim_spectra():
    # strongly deleterious DFE with a small beneficial fraction
    sim = fd.Simulation(
        sfs_neut=fd.Simulation.get_neutral_sfs(n=10, n_sites=1e7, theta=1e-2),
        params=dict(S_d=-300, b=0.4, p_b=0.05, S_b=1, h=0.5),
    )
    sim.run()
    return sim.get_spectra()


def _assert_within_bounds(inf):
    for p, v in inf.params_mle.items():
        lo, hi = inf.bounds[p]
        assert lo <= v <= hi, f'{p}={v} outside bounds [{lo}, {hi}]'


def test_base_inference_runs_sensibly(sim_spectra):
    inf = fd.BaseInference(
        sfs_neut=sim_spectra['neutral'], sfs_sel=sim_spectra['selected'],
        do_bootstrap=False, **FAST,
    )
    inf.run()

    # simulated DFE is strongly deleterious -> recovered as deleterious, all outputs sensible
    assert inf.params_mle['S_d'] < 0
    assert 0.0 <= inf.alpha <= 1.0
    assert np.isfinite(inf.likelihood)
    _assert_within_bounds(inf)


def test_base_inference_bootstrap_is_valid(sim_spectra):
    inf = fd.BaseInference(
        sfs_neut=sim_spectra['neutral'], sfs_sel=sim_spectra['selected'],
        do_bootstrap=True, n_bootstraps=3, **FAST,
    )
    inf.run()

    assert inf.bootstraps is not None and inf.bootstraps.shape[0] == 3
    # alpha is added to the bootstraps and all resampled estimates are finite
    assert 'alpha' in inf.bootstraps.columns
    assert np.isfinite(inf.bootstraps['alpha'].to_numpy()).all()


def test_joint_inference_atomic_runs_sensibly(sim_spectra):
    neut = fd.Spectra({'a': sim_spectra['neutral'].to_list(), 'b': sim_spectra['neutral'].to_list()})
    sel = fd.Spectra({'a': sim_spectra['selected'].to_list(), 'b': sim_spectra['selected'].to_list()})

    ji = fd.JointInference(sfs_neut=neut, sfs_sel=sel, do_bootstrap=False, **FAST)
    ji.run()

    assert list(ji.types) == ['a', 'b']
    # identical input per type -> each per-type joint fit is deleterious and sensible
    for t in ji.types:
        sub = ji.joint_inferences[t]
        assert sub.params_mle['S_d'] < 0
        assert 0.0 <= sub.alpha <= 1.0
        assert np.isfinite(sub.likelihood)


def test_base_inference_with_divergence_counts():
    """Divergence mode: r_div is derived from neutral divergence and alpha/omega use the counts."""
    sim = fd.Simulation(
        sfs_neut=fd.Simulation.get_neutral_sfs(n=10, n_sites=1e7, theta=1e-2),
        params=dict(S_d=-300, b=0.4, p_b=0.05, S_b=1, h=0.5), n_sites_div=1e5,
    )
    sim.run()
    sp = sim.get_spectra()

    inf = fd.BaseInference(
        sfs_neut=sp['neutral'], sfs_sel=sp['selected'],
        include_divergence=True, n_sites_div_neut=1e5, n_sites_div_sel=1e5,
        do_bootstrap=False, **FAST,
    )
    inf.run()

    assert inf._use_divergence is True
    # alpha/omega_a estimated from observed divergence are finite and in range
    assert 0.0 <= inf.get_alpha(use_divergence=True) <= 1.0
    assert np.isfinite(inf.get_omega_a())
    assert np.isfinite(inf.get_omega())


def test_folded_full_dfe_warns(caplog):
    """Estimating the full DFE (beneficial params free) on a folded SFS warns about lost info."""
    sim = fd.Simulation(
        sfs_neut=fd.Simulation.get_neutral_sfs(n=10, n_sites=1e6, theta=1e-3),
        params=dict(S_d=-300, b=0.4, p_b=0.05, S_b=1, h=0.5),
    )
    sim.run()
    sp = sim.get_spectra()

    import logging
    fastdfe_logger = logging.getLogger('fastdfe')
    propagate = fastdfe_logger.propagate
    fastdfe_logger.propagate = True  # let caplog (root) see fastdfe's warning
    try:
        with caplog.at_level(logging.WARNING):
            fd.BaseInference(
                sfs_neut=sp['neutral'].fold(), sfs_sel=sp['selected'].fold(),
                folded=True, fixed_params={'all': {'h': 0.5}},  # beneficial params NOT fixed
                do_bootstrap=False, **FAST,
            )
    finally:
        fastdfe_logger.propagate = propagate
    assert 'full DFE' in caplog.text and 'folded' in caplog.text


def test_locked_inference_cannot_run(sim_spectra):
    inf = fd.BaseInference(
        sfs_neut=sim_spectra['neutral'], sfs_sel=sim_spectra['selected'],
        locked=True, do_bootstrap=False, **FAST,
    )
    with pytest.raises(Exception, match='locked'):
        inf.run()


def test_symlog_scaled_bounds_in_bootstrap(sim_spectra):
    # a symlog-scaled parameter exercises the symlog -> linear bound conversion during bootstrap
    inf = fd.BaseInference(
        sfs_neut=sim_spectra['neutral'], sfs_sel=sim_spectra['selected'],
        scales={'S_b': 'symlog'}, do_bootstrap=True, n_bootstraps=2, **FAST,
    )
    inf.run()
    assert inf.bootstraps is not None and inf.bootstraps.shape[0] == 2


def test_discretization_midpoint_and_quad_agree():
    """The midpoint and quad integration modes must produce the same DFE->SFS transformation."""
    model = GammaExpParametrization()
    params = dict(model.x0, h=0.5)
    grid = dict(n=10, intervals_del=(-1.0e4, -1.0e-5, 30), intervals_ben=(1.0e-5, 1.0e3, 30),
                intervals_h=(0.0, 1.0, 3))

    sfs_mid = Discretization(integration_mode='midpoint', **grid).model_selection_sfs(model, params)
    sfs_quad = Discretization(integration_mode='quad', **grid).model_selection_sfs(model, params)

    assert sfs_mid.shape == sfs_quad.shape
    np.testing.assert_allclose(sfs_mid, sfs_quad, rtol=0.05)
