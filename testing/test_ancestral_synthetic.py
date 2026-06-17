"""
Maximum-likelihood ancestral-allele annotation exercised on *synthetic* site configurations fed in
directly via ``from_dataframe`` (integer-encoded bases, no VCF/FASTA needed). This covers the ML
ancestral inference core -- rate optimization, likelihood evaluation, polarization priors and
per-site inference -- which the end-to-end annotation tests (inference/slow tier) need a real
genome for. A few hundred synthetic sites optimize in well under a second.
"""
import numpy as np
import pandas as pd
import pytest

from fastdfe.annotation import (
    MaximumLikelihoodAncestralAnnotation as ML,
    JCSubstitutionModel, K2SubstitutionModel,
    KingmanPolarizationPrior, AdaptivePolarizationPrior,
)


def _synthetic_sites(n=300, n_mono=100, seed=0):
    """A mix of bi-allelic and (mostly) monomorphic sites, integer-encoded (0..3, -1 = no minor)."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        major = int(rng.integers(0, 4))
        if i < n_mono:
            minor = -1  # monomorphic
        else:
            minor = int(rng.integers(0, 4))
            if minor == major:
                minor = (major + 1) % 4
        rows.append(dict(major_base=major, minor_base=minor,
                         outgroup_bases=[major], n_major=int(rng.integers(5, 11))))
    return pd.DataFrame(rows)


@pytest.fixture(scope='module')
def fitted():
    a = ML.from_dataframe(_synthetic_sites(), n_ingroups=10, n_runs=1,
                          model=JCSubstitutionModel(), parallelize=False, seed=0)
    a.infer()
    return a


def test_ml_ancestral_infers_finite_rates(fitted):
    # rate optimization produced a finite MLE and branch rate
    assert np.isfinite(fitted.likelihood)
    assert 'K0' in fitted.params_mle
    lo, hi = fitted.model.get_bound('K')
    assert lo <= fitted.params_mle['K0'] <= hi


def test_ml_ancestral_evaluate_likelihood_matches_mle(fitted):
    # evaluating the loss at the MLE reproduces the reported likelihood
    assert np.isclose(fitted.evaluate_likelihood(fitted.params_mle), fitted.likelihood)


def test_ml_ancestral_site_info_probabilities(fitted):
    site_info = list(fitted.get_inferred_site_info())
    assert len(site_info) == 300
    # ancestral probabilities are valid
    probs = [s.p_major_ancestral for s in site_info if s.p_major_ancestral is not None]
    assert probs and all(0.0 <= p <= 1.0 for p in probs)


@pytest.mark.parametrize('model', [JCSubstitutionModel(), K2SubstitutionModel()])
def test_ml_ancestral_runs_for_both_models(model):
    a = ML.from_dataframe(_synthetic_sites(n=200, seed=1), n_ingroups=8, n_runs=1,
                          model=model, parallelize=False, seed=0)
    a.infer()
    assert np.isfinite(a.likelihood)


@pytest.mark.parametrize('prior', [KingmanPolarizationPrior(), AdaptivePolarizationPrior()])
def test_ml_ancestral_runs_with_priors(prior):
    a = ML.from_dataframe(_synthetic_sites(n=200, seed=2), n_ingroups=8, n_runs=1,
                          model=JCSubstitutionModel(), prior=prior, parallelize=False, seed=0)
    a.infer()
    assert np.isfinite(a.likelihood)
    assert a.prior is prior


def test_ml_ancestral_empty_dataframe_raises():
    with pytest.raises(ValueError):
        ML.from_dataframe(pd.DataFrame(), n_ingroups=10)
