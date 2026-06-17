"""
Fast unit tests that exercise pure-logic branches across the library to keep the unit-tier
coverage high without invoking the (resource-heavy) parsing/annotation or full inference paths.
All tests here must stay fast and deterministic (no VCF/genome/SLiM/polyDFE, no real optimization).
"""
import numpy as np
import pandas as pd
import pytest

import fastdfe as fd
from fastdfe.parametrization import (
    _from_string, _to_string, DFE, GammaExpParametrization, DiscreteParametrization,
    DiscreteFractionalParametrization, GammaDiscreteParametrization, DisplacedGammaParametrization,
)
from fastdfe.spectrum import Spectrum, Spectra
from fastdfe.discretization import Discretization
from fastdfe.io_handlers import DummyVariant
from fastdfe.annotation import (
    DegeneracyAnnotation, MaximumLikelihoodAncestralAnnotation,
    JCSubstitutionModel, K2SubstitutionModel, KingmanPolarizationPrior,
)
from fastdfe import optimization as opt


# --------------------------------------------------------------------------- parametrization

def test_from_to_string_roundtrip():
    m = GammaExpParametrization()
    assert isinstance(_from_string('GammaExpParametrization'), GammaExpParametrization)
    assert _from_string(m) is m
    assert _to_string(m) == 'GammaExpParametrization'
    assert _to_string('GammaExpParametrization') == 'GammaExpParametrization'


def test_from_string_invalid_raises():
    with pytest.raises(ValueError):
        _from_string(123)


@pytest.mark.parametrize('cls', [
    GammaExpParametrization, DisplacedGammaParametrization, GammaDiscreteParametrization,
    DiscreteParametrization, DiscreteFractionalParametrization,
])
def test_dfe_pdf_cdf_finite(cls):
    model = cls()
    dfe = DFE(params=dict(model.x0), model=model)
    S = np.array([-100.0, -10.0, -1.0, 1.0, 10.0, 100.0])
    assert np.all(np.isfinite(dfe.pdf(S)))
    assert np.all(np.isfinite(dfe.cdf(S)))


def test_dfe_bootstrap_dfes_and_discretize():
    model = GammaExpParametrization()
    params = dict(model.x0)
    bins = np.array([-100.0, -1.0, 0.0, 1.0, 100.0])

    # no bootstraps
    dfe = DFE(params=params, model=model)
    assert dfe.get_bootstrap_dfes() == []
    centers, errors = dfe.discretize(bins, confidence_intervals=False)
    assert centers is not None and errors is None

    # with bootstraps
    boot = pd.DataFrame([params for _ in range(5)])
    dfe_b = DFE(params=params, model=model, bootstraps=boot)
    assert len(dfe_b.get_bootstrap_dfes()) == 5
    centers2, errors2 = dfe_b.discretize(bins, confidence_intervals=True)
    assert centers2 is not None and errors2 is not None


# --------------------------------------------------------------------------- spectrum

def test_spectrum_to_numpy():
    data = [10, 4, 3, 2, 1]
    np.testing.assert_array_equal(Spectrum(data).to_numpy(), np.array(data))


def test_spectrum_subsample_invalid_mode_raises():
    with pytest.raises(ValueError):
        Spectrum([10, 4, 3, 2, 1]).subsample(3, mode='nonsense')


def test_get_neutral_r_length_validation():
    # wrong length r
    with pytest.raises(ValueError):
        Spectrum.get_neutral(theta=1e-3, n_sites=1e4, n=5, r=[1.0, 1.0])
    # valid r of length n - 1
    sfs = Spectrum.get_neutral(theta=1e-3, n_sites=1e4, n=5, r=[1.0, 1.0, 1.0, 1.0])
    assert sfs.n == 5


def test_spectra_dunder_and_helpers():
    s = Spectra.from_dict({'a': [10, 2, 1], 'b': [8, 3, 1]})

    # __setitem__ and __iter__
    s['c'] = Spectrum([5, 1, 1])
    assert set(iter(s)) == {'a', 'b', 'c'}

    # get_empty -> all zeros, same shape
    empty = s.get_empty()
    assert empty.n_sites.sum() == 0

    # combine
    combined = s.combine(Spectra.from_dict({'d': [7, 2, 1]}))
    assert 'd' in combined.to_dict()

    # print and resample (smoke + structural)
    s.print()
    resampled = s.resample(seed=42)
    assert set(resampled.to_dict()) == set(s.to_dict())


# --------------------------------------------------------------------------- optimization scaling

# symlog uses bounds[0] as the (positive) linear threshold and bounds[1] as the boundary;
# the inverse is exact in the log region (value well above the threshold)
@pytest.mark.parametrize('scale,bounds,value', [
    ('lin', (-5.0, 5.0), 2.0),
    ('log', (1e-2, 1e2), 2.0),
    ('symlog', (1.0, 100.0), 50.0),
])
def test_scale_unscale_roundtrip(scale, bounds, value):
    scaled = opt.scale_value(value, bounds, scale)
    back = opt.unscale_value(scaled, bounds, scale)
    assert np.isclose(back, value, rtol=1e-6)


@pytest.mark.parametrize('scale,bounds', [
    ('lin', (-5.0, 5.0)),
    ('log', (1e-2, 1e2)),
    ('symlog', (1.0, 100.0)),
])
def test_unscale_bound(scale, bounds):
    lo, hi = opt.unscale_bound(bounds, scale)
    assert lo < hi


# --------------------------------------------------------------------------- filtration dummy branches

def _dummy():
    return DummyVariant(ref='A', pos=1, chrom='chr1')


def test_all_and_no_filtration_on_dummy():
    assert fd.AllFiltration().filter_site(_dummy()) is False
    assert fd.NoFiltration().filter_site(_dummy()) is True


def test_snp_filtration_dummy_branch():
    # a dummy (mono-allelic) variant is not an SNP, so SNPFiltration drops it
    assert fd.SNPFiltration().filter_site(_dummy()) is False


def test_deviant_outgroup_retain_monomorphic_semantics():
    # with retain_monomorphic, a mono-allelic (dummy) site is kept; without, it is dropped
    keep = fd.DeviantOutgroupFiltration(outgroups=['o1'], retain_monomorphic=True)
    assert keep.filter_site(_dummy()) is True
    drop = fd.DeviantOutgroupFiltration(outgroups=['o1'], retain_monomorphic=False)
    assert drop.filter_site(_dummy()) is False


def test_existing_outgroup_keeps_dummy():
    # ExistingOutgroupFiltration only checks outgroup presence; dummy (mono-allelic) sites are kept
    assert fd.ExistingOutgroupFiltration(outgroups=['o1']).filter_site(_dummy()) is True


# --------------------------------------------------------------------------- io_handlers

def test_gff_remove_overlaps():
    from fastdfe.io_handlers import GFFHandler

    # row 0 overlaps row 1 (next start 15 <= end 20); rows 1 and 2 do not overlap their successor
    df = pd.DataFrame({'start': [10, 15, 100], 'end': [20, 25, 120]})
    result = GFFHandler.remove_overlaps(df.copy())

    # the overlapping coding sequence (row 0) is dropped, the helper column is cleaned up
    assert result['start'].tolist() == [15, 100]
    assert 'overlap' not in result.columns


# --------------------------------------------------------------------------- discretization eq/hash

def _tiny_disc(**kw):
    return Discretization(
        n=4,
        intervals_del=(-100.0, -1e-5, 10),
        intervals_ben=(1e-5, 100.0, 10),
        intervals_h=(0.0, 1.0, 3),
        **kw,
    )


def test_discretization_eq_and_hash():
    d1 = _tiny_disc()
    d2 = _tiny_disc()
    assert d1 == d2
    assert hash(d1) == hash(d2)
    assert d1 != _tiny_disc(h=0.0)
    assert d1 != 'not a discretization'


# --------------------------------------------------------------------------- annotation components

def test_codon_degeneracy_matches_genetic_code():
    # Valine GTx: 1st/2nd positions non-degenerate, 3rd position 4-fold (all synonymous)
    assert DegeneracyAnnotation._get_degeneracy('GTT', 0) == 0
    assert DegeneracyAnnotation._get_degeneracy('GTT', 2) == 4
    # Phenylalanine TTT: 3rd position 2-fold (TTT/TTC=Phe, TTA/TTG=Leu)
    assert DegeneracyAnnotation._get_degeneracy('TTT', 2) == 2


def test_degeneracy_table_is_complete():
    table = DegeneracyAnnotation._get_degeneracy_table()
    assert len(table) == 64
    assert table['GTT'] == '004'                       # Val codon degeneracies
    assert set(''.join(table.values())) <= {'0', '2', '4'}


def test_get_base_string_index_mapping():
    cls = MaximumLikelihoodAncestralAnnotation
    np.testing.assert_array_equal(cls.get_base_string(np.array([0, 1, 2, 3])),
                                  np.array(['A', 'C', 'G', 'T']))
    # an invalid index (-1) maps to the '.' placeholder
    np.testing.assert_array_equal(cls.get_base_string(np.array([0, -1])), np.array(['A', '.']))
    assert cls.get_base_string(np.array([])).size == 0


def test_jc_substitution_model():
    m = JCSubstitutionModel()
    # one rate per branch: 2 * n_outgroups - 1 branches
    assert set(m.get_bounds(2)) == {'K0', 'K1', 'K2'}
    assert m.get_bound('K') == (1e-5, 10)
    # for a small branch rate, staying on the same base is more probable than changing
    p_same = m._get_prob(0, 0, 0, {'K0': 0.5})
    p_diff = m._get_prob(0, 1, 0, {'K0': 0.5})
    assert 0 <= p_diff < p_same <= 1


def test_k2_model_has_transition_transversion_ratio():
    assert 'k' in K2SubstitutionModel().bounds


def test_kingman_polarization_prior_is_symmetric():
    prior = KingmanPolarizationPrior()._get_prior(configs=pd.DataFrame(), n_ingroups=10)
    assert len(prior) == 11
    assert np.all(np.isfinite(prior))
    # Kingman prior is symmetric across the SFS: p[i] + p[n - i] = 1
    assert np.isclose(prior[3] + prior[7], 1.0)
