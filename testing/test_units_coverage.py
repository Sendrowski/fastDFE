"""
Fast unit tests that exercise pure-logic branches across the library to keep the unit-tier
coverage high without invoking the (resource-heavy) parsing/annotation or full inference paths.
All tests here must stay fast and deterministic (no VCF/genome/SLiM/polyDFE, no real optimization).

Spectrum/Spectra, filtration, GFF/IO and annotation (degeneracy, substitution models, polarization
priors) live in sfsutils since the 1.4.0 factor-out and are tested in its own suite, so they are not
covered here.
"""
import numpy as np
import pandas as pd
import pytest

import fastdfe as fd
from fastdfe.parametrization import (
    _from_string, _to_string, DFE, GammaExpParametrization, DiscreteParametrization,
    DiscreteFractionalParametrization, GammaDiscreteParametrization, DisplacedGammaParametrization,
)
from fastdfe.discretization import Discretization
from fastdfe import optimization as opt


# --------------------------------------------------------------------------- parametrization / DFE

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


# --------------------------------------------------------------------------- package helpers

def test_linear_operator_pickle_shim_roundtrip():
    import pickle
    from scipy.sparse.linalg import aslinearoperator, LinearOperator
    op = aslinearoperator(np.eye(3))
    # reproduce the scipy >= 1.18 trigger: a LinearOperator lacking ``_xp``; the shim must
    # let it (un)pickle instead of raising, which is what breaks bootstrapping otherwise
    op.__dict__.pop('_xp', None)
    restored = pickle.loads(pickle.dumps(op))
    assert isinstance(restored, LinearOperator) and restored.shape == (3, 3)


def test_linear_operator_pickle_shim_idempotent():
    # re-installing must be a no-op (hits the already-installed / version guards)
    fd._install_linear_operator_pickle_shim()
    fd._install_linear_operator_pickle_shim()


def test_linear_operator_pickle_shim_skips_old_scipy(monkeypatch):
    import scipy
    monkeypatch.setattr(scipy, '__version__', '1.16.0')
    fd._install_linear_operator_pickle_shim()  # returns at the version guard, no error


def test_linear_operator_pickle_shim_handles_unparseable_version(monkeypatch):
    import scipy
    monkeypatch.setattr(scipy, '__version__', 'not.a.version')
    fd._install_linear_operator_pickle_shim()  # version parse fails -> guarded no-op


def test_tqdm_logging_handler_emit(capsys):
    import logging
    handler = fd.TqdmLoggingHandler()
    handler.emit(logging.LogRecord('fastdfe', logging.INFO, __file__, 1, 'hello', None, None))
    # a record whose formatting raises routes through the error handler instead of propagating
    handler.emit(logging.LogRecord('fastdfe', logging.INFO, __file__, 1, 'val=%d', ('x',), None))


def test_colored_formatter_strips_package_and_wraps_color():
    import logging
    fmt = fd.ColoredFormatter('%(name)s:%(message)s')
    out = fmt.format(logging.LogRecord('fastdfe.parser', logging.WARNING, __file__, 1, 'hi', None, None))
    assert 'parser:hi' in out and out.startswith(fmt.colors['WARNING']) and out.endswith(fmt.reset)
