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


def test_name_to_label_keeps_mapped_label_for_negative_values():
    """
    A negative value prefixes the mapped label with a minus sign. The raw parameter name was returned,
    dropping the Greek letter.
    """
    from fastdfe.visualization import Visualization

    assert Visualization.name_to_label('alpha', ['alpha'], [0.5]) == '$α$'
    assert Visualization.name_to_label('alpha', ['alpha'], [-0.5]) == '$-α$'
    assert Visualization.name_to_label('S_d', ['S_d'], [-10]) == '$-S_d$'


def test_polydfe_result_requires_postprocessing_source(tmp_path):
    """
    Parsing polyDFE output without the path to polyDFE's R postprocessing script raises an error naming the
    argument.
    """
    from fastdfe.polydfe import PolyDFEResult

    with pytest.raises(ValueError, match='postprocessing_source'):
        PolyDFEResult(output_file=str(tmp_path / 'out.txt'))


def test_plot_continuous_band_has_colour_of_its_line():
    """
    The confidence band of a DFE has the colour of its line. The bands followed a colour cycle of their own,
    so a DFE without errors shifted the colours of all later bands.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba
    from fastdfe.visualization import Visualization

    bins = np.array([-100.0, -10.0, -1.0, 1.0])
    values = [np.array([0.2, 0.3, 0.5]), np.array([0.5, 0.3, 0.2])]
    errors = [None, np.full((2, 3), 0.05)]

    _, ax = plt.subplots()
    Visualization.plot_continuous(ax=ax, bins=bins, values=values, errors=errors, show=False)

    assert len(ax.collections) == 1
    np.testing.assert_allclose(ax.collections[0].get_facecolor()[0][:3], to_rgba(ax.lines[1].get_color())[:3])
    plt.close('all')

def test_plot_continuous_leaves_out_unbounded_intervals():
    """
    The density plot leaves out intervals with an infinite bound, whose density is not defined. They were
    drawn at a density of zero, whatever mass they held.
    """
    import matplotlib.pyplot as plt
    from fastdfe.visualization import Visualization

    bins = np.array([-np.inf, -10.0, -1.0, 1.0, np.inf])
    values = [np.array([0.4, 0.2, 0.1, 0.3])]

    _, ax = plt.subplots()
    Visualization.plot_continuous(ax=ax, bins=bins, values=values, show=False)

    density = ax.lines[0].get_ydata()

    assert np.isnan(density[[0, 3]]).all()
    np.testing.assert_allclose(density[1:3], [0.2 / 9, 0.1 / 2])
    plt.close('all')

def test_boxplot_of_a_single_parameter():
    """The boxplot of inferred parameters can be drawn for a single parameter, which raised a ``TypeError``."""
    import matplotlib.pyplot as plt
    from fastdfe.visualization import Visualization

    values = dict(a=pd.DataFrame(dict(b=[0.1, 0.2, 0.3])), c=pd.DataFrame(dict(b=[0.2, 0.3, 0.4])))

    Visualization.plot_inferred_parameters_boxplot(values=values, param_names=['b'], show=False)
    plt.close('all')

def test_plots_accept_labels_that_are_not_strings():
    """Labels that are not strings, such as numbers, can be passed to the bar plots. They raised an error."""
    import matplotlib.pyplot as plt
    from fastdfe.visualization import Visualization

    assert Visualization.get_hatch(0, [1, 2]) is None
    assert Visualization.get_color(2, [1, 2]) != Visualization.get_color(1, [1, 2])

    _, ax = plt.subplots()
    Visualization.plot_discretized(
        ax=ax, values=[np.array([0.5, 0.5]), np.array([0.3, 0.7])], labels=[2020, 2021],
        intervals=np.array([-np.inf, 0, np.inf]), show=False
    )
    plt.close('all')

def test_plot_helpers_accept_positional_axes_file_and_show(tmp_path):
    """
    The plot helpers treat the axes, file and ``show`` alike whether they are passed by position or by
    keyword. Passed by position, the axes raised a ``TypeError`` and the file was not written.
    """
    import matplotlib.pyplot as plt
    from fastdfe.visualization import Visualization

    file = str(tmp_path / 'scatter.png')

    _, ax = plt.subplots()

    assert Visualization.plot_scatter([1.0, 2.0, 3.0], file, False, ax) is ax
    assert (tmp_path / 'scatter.png').exists()
    plt.close('all')

def test_adjust_ticks_show_s_labels_positions_of_s_only():
    """
    The ticks showing selection coefficients are placed on positions of ``s`` only, the last one included.
    Ticks between two positions repeated the label of the lower one, and the last position had no label.
    """
    import matplotlib.pyplot as plt
    from fastdfe.visualization import Visualization

    s = np.array([-100.0, -10.0, -1.0, 1.0])

    _, ax = plt.subplots()
    ax.plot(np.arange(4), np.ones(4))
    ax.set_xticks([-0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5])

    Visualization.adjust_ticks_show_s(s, ax)

    assert list(ax.get_xticks()) == [0, 1, 2, 3]
    assert [label.get_text() for label in ax.get_xticklabels()] == ['-1e+02', '-1e+01', '-1e+00', '1e+00']
    plt.close('all')

def test_plot_scatter_symlog_scale():
    """
    ``plot_scatter`` uses a symmetric log axis for both ``log`` and ``symlog``, and a linear one for ``lin``.
    """
    import matplotlib.pyplot as plt
    from fastdfe.visualization import Visualization

    for scale, expected in [('lin', 'linear'), ('log', 'symlog'), ('symlog', 'symlog')]:
        _, ax = plt.subplots()
        Visualization.plot_scatter(values=[-1.0, 2.0, 3.0], file=None, show=False, ax=ax, scale=scale)

        assert ax.get_yscale() == expected
        plt.close('all')


def test_loss_function_follows_changing_observed_counts():
    """
    The loss function reuses the logarithm of the factorial of the observed counts between evaluations and
    recomputes it when the observed counts change.
    """
    from fastdfe.likelihood import Likelihood

    inf = fd.BaseInference(
        sfs_neut=fd.Spectrum([177130, 997, 441, 228, 156, 117, 114, 83, 105, 109, 652]),
        sfs_sel=fd.Spectrum([797939, 1329, 499, 265, 162, 104, 117, 90, 94, 119, 794])
    )
    optimization = inf.optimization
    optimization.x0 = inf.x0
    x = opt.pack_params(opt.scale_values(optimization.x0, optimization.bounds, optimization.scales))

    modelled = np.array([4.0, 3.0, 2.5])
    observed = [np.array([5.0, 3.0, 2.0]), np.array([7.0, 1.0, 4.0])]
    current = dict(i=0)

    loss = optimization.get_loss_function(dict(all=lambda params: np.array([modelled, observed[current['i']]])))

    for i in [0, 0, 1, 1, 0]:
        current['i'] = i

        assert loss(x) == -np.sum(Likelihood.log_poisson(mu=modelled, k=observed[i]))

    np.testing.assert_array_equal(
        Likelihood.log_poisson(mu=modelled, k=observed[0], log_factorial=Likelihood.log_factorial(observed[0])),
        Likelihood.log_poisson(mu=modelled, k=observed[0])
    )
