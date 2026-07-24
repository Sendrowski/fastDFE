"""
Smoke tests for the sfsutils integration (as of the 1.4.0 factor-out).

These are deliberately lightweight and fixture-free: they check that fastdfe still exposes the SFS
parsing/spectra/annotation/filtration surface it re-exports from sfsutils, that the backward-compat
shim modules resolve to the very same classes (so jsonpickle can restore objects serialized under the
old ``fastdfe.*`` paths), and that a spectrum and the shared ``Settings`` behave. They are meant to
fail loudly if sfsutils is missing, too old, or has drifted in a way that breaks the re-exports.
"""
import importlib

import pytest

import fastdfe as fd
import sfsutils

# name -> sfsutils subpackage it must originate from
_REEXPORTS = {
    "Parser": "sfsutils.parser",
    "Spectrum": "sfsutils.spectrum",
    "Spectra": "sfsutils.spectrum",
    "Annotator": "sfsutils.annotation",
    "Filterer": "sfsutils.filtration",
    "Settings": "sfsutils.settings",
    "DegeneracyAnnotation": "sfsutils.annotation",
    "MaximumParsimonyAncestralAnnotation": "sfsutils.annotation",
    "DegeneracyStratification": "sfsutils.parser",
    "SynonymyStratification": "sfsutils.parser",
    "PolyAllelicFiltration": "sfsutils.filtration",
}

# fastdfe.<mod> compat shims and a class expected to resolve identically through them
_SHIMS = {
    "fastdfe.parser": "Parser",
    "fastdfe.spectrum": "Spectrum",
    "fastdfe.annotation": "Annotator",
    "fastdfe.filtration": "Filterer",
    "fastdfe.settings": "Settings",
    "fastdfe.io_handlers": None,
}


@pytest.mark.parametrize("name, module", sorted(_REEXPORTS.items()))
def test_reexport_present_and_from_sfsutils(name, module):
    """Each key class is importable from fastdfe and actually originates from sfsutils."""
    obj = getattr(fd, name, None)
    assert obj is not None, f"fastdfe does not re-export {name!r}"
    assert getattr(obj, "__module__", "").startswith("sfsutils"), \
        f"fastdfe.{name} came from {obj.__module__!r}, expected an sfsutils module"
    assert obj.__module__ == module, f"fastdfe.{name} is {obj.__module__}, expected {module}"


@pytest.mark.parametrize("shim, cls", sorted(_SHIMS.items()))
def test_compat_shim_resolves_to_same_object(shim, cls):
    """The backward-compat ``fastdfe.<mod>`` shims import and expose the same class objects.

    This is what lets jsonpickle restore objects serialized under the old ``fastdfe.*`` paths.
    """
    mod = importlib.import_module(shim)
    if cls is not None:
        assert getattr(mod, cls) is getattr(fd, cls), \
            f"{shim}.{cls} is not the same object as fastdfe.{cls}"


def test_old_serialized_paths_resolve():
    """Stratifications lived in fastdfe.parser before the factor-out; that path must still work."""
    import fastdfe.parser as p
    for name in ("Parser", "DegeneracyStratification", "SynonymyStratification"):
        assert getattr(p, name) is getattr(fd, name)


def test_spectrum_basic_ops():
    """A Spectrum constructs, reports its sample size, folds, and supports scalar arithmetic."""
    s = fd.Spectrum([100, 20, 10, 5, 0])
    assert s.n == 4
    assert s.to_list()[:2] == [100, 20]

    folded = s.fold()
    assert isinstance(folded, fd.Spectrum)

    scaled = s * 2
    assert isinstance(scaled, fd.Spectrum)
    assert scaled.to_list()[1] == 40


def test_spectra_from_spectra():
    """Spectra assembles from named Spectrum objects and round-trips the labels."""
    s = fd.Spectrum([100, 20, 10, 5, 0])
    spectra = fd.Spectra.from_spectra({"a": s, "b": s * 2})
    assert set(spectra.types) == {"a", "b"}


def test_settings_is_shared_and_toggles():
    """fastdfe.Settings is the very same class as sfsutils' and toggling a flag works."""
    assert fd.Settings is sfsutils.settings.Settings

    original = fd.Settings.disable_pbar
    try:
        fd.Settings.disable_pbar = not original
        assert sfsutils.Settings.disable_pbar == (not original)
    finally:
        fd.Settings.disable_pbar = original


def test_sfsutils_version_is_compatible():
    parts = tuple(int(x) for x in sfsutils.__version__.split(".")[:2])
    assert parts >= (1, 0), f"sfsutils {sfsutils.__version__} is older than the required 1.0.0"
