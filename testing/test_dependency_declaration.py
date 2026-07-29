"""
Guards on how dependencies are declared in pyproject.toml.

1.4.0 shipped with sfsutils-popgen reachable only through the ``vcf`` extra, so a plain
``pip install fastdfe`` produced a package that could not be imported. The cause is that poetry
matches ``[tool.poetry.extras]`` entries by package name alone: listing a package there tags every
entry of its dependency declaration, so a dependency that is both required and carries an optional
extra has its required entry pushed behind the ``extra ==`` marker too.

These read pyproject.toml rather than the installed metadata on purpose: a development environment
routinely carries a stale ``*.dist-info`` from an earlier install, which would make the assertions
describe a version that is no longer checked out.
"""
import tomllib
from pathlib import Path

import pytest

_PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"

# input backends that are legitimately optional, mirroring the sfsutils extras of the same names
_OPTIONAL_BACKENDS = {"cyvcf2", "zarr", "tskit"}


@pytest.fixture(scope="module")
def poetry():
    with _PYPROJECT.open("rb") as fh:
        return tomllib.load(fh)["tool"]["poetry"]


def _entries(spec):
    # a dependency is either a version string, a table, or a list of tables (multiple constraints)
    return spec if isinstance(spec, list) else [spec]


def test_extras_only_name_optional_dependencies(poetry):
    dependencies = poetry["dependencies"]

    for extra, names in poetry.get("extras", {}).items():
        for name in names:
            for entry in _entries(dependencies[name]):
                assert isinstance(entry, dict) and entry.get("optional"), (
                    f"{name!r} is named in the {extra!r} extra but has a non-optional entry; "
                    f"poetry marks every entry of {name!r} as extra-only, which hides the "
                    f"required one behind 'extra == \"{extra}\"'"
                )


def test_only_the_input_backends_are_optional(poetry):
    optional = {
        name for name, spec in poetry["dependencies"].items()
        if any(isinstance(entry, dict) and entry.get("optional") for entry in _entries(spec))
    }

    assert optional == _OPTIONAL_BACKENDS


def test_sfsutils_is_required_unconditionally(poetry):
    spec = poetry["dependencies"]["sfsutils-popgen"]

    assert not isinstance(spec, list), (
        "sfsutils-popgen must be a single constraint; a multiple-constraints declaration cannot "
        "be combined with an extra without hiding the required entry behind it"
    )
    assert isinstance(spec, str) or not spec.get("optional"), (
        "sfsutils-popgen is imported at fastdfe import time and must not be optional"
    )
