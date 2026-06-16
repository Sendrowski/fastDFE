# Top-level project Makefile. Standard targets:
#   make test        fast unit tests only (the default pytest tier)
#   make test-all    fast + inference tier, i.e. what runs on PRs (pytest -m "not slow")
#   make test-full   the entire suite incl. the slow/nightly tier
#   make docs        rebuild the HTML docs from scratch (clean + html)
#   make clean       remove the built docs
#
# The three test tiers (fast / inference / slow) are defined in pytest.ini; `make test`
# uses the default marker filter there, while the other targets override it. Tests are
# meant to run in the `dev-fastdfe` conda env. The data/precompute pipeline lives under
# snakemake/ and is driven directly via snakemake, not from here.

PYTEST ?= pytest

# Always run tests in parallel via pytest-xdist (declared in the dev/testing conda envs).
# `-n auto` uses one worker per physical core; override e.g. `make test XDIST="-n 4"` or
# disable with `make test XDIST=""` for a serial run (useful when debugging).
XDIST ?= -n auto

.PHONY: help test test-all test-full docs clean

help:
	@echo "Targets:"
	@echo "  make test       # fast unit tests (default pytest tier)"
	@echo "  make test-all   # fast + inference (PR tier): pytest -m 'not slow'"
	@echo "  make test-full  # entire suite incl. slow/nightly tier"
	@echo "  make docs       # rebuild HTML docs from scratch (clean + html)"
	@echo "  make clean      # remove the built docs"
	@echo "  (all test targets run with xdist: XDIST='$(XDIST)'; set XDIST= to disable)"

# --- tests (tiers defined in pytest.ini: fast / inference / slow) ---
test:
	$(PYTEST) $(XDIST)

test-all:
	$(PYTEST) $(XDIST) -m "not slow"

test-full:
	$(PYTEST) $(XDIST) -m "slow or not slow"

# --- docs (Sphinx + myst-nb; notebooks are not executed, nb_execution_mode='off') ---
# Full rebuild: wipe the build dir first so every page is regenerated from scratch.
docs:
	$(MAKE) -C docs clean
	$(MAKE) -C docs html
	@echo "Docs built -> docs/_build/html/index.html"

clean:
	$(MAKE) -C docs clean
