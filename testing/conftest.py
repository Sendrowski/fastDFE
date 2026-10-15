import os

import pytest
from sfsutils import Settings

collect_ignore = ["test_polydfe_wrapper.py"]

#: Time limit per test in seconds for each tier of ``pytest.ini``. A test marked with its own
#: ``pytest.mark.timeout`` keeps that limit.
TIMEOUTS = dict(fast=10, inference=60, slow=60)

#: Factor applied to the time limits on CI runners, which run the suite about ten times slower
TIMEOUT_FACTOR_CI = 5

# When running under pytest-xdist, each worker is a separate process and xdist already
# parallelizes across cores. Disable fastdfe's own multiprocessing (Settings.parallelize)
# so the inference/bootstrap pools don't oversubscribe cores on top of the xdist workers.
# xdist sets PYTEST_XDIST_WORKER (e.g. "gw0") in every worker process.
if os.environ.get("PYTEST_XDIST_WORKER"):
    Settings.parallelize = False


def pytest_collection_modifyitems(items):
    """
    Apply the time limit of its tier to every collected test.
    """
    for item in items:
        if item.get_closest_marker('timeout') is not None:
            continue

        tier = next((t for t in ('slow', 'inference') if item.get_closest_marker(t) is not None), 'fast')

        item.add_marker(pytest.mark.timeout(TIMEOUTS[tier] * (TIMEOUT_FACTOR_CI if os.environ.get('CI') else 1)))
