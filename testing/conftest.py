import os

from fastdfe.settings import Settings

collect_ignore = ["test_polydfe_wrapper.py"]

# When running under pytest-xdist, each worker is a separate process and xdist already
# parallelizes across cores. Disable fastdfe's own multiprocessing (Settings.parallelize)
# so the inference/bootstrap pools don't oversubscribe cores on top of the xdist workers.
# xdist sets PYTEST_XDIST_WORKER (e.g. "gw0") in every worker process.
if os.environ.get("PYTEST_XDIST_WORKER"):
    Settings.parallelize = False
