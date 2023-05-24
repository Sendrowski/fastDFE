import logging
import os
import sys
from pathlib import Path
from unittest import TestCase as BaseTestCase

import matplotlib
import pytest
from matplotlib import pyplot as plt

import fastdfe

# only be verbose when running on Pycharm
if 'PYCHARM_HOSTED' not in os.environ:
    matplotlib.use('Agg')
    fastdfe.disable_pbar = True
    logging.getLogger('fastdfe').setLevel(logging.WARNING)
else:
    logging.getLogger('fastdfe').setLevel(logging.INFO)


def prioritize_installed_packages():
    """
    This function prioritizes installed packages over local packages.
    """
    # Get the current working directory
    cwd = str(Path().resolve())

    # Check if the current working directory is in sys.path
    if cwd in sys.path:
        # Remove the current working directory from sys.path
        sys.path.remove(cwd)
        # Append the current working directory to the end of sys.path
        sys.path.append(cwd)


class TestCase(BaseTestCase):
    @pytest.fixture(autouse=True)
    def cleanup(self):
        yield
        plt.close('all')


prioritize_installed_packages()
