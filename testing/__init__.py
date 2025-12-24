"""
Initialization for the testing module.
"""
import logging
import os
import sys
from pathlib import Path
from unittest import TestCase as BaseTestCase

import matplotlib
import numpy as np
import pytest
from matplotlib import pyplot as plt


def prioritize_installed_packages():
    """
    This function prioritizes installed packages over local packages.
    """
    # Get the current working directory
    cwd = str(Path().resolve())

    # Check if the current working directory is in sys.path
    if cwd in sys.path:
        # Remove the current working directory from sys.path
        sys.path = [p for p in sys.path if p != cwd]
        # Append the current working directory to the end of sys.path
        sys.path.append(cwd)


# run before importing fastdfe
prioritize_installed_packages()

import fastdfe

logger = logging.getLogger('fastdfe')

logger.info(sys.version)
logger.info(f"Running tests for {fastdfe.__file__}")
logger.info(f"fastdfe version: {fastdfe.__version__}")

# only show plots when running in PyCharm
if 'PYCHARM_HOSTED' not in os.environ:
    matplotlib.use('Agg')
    #fastdfe.Settings.disable_pbar = True
    #logger.setLevel(logging.WARNING)
else:
    logger.setLevel(logging.INFO)

# check for PARALLELIZE environment variable
if 'PARALLELIZE' in os.environ and os.environ['PARALLELIZE'].lower() == 'false':
    fastdfe.Settings.parallelize = False
    logger.info("Parallelization disabled.")

# create scratch directory if it doesn't exist
if not os.path.exists('scratch'):
    os.makedirs('scratch')


class TestCase(BaseTestCase):
    @pytest.fixture(autouse=True)
    def cleanup(self):
        """

        """
        yield
        plt.close('all')

    @staticmethod
    def rel_diff(a, b, eps=1e-12):
        """
        Compute the relative difference between a and b.
        """
        return np.abs(a - b) / (np.abs(a) + np.abs(b) + eps)
