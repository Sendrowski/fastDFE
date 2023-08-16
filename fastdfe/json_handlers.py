"""
JSON handlers.
"""

__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"
__date__ = "2023-02-26"

import json
import logging

import numpy as np
import pandas as pd
from jsonpickle.handlers import BaseHandler
from scipy.optimize._lbfgsb_py import LbfgsInvHessProduct

from .optimization import SharedParams, Covariate
from .parametrization import Parametrization
from .spectrum import Spectrum, Spectra

# configure logger
logger = logging.getLogger('fastdfe')


class NumpyArrayHandler(BaseHandler):
    """
    Handler for numpy arrays.
    """

    def flatten(self, x: np.ndarray, data: dict) -> dict:
        """
        Convert Spectrum to dict.

        :param x: Numpy array
        :param data: Dictionary
        :return: Simplified dictionary
        """
        return data | dict(data=x.tolist())

    def restore(self, data: dict) -> np.ndarray:
        """
        Restore Spectrum.

        :param data: Dictionary
        :return: Numpy array
        """
        return np.array(data['data'])


class SpectrumHandler(BaseHandler):
    """
    Handler for spectrum objects.
    """

    def flatten(self, sfs: Spectrum, data: dict) -> dict:
        """
        Convert Spectrum to dict.

        :param sfs: Spectrum object
        :param data: Dictionary
        :return: Simplified dictionary
        """
        return data | dict(data=sfs.to_list())

    def restore(self, data: dict) -> Spectrum:
        """
        Restore Spectrum.

        :param data: Dictionary
        :return: Spectrum object
        """
        return Spectrum.from_list(data['data'])


class SpectraHandler(BaseHandler):
    """
    Handler for spectra objects.
    """

    def flatten(self, sfs: Spectra, data: dict) -> dict:
        """
        Convert Spectra to dict.

        :param sfs: Spectra object
        :param data: Dictionary
        :return: Simplified dictionary
        """
        return data | dict(data=sfs.to_dict())

    def restore(self, data: dict) -> Spectra:
        """
        Restore Spectra.

        :param data: Dictionary
        :return: Spectra object
        """
        return Spectra.from_dict(data['data'])


class DataframeHandler(BaseHandler):
    """
    There were also problems with dataframes, hence the custom handler.
    """

    def flatten(self, df: pd.DataFrame, data: dict) -> dict:
        """
        Convert dataframe to dict.

        :param df: Dataframe
        :param data: Dictionary
        :return: Simplified dictionary
        """
        return data | dict(data=df.to_dict())

    def restore(self, data: dict) -> pd.DataFrame:
        """
        Restore dataframe.

        :param data: Dictionary
        :return: Dataframe
        """
        return pd.DataFrame(data['data'])


class CustomEncoder(json.JSONEncoder):
    """
    Convert numpy arrays and objects to lists and primitives.
    """

    def default(self, obj):
        """
        Convert numpy arrays and objects to lists and primitives.

        :param obj: Object
        :return: Simplified object
        """
        if isinstance(obj, Spectrum):
            return obj.to_list()

        if isinstance(obj, Spectra):
            return obj.to_dict()

        if isinstance(obj, np.ndarray):
            return obj.tolist()

        if isinstance(obj, np.int64):
            return int(obj)

        if isinstance(obj, Parametrization):
            return type(obj).__name__

        if isinstance(obj, SharedParams):
            return obj.__dict__

        if isinstance(obj, Covariate):
            # Only include 'param' and 'values' to avoid cluttering
            # the config file.
            return dict(param=obj.param, values=obj.values)

        # there were serialization problems with this object
        if isinstance(obj, LbfgsInvHessProduct):
            return str(obj)

        return json.JSONEncoder.default(self, obj)
