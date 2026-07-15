"""
JSON handlers.

The ``Spectrum``/``Spectra``/``numpy``/``DataFrame`` jsonpickle handlers now live in
``sfsutils`` and are registered when ``sfsutils`` is imported. This module keeps only
the fastDFE-specific ``CustomEncoder``.
"""

__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"
__date__ = "2023-02-26"

import json
import logging

import numpy as np

from sfsutils.spectrum import Spectrum, Spectra

from .optimization import SharedParams, Covariate
from .parametrization import Parametrization

# configure logger
logger = logging.getLogger('fastdfe')


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
        from scipy.optimize._lbfgsb_py import LbfgsInvHessProduct

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
