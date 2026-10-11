"""
JSON handlers.

The ``Spectrum``/``Spectra``/``numpy``/``DataFrame`` jsonpickle handlers now live in
``sfsutils`` and are registered when ``sfsutils`` is imported. This module keeps only
the fastDFE-specific ``CustomEncoder``.
"""

__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"
__date__ = "2023-02-26"

import inspect
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

        if isinstance(obj, np.generic):
            return obj.item()

        if isinstance(obj, Parametrization):
            settings = {}

            # include the intervals of a discrete parametrization if they differ from the default
            if hasattr(obj, 'intervals'):
                intervals = obj.intervals[1:-1]

                if not np.array_equal(intervals, inspect.signature(type(obj)).parameters['intervals'].default):
                    settings['intervals'] = intervals

            # include the bounds, scales and initial values that differ from the default
            default = type(obj)(**settings)

            for key in ('bounds', 'scales', 'x0'):
                if custom := {k: v for k, v in getattr(obj, key).items() if getattr(default, key)[k] != v}:
                    settings[key] = custom

            return dict(name=type(obj).__name__, **settings) if settings else type(obj).__name__

        if isinstance(obj, SharedParams):
            return obj.__dict__

        if isinstance(obj, Covariate):
            # include the remaining settings only if they differ from the default,
            # to avoid cluttering the config file
            settings = dict((k, getattr(obj, k)) for k in ('bounds', 'x0', 'bounds_scale'))

            return dict(param=obj.param, values=obj.values) | dict(
                (k, v) for k, v in settings.items() if v != getattr(Covariate, k)
            )

        # there were serialization problems with this object
        if isinstance(obj, LbfgsInvHessProduct):
            return str(obj)

        # functions have no JSON representation
        if callable(obj):
            return None

        return json.JSONEncoder.default(self, obj)
