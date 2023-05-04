"""
fastDFE package.
"""

__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"
__date__ = "2023-03-10"

__version__ = '0.1.4-beta'

import logging
import sys
import warnings

import jsonpickle
import numpy as np
import pandas as pd

from .json_handlers import DataframeHandler, SpectrumHandler, SpectraHandler, NumpyArrayHandler
from .spectrum import Spectrum, Spectra

# register custom handles
jsonpickle.handlers.registry.register(pd.DataFrame, DataframeHandler)
jsonpickle.handlers.registry.register(Spectrum, SpectrumHandler)
jsonpickle.handlers.registry.register(Spectra, SpectraHandler)
jsonpickle.handlers.registry.register(np.ndarray, NumpyArrayHandler)

# configure logger
logger = logging.getLogger('fastdfe')
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter('%(levelname)s:%(name)s:%(message)s'))
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# whether to disable the progress bar
disable_pbar = False


def raise_on_warning(message, category, filename, lineno, file=None, line=None):
    """
    Raise exception on warning.
    """
    raise Exception(warnings.formatwarning(message, category, filename, lineno, line))


# warnings.showwarning = raise_on_warning

# configure default colormap
# plt.rcParams['image.cmap'] = 'Dark2'
# plt.rcParams['axes.prop_cycle'] = cycler('color', plt.get_cmap('Dark2').colors)

# load class from modules
from .parametrization import Parametrization, GammaExpParametrization, DiscreteParametrization, \
    GammaDiscreteParametrization, DisplacedGammaParametrization
from .config import Config
from .abstract_inference import Inference
from .base_inference import BaseInference, InferenceResults
from .shared_inference import SharedInference, SharedParams
from .optimization import Covariate
from .visualization import Visualization
from .spectrum import Spectrum, Spectra
from .parser import Parser, BaseTransitionStratification, BaseContextStratification, DegeneracyStratification, \
    TransitionTransversionStratification, ReferenceBaseStratification

__all__ = [
    'Parametrization',
    'GammaExpParametrization',
    'DiscreteParametrization',
    'GammaDiscreteParametrization',
    'DisplacedGammaParametrization',
    'Config',
    'Inference',
    'BaseInference',
    'SharedInference',
    'SharedParams',
    'Covariate',
    'Visualization',
    'Spectrum',
    'Spectra',
    'Parser',
    'BaseTransitionStratification',
    'BaseContextStratification',
    'DegeneracyStratification',
    'TransitionTransversionStratification',
    'ReferenceBaseStratification'
]
