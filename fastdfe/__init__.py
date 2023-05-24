"""
fastDFE package.
"""

__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"
__date__ = "2023-03-10"

__version__ = '0.1.6-beta'

import logging
import sys
import warnings

import jsonpickle
import numpy as np
import pandas as pd
import requests_cache
from tqdm import tqdm

from .json_handlers import DataframeHandler, SpectrumHandler, SpectraHandler, NumpyArrayHandler
from .spectrum import Spectrum, Spectra

# register custom handles
jsonpickle.handlers.registry.register(pd.DataFrame, DataframeHandler)
jsonpickle.handlers.registry.register(Spectrum, SpectrumHandler)
jsonpickle.handlers.registry.register(Spectra, SpectraHandler)
jsonpickle.handlers.registry.register(np.ndarray, NumpyArrayHandler)


class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        """
        Initialize the handler.

        :param level:
        """
        super().__init__(level)

    def emit(self, record):
        """
        Emit a record.
        """
        try:
            msg = self.format(record)

            # we write to stderr to avoid as the progress bar
            # to make the two work together
            tqdm.write(msg, file=sys.stderr)
            self.flush()
        except Exception:
            self.handleError(record)


class ColoredFormatter(logging.Formatter):
    def __init__(self, *args, **kwargs):
        """
        Initialize the formatter.
        """
        super().__init__(*args, **kwargs)
        self.colors = {
            "DEBUG": "\033[36m",  # Cyan
            "INFO": "\033[32m",  # Green
            "WARNING": "\033[33m",  # Yellow
            "ERROR": "\033[31m",  # Red
            "CRITICAL": "\033[31m",  # Red
        }
        self.reset = "\033[0m"

    def format(self, record):
        """
        Format the record.
        """
        log_color = self.colors.get(record.levelname, self.reset)

        formatted_record = super().format(record)

        return f"{log_color}{formatted_record}{self.reset}"


# configure logger
logger = logging.getLogger('fastdfe')

# don't propagate to the root logger
logger.propagate = False

# set to INFO by default
logger.setLevel(logging.INFO)

# let TQDM handle the logging
handler = TqdmLoggingHandler()

# define a Formatter with colors
formatter = ColoredFormatter('%(levelname)s:%(name)s:%(message)s')

handler.setFormatter(formatter)
logger.addHandler(handler)

# whether to disable the progress bar
disable_pbar = False

# install cache
requests_cache.install_cache('fastdfe_requests_cache', expire_after=3600 * 24)


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
from .joint_inference import JointInference, SharedParams
from .optimization import Covariate
from .visualization import Visualization
from .spectrum import Spectrum, Spectra
from .parser import Parser, Stratification, BaseTransitionStratification, BaseContextStratification, \
    DegeneracyStratification, TransitionTransversionStratification, AncestralBaseStratification, \
    SynonymyStratification, VEPStratification, SnpEffStratification
from .annotation import Annotator, Annotation, MaximumParsimonyAnnotation, DegeneracyAnnotation, SynonymyAnnotation
from .filtration import Filterer, Filtration, SNPFiltration, PolyAllelicFiltration, CodingSequenceFiltration, \
    SNVFiltration
from .vcf import VCFHandler

__all__ = [
    'Parametrization',
    'GammaExpParametrization',
    'DiscreteParametrization',
    'GammaDiscreteParametrization',
    'DisplacedGammaParametrization',
    'Config',
    'Inference',
    'BaseInference',
    'JointInference',
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
    'AncestralBaseStratification',
    'SynonymyStratification',
    'VEPStratification',
    'SnpEffStratification',
    'Annotator',
    'Annotation',
    'MaximumParsimonyAnnotation',
    'DegeneracyAnnotation',
    'SynonymyAnnotation',
    'Filtration',
    'CodingSequenceFiltration',
    'Filterer',
    'SNPFiltration',
    'SNVFiltration',
    'PolyAllelicFiltration',
    'Filterer',
    'VCFHandler',
]
