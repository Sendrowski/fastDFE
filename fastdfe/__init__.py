"""
fastDFE package.
"""

__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"
__date__ = "2023-03-10"

__version__ = '1.1.5'

import logging
import sys
import warnings

import jsonpickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from .json_handlers import DataframeHandler, SpectrumHandler, SpectraHandler, NumpyArrayHandler
from .spectrum import Spectrum, Spectra

# set the default figure size
plt.rcParams['figure.figsize'] = np.array([6.4, 4.8]) * 0.8

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

            # we write to stderr as the progress bar
            # to make the two work together
            tqdm.write(msg, file=sys.stderr)
            self.flush()
        except Exception:
            self.handleError(record)


class ColoredFormatter(logging.Formatter):
    """
    Colored formatter.
    """

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
        color = self.colors.get(record.levelname, self.reset)

        formatted = super().format(record)

        # remove package name
        formatted = formatted.replace(record.name, record.name.split('.')[-1])

        return f"{color}{formatted}{self.reset}"


# configure logger
logger = logging.getLogger('fastdfe')

# don't propagate to the root logger
logger.propagate = False

# set to INFO by default
logger.setLevel(logging.INFO)

# let TQDM handle the logging
handler = TqdmLoggingHandler()

# define a Formatter with colors
formatter = ColoredFormatter('%(levelname)s:%(name)s: %(message)s')

handler.setFormatter(formatter)
logger.addHandler(handler)


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
    GammaDiscreteParametrization, DisplacedGammaParametrization, DiscreteFractionalParametrization
from .config import Config
from .settings import Settings
from .abstract_inference import Inference
from .base_inference import BaseInference, InferenceResults
from .joint_inference import JointInference, SharedParams
from .optimization import Covariate
from .visualization import Visualization
from .spectrum import Spectrum, Spectra
from .parser import Parser, Stratification, BaseTransitionStratification, BaseContextStratification, \
    DegeneracyStratification, TransitionTransversionStratification, AncestralBaseStratification, \
    SynonymyStratification, VEPStratification, SnpEffStratification, ContigStratification, ChunkedStratification, \
    TargetSiteCounter
from .io_handlers import VCFHandler, FASTAHandler, GFFHandler, FileHandler
from .annotation import Annotator, Annotation, MaximumParsimonyAncestralAnnotation, SiteInfo, \
    MaximumLikelihoodAncestralAnnotation, DegeneracyAnnotation, SynonymyAnnotation, SubstitutionModel, \
    K2SubstitutionModel, JCSubstitutionModel, PolarizationPrior, KingmanPolarizationPrior, AdaptivePolarizationPrior
from .filtration import Filterer, Filtration, SNPFiltration, PolyAllelicFiltration, CodingSequenceFiltration, \
    SNVFiltration, DeviantOutgroupFiltration, AllFiltration, NoFiltration, BiasedGCConversionFiltration, \
    ExistingOutgroupFiltration

__all__ = [
    'Parametrization',
    'GammaExpParametrization',
    'DiscreteParametrization',
    'GammaDiscreteParametrization',
    'DisplacedGammaParametrization',
    'DiscreteFractionalParametrization',
    'Config',
    'Settings',
    'Inference',
    'BaseInference',
    'JointInference',
    'SharedParams',
    'Covariate',
    'Visualization',
    'Spectrum',
    'Spectra',
    'Parser',
    'FileHandler',
    'VCFHandler',
    'FASTAHandler',
    'GFFHandler',
    'TargetSiteCounter',
    'Stratification',
    'BaseTransitionStratification',
    'BaseContextStratification',
    'DegeneracyStratification',
    'TransitionTransversionStratification',
    'AncestralBaseStratification',
    'SynonymyStratification',
    'VEPStratification',
    'SnpEffStratification',
    'ContigStratification',
    'ChunkedStratification',
    'Annotator',
    'Annotation',
    'SiteInfo',
    'MaximumParsimonyAncestralAnnotation',
    'MaximumLikelihoodAncestralAnnotation',
    'PolarizationPrior',
    'KingmanPolarizationPrior',
    'AdaptivePolarizationPrior',
    'SubstitutionModel',
    'K2SubstitutionModel',
    'JCSubstitutionModel',
    'DegeneracyAnnotation',
    'SynonymyAnnotation',
    'Filtration',
    'CodingSequenceFiltration',
    'DeviantOutgroupFiltration',
    'AllFiltration',
    'NoFiltration',
    'Filterer',
    'SNPFiltration',
    'SNVFiltration',
    'PolyAllelicFiltration',
    'BiasedGCConversionFiltration',
    'ExistingOutgroupFiltration',
    'Filterer',
]
