"""
fastDFE package.
"""

__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"
__date__ = "2023-03-10"

__version__ = '1.3.3'

import logging
import sys

import jsonpickle
import numpy as np
import pandas as pd
from tqdm import tqdm


def _install_linear_operator_pickle_shim():
    """
    Work around ``scipy`` >= 1.18 raising ``KeyError: '_xp'`` when pickling a ``LinearOperator``
    without the ``_xp`` attribute (e.g. jsonpickle-restored results or L-BFGS-B's ``hess_inv``), which
    breaks bootstrapping. Wraps the (un)pickling hooks, falling back only on that error so it stays
    transparent wherever they already work (older scipy, or a future release that fixes the bug).
    """
    try:
        import scipy
        from scipy.sparse.linalg import LinearOperator
    except Exception:
        return

    # only the affected versions (>= 1.18); on later releases the wrappers stay transparent
    try:
        if tuple(int(p) for p in scipy.__version__.split('.')[:2]) < (1, 18):
            return
    except Exception:
        return

    if getattr(LinearOperator.__getstate__, '_fastdfe_shim', False):
        return

    orig_getstate = LinearOperator.__getstate__
    orig_setstate = LinearOperator.__setstate__

    def __getstate__(self):
        try:
            return orig_getstate(self)
        except KeyError:
            return self.__dict__.copy()

    def __setstate__(self, state):
        try:
            return orig_setstate(self, state)
        except KeyError:
            self.__dict__.update(state)

    __getstate__._fastdfe_shim = True
    LinearOperator.__getstate__ = __getstate__
    LinearOperator.__setstate__ = __setstate__


_install_linear_operator_pickle_shim()

from .json_handlers import DataframeHandler, SpectrumHandler, SpectraHandler, NumpyArrayHandler
from .spectrum import Spectrum, Spectra

# register custom handles
jsonpickle.handlers.registry.register(pd.DataFrame, DataframeHandler)
jsonpickle.handlers.registry.register(Spectrum, SpectrumHandler)
jsonpickle.handlers.registry.register(Spectra, SpectraHandler)
jsonpickle.handlers.registry.register(np.ndarray, NumpyArrayHandler)


class TqdmLoggingHandler(logging.Handler):
    """
    TQDM logging handler
    """

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

# def raise_on_warning(message, category, filename, lineno, file=None, line=None):
#    """
#    Raise exception on warning.
#    """
#    raise Exception(warnings.formatwarning(message, category, filename, lineno, line))

# warnings.showwarning = raise_on_warning

# load class from modules
from .parametrization import Parametrization, GammaExpParametrization, DiscreteParametrization, \
    GammaDiscreteParametrization, DisplacedGammaParametrization, DiscreteFractionalParametrization, DFE
from .discretization import Discretization
from .config import Config
from .settings import Settings
from .abstract_inference import Inference
from .base_inference import BaseInference, InferenceResult
from .joint_inference import JointInference, SharedParams
from .optimization import Covariate
from .simulation import Simulation
from .spectrum import Spectrum, Spectra
from .parser import Parser, Stratification, BaseTransitionStratification, BaseContextStratification, \
    DegeneracyStratification, TransitionTransversionStratification, AncestralBaseStratification, \
    SynonymyStratification, VEPStratification, SnpEffStratification, ContigStratification, ChunkedStratification, \
    RandomStratification, TargetSiteCounter
from .io_handlers import VCFHandler, FASTAHandler, GFFHandler, FileHandler
from .annotation import Annotator, Annotation, MaximumParsimonyAncestralAnnotation, SiteInfo, \
    MaximumLikelihoodAncestralAnnotation, DegeneracyAnnotation, SynonymyAnnotation, SubstitutionModel, \
    K2SubstitutionModel, JCSubstitutionModel, PolarizationPrior, KingmanPolarizationPrior, AdaptivePolarizationPrior
from .filtration import Filterer, Filtration, SNPFiltration, PolyAllelicFiltration, CodingSequenceFiltration, \
    SNVFiltration, DeviantOutgroupFiltration, AllFiltration, NoFiltration, BiasedGCConversionFiltration, \
    ExistingOutgroupFiltration, ContigFiltration, CpGFiltration

__all__ = [
    'Parametrization',
    'GammaExpParametrization',
    'DiscreteParametrization',
    'GammaDiscreteParametrization',
    'DisplacedGammaParametrization',
    'DiscreteFractionalParametrization',
    'DFE',
    'Discretization',
    'Config',
    'Settings',
    'Inference',
    'BaseInference',
    'JointInference',
    'InferenceResult',
    'Simulation',
    'SharedParams',
    'Covariate',
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
    'RandomStratification',
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
    'ContigFiltration',
    'CpGFiltration',
    'Filterer',
]
