"""
Base inference class.
"""

__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"
__date__ = "2023-02-26"

import copy
import functools
import itertools
import json
import logging
import time
from typing import List, Optional, Dict, Literal, cast, Tuple

import jsonpickle
import multiprocess as mp
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numpy.linalg import norm
from scipy.optimize._optimize import OptimizeResult
from scipy.stats import chi2
from typing_extensions import Self

from . import parametrization, optimization
from .abstract_inference import AbstractInference, Inference
from .config import Config
from .discretization import Discretization
from .json_handlers import CustomEncoder
from .optimization import Optimization, flatten_dict, pack_params, expand_fixed, scale_values, unpack_shared
from .parametrization import Parametrization, from_string
from .spectrum import Spectrum, Spectra
from .spectrum import standard_kingman
from .visualization import Visualization

# get logger
logger = logging.getLogger('fastdfe')


class BaseInference(AbstractInference):
    """
    Base inference class for inferring the SFS given one neutral and one selected SFS.
    Note that BaseInference is by default seeded.

    .. warning::
        TODO add confidence intervals for inferred SFS.
    """

    #: Default parameters not connected to the DFE parametrization
    default_x0 = dict(
        eps=0.0
    )

    #: Default parameter bounds not connected to the DFE parametrization
    default_bounds = dict(
        eps=(0, 0.15)
    )

    #: Scales for the parameters not connected to the DFE parametrization
    default_scales = dict(
        eps='lin'
    )

    #: Default options for the MLE
    default_opts_mle = dict(
        # ftol=1e-20,
        # gtol=1e-20
    )

    def __init__(
            self,
            sfs_neut: Spectra | Spectrum,
            sfs_sel: Spectra | Spectrum,
            intervals_del: (float, float, int) = (-1.0e+8, -1.0e-5, 1000),
            intervals_ben: (float, float, int) = (1.0e-5, 1.0e4, 1000),
            integration_mode: Literal['midpoint', 'quad'] = 'midpoint',
            linearized: bool = True,
            model: Parametrization | str = 'GammaExpParametrization',
            seed: int = 0,
            x0: Dict[str, Dict[str, float]] = {},
            bounds: Dict[str, Tuple[float, float]] = {},
            scales: Dict[str, Literal['lin', 'log', 'symlog']] = {},
            loss_type: Literal['likelihood', 'L2'] = 'likelihood',
            opts_mle: dict = {},
            n_runs: int = 10,
            fixed_params: Dict[str, Dict[str, float]] = {},
            do_bootstrap: bool = False,
            n_bootstraps: int = 100,
            parallelize: bool = True,
            discretization: Discretization = None,
            optimization: Optimization = None,
            locked: bool = False,
            **kwargs
    ):
        """
        Create BaseInference instance.

        :param sfs_neut: The neutral SFS. Spectra | Spectrum
        :param sfs_sel: The selected SFS.
        :param intervals_del: ``(start, stop, n_interval)`` for deleterious population-scaled
        selection coefficients. The intervals will be log10-spaced.
        :param intervals_ben: Same as for intervals_del but for beneficial selection coefficients.
        :param model: Instance of DFEParametrization which parametrized the DFE
        :param seed: Seed for the random number generator.
        :param x0: Dictionary of initial values in the form {'all': {param: value}}
        :param bounds: Bounds for the optimization in the form {param: (lower, upper)}
        :param scales: Scales for the optimization in the form {param: scale}
        :param loss_type: Type of loss function to use for optimization.
        :param opts_mle: Options for the optimization.
        :param n_runs: Number of optimization runs. The first run will use the initial values if provided.
        :param fixed_params: Fixed parameters for the optimization.
        :param do_bootstrap: Whether to do bootstrapping.
        :param n_bootstraps: Number of bootstraps.
        :param parallelize: Whether to parallelize the bootstrapping.
        :param discretization: Discretization instance. Mainly intended for internal use.
        :param optimization: Optimization instance. Mainly intended for internal use.
        :param locked: Whether to lock the instance.
        :param kwargs: Additional arguments.
        """
        super().__init__()

        # assign neutral SFS
        if isinstance(sfs_neut, Spectra):
            #: Neutral SFS
            self.sfs_neut: Spectrum = sfs_neut.all
        else:
            # assume we have Spectrum object
            self.sfs_neut: Spectrum = sfs_neut

        # assign selected SFS
        if isinstance(sfs_sel, Spectra):
            #: Selected SFS
            self.sfs_sel: Spectrum = sfs_sel.all
        else:
            # assume we have Spectrum object
            self.sfs_sel: Spectrum = sfs_sel

        #: Sample size
        self.n: int = sfs_neut.n

        #: The DFE parametrization
        self.model: Parametrization = parametrization.from_string(model)

        if discretization is None:
            # create discretization instance
            #: Discretization instance
            self.discretization: Discretization = Discretization(
                n=self.n,
                intervals_del=intervals_del,
                intervals_ben=intervals_ben,
                integration_mode=integration_mode,
                linearized=linearized
            )

        else:
            # otherwise assign instance
            self.discretization: Discretization = discretization

        #: Estimate of theta from neutral SFS
        self.theta: float = self.sfs_neut.theta

        #: MLE estimates of the initial optimization
        self.params_mle: Optional[Dict[str, float]] = None

        #: Modelled MLE SFS
        self.sfs_mle: Optional[Spectrum] = None

        #: Likelihood of the MLE, this value may be updated after bootstrapping
        self.likelihood: Optional[float] = None

        #: Likelihoods of the different ML runs, controlled by ``n_runs``
        self.likelihoods: Optional[List[float]] = None

        #: Number of MLE runs to perform
        self.n_runs: int = n_runs

        #: Numerical optimization result
        self.result: Optional[OptimizeResult] = None

        # Bootstrap options

        #: Whether to do bootstrapping
        self.do_bootstrap: bool = do_bootstrap

        #: Number of bootstraps
        self.n_bootstraps: int = n_bootstraps

        #: Whether to parallelize the bootstrapping
        self.parallelize: bool = parallelize

        # expand 'all' type
        #: Fixed parameters
        self.fixed_params: Dict[str, Dict[str, float]] = expand_fixed(fixed_params, ['all'])

        # check that the fixed parameters are valid
        self.check_fixed_params_exist()

        #: parameter scales
        self.scales: Dict[str, Literal['lin', 'log', 'symlog']] = self.model.scales | self.default_scales | scales

        #: parameter bounds
        self.bounds: Dict[str, Tuple[float, float]] = self.model.bounds | self.default_bounds | bounds

        if optimization is None:
            # create optimization instance
            # merge with default values of inference and model
            #: Optimization instance
            self.optimization: Optimization = Optimization(
                bounds=self.bounds,
                scales=self.scales,
                opts_mle=self.default_opts_mle | opts_mle,
                loss_type=loss_type,
                param_names=self.param_names,
                parallelize=self.parallelize,
                fixed_params=fixed_params,
                seed=seed
            )
        else:
            # otherwise assign instance
            self.optimization: Optimization = optimization

        #: Initial values
        self.x0 = dict(all=self.model.x0 | self.default_x0 | (x0['all'] if 'all' in x0 else {}))

        #: Bootstrapped MLE parameter estimates
        self.bootstraps: Optional[pd.DataFrame] = None

        #: L2 norm of fit minus observed SFS
        self.L2_residual: Optional[float] = None

        #: Random number generator seed
        self.seed: int | None = seed

        #: Random generator instance
        self.rng = np.random.default_rng(seed=seed)

        #: Total execution time in seconds
        self.execution_time: float = 0

        #: Whether inferences can be run from the class itself
        self.locked: bool = locked

    def get_fixed_param_names(self) -> List[str]:
        """
        Get the names of the fixed parameters.
        """
        fixed = []

        for p in self.fixed_params.values():
            fixed += list(p.keys())

        return cast(List[str], np.unique(fixed).tolist())

    def check_fixed_params_exist(self):
        """
        Check that the fixed parameters are valid.
        """
        fixed = self.get_fixed_param_names()

        if not set(fixed).issubset(set(self.param_names)):
            non_valid_params = list(set(fixed) - set(self.param_names))

            raise ValueError(f'Fixed parameters {non_valid_params} is not a valid parameter '
                             f'for this configuration. Valid parameters are {self.param_names}.')

    def raise_if_locked(self):
        """
        Raise an error if this object is locked.

        :raises Exception:
        """
        if self.locked:
            raise Exception('This object is locked as inferences ought '
                            'not be run from the class itself')

    def run_if_required(self, *args, **kwargs) -> Optional[Spectrum]:
        """
        Run if not run yet.

        :param args: Arguments.
        :param kwargs: Keyword arguments.
        :return: DFE parametrization and modelled SFS.
        """
        if self.execution_time == 0:
            logger.debug('Inference needs to be run first, triggering run.')

            return self.run(*args, **kwargs)

        logger.debug('Inference already run, not running again.')

    @staticmethod
    def run_if_required_wrapper(func):
        """
        Decorator to run inference if required.

        :param func: Function to decorate.
        :return: Decorated function.
        """

        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            """
            Wrapper function.

            :param self: Inference instance
            :param args: Positional arguments
            :param kwargs: Keyword arguments
            :return: Function output
            """
            self.run_if_required(*args, **kwargs)

            return func(self, *args, **kwargs)

        return wrapper

    def run(
            self,
            do_bootstrap: bool = None,
            pbar: bool = None,
            **kwargs

    ) -> Spectrum:
        """
        Perform the DFE inference.

        :param pbar: Whether to show a progress bar.
        :param do_bootstrap: Whether to perform bootstrapping.
        :param kwargs: Keyword arguments.
        :return: Modelled SFS.
        """
        # check if locked
        self.raise_if_locked()

        # starting time of inference
        start_time = time.time()

        # update properties
        self.update_properties(
            do_bootstrap=do_bootstrap
        )

        # perform MLE
        logger.info(f'Starting numerical optimization of {self.n_runs} '
                    'independently initialized samples which are run ' +
                    ('in parallel.' if self.parallelize else 'sequentially.'))

        # Access cached property to trigger potential pre-computation
        # of linearization. This is necessary if the optimization is
        # parallelized so that the pre-computation does not have to be
        # performed in each thread.
        _ = self.discretization.dfe_to_sfs

        # perform numerical minimization
        result, params_mle = self.optimization.run(
            x0=self.x0,
            scales=self.scales,
            bounds=self.bounds,
            get_counts=self.get_counts(),
            n_runs=self.n_runs,
            pbar=pbar
        )

        # assign likelihoods
        self.likelihoods = self.optimization.likelihoods

        # unpack shared parameters
        params_mle = unpack_shared(params_mle)

        # normalize parameters
        params_mle['all'] = self.model.normalize(params_mle['all'])

        # assign optimization result and MLE parameters
        self.assign_result(result, params_mle['all'])

        # report on optimization result
        self.report_result(result, params_mle)

        # add execution time
        self.execution_time += time.time() - start_time

        # perform bootstrap if configured
        if self.do_bootstrap:
            self.bootstrap()

        return self.sfs_mle

    def get_counts(self) -> dict:
        """
        Get callback functions for modelling SFS counts from given parameters.

        :return: Callback functions for modelling SFS counts for each type.
        """
        return dict(all=lambda params: self.model_sfs(
            params,
            sfs_neut=self.sfs_neut,
            sfs_sel=self.sfs_sel
        ))

    def evaluate_likelihood(self, params: dict) -> float:
        """
        Get loss function.
        Note that the order of the parameters has to be the same as in
        params_mle. The types also need to be specified here. For a
        BaseInference objects, the mean we need to pass dict(all=...).

        :return: The likelihood.
        """
        x0_cached = self.optimization.x0
        self.optimization.x0 = params

        # prepare parameters
        params = pack_params(self.optimization.scale_values(flatten_dict(params)))

        lk = -self.optimization.get_loss_function(self.get_counts())(params)

        self.optimization.x0 = x0_cached

        return lk

    def assign_result(self, result: OptimizeResult, params_mle: dict):
        """
        Assign optimization result and MLE parameters.

        :param params_mle: MLE parameters.
        :param result: Optimization result.
        """
        self.result = result
        self.params_mle = params_mle
        self.likelihood = -result.fun

        # get SFS for MLE parameters
        counts_mle, _ = self.model_sfs(
            params=params_mle,
            sfs_neut=self.sfs_neut,
            sfs_sel=self.sfs_sel
        )

        # add monomorphic classes and create Spectrum object
        self.sfs_mle = Spectrum.from_polymorphic(counts_mle)

        # L2 norm of fit minus observed SFS
        self.L2_residual = norm(self.sfs_mle.polymorphic - self.sfs_sel.polymorphic, 2)

    @staticmethod
    def report_result(result: OptimizeResult, params: dict):
        """
        Inform on optimization result.

        :param params: MLE parameters.
        :param result: Optimization result.
        """
        # report on optimization result
        if result.success:
            logger.info(f"Successfully finished optimization after {result.nit} iterations "
                        f"and {result.nfev} function evaluations, obtaining a log-likelihood "
                        f"of -{result.fun}.")
        else:
            logger.warning("Numerical optimization did not terminate normally "
                           f"so result might be compromised. Number of iterations: {result.nit}."
                           "Please check the result property for more information.")

        logger.info(f"Inferred parameters: {flatten_dict(params)}.")

    def update_properties(self, **kwargs):
        """
        Update the properties of this class with the given dictionary
        given that its entries are not None.

        :param kwargs: Dictionary of properties to update.
        """
        for key, value in kwargs.items():
            if value is not None:
                setattr(self, key, value)

    def bootstrap(
            self, n_samples: int = None,
            parallelize: bool = None,
            update_likelihood: bool = True
    ) -> pd.DataFrame:
        """
        Perform the parametric bootstrap.

        :param n_samples: Number of bootstrap samples.
        :param parallelize: Whether to parallelize the bootstrap.
        :param update_likelihood: Whether to update the likelihood to be the mean of the bootstrap samples.
        :return: Dataframe with bootstrap results.
        """
        # check if locked
        self.raise_if_locked()

        # perform inference first if not done yet
        self.run_if_required()

        # update properties
        self.update_properties(
            n_bootstraps=n_samples,
            parallelize=parallelize
        )

        start_time = time.time()

        # parallelize computations if desired
        if self.parallelize:

            logger.info(f"Running {self.n_bootstraps} bootstrap samples "
                        f"in parallel on {mp.cpu_count()} cores.")

            # We need to assign new random states to the subprocesses.
            # Otherwise, they would all produce the same result.
            seeds = self.rng.integers(0, high=2 ** 32, size=self.n_bootstraps)

        else:
            logger.info(f"Running {self.n_bootstraps} bootstrap samples sequentially.")

            seeds = [None] * self.n_bootstraps

        # run bootstraps
        result = optimization.parallelize(
            func=self.run_bootstrap_sample,
            data=seeds,
            parallelize=self.parallelize,
            pbar=True
        )

        # number of successful runs
        n_success = np.sum([res.success for res in result[:, 0]])

        # issue warning if some runs did not finish successfully
        if n_success < self.n_bootstraps:
            logger.warning(f"{self.n_bootstraps - n_success} out of {self.n_bootstraps} bootstrap samples "
                           "did not terminate normally during numerical optimization. "
                           "The confidence intervals might thus be unreliable.")

        # dataframe of MLE estimates
        self.bootstraps = pd.DataFrame([r['all'] for r in result[:, 1]])

        # add estimates for alpha to the bootstraps
        self.add_alpha_to_bootstraps()

        # add execution time
        self.execution_time += time.time() - start_time

        # assign average likelihood of successful runs
        if update_likelihood:
            self.likelihood = np.mean([-res.fun for res in result[:, 0] if res.success] + [self.likelihood])

        return self.bootstraps

    def add_alpha_to_bootstraps(self):
        """
        Add estimates for alpha to the bootstraps.
        """
        logger.debug('Computing estimates for alpha.')

        # add alpha estimates
        self.bootstraps['alpha'] = self.bootstraps.apply(lambda r: self.get_alpha(dict(r)), axis=1)

    def resample_sfs(self, sfs: Spectrum, seed: int = None) -> Spectrum:
        """
        Resample SFS assuming independent Poisson counts.

        :param sfs: Spectrum to resample.
        :param seed: Seed for random number generator.
        :return: Resampled spectrum.
        """
        if seed is not None:
            rng = np.random.default_rng(seed=seed)
        else:
            rng = self.rng

        # resample polymorphic sites only
        polymorphic = rng.poisson(lam=sfs.polymorphic)

        return Spectrum.from_polydfe(
            polymorphic=polymorphic,
            n_sites=sfs.n_sites,
            n_div=sfs.n_div
        )

    def run_bootstrap_sample(self, seed: int = None) -> (OptimizeResult, dict):
        """
        Resample the observed selected SFS and rerun the optimization procedure.
        We take the MLE params as initial params here.

        :param seed: Seed for random number generator.
        :return: Optimization result and MLE parameters.
        """
        # resample spectra
        sfs_sel = self.resample_sfs(self.sfs_sel, seed=seed)
        sfs_neut = self.resample_sfs(self.sfs_neut, seed=seed)

        # perform numerical minimization
        result, params_mle = self.optimization.run(
            x0=dict(all=self.params_mle),
            scales=self.get_scales_linear(),
            bounds=self.get_bounds_linear(),
            n_runs=1,
            debug_iterations=False,
            print_info=False,
            get_counts=dict(all=lambda params: self.model_sfs(
                params,
                sfs_neut=sfs_neut,
                sfs_sel=sfs_sel
            ))
        )

        # unpack shared parameters
        params_mle = unpack_shared(params_mle)

        # normalize MLE estimates
        params_mle['all'] = self.model.normalize(params_mle['all'])

        return result, params_mle

    def get_scales_linear(self) -> Dict[str, Literal['lin']]:
        """
        Get linear scales for all parameters. We do this for the bootstraps as x0 should be close to MLE.

        :return: Dictionary of scales.
        """
        return cast(Dict[str, Literal['lin']], dict((p, 'lin') for p in self.scales.keys()))

    def get_bounds_linear(self) -> Dict[str, Tuple[float, float]]:
        """
        Get linear bounds for all parameters. We do this for the bootstraps as x0 should be close to MLE.

        :return: Dictionary of bounds.
        """
        scaled_bounds = {}

        for key, bounds in self.bounds.items():

            # for symlog we need to convert the bounds to linear scale
            if self.scales[key] == 'symlog':
                scaled_bounds[key] = (-bounds[1], bounds[1])
            else:
                scaled_bounds[key] = bounds

        return scaled_bounds

    def model_sfs(self, params: dict, sfs_neut: Spectrum, sfs_sel: Spectrum) -> (np.ndarray, np.ndarray):
        """
        Model the selected SFS from the given parameters.

        :param sfs_sel: Observed spectrum of selected sites.
        :param sfs_neut: Observed spectrum of neutral sites.
        :param params: Dictionary of parameters.
        :return: Array of modelled and observed counts
        """
        # infer selected SFS
        counts_modelled = sfs_neut.theta * self.discretization.model_selection_sfs(self.model, params)

        # multiply by mutational target size
        counts_modelled *= sfs_sel.n_sites

        # add contribution of demography and polarization error
        counts_modelled = self.add_demography(sfs_neut, counts_modelled, eps=params['eps'])

        return counts_modelled, sfs_sel.polymorphic

    @staticmethod
    def adjust_polarization(counts: np.ndarray, eps: float) -> np.ndarray:
        """
        Adjust the polarization of the given SFS where
        eps is the rate of wrong ancestral misidentification.

        :param counts: Polymorphic SFS counts to adjust.
        :param eps: Rate of wrong ancestral misidentification.
        :return: Adjusted SFS counts.
        """
        return (1 - eps) * counts + eps * counts[::-1]

    def add_demography(self, sfs_neut: Spectrum, counts_sel: np.ndarray, eps: float) -> np.ndarray:
        """
        Add the effect of demography to counts_sel by considering
        how counts_neut is perturbed relative to the standard coalescent.
        The polarization error is also included here.

        :param sfs_neut: Observed spectrum of neutral sites.
        :param counts_sel: Modelled counts of selected sites.
        :param eps: Rate of wrong ancestral misidentification.
        :return: Adjusted counts of selected sites.
        """
        # normalized counts of the standard coalescent
        counts_kingman = standard_kingman(self.n).polymorphic * sfs_neut.theta * sfs_neut.n_sites

        # apply polarization error to neutral and selected counts
        counts_neut_adjusted = self.adjust_polarization(sfs_neut.polymorphic, eps)
        counts_sel_adjusted = self.adjust_polarization(counts_sel, eps)

        # These counts transform the standard Kingman case to the observed
        # neutral SFS when multiplied and thus account for demography as we assume
        # the distortion in counts_neut to be due to demography only.
        r = counts_neut_adjusted / counts_kingman

        # adjust for demography and polarization error
        return r * counts_sel_adjusted

    @run_if_required_wrapper
    def plot_continuous(
            self,
            file: str = None,
            show: bool = True,
            intervals: np.ndarray = None,
            confidence_intervals: bool = True,
            ci_level: float = 0.05,
            bootstrap_type: Literal['percentile', 'bca'] = 'percentile',
            scale_density: bool = False,
            scale: Literal['lin', 'log', 'symlog'] = 'lin',
            title: str = 'DFE',
            ax: plt.Axes = None

    ) -> plt.Axes:
        """
        Plot continuous DFE.
        The special constants np.inf and -np.inf are also valid interval bounds.
        By default, the PDF is plotted as is. Due to the logarithmic scale on
        the x-axis, we may get a wrong intuition on how the mass is distributed,
        however. To get a better intuition, we can optionally scale the density
        by the x-axis interval size using ``scale_density = True``. This has the
        disadvantage that the density now changes for x, so that even a constant
        density will look warped.

        :param title: Plot title.
        :param bootstrap_type: Type of bootstrap to use for confidence intervals.
        :param ci_level: Confidence level for confidence intervals.
        :param confidence_intervals: Whether to plot confidence intervals.
        :param file: File to save plot to.
        :param show: Whether to show plot.
        :param scale: y-scale
        :param scale_density: Whether to scale the density by the x-axis interval size
        :param intervals: Array of interval boundaries yielding ``intervals.shape[0] - 1`` bins.
        :param ax: Axes object to plot on.
        :return: Axes object
        """
        if intervals is None:
            intervals = self.discretization.bins

        return Inference.plot_continuous(
            inferences=[self],
            file=file,
            show=show,
            intervals=intervals,
            confidence_intervals=confidence_intervals,
            ci_level=ci_level,
            bootstrap_type=bootstrap_type,
            title=title,
            scale=scale,
            scale_density=scale_density,
            ax=ax
        )

    @run_if_required_wrapper
    def plot_bucket_sizes(
            self,
            file: str = None,
            show: bool = True,
            title: str = 'bucket sizes',
            ax: plt.Axes = None

    ) -> plt.Axes:
        """
        Plot mass in each bucket for the MLE DFE.
        This can be used to check if the interval bounds and spacing
        are chosen appropriately.

        :param file: File to save plot to.
        :param show: Whether to show plot.
        :param title: Plot title.
        :param ax: Axes object to plot on.
        :return: Axes object
        """
        # evaluate at fixed parameters
        sizes = self.model.discretize(self.params_mle, self.discretization.bins)

        return Visualization.plot_buckets_sizes(
            n_intervals=self.discretization.n_intervals,
            bins=self.discretization.bins,
            sizes=sizes,
            title=title,
            file=file,
            show=show,
            ax=ax
        )

    @run_if_required_wrapper
    def plot_interval_density(
            self,
            file: str = None,
            show: bool = True,
            intervals: list | np.ndarray = np.array([-np.inf, -100, -10, -1, 0, 1, np.inf]),
            interval_labels: List[str] = None,
            color: str = 'C0',
            ax: plt.Axes = None

    ) -> plt.Axes:
        """
        Plot density of the discretization intervals chosen. Note that although this plot looks similar, this is
        not the DFE!

        :param color: Color of the bars.
        :param file: File to save plot to.
        :param show: Whether to show plot.
        :param interval_labels: Labels for the intervals.
        :param intervals: Array of interval boundaries yielding ``intervals.shape[0] - 1`` bins.
        :param ax: Axes object to plot on.
        :return: Axes object
        """
        # issue warning
        if not self.discretization.linearized:
            logger.warning('Note that the interval density is not important if the DFE was not linearized.')

        return Visualization.plot_interval_density(
            density=self.discretization.get_interval_density(intervals),
            **locals()
        )

    def plot_sfs_comparison(
            self,
            types: List[Literal['modelled', 'observed', 'modelled', 'neutral']] = ['modelled', 'observed'],
            labels: List[str] = None,
            file: str = None,
            show: bool = True,
            ax: plt.Axes = None

    ) -> plt.Axes:
        """
        Plot SFS comparison.

        :param file: File to save plot to.
        :param labels: Labels for the SFS.
        :param types: Types of SFS to plot.
        :param show: Whether to show plot.
        :param ax: Axes object to plot on.
        :return: Axes object
        """
        if 'modelled' in types:
            self.run_if_required()

        mapping = dict(
            observed=self.sfs_sel,
            selected=self.sfs_sel,
            modelled=self.sfs_mle,
            neutral=self.sfs_neut
        )

        return Visualization.plot_sfs_comparison(
            spectra=[mapping[t] for t in types],
            labels=types if labels is None else labels,
            file=file,
            show=show,
            ax=ax
        )

    def plot_observed_sfs(
            self,
            labels: List[str] = None,
            file: str = None,
            show: bool = True,
            ax: plt.Axes = None

    ) -> plt.Axes:
        """
        Plot neutral and selected SFS.

        :param file: File to save plot to.
        :param labels: Labels for the SFS.
        :param show: Whether to show plot.
        :param ax: Axes object to plot on.
        :return: Axes object
        """
        return self.plot_sfs_comparison(
            types=['neutral', 'selected'],
            labels=labels,
            file=file,
            show=show,
            ax=ax
        )

    @run_if_required_wrapper
    def plot_all(self, show: bool = True):
        """
        Plot everything.
        """
        self.plot_inferred_parameters(show=show)
        self.plot_discretized(show=show)
        self.plot_sfs_comparison(show=show)
        self.plot_continuous(show=show)
        self.plot_interval_density(show=show)
        self.plot_likelihoods(show=show)

    def get_errors_discretized_dfe(
            self,
            ci_level: float = 0.05,
            intervals: np.ndarray = np.array([-np.inf, -100, -10, -1, 0, 1, np.inf]),
            bootstrap_type: Literal['percentile', 'bca'] = 'percentile'
    ) -> np.ndarray:
        """
        Get the discretized DFE errors.

        :param ci_level: Confidence interval level.
        :param intervals: Array of interval boundaries yielding ``intervals.shape[0] - 1`` bins.
        :param bootstrap_type: Type of bootstrap to use.
        :return: Arrays of errors, confidence intervals, bootstraps, means and values
        """
        return Inference.get_errors_discretized_dfe(
            params=self.params_mle,
            bootstraps=self.bootstraps,
            model=self.model,
            ci_level=ci_level,
            intervals=intervals,
            bootstrap_type=bootstrap_type
        )

    @run_if_required_wrapper
    def plot_inferred_parameters(
            self,
            confidence_intervals: bool = True,
            ci_level: float = 0.05,
            bootstrap_type: Literal['percentile', 'bca'] = 'percentile',
            file: str = None,
            show: bool = True,
            title: str = 'parameter estimates',
            scale: Literal['lin', 'log', 'symlog'] = 'log',
            legend: bool = True,
            ax: plt.Axes = None,
            **kwargs
    ) -> plt.Axes:
        """
        Visualize the inferred parameters and their confidence intervals.

        :param scale: y-scale of the plot.
        :param title: Plot title.
        :param legend: Whether to show the legend.
        :param confidence_intervals: Whether to show confidence intervals.
        :param bootstrap_type: Type of bootstrap to use.
        :param ci_level: Confidence level for the confidence intervals.
        :param file: File to save plot to.
        :param show: Whether to show plot.
        :param ax: Axes object to plot on.
        :return: Axes object
        """
        return Inference.plot_inferred_parameters(
            inferences=[self],
            labels=['all'],
            file=file,
            show=show,
            title=title,
            ci_level=ci_level,
            confidence_intervals=confidence_intervals,
            bootstrap_type=bootstrap_type,
            scale=scale,
            ax=ax
        )

    @run_if_required_wrapper
    def plot_likelihoods(
            self,
            file: str = None,
            show: bool = True,
            title: str = 'likelihoods',
            scale: Literal['lin', 'log', 'symlog'] = 'lin',
            ax: plt.Axes = None,
            **kwargs
    ) -> plt.Axes:
        """
        Visualize the likelihoods of the optimization runs.

        :param scale: y-scale of the plot.
        :param title: Plot title.
        :param file: File to save plot to.
        :param show: Whether to show plot.
        :param ax: Axes object to plot on.
        :return: Axes object
        """
        return Visualization.plot_likelihoods(
            likelihoods=self.likelihoods,
            file=file,
            show=show,
            title=title,
            scale=scale,
            ax=ax
        )

    @staticmethod
    def lrt(ll_simple: float, ll_complex: float, df: int = 1) -> float:
        """
        Perform the likelihood ratio test (LRT).

        :param ll_simple: Log-likelihood of the simple model.
        :param ll_complex: Log-likelihood of the complex model.
        :param df: Degrees of freedom.
        :return: p-value
        """
        lr = -2 * (ll_simple - ll_complex)

        # issue info message
        logger.info(f"Simple model likelihood: {ll_simple}, "
                    f"complex model likelihood: {ll_complex}, "
                    f"degrees of freedom: {df}.")

        return chi2.sf(lr, df)

    def compare_nested_likelihoods(self, complex: 'BaseInference') -> float | None:
        """
        Perform likelihood ratio test with given more complex model.
        The given model's fixed parameters need to be a proper
        subset of this model's fixed parameters.

        :param complex: More complex model.
        :return: p-value
        """
        # optimization holds the flattened dictionary
        fixed_complex = set(complex.optimization.fixed_params.keys())
        fixed_simple = set(self.optimization.fixed_params.keys())

        # check that the models are nested
        if fixed_complex < fixed_simple:
            # determine degree of freedom
            d = len(fixed_simple - fixed_complex)
        else:
            return None

        return self.lrt(self.likelihood, complex.likelihood, d)

    @run_if_required_wrapper
    @functools.lru_cache
    def compare_nested_models(self, do_bootstrap: bool = True) -> (np.ndarray, Dict[str, 'BaseInference']):
        """
        Compare the various nested versions of the specified
        model using likelihood ratio tests.

        :param do_bootstrap: Whether to perform bootstrapping. This is recommended to get more accurate p-values.
        :return: Matrix of p-values, dict of base inference objects
        """

        # get sub-model specifications
        submodels_dfe = from_string(self.model).submodels
        submodels_outer = dict(
            no_anc=dict(eps=0),
            anc={}
        )

        # take outer product to get fixed parameters for each model
        inferences: Dict[str, BaseInference] = {}
        for p in itertools.product(submodels_dfe.keys(), submodels_outer.keys()):
            # create deep copy of object
            inference = copy.deepcopy(self)

            # disable bootstraps
            inference.do_bootstrap = do_bootstrap

            # dict of params to be fixed
            params = dict(all=submodels_dfe[p[0]] | submodels_outer[p[1]])

            # assign fixed parameters
            inference.set_fixed_params(params)

            # inform about fixed parameters
            logger.info(f'Holding parameters fixed to {params}.')

            # run inference
            inference.run()

            # define name
            name = '.'.join(p).rstrip('_')

            # save to dict
            inferences[name] = inference

        # number of models
        n = len(inferences)

        # create input matrix
        inputs = list(itertools.product(inferences.items(), inferences.items()))

        # create likelihood ratio matrix
        P = np.reshape(
            [cast(BaseInference, i[0][1]).compare_nested_likelihoods(cast(BaseInference, i[1][1])) for i in inputs],
            (n, n)
        )

        return P, inferences

    def plot_nested_likelihoods(
            self,
            file: str = None,
            show: bool = True,
            remove_empty: bool = False,
            transpose: bool = False,
            cmap: str = None,
            title: str = 'nested model comparison',
            ax: plt.Axes = None,
            do_bootstrap: bool = True

    ) -> plt.Axes:
        """
        Plot the p-values of nested likelihoods.

        :param file: File to save plot to.
        :param show: Whether to show plot.
        :param remove_empty: Whether to remove empty rows and columns.
        :param transpose: Whether to transpose the matrix.
        :param cmap: Colormap to use.
        :param title: Plot title.
        :param ax: Axes object to plot on.
        :param do_bootstrap: Whether to perform bootstrapping. This is recommended to get more accurate p-values.
        :return: Axes object
        """
        # get p-values and names
        P, inferences = self.compare_nested_models(do_bootstrap=do_bootstrap)

        # define labels
        labels_x = np.array(list(inferences.keys()))
        labels_y = np.array(list(inferences.keys()))

        if remove_empty:
            # remove empty columns
            nonempty_cols = np.sum(~np.equal(P, None), axis=0) != 0
            P = P[:, nonempty_cols]
            labels_x = labels_x[nonempty_cols]

            # remove empty rows
            nonempty_rows = np.sum(~np.equal(P, None), axis=1) != 0
            P = P[nonempty_rows]
            labels_y = labels_y[nonempty_rows]

        # take transpose if specified
        if transpose:
            P = P.T
            labels_x, labels_y = labels_y, labels_x

        return Visualization.plot_nested_likelihoods(
            P=P,
            labels_x=labels_x,
            labels_y=labels_y,
            file=file,
            show=show,
            cmap=cmap,
            title=title,
            ax=ax
        )

    def get_alpha(self, params: dict = None) -> float:
        """
        Get alpha, the proportion of beneficial non-synonymous substitutions.

        :param params: Parameters to use for calculation.
        """
        if params is None:
            params = self.params_mle

        return self.discretization.get_alpha(self.model, params)

    @functools.cached_property
    def alpha(self) -> float:
        """
        Cache alpha, the proportion of beneficial non-synonymous substitutions.
        """
        return self.get_alpha()

    def get_bootstrap_params(self) -> Dict[str, float]:
        """
        Get the parameters to be included in the bootstraps.

        :return: Parameters to be included in the bootstraps.
        """
        return self.params_mle | dict(alpha=self.alpha)

    def get_bootstrap_param_names(self) -> List[str]:
        """
        Get the parameters to be included in the bootstraps.

        :return: Parameters to be included in the bootstraps.
        """
        return list(self.get_bootstrap_params().keys())

    def get_optimized_param_names(self) -> List[str]:
        """
        Get the parameters names for the parameters that are optimized.

        :return: List of parameter names.
        """
        return list(set(flatten_dict(self.get_x0())) - set(flatten_dict(self.fixed_params)))

    def get_n_optimized(self) -> int:
        """
        Get the number of parameters that are optimized.

        :return: Number of parameters that are optimized.
        """
        return len(self.get_optimized_param_names())

    def create_config(self) -> 'Config':
        """
        Create a config object from the inference object.

        :return: Config object.
        """

        return Config(
            sfs_neut=Spectra.from_spectrum(self.sfs_neut),
            sfs_sel=Spectra.from_spectrum(self.sfs_sel),
            intervals_ben=self.discretization.intervals_ben,
            intervals_del=self.discretization.intervals_del,
            integration_mode=self.discretization.integration_mode,
            linearized=self.discretization.linearized,
            model=self.model.__class__.__name__,
            seed=self.seed,
            x0=self.x0,
            bounds=self.bounds,
            scales=self.scales,
            loss_type=self.optimization.loss_type,
            opts_mle=self.optimization.opts_mle,
            n_runs=self.n_runs,
            fixed_params=self.fixed_params,
            do_bootstrap=self.do_bootstrap,
            n_bootstraps=self.n_bootstraps,
            parallelize=self.parallelize
        )

    @classmethod
    def from_config(cls, config: 'Config') -> Self:
        """
        Load from config object.

        :param config: Config object.
        :return: Inference object.
        """
        return cls(**config.data)

    @classmethod
    def from_config_file(cls, file: str) -> Self:
        """
        Load from config file.

        :param file: Config file path.
        :return: Inference object.
        """
        from fastdfe import Config

        return cls.from_config(Config.from_file(file))

    def get_summary(self) -> 'InferenceResults':
        """
        Get summary.

        :return: Inference results.
        """
        return InferenceResults(self)

    def get_x0(self) -> Dict[str, Dict[str, float]]:
        """
        Get initial values.

        :return: Initial values.
        """
        # filter by parameter names
        return dict(all=dict((p, self.x0['all'][p]) for p in self.optimization.param_names))

    @property
    def param_names(self) -> List[str]:
        """
        Parameter names.

        :return: List of parameter names.
        """
        return self.model.param_names + ['eps']

    def set_fixed_params(self, params: Dict[str, Dict[str, float]]):
        """
        Set fixed parameters.

        :param params: Fixed parameters.
        """
        self.fixed_params = expand_fixed(params, ['all'])

        self.optimization.set_fixed_params(self.fixed_params)

    def to_json(self) -> str:
        """
        Serialize object.

        :return: JSON string
        """
        # using make_ref=True resulted in weird behaviour when unserializing.
        return jsonpickle.encode(self, indent=4, warn=True, make_refs=False)


class InferenceResults:
    def __init__(self, inference: BaseInference):
        """
        Inference results.

        :param inference: Inference object.
        """
        self.inference = inference

    def to_json(self) -> str:
        """
        Convert object to JSON.

        :return: JSON string.
        """
        return json.dumps(
            dict(
                likelihood=self.inference.likelihood,
                L2_residual=self.inference.L2_residual,
                params_mle=self.inference.params_mle,
                alpha=self.inference.alpha,
                result=self.inference.result,
                execution_time=self.inference.execution_time,
                sfs_mle=self.inference.sfs_mle,
                fixed_params=self.inference.optimization.fixed_params,
                bounds=self.inference.optimization.bounds,
                config=self.inference.create_config().to_dict(),
            ),
            indent=4,
            cls=CustomEncoder
        )

    def to_file(self, file: str):
        """
        Save object to file.

        :param file: File path.
        """
        with open(file, 'w') as fh:
            fh.write(self.to_json())
