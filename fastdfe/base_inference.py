"""
Base inference class.
"""

__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"
__date__ = "2023-02-26"

import copy
import functools
import itertools
import logging
import time
from math import comb
from typing import List, Optional, Dict, Literal, cast, Tuple, Sequence, Callable

import jsonpickle
import multiprocess as mp
import numpy as np
import pandas as pd
from numpy.linalg import norm
from scipy.stats import chi2
from typing_extensions import Self

from . import optimization
from . import parametrization
from .abstract_inference import AbstractInference, Inference
from .config import Config
from .discretization import Discretization
from .optimization import Optimization, flatten_dict, pack_params, expand_fixed, unpack_shared
from .parametrization import Parametrization, _from_string, DFE
from .spectrum import Spectrum, Spectra
from .spectrum import standard_kingman
from .utils import Serializable

# get logger
logger = logging.getLogger('fastdfe')


class BaseInference(AbstractInference):
    """
    Base inference class for inferring the SFS given one neutral and one selected SFS.
    Note that BaseInference is by default seeded.

    Example usage:

    ::

        import fastdfe as fd

        sfs_neut = fd.Spectrum([177130, 997, 441, 228, 156, 117, 114, 83, 105, 109, 652])
        sfs_sel = fd.Spectrum([797939, 1329, 499, 265, 162, 104, 117, 90, 94, 119, 794])

        # create inference object
        inf = fd.BaseInference(
            sfs_neut=sfs_neut,
            sfs_sel=sfs_sel,
            n_runs=10,
            n_bootstraps=100,
            do_bootstrap=True
        )

        # run inference
        inf.run()

        # plot discretized DFE
        inf.plot_discretized()

        # plot SFS comparison
        inf.plot_sfs_comparison()

    """

    #: Default parameters not connected to the DFE parametrization
    _default_x0 = dict(
        eps=0.0,
        h=0.5
    )

    #: Default parameter bounds not connected to the DFE parametrization
    _default_bounds = dict(
        eps=(0, 0.15),
        h=(0, 1)
    )

    #: Default scales for the parameters not connected to the DFE parametrization
    _default_scales = dict(
        eps='lin',
        h='lin'
    )

    #: Default options for the maximum likelihood optimization
    _default_opts_mle = dict(
        # ftol=1e-10,
        # gtol=1e-10
    )

    # Optimization runs. Static for backward compatibility.
    runs: Optional[pd.DataFrame] = None

    def __init__(
            self,
            sfs_neut: Spectra | Spectrum,
            sfs_sel: Spectra | Spectrum,
            intervals_del: Tuple[float, float, int] = (-1.0e+8, -1.0e-5, 1000),
            intervals_ben: Tuple[float, float, int] = (1.0e-5, 1.0e4, 1000),
            intervals_h: Tuple[float, float, int] = (0.0, 1.0, 21),
            h_callback: Callable[[np.ndarray], np.ndarray] = None,
            integration_mode: Literal['midpoint', 'quad'] = 'midpoint',
            linearized: bool = True,
            model: Parametrization | str = 'GammaExpParametrization',
            seed: int = 0,
            x0: Dict[str, Dict[str, float]] = {},
            bounds: Dict[str, Tuple[float, float]] = {},
            scales: Dict[str, Literal['lin', 'log', 'symlog']] = {},
            loss_type: Literal['likelihood', 'L2'] = 'likelihood',
            opts_mle: dict = {},
            method_mle: str = 'L-BFGS-B',
            n_runs: int = 10,
            fixed_params: Dict[str, Dict[str, float]] = None,
            do_bootstrap: bool = True,
            n_bootstraps: int = 100,
            n_bootstrap_retries: int = 2,
            parallelize: bool = True,
            folded: bool = None,
            discretization: Discretization = None,
            optimization: Optimization = None,
            locked: bool = False,
            **kwargs
    ):
        """
        Create BaseInference instance.

        :param sfs_neut: The neutral SFS. Note that we require monomorphic counts to be specified in order to infer
            the mutation rate. If a :class:`~fastdfe.spectrum.Spectra` object with more than one SFS is provided, the
            ``all`` attribute will be used.
        :param sfs_sel: The selected SFS. Note that we require monomorphic counts to be specified in order to infer
            the mutation rate. If a :class:`~fastdfe.spectrum.Spectra` object with more than one SFS is provided, the
            ``all`` attribute will be used.
        :param intervals_del: ``(start, stop, n_interval)`` for deleterious population-scaled
            selection coefficients. The intervals will be log10-spaced. Decreasing the number of intervals to ``100``
            provides nearly identical results while increasing speed, especially when precomputing across dominance
            coefficients.
        :param intervals_ben: Same as ``intervals_del`` but for positive selection coefficients. Decreasing the number
            of intervals to ``100`` provides nearly identical results while increasing speed, especially when
            precomputing across dominance coefficients.
        :param intervals_h: ``(start, stop, n_interval)`` for dominance coefficients which are linearly spaced.
            This is only used when inferring dominance coefficients. Values of ``h`` between the edges will be
            interpolated linearly.
        :param h_callback: A function mapping the scalar parameter ``h`` and the array of selection
            coefficients ``S`` to dominance coefficients of the same shape, allowing models where ``h``
            depends on ``S``. The default is ``lambda h, S: np.full_like(S, h)``, keeping ``h`` constant.
            Expected allele counts for a given dominance value are obtained by linear interpolation
            between precomputed values in ``intervals_h``. The inferred parameter is still named ``h``,
            even if transformed by ``h_callback``, and its bounds, scales, and initial values can be set
            via ``bounds``, ``scales``, and ``x0``. The fitness of heterozygotes and mutation homozygotes is defined as
            ``1 + 2hs`` and ``1 + 2s``, respectively.
        :param integration_mode: Integration mode when computing expected SFS under semidominance.
            ``quad`` is not recommended.
        :param linearized: Whether to discretize and cache the linearized integral mapping DFE to SFS or use
            ``scipy.integrate.quad`` in each call. ``False`` not recommended.
        :param model: Instance of DFEParametrization which parametrized the DFE
        :param seed: Seed for the random number generator. Use ``None`` for no seed.
        :param x0: Dictionary of initial values in the form ``{'all': {param: value}}``
        :param bounds: Bounds for the optimization in the form ``{param: (lower, upper)}``
        :param scales: Scales for the optimization in the form ``{param: scale}``
        :param loss_type: Type of loss function to use for optimization.
        :param opts_mle: Options for the optimization.
        :param method_mle: Method to use for optimization. See ``scipy.optimize.minimize`` for available methods.
        :param n_runs: Number of independent optimization runs out of which the best one is chosen. The first run
            will use the initial values if specified. Consider increasing this number if the optimization does not
            produce good results.
        :param fixed_params: Parameters kept constant during optimization, given as ``{'all': {param: value}}``.
            By default, the ancestral misidentification rate ``eps`` is fixed to zero, ``h`` to 0.5, and the DFE
            parameters are fixed so that only the deleterious component of the DFE is estimated.
        :param do_bootstrap: Whether to do bootstrapping.
        :param n_bootstraps: Number of bootstraps.
        :param n_bootstrap_retries: Number of optimization runs for each bootstrap sample. This parameter previously
            defined the number of retries per bootstrap sample when subsequent runs failed, but now it defines the
            total number of runs per bootstrap sample, taking the most likely one.
        :param pbar: Whether to show a progress bar.
        :param parallelize: Whether to parallelize computations.
        :param folded: Whether the SFS are folded. If not specified, the SFS will be folded if both of the given
            SFS appear to be folded.
        :param discretization: Discretization instance. Mainly intended for internal use.
        :param optimization: Optimization instance. Mainly intended for internal use.
        :param locked: Whether inferences can be run from the class itself. Used internally.
        :param kwargs: Additional keyword arguments which are ignored.
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

        # check that we have monomorphic counts
        if self.sfs_neut.to_list()[0] == 0 or self.sfs_sel.to_list()[0] == 0:
            raise ValueError('Some of the provided SFS have zero ancestral monomorphic counts. '
                             'Note that we require monomorphic counts in order to infer the mutation rate.')

        # check that we have polymorphic counts
        if self.sfs_neut.n_polymorphic == 0 or self.sfs_neut.n_polymorphic == 0:
            raise ValueError('Some of the provided SFS have zero polymorphic counts. '
                             'Note that we require polymorphic counts in order to infer the DFE.')

        if self.sfs_neut.n != self.sfs_sel.n:
            raise ValueError('The sample sizes of the neutral and selected SFS must be equal.')

        #: The DFE parametrization
        self.model: Parametrization = parametrization._from_string(model)

        if folded is None:
            #: Whether the SFS are folded
            self.folded: bool = self.sfs_sel.is_folded() and self.sfs_neut.is_folded()
        else:
            self.folded = folded

        #: Sample size
        self.n: int = sfs_neut.n

        if self.n > 400:
            raise ValueError(
                f"Population sample size n={self.n} above allowed maximum of 400. "
                f"Note that high sample sizes (e.g. n > 50) don't carry much additional "
                f"information over moderate sample sizes and don't necessarily improve DFE inference."
            )

        if fixed_params is None:
            fixed_params = {'all': {'eps': 0, 'h': 0.5} | self.model.submodels['dele']}

        if not 'all' in fixed_params:
            fixed_params['all'] = {}

        # expand 'all' type
        #: Fixed parameters
        self.fixed_params: Dict[str, Dict[str, float]] = expand_fixed(fixed_params, ['all'])

        # check that the fixed parameters are valid
        self.check_fixed_params_exist()

        if discretization is None:
            # create discretization instance
            #: Discretization instance
            self.discretization: Discretization = Discretization(
                n=self.n,
                h=self.fixed_params.get('all', {}).get('h', None),
                h_callback=h_callback,
                intervals_del=intervals_del,
                intervals_ben=intervals_ben,
                intervals_h=intervals_h,
                integration_mode=integration_mode,
                linearized=linearized,
                parallelize=parallelize
            )

        else:
            # otherwise assign instance
            self.discretization: Discretization = discretization

        #: Estimate of theta from neutral SFS
        self.theta: float = self.sfs_neut.theta

        # raise warning if theta is unusually large
        if self.theta > 0.05 or self.sfs_sel.theta > 0.05:
            self._logger.warning("Mutation rate appears unusually high. "
                                 "This is a reminder to provide monomorphic site counts.")

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

        #: Dataframe holding information on all optimization runs
        self.runs: Optional[pd.DataFrame] = None

        #: Numerical optimization result
        self.result: Optional['scipy.optimize.OptimizeResult'] = None

        # Bootstrap options

        #: Whether to do bootstrapping
        self.do_bootstrap: bool = do_bootstrap

        #: Number of bootstraps
        self.n_bootstraps: int = n_bootstraps

        #: Number of optimization runs for each bootstrap sample
        self.n_bootstrap_retries: int = n_bootstrap_retries

        #: Whether to parallelize computations
        self.parallelize: bool = parallelize

        #: Parameter scales
        self.scales: Dict[str, Literal['lin', 'log', 'symlog']] = self.model.scales | self._default_scales | scales

        #: Parameter bounds
        self.bounds: Dict[str, Tuple[float, float]] = self.model.bounds | self._default_bounds | bounds

        if optimization is None:
            # create optimization instance
            # merge with default values of inference and model
            #: Optimization instance
            self.optimization: Optimization = Optimization(
                bounds=self.bounds,
                scales=self.scales,
                opts_mle=self._default_opts_mle | opts_mle,
                method_mle=method_mle,
                loss_type=loss_type,
                param_names=self.param_names,
                parallelize=self.parallelize,
                fixed_params=fixed_params,
                seed=seed
            )
        else:
            # otherwise assign instance
            self.optimization: Optimization = optimization

        # warn if folded is true, and we estimate the full DFE
        if self.folded:
            fixed_dele = set(self.model.submodels['dele'].keys()) if 'dele' in self.model.submodels else set()
            fixed = set(self.fixed_params['all'].keys()) if 'all' in self.fixed_params else set()

            if not fixed_dele.issubset(fixed):
                self._logger.warning("You are estimating the full DFE, but the SFS are folded. "
                                     "This is not recommend as the folded SFS contains little information "
                                     "on beneficial mutations.")

        #: Initial values
        self.x0: Dict[str, Dict[str, float]] = dict(
            all=self.model.x0 | self._default_x0 | (x0['all'] if 'all' in x0 else {})
        )

        #: Bootstrapped MLE parameter estimates
        self.bootstraps: Optional[pd.DataFrame] = None

        #: Bootstrap optimization results
        self.bootstrap_results: Optional[List['scipy.optimize.OptimizeResult']] = None

        #: L2 norm of differential between modelled and observed SFS
        self.L2_residual: Optional[float] = None

        #: Random number generator seed
        self.seed: int | None = int(seed) if seed is not None else None

        #: Random generator instance
        self.rng: np.random.Generator = np.random.default_rng(seed=seed)

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
            self._logger.debug('Inference needs to be run first, triggering run.')

            return self.run(*args, **kwargs)

        self._logger.debug('Inference already run, not running again.')

    @staticmethod
    def _run_if_required_wrapper(func):
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
        :param kwargs: Additional keyword arguments which are ignored.
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
        self._logger.debug(f'Starting numerical optimization of {self.n_runs} '
                           'independently initialized samples which are run ' +
                           ('in parallel.' if self.parallelize else 'sequentially.'))

        # precompute discretization
        self.discretization.precompute()

        # perform numerical minimization
        result, params_mle = self.optimization.run(
            x0=self.x0,
            scales=self.scales,
            bounds=self.bounds,
            get_counts=self.get_counts(),
            n_runs=self.n_runs,
            pbar=pbar,
            desc=f'{self.__class__.__name__}>Performing inference'
        )

        # assign runs dataframe and likelihoods
        self.runs = self.optimization.runs
        self.likelihoods = self.runs.likelihood.to_list()

        # unpack shared parameters
        params_mle = unpack_shared(params_mle)

        # normalize parameters
        params_mle['all'] = self.model._normalize(params_mle['all'])

        # assign optimization result and MLE parameters
        self._assign_result(result, params_mle['all'])

        # report on optimization result
        self._report_result(result, params_mle)

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
        return dict(all=lambda params: self._model_sfs(
            discretization=self.discretization,
            model=self.model,
            params=params,
            sfs_neut=self.sfs_neut,
            sfs_sel=self.sfs_sel,
            folded=self.folded
        ))

    def evaluate_likelihood(self, params: Dict[str, Dict[str, float]]) -> float:
        """
        Evaluate likelihood function at given parameters.

        :param params: Parameters indexed by type and parameter name.
        :return: The likelihood.
        """
        x0_cached = self.optimization.x0
        self.optimization.x0 = params

        # prepare parameters
        params = pack_params(self.optimization.scale_values(flatten_dict(params)))

        lk = -self.optimization.get_loss_function(self.get_counts())(params)

        self.optimization.x0 = x0_cached

        return lk

    def _assign_result(self, result: 'scipy.optimize.OptimizeResult', params_mle: dict):
        """
        Assign optimization result and MLE parameters.

        :param params_mle: MLE parameters.
        :param result: Optimization result.
        """
        self.result = result
        self.params_mle = params_mle
        self.likelihood = -result.fun

        # get SFS for MLE parameters
        counts_mle, _ = self._model_sfs(
            params=params_mle,
            discretization=self.discretization,
            model=self.model,
            sfs_neut=self.sfs_neut,
            sfs_sel=self.sfs_sel,
            folded=self.folded
        )

        # create spectrum object from polymorphic counts
        self.sfs_mle = Spectrum.from_polymorphic(counts_mle)

        # L2 norm of fit minus observed SFS
        self.L2_residual = self.get_residual(2)

        # check L2 residual
        self._check_L2_residual()

    def get_residual(self, k: int) -> float:
        """
        Residual of the modelled and observed SFS.

        :param k: Order of the norm
        :return: L2 residual
        """
        return norm(self.sfs_mle.polymorphic - self.sfs_sel.polymorphic, k)

    def _check_L2_residual(self):
        """
        Check L2 residual.
        """
        ratio = self.get_residual(1) / self.sfs_sel.n_polymorphic

        if ratio > 0.1:
            self._logger.warning("The L1 residual comparing the modelled and observed SFS is rather large: "
                                 f"`norm(sfs_modelled - sfs_observed, 1) / sfs_observed` = {ratio:.3f}. "
                                 "This may indicate that the model does not fit the data well.")

    def _report_result(self, result: 'scipy.optimize.OptimizeResult', params: dict):
        """
        Inform on optimization result.

        :param params: MLE parameters.
        :param result: Optimization result.
        """
        # report on optimization result
        if result.success:
            self._logger.debug(f"Successfully finished optimization after {result.nit} iterations "
                              f"and {result.nfev} function evaluations, obtaining a log-likelihood "
                              f"of -{result.fun:.4g}")
        else:
            self._logger.warning("Numerical optimization did not terminate normally, so "
                                 "the result might be unreliable. Consider adjusting "
                                 "the optimization parameters (increasing `gtol` or `n_runs`) "
                                 "or decreasing the number of optimized parameters.")

        self._logger.info(
            "Inference results: %s (best_run ± std_across_runs)",
            self._format_mean_std(
                self.runs[self.runs.likelihood == self.runs.likelihood.max()].select_dtypes("number").iloc[0],
                self.runs.select_dtypes("number").std().fillna(0)
            )
        )

    @staticmethod
    def _format_mean_std(mean: dict, std: dict) -> str:
        """
        Format numerical dictionary for printing as mean ± std.

        :param mean: Dictionary of means.
        :param std: Dictionary of standard deviations.
        :return: Formatted string.
        """
        mean = flatten_dict(mean)
        std = flatten_dict(std)

        return str({
            k: f"{mean[k]:.4g} ± {std.get(k, 0):.2g}"
            for k in mean
        }).replace("'", "")

    def update_properties(self, **kwargs) -> Self:
        """
        Update the properties of this class with the given dictionary
        given that its entries are not ``None``.

        .. warning::
            This will fail to update the optimization instance among other things. Avoid using this method.

        :meta private:
        :param kwargs: Dictionary of properties to update.
        :return: Self.
        """
        for key, value in kwargs.items():
            if value is not None:
                setattr(self, key, value)

        return self

    def bootstrap(
            self,
            n_samples: int = None,
            parallelize: bool = None,
            update_likelihood: bool = True,
            n_retries: int = None,
            pbar: bool = True,
            pbar_title: str = 'Bootstrapping'
    ) -> pd.DataFrame:
        """
        Perform the parametric bootstrap.

        :param n_samples: Number of bootstrap samples. Defaults to :attr:`n_bootstraps`.
        :param parallelize: Whether to parallelize the bootstrap. Defaults to :attr:`parallelize`.
        :param update_likelihood: Whether to update the likelihood to be the mean of the bootstrap samples.
        :param n_retries: Number of optimization runs for each bootstrap sample. Defaults to
            :attr:`n_bootstrap_retries`.
        :param pbar: Whether to show a progress bar.
        :param pbar_title: Title for the progress bar.
        :return: Dataframe with bootstrap results.
        """
        self._bootstrap(
            n_samples=n_samples,
            parallelize=parallelize,
            update_likelihood=update_likelihood,
            n_retries=n_retries,
            pbar=pbar,
            pbar_title=pbar_title
        )

        # remove 'all.' prefix from columns
        self.bootstraps.columns = self.bootstraps.columns.str.removeprefix("all.")

        # add estimates for alpha to the bootstraps
        self._add_alpha_to_bootstraps()

        return self.bootstraps

    def _bootstrap(
            self,
            n_samples: int = None,
            parallelize: bool = None,
            update_likelihood: bool = True,
            n_retries: int = None,
            pbar: bool = True,
            pbar_title: str = 'Bootstrapping',
    ) -> pd.DataFrame:
        """
        Perform the parametric bootstrap.

        :param n_samples: Number of bootstrap samples. Defaults to :attr:`n_bootstraps`.
        :param parallelize: Whether to parallelize the bootstrap. Defaults to :attr:`parallelize`.
        :param update_likelihood: Whether to update the likelihood to be the mean of the bootstrap samples.
        :param n_retries: Number of optimization runs for each bootstrap sample. Defaults to
            :attr:`n_bootstrap_retries`.
        :param pbar: Whether to show a progress bar.
        :param pbar_title: Title for the progress bar.
        :return: Dataframe with bootstrap results.
        """
        # check if locked
        self.raise_if_locked()

        # perform inference first if not done yet
        self.run_if_required()

        # precompute discretization if not done yet
        self.discretization.precompute()

        # update properties
        self.update_properties(
            n_bootstraps=n_samples,
            parallelize=parallelize,
            n_bootstrap_retries=n_retries,
        )

        n_bootstraps = int(self.n_bootstraps)

        start_time = time.time()

        # parallelize computations if desired
        if self.parallelize:

            self._logger.debug(f"Running {n_bootstraps} bootstrap samples "
                               f"in parallel on {min(mp.cpu_count(), n_bootstraps)} cores.")

        else:
            self._logger.debug(f"Running {n_bootstraps} bootstrap samples sequentially.")

        # seeds for bootstraps
        seeds = self.rng.integers(0, high=2 ** 32, size=n_bootstraps)

        # run bootstraps
        results = optimization.parallelize(
            func=self._get_run_bootstrap_sample(),
            data=seeds,
            parallelize=self.parallelize,
            pbar=pbar,
            desc=f'{self.__class__.__name__}>{pbar_title} ({self.n_bootstrap_retries} runs each)'
        )

        # select best result for each bootstrap sample
        i_best = [np.argmax([-r[0].fun if r[0].success else -np.inf for r in res]) for res in results]

        # get best results
        result = results[np.arange(n_bootstraps), i_best]

        # number of successful runs
        n_success = np.sum([res.success for res in result[:, 0]])

        # dataframe of MLE estimates
        self.bootstraps = pd.DataFrame([flatten_dict(r) for r in result[:, 1]])

        # assign additional info to bootstraps
        self.bootstraps['likelihood'] = [-r.fun for r in result[:, 0]]
        self.bootstraps['success'] = [r.success for r in result[:, 0]]
        self.bootstraps['result'] = [str(r) for r in result[:, 0]]
        self.bootstraps['x0'] = [flatten_dict(r) for r in result[:, 2]]
        self.bootstraps['i_best_run'] = i_best
        self.bootstraps['likelihoods_runs'] = [[-r[0].fun for r in res] for res in results]
        self.bootstraps['likelihoods_std'] = self.bootstraps['likelihoods_runs'].apply(np.std)
        self.bootstraps['results_runs'] = [[str(r[0]) for r in res] for res in results]
        self.bootstraps['x0_runs'] = [[flatten_dict(r[2]) for r in res] for res in results]

        # assign bootstrap results
        self.bootstrap_results = list(result[:, 0])

        self._logger.info(
            "Bootstrap summary: %s (mean ± std)",
            self._format_mean_std(
                self.bootstraps.select_dtypes("number").mean().fillna(0),
                self.bootstraps.select_dtypes("number").std().fillna(0)
            )
        )

        # issue warning if some runs did not finish successfully
        if n_success < n_bootstraps:
            self._logger.warning(
                f"{n_bootstraps - n_success} out of {n_bootstraps} bootstrap samples "
                "did not terminate normally during numerical optimization. "
                "The confidence intervals might thus be unreliable. Consider "
                "increasing the number of optimization runs per bootstrap sample (`n_retries`), "
                "adjusting the optimization parameters (increasing `gtol` or `n_runs`), "
                "or decreasing the number of optimized parameters."
            )

        # add execution time
        self.execution_time += time.time() - start_time

        # assign average likelihood of successful runs
        if update_likelihood:
            self.likelihood = np.mean([-res.fun for res in result[:, 0] if res.success] + [self.likelihood])

        return self.bootstraps

    def _add_alpha_to_bootstraps(self):
        """
        Add estimates for alpha to the bootstraps.
        """
        self._logger.debug('Computing estimates for alpha.')

        # add alpha estimates
        self.bootstraps['alpha'] = self.bootstraps.apply(lambda r: self.get_alpha(dict(r)), axis=1)

    def _get_run_bootstrap_sample(self) -> Callable[[int], Tuple['scipy.optimize.OptimizeResult', dict, dict, int]]:
        """
        Get function which runs a single bootstrap sample.

        :return: Static function which runs a single bootstrap sample, taking an optional seed and returning the
            optimization result of best run, MLE parameters, the best x0, and its index.
        """

        optimization = self.optimization
        discretization = self.discretization
        model = self.model
        folded = self.folded
        scales = self.scales
        bounds = self.bounds
        n_retries = self.n_bootstrap_retries
        sfs_sel = self.sfs_sel
        sfs_neut = self.sfs_neut
        params_mle_all = dict(all=self.params_mle)

        def run_bootstrap_sample(seed: int):
            """
            Resample the selected and neutral SFS and rerun optimization.
            Initial point is the MLE; each retry perturbs x0.
            """
            results = []
            x0 = params_mle_all
            rng = np.random.default_rng(seed)

            # resample once
            sel = sfs_sel.resample(seed=rng)
            neut = sfs_neut.resample(seed=rng)

            for _ in range(max(n_retries, 1)):
                # run optimizer
                result, params_mle = optimization.run(
                    x0=x0,
                    scales=scales,
                    bounds=bounds,
                    n_runs=1,
                    debug_iterations=False,
                    print_info=False,
                    get_counts=dict(all=lambda p: BaseInference._model_sfs(
                        discretization=discretization,
                        model=model,
                        params=p,
                        sfs_neut=neut,
                        sfs_sel=sel,
                        folded=folded
                    )),
                    desc=f"{self.__class__.__name__}>Bootstrapping"
                )

                # unpack shared + normalize
                params_mle = unpack_shared(params_mle)

                # normalize MLE estimates
                params_mle['all'] = model._normalize(params_mle['all'])

                results += [(result, params_mle, x0)]

                x0 = optimization.sample_x0(params_mle_all, random_state=rng)

            return results

        return run_bootstrap_sample

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

    @classmethod
    def _model_sfs(
            cls,
            discretization: Discretization,
            model: Parametrization,
            params: dict,
            sfs_neut: Spectrum,
            sfs_sel: Spectrum,
            folded: bool
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Model the selected SFS from the given parameters.

        :param discretization: Discretization instance.
        :param model: Parametrization instance.
        :param params: Dictionary of parameters.
        :param sfs_sel: Observed spectrum of selected sites.
        :param sfs_neut: Observed spectrum of neutral sites.
        :param folded: Whether the SFS are folded.
        :return: Array of modelled and observed counts
        """
        # obtain modelled selected SFS
        counts_modelled = discretization.model_selection_sfs(model, params)

        # adjust for mutation rate and mutational target size
        counts_modelled *= sfs_neut.theta * sfs_sel.n_sites

        # add contribution of demography and adjust polarization error
        counts_modelled = cls._add_demography(sfs_neut, counts_modelled, eps=params['eps'])

        # fold if necessary
        if folded:
            counts_observed = sfs_sel.fold().polymorphic
            counts_modelled = Spectrum.from_polymorphic(counts_modelled).fold().polymorphic
        else:
            counts_observed = sfs_sel.polymorphic

        return counts_modelled, counts_observed

    @staticmethod
    def _adjust_polarization(counts: np.ndarray, eps: float) -> np.ndarray:
        """
        Adjust the polarization of the given SFS where
        eps is the rate of ancestral misidentification.

        :param counts: Polymorphic SFS counts to adjust.
        :param eps: Rate of ancestral misidentification.
        :return: Adjusted SFS counts.
        """
        return (1 - eps) * counts + eps * counts[::-1]

    @classmethod
    def _add_demography(
            cls,
            sfs_neut: Spectrum,
            counts_sel: np.ndarray,
            eps: float
    ) -> np.ndarray:
        """
        Add the effect of demography to counts_sel by considering
        how counts_neut is perturbed relative to the standard coalescent.
        The polarization error is also included here.

        :param sfs_neut: Observed spectrum of neutral sites.
        :param counts_sel: Modelled counts of selected sites.
        :param eps: Rate of ancestral misidentification.
        :return: Adjusted counts of selected sites.
        """
        # normalized counts of the standard coalescent
        neutral_modelled = standard_kingman(sfs_neut.n).polymorphic * sfs_neut.theta * sfs_neut.n_sites

        # Apply polarization error to neutral and selected counts.
        # Note that we fail to adjust for polarization if we were to
        # apply the error to both the selected and neutral modelled counts.
        # In addition, note that eps is not easily interpretable,
        # as we apply it to both neutral and selected counts, which
        # is of course necessary. Testing this on sample datasets
        # yields very reasonable results, compatible with polyDFE.
        neutral_observed = cls._adjust_polarization(sfs_neut.polymorphic, eps)
        selected_modelled = cls._adjust_polarization(counts_sel, eps)

        # These counts transform the standard Kingman case to the observed
        # neutral SFS when multiplied and thus account for demography as we assume
        # the distortion in sfs_neut to be due to demography alone.
        # Note that the nuisance parameters naturally are sensitive to a large
        # sampling variance in the neutral SFS.
        # However, DFE inference on `wiggly` spectra is not advisable anyway.
        r = neutral_observed / neutral_modelled

        # adjust for demography and polarization error
        return r * selected_modelled

    @_run_if_required_wrapper
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
            kwargs_legend: dict = dict(prop=dict(size=8)),
            ax: 'plt.Axes' = None

    ) -> 'plt.Axes':
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
        :param kwargs_legend: Keyword arguments passed to :meth:`plt.legend`. Only for Python visualization backend.
        :param ax: Axes to plot on. Only for Python visualization backend.
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
            kwargs_legend=kwargs_legend,
            ax=ax
        )

    @_run_if_required_wrapper
    def plot_bucket_sizes(
            self,
            file: str = None,
            show: bool = True,
            title: str = 'bucket sizes',
            ax: 'plt.Axes' = None

    ) -> 'plt.Axes':
        """
        Plot mass in each bucket for the MLE DFE.
        This can be used to check if the interval bounds and spacing
        are chosen appropriately.

        :param file: File to save plot to.
        :param show: Whether to show plot.
        :param title: Plot title.
        :param ax: Axes to plot on. Only for Python visualization backend.
        :return: Axes object
        """
        from .visualization import Visualization

        # evaluate at fixed parameters
        sizes = self.model._discretize(self.params_mle, self.discretization.bins)

        return Visualization.plot_buckets_sizes(
            n_intervals=self.discretization.n_intervals,
            bins=self.discretization.bins,
            sizes=sizes,
            title=title,
            file=file,
            show=show,
            ax=ax
        )

    @_run_if_required_wrapper
    def plot_interval_density(
            self,
            file: str = None,
            show: bool = True,
            intervals: Sequence = np.array([-np.inf, -100, -10, -1, 0, 1, np.inf]),
            interval_labels: List[str] = None,
            color: str = 'C0',
            ax: 'plt.Axes' = None
    ) -> 'plt.Axes':
        """
        Plot density of the discretization intervals chosen. Note that although this plot looks similar, this is
        not the DFE!

        :param color: Color of the bars.
        :param file: File to save plot to.
        :param show: Whether to show plot.
        :param interval_labels: Labels for the intervals.
        :param intervals: Array of interval boundaries yielding ``intervals.shape[0] - 1`` bins.
        :param ax: Axes to plot on. Only for Python visualization backend.
        :return: Axes object
        """
        from .visualization import Visualization

        # issue warning
        if not self.discretization.linearized:
            self._logger.warning('Note that the interval density is not important if the DFE was not linearized.')

        return Visualization.plot_interval_density(
            density=self.discretization.get_interval_density(intervals),
            file=file,
            show=show,
            intervals=intervals,
            interval_labels=interval_labels,
            color=color,
            ax=ax
        )

    def plot_sfs_comparison(
            self,
            sfs_types: List[Literal['modelled', 'observed', 'selected', 'neutral']] = ['observed', 'modelled'],
            labels: List[str] = None,
            colors: List[str] = None,
            file: str = None,
            show: bool = True,
            ax: 'plt.Axes' = None,
            title: str = 'SFS comparison',
            use_subplots: bool = False,
            show_monomorphic: bool = False,
            kwargs_legend: dict = dict(prop=dict(size=8)),
    ) -> 'plt.Axes':
        """
        Plot SFS comparison.

        :param file: File to save plot to.
        :param labels: Labels for the SFS.
        :param colors: Colors for the SFS. Only for Python visualization backend.
        :param sfs_types: Types of SFS to plot.
        :param show: Whether to show plot.
        :param ax: Axes to plot on. Only for Python visualization backend and if ``use_subplots`` is ``False``.
        :param title: Plot title
        :param use_subplots: Whether to use subplots. Only for Python visualization backend.
        :param show_monomorphic: Whether to show monomorphic counts
        :param kwargs_legend: Keyword arguments passed to :meth:`plt.legend`. Only for Python visualization backend.
        :return: Axes object
        """
        from .visualization import Visualization

        if 'modelled' in sfs_types:
            self.run_if_required()

        mapping = dict(
            observed=self.sfs_sel,
            selected=self.sfs_sel,
            modelled=self.sfs_mle,
            neutral=self.sfs_neut
        )

        return Visualization.plot_spectra(
            spectra=[list(mapping[t]) for t in sfs_types],
            labels=sfs_types if labels is None else labels,
            colors=colors,
            file=file,
            show=show,
            ax=ax,
            title=title,
            use_subplots=use_subplots,
            show_monomorphic=show_monomorphic,
            kwargs_legend=kwargs_legend
        )

    def plot_observed_sfs(
            self,
            labels: List[str] = None,
            file: str = None,
            show: bool = True,
            ax: 'plt.Axes' = None
    ) -> 'plt.Axes':
        """
        Plot neutral and selected SFS.

        :param file: File to save plot to.
        :param labels: Labels for the SFS.
        :param show: Whether to show plot.
        :param ax: Axes to plot on. Only for Python visualization backend.
        :return: Axes object
        """
        return self.plot_sfs_comparison(
            sfs_types=['neutral', 'selected'],
            labels=labels,
            file=file,
            show=show,
            ax=ax
        )

    @_run_if_required_wrapper
    def plot_all(self, show: bool = True):
        """
        Plot a bunch of plots.

        :param show: Whether to show plot.
        """
        self.plot_inferred_parameters(show=show)
        self.plot_discretized(show=show)
        self.plot_sfs_comparison(show=show)
        self.plot_continuous(show=show)
        self.plot_interval_density(show=show)
        self.plot_likelihoods(show=show)

    def get_stats_discretized(
            self,
            ci_level: float = 0.05,
            intervals: np.ndarray = np.array([-np.inf, -100, -10, -1, 0, 1, np.inf]),
            bootstrap_type: Literal['percentile', 'bca'] = 'percentile',
            point_estimate: Literal['original', 'mean', 'median'] = 'mean'
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Get statistics for the discretized DFE.

        :param ci_level: Confidence interval level.
        :param intervals: Array of interval boundaries yielding ``intervals.shape[0] - 1`` bins.
        :param bootstrap_type: Type of bootstrap to use.
        :param point_estimate: Whether to use 'original' MLE values, 'mean' or 'median' of bootstraps as point estimate.
        :return: Center values, errors around center, and confidence intervals.
        """
        return Inference.get_stats_discretized(
            params=self.params_mle,
            bootstraps=self.bootstraps,
            model=self.model,
            ci_level=ci_level,
            intervals=intervals,
            bootstrap_type=bootstrap_type,
            point_estimate=point_estimate
        )

    @_run_if_required_wrapper
    def plot_inferred_parameters(
            self,
            confidence_intervals: bool = True,
            ci_level: float = 0.05,
            bootstrap_type: Literal['percentile', 'bca'] = 'percentile',
            file: str = None,
            show: bool = True,
            title: str = 'parameter estimates',
            scale: Literal['lin', 'log', 'symlog'] = 'log',
            ax: 'plt.Axes' = None,
            kwargs_legend: dict = dict(prop=dict(size=8), loc='upper right'),
            **kwargs
    ) -> 'plt.Axes':
        """
        Visualize the inferred parameters and their confidence intervals.

        :param scale: y-scale of the plot.
        :param title: Plot title.
        :param confidence_intervals: Whether to show confidence intervals.
        :param bootstrap_type: Type of bootstrap to use.
        :param ci_level: Confidence level for the confidence intervals.
        :param file: File to save plot to.
        :param show: Whether to show plot.
        :param ax: Axes to plot on. Only for Python visualization backend.
        :param kwargs_legend: Keyword arguments passed to :meth:`plt.legend`. Only for Python visualization backend.
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
            ax=ax,
            kwargs_legend=kwargs_legend,
        )

    @_run_if_required_wrapper
    def plot_inferred_parameters_boxplot(
            self,
            file: str = None,
            show: bool = True,
            title: str = 'parameter estimates',
            **kwargs
    ) -> 'plt.Axes':
        """
        Visualize the inferred parameters and their confidence intervals.

        :param title: Plot title.
        :param file: File to save plot to.
        :param show: Whether to show plot.
        :return: Axes object
        :raises ValueError: If no inference objects are given or no bootstraps are found.
        """
        return Inference.plot_inferred_parameters_boxplot(
            inferences=[self],
            labels=['all'],
            file=file,
            show=show,
            title=title
        )

    @_run_if_required_wrapper
    def plot_likelihoods(
            self,
            file: str = None,
            show: bool = True,
            title: str = 'likelihoods',
            scale: Literal['lin', 'log'] = 'lin',
            ax: 'plt.Axes' = None,
            ylabel: str = 'lnl',
            **kwargs
    ) -> 'plt.Axes':
        """
        Visualize the likelihoods of the optimization runs using a scatter plot.

        :param scale: y-scale of the plot.
        :param title: Plot title.
        :param file: File to save plot to.
        :param show: Whether to show plot.
        :param ax: Axes to plot on. Only for Python visualization backend.
        :param ylabel: Label for the y-axis.
        :return: Axes object
        """
        from .visualization import Visualization

        return Visualization.plot_scatter(
            values=self.likelihoods,
            file=file,
            show=show,
            title=title,
            scale=scale,
            ax=ax,
            ylabel=ylabel,
        )

    def _lrt_components(
            self,
            ll_simple: float,
            ll_complex: float,
            df: int,
            m: int = 0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform a likelihood ratio test (LRT) with correction for boundary constraints as described in
        Molenberghs et Verbeke (2007) (https://doi.org/10.2307/1403361).

        :param ll_simple: Log-likelihood of the simple model.
        :param ll_complex: Log-likelihood of the complex model.
        :param df: Total number of parameters being tested.
        :param m: Number of parameters at the boundary.
        :return: Weights and components (p-values), and degrees of freedom.
        :raises ValueError: If the number of parameters at the boundary exceeds the total number of parameters.
        """
        if m > df:
            raise ValueError('Number of parameters at the boundary exceeds total number of parameters.')

        lr = -2 * (ll_simple - ll_complex)

        # Issue info message
        self._logger.info(f"Simple model likelihood: {ll_simple}, "
                          f"Complex model likelihood: {ll_complex}, "
                          f"Total degrees of freedom: {df}, "
                          f"Parameters at boundary: {m}.")

        # compute mixture weights
        weights = np.array([comb(m, j - (df - m)) / (2 ** m) for j in range(df - m, df + 1)])

        # zero degrees of freedom is point mass at 0
        components = np.array([(0 if j == 0 else chi2.sf(lr, j)) for j in range(df - m, df + 1)])
        df = np.array([j for j in range(df - m, df + 1)])

        self._logger.debug(
            " + ".join([f"{weight} * {component:.3f}" for weight, component in zip(weights, components)])
        )

        return weights, components, df

    def lrt(self, ll_simple: float, ll_complex: float, df: int, m: int = 0) -> float:
        """
        Perform a likelihood ratio test (LRT) with correction for boundary constraints as described in
        Molenberghs et Verbeke (2007) (https://doi.org/10.2307/1403361).

        :param ll_simple: Log-likelihood of the simple model.
        :param ll_complex: Log-likelihood of the complex model.
        :param df: Total number of parameters being tested.
        :param m: Number of parameters at the boundary.
        :return: p-value
        """
        weights, components, _ = self._lrt_components(ll_simple, ll_complex, df, m)

        return np.sum(weights * components)

    def _is_nested(self, complex: 'BaseInference') -> bool:
        """
        Check if the given model is nested within the current model.
        The given model's fixed parameters need to be a proper subset of this model's fixed parameters.

        :param complex: More complex model.
        :return: Whether the models are nested.
        """
        if type(self.model) != type(complex.model):
            return False

        # optimization holds the flattened dictionary
        fixed_complex = set(complex.optimization.fixed_params.keys())
        fixed_simple = set(self.optimization.fixed_params.keys())

        return fixed_complex < fixed_simple

    def _get_df_nested(self, complex: 'BaseInference') -> Tuple[int, int]:
        """
        Get the degrees of freedom for the likelihood ratio test.

        :param complex: More complex model.
        :return: Degrees of freedom and number of parameters at bounds.
        :raises ValueError: If the models are not nested.
        """
        if type(self.model) != type(complex.model):
            raise ValueError(f'DFE parametrizations are not the same: {type(self.model)} vs. {type(complex.model)}.')

        if not self._is_nested(complex):
            raise ValueError('Models are not nested.')

        # optimization holds the flattened dictionary
        fixed_complex = set(complex.optimization.fixed_params.keys())
        fixed_simple = set(self.optimization.fixed_params.keys())

        # fixed parameters
        fixed = fixed_simple - fixed_complex
        fixed_params = self.optimization.fixed_params

        # parameter bounds
        bounds = self.optimization.bounds

        # number of parameters at bounds
        m = sum([fixed_params[p] == bounds[p.split('.')[1]][0] or
                 fixed_params[p] == bounds[p.split('.')[1]][1] for p in fixed])

        # degrees of freedom
        df = len(fixed)

        return df, m

    def compare_nested(self, complex: 'BaseInference') -> float | None:
        """
        Perform likelihood ratio test with given more complex model.
        The given model's fixed parameters need to be a proper
        subset of this model's fixed parameters.
        Low p-values indicate that the more complex model provides a significantly better fit.

        :param complex: More complex model.
        :return: p-value or None if the models are not nested.
        :raises ValueError: If the inference objects are not nested.
        """
        df, m = self._get_df_nested(complex)

        # run inference if not done yet
        self.run_if_required()
        complex.run_if_required()

        return self.lrt(self.likelihood, complex.likelihood, df, m)

    @_run_if_required_wrapper
    @functools.lru_cache
    def compare_nested_models(self, do_bootstrap: bool = True) -> Tuple[np.ndarray, Dict[str, 'BaseInference']]:
        """
        Compare the various nested versions of the specified
        model using likelihood ratio tests. We check for inclusion of ancestral misidentification
        and beneficial mutations.

        :param do_bootstrap: Whether to perform bootstrapping. This is recommended to get more accurate p-values.
        :return: Matrix of p-values, dict of base inference objects
        """

        # get sub-model specifications
        submodels_dfe = _from_string(self.model).submodels
        submodels_outer = dict(
            no_anc=dict(eps=0),
            anc={}
        )
        if 'all' in self.fixed_params and 'h' in self.fixed_params['all']:
            submodels_h = dict(h=self.fixed_params['all']['h'])
        else:
            submodels_h = {}

        # take outer product to get fixed parameters for each model
        inferences: Dict[str, BaseInference] = {}
        for p in itertools.product(submodels_dfe.keys(), submodels_outer.keys()):
            # create deep copy of object
            inference = copy.deepcopy(self)

            # carry over bootstrap setting
            inference.do_bootstrap = do_bootstrap

            # dict of params to be fixed
            params = dict(all=submodels_dfe[p[0]] | submodels_outer[p[1]] | submodels_h)

            # assign fixed parameters
            inference._set_fixed_params(params)

            # inform about fixed parameters
            self._logger.info(f'Holding parameters fixed to {params}.')

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
        P = []
        for i in inputs:
            try:
                P.append(i[0][1].compare_nested(i[1][1]))
            except ValueError:
                P.append(None)

        P = np.reshape(P, (n, n))

        return P, inferences

    def plot_nested_models(
            self,
            file: str = None,
            show: bool = True,
            remove_empty: bool = False,
            transpose: bool = False,
            cmap: str = None,
            title: str = 'nested model comparison',
            ax: 'plt.Axes' = None,
            do_bootstrap: bool = True,

    ) -> 'plt.Axes':
        """
        Plot the p-values of nested likelihoods.

        :param file: File to save plot to.
        :param show: Whether to show plot.
        :param remove_empty: Whether to remove empty rows and columns.
        :param transpose: Whether to transpose the matrix.
        :param cmap: Colormap to use.
        :param title: Plot title.
        :param ax: Axes to plot on. Only for Python visualization backend.
        :param do_bootstrap: Whether to perform bootstrapping. This is recommended to get more accurate p-values.
        :return: Axes object
        """
        from .visualization import Visualization

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

        return Visualization.plot_nested_models(
            P=P,
            labels_x=labels_x,
            labels_y=labels_y,
            file=file,
            show=show,
            cmap=cmap,
            title=title,
            ax=ax
        )

    @functools.wraps(AbstractInference.plot_discretized)
    @_run_if_required_wrapper
    def plot_discretized(self, *args, **kwargs):
        return super().plot_discretized(*args, **kwargs)

    def get_dfe(self) -> DFE:
        """
        Return a frozen DFE object containing the inferred parameters and (if available) the bootstrap samples.

        :return: DFE object
        """
        # ensure inference is run
        self.run_if_required()

        return DFE(
            model=self.model,
            params=self.params_mle,
            bootstraps=self.bootstraps
        )

    def get_spectra(self) -> Spectra:
        """
        Return a Spectra object containing the neutral, selected and modelled SFS.

        :return: Spectra object with types ``neutral``, ``selected``, and ``modelled``.
        """
        return Spectra.from_spectra(dict(
            neutral=self.sfs_neut,
            selected=self.sfs_sel,
            modelled=self.sfs_mle
        ))

    def get_alpha(self, params: dict = None) -> float:
        """
        Get alpha, the proportion of beneficial non-synonymous substitutions.

        :param params: DFE Parameters to use for calculation.
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
            method_mle=self.optimization.method_mle,
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
    def from_config_file(cls, file: str, cache: bool = True) -> Self:
        """
        Load from config file.

        :param file: Config file path, possibly URL.
        :param cache: Whether to use the cache if available.
        :return: Inference object.
        """
        from . import Config

        return cls.from_config(Config.from_file(file, cache=cache))

    def get_summary(self) -> 'InferenceResult':
        """
        Get summary.

        :return: Inference results.
        """
        return InferenceResult(self)

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
        return self.model.param_names + ['eps', 'h']

    def _set_fixed_params(self, params: Dict[str, Dict[str, float]]):
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


class InferenceResult(Serializable):
    """
    Inference result.
    """

    def __init__(self, inference: BaseInference):
        """
        Initialize object.

        :param inference: Inference object.
        """
        #: Configuration of the inference
        self.config: 'Config' = inference.create_config()

        #: Inferred DFE
        self.dfe: DFE = inference.get_dfe()

        #: Spectra
        self.spectra: Spectra = inference.get_spectra()

        #: Bootstrap samples
        self.bootstraps: pd.DataFrame = inference.bootstraps

        #: Optimization runs
        self.runs: pd.DataFrame = inference.runs
