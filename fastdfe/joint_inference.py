"""
Joint inference module.
"""

__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"
__date__ = "2023-02-26"

import copy
import functools
import logging
import time
from typing import List, Dict, Tuple, Literal, Optional, cast, Callable

import jsonpickle
import multiprocess as mp
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numpy.linalg import norm
from scipy.optimize import OptimizeResult
from tqdm import tqdm

from .abstract_inference import Inference
from .base_inference import BaseInference
from .bootstrap import Bootstrap
from .config import Config
from .optimization import Optimization, SharedParams, pack_shared, expand_shared, \
    Covariate, flatten_dict, merge_dicts, correct_values, parallelize as parallelize_func, expand_fixed, \
    collapse_fixed, unpack_shared
from .parametrization import Parametrization
from .settings import Settings
from .spectrum import Spectrum, Spectra
from .visualization import Visualization

# get logger
logger = logging.getLogger('fastdfe')


class JointInference(BaseInference):
    """
    Enabling the sharing of parameters among several Inference objects.

    Example usage:

    ::

        import fastdfe as fd

        # neutral SFS for two types
        sfs_neut = fd.Spectra(dict(
            pendula=[177130, 997, 441, 228, 156, 117, 114, 83, 105, 109, 652],
            pubescens=[172528, 3612, 1359, 790, 584, 427, 325, 234, 166, 76, 31]
        ))

        # selected SFS for two types
        sfs_sel = fd.Spectra(dict(
            pendula=[797939, 1329, 499, 265, 162, 104, 117, 90, 94, 119, 794],
            pubescens=[791106, 5326, 1741, 1005, 756, 546, 416, 294, 177, 104, 41]
        ))

        # create inference object
        inf = fd.JointInference(
            sfs_neut=sfs_neut,
            sfs_sel=sfs_sel,
            fixed_params=dict(eps=0), # fix eps to 0
            do_bootstrap=True
        )

        # run inference
        inf.run()

    """

    def __init__(
            self,
            sfs_neut: Spectra,
            sfs_sel: Spectra,
            include_divergence: bool = None,
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
            shared_params: List[SharedParams] = [],
            covariates: List[Covariate] = [],
            do_bootstrap: bool = False,
            n_bootstraps: int = 100,
            n_bootstrap_retries: int = 2,
            parallelize: bool = True,
            folded: bool = None,
            **kwargs
    ):
        """
        Create instance.

        :param sfs_neut: Neutral SFS. Note that we require monomorphic counts to be specified in order to infer
            the mutation rate.
        :param sfs_sel: Selected SFS. Note that we require monomorphic counts to be specified in order to infer
            the mutation rate.
        :param include_divergence: Whether to include divergence in the likelihood
        :param intervals_del: ``(start, stop, n_interval)`` for deleterious population-scaled
            selection coefficients. The intervals will be log10-spaced.
        :param intervals_ben: Same as intervals_del but for beneficial selection coefficients
        :param integration_mode: Integration mode, ``quad`` not recommended
        :param linearized: Whether to use the linearized model, ``False`` not recommended
        :param model: DFE parametrization
        :param seed: Random seed. Use ``None`` for no seed.
        :param x0: Dictionary of initial values in the form ``{type: {param: value}}``
        :param bounds: Bounds for the optimization in the form {param: (lower, upper)}
        :param scales: Scales for the optimization in the form {param: scale}
        :param loss_type: Loss type
        :param opts_mle: Options for the optimization
        :param n_runs: Number of independent optimization runs out of which the best one is chosen. The first run
            will use the initial values if specified. Consider increasing this number if the optimization does not
            produce good results.
        :param fixed_params: Dictionary of fixed parameters in the form ``{type: {param: value}}``
        :param shared_params: List of shared parameters
        :param do_bootstrap: Whether to perform bootstrapping
        :param n_bootstraps: Number of bootstraps
        :param n_bootstrap_retries: Number of retries for bootstraps that did not terminate normally.
        :param parallelize: Whether to parallelize the optimization
        :param folded: Whether the SFS are folded. If not specified, the SFS will be folded if all of the given
            SFS appear to be folded.
        :param kwargs: Additional keyword arguments which are ignored.
        """
        if sfs_neut.has_dots() or sfs_sel.has_dots():
            raise ValueError('Type names cannot contain dots are not allowed as they are used internally.'
                             'You can use the `replace_dots` method to replace dots with another character.')

        # check whether types are equal
        if set(sfs_neut.types) != set(sfs_sel.types):
            raise ValueError('The neutral and selected spectra must have exactly the same types.')

        #: SFS types
        self.types: List[str] = sfs_neut.types

        # initialize parent
        # the `self.folded` property will generalize nicely as we
        # evaluate ``sfs_sel.all.is_folded and sfs_neut.all.is_folded``
        BaseInference.__init__(**locals())

        #: original MLE parameters before adding covariates and unpacked shared
        self.params_mle_raw: Optional[Dict[str, Dict[str, float]]] = None

        #: Shared parameters with expanded 'all' type
        self.shared_params = expand_shared(shared_params, self.types, self.optimization.param_names)

        # add covariates as shared parameters
        self._add_covariates_as_shared(covariates)

        # check if the shared parameters were specified correctly
        self.check_shared_params()

        # throw error if joint inference does not make sense
        if not self.joint_inference_makes_sense():
            self._logger.warning('Joint inference does not make sense as no parameters are shared '
                                 'across more than one type. If this is not intended consider running '
                                 'several marginal inferences instead, which is more efficient.')

        # issue notice
        self._logger.info(f'Using shared parameters {self.shared_params}.')

        #: Covariates indexed by parameter name
        self.covariates: Dict[str, Covariate] = {f"c{i}": cov for i, cov in enumerate(covariates)}

        # only show param and values
        cov_repr = dict((k, dict(param=v.param, values=v.values)) for k, v in self.covariates.items())

        # issue notice
        self._logger.info(f'Including covariates: {cov_repr}.')

        # parameter names for covariates
        args_cov = list(self.covariates.keys())

        # bounds for covariates
        bounds_cov = dict((k, cov.bounds) for k, cov in self.covariates.items())

        #: Initial values for covariates
        self._x0_cov = dict((k, cov.x0) for k, cov in self.covariates.items())

        # use linear scale for covariates
        scales_cov = dict((k, cov.bounds_scale) for k, cov in self.covariates.items())

        #: fixed parameters with expanded 'all' type
        self.fixed_params: Dict[str, Dict[str, float]] = expand_fixed(fixed_params, self.types)

        # collapse fixed parameters
        fixed_collapsed = collapse_fixed(self.fixed_params, self.types)

        # include 'all' type with infers the DFE for all spectra added together
        #: Dictionary of marginal inferences indexed by type
        self.marginal_inferences: Dict[str, BaseInference] = dict(
            all=BaseInference(
                sfs_neut=sfs_neut,
                sfs_sel=sfs_sel,
                discretization=self.discretization,
                include_divergence=include_divergence,
                model=model,
                seed=seed,
                x0=x0,
                bounds=bounds,
                scales=scales,
                loss_type=loss_type,
                opts_mle=opts_mle,
                fixed_params=dict(all=fixed_collapsed['all']) if 'all' in fixed_collapsed else {},
                do_bootstrap=do_bootstrap,
                n_bootstraps=n_bootstraps,
                n_bootstrap_retries=n_bootstrap_retries,
                parallelize=parallelize,
                folded=self.folded,
                n_runs=n_runs)
        )

        # check that the fixed parameters are valid
        self.check_fixed_params_exist()

        # check if the fixed parameters are compatible with the shared parameters
        self.check_no_shared_params_fixed()

        #: parameter scales
        self.scales: Dict[str, Literal['lin', 'log', 'symlog']] = \
            self.model.scales | self._default_scales | scales_cov | scales

        #: parameter bounds
        self.bounds: Dict[str, Tuple[float, float]] = self.model.bounds | self._default_bounds | bounds | bounds_cov

        # create optimization instance for joint inference
        # take initial values and bounds from marginal inferences
        # and from this inference for type 'all'
        #: Joint optimization instance
        self.optimization: Optimization = Optimization(
            bounds=self.bounds,
            scales=self.scales,
            opts_mle=self.optimization.opts_mle,
            loss_type=self.optimization.loss_type,
            param_names=self.model.param_names + ['eps'] + args_cov,
            parallelize=self.parallelize,
            fixed_params=self.fixed_params,
            seed=self.seed
        )

        # Construct the marginal inference object for each type.
        # Note that we use the same discretization
        # instance to avoid precomputing the linearization
        # several times.
        for t in sfs_neut.types:
            self.marginal_inferences[t] = BaseInference(
                # pass subtypes
                sfs_neut=sfs_neut[[t]],
                sfs_sel=sfs_sel[[t]],
                discretization=self.discretization,
                include_divergence=include_divergence,
                model=model,
                seed=seed,
                x0=x0,
                bounds=bounds,
                scales=scales,
                loss_type=loss_type,
                opts_mle=opts_mle,
                fixed_params=dict(all=self.fixed_params[t]) if t in self.fixed_params else {},
                do_bootstrap=do_bootstrap,
                n_bootstraps=n_bootstraps,
                n_bootstrap_retries=n_bootstrap_retries,
                parallelize=parallelize,
                folded=self.folded,
                n_runs=n_runs
            )

        #: Joint inference object indexed by type
        self.joint_inferences: Dict[str, BaseInference] = {}

        # add base inference object for each type
        for t in sfs_neut.types:
            self.joint_inferences[t] = BaseInference(
                # pass subtypes
                sfs_neut=sfs_neut[[t]],
                sfs_sel=sfs_sel[[t]],
                discretization=self.discretization,
                include_divergence=include_divergence,
                model=model,
                seed=seed,
                x0=x0,
                bounds=bounds,
                scales=scales,
                loss_type=loss_type,
                opts_mle=opts_mle,
                fixed_params=dict(all=self.fixed_params[t]) if t in self.fixed_params else {},
                do_bootstrap=do_bootstrap,
                n_bootstraps=n_bootstraps,
                n_bootstrap_retries=n_bootstrap_retries,
                parallelize=parallelize,
                folded=self.folded,
                n_runs=n_runs,
                locked=True
            )

    def check_shared_params(self):
        """
        Check if the shared parameters were specified correctly.
        """
        for shared in self.shared_params:
            # check parameters
            if len(set(shared.params) - set(self.param_names)) != 0:
                raise ValueError(
                    f'Specified shared parameters {list(set(shared.params) - set(self.param_names))} '
                    f"don't match with any of {self.param_names}."
                )

            # check types
            if len(set(shared.types) - set(self.types)) != 0:
                raise ValueError(
                    f'Specified types {list(set(shared.types) - set(self.types))} '
                    f" in shared parameters don't match with any of {self.types}."
                )

    def check_no_shared_params_fixed(self):
        """
        Check that no shared parameters are fixed and raise an error otherwise.
        """
        for shared in self.shared_params:
            for t in shared.types:
                if t in self.fixed_params:
                    for p in shared.params:
                        if p in self.fixed_params[t]:
                            raise ValueError(f"Parameter '{p}' in type '{t}' is both "
                                             f"shared and fixed, which is not allowed. "
                                             f"Note that covariates are automatically shared.")

    def get_shared_param_names(self) -> List[str]:
        """
        Get the names of the shared parameters.

        :return: Names of the shared parameters.
        """
        shared = []

        for p in self.shared_params:
            shared += p.params

        return cast(List[str], np.unique(shared).tolist())

    def _add_covariates_as_shared(self, covariates: List[Covariate]):
        """
        Add covariates as shared parameters.

        :param covariates: List of covariates.
        """
        cov_but_not_shared = self._determine_unshared_covariates(covariates)

        # add parameters with covariates to shared parameters
        if len(cov_but_not_shared) > 0:
            self._logger.info(f'Parameters {cov_but_not_shared} have '
                              f'covariates and thus need to be shared. '
                              f'Adding them to shared parameters.')

            # add to shared parameters
            self.shared_params.append(SharedParams(params=cov_but_not_shared, types=self.types))

    def _determine_unshared_covariates(self, covariates):
        """
        Determine which covariates are not shared.

        :param covariates:
        :return:
        """
        # determine completely shared parameters
        completely_shared = []
        for shared in self.shared_params:
            if len(shared.types) == len(self.types):
                completely_shared += shared.params

        # determine parameters with covariates
        params_with_covariates = [cov.param for cov in covariates]

        # determine parameters with covariates that are not shared
        return list(set(params_with_covariates) - set(completely_shared))

    def run(
            self,
            **kwargs
    ) -> Spectrum:
        """
        Run inference.

        :param kwargs: Additional keyword arguments
        :return: Modelled SFS.
        """

        # run marginal optimization
        self._run_marginal()

        # run joint optimization
        return self._run_joint()

    def _run_marginal(self):
        """
        Run marginal optimization.

        :return: Dict of marginal inferences indexed by type.
        """

        def run_marginal(data: Tuple[str, BaseInference]) -> Tuple[str, BaseInference]:
            """
            Run marginal inference. Start with initial values
            obtained from 'all' type.

            :param data: Tuple of type and inference object.
            :return: Tuple of type and inference object.
            """
            # issue notice
            self._logger.info(f"Running marginal inference for type '{data[0]}'.")

            data[1].run_if_required(
                do_bootstrap=False,
                pbar=False
            )

            return data

        # run for 'all' type
        run_marginal(('all', self.marginal_inferences['all']))

        # update initial values of marginal inferences
        # with MLE estimate of 'all' type
        for inf in self.marginal_inferences.values():
            inf.x0 = dict(all=self.marginal_inferences['all'].params_mle)

        # skip marginal inference if only one type was specified
        if len(self.types) == 1:
            self._logger.info('Skipping marginal inference as only one type was specified.')
        else:
            # issue notice
            self._logger.info(f'Running marginal inferences for types {self.types}.')

        # optionally parallelize marginal inferences
        run_inferences = dict(parallelize_func(
            func=run_marginal,
            data=list(self.marginals_without_all().items()),
            parallelize=False,
            pbar=False
        ))

        # reassign marginal inferences
        self.marginal_inferences = dict(all=self.marginal_inferences['all']) | run_inferences

        return self.marginal_inferences

    def marginals_without_all(self) -> Dict[str, BaseInference]:
        """
        Get marginal inference without 'all' type.

        :return: Dict of marginal inferences indexed by type.
        """
        return dict((t, inf) for t, inf in self.marginal_inferences.items() if t != 'all')

    def _run_joint(self) -> Spectrum:
        """
        Run joint optimization.

        :return: Modelled SFS.
        """
        # issue notice
        self._logger.info(f'Running joint inference for types {self.types}.')

        # issue notice
        self._logger.debug(f'Starting numerical optimization of {self.n_runs} '
                           'independently initialized samples which are run ' +
                           ('in parallel.' if self.parallelize else 'sequentially.'))

        # starting time of joint inference
        start_time = time.time()

        # Perform joint optimization.
        self.result, params_mle = self.optimization.run(
            x0=self.get_x0(),
            scales=self.scales,
            bounds=self.bounds,
            n_runs=self.n_runs,
            get_counts=self.get_counts(),
            desc=f'{self.__class__.__name__}>Performing joint inference'
        )

        # assign likelihoods
        self.likelihoods = self.optimization.likelihoods

        # store packed MLE params for later usage
        self.params_mle_raw = copy.deepcopy(params_mle)

        # unpack shared parameters
        params_mle = unpack_shared(params_mle)

        # normalize parameters for each type
        for t in self.types:
            params_mle[t] = self.model._normalize(params_mle[t])

        # report on optimization result
        self._report_result(self.result, params_mle)

        # assign optimization result and MLE parameters for each type
        for t, inf in self.joint_inferences.items():
            params_mle[t] = correct_values(
                params=Covariate._apply(self.covariates, params_mle[t], t),
                bounds=self.bounds,
                scales=self.scales
            )

            # remove effect of covariates and assign result
            inf._assign_result(self.result, params_mle[t])

            # assign execution time
            inf.execution_time = time.time() - start_time

        # assign MLE params
        self.params_mle = params_mle

        # assign joint likelihood
        self.likelihood = -self.result.fun

        # calculate L2 residual
        self.L2_residual = self.get_residual(2)

        # check L2 residual
        self._check_L2_residual()

        # add execution time
        self.execution_time += time.time() - start_time

        # perform bootstrap if configured
        if self.do_bootstrap:
            self.bootstrap()

        return self.sfs_mle

    def joint_inference_makes_sense(self) -> bool:
        """
        Check if joint inference makes sense.

        :return: Whether joint inference makes sense.
        """
        return len(self.types) >= 2 and len(self.shared_params) != 0

    def get_counts(self) -> dict:
        """
        Get callback functions for modelling SFS counts from given parameters.

        :return: Dict of callback functions indexed by type.
        """
        # Note that it's important we bind t into the lambda function
        # at the time of creation.
        return dict((t, (lambda params, t=t: inf._model_sfs(
            discretization=self.discretization,
            model=self.model,
            params=correct_values(Covariate._apply(self.covariates, params, t), self.bounds, self.scales),
            sfs_neut=self.joint_inferences[t].sfs_neut,
            sfs_sel=self.joint_inferences[t].sfs_sel,
            folded=self.folded
        ))) for t, inf in self.joint_inferences.items())

    @BaseInference._run_if_required_wrapper
    @functools.lru_cache
    def run_joint_without_covariates(self, do_bootstrap: bool = True) -> 'JointInference':
        """
        Run joint inference without covariates. Note that the result of this function is cached.

        :return: Joint inference instance devoid of covariates.
        """
        config = self.create_config()

        # retain shared parameters but remove covariates
        config.update(
            shared_params=self.shared_params,
            covariates={},
            do_bootstrap=do_bootstrap
        )

        # create copy
        other = JointInference.from_config(config)

        # issue notice
        self._logger.info('Running joint inference without covariates.')

        # run inference
        other.run()

        return other

    def create_config(self) -> 'Config':
        """
        Create a config object from the inference object.

        :return: Config object.
        """
        return Config(
            sfs_neut=Spectra.from_spectra(dict((t, inf.sfs_neut) for t, inf in self.marginals_without_all().items())),
            sfs_sel=Spectra.from_spectra(dict((t, inf.sfs_sel) for t, inf in self.marginals_without_all().items())),
            intervals_ben=self.discretization.intervals_ben,
            intervals_del=self.discretization.intervals_del,
            integration_mode=self.discretization.integration_mode,
            linearized=self.discretization.linearized,
            model=self.model,
            seed=self.seed,
            opts_mle=self.optimization.opts_mle,
            x0=self.x0,
            bounds=self.bounds,
            scales=self.scales,
            loss_type=self.optimization.loss_type,
            fixed_params=self.fixed_params,
            covariates=[c for c in self.covariates.values()],
            shared_params=self.shared_params,
            do_bootstrap=self.do_bootstrap,
            n_bootstraps=self.n_bootstraps,
            parallelize=self.parallelize,
            n_runs=self.n_runs
        )

    @BaseInference._run_if_required_wrapper
    def perform_lrt_covariates(self, do_bootstrap: bool = True) -> float:
        """
        Perform likelihood ratio test against joint inference without covariates.
        In the simple model we share parameters across types. Low p-values indicate that
        the covariates provide a significant improvement in the fit.

        To access the JointInference object without covariates, you can call :meth:`run_joint_without_covariates`,
        which is cached.

        :param do_bootstrap: Whether to bootstrap. This improves the accuracy of the p-value. Note
            that if bootstrapping was performed previously without updating the likelihood, this won't have any effect.
        :return: Likelihood ratio test p-value.
        """
        if len(self.covariates) == 0:
            raise ValueError('No covariates were specified.')

        # bootstrap if required
        if do_bootstrap:
            self.bootstrap_if_required()

        # run joint inference without covariates
        simple = self.run_joint_without_covariates(do_bootstrap=do_bootstrap)

        return self.lrt(simple.likelihood, self.likelihood, len(self.covariates))

    def _get_run_bootstrap_sample(self) -> Callable[[int], Tuple[OptimizeResult, dict]]:
        """
        Get function which runs a single bootstrap sample.

        :return: Static function which runs a single bootstrap sample, taking an optional seed and returning the
            optimization result and the MLE parameters.
        """
        optimization = self.optimization
        discretization = self.discretization
        covariates = self.covariates
        model = self.model
        types = self.types
        folded = self.folded
        params_mle_raw = self.params_mle_raw
        scales_linear = self.get_scales_linear()
        bounds_linear = self.get_bounds_linear()
        scales = self.scales
        bounds = self.bounds
        n_retries = self.n_bootstrap_retries
        sfs_neut = dict((t, self.marginal_inferences[t].sfs_neut) for t in self.types)
        sfs_sel = dict((t, self.marginal_inferences[t].sfs_sel) for t in self.types)

        def run_bootstrap_sample(seed: int) -> (OptimizeResult, dict):
            """
            Resample the observed selected SFS and rerun the optimization procedure.
            We take the MLE params as initial params here.
            We make this function static to improve performance when parallelizing.
            In case the optimization does not terminate normally, we retry up to `n_retries` times.

            :return: Optimization result and dictionary of MLE params
            """
            result, params_mle = None, None

            # retry up to `n_retries` times
            for i in range(max(n_retries, 0) + 1):

                # perform joint optimization
                # Note that it's important we bind t into the lambda function
                # at the time of creation.
                result, params_mle = optimization.run(
                    x0=params_mle_raw,
                    scales=scales_linear,
                    bounds=bounds_linear,
                    n_runs=1,
                    debug_iterations=False,
                    print_info=False,
                    desc=f"{self.__class__.__name__}>Bootstrapping joint inference",
                    get_counts=dict((t, lambda params, t=t: BaseInference._model_sfs(
                        discretization=discretization,
                        model=model,
                        params=correct_values(Covariate._apply(covariates, params, t), bounds, scales),
                        sfs_neut=sfs_neut[t].resample(seed=seed + i),
                        sfs_sel=sfs_sel[t].resample(seed=seed + i),
                        folded=folded
                    )) for t in types)
                )

                # unpack shared parameters
                params_mle = unpack_shared(params_mle)

                for t in types:
                    # normalize parameters for each type
                    params_mle[t] = model._normalize(params_mle[t])

                    # add covariates for each type
                    params_mle[t] = correct_values(
                        params=Covariate._apply(covariates, params_mle[t], t),
                        bounds=bounds,
                        scales=scales
                    )

                if result.success:
                    return result, params_mle

            return result, params_mle

        return run_bootstrap_sample

    def bootstrap(
            self,
            n_samples: int = None,
            parallelize: bool = None,
            n_retries: int = None,
            update_likelihood: bool = True,
            **kwargs
    ) -> Optional[pd.DataFrame]:
        """
        Perform the parametric bootstrap both for the marginal and joint inferences.

        :param n_samples: Number of bootstrap samples. Defaults to :attr:`n_bootstraps`.
        :param parallelize: Whether to parallelize computations. Defaults to :attr:`parallelize`.
        :param n_retries: Number of retries for bootstraps that did not terminate normally. Defaults to
            :attr:`n_bootstrap_retries`.
        :param update_likelihood: Whether to update the likelihood
        :return: DataFrame with bootstrap samples
        """
        # perform inference first if not done yet
        self.run_if_required()

        # update properties
        self.update_properties(
            n_bootstraps=n_samples,
            parallelize=parallelize,
            n_bootstrap_retries=n_retries
        )

        n_bootstraps = int(self.n_bootstraps)

        with tqdm(
                total=len(self.marginal_inferences) * n_bootstraps,
                disable=Settings.disable_pbar,
                desc=f"{self.__class__.__name__}>Bootstrapping marginal inferences"
        ) as pbar:

            # bootstrap marginal inferences
            for t, inf in self.marginal_inferences.items():
                self._logger.info(f"Bootstrapping type '{t}'.")

                inf.bootstrap(
                    n_samples=n_bootstraps,
                    parallelize=self.parallelize,
                    n_retries=self.n_bootstrap_retries,
                    update_likelihood=update_likelihood,
                    pbar=False
                )

                pbar.update(n_bootstraps)

        start_time = time.time()

        # parallelize computations if desired
        if self.parallelize:

            self._logger.debug(f"Running {n_bootstraps} joint bootstrap samples "
                               f"in parallel on {min(mp.cpu_count(), n_bootstraps)} cores.")

        else:
            self._logger.debug(f"Running {n_bootstraps} joint bootstrap samples sequentially.")

        # seeds for bootstraps
        seeds = self.rng.integers(0, high=2 ** 32, size=n_bootstraps)

        # run bootstraps
        result = parallelize_func(
            func=self._get_run_bootstrap_sample(),
            data=seeds,
            parallelize=self.parallelize,
            pbar=True,
            desc=f"{self.__class__.__name__}>Bootstrapping joint inference"
        )

        # number of successful runs
        n_success = np.sum([res.success for res in result[:, 0]])

        # issue warning if some runs did not finish successfully
        if n_success < n_bootstraps:
            self._logger.warning(
                f"{n_bootstraps - n_success} out of {n_bootstraps} bootstrap samples "
                "did not terminate normally during numerical optimization. "
                "The confidence intervals might thus be unreliable. Consider "
                "increasing the number of retries (`n_retries`), "
                "adjusting the optimization parameters (increasing `gtol` or `n_runs`), "
                "or decreasing the number of optimized parameters."
            )

        # dataframe of MLE estimates in flattened format
        self.bootstraps = pd.DataFrame([flatten_dict(r) for r in result[:, 1]])

        # assign bootstrap results
        self.bootstrap_results = list(result[:, 0])

        # assign bootstrap parameters to joint inference objects
        for t, inf in self.joint_inferences.items():
            # filter for columns belonging to the current type
            inf.bootstraps = self.bootstraps.filter(regex=f'{t}\\..*').rename(columns=lambda x: x.split('.')[-1])

            # add estimates for alpha to the bootstraps
            inf._add_alpha_to_bootstraps()

        # add execution time
        self.execution_time += time.time() - start_time

        # assign average likelihood of successful runs
        if update_likelihood:
            self.likelihood = np.mean([-res.fun for res in result[:, 0] if res.success] + [self.likelihood])

        return self.bootstraps

    def bootstrap_if_required(self):
        """
        Bootstrap if not done yet.
        """
        if self.bootstraps is None:
            self.bootstrap()

    def get_x0(self) -> Dict[str, Dict[str, float]]:
        """
        Get initial values for joint inference.

        :return: Dictionary of initial values indexed by type
        """
        x0 = {}

        # create initial values from marginal inferences
        for t, inf in self.marginals_without_all().items():

            # the MLE params might not be defined
            if inf.params_mle is not None:
                x0[t] = inf.params_mle
            else:
                x0[t] = self.marginal_inferences['all'].x0['all']

        # get shared parameters from last inference and merge
        # with parameters for type 'all'
        shared = {}

        # get dict of shared parameters
        for s in self.shared_params:
            for p in s.params:
                # take mean value over types
                shared[':'.join(s.types) + '.' + p] = cast(float, np.mean([x0[t][p] for t in s.types]))

        # pack shared parameters
        packed = pack_shared(x0, self.shared_params, shared)

        # add parameters for covariates and return
        return merge_dicts(packed, {':'.join(self.types): self._x0_cov})

    @BaseInference._run_if_required_wrapper
    def perform_lrt_shared(self, do_bootstrap: bool = True) -> float:
        """
        Compare likelihood of joint inference with product of marginal likelihoods.
        This provides information about the goodness of fit achieved by the parameter sharing.
        Low p-values indicate that parameter sharing is not justified, i.e., that the marginal
        inferences provide a better fit to the data. Note that it is more difficult to properly
        optimize the joint likelihood, which makes this test conservative, i.e., the reported p-value
        might be larger than what it really is.

        :param do_bootstrap: Whether to perform bootstrapping. This improves the accuracy of the p-value. Note
            that if bootstrapping was performed previously without updating the likelihood, this won't have any effect.
        :return: p-value
        """
        if do_bootstrap:
            self.bootstrap_if_required()

        # determine likelihood of marginal inferences
        ll_marginal = sum([inf.likelihood for inf in self.marginals_without_all().values()])

        # determine number of parameters
        n_marginal = np.sum([len(inf.params_mle) for inf in self.marginals_without_all().values()])
        n_joint = len(flatten_dict(self.get_x0()))

        return self.lrt(ll_simple=self.likelihood, ll_complex=ll_marginal, df=n_marginal - n_joint)

    def get_inferences(
            self,
            types: List[str] = None,
            labels: List[str] = None,
            show_marginals: bool = True
    ) -> Dict[str, 'BaseInference']:
        """
        Get all inference objects as dictionary.

        :param types: Types to include
        :param labels: Labels for types
        :param show_marginals: Whether to also show marginal inferences
        :return: Dictionary of base inference objects indexed by type and joint vs marginal subtypes
        """

        def get(infs: Dict[str, BaseInference], prefix: str) -> Dict[str, BaseInference]:
            """
            Get filtered and prefixed inferences.

            :param infs: Dictionary of inferences.
            :param prefix: Prefix
            :return: Dictionary of inferences
            """
            # filter types if types are given
            if types is not None:
                infs = dict((k, v) for k, v in infs.items() if k in types)

            # add prefix to keys
            return dict((prefix + '.' + k, v) for k, v in infs.items())

        # get joint inferences
        inferences = get(self.joint_inferences, 'joint')

        if show_marginals:
            # include marginal inferences
            inferences = get(self.marginal_inferences, 'marginal') | inferences

        # use labels as keys if given
        if labels is not None:

            if len(labels) != len(inferences):
                raise ValueError(f'Number of labels ({len(labels)}) does not match '
                                 f'number of inferences ({len(inferences)}).')

            inferences = dict(zip(labels, inferences.values()))

        return inferences

    @BaseInference._run_if_required_wrapper
    def plot_discretized(
            self,
            file: str = None,
            show: bool = True,
            intervals: np.ndarray = np.array([-np.inf, -100, -10, -1, 0, 1, np.inf]),
            show_marginals: bool = True,
            confidence_intervals: bool = True,
            ci_level: float = 0.05,
            bootstrap_type: Literal['percentile', 'bca'] = 'percentile',
            title: str = 'discretized DFE comparison',
            labels: List[str] = None,
            kwargs_legend: dict = dict(prop=dict(size=8)),
            ax: plt.Axes = None
    ) -> plt.Axes:
        """
        Plot discretized DFE comparing the different types.

        :param labels: Labels for types
        :param title: Title of plot
        :param intervals: Array of interval boundaries yielding ``intervals.shape[0] - 1`` bars.
        :param show_marginals: Whether to also show marginal inferences
        :param bootstrap_type: Type of bootstrap
        :param ci_level: Confidence level
        :param confidence_intervals: Whether to plot confidence intervals
        :param file: File to save plot to
        :param show: Whether to show plot
        :param ax: Axes to plot on. Only for Python visualization backend.
        :param kwargs_legend: Keyword arguments passed to :meth:`plt.legend`. Only for Python visualization backend.
        :return: Axes object
        """
        labels, inferences = zip(*self.get_inferences(labels=labels, show_marginals=show_marginals).items())

        return Inference.plot_discretized(**locals())

    def plot_sfs_comparison(
            self,
            sfs_types: List[Literal['modelled', 'observed', 'selected', 'neutral']] = ['observed', 'modelled'],
            types: List[str] = None,
            labels: List[str] = None,
            colors: List[str] = None,
            file: str = None,
            show: bool = True,
            ax: plt.Axes = None,
            title: str = 'SFS comparison',
            use_subplots: bool = False,
            show_monomorphic: bool = False,
            kwargs_legend: dict = dict(prop=dict(size=8)),

    ) -> plt.Axes:
        """
        Plot SFS comparison.

        :param types: Types to plot
        :param file: File to save plot to
        :param labels: Labels for types.
        :param colors: Colors for types. Only for Python visualization backend.
        :param sfs_types: Types of SFS to plot
        :param show: Whether to show plot
        :param ax: Axes to plot on. Only for Python visualization backend and if ``use_subplots`` is ``False``.
        :param title: Plot title
        :param use_subplots: Whether to use subplots. Only for Python visualization backend.
        :param show_monomorphic: Whether to show monomorphic counts
        :param kwargs_legend: Keyword arguments passed to :meth:`plt.legend`. Only for Python visualization backend.
        :return: Axes object
        """
        if 'modelled' in sfs_types:
            self.run_if_required()

        mapping = dict(
            observed='sfs_sel',
            selected='sfs_sel',
            modelled='sfs_mle',
            neutral='sfs_neut'
        )

        inferences = self.get_inferences(types=types)

        def get_label(t: str, sfs_type: str) -> str:
            """
            Get label for types.

            :param t: Type
            :param sfs_type: SFS type
            :return: Label
            """
            subtypes = t.split('.')

            # insert sfs type at second position
            return '.'.join([subtypes[0]] + [sfs_type] + subtypes[1:])

        # get spectra
        spectra = {get_label(t, sfs): getattr(inf, mapping[sfs]) for t, inf in inferences.items() for sfs in sfs_types}

        return Visualization.plot_spectra(
            spectra=[list(v) for v in spectra.values()],
            labels=list(spectra.keys()) if labels is None else labels,
            colors=colors,
            file=file,
            show=show,
            ax=ax,
            title=title,
            use_subplots=use_subplots,
            show_monomorphic=show_monomorphic,
            kwargs_legend=kwargs_legend
        )

    @BaseInference._run_if_required_wrapper
    def plot_continuous(
            self,
            file: str = None,
            show: bool = True,
            intervals: np.ndarray = None,
            confidence_intervals: bool = True,
            ci_level: float = 0.05,
            bootstrap_type: Literal['percentile', 'bca'] = 'percentile',
            title: str = 'DFE comparison',
            labels: List[str] = None,
            scale_density: bool = False,
            scale: Literal['lin', 'log', 'symlog'] = 'lin',
            kwargs_legend: dict = dict(prop=dict(size=8)),
            ax: plt.Axes = None
    ) -> plt.Axes:
        """
        Plot discretized DFE. The special constants ``np.inf`` and ``-np.inf`` are also valid interval bounds.
        By default, the PDF is plotted as is. Due to the logarithmic scale on
        the x-axis, we may get a wrong intuition on how the mass is distributed,
        however. To get a better intuition, we can optionally scale the density
        by the x-axis interval size using ``scale_density = True``. This has the
        disadvantage that the density now changes for x, so that even a constant
        density will look warped.

        :param scale_density: Whether to scale the density by the x-axis interval size
        :param scale: y-scale
        :param labels: Labels for types
        :param title: Title of plot
        :param bootstrap_type: Type of bootstrap
        :param ci_level: Confidence level
        :param confidence_intervals: Whether to plot confidence intervals
        :param file: File to save plot to
        :param show: Whether to show plot
        :param intervals: Array of interval boundaries yielding ``intervals.shape[0] - 1`` bins.
        :param kwargs_legend: Keyword arguments passed to :meth:`plt.legend`. Only for Python visualization backend.
        :param ax: Axes to plot on. Only for Python visualization backend.
        :return: Axes object
        """
        if intervals is None:
            intervals = self.discretization.bins

        labels, inferences = zip(*self.get_inferences(labels=labels).items())

        return Inference.plot_continuous(**locals())

    @BaseInference._run_if_required_wrapper
    def plot_inferred_parameters(
            self,
            file: str = None,
            confidence_intervals: bool = True,
            bootstrap_type: Literal['percentile', 'bca'] = 'percentile',
            ci_level: float = 0.05,
            show: bool = True,
            title: str = 'inferred parameters',
            labels: List[str] = None,
            ax: plt.Axes = None,
            scale: Literal['lin', 'log', 'symlog'] = 'log',
            kwargs_legend: dict = dict(prop=dict(size=8), loc='upper right'),
            **kwargs: List[str]
    ) -> plt.Axes:
        """
        Plot discretized DFE comparing the different types.

        :param labels: Labels for types
        :param title: Title of plot
        :param bootstrap_type: Type of bootstrap
        :param ci_level: Confidence level
        :param confidence_intervals: Whether to plot confidence intervals
        :param file: File to save plot to
        :param show: Whether to show plot
        :param ax: Axes to plot on. Only for Python visualization backend.
        :param scale: y-scale
        :param kwargs_legend: Keyword arguments passed to :meth:`plt.legend`. Only for Python visualization backend.
        :return: Axes object
        """
        labels, inferences = zip(*self.get_inferences(labels=labels).items())

        return Inference.plot_inferred_parameters(**locals())

    @BaseInference._run_if_required_wrapper
    def plot_inferred_parameters_boxplot(
            self,
            file: str = None,
            show: bool = True,
            title: str = 'inferred parameters',
            labels: List[str] = None,
            **kwargs: List[str]
    ) -> plt.Axes:
        """
        Plot discretized DFE comparing the different types.

        :param labels: Labels for types
        :param title: Title of plot
        :param file: File to save plot to
        :param show: Whether to show plot
        :return: Axes object
        :raises ValueError: If no inference objects are given or no bootstraps are found.
        """
        labels, inferences = zip(*self.get_inferences(labels=labels).items())

        return Inference.plot_inferred_parameters_boxplot(**locals())

    @BaseInference._run_if_required_wrapper
    def plot_covariate(
            self,
            index: int = 0,
            file: str = None,
            show: bool = True,
            title: str = None,
            bootstrap_type: Literal['percentile', 'bca'] = 'percentile',
            show_types: bool = True,
            ci_level: float = 0.05,
            xlabel: str = "cov",
            ylabel: str = None,
            ax: plt.Axes = None
    ) -> plt.Axes:
        """
        Plot the covariate given by the index.

        :param index: The index of the covariate.
        :param file: File to save plot to.
        :param show: Whether to show plot.
        :param title: Plot title.
        :param bootstrap_type: Bootstrap type.
        :param show_types: Whether to show types on second x-axis.
        :param ci_level: Confidence level.
        :param xlabel: X-axis label.
        :param ylabel: Y-axis label, defaults to the covariate parameter name.
        :param ax: Axes to plot on. Only for Python visualization backend.
        :return: Axes object.
        """
        key = f"c{index}"

        # check if covariate exists
        if key not in self.covariates:
            raise ValueError(f"Covariate with index {index} does not exist.")

        # default title
        if title is None:
            title = f'covariate {key}'

        cov = self.covariates[key]
        values = [self.params_mle[t][cov.param] for t in self.types]

        # get errors if bootstrapped
        errors = None
        if self.bootstraps is not None:
            bootstraps = np.array([self.bootstraps[f"{t}.{cov.param}"] for t in self.types]).T

            # compute errors
            errors = Bootstrap.get_errors(
                values=values,
                bs=bootstraps,
                bootstrap_type=bootstrap_type,
                ci_level=ci_level
            )[0]

            # take mean of bootstraps as values
            values = bootstraps.mean(axis=0)

        return Visualization.plot_covariate(
            covariates=[cov.values[t] for t in self.types],
            values=values,
            errors=errors,
            file=file,
            show=show,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel or cov.param,
            labels=self.types if show_types else None,
            ax=ax
        )

    def get_cis_params_mle(
            self,
            bootstrap_type: Literal['percentile', 'bca'] = 'percentile',
            ci_level: float = 0.05,
            param_names: Optional[list[str]] = None
    ):
        """
        Get confidence intervals for the parameters.

        :return: Confidence intervals for the parameters
        """
        # get dict of inferences
        inferences = self.get_inferences()

        # get param names if not given
        if param_names is None:
            param_names = list(inferences.values())[0].get_bootstrap_param_names()

        return Inference.get_cis_params_mle(
            inferences=list(inferences.values()),
            bootstrap_type=bootstrap_type,
            ci_level=ci_level,
            param_names=param_names,
            labels=list(inferences.keys())
        )

    def get_discretized(
            self,
            intervals: np.ndarray = np.array([-np.inf, -100, -10, -1, 0, 1, np.inf]),
            confidence_intervals: bool = True,
            ci_level: float = 0.05,
            bootstrap_type: Literal['percentile', 'bca'] = 'percentile'
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Get discretized DFEs.

        :param bootstrap_type: Type of bootstrap
        :param ci_level: Confidence interval level
        :param confidence_intervals: Whether to return confidence intervals
        :param intervals: Array of interval boundaries yielding ``intervals.shape[0] - 1`` bins.
        :return: Dictionary of array of values and array of errors indexed by inference type
        """
        return Inference.get_discretized(
            inferences=list(self.get_inferences().values()),
            labels=list(self.get_inferences().keys()),
            intervals=intervals,
            confidence_intervals=confidence_intervals,
            ci_level=ci_level,
            bootstrap_type=bootstrap_type
        )

    def get_bootstrap_params(self) -> Dict[str, float]:
        """
        Get bootstrap parameters.

        :return: Bootstrap parameters
        """
        return flatten_dict(dict((t, self.joint_inferences[t].get_bootstrap_params()) for t in self.types))

    def to_json(self) -> str:
        """
        Serialize object. Note that the deserialized inference objects no
        longer share the same optimization instance among other things.

        :return: JSON string
        """
        # using make_ref=True resulted in weird behaviour when unserializing.
        return jsonpickle.encode(self, indent=4, warn=True, make_refs=False)

    @functools.cached_property
    def alpha(self) -> Optional[float]:
        """
        The is no single alpha for the joint inference. Please refer
        to the ``self.joint_inferences[t].alpha``.

        :return: None
        """
        return

    def _set_fixed_params(self, params: Dict[str, Dict[str, float]]):
        """
        Set fixed parameters.

        :param params: Fixed parameters
        """
        # set for 'all' type
        self.marginal_inferences['all']._set_fixed_params(params)

        # expand types
        self.fixed_params = expand_fixed(params, self.types)

        # propagate to inference objects
        for t in self.types:
            self.marginal_inferences[t]._set_fixed_params(dict(all=self.fixed_params[t]))
            self.joint_inferences[t]._set_fixed_params(dict(all=self.fixed_params[t]))

        # propagate to optimization
        self.optimization.set_fixed_params(self.fixed_params)

        # check if the fixed parameters are compatible with the shared parameters
        self.check_no_shared_params_fixed()

    def get_residual(self, k: int) -> float:
        """
        Residual of joint inference. We calculate the residual over the jointly inferred SFS for all types.

        :param k: Order of the norm
        :return: L2 residual
        """
        counts_mle = np.array([inf.sfs_mle.polymorphic for inf in self.joint_inferences.values()]).flatten()
        counts_sel = np.array([inf.sfs_sel.polymorphic for inf in self.joint_inferences.values()]).flatten()

        return norm(counts_mle - counts_sel, k)
