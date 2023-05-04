"""
Shared inference module.
"""

__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"
__date__ = "2023-02-26"

import copy
import functools
import logging
import time
from typing import List, Dict, Tuple, Literal, Optional, cast

import jsonpickle
import multiprocess as mp
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import OptimizeResult
from numpy.linalg import norm

from . import Config
from .abstract_inference import Inference
from .base_inference import BaseInference
from .optimization import Optimization, SharedParams, pack_shared, expand_shared, \
    Covariate, flatten_dict, merge_dicts, correct_values, parallelize as parallelize_func, expand_fixed, collapse_fixed, \
    unpack_shared
from .parametrization import Parametrization
from .spectrum import Spectrum, Spectra

# get logger
logger = logging.getLogger('fastdfe')


class SharedInference(BaseInference):
    """
    Enabling the sharing of parameters among several Inference objects.
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
            parallelize: bool = True,
    ):
        """
        Create instance.

        :param sfs_neut: Neutral SFS
        :param sfs_sel: Selected SFS
        :param include_divergence: Whether to include divergence in the likelihood
        :param intervals_del: ``(start, stop, n_interval)`` for deleterious population-scaled
        selection coefficients. The intervals will be log10-spaced.
        :param intervals_ben: Same as intervals_del but for beneficial selection coefficients
        :param integration_mode: Integration mode
        :param linearized: Whether to use the linearized model
        :param model: DFE parametrization
        :param seed: Random seed
        :param x0: Dictionary of initial values in the form ``{type: {param: value}}``
        :param bounds: Bounds for the optimization in the form {param: (lower, upper)}
        :param scales: Scales for the optimization in the form {param: scale}
        :param loss_type: Loss type
        :param opts_mle: Options for the optimization
        :param n_runs: Number of independent optimization runs
        :param fixed_params: dictionary of fixed parameters in the form ``{type: {param: value}}``
        :param shared_params: List of shared parameters
        :param do_bootstrap: Whether to perform bootstrapping
        :param n_bootstraps: Number of bootstraps
        :param parallelize: Whether to parallelize the optimization
        """
        # check whether types are equal
        if set(sfs_neut.types) != set(sfs_sel.types):
            raise Exception('The neutral and selected spectra must have exactly the same types.')

        #: SFS types
        self.types: List[str] = sfs_neut.types

        # initialize parent
        BaseInference.__init__(**locals())

        #: original MLE parameters before adding covariates and unpacked shared
        self.params_mle_raw: Optional[Dict[str, Dict[str, float]]] = None

        #: Shared parameters with expanded 'all' type
        self.shared_params = expand_shared(shared_params, self.types, self.optimization.param_names)

        # add covariates as shared parameters
        self.add_covariates_as_shared(covariates)

        # check if the shared parameters were specified correctly
        self.check_shared_params()

        # throw error if joint inference does not make sense
        if not self.joint_inference_makes_sense():
            logger.warning('Joint inference does not make sense as there are no parameters to be shared '
                           'across more than one types. If this is not intended consider running '
                           'several marginal inferences instead, which is more efficient.')

        # issue notice
        logger.info(f'Using shared parameters {self.shared_params}.')

        #: Covariates indexed by parameter name
        self.covariates: Dict[str, Covariate] = {f"c{i}": cov for i, cov in enumerate(covariates)}

        # only show param and values
        cov_repr = dict((k, dict(param=v.param, values=v.values)) for k, v in self.covariates.items())

        # issue notice
        logger.info(f'Including covariates: {cov_repr}.')

        # parameter names for covariates
        args_cov = list(self.covariates.keys())

        # bounds for covariates
        bounds_cov = dict((k, cov.bounds) for k, cov in self.covariates.items())

        #: Initial values for covariates
        self.x0_cov = dict((k, cov.x0) for k, cov in self.covariates.items())

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
                parallelize=parallelize,
                n_runs=n_runs)
        )

        # check that the fixed parameters are valid
        self.check_fixed_params_exist()

        # check if the fixed parameters are compatible with the shared parameters
        self.check_no_shared_params_fixed()

        #: parameter scales
        self.scales: Dict[str, Literal['lin', 'log', 'symlog']] = \
            self.model.scales | self.default_scales | scales_cov | scales

        #: parameter bounds
        self.bounds: Dict[str, Tuple[float, float]] = self.model.bounds | self.default_bounds | bounds | bounds_cov

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
                parallelize=parallelize,
                n_runs=n_runs
            )

        #: Joint inference object indexed by type
        self.joint_inferences: Dict[str, BaseInference] = {}

        # only add if more than one type and at least one shared parameter is given
        if len(self.types) > 1 and len(self.shared_params) > 0:
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
                    parallelize=parallelize,
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

    def add_covariates_as_shared(self, covariates: List[Covariate]):
        """
        Add covariates as shared parameters.

        :param covariates: List of covariates.
        """
        cov_but_not_shared = self.determine_unshared_covariates(covariates)

        # add parameters with covariates to shared parameters
        if len(cov_but_not_shared) > 0:
            logger.info(f'Parameters {cov_but_not_shared} have '
                        f'covariates and thus need to be shared. '
                        f'Adding them to shared parameters.')

            # add to shared parameters
            self.shared_params.append(SharedParams(params=cov_but_not_shared, types=self.types))

    def determine_unshared_covariates(self, covariates):
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
            do_bootstrap: bool = None,
            pbar: bool = True,
            **kwargs
    ) -> Spectrum:
        """
        Run inference.

        :param do_bootstrap: Whether to perform bootstrapping.
        :param pbar: Whether to show progress bar.
        :param kwargs: Additional keyword arguments passed to
        :return: Modelled SFS.
        """

        # run marginal optimization
        self.run_marginal()

        # run joint optimization
        return self.run_joint()

    def run_marginal(self):
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
            data[1].run_if_required(
                do_bootstrap=False,
                pbar=False
            )

            return data

        # issue notice
        logger.info(f"Running marginal inferences for type 'all'.")

        # run for 'all' type
        run_marginal(('all', self.marginal_inferences['all']))

        # update initial values of marginal inferences
        # with MLE estimate of 'all' type
        for inf in self.marginal_inferences.values():
            inf.x0 = dict(all=self.marginal_inferences['all'].params_mle)

        # skip marginal inference if only one type was specified
        if len(self.types) == 1:
            logger.info('Skipping marginal inference as only one type was specified.')
        else:
            # issue notice
            logger.info(f'Running marginal inferences for types {self.types}.')

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

    def run_joint(self):
        """
        Run joint optimization.

        :return: Modelled SFS.
        """
        # issue notice
        logger.info(f'Running joint inference.')

        # issue notice
        logger.info(f'Starting numerical optimization of {self.n_runs} '
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
            get_counts=self.get_counts()
        )

        # assign likelihoods
        self.likelihoods = self.optimization.likelihoods

        # store packed MLE params for later usage
        self.params_mle_raw = copy.deepcopy(params_mle)

        # unpack shared parameters
        params_mle = unpack_shared(params_mle)

        # normalize parameters for each type
        for t in self.types:
            params_mle[t] = self.model.normalize(params_mle[t])

        # report on optimization result
        self.report_result(self.result, params_mle)

        # assign optimization result and MLE parameters for each type
        for t, inf in self.joint_inferences.items():
            params_mle[t] = correct_values(
                params=self.add_covariates(params_mle[t], t),
                bounds=self.bounds,
                scales=self.scales
            )

            # remove effect of covariates and assign result
            inf.assign_result(self.result, params_mle[t])

            # assign execution time
            inf.execution_time = time.time() - start_time

        # assign MLE params
        self.params_mle = params_mle

        # assign joint likelihood
        self.likelihood = -self.result.fun

        # calculate L2 residual
        self.L2_residual = self.get_L2_residual()

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
        return dict((t, (lambda params, t=t: inf.model_sfs(
            correct_values(self.add_covariates(params, t), self.bounds, self.scales),
            sfs_neut=self.joint_inferences[t].sfs_neut,
            sfs_sel=self.joint_inferences[t].sfs_sel
        ))) for t, inf in self.joint_inferences.items())

    @BaseInference.run_if_required_wrapper
    @functools.lru_cache
    def run_joint_without_covariates(self, do_bootstrap: bool = True) -> 'SharedInference':
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
        other = SharedInference.from_config(config)

        # issue notice
        logger.info('Running joint inference without covariates.')

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
            model=self.model.__class__.__name__,
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

    @BaseInference.run_if_required_wrapper
    def perform_lrt_covariates(self, do_bootstrap: bool = True) -> float:
        """
        Perform likelihood ratio test against joint inference without covariates.
        In the simple model we share parameters across types. Low p-values indicate that
        the covariates provide a significant improvement in the fit.

        :param do_bootstrap: Whether to bootstrap. This improves the accuracy of the p-value. Note
        that if bootstrapping was performed previously without updating the likelihood, this won't have any effect.
        :return: Likelihood ratio test statistic.
        """
        if len(self.covariates) == 0:
            raise ValueError('No covariates were specified.')

        # bootstrap if required
        if do_bootstrap:
            self.bootstrap_if_required()

        # run joint inference without covariates
        simple = self.run_joint_without_covariates(do_bootstrap=do_bootstrap)

        return self.lrt(simple.likelihood, self.likelihood, len(self.covariates))

    def add_covariates(self, params: dict, type: str) -> dict:
        """
        Add covariates to parameters.

        :param params: Dict of parameters
        :param type: SFS type
        :return: Dict of parameters with covariates added
        """
        for k, cov in self.covariates.items():
            params = cov.apply(
                covariate=params[k],
                type=type,
                params=params
            )

        return params

    def bootstrap(
            self,
            n_samples: int = None,
            parallelize: bool = None,
            update_likelihood: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        Perform the parametric bootstrap both for the marginal and joint inferences.

        :param n_samples: Number of bootstrap samples
        :param parallelize: Whether to parallelize computations
        :param update_likelihood: Whether to update the likelihood
        :return: DataFrame with bootstrap samples
        """
        # perform inference first if not done yet
        self.run_if_required()

        # update properties
        self.update_properties(
            n_bootstraps=n_samples,
            parallelize=parallelize
        )

        logger.info("Bootstrapping marginal inferences.")

        # bootstrap marginal inferences
        for inf in self.marginal_inferences.values():
            inf.bootstrap(
                n_samples=n_samples,
                parallelize=self.parallelize,
                update_likelihood=update_likelihood
            )

        start_time = time.time()

        logger.info("Bootstrapping joint inference.")

        # parallelize computations if desired
        if self.parallelize:

            logger.info(f"Running {self.n_bootstraps} joint bootstrap samples "
                        f"in parallel on {mp.cpu_count()} cores.")

            # We need to assign new random states to the subprocesses.
            # Otherwise, they would all produce the same result.
            seeds = self.rng.integers(0, high=2 ** 32, size=self.n_bootstraps)

        else:
            logger.info(f"Running {self.n_bootstraps} joint bootstrap samples sequentially.")

            seeds = [None] * self.n_bootstraps

        # run bootstraps
        result = parallelize_func(
            func=self.run_joint_bootstrap_sample,
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

        # dataframe of MLE estimates in flattened format
        self.bootstraps = pd.DataFrame([flatten_dict(r) for r in result[:, 1]])

        # assign bootstrap parameters to joint inferences
        for t, inf in self.joint_inferences.items():
            inf.bootstraps = self.bootstraps.filter(regex=f'{t}.*').rename(columns=lambda x: x.split('.')[-1])

            # add estimates for alpha to the bootstraps
            inf.add_alpha_to_bootstraps()

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

    def run_joint_bootstrap_sample(self, seed: int = None) -> (OptimizeResult, dict):
        """
        Resample the observed selected SFS and rerun the optimization procedure.
        We take the MLE params as initial params here.

        :return: Optimization result and dictionary of MLE params
        """
        # perform joint optimization
        # Note that it's important we bind t into the lambda function
        # at the time of creation.
        result, params_mle = self.optimization.run(
            x0=self.params_mle_raw,
            scales=self.get_scales_linear(),
            bounds=self.get_bounds_linear(),
            n_runs=1,
            debug_iterations=False,
            print_info=False,
            get_counts=dict((t, lambda params, t=t: inf.model_sfs(
                correct_values(self.add_covariates(params, t), self.bounds, self.scales),
                sfs_neut=self.resample_sfs(self.marginal_inferences[t].sfs_neut, seed=seed),
                sfs_sel=self.resample_sfs(self.marginal_inferences[t].sfs_sel, seed=seed)
            )) for t, inf in self.marginals_without_all().items())
        )

        # unpack shared parameters
        params_mle = unpack_shared(params_mle)

        for t in self.types:
            # normalize parameters for each type
            params_mle[t] = self.model.normalize(params_mle[t])

            # add covariates for each type
            params_mle[t] = correct_values(
                params=self.add_covariates(params_mle[t], t),
                bounds=self.bounds,
                scales=self.scales
            )

        return result, params_mle

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
        return merge_dicts(packed, {':'.join(self.types): self.x0_cov})

    @BaseInference.run_if_required_wrapper
    def perform_lrt_shared(self, do_bootstrap: bool = True) -> float:
        """
        Compare likelihood of shared inference with product of marginal likelihoods.
        This provides information about the goodness of fit achieved by the parameter sharing.
        Low p-values indicate that parameter sharing is not justified, i.e., that the marginal
        inferences provide a better fit to the data. Note that it is more difficult to properly
        optimize the joint likelihood, which makes this test conservative, i.e., the p-value
        might be larger than it should be.

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
            labels: List[str] = None
    ) -> Dict[str, 'BaseInference']:
        """
        Get all inference objects as dictionary.

        :return: Dictionary of base inference objects indexed by type and joint vs marginal subtypes
        """
        marginal = self.marginal_inferences
        joint = self.joint_inferences

        # filter types if types are given
        if types is not None:
            marginal = dict((k, v) for k, v in marginal.items() if k in types)
            joint = dict((k, v) for k, v in joint.items() if k in types)

        # add prefix to keys
        marginal = dict(('marginal.' + k, v) for k, v in marginal.items())
        joint = dict(('joint.' + k, v) for k, v in joint.items())

        # merge dictionaries
        inferences = marginal | joint

        # use labels as keys if given
        if labels is not None:
            inferences = dict(zip(labels, inferences.values()))

        return inferences

    @BaseInference.run_if_required_wrapper
    def plot_discretized(
            self,
            file: str = None,
            show: bool = True,
            intervals: np.ndarray = np.array([-np.inf, -100, -10, -1, 0, 1, np.inf]),
            confidence_intervals: bool = True,
            ci_level: float = 0.05,
            bootstrap_type: Literal['percentile', 'bca'] = 'percentile',
            intervals_del: (float, float, int) = (-1.0e+8, -1.0e-5, 1000),
            intervals_ben: (float, float, int) = (1.0e-5, 1.0e4, 1000),
            title: str = 'discretized DFE comparison',
            labels: List[str] = None,
            ax: plt.Axes = None
    ) -> plt.Axes:
        """
        Plot discretized DFE comparing the different types.

        :param labels: Labels for types
        :param title: Title of plot
        :param intervals_ben: Interval boundaries for beneficial DFE
        :param intervals_del: Interval boundaries for deleterious DFE
        :param bootstrap_type: Type of bootstrap
        :param ci_level: Confidence level
        :param confidence_intervals: Whether to plot confidence intervals
        :param file: File to save plot to
        :param show: Whether to show plot
        :param intervals: Array of interval boundaries yielding ``intervals.shape[0] - 1`` bars.
        :param ax: Axes object
        :return: Axes object
        """
        labels, inferences = zip(*self.get_inferences(labels=labels).items())

        return Inference.plot_discretized(**locals())

    def plot_sfs_comparison(
            self,
            sfs_types: List[Literal['modelled', 'observed', 'modelled', 'neutral']] = ['modelled', 'observed'],
            types: List[str] = None,
            labels: List[str] = None,
            file: str = None,
            show: bool = True,
            ax: plt.Axes = None

    ) -> plt.Axes:
        """
        Plot SFS comparison.

        :param types: Types to plot
        :param file: File to save plot to
        :param labels: Labels for types
        :param sfs_types: Types of SFS to plot
        :param show: Whether to show plot
        :param ax: Axes object
        :return: Axes object
        """
        from fastdfe import Visualization

        if 'modelled' in sfs_types:
            self.run_if_required()

        mapping = dict(
            observed='sfs_sel',
            selected='sfs_sel',
            modelled='sfs_mle',
            neutral='sfs_neut'
        )

        inferences = self.get_inferences(types=types).items()

        def get_label(type: str, sfs_type: str) -> str:
            """
            Get label for types.

            :param type: Type
            :param sfs_type: SFS type
            :return: Label
            """
            subtypes = type.split('.')

            # insert sfs type at second position
            return '.'.join(np.insert(subtypes, 1, sfs_type))

        # get spectra
        spectra = {get_label(t, sfs): getattr(inf, mapping[sfs]) for t, inf in inferences for sfs in sfs_types}

        return Visualization.plot_sfs_comparison(
            spectra=list(spectra.values()),
            labels=list(spectra.keys()) if labels is None else labels,
            file=file,
            show=show,
            ax=ax
        )

    @BaseInference.run_if_required_wrapper
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
        :param ax: Axes object
        :return: Axes object
        """
        if intervals is None:
            intervals = self.discretization.bins

        labels, inferences = zip(*self.get_inferences(labels=labels).items())

        return Inference.plot_continuous(**locals())

    @BaseInference.run_if_required_wrapper
    def plot_inferred_parameters(
            self,
            file: str = None,
            confidence_intervals: bool = True,
            bootstrap_type: Literal['percentile', 'bca'] = 'percentile',
            ci_level: float = 0.05,
            show: bool = True,
            title: str = 'inferred parameters',
            labels: List[str] = None,
            legend: bool = True,
            ax: plt.Axes = None,
            **kwargs: List[str]
    ) -> plt.Axes:
        """
        Plot discretized DFE comparing the different types.

        :param legend: Whether to show legend
        :param labels: Labels for types
        :param title: Title of plot
        :param bootstrap_type: Type of bootstrap
        :param ci_level: Confidence level
        :param confidence_intervals: Whether to plot confidence intervals
        :param file: File to save plot to
        :param show: Whether to show plot
        :param ax: Axes object
        :return: Axes object
        """
        labels, inferences = zip(*self.get_inferences(labels=labels).items())

        return Inference.plot_inferred_parameters(**locals())

    def get_bootstrap_params(self) -> Dict[str, float]:
        """
        Get bootstrap parameters.

        :return: Bootstrap parameters
        """
        return flatten_dict(dict((t, self.joint_inferences[t].get_bootstrap_params()) for t in self.types))

    @BaseInference.run_if_required_wrapper
    def plot_covariates(
            self,
            file: str = None,
            show: bool = True,
            axs: List[plt.Axes] = None
    ) -> List[plt.Axes]:
        """
        Plot inferred parameters of joint inference vs inferred
        parameters of marginal inferences side by side.

        :param file: File to save plot to
        :param show: Whether to show plot
        :param axs: List of axes object, one for each covariate
        :return: Axes object
        """
        from . import Visualization

        return Visualization.plot_covariates(
            covariates=self.covariates,
            params_marginal=dict((t, inf.params_mle) for t, inf in self.marginal_inferences.items()),
            params_joint=self.params_mle,
            file=file,
            show=show,
            axs=axs
        )

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
        return None

    def set_fixed_params(self, params: Dict[str, Dict[str, float]]):
        """
        Set fixed parameters.

        :param params: Fixed parameters
        """
        # set for 'all' type
        self.marginal_inferences['all'].set_fixed_params(params)

        # expand types
        self.fixed_params = expand_fixed(params, self.types)

        # propagate to inference objects
        for t in self.types:
            self.marginal_inferences[t].set_fixed_params(dict(all=self.fixed_params[t]))
            self.joint_inferences[t].set_fixed_params(dict(all=self.fixed_params[t]))

        # propagate to optimization
        self.optimization.set_fixed_params(self.fixed_params)

        # check if the fixed parameters are compatible with the shared parameters
        self.check_no_shared_params_fixed()

    def get_L2_residual(self) -> float:
        """
        L2 residual of joint inference. We calculate the residual over
        the jointly inferred SFS for all types.
        """
        counts_mle = np.array([inf.sfs_mle.polymorphic for inf in self.joint_inferences.values()]).flatten()
        counts_sel = np.array([inf.sfs_sel.polymorphic for inf in self.joint_inferences.values()]).flatten()

        return norm(counts_mle - counts_sel, 2)
