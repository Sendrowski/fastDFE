"""
Configuration class.
"""

__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"
__date__ = "2023-02-26"

import json
import logging
from typing import List, Literal, Dict, Tuple, Callable

import numpy as np
import yaml

from .discretization import Discretization
from .io_handlers import download_if_url
from .json_handlers import CustomEncoder
from .optimization import Covariate
from .optimization import SharedParams, merge_dicts
from .parametrization import Parametrization, _from_string, _to_string
from .polydfe_utils import create_sfs_config, parse_init_file, create_init_file, models
from .spectrum import Spectra, parse_polydfe_sfs_config, Spectrum

logger = logging.getLogger('fastdfe').getChild('Config')


class Config:
    """
    Configuration class to be used for :class:`~fastdfe.base_inference.BaseInference` and
    :class:`~fastdfe.joint_inference.JointInference`.
    """

    def __init__(
            self,
            polydfe_spectra_config: str = None,
            polydfe_init_file: str = None,
            polydfe_init_file_id: int = 1,
            sfs_neut: Spectra | Spectrum = None,
            sfs_sel: Spectra | Spectrum = None,
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
            shared_params: List[SharedParams] = [],
            covariates: List[Covariate] = [],
            do_bootstrap: bool = True,
            n_bootstraps: int = 100,
            n_bootstrap_retries: int = 2,
            parallelize: bool = True,
            **kwargs
    ):
        """
        Create config object.

        :param polydfe_spectra_config: Path to polyDFE SFS config file.
        :param polydfe_init_file: Path to polyDFE init file.
        :param polydfe_init_file_id: ID of polyDFE init file.
        :param sfs_neut: Neutral SFS. Note that we require monomorphic counts to be specified in order to infer
            the mutation rate.
        :param sfs_sel: Selected SFS. Note that we require monomorphic counts to be specified in order to infer
            the mutation rate.
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
        :param model: Parametrization of the DFE.
        :param seed: Seed for the random number generator. Use ``None`` for no seed.
        :param x0: Dictionary of initial values in the form ``{type: {param: value}}``
        :param bounds: Bounds for the optimization in the form {param: (lower, upper)}
        :param scales: Scales for the optimization in the form {param: scale}
        :param loss_type: Loss function to use.
        :param opts_mle: Options for the optimization.
        :param method_mle: Method to use for optimization. See ``scipy.optimize.minimize`` for available methods.
        :param n_runs: Number of independent optimization runs out of which the best one is chosen. The first run
            will use the initial values if specified. Consider increasing this number if the optimization does not
            produce good results.
        :param fixed_params: Fixed parameters for the optimization.
        :param shared_params: Shared parameters for the optimization.
        :param covariates: Covariates for the optimization.
        :param do_bootstrap: Whether to do bootstrapping automatically.
        :param n_bootstraps: Number of bootstraps.
        :param n_bootstrap_retries: Number of optimization runs for each bootstrap sample. This parameter previously
            defined the number of retries per bootstrap sample when subsequent runs failed, but now it defines the
            total number of runs per bootstrap sample, taking the most likely one.
        :param parallelize: Whether to parallelize the optimization.
        :param kwargs: Additional keyword arguments which are ignored.
        """

        # save options
        self.data = dict(
            model=model,
            intervals_del=intervals_del,
            intervals_ben=intervals_ben,
            intervals_h=intervals_h,
            h_callback=h_callback,
            integration_mode=integration_mode,
            linearized=linearized,
            seed=seed,
            opts_mle=opts_mle,
            method_mle=method_mle,
            n_runs=n_runs,
            x0=x0,
            bounds=bounds,
            scales=scales,
            fixed_params=fixed_params,
            shared_params=shared_params,
            covariates=covariates,
            sfs_neut=Spectra.from_spectrum(sfs_neut) if isinstance(sfs_neut, Spectrum) else sfs_neut,
            sfs_sel=Spectra.from_spectrum(sfs_sel) if isinstance(sfs_sel, Spectrum) else sfs_sel,
            do_bootstrap=do_bootstrap,
            n_bootstraps=n_bootstraps,
            n_bootstrap_retries=n_bootstrap_retries,
            parallelize=parallelize,
            loss_type=loss_type
        )

        # parse spectra file if specified
        if polydfe_spectra_config is not None:
            self.parse_polydfe_sfs_config(polydfe_spectra_config)

        # parse init file if specified
        if polydfe_init_file is not None:
            self.parse_polydfe_init_file(polydfe_init_file, polydfe_init_file_id)

    def update(self, **kwargs) -> 'Config':
        """
        Update config with given data.

        :param kwargs: Data to update.
        :return: Updated config.
        """
        # convert spectrum to spectra objects
        for key in ['sfs_neut', 'sfs_sel']:
            if key in kwargs and isinstance(kwargs[key], Spectrum):
                kwargs[key] = Spectra.from_spectrum(kwargs[key])

        self.data |= kwargs

        return self

    def parse_polydfe_init_file(self, file: str, id: int = 1, type='all'):
        """
        Parse polyDFE init file.
        This will define the initial parameters and
        which ones will be held fixed during the optimization.

        :param type: Type of parameters to parse for.
        :param id: ID of the init file.
        :param file: Path to the init file.
        """
        fixed_params, x0 = parse_init_file(_from_string(self.data['model']).param_names, file, id)

        # merge with existing config
        self.data['fixed_params'] = merge_dicts(
            self.data['fixed_params'] if isinstance(self.data['fixed_params'], dict) else {},
            dict(all=fixed_params)
        )
        self.data['x0'] |= {type: x0}

    def create_polydfe_init_file(self, file: str, n: int, type: str = 'all'):
        """
        Create an init file for polyDFE.

        :param type: Type to use for the init file.
        :param n: SFS samples size.
        :param file: Path to the init file to be created.
        """
        create_init_file(
            file=file,
            fixed_params=list(self.data['fixed_params'][type].keys()) if type in self.data['fixed_params'] else [],
            x0=self.data['x0'][type] if type in self.data['x0'] else {},
            dfe_params=_from_string(self.data['model']).param_names,
            n=n
        )

    def parse_polydfe_sfs_config(self, file: str):
        """
        Parse frequency spectra and mutational target site from
        polyDFE configuration file.

        :param file: Path to the polyDFE config file.
        """
        spectra = parse_polydfe_sfs_config(file)

        # merge into data dictionary
        self.data |= dict(
            sfs_neut=Spectra.from_spectrum(spectra['sfs_neut']),
            sfs_sel=Spectra.from_spectrum(spectra['sfs_sel'])
        )

    def create_polydfe_sfs_config(self, file: str):
        """
        Create a sfs config file for polyDFE.

        :param file: Path to the sfs config file to be created.
        """
        create_sfs_config(
            file=file,
            sfs_neut=self.data['sfs_neut'].all if isinstance(self.data['sfs_neut'], Spectra) else self.data['sfs_neut'],
            sfs_sel=self.data['sfs_sel'].all if isinstance(self.data['sfs_sel'], Spectra) else self.data['sfs_sel']
        )

    def to_dict(self) -> dict:
        """
        Represent config as dictionary.

        :return: Dictionary representation of config.
        """
        return self.data

    def to_json(self) -> str:
        """
        Create JSON representation of object.

        :return: JSON string
        """
        return json.dumps(self.data, indent=4, cls=CustomEncoder)

    def to_yaml(self) -> str:
        """
        Create YAML representation of object.

        :return: YAML string
        """
        return yaml.dump(json.loads(self.to_json()), sort_keys=False)

    def to_file(self, file: str):
        """
        Save object to file.

        :param file: Path to file.
        """
        with open(file, 'w') as fh:
            fh.write(self.to_yaml())

    @staticmethod
    def from_dict(data: dict) -> 'Config':
        """
        Load config from dictionary.

        :return: Config object.
        """
        # recreate spectra objects
        for sfs in ['sfs_neut', 'sfs_sel']:
            if sfs in data and data[sfs] is not None:
                data[sfs] = Spectra.from_dict(data=data[sfs])

        # recreate SharedParams objects
        if 'shared_params' in data and data['shared_params'] is not None:
            data['shared_params'] = [SharedParams(**p) for p in data['shared_params']]

        # recreate Covariate objects
        if 'covariates' in data and data['covariates'] is not None:
            data['covariates'] = [Covariate(**c) for c in data['covariates']]

        # cast to tuple of (float, float, int)
        # useful when restoring from YAML file
        for key in ['intervals_ben', 'intervals_del', 'intervals_h']:
            if key in data:
                data[key] = (float(data[key][0]), float(data[key][1]), int(data[key][2]))

        return Config(**data)

    @staticmethod
    def from_json(data: str) -> 'Config':
        """
        Load config from JSON str.

        :param data: JSON string.
        :return: Config object.
        """
        return Config.from_dict(json.loads(data))

    @staticmethod
    def from_yaml(data: str) -> 'Config':
        """
        Load config from YAML str.

        :param data: YAML string.
        :return: Config object.
        """
        return Config.from_dict(yaml.load(data, Loader=yaml.Loader))

    @classmethod
    def from_file(cls, file: str, cache: bool = True) -> 'Config':
        """
        Load object from file.

        :param file: Path to file, possibly gzipped or a URL.
        :param cache: Whether to use the cache if available.
        :return: Config object.
        """
        with open(download_if_url(file, cache=cache, desc=f'{cls.__name__}>Downloading file'), 'r') as fh:
            return Config.from_yaml(fh.read())

    def get_polydfe_model(self) -> str:  # pragma: no cover
        """
        Get the model name in polyDFE that corresponds
        to the configured DFE parametrization.

        :return: polyDFE model name.
        """
        # get name of configured model
        name = _to_string(self.data['model'])

        # return polyDFE model name if it exists
        if name in models:
            return models[name]

        # raise error otherwise
        raise NotImplementedError(f'There is no polyDFE equivalent of {name}')
