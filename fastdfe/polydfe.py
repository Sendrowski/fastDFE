"""
Infer the DFE from the SFS using discretized DFE.
"""

__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"
__date__ = "2023-02-26"

import copy
import json
import logging
import os
import subprocess
import tempfile
import time
from typing import Callable, List, Optional, TextIO, Literal

import numpy as np
import pandas as pd
import rpy2.robjects as ro
from matplotlib import pyplot as plt
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import ListVector
from typing_extensions import Self

from . import config
from .abstract_inference import AbstractInference
from .parametrization import from_string, Parametrization, DiscreteParametrization
from .spectrum import Spectrum
from .visualization import Visualization

# get logger
logger = logging.getLogger('fastdfe')

# model mapping
models = dict(
    DisplacedGammaParametrization='A',
    GammaDiscreteParametrization='B',
    GammaExpParametrization='C',
    DiscreteParametrization='D'
)


def create_sfs_config(
        file: str,
        sfs_neut: Spectrum,
        sfs_sel: Spectrum
):
    """
    Create a sfs config file for polyDFE.

    :param sfs_sel: Selected sfs
    :param sfs_neut: Neutral sfs
    :param file: Path to config file
    """

    def write_line(f: TextIO, sfs: Spectrum, sep: str = '\t'):
        """
        Concatenate given array and write to stream.

        :param sfs: SFS
        :param f: File stream
        :param sep: separator
        """
        # SFS and number of mutational target sites
        # we ignore divergence data here
        data = list(sfs.polymorphic) + [sfs.n_sites]

        f.write(sep.join(map(str, data)) + os.linesep)

    # number of spectra per time
    m_neut = 1
    m_sel = 1
    with open(file, 'w') as fh:
        fh.write(' '.join(map(str, [m_neut, m_sel, sfs_sel.n])) + os.linesep)

        write_line(fh, sfs_neut)
        write_line(fh, sfs_sel)


def parse_init_file(
        dfe_params: list,
        file: str,
        id: int = 1,
        ignore: list = ['eps_deprecated', 'r']
) -> (dict, dict):
    """
    Parse polyDFE init file.
    This will define the initial parameters and
    which ones will be held fixed during the optimization.
    We do not support fixing some of the polyDFE's parameters
    like theta and the nuisance parameters. Some other parameters
    are not supported altogether.

    :param dfe_params: List of dfe parameters
    :param file: Path to init file
    :param id: Init id
    :param ignore: List of parameters to ignore
    :return Dictionary of fixed parameters and dictionary of initial values
    """
    logger.info('Parsing init file.')

    # columns except for nuisance parameters
    columns = PolyDFE.columns + list(dfe_params)

    # load init file into dataframe
    df = pd.read_csv(file, delimiter=r"\s+", comment='#', header=None, index_col=0)

    # check if id is present
    if id not in df.index:
        raise Exception(f'Id {id} not found in init file. Possible ids: {str(df.index.to_list())}.')

    # extract row with specified id
    data = df.loc[id]

    # Retrieve parameters and their modes.
    # Note that the parameter order is important
    # when recreating the init file.
    params, params_mode = {}, {}
    for i, col in enumerate(columns):
        params[col] = data[2 * i + 2]
        params_mode[col] = int(data[2 * i + 1])

    # assign nuisance parameters
    params['r'] = list(data[2 * len(columns) + 1:])
    params_mode['r'] = int(data[2 * len(columns) + 1])

    # determine fixed parameters and initial conditions
    fixed_params = {}
    x0 = {}
    for param in params_mode.keys():
        if param not in ignore:
            # here we don't distinguish between shared and independent parameters
            if params_mode[param] == PolyDFE.ParamMode.fixed:
                fixed_params[param] = params[param]

            x0[param] = params[param]

    # issue notice
    logger.info(f"Found initial params: {x0} and fixed params: {fixed_params} when parsing init file.")

    return fixed_params, x0


def create_init_file(
        file: str,
        fixed_params: List,
        x0: dict,
        dfe_params: List,
        n: int,
        id: int = 1
):
    """
    Create an init file for polyDFE.

    :param id: Init id
    :param n: SFS sample size
    :param dfe_params: List of dfe parameters
    :param x0: Dictionary of initial values
    :param fixed_params: List of fixed parameters
    :param file: File path to write init file to
    """
    columns = PolyDFE.columns + list(dfe_params) + ['r']

    # copy dict to avoid modification
    x0 = x0.copy()

    if 'r' in x0:
        # assign existing nuisance parameters
        x0['r'] = ' '.join([str(r) for r in x0['r']])
    else:
        # make up initial nuisance parameters
        x0['r'] = ' '.join(['1'] * (n - 1))

    # create list of initial parameters
    params = []
    for col in columns:
        if col in x0:
            params += [x0[col]]
        else:
            params += [0]

    # create list of parameter modes
    params_mode = []
    for col in columns:
        if col in fixed_params:
            params_mode += [PolyDFE.ParamMode.fixed]
        else:
            params_mode += [PolyDFE.ParamMode.independent]

    # concatenate parameters and their modes
    params_with_mode = [str(a) + ' ' + str(b) for a, b in zip(list(params_mode), params)]

    # prepend an id value and join
    line = '\t'.join([str(id)] + params_with_mode)

    # write line to file
    with open(file, 'w') as fh:
        fh.write(line)


class PolyDFEResult:
    # default postprocessing source
    default_postprocessing_source = 'resources/polydfe/postprocessing/script.R'

    # mapping of polyDFE params to new name
    param_mapping = dict(
        eps_an='eps'
    )

    def __init__(
            self,
            output_file: str,
            additional_data: dict = {},
            postprocessing_source: str = default_postprocessing_source
    ):
        """
        Create summary object.

        :param output_file: Path to polyDFE output file
        :param additional_data: Additional data to add to summary
        :param postprocessing_source: Path to polyDFE postprocessing script
        """
        self.output_file = output_file
        self.postprocessing_source = postprocessing_source

        self.data = self.parse_output() | additional_data

    @staticmethod
    def map_name(name: str) -> str:
        """
        Map polyDFE parameter name to new name.

        :param name: polyDFE parameter name
        :return: New name
        """
        if name in PolyDFEResult.param_mapping:
            return PolyDFEResult.param_mapping[name]

        return name

    def parse_output(self) -> dict:
        """
        Parse output from polyDFE output file.

        :return: Dictionary of parsed data
        """
        # use the polyDFE R postprocessing script to parse the output file
        output = self.get_postprocessing_wrapper().parseOutput(self.output_file)[0]

        # convert to JSON
        rj = importr('RJSONIO')
        json_str = rj.toJSON(output)[0]

        # load dict from JSON string
        data = json.loads(json_str)

        return dict(
            output_file=data['input'],
            model=data['model'],
            likelihood=data['lk'],
            criteria=data['criteria'],
            params_mle=dict(
                # non-nuisance parameters
                # map polyDFE param name to new name
                (PolyDFEResult.map_name(k), v) for k, v in data['values'][0].items() if not k.startswith('r ')
            ) | dict(
                # nuisance parameters
                r=[data['values'][0][f"r {i}"] for i in range(2, data['n'])]
            ),
            n=data['n'],
            expec=data['expec'][0],
            alpha=self.get_alpha(output)
        )

    def get_alpha(self, parsed_output: ListVector) -> float:
        """
        Get alpha, the proportion of beneficial non-synonymous substitutions.

        :return: Parsed output from polyDFE
        """
        return self.get_postprocessing_wrapper().estimateAlpha(parsed_output)[0]

    def get_postprocessing_wrapper(self) -> ro.R:
        """
        Get the wrapped polyDFE postprocessing source.

        :return: R object
        """
        ps = ro.r
        ps.source(self.postprocessing_source)

        return ps

    def to_json(self) -> str:
        """
        Convert object to JSON.

        :return: JSON string
        """
        return json.dumps(self.data, indent=4)

    def to_file(self, file: str):
        """
        Save object to file.

        :return: File path to save to
        """
        with open(file, 'w') as fh:
            fh.write(self.to_json())


class PolyDFE(AbstractInference):
    """
    polyDFE wrapper.
    Currently only model C is fully implemented.
    """

    # leading column names in init file
    columns = [
        'eps',
        'eps_deprecated',
        'lambda',
        'theta_bar',
        'a'
    ]

    class ParamMode:
        """
        Parameter modes in polyDFE input files.
        """
        independent = 0
        fixed = 1
        shared = 2

    def __init__(self, config: 'config.Config'):
        """
        Create polyDFE wrapper.

        :param config: Config object
        """
        super().__init__()

        self.config = config

        # polyDFE output file
        self.output_file: Optional[str] = None

        # the total execution time in seconds
        self.execution_time: float = 0

        # polyDFE summary
        self.summary: Optional[PolyDFEResult] = None

        # bootstrap samples
        self.bootstraps: Optional[pd.DataFrame] = None

    @staticmethod
    def map_polydfe_model(model: str) -> str:
        """
        Map polyDFE model name to native equivalent.

        :param model: polyDFE model name
        :return: Native model name
        """
        return {v: k for k, v in models.items()}[model]

    @classmethod
    def from_config(cls, config: 'Config') -> Self:
        """
        Load from config object.

        :param config: Config object
        """
        return cls(config)

    @classmethod
    def from_config_file(cls, file: str) -> Self:
        """
        Load from config object.
        """
        from fastdfe import Config

        return cls.from_config(Config.from_file(file))

    def run(
            self,
            output_file: str,
            bin: str = 'polydfe',
            wd: str = None,
            execute: Callable = None,
            postprocessing_source: str = PolyDFEResult.default_postprocessing_source
    ) -> PolyDFEResult:
        """
        Run polyDFE.

        :param postprocessing_source: polyDFE postprocessing source
        :param execute: Function for executing shell commands
        :param wd: Working directory
        :param bin: polyDFE binary
        :param output_file: Output file
        :return: polyDFE summary
        """
        start_time = time.time()

        def shell(command: str):
            """
            Execute shell command.

            :param command: Command string
            """
            return subprocess.run(command, check=True, cwd=wd, shell=True)

        # define default function for executing command
        if execute is None:
            execute = shell

        # save the spectra and init file, so they can be reviewed if necessary
        # use tempfile to generate the file name.
        with open(tempfile.NamedTemporaryFile().name, 'w') as spectra_file:
            with open(tempfile.NamedTemporaryFile().name, 'w') as init_file:
                # save files
                self.config.create_polydfe_sfs_config(spectra_file.name)
                self.config.create_polydfe_init_file(init_file.name, n=self.config.data['sfs_neut'].n)

                # add number of fragment if model is DiscreteParametrization
                model = from_string(self.config.data['model'])
                k = str(model.k - 1) + ' ' if isinstance(model, DiscreteParametrization) else ''

                # construct command string
                command = (f"{bin} "
                           f"-d {spectra_file.name} "
                           f"-m {self.config.get_polydfe_model()} {k}"
                           f"-i {init_file.name} 1 "
                           f"-v 1 > {output_file}")

                # log command signature
                logger.info(f"Running: {command}")

                # execute command
                execute(command)

        # add execution time
        self.execution_time += time.time() - start_time

        # create summary from output file
        self.summary = PolyDFEResult(
            output_file=output_file,
            additional_data=dict(execution_time=self.execution_time),
            postprocessing_source=postprocessing_source
        )

        return self.summary

    def create_bootstrap(self) -> 'config.Config':
        """
        Create a bootstrap sample using polyDFE's resampling.

        :return: Config object
        """
        if self.summary is None:
            raise Exception('PolyDFE needs to be run before creating bootstrap samples.')

        # postprocessing wrapper
        ps = self.summary.get_postprocessing_wrapper()

        # create temporary SFS config file
        with tempfile.NamedTemporaryFile() as tmp:
            create_sfs_config(
                file=tmp.name,
                sfs_neut=self.config.data['sfs_neut']['all'],
                sfs_sel=self.config.data['sfs_sel']['all']
            )

            # create bootstrap SFS config
            ps.bootstrapData(tmp.name, rep=1)

            # file name of resampled SFS config
            sfs_config_file = tmp.name + '_1'

            # use exiting config as template and load init and SFS file
            config = copy.deepcopy(self.config)
            config.parse_polydfe_sfs_config(sfs_config_file)
            config.data['x0'] = dict(all=self.summary.data['params_mle'])

        return config

    def add_bootstraps(self, bootstraps: 'List[PolyDFE]'):
        """
        Add bootstraps samples.

        :param bootstraps: List of bootstrap samples
        """
        # load MLE params into dataframe
        self.bootstraps = pd.DataFrame([bs.get_bootstrap_params() for bs in bootstraps])

        # drop nuisance parameters and eps_cont
        self.bootstraps.drop(columns=['r', 'eps_cont'], inplace=True)

        # update execution time
        self.summary.data['execution_time'] += np.sum([bs.summary.data['execution_time'] for bs in bootstraps])

    def plot_all(self, show: bool = True):
        """
        Plot everything.
        """
        self.plot_inferred_parameters(show=show)
        self.plot_discretized(show=show)

    def plot_inferred_parameters(
            self,
            file: str = None,
            show: bool = True,
            scale: Literal['lin', 'log'] = 'log',
            ax: plt.Axes = None
    ) -> plt.Axes:
        """
        Visualize the inferred parameters and their confidence intervals.

        :param scale: Scale of the y-axis
        :param file: File to save the plot to
        :param show: Show the plot
        :param ax: Axes object
        :return: Axes object
        """
        param_names = self.get_bootstrap_param_names()

        return Visualization.plot_inferred_parameters(
            params=[self.get_bootstrap_params()],
            bootstraps=[self.bootstraps[param_names]] if self.bootstraps is not None else [None],
            scale=scale,
            file=file,
            show=show,
            ax=ax
        )

    def get_bootstrap_param_names(self) -> List[str]:
        """
        Parameter names for parameters included in bootstraps.

        :return: List of parameter names
        """
        return from_string(self.config.data['model']).param_names + ['eps', 'alpha']

    def get_bootstrap_params(self) -> dict:
        """
        Get the parameters to be included in the bootstraps.

        :return: Dict of parameters
        """
        params = self.summary.data['params_mle'] | dict(alpha=self.summary.data['alpha'])

        # filter params
        return dict((k, params[k]) for k in self.get_bootstrap_param_names())

    @property
    def params_mle(self) -> dict:
        """
        Get the maximum likelihood estimate of the parameters.

        :return: Dict of parameters
        """
        return self.summary.data['params_mle']

    @params_mle.setter
    def params_mle(self, value):
        """
        Set the maximum likelihood estimate of the parameters.

        :param value: Dict of parameters
        """
        pass

    @property
    def model(self) -> Parametrization:
        """
        Get the DFE parametrization.

        :return: Parametrization
        """

        return from_string(self.config.data['model'])

    @model.setter
    def model(self, value):
        """
        Set the DFE parametrization.

        :param value: Parametrization
        """
        pass
