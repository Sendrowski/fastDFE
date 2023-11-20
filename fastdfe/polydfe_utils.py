"""
Utilities for parsing polyDFE files.
"""

__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"
__date__ = "2023-02-26"

import logging
import os
from typing import TextIO, List

import pandas as pd

from .spectrum import Spectrum

# get logger
logger = logging.getLogger('fastdfe').getChild('polydfe')

#: model mapping
models = dict(
    DisplacedGammaParametrization='A',
    GammaDiscreteParametrization='B',
    GammaExpParametrization='C',
    DiscreteParametrization='D'
)

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
    cols = columns + list(dfe_params)

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
    for i, col in enumerate(cols):
        params[col] = data[2 * i + 2]
        params_mode[col] = int(data[2 * i + 1])

    # assign nuisance parameters
    params['r'] = list(data[2 * len(cols) + 1:])
    params_mode['r'] = int(data[2 * len(cols) + 1])

    # determine fixed parameters and initial conditions
    fixed_params = {}
    x0 = {}
    for param in params_mode.keys():
        if param not in ignore:
            # here we don't distinguish between shared and independent parameters
            if params_mode[param] == ParamMode.fixed:
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
    cols = columns + list(dfe_params) + ['r']

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
    for col in cols:
        if col in x0:
            params += [x0[col]]
        else:
            params += [0]

    # create list of parameter modes
    params_mode = []
    for col in cols:
        if col in fixed_params:
            params_mode += [ParamMode.fixed]
        else:
            params_mode += [ParamMode.independent]

    # concatenate parameters and their modes
    params_with_mode = [str(a) + ' ' + str(b) for a, b in zip(list(params_mode), params)]

    # prepend an id value and join
    line = '\t'.join([str(id)] + params_with_mode)

    # write line to file
    with open(file, 'w') as fh:
        fh.write(line)
