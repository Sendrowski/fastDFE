"""
Check for covariates in the great ape dataset.
"""

__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"
__date__ = "2023-05-31"

import pandas as pd

try:
    import sys

    # necessary to import dfe module
    sys.path.append('..')
    testing = False
    types = snakemake.params.types
    param = snakemake.params.param
    sfs_neutral = snakemake.input.neutral
    sfs_selected = snakemake.input.selected
    out = snakemake.output[0]
except NameError:
    # testing
    testing = True
    types = [
        'bonobo',
        'bornean_orang',
        'central_chimp',
        'eastern_chimp',
        'human',
        'NC_chimp',
        'sumatran_orang',
        'western_chimp',
        'western_lowland_gorilla'
    ]
    param = 'S_d'
    sfs_neutral = [f'../resources/SFS/hominidae/uSFS/{t}_4fold_all_sfs.txt' for t in types]
    sfs_selected = [f'../resources/SFS/hominidae/uSFS/{t}_0fold_all_sfs.txt' for t in types]
    out = "scratch/joint_inference.yaml"

from fastdfe import Config, Spectra, Spectrum, Covariate, SharedParams


def get_sfs(file: str) -> Spectrum:
    """
    Get SFS from file.

    :param file: The file to read the SFS from.
    :return: The SFS.
    """
    return Spectrum(pd.read_csv(file, header=None).iloc[0].tolist())


config = Config(
    sfs_neut=Spectra.from_spectra(dict((t, get_sfs(sfs_neutral[i])) for i, t in enumerate(types))),
    sfs_sel=Spectra.from_spectra(dict((t, get_sfs(sfs_selected[i])) for i, t in enumerate(types))),
    do_bootstrap=True,
    n_bootstraps=100,
    n_runs=100,
    parallelize=True,
    # fixed_params=dict(all=dict(eps=0, p_b=0, S_b=1)),
    shared_params=[SharedParams(params='all', types='all')],
    covariates=[Covariate(
        param=param,
        values=dict((t, get_sfs(sfs_neutral[i]).theta / (4 * 1e-9)) for i, t in enumerate(types))
    )],
)

config.to_file(out)
