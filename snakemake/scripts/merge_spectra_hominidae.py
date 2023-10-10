"""
Merge spectra from the great apes study.
"""

__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"
__date__ = "2023-05-31"

import pandas as pd

try:
    import sys

    # necessary to import fastdfe locally
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

import fastdfe as fd


def get_sfs(file: str) -> fd.Spectrum:
    """
    Get SFS from file.

    :param file: The file to read the SFS from.
    :return: The SFS.
    """
    return fd.Spectrum(pd.read_csv(file, header=None).iloc[0].tolist())


neutral = fd.Spectra.from_spectra(dict((t, get_sfs(sfs_neutral[i])) for i, t in enumerate(types)))
selected = fd.Spectra.from_spectra(dict((t, get_sfs(sfs_selected[i])) for i, t in enumerate(types)))

spectra = neutral.prefix('neutral') + selected.prefix('selected')

pass
