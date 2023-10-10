"""
Prepare fastdfe config for SFS of the great apes study.
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
    sfs_neutral = snakemake.input.neutral
    sfs_selected = snakemake.input.selected
    out = snakemake.output[0]
except NameError:
    # testing
    testing = True
    type = 'human'
    sfs_neutral = f'../resources/SFS/hominidae/uSFS/{type}_4fold_all_sfs.txt'
    sfs_selected = f'../resources/SFS/hominidae/uSFS/{type}_0fold_all_sfs.txt'
    out = 'scratch/config.yaml'

from fastdfe import Config, Spectra, Spectrum

sfs = Spectra.from_spectra(dict(
    neutral=Spectrum(pd.read_csv(sfs_neutral, header=None).iloc[0].tolist()),
    selected=Spectrum(pd.read_csv(sfs_selected, header=None).iloc[0].tolist())
))

config = Config(
    sfs_neut=sfs['neutral'],
    sfs_sel=sfs['selected'],
    do_bootstrap=True,
    n_bootstraps=100
)

config.to_file(out)

pass
