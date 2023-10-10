"""
Create config file for joint inference of Arabidopsis thaliana.
"""

__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"
__date__ = "2023-06-06"

import pandas as pd

try:
    import sys

    # necessary to import fastdfe locally
    sys.path.append('..')
    testing = False
    sfs_file = snakemake.input[0]
    col = snakemake.params.col
    param = snakemake.params.param
    out = snakemake.output[0]
except NameError:
    # testing
    testing = True
    sfs_file = "results/sfs_covariates/arabidopsis.csv"
    col = 'age.rsa'
    param = 'b'
    out = "scratch/sfs_arabidopsis.yaml"

from fastdfe import Config, Spectra, JointInference, SharedParams, Covariate, DiscreteFractionalParametrization

genes = pd.read_csv(sfs_file, sep="\t")

grouped = genes.groupby(col).sum()

cols_neut = ['Lps'] + [f'sfsS{i}' for i in range(1, 105)] + ['Ds']
cols_sel = ['Lpn'] + [f'sfsN{i}' for i in range(1, 105)] + ['Dn']

sfs_neut = Spectra.from_dict(
    {'.'.join([col, str(int(i))]): r.values for i, r in grouped[cols_neut].iterrows()}
).replace_dots()

sfs_sel = Spectra.from_dict(
    {'.'.join([col, str(int(i))]): r.values for i, r in grouped[cols_sel].iterrows()}
).replace_dots()

config = Config(
    sfs_neut=sfs_neut,
    sfs_sel=sfs_sel,
    shared_params=[SharedParams(params='all', types='all')],
    covariates=[Covariate(
        param=param,
        values=dict((t, int(t.split('_')[-1])) for t in sfs_neut.types)
    )],
    n_runs=1000,
    n_bootstraps=200,
    do_bootstrap=True,
    parallelize=True
)

Spectra.from_spectra(dict(neutral=sfs_neut.all, selected=sfs_sel.all)).plot(show=testing)

config.to_file(out)

pass
