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
    param = 'S_d'
    out = "scratch/sfs_arabidopsis.yaml"

import fastdfe as fd

genes = pd.read_csv(sfs_file, sep="\t")

genes['group'] = pd.cut(
    x=genes[col],
    bins=[0, 1, 2, 7, 10, 11, 13, 18],
    labels=['0-1', '1-2', '2-4', '4-10', '10-11', '11-13', '13-18']
)

grouped = genes.groupby('group', observed=True).sum()

cols_neut = ['Lps'] + [f'sfsS{i}' for i in range(1, 105)] + ['Ds']
cols_sel = ['Lpn'] + [f'sfsN{i}' for i in range(1, 105)] + ['Dn']

sfs_neut = fd.Spectra.from_dict(
    {'.'.join([col, k]): r.values for k, r in grouped[cols_neut].iterrows()}
).replace_dots()

sfs_sel = fd.Spectra.from_dict(
    {'.'.join([col, k]): r.values for k, r in grouped[cols_sel].iterrows()}
).replace_dots()

config = fd.Config(
    sfs_neut=sfs_neut.subsample(20),
    sfs_sel=sfs_sel.subsample(20),
    shared_params=[fd.SharedParams(params='all', types='all')],
    #covariates=[fd.Covariate(
    #    param=param,
    #    values=dict((t, int(t.split('_')[-1])) for t in sfs_neut.types)
    #)],
    n_runs=10,
    n_bootstraps=100,
    do_bootstrap=True,
    parallelize=True,
    seed=1234,
)

# Spectra.from_spectra(dict(neutral=sfs_neut.all, selected=sfs_sel.all)).plot(show=testing)

config.to_file(out)

inf = fd.JointInference.from_config(config)

inf.run()

inf.plot_discretized()

pass
