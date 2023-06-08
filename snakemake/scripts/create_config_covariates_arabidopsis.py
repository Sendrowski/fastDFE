"""
Create config file for joint inference of Arabidopsis thaliana.
"""

__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"
__date__ = "2023-06-06"

import pandas as pd

try:
    import sys

    # necessary to import dfe module
    sys.path.append('..')
    testing = False
    col = snakemake.params.col
    param = snakemake.params.param
    sfs_file = snakemake.input[0]
    out = snakemake.output[0]
except NameError:
    # testing
    testing = True
    sfs_file = "results/sfs_covariates/arabidopsis.csv"
    col = 'age.rsa'
    param = 'b'
    out = "scratch/joint_inference.yaml"

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
    n_runs=100,
    n_bootstraps=100,
    do_bootstrap=True,
    parallelize=True,
    #model=DiscreteFractionalParametrization()
)

config.to_file(out)

Spectra.from_spectra(dict(neutral=sfs_neut.all, selected=sfs_sel.all)).plot()

inf = JointInference.from_config(config)

# inf.plot_likelihoods()

# inf.plot_sfs_comparison(sfs_types=['neutral', 'selected'], use_subplots=True)

inf.plot_discretized(kwargs_legend=dict(bbox_to_anchor=(1, 0.5), ncol=3, fontsize=8))

inf.perform_lrt_covariates()

pass
