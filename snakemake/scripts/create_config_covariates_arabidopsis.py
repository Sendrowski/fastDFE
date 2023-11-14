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
    n_bins = snakemake.params.n_bins
    subsample_size = snakemake.params.subsample_size
    param = snakemake.params.param
    out = snakemake.output[0]
except NameError:
    # testing
    testing = True
    sfs_file = "results/sfs_covariates/arabidopsis.csv"
    col = 'mean.rsa'
    n_bins = 15
    subsample_size = 20
    param = 'S_d'
    out = "scratch/sfs_arabidopsis.yaml"

import fastdfe as fd

# read SFS stratified by gene
genes = pd.read_csv(sfs_file, sep="\t")

genes['count'] = 1

# create bins
genes['group'] = pd.qcut(
    x=genes[col],
    q=n_bins,
    labels=[f"bin_{i}" for i in range(n_bins)],
)

# group by bins
grouped = genes.groupby('group', observed=True)

# extract SFS
cols_neut = ['Lps'] + [f'sfsS{i}' for i in range(1, 105)] + ['Ds']
cols_sel = ['Lpn'] + [f'sfsN{i}' for i in range(1, 105)] + ['Dn']

# create spectra objects
sfs_neut = fd.Spectra.from_dict(
    {k: r.values for k, r in grouped.sum()[cols_neut].iterrows()}
)

sfs_sel = fd.Spectra.from_dict(
    {k: r.values for k, r in grouped.sum()[cols_sel].iterrows()}
)

# determine average value of covariate in each bin
cov = grouped[col].mean()

# shuffle covariate values for testing
# cov = dict(zip(cov.index, np.random.permutation(cov.values)))

# create covariate object
covariates = fd.Covariate(
    param=param,
    values=dict((t, cov[t]) for t in sfs_neut.types)
)

# create config object
config = fd.Config(
    sfs_neut=sfs_neut.subsample(subsample_size),
    sfs_sel=sfs_sel.subsample(subsample_size),
    shared_params=[fd.SharedParams(params='all', types='all')],
    covariates=[covariates],
    n_runs=20,
    n_bootstraps=100,
    do_bootstrap=True,
    parallelize=True
)

# plot SFS
if testing:
    (config.data['sfs_neut'].prefix('neutral') + config.data['sfs_sel'].prefix('selected')).plot(
        show=testing,
        use_subplots=True
    )

# save config
config.to_file(out)

pass
