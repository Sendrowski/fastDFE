"""
Infer the DFE from spectra generated by SLiM.
"""

__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"
__date__ = "2024-02-26"

import numpy as np

try:
    import sys

    # necessary to import fastdfe locally
    sys.path.append('..')

    testing = False
    sfs_file = snakemake.input[0]
    out_summary = snakemake.output.summary
    out_serialized = snakemake.output.serialized
    out_dfe = snakemake.output.get('dfe', None)
    out_model_fit = snakemake.output.get('model_fit', None)
    out_spectra = snakemake.output.get('spectra', None)
    out_params = snakemake.output.get('params', None)
except NameError:
    # testing
    testing = True
    sfs_file = 'snakemake/results/slim/n_replicate=1/n_chunks=40/g=1e4/L=1e7/mu=1e-8/r=1e-7/N=1e3/s_b=1e-9/b=2/s_d=1e0/p_b=0/n=20/unfolded/sfs.csv'
    out_summary = "scratch/summary.json"
    out_serialized = "scratch/serialized.json"
    out_dfe = "scratch/dfe.png"
    out_model_fit = "scratch/model_fit.png"
    out_spectra = "scratch/spectra.png"
    out_params = "scratch/params.png"

import fastdfe as fd

s = fd.Spectra.from_file(sfs_file)

# create from config
inf = fd.BaseInference(
    sfs_neut=s['neutral'],
    sfs_sel=s['selected'],
    do_bootstrap=True,
    parallelize=True,
    fixed_params=dict(all=dict(eps=0))
)

# perform inference
inf.run()

# save object in serialized form
inf.to_file(out_serialized)

# save summary
inf.get_summary().to_file(out_summary)

params = {k: v for k, v in inf.bootstraps.median().to_dict().items() if k not in ['alpha']}

# plot results
inf.plot_inferred_parameters(file=out_params, show=testing)
inf.plot_sfs_comparison(file=out_model_fit, show=testing)
inf.plot_discretized(
    file=out_dfe,
    show=testing,
    title="Inferred DFE\n" + ", ".join([f"${k}$={round(v, 2)}" for k, v in params.items()]),
    intervals=[-np.inf, -100, -10, -1, 0, 1, np.inf]
)
s.plot(file=out_spectra, show=testing)

pass