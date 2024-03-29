"""
Infer the joint DFE from the SFS using fastDFE.
"""

__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"
__date__ = "2023-02-26"

try:
    import sys

    # necessary to import fastdfe locally
    sys.path.append('..')
    testing = False
    config_file = snakemake.input[0]
    kwargs_legend = snakemake.params.get('legend', {})
    figsize = snakemake.params.get('figsize', (20, 10))
    out_summary = snakemake.output.summary
    out_serialized = snakemake.output.serialized
    out_dfe = snakemake.output.get('dfe', None)
    out_covariates = snakemake.output.get('covariates', None)
    out_spectra = snakemake.output.get('spectra', None)
    out_params = snakemake.output.get('params', None)
except NameError:
    # testing
    testing = True
    # config_file = 'results/configs/example_1_C_full_anc/config.yaml'
    # config_file = 'results/configs/example_1_C_deleterious_anc_bootstrapped_100/config.yaml'
    # config_file = 'results/configs/pendula_C_full_anc_bootstrapped_100/config.yaml'
    config_file = 'results/fastdfe/arabidopsis/cov/S_d.mean.rsa.10.20.yaml'
    kwargs_legend = {}
    figsize = (20, 10)
    out_summary = "scratch/summary.json"
    out_serialized = "scratch/serialized.json"
    out_dfe = "scratch/dfe.png"
    out_covariates = "scratch/covariates.png"
    out_spectra = "scratch/spectra.png"
    out_params = "scratch/params.png"

import fastdfe as fd

# create from config
inf = fd.JointInference.from_config_file(config_file)

# perform inference
inf.run()

# save object in serialized form
inf.to_file(out_serialized)

# save summary
inf.get_summary().to_file(out_summary)

p = f"p: {inf.perform_lrt_covariates():.2e}"
c0 = f"c0: ({inf.bootstraps.mean()[inf.types[0] + '.c0']:.2e}, {inf.bootstraps.var()[inf.types[0] + '.c0']:.2e})"

# plot results
inf.plot_inferred_parameters(
    file=out_params,
    show=testing,
    title=f'inferred parameters, {p}, {c0}',
    kwargs_legend=kwargs_legend
)

inf.plot_sfs_comparison(
    use_subplots=True,
    file=out_spectra,
    show=testing,
    title=f'SFS comparison, {p}, {c0}',
    kwargs_legend=kwargs_legend
)

inf.plot_discretized(
    file=out_dfe,
    show=testing,
    title=f'DFE comparison, {p}, {c0}',
    kwargs_legend=kwargs_legend
)

inf.plot_covariate(
    file=out_covariates,
    show=testing,
    title=f'covariate comparison, {p}, {c0}'
)

pass
