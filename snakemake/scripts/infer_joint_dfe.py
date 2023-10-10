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
    out_summary = snakemake.output.summary
    out_serialized = snakemake.output.serialized
    out_dfe = snakemake.output.get('dfe', None)
    out_spectra = snakemake.output.get('spectra', None)
    out_params = snakemake.output.get('params', None)
except NameError:
    # testing
    testing = True
    # config_file = 'results/configs/example_1_C_full_anc/config.yaml'
    # config_file = 'results/configs/example_1_C_deleterious_anc_bootstrapped_100/config.yaml'
    # config_file = 'results/configs/pendula_C_full_anc_bootstrapped_100/config.yaml'
    config_file = 'scratch/bonobo.yaml'
    out_summary = "scratch/summary.json"
    out_serialized = "scratch/serialized.json"
    out_dfe = "scratch/dfe.png"
    out_spectra = "scratch/spectra.png"
    out_params = "scratch/params.png"

from fastdfe import JointInference

# create from config
inference = JointInference.from_config_file(config_file)

# perform inference
inference.run()

# save object in serialized form
inference.to_file(out_serialized)

# save summary
inference.get_summary().to_file(out_summary)

# plot results
inference.plot_inferred_parameters(file=out_params, show=testing)
inference.plot_sfs_comparison(file=out_spectra, show=testing)
inference.plot_discretized(file=out_dfe, show=testing)

pass
