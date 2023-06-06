"""
Visualize a DFE inference.
"""

__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"
__date__ = "2023-02-26"

try:
    import sys

    # necessary to import dfe module
    sys.path.append('..')
    testing = False
    serialized = snakemake.input[0]
    out_dfe = snakemake.output.get('dfe', None)
    out_spectra = snakemake.output.get('spectra', None)
    out_params = snakemake.output.get('params', None)
except NameError:
    # testing
    testing = True
    # config_file = 'results/configs/example_1_C_full_anc/config.yaml'
    # config_file = 'results/configs/example_1_C_deleterious_anc_bootstrapped_100/config.yaml'
    # config_file = 'results/configs/pendula_C_full_anc_bootstrapped_100/config.yaml'
    serialized = 'results/fastdfe/hominidae/cov/S_d.serialized.json'
    out_dfe = "scratch/dfe.png"
    out_spectra = "scratch/spectra.png"
    out_params = "scratch/params.png"

from fastdfe import JointInference

# create from config
inf = JointInference.from_file(serialized)

# plot results
inf.plot_inferred_parameters(file=out_params, show=testing)
inf.plot_sfs_comparison(file=out_spectra, show=testing)
inf.plot_discretized(file=out_dfe, show=testing)

pass
