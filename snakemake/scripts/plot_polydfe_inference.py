"""
Visualize polyDFE inference results.
"""

__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"
__date__ = "2023-03-11"

try:
    import sys

    # necessary to import dfe module
    sys.path.append('..')

    testing = False
    input = snakemake.input[0]
    out_dfe_discretized = snakemake.output.dfe_discretized
    out_mle_params = snakemake.output.mle_params
except NameError:
    # testing
    testing = True
    # input = "scripts/polydfe/pendula_C_full_anc/serialized.json"
    input = "results/fastdfe/pendula_C_full_bootstrapped_100/serialized.json"
    out_dfe_discretized = "scratch/dfe_discretized.png"
    out_mle_params = "scratch/mle_params.png"

from fastdfe.polydfe import PolyDFE

inference = PolyDFE.from_file(input)

inference.plot_inferred_parameters(out_mle_params, show=testing)
inference.plot_discretized(out_dfe_discretized, show=testing)

pass
