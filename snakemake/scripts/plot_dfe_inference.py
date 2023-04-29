"""
Visualize DFE inference results.
"""

__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"
__date__ = "2023-03-02"

try:
    import sys

    # necessary to import dfe module
    sys.path.append('..')

    testing = False
    input = snakemake.input[0]
    out_dfe_discretized = snakemake.output.dfe_discretized
    out_dfe_log = snakemake.output.dfe_log
    out_sfs_comparison = snakemake.output.sfs_comparison
    out_mle_params = snakemake.output.mle_params
    out_bucket_sizes = snakemake.output.bucket_sizes
except NameError:
    # testing
    testing = True
    input = "results/fastdfe/example_1_C_deleterious_anc_bootstrapped_100/serialized.json"
    out_dfe_discretized = "scratch/dfe_discretized.png"
    out_dfe_log = "scratch/dfe_log.png"
    out_sfs_comparison = "scratch/sfs_comparison.png"
    out_mle_params = "scratch/mle_params.png"
    out_bucket_sizes = "scratch/bucket_sizes.png"

import fastdfe

inference = fastdfe.BaseInference.from_file(input)

inference.plot_continuous(file=out_dfe_log, show=testing)
inference.plot_discretized(file=out_dfe_discretized, show=testing)
inference.plot_inferred_parameters(file=out_mle_params, show=testing)
inference.plot_bucket_sizes(file=out_bucket_sizes, show=testing)
inference.plot_sfs_comparison(file=out_sfs_comparison, show=testing)
inference.plot_interval_density(show=testing)

pass
