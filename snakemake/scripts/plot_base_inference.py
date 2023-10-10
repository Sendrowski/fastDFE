"""
Visualize DFE inference results.
"""

__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"
__date__ = "2023-03-02"

try:
    import sys

    # necessary to import fastdfe locally
    sys.path.append('..')

    testing = False
    input = snakemake.input[0]
    out_dfe_discretized = snakemake.output.get('dfe_discretized', None)
    out_dfe_continuous = snakemake.output.get('dfe_continuous', None)
    out_sfs_comparison = snakemake.output.get('sfs_comparison', None)
    out_mle_params = snakemake.output.get('mle_params', None)
    out_bucket_sizes = snakemake.output.get('bucket_sizes', None)
except NameError:
    # testing
    testing = True
    input = "results/fastdfe/example_1_C_deleterious_anc_bootstrapped_100/serialized.json"
    out_dfe_discretized = "scratch/dfe_discretized.png"
    out_dfe_continuous = "scratch/dfe_continuous.png"
    out_sfs_comparison = "scratch/sfs_comparison.png"
    out_mle_params = "scratch/mle_params.png"
    out_bucket_sizes = "scratch/bucket_sizes.png"

import fastdfe as fd

inference = fd.BaseInference.from_file(input)

if out_dfe_continuous is not None:
    inference.plot_continuous(file=out_dfe_continuous, show=testing)

if out_dfe_discretized is not None:
    inference.plot_discretized(file=out_dfe_discretized, show=testing)

if out_mle_params is not None:
    inference.plot_inferred_parameters(file=out_mle_params, show=testing)

if out_sfs_comparison is not None:
    inference.plot_sfs_comparison(file=out_sfs_comparison, show=testing)

if out_bucket_sizes is not None:
    inference.plot_bucket_sizes(file=out_bucket_sizes, show=testing)

pass
