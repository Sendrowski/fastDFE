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
    sfs_comparison_detailed = snakemake.output.get('sfs_comparison_detailed', None)
    out_sfs_input = snakemake.output.get('sfs_input', None)
    out_mle_params = snakemake.output.get('mle_params', None)
    out_bucket_sizes = snakemake.output.get('bucket_sizes', None)
except NameError:
    # testing
    testing = True
    input = "results/fastdfe/example_3_C_full_anc_bootstrapped_100/serialized.json"
    out_dfe_discretized = "scratch/dfe_discretized.png"
    out_dfe_continuous = "scratch/dfe_continuous.png"
    out_sfs_comparison = "scratch/sfs_comparison.png"
    sfs_comparison_detailed = "scratch/sfs_comparison_detailed.png"
    out_sfs_input = "scratch/sfs_input.png"
    out_mle_params = "scratch/mle_params.png"
    out_bucket_sizes = "scratch/bucket_sizes.png"

import fastdfe as fd

inf = fd.BaseInference.from_file(input)

if out_dfe_continuous is not None:
    inf.plot_continuous(file=out_dfe_continuous, show=testing)

if out_dfe_discretized is not None:
    inf.plot_discretized(file=out_dfe_discretized, show=testing)

if out_mle_params is not None:
    inf.plot_inferred_parameters(file=out_mle_params, show=testing)

if out_sfs_comparison is not None:
    inf.plot_sfs_comparison(
        file=out_sfs_comparison,
        show=testing,
        title=f"SFS comparison, L1 norm:{inf.get_residual(1) / inf.sfs_sel.n_polymorphic:.2f}"
    )

if sfs_comparison_detailed is not None:
    inf.plot_sfs_comparison(
        sfs_types=["neutral", "selected", "modelled"],
        file=sfs_comparison_detailed,
        show=testing,
        title=''
    )

if out_sfs_input is not None:
    inf.plot_sfs_comparison(
        sfs_types=["neutral", "selected"],
        file=out_sfs_input,
        show=testing
    )

if out_bucket_sizes is not None:
    inf.plot_bucket_sizes(file=out_bucket_sizes, show=testing)

pass
