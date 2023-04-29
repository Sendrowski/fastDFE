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
    input_fastdfe = snakemake.input.fastdfe
    input_polydfe = snakemake.input.polydfe
    out_discretized = snakemake.output.discretized
    out_params = snakemake.output.inferred_params
except NameError:
    # testing
    testing = True
    input_fastdfe = "results/fastdfe/pubescens_C_full_anc_bootstrapped_100/serialized.json"
    input_polydfe = "results/polydfe/pubescens_C_full_anc_bootstrapped_100/serialized.json"
    out_discretized = "scratch/comp_discretized.png"
    out_params = "scratch/comp_inferred_params.png"

from fastdfe import BaseInference, Inference, PolyDFE

inferences = [BaseInference.from_file(input_fastdfe), PolyDFE.from_file(input_polydfe)]

Inference.plot_discretized(inferences, labels=['fastDFE', 'polyDFE'], show=testing, file=out_discretized)
Inference.plot_inferred_parameters(inferences, labels=['fastDFE', 'polyDFE'], show=testing, file=out_params)

pass
