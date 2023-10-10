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
    inputs = snakemake.input
    out_dfe_discretized = snakemake.output[0]
except NameError:
    # testing
    testing = True
    inputs = [
        "results/fastdfe/pendula_C_full_bootstrapped_100/serialized.json",
        "results/fastdfe/pubescens_C_full_bootstrapped_100/serialized.json"
    ]
    out = "scratch/comp.png"

import fastdfe as fd

inferences = [fd.BaseInference.from_file(input) for input in inputs]

fd.Inference.plot_discretized(inferences, labels=['pendula', 'pubescens'])

pass
