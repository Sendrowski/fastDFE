"""
Compare different model type using likelihood ratio tests.
"""

__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"
__date__ = "2023-03-16"

try:
    import sys

    # necessary to import fastdfe locally
    sys.path.append('..')

    testing = False
    input = snakemake.input[0]
    out = snakemake.output[0]
except NameError:
    # testing
    testing = True
    input = "results/fastdfe/pendula_C_full_bootstrapped_100/serialized.json"
    out = "scratch/lrt.png"

import fastdfe as fd

fd.logger.setLevel(logging.INFO)

# load object from file
inference = fd.BaseInference.from_file(input)

# compare nested models
inference.plot_nested_models(file=out, show=testing)

pass
