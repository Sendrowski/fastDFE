"""
Visualize DFE inference results.
"""

__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"
__date__ = "2023-03-02"

import numpy as np
from matplotlib import pyplot as plt

try:
    import sys

    # necessary to import fastdfe locally
    sys.path.append('..')

    testing = False
    input_fastdfe = snakemake.input.fastdfe
    input_polydfe = snakemake.input.polydfe
    out_discretized = snakemake.output.discretized
    out_params = snakemake.output.inferred_params
except NameError:
    # testing
    testing = True
    input_fastdfe = "results/fastdfe/pubescens_C_full_bootstrapped_100/serialized.json"
    input_polydfe = "results/polydfe/pubescens_C_full_bootstrapped_100/serialized.json"
    out_discretized = "scratch/comp_discretized.png"
    out_params = "scratch/comp_inferred_params.png"

from fastdfe import BaseInference, Inference
from fastdfe.polydfe import PolyDFE

inferences = [BaseInference.from_file(input_fastdfe), PolyDFE.from_file(input_polydfe)]

Inference.plot_discretized(inferences, labels=['fastDFE', 'polyDFE'], show=testing, file=out_discretized, title='')
Inference.plot_inferred_parameters(inferences, labels=['fastDFE', 'polyDFE'], show=False, file=None, title='')

# use symlog scale for the inferred parameters
plt.gca().set_yscale('symlog', linthresh=0.1)

plt.savefig(out_params, bbox_inches='tight')

if testing:
    plt.show()

pass
