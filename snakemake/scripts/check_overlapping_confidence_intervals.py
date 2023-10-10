"""
Testing the inference results against the results of polyDFE.
"""

__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"
__date__ = "2023-03-16"

import numpy as np

try:
    from snakemake.shell import shell

    import sys

    # necessary to import fastdfe locally
    sys.path.append('..')

    testing = False
    input_fastdfe = snakemake.input.fastdfe
    input_polydfe = snakemake.input.polydfe
except ModuleNotFoundError:
    # testing
    testing = True
    input_fastdfe = 'results/fastdfe/pendula_C_full_bootstrapped_100/serialized.json'
    input_polydfe = 'results/polydfe/pendula_C_full_bootstrapped_100/serialized.json'

from fastdfe import BaseInference
from fastdfe.polydfe import PolyDFE

native = BaseInference.from_file(input_fastdfe)
polydfe = PolyDFE.from_file(input_polydfe)

res = dict(
    fastdfe=native.get_discretized(),
    polydfe=polydfe.get_discretized()
)

# fastdfe.plot_discretized(title='fastDFE')
# polydfe.plot_discretized(title='polyDFE')

# check that the confidence interval overlap
assert np.all(res['fastdfe'][1][0] < res['polydfe'][1][1])
assert np.all(res['polydfe'][1][0] < res['fastdfe'][1][1])
