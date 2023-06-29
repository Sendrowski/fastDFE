"""
Visualize linearized DFE to SFS transformation.
"""

__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"
__date__ = "2023-03-27"

import numpy as np
from matplotlib import pyplot as plt

try:
    from snakemake.shell import shell

    import sys

    # necessary to import dfe module
    sys.path.append('..')

    testing = False
    n = snakemake.params.n
    S = snakemake.params.S
    out = snakemake.output[0]
except ModuleNotFoundError:
    # testing
    testing = True
    n = 7
    S = 7
    out = "scratch/sfs_selected.png"

from fastdfe.discretization import Discretization
from fastdfe import Spectrum

d = Discretization(
    n=n,
    intervals_ben=(1e-15, 1000, 1000),
    intervals_del=(-1000000, -1e-15, 1000)
)

k = np.arange(1, n)
s = Spectrum.from_polymorphic(d.get_allele_count_regularized(S * np.ones_like(k), k))

s.plot(file=out, show=testing)

pass
