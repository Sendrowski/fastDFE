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

    # necessary to import fastdfe locally
    sys.path.append('..')

    testing = False
    n = snakemake.params.n
    S = snakemake.params.S
    out = snakemake.output[0]
except ModuleNotFoundError:
    # testing
    testing = True
    n = 7
    S = -10
    out = "scratch/sfs_selected.png"

from fastdfe.discretization import Discretization
from fastdfe import Spectrum

d = Discretization(
    n=n,
    intervals_ben=(1e-15, 1000, 1000),
    intervals_del=(-1000000, -1e-15, 1000)
)

k = np.arange(1, n)
s = Spectrum.from_polymorphic(d.get_counts_semidominant_regularized(S * np.ones_like(k), k))

plt.figure(figsize=(2, 2))
s.plot(file=out, show=False, title=f"S = {S}", ax=plt.gca())
plt.gca().set_ylim(bottom=0, top=1.25)
plt.show()

pass
