"""
Visualize linearized DFE to SFS transformation.
"""

__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"
__date__ = "2023-03-27"

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import FancyBboxPatch

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
    S = 10
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

plt.figure(figsize=(2.4, 1.5), dpi=200)

ax = plt.gca()
s.plot(
    file=None,
    show=False,
    title=None,
    ax=ax,
)

# axis limits
ax.set_ylim(0, 1.2)

# aesthetics
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_linewidth(0.8)
ax.spines["bottom"].set_linewidth(0.8)

ax.tick_params(axis="both", labelsize=7, width=0.8, length=3)
ax.set_yticks([0, 0.5, 1.0])
ax.set_yticklabels(["0", "0.5", "1"])
ax.grid(axis="y", alpha=0.2, linewidth=0.6)
ax.set_xlabel('')

# margins (prevents frame cutoff)
plt.tight_layout(pad=0.6)

# save
plt.show()
plt.savefig(out, bbox_inches="tight")
plt.close()
