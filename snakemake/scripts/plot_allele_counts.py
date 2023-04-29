"""
Visualize the expected allele counts.
"""

__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"
__date__ = "2023-03-06"

import numpy as np
from matplotlib import pyplot as plt

try:
    from snakemake.shell import shell

    import sys

    # necessary to import dfe module
    sys.path.append('..')

    testing = False
    input = snakemake.input[0]
    postprocessing_source = '../resources/polydfe/postprocessing/script.R'
    out = snakemake.output[0]
except ModuleNotFoundError:
    # testing
    testing = True
    n = 7
    out = "scratch/allele_counts.png"

from fastdfe.discretization import Discretization

d = Discretization(
    n=n,
    intervals_ben=(1e-15, 1000, 1000),
    intervals_del=(-1000000, -1e-15, 1000)
)

fig, (ax1, ax2) = plt.subplots(nrows=2)
fig.tight_layout(pad=4)

for i in range(1, n):
    ax1.plot(np.arange(d.n_intervals), d.get_allele_count(d.s, i * np.ones_like(d.s)), label=i)
    ax2.plot(np.arange(d.n_intervals), d.get_allele_count_regularized(d.s, i * np.ones_like(d.s)),
             label=i)

ax1.set_title('allele counts')
ax2.set_title('regularized allele counts')

for ax in (ax1, ax2):
    ax.set_xlabel('S')
    ax.set_ylabel('count')

    xticks = plt.gca().get_xticks()
    labels = ["{:.0e}".format(d.s[int(l)]) if 0 <= int(l) < d.n_intervals else None for l in xticks]
    ax.set_xticklabels(labels)

    ax.legend(ncol=3, prop=dict(size=5))

    ax.margins(x=0)

    ax.set_xlim(left=0, right=d.n_intervals - 1)

plt.savefig(out, dpi=200, bbox_inches='tight', pad_inches=0.1)

if testing:
    plt.show()

pass
