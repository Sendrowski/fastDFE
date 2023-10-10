"""
Visualize a DFE.
"""

__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"
__date__ = "2023-03-06"

import numpy as np
from matplotlib import pyplot as plt

try:
    from snakemake.shell import shell

    import sys

    # necessary to import fastdfe locally
    sys.path.append('..')

    testing = False
    input = snakemake.input[0]
    postprocessing_source = '../resources/polydfe/postprocessing/script.R'
    out = snakemake.output[0]
except ModuleNotFoundError:
    # testing
    testing = True
    n = 20
    params = {
        'S_d': -1000,
        'b': 0.1,
        'p_b': 0.1,
        'S_b': 0.1
    }
    out = "scratch/parametrization.png"

import fastdfe as fd

p = fd.GammaExpParametrization()
d = fd.Discretization(
    n=n,
    intervals_ben=(1e-15, 1000, 1000),
    intervals_del=(-1000000, -1e-15, 1000)
)

d1 = p.get_pdf(**params)(d.s)
d2 = p._discretize(params, d.bins) / d.interval_sizes

plt.plot(np.arange(d.n_intervals), d1, alpha=0.5, label='midpoints')
plt.plot(np.arange(d.n_intervals), d2, alpha=0.5, label='exact')

plt.legend()

ax = plt.gca()
xticks = ax.get_xticks()
labels = ["{:.0e}".format(d.s[int(l)]) if 0 <= int(l) < d.n_intervals else None for l in xticks]
ax.set_xticklabels(labels)

plt.title(params)
plt.margins(x=0)

plt.xlabel('S')
plt.ylabel('f(S)')

plt.gcf().set_size_inches(np.array([1, 0.35]) * np.array(plt.rcParams["figure.figsize"]))

plt.savefig(out, dpi=200, bbox_inches='tight', pad_inches=0.1)

if testing:
    plt.show()

pass
