"""
Base inference example for presentation
"""
import numpy as np

import fastdfe as fd

inf = fd.BaseInference(
    sfs_neut=fd.Spectrum(
        [66200, 410, 120, 60, 42, 43, 52, 65, 0]
    ),
    sfs_sel=fd.Spectrum(
        [281937, 600, 180, 87, 51, 43, 49, 61, 0]
    ),
    model=fd.GammaExpParametrization(),
    n_runs=10,
    n_bootstraps=100,
    do_bootstrap=True
)

inf.run()

import matplotlib.pyplot as plt

# create subplots
axs = plt.subplots(3, 1, figsize=(3.5, 6))[1].flatten()

# plot results
types = ['neutral', 'selected']
inf.plot_sfs_comparison(ax=axs[0], show=False, sfs_types=types)
inf.plot_sfs_comparison(ax=axs[1], show=False, colors=['C1', 'C5'])
inf.plot_discretized(ax=axs[2], show=False, intervals=[-np.inf, -100, -1, 0, np.inf])

plt.savefig('scratch/base_inference_presentation.png', dpi=400)

plt.show()

pass