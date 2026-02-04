"""
Base inference example for presentation
"""
import matplotlib.pyplot as plt
import numpy as np

import fastdfe as fd

sim = fd.Simulation(
    params=fd.GammaExpParametrization().x0,
    sfs_neut=fd.Spectrum.get_neutral(n=7, n_sites=1e8, theta=1e-4),
)
sim.run()

inf = fd.BaseInference(
    sfs_neut=sim.sfs_neut * 0.25,
    sfs_sel=sim.sfs_sel,
)

inf.run()

# create subplots
axs = plt.subplots(1, 3, figsize=(11, 3), dpi=400)[1].flatten()

# plot results
types = ['neutral', 'selected']
inf.plot_sfs_comparison(ax=axs[0], show=False, sfs_types=types)
inf.plot_sfs_comparison(ax=axs[1], show=False, colors=['C1', 'C5'])
inf.plot_discretized(ax=axs[2], show=False, intervals=[-np.inf, -100, -1, 0, np.inf])
axs[2].set_ylabel('')
# decrease x-axis tick labels in third plot
axs[2].tick_params(axis='x', labelsize=9)
axs[2].set_xlabel('$S = 4 N_e s$', fontsize=10)

# use scientific notation for y-axis in first two plots
for ax in axs[:2]:
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

plt.tight_layout(pad=2.5)
plt.savefig('scratch/base_inference_presentation.png', dpi=400)

plt.show()

pass
