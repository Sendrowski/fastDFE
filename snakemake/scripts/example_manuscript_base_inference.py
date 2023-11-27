import fastdfe as fd

# create inference object
inf = fd.BaseInference(
    sfs_neut=fd.Spectrum([66200, 410, 120, 60, 42, 43, 52, 65, 0]),
    sfs_sel=fd.Spectrum([281937, 600, 180, 87, 51, 43, 49, 61, 0]),
    model=fd.GammaExpParametrization(),  # the model to use
    n_runs=10,  # number of optimization runs
    n_bootstraps=100,  # number of bootstrap replicates
    do_bootstrap=True
)

# run inference
inf.run()

import matplotlib.pyplot as plt

# create subplots
axs = plt.subplots(2, 2, figsize=(11, 7))[1].flatten()

# plot results
types = ['neutral', 'selected']
inf.plot_sfs_comparison(ax=axs[0], show=False, sfs_types=types)
inf.plot_sfs_comparison(ax=axs[1], show=False, colors=['C1', 'C5'])
inf.plot_inferred_parameters(ax=axs[2], show=False)
inf.plot_discretized(ax=axs[3], show=True)

pass