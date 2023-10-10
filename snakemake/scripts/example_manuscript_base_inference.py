import fastdfe as fd

# create inference object
inf = fd.BaseInference(
    # neutral SFS
    sfs_neut=fd.Spectrum(
        [177130, 997, 441, 228, 156, 117, 114, 83, 105, 109, 652]
    ),
    # selected SFS
    sfs_sel=fd.Spectrum(
        [797939, 1329, 499, 265, 162, 104, 117, 90, 94, 119, 794]
    ),
    model=fd.GammaExpParametrization(),  # the model to use
    n_runs=10,  # number of independent optimization runs
    n_bootstraps=100,  # number of bootstrap replicates
    do_bootstrap=True
)

# run inference
inf.run()

import matplotlib.pyplot as plt

# create subplots
axs = plt.subplots(2, 2, figsize=(11, 10))[1].flatten()

# plot results
inf.plot_sfs_comparison(ax=axs[0], show=False)
inf.plot_discretized(ax=axs[1], show=False)
inf.plot_inferred_parameters(ax=axs[2], show=False)
inf.plot_nested_models(ax=axs[3], show=False)

plt.show()
