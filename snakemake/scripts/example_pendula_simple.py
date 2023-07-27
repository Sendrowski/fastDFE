import matplotlib.pyplot as plt

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
    n_runs=10,
    n_bootstraps=100,
    do_bootstrap=True
)

inf.run()

axs = plt.subplots(nrows=2, ncols=2, figsize=(11, 10))[1].flatten()

inf.plot_sfs_comparison(ax=axs[0], sfs_types=['neutral', 'selected', 'modelled'], show=False)
inf.plot_discretized(ax=axs[1], show=False)
inf.plot_inferred_parameters(ax=axs[2], show=False)
inf.plot_nested_likelihoods(ax=axs[3])

pass
