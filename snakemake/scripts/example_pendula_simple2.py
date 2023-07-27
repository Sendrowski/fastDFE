import numpy as np

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
    #model=fd.DiscreteFractionalParametrization(np.array([-100000, -100, -10, -1, 1, 1000])),
    n_runs=10,
    n_bootstraps=100,
    do_bootstrap=True
)

inf.run()

inf.plot_discretized()

pass
