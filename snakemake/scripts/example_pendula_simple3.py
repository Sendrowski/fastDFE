import numpy as np

import fastdfe as fd

p = fd.Parser(
    n=10,
    vcf="../resources/genome/betula/all.vcf.gz",
    stratifications=[fd.DegeneracyStratification()]
)

# parse SFS
s: fd.Spectra = p.parse()

# create inference object
inf = fd.BaseInference(
    # neutral SFS
    sfs_neut=s['neutral'],
    # selected SFS
    sfs_sel=s['selected'],
    model=fd.DiscreteFractionalParametrization(np.array([-100000, -100, -10, -1, 1, 1000])),
    n_runs=10,
    n_bootstraps=100,
    do_bootstrap=True
)

inf.run()

inf.plot_discretized()

pass
