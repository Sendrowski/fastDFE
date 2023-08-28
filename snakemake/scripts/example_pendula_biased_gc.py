import numpy as np

import fastdfe as fd

p = fd.Parser(
    n=10,
    vcf="../resources/genome/betula/all.vcf.gz",
    stratifications=[fd.DegeneracyStratification(), fd.BaseTransitionStratification()]
)

# parse SFS
s: fd.Spectra = p.parse()

s.plot(use_subplots=True)

# extract neutral and selected SFS
neut = s['neutral.*'].merge_groups(1)
sel = s['selected.*'].merge_groups(1)

"""
# create inference objects
inferences = [fd.BaseInference(
    sfs_neut=neut[t],
    sfs_sel=sel[t],
    model=fd.DiscreteFractionalParametrization(np.array([-100000, -100, -10, -1, 1, 1000])),
    do_bootstrap=True
) for t in neut.types]

# run inference
[i.run() for i in inferences]

fd.Inference.plot_discretized(inferences, labels=neut.types)
"""

inf = fd.JointInference(
    sfs_neut=neut,
    sfs_sel=sel,
    shared_params=[fd.SharedParams(params=['S1', 'S4'], types='all')],
    model=fd.DiscreteFractionalParametrization(np.array([-100000, -100, -10, -1, 1, 1000])),
    covariates=[fd.Covariate(
        param='S3',
        values=dict((t, int(t not in ['A>T', 'T>A', 'G>C', 'C>G'])) for t in neut.types)
    )],
    fixed_params=dict(all=dict(eps=0)),
    do_bootstrap=True
)

inf.run()

inf.plot_discretized(kwargs_legend=dict(prop=dict(size=5)))

pass
