import sys
sys.path.append('.')

import fastdfe as fd

inf = fd.BaseInference(
    sfs_neut=fd.Spectrum([177130, 997, 441, 228, 156, 117, 114, 83, 105, 109, 652]),
    sfs_sel=fd.Spectrum([797939, 1329, 499, 265, 162, 104, 117, 90, 94, 119, 794]),
    fixed_params=dict(all=dict(h=0.5)),
    parallelize=False
)

inf.run()

inf = fd.BaseInference(
    sfs_neut=fd.Spectrum([177130, 997, 441, 228, 156, 117, 114, 83, 105, 109, 652]),
    sfs_sel=fd.Spectrum([797939, 1329, 499, 265, 162, 104, 117, 90, 94, 119, 794]),
    fixed_params=dict(all=dict(h=0.5)),
    parallelize=True
)

inf.run()
