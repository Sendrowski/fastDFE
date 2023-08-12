import fastdfe as fd

# create inference object
inf = fd.JointInference(
    sfs_neut=fd.Spectra(dict(
        pendula=[177130, 997, 441, 228, 156, 117, 114, 83, 105, 109, 652],
        pubescens=[172528, 3612, 1359, 790, 584, 427, 325, 234, 166, 76, 31]
    )),
    sfs_sel=fd.Spectra(dict(
        pendula=[797939, 1329, 499, 265, 162, 104, 117, 90, 94, 119, 794],
        pubescens=[791106, 5326, 1741, 1005, 756, 546, 416, 294, 177, 104, 41]
    )),
    # fix eps to 0
    fixed_params=dict(all=dict(eps=0)),
    # share S_b and p_b between types
    shared_params=[fd.SharedParams(types='all', params=['S_b', 'p_b'])],
    do_bootstrap=True
)

# run inference
inf.run()

# plot discretized DFE
inf.plot_discretized()

# inf.plot_inferred_parameters()

pass
