import fastdfe as fd

spectra = fd.Spectra.from_file(
    "https://github.com/Sendrowski/fastDFE/blob/dev/"
    "resources/SFS/betula/spectra.20.csv?raw=true"
)

inf = fd.JointInference(
    sfs_neut=spectra['neutral.*'].merge_groups(1),
    sfs_sel=spectra['selected.*'].merge_groups(1),
    do_bootstrap=True,
    model=fd.DiscreteFractionalParametrization(),
    shared_params=[fd.SharedParams(params=['S1', 'S2'], types='all')]
)

inf.run()

inf.plot_discretized()
#inf.plot_inferred_parameters(scale='lin')

pass
