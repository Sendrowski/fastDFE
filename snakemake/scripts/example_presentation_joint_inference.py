import matplotlib.pyplot as plt
import fastdfe as fd

spectra = fd.Spectra.from_file(
    "../resources/SFS/betula/spectra.20.csv"
)

inf = fd.JointInference(
    sfs_neut=spectra['neutral.*'].merge_groups(1),
    sfs_sel=spectra['selected.*'].merge_groups(1),
    do_bootstrap=True,
    model=fd.DiscreteFractionalParametrization(),
    shared_params=[fd.SharedParams(params=['S1', 'S2', 'S4'], types='all')]
)

inf.run()


fig = plt.figure(figsize=(4, 3))
fd.Inference.plot_inferred_parameters(
    ax=plt.gca(),
    inferences=list(inf.joint_inferences.values()),
    labels=['type 1', 'type 2'],
    scale='lin',
    show=False
)

plt.gca().set_xticklabels([f"$p{i}$" for i in range(1, 8)])
plt.show()

pass
