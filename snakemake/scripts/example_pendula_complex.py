import fastdfe as fd

# instantiate parser
p = fd.Parser(
    n=10,
    vcf="../resources/genome/betula/all.vcf.gz",
    stratifications=[fd.DegeneracyStratification(), fd.AncestralBaseStratification()]
)

# parse SFS
spectra: fd.Spectra = p.parse()

# create inference object
inf = fd.JointInference(
    # select neutral and selected spectra from stratified spectra
    sfs_neut=spectra['neutral.*'].merge_groups(1),
    sfs_sel=spectra['selected.*'].merge_groups(1),
    # fix ancestral misidentification rate to 0
    fixed_params=dict(all=dict(eps=0)),
    # share S_b and p_b across types
    shared_params=[fd.SharedParams(params=['p_b', 'S_b'], types='all')],
    do_bootstrap=True
)

# run inference
inf.run()

# plot discretized DFE
inf.plot_discretized(show=False, kwargs_legend=dict(framealpha=0))

pass
