"""
Simulate the selected SFS using fastDFE.
"""

__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"
__date__ = "2024-12-25"

from matplotlib import pyplot as plt

try:
    import sys

    # necessary to import fastdfe locally
    sys.path.append('..')

    testing = False
    sfs_file = snakemake.input[0]
    s_b = snakemake.params.s_b
    b = snakemake.params.b
    s_d = snakemake.params.s_d
    p_b = snakemake.params.p_b
    n = snakemake.params.n
    mu = snakemake.params.mu
    title = snakemake.params.title
    out_sfs = snakemake.output.sfs
    out_comparison = snakemake.output.comp
except NameError:
    # testing
    testing = True
    sfs_file = 'results/slim/n_replicate=1/n_chunks=100/g=1e4/L=1e7/mu=1e-8/r=1e-7/N=1e3/s_b=1e-9/b=5/s_d=3e-1/p_b=0/n=20/unfolded/sfs.csv'
    s_b = 1e-9
    b = 1
    s_d = 1e-3
    p_b = 0.2
    n = 10
    mu = 1e-8
    title = "g=1e4/L=1e8/mu=1e-8/r=1e-7/N=1e3/s_b=1e-9/b=1/s_d=1e-1/p_b=0.2"
    out_sfs = "scratch/sfs.csv"
    out_comparison = "scratch/comp.png"

import fastdfe as fd

spectra = fd.Spectra.from_file(sfs_file)

theta = spectra['neutral'].theta
Ne = theta / (4 * mu)

model = fd.GammaExpParametrization()
model.bounds['S_b'] = (1e-10, 100)

sim = fd.Simulation(
    params=dict(
        S_b=4 * Ne * s_b,
        b=b,
        S_d=-4 * Ne * s_d,
        p_b=p_b,
    ),
    sfs_neut=spectra['neutral'],
    model=model
)

sfs_sel = sim.run()

sfs_sel.to_file(out_sfs)

comp = fd.Spectra(dict(
    slim=spectra['selected'],
    fastdfe=sfs_sel
))

plt.rcParams['axes.titlesize'] = 11
comp.plot(file=out_comparison, show=testing, title=title)

pass
