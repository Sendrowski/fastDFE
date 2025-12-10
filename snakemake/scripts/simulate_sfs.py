"""
Simulate the selected SFS using fastDFE.
"""

__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"
__date__ = "2024-12-25"

import numpy as np
import re
from matplotlib import pyplot as plt

try:
    import sys

    # necessary to import fastdfe locally
    sys.path.append('.')

    testing = False
    sfs_file = snakemake.input[0]
    s_b = snakemake.params.s_b
    b = snakemake.params.b
    s_d = snakemake.params.s_d
    p_b = snakemake.params.p_b
    n = snakemake.params.n
    h = snakemake.params.h
    mu = snakemake.params.mu
    demography = snakemake.params.demography
    parallelize = snakemake.params.get('parallelize', False)
    title = snakemake.params.title
    out_sfs = snakemake.output.sfs
    out_comparison = snakemake.output.comp
except NameError:

    def get_param(string, param):
        return float(re.search(rf"{param}=([\d.e+-]+)", string).group(1))

    # testing
    testing = True
    sfs_file = 'results/slim/n_replicate=3/n_chunks=8/g=1e4/L=1e7/mu=1e-8/r=1e-6/N=1e3/s_b=1e-3/b=0.3/s_d=3e-1/p_b=0.00/n=20/dominance_function_10/unfolded/sfs.csv'
    s_b = get_param(sfs_file, 's_b')
    b = get_param(sfs_file, '/b')
    s_d = get_param(sfs_file, 's_d')
    p_b = get_param(sfs_file, 'p_b')
    n = 20
    h = float(re.search(r"dominance_function_([\d.]+)", sfs_file).group(1))
    mu = 1e-8
    demography = "dominance_function"
    parallelize = True
    title = f"$s_b$={s_b:.0e}, $b$={b}, $s_d$={s_d:.0e}, $p_b$={p_b}"
    out_sfs = "scratch/sfs.csv"
    out_comparison = "scratch/comp.png"

import fastdfe as fd

spectra = fd.Spectra.from_file(sfs_file)

theta = spectra['neutral'].theta
Ne = theta / (4 * mu)

model = fd.GammaExpParametrization()
model.bounds['S_b'] = (1e-10, 100)
model.bounds['S_d'] = (-1e6, -1e-2)

if demography == 'dominance_function':
    h_callback = lambda k, S: np.maximum(0.4 * np.exp(-h * abs(S / (4 * Ne))), 0.1)
    #h_callback = lambda k, S: np.full_like(S, 0.7)
else:
    h_callback = lambda h, S: np.full_like(S, h)

sim = fd.Simulation(
    params=dict(
        S_b=4 * Ne * s_b,
        b=b,
        S_d=-4 * Ne * s_d,
        p_b=p_b,
        h=h,
    ),
    intervals_del=(-1.0e+8, -1.0e-5, 100),
    intervals_ben=(1.0e-5, 1.0e4, 100),
    h_callback=h_callback,
    sfs_neut=spectra['neutral'],
    model=model,
    parallelize=parallelize,
)

sfs_sel = sim.run()

sfs_sel.to_file(out_sfs)

fig, ax = plt.subplots(figsize=np.array([4, 3]) * 0.9)

comp = fd.Spectra(dict(
    slim=spectra['selected'],
    fastdfe=sfs_sel
))

plt.rcParams['axes.titlesize'] = 11
comp.plot(file=out_comparison, show=testing, title=title, ax=ax)

pass
