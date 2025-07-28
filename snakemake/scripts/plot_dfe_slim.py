"""
Plot the distribution of selection coefficients from SLiM simulations.
"""

__author__ = "Janek Sendrowski"
__contact__ = "j.sendrowski18@gmail.com"
__date__ = "2024-03-03"

import numpy as np
from matplotlib import pyplot as plt

try:
    import sys

    # necessary to import fastdfe locally
    sys.path.append('..')

    testing = False
    spectra_file = snakemake.input[0]
    p_b = snakemake.params.p_b
    s_b = snakemake.params.s_b
    s_d = snakemake.params.s_d
    b = snakemake.params.b
    mu = snakemake.params.mu
    out = snakemake.output[0]
except NameError:
    # testing
    testing = True
    spectra_file = "results/slim/n_replicate=1/n_chunks=100/g=1e4/L=1e7/mu=1e-8/r=1e-7/N=1e3/s_b=1e-9/b=5/s_d=3e-1/p_b=0/n=20/unfolded/sfs.csv"
    p_b = 0.4
    s_b = 18
    s_d = 67
    b = 7
    mu = 1e-9
    out = "scratch/slim_dfe.png"

import fastdfe as fd

spectra = fd.Spectra.from_file(spectra_file)

Ne = spectra['neutral'].theta / (4 * mu)

params = dict(
    S_d=-4 * Ne * s_d,
    b=b,
    p_b=p_b,
    S_b=4 * Ne * s_b,
    N_e=Ne
)

fig, ax = plt.subplots(figsize=(4.5, 3.2))

fd.GammaExpParametrization().plot(
    params=dict(S_d=min(params['S_d'], -1e-16), S_b=max(params['S_b'], 1e-16), b=b, p_b=p_b),
    intervals=[-np.inf, -100, -10, -1, 1, np.inf],
    title='Simulated DFE\n' + ", ".join([f"${k}$={round(v, 2)}" for k, v in params.items()]),
    show=testing,
    file=out,
    ax=ax
)

pass
