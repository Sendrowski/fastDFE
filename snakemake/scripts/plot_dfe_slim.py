"""
Plot the distribution of selection coefficients from SLiM simulations.
"""

__author__ = "Janek Sendrowski"
__contact__ = "j.sendrowski18@gmail.com"
__date__ = "2024-03-03"

import numpy as np

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
    spectra_file = "results/slim/n_replicate=1/g=10000/L=10000000/mu=1e-07/r=1e-08/N=5000/s_b=0.1/b=0.2/s_d=0.03/p_b=0.1/n=10/sfs.csv"
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
    S_b=4 * Ne * s_b,
    b=b,
    p_b=p_b,Ne=Ne
)

fd.GammaExpParametrization().plot(
    params=dict(S_d=min(params['S_d'], -1e-16), S_b=max(params['S_b'], 1e-16), b=b, p_b=p_b),
    intervals=[-np.inf, -100, -10, -1, 1, np.inf],
    title='Simulated DFE\n' + ", ".join([f"${k}$={round(v, 2)}" for k, v in params.items()]),
    show=testing,
    file=out
)

pass
