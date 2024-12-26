"""
Sum up multiple SFS.
"""

try:
    import sys

    # necessary to import fastdfe locally
    sys.path.append('..')

    testing = False
    files = snakemake.input
    out = snakemake.output[0]
except NameError:
    # testing
    testing = True
    files = [
        "results/slim/n_replicate=1/n_chunks=10/g=1e4/L=1e8/mu=1e-8/r=1e-7/N=1e3/s_b=1e-9/b=1/s_d=1e-1/p_b=0.2/n=30/chunk=0/sfs.csv",
        "results/slim/n_replicate=1/n_chunks=10/g=1e4/L=1e8/mu=1e-8/r=1e-7/N=1e3/s_b=1e-9/b=1/s_d=1e-1/p_b=0.2/n=30/chunk=1/sfs.csv"
    ]
    out = f"scratch/summed_sfs.csv"

import fastdfe as fd

spectra = [fd.Spectra.from_file(file) for file in files]

summed = spectra[0]
for s in spectra[1:]:
    summed += s

summed.to_file(out)
