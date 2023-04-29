"""
Merge several spectra.
"""

__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"
__date__ = "2023-03-30"

try:
    import sys

    # necessary to import dfe module
    sys.path.append('..')

    testing = False
    spectra_files = snakemake.input
    out = snakemake.output[0]
except NameError:
    # testing
    testing = True
    spectra_files = [
        "results/sfs/pendula/16.csv",
        "results/sfs/pendula/15.csv",
        "results/sfs/pendula/14.csv"
    ]
    out = "scratch/out.txt"

import numpy as np

from fastdfe import Spectra

s = Spectra.from_file(spectra_files[0])
for f in spectra_files[1:]:
    s += Spectra.from_file(f)

s.to_file(out)

pass
