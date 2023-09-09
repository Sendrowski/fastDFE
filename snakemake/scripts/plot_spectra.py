"""
Plot spectra.
"""

__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"
__date__ = "2023-03-30"

try:
    import sys

    # necessary to import dfe module
    sys.path.append('..')

    testing = False
    file = snakemake.input[0]
    out = snakemake.output[0]
except NameError:
    # testing
    testing = True
    file = "results/sfs/pendula/all.csv"
    out = "scratch/spectra.png"

from fastdfe import Spectra

s = Spectra.from_file(file)

s = s.merge_groups([0])

s.plot(show=testing, file=out, show_monomorphic=False, use_subplots=False)

pass
