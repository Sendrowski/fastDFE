"""
Create spectra objects.
"""

__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"
__date__ = "2022-30-03"

try:
    import sys

    # necessary to import dfe module
    sys.path.append('..')

    testing = False
    files = snakemake.input
    names = snakemake.params.names
    out = snakemake.output[0]
except NameError:
    # testing
    testing = True
    files = [
        'resources/polydfe/pendula/spectra/sfs.txt',
        'resources/polydfe/pubescens/spectra/sfs.txt'
    ]
    names = [
        'pendula',
        'pubescens'
    ]
    out = "scratch/spectra.csv"

import fastdfe
from fastdfe import Spectra

spectra = Spectra({})
for name, file in zip(names, files):
    s = fastdfe.spectrum.parse_polydfe_sfs_config(file)

    spectra['sfs_neut.' + name] = s['sfs_neut']
    spectra['sfs_sel.' + name] = s['sfs_sel']

spectra.to_file(out)
