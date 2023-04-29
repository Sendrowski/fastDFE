"""
Merge bootstraps and original inference result.
"""

__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"
__date__ = "2023-03-10"

try:
    from snakemake.shell import shell

    import sys

    # necessary to import dfe module
    sys.path.append('..')

    testing = False
    file_original = snakemake.input.original
    files_bootstrap = snakemake.input.bootstraps
    out = snakemake.output[0]
except NameError:
    # testing
    testing = True
    file_original = 'results/polydfe/pendula_C_full_anc/serialized.json'
    files_bootstrap = ['results/polydfe/pendula_C_full_anc/serialized.json' for i in range(10)]
    out = 'scratch/polydfe_bs.json'

import fastdfe

# load from file
original = fastdfe.PolyDFE.from_file(file_original)
bootstraps = [fastdfe.PolyDFE.from_file(f) for f in files_bootstrap]

# add bootstraps
original.add_bootstraps(bootstraps)

# save to file
original.to_file(out)
