"""
Testing the inference results against the results of polyDFE.
"""

__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"
__date__ = "2023-03-31"

try:
    from snakemake.shell import shell

    import sys

    # necessary to import dfe module
    sys.path.append('..')

    testing = False
    input_fastdfe = snakemake.input.fastdfe
    input_polydfe = snakemake.input.polydfe
except ModuleNotFoundError:
    # testing
    testing = True
    spectra_files = [
        'results/sfs/pendula/DegeneracyStratification/all.csv',
        'results/sfs/pendula/DegeneracyStratification.BaseContextStratification/all.csv',
        'results/sfs/pendula/DegeneracyStratification.BaseTransitionStratification/all.csv',
        'results/sfs/pendula/DegeneracyStratification.BaseContextStratification.BaseTransitionStratification/all.csv'
    ]

from fastdfe import Spectra

spectra = [Spectra.from_file(s) for s in spectra_files]

pass
