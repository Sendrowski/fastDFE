"""
Create config file for bootstrapped samples to be run with polyDFE.
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
    input = snakemake.input[0]
    out = snakemake.output[0]
except NameError:
    # testing
    testing = True
    input = 'scratch/polydfe_wrapper.json'
    out = 'scratch/polydfe_config.yaml'

from fastdfe.polydfe import PolyDFE

# restore polyDFE wrapper from file
polydfe = PolyDFE.from_file(input)

# create bootstrap
config = polydfe.create_bootstrap()

# save polyDFE wrapper object to file
config.to_file(out)
