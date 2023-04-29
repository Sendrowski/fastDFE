"""
Run polyDFE.
"""

__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"
__date__ = "2023-03-06"

try:
    from snakemake.shell import shell

    import sys

    # necessary to import dfe module
    sys.path.append('..')

    execute = shell
    testing = False
    config_file = snakemake.input[0]
    bin = snakemake.params.get('bin', 'resources/polydfe/bin/polyDFE-2.0-macOS-64-bit')
    out = snakemake.output.polydfe
    out_summary = snakemake.output.summary
    out_wrapper = snakemake.output.wrapper
except ModuleNotFoundError:
    # testing
    testing = True
    execute = None
    config_file = 'results/configs/pendula_D_full/config.yaml'
    bin = 'resources/polydfe/bin/polyDFE-2.0-macOS-64-bit'
    out = 'scratch/polydfe_out.txt'
    out_summary = 'scratch/polydfe_out.json'
    out_wrapper = 'scratch/polydfe_wrapper.json'

    import logging

    # set log level to debug
    logging.getLogger('fastdfe').setLevel(logging.DEBUG)

from fastdfe import Config
from fastdfe.polydfe import PolyDFE
import os

config = Config.from_file(config_file)

polydfe = PolyDFE(config)
summary = polydfe.run(out, bin=bin, wd=os.getcwd(), execute=execute)

# save summary to file
summary.to_file(out_summary)

# save polyDFE wrapper object to file
polydfe.to_file(out_wrapper)

pass
