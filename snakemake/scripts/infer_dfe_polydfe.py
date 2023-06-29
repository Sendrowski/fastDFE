"""
Run polyDFE.
"""

__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"
__date__ = "2023-03-06"

import os

try:
    from snakemake.shell import shell

    import sys

    # necessary to import dfe module
    sys.path.append('..')

    execute = shell
    testing = False
    config_file = snakemake.input[0]
    bin = snakemake.params.get('bin', '../resources/polydfe/bin/polyDFE-2.0-macOS-64-bit')
    out = snakemake.output.polydfe
    out_summary = snakemake.output.summary
    out_serialized = snakemake.output.serialized
    out_dfe = snakemake.output.get('dfe', None)
    out_params = snakemake.output.get('params', None)
except ModuleNotFoundError:
    # testing
    testing = True
    execute = None
    config_file = 'results/configs/pendula_D_full/config.yaml'
    bin = '../resources/polydfe/bin/polyDFE-2.0-macOS-64-bit'
    out = 'scratch/polydfe_out.txt'
    out_summary = 'scratch/polydfe_out.json'
    out_serialized = 'scratch/polydfe_serialized.json'
    out_dfe = "scratch/dfe.png"
    out_params = "scratch/params.png"

from fastdfe import Config
from fastdfe.polydfe import PolyDFE

config = Config.from_file(config_file)

polydfe = PolyDFE(config)

summary = polydfe.run(out, bin=bin, wd=os.getcwd(), execute=execute)

# save summary to file
summary.to_file(out_summary)

# save polyDFE wrapper object to file
polydfe.to_file(out_serialized)

polydfe.plot_inferred_parameters(file=out_params, show=testing)
polydfe.plot_discretized(file=out_dfe, show=testing)

pass
