"""
Infer the DFE from the SFS for the HGDP dataset using polyDFE.
"""

__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"
__date__ = "2023-05-29"

import os

try:
    from snakemake.shell import shell

    import sys

    # necessary to import dfe module
    sys.path.append('..')

    execute = shell
    testing = False
    config_file = snakemake.input.config
    spectra_file = snakemake.input.spectra
    bin = snakemake.params.get('bin', 'resources/polydfe/bin/polyDFE-2.0-macOS-64-bit')
    out_summary = snakemake.output.summary
    out_serialized = snakemake.output.serialized
    out_polydfe = snakemake.output.polydfe
    out_dfe = snakemake.output.dfe
    out_spectra = snakemake.output.spectra
    out_params = snakemake.output.params
except ModuleNotFoundError:
    # testing
    testing = True
    execute = None
    config_file = '../resources/configs/HGDP/polydfe.yaml'
    spectra_file = "results/spectra/hgdp/1/opts.n.10/all.csv"
    bin = '../resources/polydfe/bin/polyDFE-2.0-macOS-64-bit'
    out_summary = "scratch/summary.json"
    out_serialized = "scratch/serialized.json"
    out_polydfe = "scratch/polydfe.txt"
    out_dfe = "scratch/dfe.png"
    out_spectra = "scratch/spectra.png"
    out_params = "scratch/params.png"

from fastdfe import Config, Spectra
from fastdfe.polydfe import PolyDFE

# load config from file
config = Config.from_file(config_file)

# load spectra from file
spectra = Spectra.from_file(spectra_file)

# update config with spectra
config.update(
    sfs_neut=spectra['neutral'],
    sfs_sel=spectra['selected'],
    do_bootstrap=False
)

# create from config
inference = PolyDFE.from_config(config)

# perform inference
summary = inference.run(out_polydfe, bin=bin, wd=os.getcwd(), execute=execute)

# save object in serialized form
inference.to_file(out_serialized)

# save summary
summary.to_file(out_summary)

inference.plot_inferred_parameters(file=out_params, show=testing)
inference.plot_discretized(file=out_dfe, show=testing)

pass
