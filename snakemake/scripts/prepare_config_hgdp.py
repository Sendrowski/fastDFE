"""
Prepare fastDFE config for HGDP data.
"""

__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"
__date__ = "2023-05-30"

try:
    import sys

    # necessary to import fastdfe locally
    sys.path.append('..')

    testing = False
    config_file = snakemake.input.config
    spectra_file = snakemake.input.spectra
    out = snakemake.output[0]
except NameError:
    # testing
    testing = True
    config_file = '../resources/configs/HGDP/test.yaml'
    spectra_file = "results/spectra/hgdp/1/opts.n.20/all.csv"
    out = "scratch/config.yaml"

from fastdfe import Config, Spectra

# load config from file
config = Config.from_file(config_file)

# load spectra from file
spectra = Spectra.from_file(spectra_file)

# update config with spectra
config.update(
    sfs_neut=spectra['neutral'],
    sfs_sel=spectra['selected']
)

# save config
config.to_file(out)
