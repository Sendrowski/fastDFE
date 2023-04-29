"""
Create a config file given a spectra object.
"""

__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"
__date__ = "2023-03-07"

try:
    from snakemake.shell import shell

    import sys

    # necessary to import dfe module
    sys.path.append('..')

    testing = False
    spectra_file = snakemake.input.spectra
    init_file = snakemake.input.init
    levels = snakemake.params.get('levels', [1])
    out = snakemake.output[0]
except ModuleNotFoundError:
    # testing
    testing = True
    spectra_file = 'results/spectra/pendula.pubescens.example_1.example_2.example_3.csv'
    init_file = 'resources/polydfe/init/C.full_anc_init'
    levels = [1]
    out = 'scratch/config_merged.yaml'

from fastdfe import Spectra, Config

spectra = Spectra.from_file(spectra_file)

config = Config(
    sfs_neut=spectra[['sfs_neut.*']].merge_groups(levels),
    sfs_sel=spectra[['sfs_sel.*']].merge_groups(levels),
    polydfe_init_file=init_file
)

config.to_file(out)
