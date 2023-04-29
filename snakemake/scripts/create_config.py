"""
Create a config file which can be used both to run
polyDFE and the new inference tool.
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
    spectra_file = snakemake.input.sfs
    init_file = snakemake.input.init
    do_bootstrap = snakemake.params.get('do_bootstrap', None)
    model = snakemake.params.get('model', 'GammaExpParametrization')
    out = snakemake.output[0]
except ModuleNotFoundError:
    # testing
    testing = True
    spectra_file = 'resources/polydfe/pubescens/spectra/sfs.txt'
    init_file = 'resources/polydfe/init/D.full_init'
    do_bootstrap = None
    model = 'DiscreteParametrization'
    out = 'scratch/config.yaml'

import fastdfe

# create config object
config = fastdfe.Config(
    polydfe_spectra_config=spectra_file,
    polydfe_init_file=init_file,
    do_bootstrap=do_bootstrap,
    model=model
)

# save config to file
config.to_file(out)

pass
