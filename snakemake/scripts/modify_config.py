"""
Modify an existing config file.
"""

__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"
__date__ = "2023-03-13"

try:
    from snakemake.shell import shell

    import sys

    # necessary to import dfe module
    sys.path.append('..')

    testing = False
    config_file = snakemake.input[0]
    opts = snakemake.params.get('opts', {})
    out = snakemake.output[0]
except NameError:
    # testing
    testing = True
    config_file = 'results/configs/pendula_C_full_anc/config.yaml'
    opts = dict(
        do_bootstrap=True
    )
    out = 'scratch/config_modified.yaml'

import fastdfe

# create config object
config = fastdfe.Config.from_file(config_file)

# merge config with new options
config.data |= opts

# save config to file
config.to_file(out)
