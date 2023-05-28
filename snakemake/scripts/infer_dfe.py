"""
Infer the DFE from the SFS.
"""

__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"
__date__ = "2023-02-26"

try:
    import sys

    # necessary to import dfe module
    sys.path.append('..')
    testing = False
    config_file = snakemake.input[0]
    out_summary = snakemake.output.summary
    out_serialized = snakemake.output.serialized
except NameError:
    # testing
    testing = True
    # config_file = 'results/configs/example_1_C_full_anc/config.yaml'
    # config_file = 'results/configs/example_1_C_deleterious_anc_bootstrapped_100/config.yaml'
    # config_file = 'results/configs/pendula_C_full_anc_bootstrapped_100/config.yaml'
    config_file = '../resources/configs/shared/pendula_tutorial/config.yaml'
    out_summary = "scratch/summary.json"
    out_serialized = "scratch/serialized.json"

    import logging

    # set log level to debug
    logging.getLogger('fastdfe').setLevel(logging.INFO)

from fastdfe.parametrization import GammaExpParametrization, DiscreteParametrization, DisplacedGammaParametrization, \
    GammaDiscreteParametrization
from fastdfe import Config, BaseInference

# load config from file
config = Config.from_file(config_file)
"""config.update(
    parallelize=False,
    model=DiscreteParametrization(),
    n_runs=20
)"""

# create from config
inference = BaseInference.from_config(config)

# perform inference
inference.run()

# save object in serialized form
inference.to_file(out_serialized)

# save summary
inference.get_summary().to_file(out_summary)

if testing:
    inference.plot_all()
    # inference.plot_sfs_comparison()
    # inference.plot_dfe_continuous()

pass
