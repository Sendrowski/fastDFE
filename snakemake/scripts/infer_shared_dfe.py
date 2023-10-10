"""
Perform joint inference.
"""

__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"
__date__ = "2023-03-19"

try:
    import sys

    # necessary to import fastdfe locally
    sys.path.append('..')
    testing = False
    config_file = snakemake.input[0]
    out_summary = snakemake.output.summary
    out_serialized = snakemake.output.serialized
except NameError:
    # testing
    testing = True
    # config_file = 'resources/configs/shared/covariates_dummy_example_1/config.yaml'
    # config_file = 'results/configs/pendula.pubescens.example_1.example_2.example_3_C_full_anc/config.yaml'
    config_file = '../resources/configs/shared/pendula_betula_tutorial/config.yaml'
    out_summary = "scratch/summary.json"
    out_serialized = "scratch/serialized_shared.json"

import logging

from fastdfe import Config, JointInference

# set log level to debug
logging.getLogger('fastdfe').setLevel(logging.INFO)

# load config from file
config = Config.from_file(config_file)

# update config
"""config.update(
    model=DiscreteParametrization(),
    fixed_params={},
    shared_params=[SharedParams(types='all', params=['S2'])],
    covariates=[],
    parallelize=False,
    n_runs=1
)"""

# create from config
inference = JointInference.from_config(config)

# perform inference
inference.run()

# bootstrap
# inference.bootstrap(10)

# save object in serialized form
inference.to_file(out_serialized)

# save summary
inference.get_summary().to_file(out_summary)

if testing:
    inference.plot_discretized()
    inference.plot_continuous()
    inference.plot_inferred_parameters()
    inference.perform_lrt_covariates()
    inference.perform_lrt_shared()

pass
