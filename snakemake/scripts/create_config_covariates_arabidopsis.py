"""
Create config file for joint inference of Arabidopsis thaliana.
"""

__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"
__date__ = "2023-06-06"

import pandas as pd

try:
    import sys

    # necessary to import dfe module
    sys.path.append('..')
    testing = False
    sfs_file = snakemake.input[0]
    out = snakemake.output[0]
except NameError:
    # testing
    testing = True
    sfs_file = "results/sfs_covariates/arabidopsis.csv"
    out = "scratch/joint_inference.yaml"

from fastdfe import Config

df = pd.read_csv(sfs_file, sep="\t")

config = Config()

config.to_file(out)
