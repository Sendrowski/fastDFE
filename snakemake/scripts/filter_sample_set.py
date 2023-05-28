"""
Filter the given sample set to satisfy `samples[filter_col] == filter_val`.
"""

__author__ = "Janek Sendrowski"
__contact__ = "j.sendrowski18@gmail.com"
__date__ = "2022-05-31"

import pandas as pd

try:
    samples_file = snakemake.input[0]
    filter_col = snakemake.params.get('filter_col', 'species')
    filter_val = snakemake.params.filter_val
    sep = snakemake.params.get('sep', '\t')
    out = snakemake.output[0]
except NameError:
    samples_file = "resources/quercus/ingroups.csv"
    filter_col = 'species'
    filter_val = 'Qa'
    sep = '\t'
    out = "scratch/acutissima.csv"

samples = pd.read_csv(samples_file, sep=sep)

samples[samples[filter_col] == filter_val].to_csv(out, index=False, sep=sep)
