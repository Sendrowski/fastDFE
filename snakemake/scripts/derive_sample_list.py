"""
Derive a with sample names separated by new lines.
"""

__author__ = "Janek Sendrowski"
__contact__ = "j.sendrowski18@gmail.com"
__date__ = "2022-05-31"

import pandas as pd

try:
    samples_file = snakemake.input[0]
    name_col = snakemake.params.get('name_col', 'name')
    sep = snakemake.params.get('sep', '\t')
    out = snakemake.output[0]
except NameError:
    samples_file = "results/sample_sets/hgdp/Maya.csv"
    sep = snakemake.params.get('sep', '\t')
    name_col = 'sample'
    out = "scratch/all.args"

samples = pd.read_csv(samples_file, sep=sep)

samples[name_col].to_csv(out, header=False, index=False, sep=sep)
