"""
Set up the config for est-sfs.
We use the substitution model (1) recommended by the paper.
"""

__author__ = "Janek Sendrowski"
__contact__ = "j.sendrowski18@gmail.com"
__date__ = "2022-05-31"

try:
    out = snakemake.output[0]
    config = snakemake.config
    max_sites = snakemake.config['max_sites']
except NameError:
    # testing
    out = "scratch/est-sfs.config"
    config = {
        'n_outgroup': 2,
        'model': 1,
        'nrandom': 5
    }

with open(out, 'w') as f:
    for param in ["n_outgroup", "model", 'nrandom']:
        f.write(f"{param} {config[param]}\n")
