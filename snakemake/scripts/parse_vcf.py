"""
Parse a VCF file.
"""

__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"
__date__ = "2023-29-03"

import numpy as np

try:
    import sys

    # necessary to import fastdfe locally
    sys.path.append('..')

    testing = False
    vcf_file = snakemake.input.vcf
    fasta = snakemake.input.ref
    n = snakemake.params.n
    max_sites = snakemake.params.get('max_sites', np.inf)
    stratifications = snakemake.params.get('stratifications', ['DegeneracyStratification'])
    out = snakemake.output[0]
except NameError:

    # testing
    testing = True
    vcf_file = 'results/vcf/betula/vcf/all.vcf.gz'
    fasta = '../resources/genome/betula/genome.fasta'
    n = 20
    max_sites = np.inf
    stratifications = [
        'DegeneracyStratification',
        'AncestralBaseStratification'
    ]
    out = "scratch/sfs_parsed.csv"

import fastdfe as fd

# instantiate stratifications
for i, s in enumerate(stratifications):
    if s == 'BaseContextStratification':
        stratifications[i] = fd.BaseContextStratification(fasta=fasta)
    else:
        stratifications[i] = getattr(fastdfe.parser, s)()

# instantiate parser
p = fd.Parser(
    n=n,
    vcf=vcf_file,
    max_sites=max_sites,
    stratifications=stratifications
)

# parse SFS
sfs = p.parse()

# save to file
sfs.to_file(out)

pass
