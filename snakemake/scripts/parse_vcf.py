"""
Parse a VCF file.
"""

__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"
__date__ = "2023-29-03"

import numpy as np

try:
    import sys

    # necessary to import dfe module
    sys.path.append('..')

    testing = False
    vcf_file = snakemake.input.vcf
    fasta_file = snakemake.input.ref
    n = snakemake.params.n
    max_sites = snakemake.params.get('max_sites', np.inf)
    stratifications = snakemake.params.get('stratifications', ['DegeneracyStratification'])
    out = snakemake.output[0]
except NameError:

    # testing
    testing = True
    vcf_file = 'results/vcf/betula/vcf/all.vcf.gz'
    fasta_file = '../resources/genome/betula/genome.fasta'
    n = 20
    max_sites = np.inf
    stratifications = [
        'DegeneracyStratification',
        'ReferenceBaseStratification'
    ]
    out = "scratch/sfs_parsed.csv"

import fastdfe
from fastdfe import Parser
from fastdfe.parser import BaseContextStratification

# instantiate stratifications
for i, s in enumerate(stratifications):
    if s == 'BaseContextStratification':
        stratifications[i] = BaseContextStratification(fasta_file=fasta_file)
    else:
        stratifications[i] = getattr(fastdfe.parser, s)()

# instantiate parser
p = Parser(
    n=n,
    vcf_file=vcf_file,
    max_sites=max_sites,
    stratifications=stratifications
)

# parse SFS
sfs = p.parse()

# save to file
sfs.to_file(out)

pass
