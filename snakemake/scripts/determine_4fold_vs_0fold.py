"""
Determine the ratio of 4-fold to 0-fold sites in a genome by using TargetSiteCounter.
"""

import fastdfe as fd

p = fd.Parser(
    n=20,  # number of individuals not important
    # choose sufficiently many sites since TargetSiteCounter is restricted to the same genomic interval
    max_sites=10000,
    vcf="https://github.com/Sendrowski/fastDFE/blob/dev/resources/"
        "genome/betula/biallelic.polarized.subset.50000.vcf.gz?raw=true",
    fasta="https://github.com/Sendrowski/fastDFE/blob/dev/resources/"
          "genome/betula/genome.subset.1000.fasta.gz?raw=true",
    gff="https://github.com/Sendrowski/fastDFE/blob/dev/resources/"
        "genome/betula/genome.gff.gz?raw=true",
    target_site_counter=fd.TargetSiteCounter(
        n_target_sites=100000,  # doesn't matter since we're only interested in the ratio
        n_samples=10000  # choose sufficiently many samples for the ratio to be accurate
    ),
    stratifications=[fd.DegeneracyStratification()],  # stratify by 4-fold and 0-fold sites
    annotations=[fd.DegeneracyAnnotation()],  # annotate sites with degeneracy on-the-fly
    filtrations=[fd.SNPFiltration()],
)

sfs = p.parse()

r = sfs['neutral'].data[0] / sfs['selected'].data[0]

pass
