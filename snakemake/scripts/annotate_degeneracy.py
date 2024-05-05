"""
Annotate degeneracy of variants in a VCF file.
"""

__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"
__date__ = "2023-05-12"

try:
    import sys

    # necessary to import fastdfe locally
    sys.path.append('..')

    testing = False
    vcf_file = snakemake.input.vcf
    fasta = snakemake.input.ref
    gff = snakemake.input.gff
    out = snakemake.output[0]
except ModuleNotFoundError:
    # testing
    testing = True
    vcf_file = "resources/genome/betula/all.polarized.subset.200000.vcf.gz"
    fasta = "resources/genome/betula/genome.fasta"
    gff = "resources/genome/betula/genome.gff.gz"
    out = 'resources/genome/betula/all.polarized.deg.subset.200000.vcf.gz'

import fastdfe as fd

fd.logger.setLevel('DEBUG')

# initialize annotator
ann = fd.Annotator(
    vcf=vcf_file,
    output=out,
    fasta=fasta,
    gff=gff,
    annotations=[fd.DegeneracyAnnotation()],
)

# run annotator
ann.annotate()
