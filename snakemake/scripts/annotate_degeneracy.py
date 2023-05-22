"""
Run the annotator on a VCF file.
"""

__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"
__date__ = "2023-05-12"

try:
    from snakemake.shell import shell

    import sys

    # necessary to import dfe module
    sys.path.append('..')

    testing = False
    vcf_file = snakemake.input.vcf
    fasta_file = snakemake.input._ref
    gff_file = snakemake.input.gff
    out = snakemake.output[0]
except ModuleNotFoundError:
    # testing
    testing = True
    vcf_file = "resources/genome/betula/all.vcf.gz"
    fasta_file = "resources/genome/betula/genome.fasta"
    gff_file = "resources/genome/betula/genome.gff.gz"
    out = 'scratch/degeneracy.vcf'

from fastdfe import Annotator, DegeneracyAnnotation
import logging

logging.getLogger('fastdfe').setLevel(logging.DEBUG)

# initialize annotator
ann = Annotator(
    vcf=vcf_file,
    output=out,
    annotations=[
        DegeneracyAnnotation(
            fasta_file=fasta_file,
            gff_file=gff_file
        )
    ],
)

# run annotator
ann.annotate()
