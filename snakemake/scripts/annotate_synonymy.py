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
    fasta_file = snakemake.input.ref
    gff_file = snakemake.input.gff
    aliases = snakemake.params.aliases
    out = snakemake.output[0]
except ModuleNotFoundError:
    # testing
    testing = True
    vcf_file = "https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/1000_genomes_project/release/" \
               "20181203_biallelic_SNV/ALL.chr21.shapeit2_integrated_v1a.GRCh38.20181129.phased.vcf.gz"
    fasta_file = "https://ftp.ensembl.org/pub/release-109/fasta/homo_sapiens/" \
                 "dna/Homo_sapiens.GRCh38.dna.chromosome.21.fa.gz"
    gff_file = "https://ftp.ensembl.org/pub/release-109/gff3/homo_sapiens/" \
               "Homo_sapiens.GRCh38.109.chromosome.21.gff3.gz"
    aliases=dict(chr21=['21'])
    out = 'scratch/synonymy.vcf'

from fastdfe import Annotator, SynonymyAnnotation
import logging

logging.getLogger('fastdfe').setLevel(logging.DEBUG)

# initialize annotator
ann = Annotator(
    vcf=vcf_file,
    output=out,
    annotations=[
        SynonymyAnnotation(
            fasta_file=fasta_file,
            gff_file=gff_file,
            aliases=aliases
        )
    ],
)

# run annotator
ann.annotate()

pass
