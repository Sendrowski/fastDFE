"""
Parse betula VCF.
"""

__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"
__date__ = "2023-08-06"

import pandas as pd

try:
    import sys

    # necessary to import dfe module
    sys.path.append('..')

    testing = False
    vcf_file = snakemake.input.vcf
    fasta_file = snakemake.input.ref
    gff_file = snakemake.input.gff
    samples_file = snakemake.input.samples
    n = snakemake.params.n
    out_csv = snakemake.output.csv
    out_png = snakemake.output.png
except NameError:

    # testing
    testing = True
    vcf_file = '../resources/genome/betula/all.vcf.gz'
    fasta_file = '../resources/genome/betula/genome.fasta'
    gff_file = '../resources/genome/betula/genome.gff.gz'
    samples_file = '../resources/genome/betula/sample_sets/pendula.args'
    n = 20
    out_csv = "scratch/sfs_parsed.csv"
    out_png = "scratch/sfs_parsed.png"

from fastdfe import Parser, CodingSequenceFiltration, DegeneracyAnnotation, DegeneracyStratification

p = Parser(
    vcf=vcf_file,
    n=20,
    annotations=[
        DegeneracyAnnotation(
            fasta_file=fasta_file,
            gff_file=gff_file
        )
    ],
    filtrations=[
        CodingSequenceFiltration(
            gff_file=gff_file
        )
    ],
    stratifications=[DegeneracyStratification()],
    samples=pd.read_csv(samples_file).iloc[:, 0].tolist()
)

sfs = p.parse()

sfs.plot(show=testing, file=out_png)

sfs.to_file(out_csv)
