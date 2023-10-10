"""
Parse betula VCF.
"""

__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"
__date__ = "2023-08-06"

import pandas as pd

try:
    import sys

    # necessary to import fastdfe locally
    sys.path.append('..')

    testing = False
    vcf_file = snakemake.input.vcf
    fasta = snakemake.input.ref
    gff = snakemake.input.gff
    samples_file = snakemake.input.samples
    max_sites = snakemake.params.max_sites.get("max_sites", np.inf)
    n = snakemake.params.n
    out_csv = snakemake.output.csv
    out_png = snakemake.output.png
except NameError:

    # testing
    testing = True
    vcf_file = '../resources/genome/betula/all.with_outgroups.vcf.gz'
    fasta = '../resources/genome/betula/genome.fasta'
    gff = '../resources/genome/betula/genome.gff.gz'
    samples_file = '../resources/genome/betula/sample_sets/pendula.args'
    max_sites = 100000
    n = 10
    out_csv = "scratch/sfs_parsed.csv"
    out_png = "scratch/sfs_parsed.png"

import fastdfe as fd

ingroups = pd.read_csv(samples_file).iloc[:, 0].tolist()

p = fd.Parser(
    vcf=vcf_file,
    fasta=fasta,
    gff=gff,
    n=n,
    annotations=[
        fd.DegeneracyAnnotation(),
        fd.MaximumLikelihoodAncestralAnnotation(
            n_ingroups=20,
            ingroups=ingroups,
            outgroups=["ERR2103730"],
            max_sites=10000
        )
    ],
    filtrations=[
        fd.CodingSequenceFiltration(),
        fd.PolyAllelicFiltration(),
    ],
    stratifications=[fd.DegeneracyStratification()],
    include_samples=ingroups,
    max_sites=max_sites
)

sfs = p.parse()

sfs.plot(show=testing, file=out_png)

sfs.to_file(out_csv)

pass
