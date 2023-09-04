"""
Parse SFS from HGDP VCF.
"""

__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"
__date__ = "2023-05-25"

import numpy as np

try:
    import sys

    # necessary to import dfe module
    sys.path.append('..')
    testing = False
    chr = snakemake.params.chr
    vcf_file = snakemake.input.vcf
    fasta = snakemake.input.fasta
    gff = snakemake.input.gff
    samples_file = snakemake.input.samples
    out_csv = snakemake.output.csv
    out_png = snakemake.output.png
    aliases = snakemake.params.aliases
    max_sites = snakemake.params.get('max_sites', np.inf)
    n_samples = snakemake.params.get('n_samples', 1000000)
    n = snakemake.params.get('n', 20)
except NameError:
    # testing
    testing = True
    chr = "9"
    vcf_file = f"results/vcf/hgdp/{chr}/opts.vcf.gz"
    fasta = f"results/fasta/hgdp/{chr}.fasta.gz"
    gff = f"results/gff/hgdp/{chr}.corrected.gff3.gz"
    samples_file = "results/sample_lists/hgdp/all.args"
    out_csv = "scratch/parse_csv_hgdp.spectra.csv"
    out_png = "scratch/parse_csv_hgdp.spectra.png"
    aliases = {f"chr{chr}": [chr]}
    max_sites = 10000
    n_samples = 10000
    n = 20

import pandas as pd

import fastdfe as fd

samples = pd.read_csv(samples_file).iloc[:, 0].tolist()

# setup parser
p = fd.Parser(
    vcf=vcf_file,
    fasta=fasta,
    gff=gff,
    aliases=aliases,
    max_sites=max_sites,
    target_site_counter=fd.TargetSiteCounter(
        n_samples=n_samples,
        n_target_sites=fd.Annotation.count_target_sites(
            file=gff
        )[chr]
    ),
    n=n,
    annotations=[fd.DegeneracyAnnotation()],
    filtrations=[
        fd.CodingSequenceFiltration(),
        fd.SNVFiltration(),
        fd.PolyAllelicFiltration()
    ],
    stratifications=[fd.DegeneracyStratification()],
    include_samples=samples,
    info_ancestral='AA_ensembl'
)

# parse spectra
spectra = p.parse()

# save to file
spectra.to_file(out_csv)

# plot data and save to file
spectra.plot(show=testing, file=out_png)

pass
