"""
Parse SFS from HGDP VCF.
"""

__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"
__date__ = "2023-05-25"

try:
    import sys

    # necessary to import dfe module
    sys.path.append('..')
    testing = False
    chr = snakemake.params.chr
    vcf_file = snakemake.input.vcf
    fasta_file = snakemake.input.fasta
    gff_file = snakemake.input.gff
    samples_file = snakemake.input.samples
    out_csv = snakemake.output.csv
    out_png = snakemake.output.png
    aliases = snakemake.params.aliases
    n = snakemake.params.get('n', 20)
except NameError:
    # testing
    testing = True
    chr = "1"
    vcf_file = f"results/vcf/hgdp/{chr}/opts.vcf.gz"
    fasta_file = f"results/fasta/hgdp/{chr}.fasta.gz"
    gff_file = f"results/gff/hgdp/{chr}.corrected.gff3.gz"
    samples_file = "results/sample_lists/hgdp/all.args"
    out_csv = "scratch/parse_csv_hgdp.spectra.csv"
    out_png = "scratch/parse_csv_hgdp.spectra.png"
    aliases = {f"chr{chr}": [chr]}
    n = 15

import pandas as pd

import fastdfe as fd

samples = pd.read_csv(samples_file).iloc[:, 0].tolist()

# setup parser
p = fd.Parser(
    vcf=vcf_file,
    target_site_counter=fd.TargetSiteCounter(
        fasta_file=fasta_file,
        aliases=aliases,
        n_samples=1000000,
        n_target_sites=fd.Annotation.count_target_sites(
            file=gff_file
        )[chr]
    ),
    n=n,
    annotations=[
        fd.DegeneracyAnnotation(
            fasta_file=fasta_file,
            gff_file=gff_file,
            aliases=aliases
        )
    ],
    filtrations=[
        fd.CodingSequenceFiltration(
            gff_file=gff_file,
            aliases=aliases
        ),
        fd.SNVFiltration(),
        fd.PolyAllelicFiltration()
    ],
    stratifications=[fd.DegeneracyStratification()],
    samples=samples,
    info_ancestral='AA_ensembl',
)

# parse spectra
spectra = p.parse()

# save to file
spectra.to_file(out_csv)

# plot data and save to file
spectra.plot(show=testing, file=out_png)

pass
