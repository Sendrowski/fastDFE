"""
Infer the DFE from the SFS.
"""

__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"
__date__ = "2023-05-25"

try:
    import sys

    # necessary to import dfe module
    sys.path.append('..')
    testing = False
    vcf_file = snakemake.input.vcf
    fasta_file = snakemake.input.fasta
    gff_file = snakemake.input.gff
    samples_file = snakemake.input.samples
    profile = snakemake.params.profile
    out_csv = snakemake.output.csv
    out_png = snakemake.output.png
    aliases = snakemake.params.aliases
    n_target_sites = snakemake.params.get('n_target_sites', None)
    n = snakemake.params.get('n', 20)
except NameError:
    # testing
    testing = True
    chr = 21
    vcf_file = f"results/vcf/hgdp/21/opts.vcf.gz"
    fasta_file = f"results/fasta/hgdp/{chr}.fasta.gz"
    gff_file = f"results/gff/hgdp/{chr}.gff3.gz"
    samples_file = "results/sample_lists/hgdp/French.args"
    profile = "default"
    out_csv = "scratch/parse_spectra_from_url.spectra.csv"
    out_png = "scratch/parse_spectra_from_url.spectra.png"
    aliases = {f"chr{chr}": [f"{chr}"]}
    n_target_sites = None
    n = 20

import pandas as pd

from fastdfe import Parser, CodingSequenceFiltration, VEPStratification, \
    SNPFiltration, PolyAllelicFiltration, SynonymyAnnotation, SynonymyStratification

# setup parser
p = dict(
    default=Parser(
        vcf=vcf_file,
        n=n,
        annotations=[
            SynonymyAnnotation(
                fasta_file=fasta_file,
                gff_file=gff_file,
                aliases=aliases
            )
        ],
        filtrations=[
            CodingSequenceFiltration(
                gff_file=gff_file,
                aliases=aliases
            ),
            SNPFiltration(),
            PolyAllelicFiltration()
        ],
        stratifications=[SynonymyStratification()],
        samples=pd.read_csv(samples_file).iloc[:, 0].tolist(),
        info_ancestral='AA_ensembl'
    ),
    vep=Parser(
        vcf=vcf_file,
        n=n,
        filtrations=[],
        stratifications=[VEPStratification()],
        samples=pd.read_csv(samples_file).iloc[:, 0].tolist(),
        info_ancestral='AA_ensembl'
    )
)[profile]

# parse spectra
spectra = p.parse()

# save to file
spectra.to_file(out_csv)

# plot data and save to file
spectra.plot(show=testing, file=out_png)

pass
