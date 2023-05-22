"""
Predict the variant effects with SNPEff.
"""

__author__ = "Janek Sendrowski"
__contact__ = "j.sendrowski18@gmail.com"
__date__ = "2023-05-21"

from snakemake.shell import shell

vcf = snakemake.input.vcf
species = snakemake.params.species
out = snakemake.output[0]

# download database
shell("snpEff download {species}")

# annotate
shell(
    f"snpEff "
    f"-v {species} "
    f"{vcf} "
    f"| bgzip > {out}"
)
