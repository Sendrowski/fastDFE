"""
Convert VCF file to plink format.
"""

__author__ = "Janek Sendrowski"
__contact__ = "j.sendrowski18@gmail.com"
__date__ = "2022-05-31"

from snakemake.shell import shell

vcf = snakemake.input[0]
out = snakemake.output[0]
prefix = out[:out.rfind(".")]

# create bed, bim and fam files
shell(f"plink --vcf {vcf} --out {prefix} --allow-extra-chr")
