"""
Predict the variant effects with VEP.
"""

__author__ = "Janek Sendrowski"
__contact__ = "j.sendrowski18@gmail.com"
__date__ = "2023-05-13"

from snakemake.shell import shell

ref = snakemake.input.ref
vcf = snakemake.input.vcf
gff = snakemake.input.gff
species = snakemake.params.species
out = snakemake.output[0]

shell(
    f"vep "
    f"--gff {gff} "
    f"--fasta {ref} "
    f"-i {vcf} "
    f"--vcf "
    f"--species {species} "
    f"--allow_non_variant "
    f"-o {out} "
    f"--fields 'Consequence,Codons' "
    f"--compress_output bgzip"
)
