"""
Remove polarization from VCF file.
"""

__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"
__date__ = "2023-12-09"

from cyvcf2 import VCF, Writer
from tqdm import tqdm

try:
    from snakemake.shell import shell

    import sys

    # necessary to import dfe module
    sys.path.append('..')

    testing = False
    vcf_in = snakemake.input.vcf
    tags = snakemake.params.tags
    vcf_out = snakemake.output.vcf
except ModuleNotFoundError:
    # testing
    testing = True
    vcf_in = "resources/genome/betula/all.with_outgroups.subset.10000.vcf.gz"
    tags = ["AA", "Degeneracy", "Degeneracy_Info", "Synonymy", "Synonymy_Info", "EST_SFS_input", "EST_SFS_probs"]
    vcf_out = "resources/genome/betula/all.with_outgroups.subset.10000.vcf.gz.new"

# remove polarization (AA tag) using cyvcf2
vcf = VCF(vcf_in)

writer = Writer(vcf_out, vcf)

# remove polarization
for v in tqdm(vcf):
    for tag in tags:
        del v.INFO[tag]

    writer.write_record(v)

pass
