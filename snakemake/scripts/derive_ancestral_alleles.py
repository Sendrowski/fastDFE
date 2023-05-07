"""
Derive the ancestral alleles using EST-SFS.
"""

__author__ = "Janek Sendrowski"
__contact__ = "j.sendrowski18@gmail.com"
__date__ = "2022-05-31"

from snakemake.shell import shell

data = snakemake.input.data
seed = snakemake.input.seed
config = snakemake.input.config
bin = snakemake.input.bin
out_sfs = snakemake.output.sfs
out_probs = snakemake.output.probs
tmp_dir = snakemake.resources.tmpdir

# execute command
shell(f"{bin} {config} {data} {seed} {out_sfs} {out_probs}")
