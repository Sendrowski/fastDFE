"""
Run the GATK.
"""

__author__ = "Janek Sendrowski"
__contact__ = "j.sendrowski18@gmail.com"
__date__ = "2022-05-31"

from snakemake.shell import shell
from snakemake_wrapper_utils.java import get_java_opts

java_opts = get_java_opts(snakemake)
flags = snakemake.params.flags
command = snakemake.params.command

cmd = f"gatk --java-options '{java_opts}' {command}"

if '-O' not in flags:
    flags['-O'] = snakemake.output[0]

for flag, val in flags.items():
    cmd += f" {flag} {val}"

shell(cmd)
