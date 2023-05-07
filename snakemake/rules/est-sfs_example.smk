"""
Example configuration file for EST-SFS module.
"""
import logging

rule all:
    input:
        "../resources/genome/betula/all.with_outgroups.subset.10000.polarized.vcf.gz"

# load EST-SFS module
module est_sfs:
    snakefile:
        "est-sfs.smk"
    config:
        {
            'vcf_in': "../resources/genome/betula/all.with_outgroups.subset.10000.vcf.gz",
            'vcf_out': "../resources/genome/betula/all.with_outgroups.subset.10000.polarized.vcf.gz",
            'wildcards': {},
            'ingroups': "../resources/genome/betula/sample_sets/birch.args",
            'outgroups': "../resources/genome/betula/sample_sets/outgroups.args",
            'basepath': "results/est-sfs/",
            'basepath_vcf': "results/est-sfs/",
            'max_sites': 1000000,
            'n_samples': 50,
            'n_outgroup': 3,
            'model': 1,
            'nrandom': 10,
            'log_level': logging.INFO,
        }

# load all rules from EST-SFS module
use rule * from est_sfs as *
