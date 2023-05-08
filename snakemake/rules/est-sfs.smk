"""
Module for running EST-SFS on a set of VCF files.
EST-SFS is limited by the number of sites it can handle,
so we scatter and gather the input files so as not
to exceed this limit (given by ``max_sites``).
The EST-SFS are automatically fetched and installed.
This workflow only infers ancestral allele of bi-allelic sites that
have at least `n_outgroup` called haplotypes. This is to make sure
enough information is available to reliably infer the ancestral allele.
The `AA` info tag is added to the VCF files to indicate the ancestral allele.
If not enough outgroup haplotypes are available, `AA` is set to '.'.
The additional info tags `EST_SFS_input` and `EST_SFS_output` denote the
EST-SFS input and output as described in the manual.

Note that EST-SFS only runs on Linux and that the wildcard values specifying
the set of VCF files need to be specified explicitly.

config example:

The path names support wildcards which need to be specified explicitly.

``{
    'vcf_in': "vcf/all/{chunk}.{opts}.vcf.gz", # input VCF
    'vcf_out': "vcf/all/{chunk}.{opts}.polarized.vcf.gz", # annotated output VCF where 'AA' is added to the INFO field
    'wildcards': {'chunk': chunks_vcf}, # chunks_vcf is a list of chunk names
    'ingroups': "sample_lists/ingroups.args", # sample lists for ingroups separated by new lines
    'outgroups': "sample_lists/outgroups.args", # sample lists for outgroups separated by new lines
    'basepath': "est-sfs/{opts}", # base path for EST-SFS output
    'basepath_vcf': "est-sfs/{chunk}.{opts}" # base path for EST-SFS input

    # maximum number of sites used for EST-SFS
    # this number is hard-coded and has to be replaced at compilation time
    # the compilation seems to fail for values larger than around 1000000
    'max_sites': 1000000

    # maximum number of samples
    # EST-SFS doesn't allow more than 200 alleles (including the outgroups)
    # a size of 50 produces an SFS sufficiently smooth
    'n_samples': 50

    # Number of outgroup haplotypes used, max 3.
    # Note that even one outgroup individual may have more
    # one haplotype depending on its ploidy. The haplotypes
    # are randomly sampled from the given set of outgroups species.
    'n_outgroup': 3

    'model': 1 # substitution model used for EST-SFS
    'nrandom': 10 # number of ML runs
    'debug': False # whether to print local variables in this snakefile
    'seed': 0 # seed for random number generator

    'log_level' # log level for snakemake
}``
"""
from typing import List

import numpy as np

basepath = config['basepath']
basepath_vcf = config['basepath_vcf']

if 'debug' in config and config['debug']:
    print(locals())

# create a config file for EST-SFS
rule setup_config_est_sfs:
    output:
        "resources/est-sfs/config.txt"
    conda:
        "../envs/est-sfs.yaml"
    script:
        "../scripts/setup_config_est_sfs.py"

# create a file containing a seed
rule create_seed_file_est_sfs:
    output:
        basepath + ".seed.{i}.txt"
    conda:
        "../envs/est-sfs.yaml"
    script:
        "../scripts/create_seed_file_est_sfs.py"

# download and install EST-SFS
rule install_est_sfs:
    input:
        "resources/est-sfs/config.txt"
    output:
        "resources/est-sfs/bin"
    conda:
        "../envs/est-sfs.yaml"
    script:
        "../scripts/install_est_sfs.py"

# prepare the input files for EST-SFS
rule prepare_input_est_sfs:
    input:
        ingroups=config['ingroups'],
        outgroups=config['outgroups'],
        vcf=config['vcf_in']
    output:
        data=basepath_vcf + '.data.txt'
    conda:
        "../envs/est-sfs.yaml"
    script:
        "../scripts/prepare_input_est_sfs.py"

# EST-SFS needs the information of all sites to correctly
# predict the ancestral state. If used on a small subset, the high variation
# of frequencies seems to inflate the number of high frequency derived alleles.
# We thus gather all subsets here and split the result again for
# further processing.
checkpoint gather_input_est_sfs:
    input:
        expand(basepath_vcf + '.data.txt',**config['wildcards'],allow_missing=True)
    output:
        basepath + ".data.all.txt"
    conda:
        "../envs/est-sfs.yaml"
    script:
        "../scripts/gather_input_est_sfs.py"

# scatter the data files again into chunk sufficiently small
# to be passed to EST-SFS
rule scatter_data_est_sfs:
    input:
        basepath + ".data.all.txt"
    output:
        basepath + ".data.{i}.{n_chunks}.txt"
    params:
        n_chunks=lambda w: int(w.n_chunks),
        i=lambda w: int(w.i)
    conda:
        "../envs/est-sfs.yaml"
    script:
        "../scripts/scatter_lines.jl"

# determine the ancestral alleles using the outgroups
rule derive_ancestral_alleles:
    input:
        data=basepath + ".data.{i}.{n_chunks}.txt",
        seed=ancient(basepath + ".seed.{i}.txt"),
        config=rules.setup_config_est_sfs.output[0],
        bin=rules.install_est_sfs.output[0]
    output:
        sfs=basepath + ".sfs.{i}.{n_chunks}.txt",
        probs=basepath + ".probs.comments.{i}.{n_chunks}.txt"
    conda:
        "../envs/est-sfs.yaml"
    script:
        "../scripts/derive_ancestral_alleles.py"

# strip the comment lines from the output file
rule strip_comment_lines:
    input:
        basepath + ".probs.comments.{i}.{n_chunks}.txt"
    output:
        basepath + ".probs.{i,\d+}.{n_chunks,\d+}.txt"
    shell:
        "grep -v '^0' {input} > {output}"


# we need to
def get_n_chunks(w) -> int:
    """
    Determine the number of partitions used for EST-SFS
    depending on the maximum number of sites allowed per run.

    :param w: Wildcards
    :return: Number of partitions
    """
    data_file = checkpoints.gather_input_est_sfs.get(**w).output[0]

    n_lines = sum(1 for _ in open(data_file))
    return int(np.ceil(n_lines / config['max_sites']))


# get the names of chunked files containing the probabilities
def get_partitions(w) -> List[str]:
    """
    Get the names of chunked files containing the probabilities.

    :param w: Wildcards
    :return: List of file names
    """
    n_chunks = get_n_chunks(w)

    return [basepath + f".probs.{i}.{n_chunks}.txt" for i in range(1,n_chunks + 1)]


# gather the output probabilities
rule gather_props_est_sfs:
    input:
        get_partitions
    output:
        basepath + ".probs.txt"
    params:
        n_chunks=lambda w: get_n_chunks(w)
    conda:
        "../envs/est-sfs.yaml"
    script:
        "../scripts/gather_lines.jl"

# split gather EST-SFS output into original chunks
rule scatter_output_est_sfs:
    input:
        probs=rules.gather_props_est_sfs.output[0],
        data=expand(basepath_vcf + '.data.txt',**config['wildcards'],allow_missing=True)
    output:
        probs=expand(basepath_vcf + '.probs.txt',**config['wildcards'],allow_missing=True)
    conda:
        "../envs/est-sfs.yaml"
    script:
        "../scripts/scatter_output_est_sfs.py"

# correct the ancestral alleles
# here gzipped output files are not supported
# so we zip them later
rule recode_ancestral_alleles_vcf:
    input:
        vcf=config['vcf_in'],
        probs=basepath_vcf + '.probs.txt',
        data=basepath_vcf + '.data.txt',
        ingroups=config['ingroups'],
        outgroups=config['outgroups']
    log:
        basepath_vcf + '.log'
    output:
        vcf=temp(config['vcf_out'].replace('.gz',''))
    conda:
        "../envs/est-sfs.yaml"
    script:
        "../scripts/recode_ancestral_alleles_vcf.py"

# import rule to gzip output VCF files
module zip:
    snakefile:
        "zip.smk"
    config:
        config

# compress the output VCF file
use rule compress_bgzip from zip as compress_vcf with:
    input:
        rules.recode_ancestral_alleles_vcf.output.vcf
    output:
        config['vcf_out']
