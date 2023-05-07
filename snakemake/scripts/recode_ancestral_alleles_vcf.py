"""
Annotate the VCF with ancestral allele information.
An AA info tag is added in addition to further information.
"""

__author__ = "Janek Sendrowski"
__contact__ = "j.sendrowski18@gmail.com"
__date__ = "2022-05-31"

import logging
import numpy as np
import sys
from collections import Counter
from tqdm import tqdm

import pandas as pd
from cyvcf2 import VCF, Writer

try:
    vcf_file = snakemake.input.vcf
    probs = snakemake.input.probs
    data = snakemake.input.data
    samples_file = snakemake.input.samples
    log = snakemake.log[0]
    out = snakemake.output.vcf
    log_level = snakemake.config.get('log_level', 20)
except NameError:
    # testing
    vcf_file = "../resources/genome/betula/all.with_outgroups.subset.10000.vcf.gz"
    probs = "results/est-sfs/.probs.txt"
    data = "results/est-sfs/.data.txt"
    samples_file = "../resources/genome/betula/sample_sets/birch.args"
    log = "scratch/est-sfs.log"
    out = "scratch/1.polarized.vcf"
    log_level = logging.DEBUG

# configure logger to log stdout
logger = logging.getLogger()

# add file handler
file_handler = logging.FileHandler(log)
file_handler.setLevel(logging.DEBUG)
logger.addHandler(file_handler)

# add stream handler
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(log_level)
logger.addHandler(stream_handler)


def get_called_bases(calls: np.ndarray) -> np.ndarray:
    """
    Get the called bases from a list of calls.

    :param calls: Array of calls.
    :return: Array of called bases.
    """
    return np.array([b for b in '/'.join(calls).replace('|', '/') if b in 'ACGT'])


samples = pd.read_csv(samples_file, header=None, index_col=False)[0].tolist()

vcf_reader = VCF(vcf_file)

# Add AA info field to the header
vcf_reader.add_info_to_header({'ID': 'AA', 'Number': 1, 'Type': 'String', 'Description': 'Ancestral Allele'})
vcf_reader.add_info_to_header(
    {'ID': 'EST_SFS_output', 'Number': 1, 'Type': 'String', 'Description': 'EST-SFS probabilities'})
vcf_reader.add_info_to_header({'ID': 'EST_SFS_input', 'Number': 1, 'Type': 'String', 'Description': 'EST-SFS input'})

probs_reader = open(probs, 'r')
data_reader = open(data, 'r')
writer = Writer(out, vcf_reader)

# get a list of ingroup samples
ingroups = pd.read_csv(samples_file, header=None, index_col=False)[0].tolist()

# get indices of ingroup samples
ingroup_mask = [sample in ingroups for sample in vcf_reader.samples]

# write to data file
for variant in tqdm(vcf_reader):

    # Only do inference for bi-allelic SNPs.
    # Since EST-SFS only reports the probability of
    # the major allele being ancestral, we cannot do more.
    if variant.is_snp and len(variant.ALT) == 1:
        line = probs_reader.readline()

        # read est-sfs input data
        est_input = data_reader.readline()

        # get probability of major allele
        prob_major_allele = float(line.split(' ')[2])

        # restrict haplotypes to the non-outgroup birch samples
        bases = get_called_bases(variant.gt_bases[ingroup_mask])

        major_allele = '.'
        ancestral_allele = '.'

        if len(bases) > 0:
            # determine major allele
            major_allele = Counter(bases).most_common()[0][0]

            # take the major allele to be the ancestral allele
            # if its probability is greater than equal 0.5
            if prob_major_allele >= 0.5:
                ancestral_allele = major_allele
            else:
                # there are exactly two alleles
                ancestral_allele = variant.ALT[0]

        # add ancestral allele annotation to record
        variant.INFO['AA'] = ancestral_allele

        # add additional information from est-sfs
        variant.INFO['EST_SFS_output'] = line.replace('\n', '').replace(' ', '|')
        variant.INFO['EST_SFS_input'] = est_input.replace('\n', '').replace(' ', '|').replace(',', ':')

        # only log if major allele, ancestral allele and reference allele are not the same
        if not (major_allele == ancestral_allele == variant.REF):
            logger.debug(f"site: {variant.CHROM}:{variant.POS}, ancestral allele: {ancestral_allele}, "
                         f"major allele: {major_allele}, reference: {variant.REF}, "
                         f"prob major allele: {prob_major_allele}")
    else:

        # simply assign the ancestral allele to be the reference allele
        # if the record is monomorphic
        variant.INFO['AA'] = variant.REF
        variant.INFO['EST_SFS_output'] = '.'
        variant.INFO['EST_SFS_input'] = '.'

    writer.write_record(variant)

# raise error if there are lines left from the EST-SFS output
if next(probs_reader, None) is not None:
    raise AssertionError("Number of sites don't match.")

# raise error if there are lines left from the EST-SFS input
probs_reader.close()
