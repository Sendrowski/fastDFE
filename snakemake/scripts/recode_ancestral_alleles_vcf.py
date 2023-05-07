"""
Annotate the VCF with regards to the ancestral alleles.
An AA info tag will added in addition to further information.
This is same tag dadi looks for when parsing a unfolded SFS.
"""

__author__ = "Janek Sendrowski"
__contact__ = "j.sendrowski18@gmail.com"
__date__ = "2022-05-31"

import logging
import re
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
except NameError:
    # testing
    vcf_file = "results/test/remote/vcf/all/chr1_test.passed.biallelic.90.vcf.gz"
    probs = "results/test/remote/est-sfs/chr1_test.passed.biallelic.90.probs.txt"
    data = "results/test/remote/est-sfs/chr1_test.passed.biallelic.90.data.txt"
    samples_file = "results/test/remote/sample_lists/ingroups.args"
    log = "scratch/est-sfs.log"
    out = "scratch/1.polarized.vcf"

# configure logger to log stdout
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

logger.addHandler(logging.FileHandler(log))
logger.addHandler(logging.StreamHandler(sys.stdout))


# get a list of all called haplotypes
def haplotypes(calls):
    haplotypes = []

    for call in calls:
        if call.gt_bases:
            bases = re.split("/|\|", call.gt_bases)

            for base in bases:
                if base in ['A', 'C', 'G', 'T']:
                    haplotypes.append(base)

    return haplotypes


# restrict the calls to the sample set
def restrict(calls, names, exclude=False):
    if exclude:
        return list(filter((lambda call: call.sample not in names), calls))

    return list(filter((lambda call: call.sample in names), calls))


samples = pd.read_csv(samples_file, header=None, index_col=False)[0].tolist()

vcf_reader = VCF(vcf_file)

# Add AA info field to the header
vcf_reader.add_info_to_header({'ID': 'AA', 'Number': 1, 'Type': 'String', 'Description': 'Ancestral Allele'})
vcf_reader.add_info_to_header(
    {'ID': 'EST_SFS_probs', 'Number': 1, 'Type': 'String', 'Description': 'EST-SFS probabilities'})
vcf_reader.add_info_to_header({'ID': 'EST_SFS_input', 'Number': 1, 'Type': 'String', 'Description': 'EST-SFS input'})

probs_reader = open(probs, 'r')
data_reader = open(data, 'r')
writer = Writer(out, vcf_reader)

# write to data file
i = 0
for variant in tqdm(vcf_reader):

    # simply assign the ancestral allele to be the reference allele
    # if the record is monomorphic
    if not variant.is_snp:
        variant.INFO['AA'] = variant.REF
        variant.INFO['EST_SFS_probs'] = None
        variant.INFO['EST_SFS_input'] = None
    else:

        line = probs_reader.readline()

        # read est-sfs input data
        est_input = data_reader.readline()

        # get probability of major allele
        prob_major_allele = float(line.split(' ')[2])

        # restrict haplotypes to the non-outgroup birch samples
        hps = haplotypes(restrict(variant.samples, samples))

        major_allele = '.'

        if hps:
            # determine major allele
            major_allele = Counter(hps).most_common()[0][0]

            # take the major allele to be the ancestral allele
            # if its probability is greater than equal 0.5
            if prob_major_allele >= 0.5:
                ancestral_allele = major_allele
            else:
                # there are exactly two alleles
                ancestral_allele = None
                for allele in variant.alleles:
                    if str(allele) != major_allele:
                        ancestral_allele = str(allele)
        else:
            ancestral_allele = '.'

        # add ancestral allele annotation to record
        variant.INFO['AA'] = ancestral_allele

        # add additional information from est-sfs
        variant.INFO['EST_SFS_probs'] = line.replace('\n', '').replace(' ', '|')
        variant.INFO['EST_SFS_input'] = est_input.replace('\n', '').replace(' ', '|').replace(',', ':')

        # log change
        if not (major_allele == ancestral_allele == variant.REF):
            logging.debug(f"site: {variant.CHROM}:{variant.POS}, ancestral allele: {ancestral_allele}, "
                          f"major allele: {major_allele}, reference: {variant.REF}, "
                          f"prob major allele: {prob_major_allele}")

    writer.write_record(variant)

    i += 1
    if i % 1000 == 0: logging.debug(f"Processed sites: {i}")

# raise error if there are lines left from the EST-SFS output
if next(probs_reader, None) is not None:
    raise AssertionError("Number of sites don't match.")

# raise error if there are lines left from the EST-SFS input
probs_reader.close()
