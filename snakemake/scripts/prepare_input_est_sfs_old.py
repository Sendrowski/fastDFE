"""
Prepare the input for est-sfs.
"""

__author__ = "Janek Sendrowski"
__contact__ = "j.sendrowski18@gmail.com"
__date__ = "2022-05-31"

import os
import re

import numpy as np
import pandas as pd
import vcf

try:
    vcf_file = snakemake.input.vcf
    ingroups_file = snakemake.input.ingroups
    outgroups_file = snakemake.input.outgroups
    out_data = snakemake.output.data
    n_outgroups = snakemake.config['n_outgroup']
    n_max_subsamples = snakemake.config['n_samples']
except NameError:
    # testing
    vcf_file = "../resources/genome/betula/all.with_outgroups.subset.10000.vcf.gz"
    ingroups_file = "../resources/genome/betula/sample_sets/birch.args"
    outgroups_file = "../resources/genome/betula/sample_sets/outgroups.args"
    out_data = "scratch/est-sfs.data.txt"
    n_outgroups = 2
    n_max_subsamples = 50


# obtain a subsample from the given set of haplotypes
def subsample(haplotypes, size):
    # return an empty array when there are no haplotypes
    # this can happen for an uncalled outgroup site
    if len(haplotypes) == 0:
        return []

    return np.array(haplotypes)[np.random.choice(len(haplotypes), size=size, replace=False)]


# returns dict of counts indexed by A, C, G and T
def count(haplotypes):
    counts = {'A': 0, 'C': 0, 'G': 0, 'T': 0}

    for key, value in zip(*np.unique(haplotypes, return_counts=True)):
        counts[key] = value

    return counts


# returns 'A,C,G,T'
def base_dict_to_string(d):
    return ','.join(map(str, [d['A'], d['C'], d['G'], d['T']]))


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


# load ingroup and outgroup samples
ingroups = pd.read_csv(ingroups_file, header=None, index_col=False)[0].tolist()
outgroups = pd.read_csv(outgroups_file, header=None, index_col=False)[0].tolist()

# seed rng
np.random.seed(seed=0)

# number of subsamples to sample from haplotypes
n_subsamples = min(n_max_subsamples, len(ingroups))

# write to data file
with open(out_data, 'w') as f:
    i = 0
    for record in vcf.Reader(filename=vcf_file):

        # only do inference for polymorphic sites
        if not record.is_monomorphic:
            ingroup = count(subsample(haplotypes(restrict(record.samples, outgroups, True)), n_subsamples))

            outgroup = []
            for i in range(n_outgroups):
                outgroup.append(count(subsample(haplotypes(restrict(record.samples, [outgroups[i]])), 1)))

            f.write(' '.join([base_dict_to_string(r) for r in [ingroup] + outgroup]) + os.linesep)

        i += 1
        if i % 1000 == 0: print(f"Processed sites: {i}", flush=True)
