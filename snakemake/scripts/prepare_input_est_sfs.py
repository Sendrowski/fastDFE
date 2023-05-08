"""
Prepare the input for est-sfs.
"""

__author__ = "Janek Sendrowski"
__contact__ = "j.sendrowski18@gmail.com"
__date__ = "2022-05-31"

import os
from typing import Dict

import numpy as np
import pandas as pd
from cyvcf2 import VCF
from tqdm import tqdm

try:
    vcf_file = snakemake.input.vcf
    ingroups_file = snakemake.input.ingroups
    outgroups_file = snakemake.input.outgroups
    out_data = snakemake.output.data
    n_outgroups = snakemake.config['n_outgroup']
    n_max_subsamples = snakemake.config['n_samples']
    seed = snakemake.config.get('seed', 0)
except NameError:
    # testing
    vcf_file = "../resources/genome/betula/all.with_outgroups.subset.10000.vcf.gz"
    ingroups_file = "../resources/genome/betula/sample_sets/birch.args"
    outgroups_file = "../resources/genome/betula/sample_sets/outgroups.args"
    out_data = "scratch/est-sfs.data.txt"
    n_outgroups = 2
    n_max_subsamples = 50
    seed = 0

rng = np.random.default_rng(seed=seed)


def subsample(bases: np.ndarray, size: int) -> np.ndarray:
    """
    Subsample a set of bases.

    :param bases: A list of bases.
    :param size: The size of the subsample.
    :param allow_less: Whether to allow a subsample smaller than the requested size.
    :return: A subsample of the bases.
    """
    # return an empty array when there are no haplotypes
    # this can happen for an uncalled outgroup site
    if len(bases) == 0:
        return np.array([])

    # sample with replacement if the number of haplotypes is less than the subsample size
    if len(bases) < size:
        return np.array(bases)[rng.choice(len(bases), size=size, replace=True)]

    return np.array(bases)[rng.choice(len(bases), size=size, replace=False)]


def count(bases: np.ndarray) -> Dict[str, int]:
    """
    Count the number of bases in a list of haplotypes.

    :param bases: Array of bases.
    :return: A dictionary of base counts.
    """
    counts = {'A': 0, 'C': 0, 'G': 0, 'T': 0}

    for key, value in zip(*np.unique(bases, return_counts=True)):
        counts[key] = value

    return counts


def base_dict_to_string(d: Dict[str, int]) -> str:
    """
    Convert a dictionary of bases to a string.

    :param d: A dictionary of base counts.
    :return: A string of base counts.
    """
    return ','.join(map(str, [d['A'], d['C'], d['G'], d['T']]))


def get_called_bases(calls: np.ndarray) -> np.ndarray:
    """
    Get the called bases from a list of calls.

    :param calls: Array of calls.
    :return: Array of called bases.
    """
    return np.array([b for b in '/'.join(calls).replace('|', '/') if b in 'ACGT'])


# load ingroup and outgroup samples
ingroups = pd.read_csv(ingroups_file, header=None, index_col=False)[0].tolist()
outgroups = pd.read_csv(outgroups_file, header=None, index_col=False)[0].tolist()

# number of subsamples to sample from haplotypes
n_subsamples = min(n_max_subsamples, len(ingroups))

# write to data file
with open(out_data, 'w') as f:
    # create reader
    vcf_reader = VCF(vcf_file)

    # raise error if outgroup samples are not in VCF
    if not set(outgroups).issubset(set(vcf_reader.samples)):
        raise ValueError("Outgroup samples not in VCF.")

    # get indices of ingroup and outgroup samples
    ingroup_mask = [sample in ingroups for sample in vcf_reader.samples]
    outgroup_mask = [sample in outgroups for sample in vcf_reader.samples]

    for variant in tqdm(vcf_reader):

        # Only do inference for bi-allelic SNPs.
        if variant.is_snp:

            # get base counts for outgroup samples
            outgroup_counts = get_called_bases(variant.gt_bases[outgroup_mask])

            # Only do inference for bi-allelic SNPs for which
            # at least ``n_outgroups`` outgroup samples are called
            if len(variant.ALT) == 1 and len(outgroup_counts) >= n_outgroups:

                # get base counts for ingroup samples
                ingroup_counts = count(subsample(get_called_bases(variant.gt_bases[ingroup_mask]), n_subsamples))

                # subsample outgroup samples
                outgroup_subsamples = subsample(outgroup_counts, n_outgroups)

                # create a base count dictionary for each haplotype
                outgroup_dicts = [dict(A=0, C=0, G=0, T=0) for _ in range(n_outgroups)]
                for i, sub in enumerate(outgroup_subsamples):
                    outgroup_dicts[i][sub] = 1

                f.write(' '.join([base_dict_to_string(r) for r in [ingroup_counts] + outgroup_dicts]) + os.linesep)
