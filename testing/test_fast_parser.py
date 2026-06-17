"""
Fast parser tests on a small, committed VCF subset with a capped number of sites. These cover the
core VCF -> SFS path and the VCF-only stratifications without needing the (gitignored, 435 MB) full
genome FASTA, staying well under a second each. Degeneracy/ancestral annotation, which require the
full reference genome, are covered in the inference/slow tiers.
"""
import numpy as np

import fastdfe as fd

VCF = 'resources/genome/betula/all.polarized.subset.10000.vcf.gz'


def test_parse_plain_sfs():
    sfs = fd.Parser(vcf=VCF, n=10, max_sites=1000).parse()

    assert list(sfs.types) == ['all']
    # an SFS over n=10 haplotypes has n + 1 = 11 entries, all non-negative, with real sites
    counts = sfs.all.to_list()
    assert len(counts) == 11
    assert all(c >= 0 for c in counts)
    assert sfs.n_sites.sum() > 0


def test_parse_subsample_smaller_n():
    # subsampling to a smaller n yields a correspondingly shorter SFS
    sfs = fd.Parser(vcf=VCF, n=6, max_sites=1000).parse()
    assert len(sfs.all.to_list()) == 7
    assert sfs.n_sites.sum() > 0


def test_parse_transition_transversion_stratification():
    sfs = fd.Parser(
        vcf=VCF, n=8, max_sites=1000,
        stratifications=[fd.TransitionTransversionStratification()],
    ).parse()

    # both mutation classes are present and partition the sites
    assert set(sfs.types) == {'transition', 'transversion'}
    assert sfs.n_sites.sum() > 0
