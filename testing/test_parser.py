import logging

from testing import prioritize_installed_packages

prioritize_installed_packages()

from unittest import TestCase

import dadi
import numpy as np

import fastdfe
from fastdfe import DegeneracyStratification, BaseTransitionStratification, TransitionTransversionStratification, \
    BaseContextStratification, AncestralBaseStratification

logging.getLogger('fastdfe').setLevel(logging.DEBUG)


class ParserTestCase(TestCase):
    """
    Test the inference.

    TODO test parser in more once we have annotators
    """
    vcf_file = 'resources/genome/betula/biallelic.subset.10000.vcf.gz'
    vcf_file_with_monomorphic = 'resources/genome/betula/all.with_outgroups.subset.10000.vcf.gz'
    fasta_file = 'resources/genome/betula/genome.subset.2.fasta'
    pop_file = 'resources/genome/betula/pops_dadi.txt'

    def test_compare_sfs_with_data(self):
        """
        Compare the sfs from dadi with the one from the data.
        """
        p = fastdfe.Parser(vcf=self.vcf_file, n=20, stratifications=[], seed=2)

        sfs = p.parse().all

        sfs.plot()

        data_dict = dadi.Misc.make_data_dict_vcf(self.vcf_file, self.pop_file)

        sfs2 = dadi.Spectrum.from_data_dict(data_dict, ['pop0'], [20], polarized=True)

        diff_rel = np.abs(sfs.to_numpy() - sfs2.data) / sfs2.data

        # assert total number of sites
        assert np.sum(sfs.data) == 4304 - p.n_skipped

        # check that the sum of the sfs is the same
        self.assertAlmostEqual(np.sum(sfs.data), np.sum(sfs2.data))

        # this is better for large VCF files
        assert np.max(diff_rel) < 0.8

    def test_degeneracy_stratification(self):
        """
        Test the degeneracy stratification.
        """
        p = fastdfe.Parser(vcf=self.vcf_file, n=20, stratifications=[DegeneracyStratification()])

        sfs = p.parse()

        sfs.plot()

        # assert total number of sites
        assert np.sum(sfs.all.data) == 4304 - p.n_skipped

    def test_base_transition_stratification(self):
        """
        Test the base transition stratification.
        """
        p = fastdfe.Parser(
            vcf=self.vcf_file_with_monomorphic,
            n=20,
            stratifications=[BaseTransitionStratification()]
        )

        sfs = p.parse()

        # check that probabilities sum up to 1
        self.assertAlmostEqual(1, sum(p.stratifications[0].probabilities.values()))

        sfs.plot()

        # assert total number of sites
        assert np.sum(sfs.all.data) == 4306 - p.n_skipped

    def test_transition_transversion_stratification(self):
        """
        Test the transition transversion stratification.
        """
        p = fastdfe.Parser(
            vcf=self.vcf_file_with_monomorphic,
            n=20,
            stratifications=[TransitionTransversionStratification()]
        )

        sfs = p.parse()

        # check that probabilities sum up to 1
        self.assertAlmostEqual(1, sum(p.stratifications[0].probabilities.values()))

        sfs.plot()

        # assert total number of sites
        assert np.sum(sfs.all.data) == 4306 - p.n_skipped

    def test_base_context_stratification(self):
        """
        Test the base context stratification.
        """
        p = fastdfe.Parser(
            vcf=self.vcf_file,
            n=20,
            stratifications=[BaseContextStratification(fasta_file=self.fasta_file)]
        )

        sfs = p.parse()

        sfs.plot()

        # assert total number of sites
        assert np.sum(sfs.all.data) == 4304 - p.n_skipped

    def test_reference_base_stratification(self):
        """
        Test the reference base stratification.
        """
        p = fastdfe.Parser(
            vcf=self.vcf_file,
            n=20,
            stratifications=[AncestralBaseStratification()]
        )

        sfs = p.parse()

        sfs.plot()

        # assert total number of sites
        assert np.sum(sfs.all.data) == 4304 - p.n_skipped
