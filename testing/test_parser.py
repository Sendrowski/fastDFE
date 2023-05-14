import logging

from testing import prioritize_installed_packages

prioritize_installed_packages()

from unittest import TestCase

import dadi
import numpy as np

from fastdfe import DegeneracyStratification, BaseTransitionStratification, TransitionTransversionStratification, \
    BaseContextStratification, AncestralBaseStratification, Parser, DegeneracyAnnotation, MaximumParsimonyAnnotation, \
    CodingSequenceFiltration

logging.getLogger('fastdfe').setLevel(logging.DEBUG)


class ParserTestCase(TestCase):
    """
    Test the inference.

    TODO test parser in more once we have annotators
    """
    vcf_file = 'resources/genome/betula/biallelic.subset.10000.vcf.gz'
    vcf_file_with_monomorphic = 'resources/genome/betula/all.with_outgroups.subset.10000.vcf.gz'
    fasta_file = 'resources/genome/betula/genome.subset.20.fasta'
    pop_file = 'resources/genome/betula/pops_dadi.txt'

    def test_compare_sfs_with_data(self):
        """
        Compare the sfs from dadi with the one from the data.
        """
        p = Parser(vcf=self.vcf_file, n=20, stratifications=[], seed=2)

        sfs = p.parse().all

        sfs.plot()

        data_dict = dadi.Misc.make_data_dict_vcf(self.vcf_file, self.pop_file)

        sfs2 = dadi.Spectrum.from_data_dict(data_dict, ['pop0'], [20], polarized=True)

        diff_rel = np.abs(sfs.to_numpy() - sfs2.data) / sfs2.data

        # assert total number of sites
        assert np.sum(sfs.data) == 10000 - p.n_skipped

        # check that the sum of the sfs is the same
        self.assertAlmostEqual(np.sum(sfs.data), np.sum(sfs2.data))

        # this is better for large VCF files
        assert np.max(diff_rel) < 0.8

    def test_degeneracy_stratification(self):
        """
        Test the degeneracy stratification.
        """
        p = Parser(vcf=self.vcf_file, n=20, stratifications=[DegeneracyStratification()])

        sfs = p.parse()

        sfs.plot()

        # assert total number of sites
        assert sfs.all.data.sum() == 10000 - p.n_skipped

    def test_base_transition_stratification(self):
        """
        Test the base transition stratification.
        """
        p = Parser(
            vcf=self.vcf_file_with_monomorphic,
            n=20,
            stratifications=[BaseTransitionStratification()]
        )

        sfs = p.parse()

        # check that probabilities sum up to 1
        self.assertAlmostEqual(1, sum(p.stratifications[0].probabilities.values()))

        sfs.plot()

        # assert total number of sites
        assert sfs.all.data.sum() == 10000 - p.n_skipped

    def test_transition_transversion_stratification(self):
        """
        Test the transition transversion stratification.
        """
        p = Parser(
            vcf=self.vcf_file_with_monomorphic,
            n=20,
            stratifications=[TransitionTransversionStratification()]
        )

        sfs = p.parse()

        # check that probabilities sum up to 1
        self.assertAlmostEqual(1, sum(p.stratifications[0].probabilities.values()))

        sfs.plot()

        # assert total number of sites
        assert sfs.all.data.sum() == 10000 - p.n_skipped

    def test_base_context_stratification(self):
        """
        Test the base context stratification.
        """
        p = Parser(
            vcf=self.vcf_file,
            n=20,
            stratifications=[BaseContextStratification(fasta_file=self.fasta_file)]
        )

        sfs = p.parse()

        sfs.plot()

        # assert total number of sites
        assert sfs.all.data.sum() == 10000 - p.n_skipped

    def test_reference_base_stratification(self):
        """
        Test the reference base stratification.
        """
        p = Parser(
            vcf=self.vcf_file,
            n=20,
            stratifications=[AncestralBaseStratification()]
        )

        sfs = p.parse()

        sfs.plot()

        # assert total number of sites
        assert sfs.all.data.sum() == 10000 - p.n_skipped

    def test_parser_load_vcf_from_url(self):
        """
        Test the parser loading a VCF from a URL.
        """
        p = Parser(
            vcf='https://github.com/Sendrowski/fastDFE/blob/master/resources/genome/betula/biallelic.subset.10000.vcf.gz?raw=true',
            n=20
        )

        sfs = p.parse()

        assert sfs.all.data.sum() == 10000 - p.n_skipped

    def test_parse_vcf_without_AA_yields_empty_sfs(self):
        """
        Test that parsing a VCF file without AA info field yields an empty SFS
        """
        p = Parser(
            vcf="resources/genome/sapiens/chr21_test.vcf.gz",
            n=20,
            ignore_not_polarized=True
        )

        sfs = p.parse()

        # assert total number of sites is 0
        assert sfs.all.data.sum() == 0

    def test_parse_vcf(self):
        """
        Parse the VCF file using a remote fasta
        :return:
        """
        deg = DegeneracyAnnotation(
            fasta_file="https://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/chr21.fa.gz",
            gff_file="resources/genome/sapiens/hg38.gtf"
        )

        aa = MaximumParsimonyAnnotation()

        f = CodingSequenceFiltration(
            gff_file="resources/genome/sapiens/hg38.gtf"
        )

        p = Parser(
            vcf="resources/genome/sapiens/chr21_test.vcf.gz",
            n=20,
            ignore_not_polarized=True,
            annotations=[deg, aa],
            filtrations=[f],
        )

        sfs = p.parse()

        assert sfs.all.data.sum() == 6
