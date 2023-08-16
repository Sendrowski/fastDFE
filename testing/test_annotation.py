import itertools
import logging
from unittest.mock import MagicMock, PropertyMock, patch

import numpy as np
import pandas as pd
from numpy import testing
from pandas.testing import assert_frame_equal

from fastdfe.bio_handlers import count_sites, GFFHandler
from testing import prioritize_installed_packages

logging.getLogger('fastdfe').setLevel(logging.DEBUG)

prioritize_installed_packages()

from testing import TestCase
import pytest

from fastdfe import Annotator, MaximumParsimonyAnnotation, Parser, DegeneracyAnnotation, SynonymyAnnotation, \
    Annotation, OutgroupAncestralAlleleAnnotation, K2SubstitutionModel, JCSubstitutionModel


class AnnotatorTestCase(TestCase):
    """
    Test the annotators.
    """

    vcf_file = 'resources/genome/betula/biallelic.subset.10000.vcf.gz'
    fasta_file = 'resources/genome/betula/genome.subset.20.fasta'
    gff_file = 'resources/genome/betula/genome.gff.gz'

    @staticmethod
    def test_maximum_parsimony_annotation_annotate_site():
        """
        Test the maximum parsimony annotation for a single site.
        """
        # Test data
        test_cases = [
            {
                "ingroups": ["sample1", "sample2"],
                "samples": ["sample1", "sample2", "sample3"],
                "gt_bases": np.array(["A/A", "A/G", "G/G"]),
                "expected": "A",
            },
            {
                "ingroups": ["sample1", "sample2"],
                "samples": ["sample1", "sample2", "sample3"],
                "gt_bases": np.array(["./.", "C/C", "C/C"]),
                "expected": "C",
            },
            {
                "ingroups": ["sample1", "sample2"],
                "samples": ["sample1", "sample2", "sample3"],
                "gt_bases": np.array(["T/T", "./.", "./."]),
                "expected": "T",
            },
        ]

        # Mock Annotator
        mock_annotator = MagicMock()
        type(mock_annotator).info_ancestral = PropertyMock(return_value="AA")

        for test_case in test_cases:
            annotation = MaximumParsimonyAnnotation(samples=test_case["ingroups"])
            annotation.annotator = mock_annotator

            # Mock the VCF reader with samples
            mock_reader = MagicMock()
            mock_reader.samples = test_case["samples"]
            annotation.reader = mock_reader

            # Create ingroup mask
            annotation.samples_mask = np.isin(annotation.reader.samples, annotation.samples)

            # Mock variant with a dictionary for INFO
            mock_variant = MagicMock()
            mock_variant.gt_bases = test_case["gt_bases"]
            mock_variant.INFO = {}

            # Run the method
            annotation.annotate_site(mock_variant)

            # Check if the result matches the expectation
            assert mock_variant.INFO[annotation.annotator.info_ancestral] == test_case["expected"]

    def test_maximum_parsimony_annotation(self):
        """
        Test the maximum parsimony annotator.
        """
        ann = Annotator(
            vcf=self.vcf_file,
            output='scratch/test_maximum_parsimony_annotation.vcf',
            annotations=[MaximumParsimonyAnnotation()],
            info_ancestral='BB'
        )

        ann.annotate()

        Parser(self.vcf_file, n=20, info_ancestral='BB').parse().plot(title="Original")
        Parser(ann.output, n=20, info_ancestral='BB').parse().plot(title="Annotated")

        # assert number of sites is the same
        assert count_sites(self.vcf_file) == count_sites(ann.output)

    @staticmethod
    def test_degeneracy_annotation_human_test_genome():
        """
        Test the degeneracy annotator on a small human genome.
        """
        deg = DegeneracyAnnotation(
            fasta_file="http://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/chr21.fa.gz",
            gff_file="resources/genome/sapiens/hg38.sorted.gtf.gz"
        )

        vcf = "resources/genome/sapiens/chr21_test.vcf.gz"

        ann = Annotator(
            vcf=vcf,
            output='scratch/test_degeneracy_annotation_human_test_genome.vcf',
            annotations=[deg],
        )

        ann.annotate()

        # assert number of sites is the same
        assert count_sites(vcf) == count_sites(ann.output)

        # assert number of annotated sites and total number of sites
        assert deg.n_annotated == 7
        assert ann.n_sites == 1517

    @staticmethod
    def test_degeneracy_annotation_human_test_genome_remote_fasta_gzipped():
        """
        Test the degeneracy annotator on a small human genome with a remote gzipped fasta file.
        """
        deg = DegeneracyAnnotation(
            fasta_file="http://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/chr21.fa.gz",
            gff_file="resources/genome/sapiens/hg38.sorted.gtf.gz"
        )

        vcf = "resources/genome/sapiens/chr21_test.vcf.gz"

        ann = Annotator(
            vcf=vcf,
            output='scratch/test_degeneracy_annotation_human_test_genome_remote_fasta_gzipped.vcf',
            annotations=[deg],
        )

        ann.annotate()

        # assert number of annotated sites and total number of sites
        assert deg.n_annotated == 7
        assert ann.n_sites == 1517

    @staticmethod
    def test_degeneracy_annotation_betula_subset():
        """
        Test the degeneracy annotator.
        """
        vcf = "resources/genome/betula/all.subset.100000.vcf.gz"

        ann = Annotator(
            vcf=vcf,
            output='scratch/test_degeneracy_annotation_betula_subset.vcf',
            annotations=[
                DegeneracyAnnotation(
                    fasta_file="resources/genome/betula/genome.subset.20.fasta",
                    gff_file="resources/genome/betula/genome.gff.gz"
                )
            ],
        )

        ann.annotate()

        # assert number of sites is the same
        assert count_sites(vcf) == count_sites(ann.output)

    @staticmethod
    def test_annotator_load_vcf_from_url():
        """
        Test the annotator loading a VCF from a URL.
        """
        ann = Annotator(
            vcf="resources/genome/betula/biallelic.subset.10000.vcf.gz",
            output='scratch/test_annotator_load_vcf_from_url.vcf',
            annotations=[
                DegeneracyAnnotation(
                    fasta_file="resources/genome/betula/genome.subset.20.fasta",
                    gff_file="resources/genome/betula/genome.gff.gz"
                )
            ],
        )

        ann.annotate()

        # assert number of sites is the same
        assert ann.n_sites == 10000
        assert count_sites(ann.output) == 10000

    @staticmethod
    def test_compare_synonymy_annotation_with_vep_betula():
        """
        Compare the synonymy annotation with VEP.
        """
        syn = SynonymyAnnotation(
            fasta_file="resources/genome/betula/genome.subset.20.fasta",
            gff_file="resources/genome/betula/genome.gff.gz"
        )

        ann = Annotator(
            vcf="resources/genome/betula/biallelic.subset.10000.vcf.gz",
            output='scratch/test_compare_synonymy_annotation_with_vep_betula.vcf',
            annotations=[syn],
        )

        ann.annotate()

        assert syn.n_vep_comparisons == 3566
        assert len(syn.vep_mismatches) == 0

    @pytest.mark.slow
    def test_compare_synonymy_annotation_with_vep_human_chr21(self):
        """
        Compare the synonymy annotation with SnpEff and VEP.
        """
        syn = SynonymyAnnotation(
            fasta_file="resources/genome/sapiens/chr21.fasta",
            gff_file="resources/genome/sapiens/chr21.sorted.gff3",
            aliases=dict(chr21=['21'])
        )

        ann = Annotator(
            vcf="snakemake/results/vcf/sapiens/chr21.vep.vcf.gz",
            output="scratch/test_compare_synonymy_annotation_with_vep_human_chr21.vcf",
            annotations=[syn]
        )

        ann.annotate()

        assert syn.n_vep_comparisons == 9804
        assert len(syn.vep_mismatches) == 42

    @pytest.mark.slow
    def test_compare_synonymy_annotation_with_snpeff_human_chr21(self):
        """
        Compare the synonymy annotation with VEP.
        """
        syn = SynonymyAnnotation(
            fasta_file="resources/genome/sapiens/chr21.fasta",
            gff_file="resources/genome/sapiens/hg38.sorted.gtf.gz",
            aliases=dict(chr21=['21'])
        )

        ann = Annotator(
            vcf="snakemake/results/vcf/sapiens/chr21.snpeff.vcf.gz",
            output="scratch/test_compare_synonymy_annotation_with_snpeff_human_chr21.vcf",
            annotations=[syn]
        )

        ann.annotate()

        assert syn.n_snpeff_comparisons == 11240
        assert len(syn.snpeff_mismatches) == 233

    def test_get_degeneracy(self):
        """
        Test the get_degeneracy function.
        """
        # List of test cases. Each case is a tuple (codon, pos, expected_degeneracy).
        test_cases = [
            ('ATG', 0, 0),  # Codon for Methionine (Start), non-degenerate
            ('ATG', 1, 0),  # Codon for Methionine (Start), non-degenerate
            ('ATG', 2, 0),  # Codon for Methionine (Start), non-degenerate
            ('TGG', 0, 0),  # Codon for Tryptophan, non-degenerate
            ('TGG', 1, 0),  # Codon for Tryptophan, non-degenerate
            ('TGG', 2, 0),  # Codon for Tryptophan, non-degenerate
            ('GCT', 0, 0),  # Codon for Alanine, non-degenerate
            ('GCT', 1, 0),  # Codon for Alanine, non-degenerate
            ('GCT', 2, 4),  # Codon for Alanine, 4-fold degenerate
            ('CGT', 0, 0),  # Codon for Arginine, non-degenerate
            ('CGT', 1, 0),  # Codon for Arginine, non-degenerate
            ('CGT', 2, 4),  # Codon for Arginine, 4-fold degenerate
            ('ACT', 0, 0),  # Codon for Threonine, non-degenerate
            ('ACT', 1, 0),  # Codon for Threonine, non-degenerate
            ('ACT', 2, 4),  # Codon for Threonine, 4-fold degenerate
            ('GAT', 0, 0),  # Codon for Aspartic Acid, non-degenerate
            ('GAT', 1, 0),  # Codon for Aspartic Acid, non-degenerate
            ('GAT', 2, 2),  # Codon for Aspartic Acid, 4-fold degenerate
        ]

        for codon, pos, expected in test_cases:
            degeneracy = DegeneracyAnnotation._get_degeneracy(codon, pos)
            self.assertEqual(degeneracy, expected)

    def test_is_synonymous(self):
        # List of test cases. Each case is a tuple (codon1, codon2, expected_result).
        test_cases = [
            ('ATG', 'ATG', True),  # Same codons are always synonymous
            ('GCT', 'GCC', True),  # Both code for Alanine
            ('GCT', 'GTT', False),  # GCT codes for Alanine, GTT codes for Valine
            ('TAA', 'TAG', True),  # Both are stop codons
            ('TAA', 'ATG', False),  # TAA is a stop codon, ATG codes for Methionine
            ('CGT', 'CGG', True),  # Both code for Arginine
            ('GAT', 'GAC', True),  # Both code for Aspartic Acid
            ('TTT', 'TTC', True),  # Both code for Phenylalanine
            ('AAA', 'AAG', True),  # Both code for Lysine
            ('CAA', 'CAG', True),  # Both code for Glutamine
            ('AAT', 'AAC', True),  # Both code for Asparagine
            ('GAA', 'GAG', True),  # Both code for Glutamic Acid
            ('AGA', 'AGG', True),  # Both code for Arginine
            ('CTT', 'CTC', True),  # Both code for Leucine
            ('TAT', 'TAC', True),  # Both code for Tyrosine
            ('CGT', 'AGT', False),  # CGT codes for Arginine, AGT codes for Serine
            ('GGT', 'GCT', False),  # GGT codes for Glycine, GCT codes for Alanine
            ('GAT', 'GGT', False),  # GAT codes for Aspartic Acid, GGT codes for Glycine
            ('AAA', 'AGA', False),  # AAA codes for Lysine, AGA codes for Arginine
            ('CAA', 'GAA', False),  # CAA codes for Glutamine, GAA codes for Glutamic Acid
            ('AAT', 'ACT', False),  # AAT codes for Asparagine, ACT codes for Threonine
            ('GAA', 'GGA', False),  # GAA codes for Glutamic Acid, GGA codes for Glycine
            ('AGA', 'ACA', False),  # AGA codes for Arginine, ACA codes for Threonine
            ('CTT', 'CCT', False),  # CTT codes for Leucine, CCT codes for Proline
            ('TAT', 'TGT', False),  # TAT codes for Tyrosine, TGT codes for Cysteine
        ]

        for codon1, codon2, expected in test_cases:
            with self.subTest(codon1=codon1, codon2=codon2):
                synonymy = SynonymyAnnotation.is_synonymous(codon1, codon2)
                self.assertEqual(synonymy, expected)

    @staticmethod
    def test_count_target_sites():
        """
        Test the count_target_sites function.
        """
        # Define a mock dataframe to be returned by _load_cds
        mock_df = pd.DataFrame({
            'seqid': ['chr1', 'chr1', 'chr2', 'chr2', 'chr2'],
            'start': [1, 100, 1, 100, 200],
            'end': [10, 200, 50, 150, 250]
        })

        # We'll patch the _load_cds function to return our mock dataframe
        with patch.object(GFFHandler, "_load_cds", return_value=mock_df):
            result = GFFHandler("mock_file.gff")._count_target_sites()

        # Our expected result is the sum of the differences between end and start for each 'seqid'
        expected_result = {
            'chr1': (200 - 100 + 1) + (10 - 1 + 1),
            'chr2': (50 - 1 + 1) + (150 - 100 + 1) + (250 - 200 + 1)
        }

        assert result == expected_result

    @staticmethod
    def test_count_target_sites_betula_gff():
        """
        Test the count_target_sites function on the Betula gff file
        """
        gff = "resources/genome/betula/genome.gff.gz"

        result = Annotation.count_target_sites(gff)

        pass

    def test_outgroup_ancestral_allele_annotation_negative_lower_bounds_raises_value_error(self):
        """
        Test that a ValueError is raised when the lower bound of a parameter is negative.
        """
        with self.assertRaises(ValueError):
            OutgroupAncestralAlleleAnnotation(
                outgroups=["ERR2103730", "ERR2103731"],
                n_runs_rate=3,
                n_ingroups=5
            )

    def test_outgroup_ancestral_allele_annotation_upper_bounds_larger_than_lower_bounds_raises_value_error(self):
        """
        Test that a ValueError is raised when the lower bound of a parameter is negative.
        """
        with self.assertRaises(ValueError):
            OutgroupAncestralAlleleAnnotation(
                outgroups=["ERR2103730", "ERR2103731"],
                n_runs_rate=3,
                n_ingroups=5
            )

    @staticmethod
    def test_outgroup_ancestral_allele_annotation_pendula_thorough():
        """
        Test the MLEAncestralAlleleAnnotation class on the Betula pendula vcf file.
        """
        anc = OutgroupAncestralAlleleAnnotation(
            outgroups=["ERR2103730", "ERR2103731"],
            n_runs_rate=50,
            n_ingroups=10
        )

        ann = Annotator(
            vcf="resources/genome/betula/all.with_outgroups.subset.10000.vcf.gz",
            output='scratch/test_outgroup_ancestral_allele_annotation_pendula.vcf',
            annotations=[anc]
        )

        ann.annotate()

        anc.evaluate_likelihood_rates(anc.params_mle)

    @staticmethod
    def test_outgroup_ancestral_allele_annotation_pendula_use_prior_K2_model():
        """
        Test the MLEAncestralAlleleAnnotation class on the Betula pendula vcf file.
        """
        anc = OutgroupAncestralAlleleAnnotation(
            outgroups=["ERR2103730", "ERR2103731"],
            n_runs_rate=3,
            n_ingroups=5,
            model=K2SubstitutionModel(bounds=dict(k=(0.001, 100), K=(0.01, 0.1)))
        )

        ann = Annotator(
            vcf="resources/genome/betula/all.with_outgroups.subset.10000.vcf.gz",
            output='scratch/test_outgroup_ancestral_allele_annotation_pendula.vcf',
            annotations=[anc],
            max_sites=1000
        )

        ann.annotate()

        pass

    @staticmethod
    def test_outgroup_ancestral_allele_annotation_pendula_use_prior_JC_model():
        """
        Test the MLEAncestralAlleleAnnotation class on the Betula pendula vcf file.
        """
        anc = OutgroupAncestralAlleleAnnotation(
            outgroups=["ERR2103730", "ERR2103731"],
            n_runs_rate=3,
            n_ingroups=5,
            model=JCSubstitutionModel(bounds=dict(K=(0.01, 0.1)))
        )

        ann = Annotator(
            vcf="resources/genome/betula/all.with_outgroups.subset.10000.vcf.gz",
            output='scratch/test_outgroup_ancestral_allele_annotation_pendula.vcf',
            annotations=[anc],
            max_sites=1000
        )

        ann.annotate()

        ll1 = anc.evaluate_likelihood_rates({'K0': 0, 'K1': 0, 'K2': 0})
        ll2 = anc.evaluate_likelihood_rates({'K0': 0.01, 'K1': 0.01, 'K2': 0.01})

        pass

    @staticmethod
    def test_outgroup_ancestral_allele_annotation_pendula_not_use_prior():
        """
        Test the MLEAncestralAlleleAnnotation class on the Betula pendula vcf file.
        """
        anc = OutgroupAncestralAlleleAnnotation(
            outgroups=["ERR2103730", "ERR2103731"],
            n_runs_rate=3,
            n_ingroups=5,
            use_prior=False
        )

        ann = Annotator(
            vcf="resources/genome/betula/all.with_outgroups.subset.10000.vcf.gz",
            output='scratch/test_outgroup_ancestral_allele_annotation_pendula.vcf',
            annotations=[anc],
            max_sites=1000
        )

        ann.annotate()

    def test_get_p_tree_outgroup_ancestral_allele_annotation_one_outgroup(self):
        """
        Test the get_p_tree function with one outgroup.
        """
        params = dict(
            k=2,
            K0=0.5
        )

        model = K2SubstitutionModel()

        for base, outgroup_base in itertools.product(range(4), range(4)):
            p = OutgroupAncestralAlleleAnnotation.get_p_tree(
                base=base,
                n_outgroups=1,
                internal_nodes=[],
                outgroup_bases=[outgroup_base],
                model=model,
                params=dict(
                    k=2,
                    K0=0.5
                )
            )

            p_expected = model.get_prob(
                b1=base,
                b2=outgroup_base,
                i=0,
                params=params
            )

            self.assertEqual(p, p_expected)

    def test_get_p_tree_outgroup_ancestral_allele_annotation_two_outgroups(self):
        """
        Test the get_p_tree function with two outgroups.
        """
        params = dict(
            k=2,
            K0=0.5,
            K1=0.25,
            K2=0.125
        )

        model = K2SubstitutionModel()

        for base, outgroup_base1, outgroup_base2, internal_node in itertools.product(range(4), repeat=4):
            p = OutgroupAncestralAlleleAnnotation.get_p_tree(
                base=base,
                n_outgroups=2,
                internal_nodes=[internal_node],
                outgroup_bases=[outgroup_base1, outgroup_base2],
                params=params,
                model=model
            )

            p_expected = (model.get_prob(base, internal_node, 0, params) *
                          model.get_prob(internal_node, outgroup_base1, 1, params) *
                          model.get_prob(internal_node, outgroup_base2, 2, params))

            self.assertEqual(p, p_expected)

    def test_get_p_site_outgroup_ancestral_allele_annotation_three_outgroups(self):
        """
        Test the get_p_site function with three outgroups.
        """
        params = dict(
            k=2,
            K0=0.5,
            K1=0.25,
            K2=0.125,
            K3=0.0625,
            K4=0.03125
        )

        model = K2SubstitutionModel()

        for base, out1, out2, out3, int_node1, int_node2 in itertools.product(range(4), repeat=6):
            p = OutgroupAncestralAlleleAnnotation.get_p_tree(
                base=base,
                n_outgroups=3,
                internal_nodes=[int_node1, int_node2],
                outgroup_bases=[out1, out2, out3],
                model=model,
                params=dict(
                    k=2,
                    K0=0.5,
                    K1=0.25,
                    K2=0.125,
                    K3=0.0625,
                    K4=0.03125
                )
            )

            p_expected = (model.get_prob(base, int_node1, 0, params) *
                          model.get_prob(int_node1, out1, 1, params) *
                          model.get_prob(int_node1, int_node2, 2, params) *
                          model.get_prob(int_node2, out2, 3, params) *
                          model.get_prob(int_node2, out3, 4, params))

            self.assertEqual(p, p_expected)

    def test_outgroup_ancestral_allele_annotation_from_data(self):
        """
        Test the OutgroupAncestorAlleleAnnotation class on the Betula pendula vcf file.
        """
        anc = OutgroupAncestralAlleleAnnotation.from_data(
            n_major=[13, 15, 17, 11],
            major_bases=['A', 'C', 'G', 'T'],
            minor_bases=['C', 'G', 'T', 'A'],
            outgroup_bases=[['A', 'C'], ['G', 'G'], ['G', 'G'], ['A', 'A']],
            n_ingroups=20
        )

        anc.infer()

        probs = anc.get_probs()

        testing.assert_array_almost_equal(probs, [0.97479028, 0.04345338, 0.98914882, 0.02710116], decimal=5)

    @staticmethod
    def test_get_likelihood_outgroup_ancestral_allele_annotation():
        """
        Test the get_likelihood function.
        """
        anc = OutgroupAncestralAlleleAnnotation(outgroups=["ERR2103730", "ERR2103731"], n_ingroups=10)

        ann = Annotator(
            vcf="resources/genome/betula/all.with_outgroups.subset.10000.vcf.gz",
            output='scratch/test_outgroup_ancestral_allele_annotation_pendula.vcf',
            annotations=[anc],
            max_sites=1000
        )

        ann.annotate()

        ll = anc.evaluate_likelihood_rates(dict(
            k=2,
            K0=0.5,
            K1=0.5,
            K2=0.5
        ))

        pass

    @staticmethod
    def test_from_est_sfs_input():
        """
        Test the from_est_sfs_input function.
        """
        anc = OutgroupAncestralAlleleAnnotation.from_est_sfs(
            file="resources/EST-SFS/TEST-DATA.TXT",
            model=JCSubstitutionModel(),
            n_runs_rate=2
        )

        anc.infer()

        probs = anc.get_probs()

        sfs = anc.get_sfs().to_numpy()

        pass

    @staticmethod
    def test_from_est_sfs_chunked():
        """
        Test that the chunked and non-chunked version of the from_est_sfs function return the same results.
        """
        anc1 = OutgroupAncestralAlleleAnnotation.from_est_sfs(
            file="resources/EST-SFS/TEST-DATA.TXT",
            chunk_size=5
        )

        anc2 = OutgroupAncestralAlleleAnnotation.from_est_sfs(
            file="resources/EST-SFS/TEST-DATA.TXT",
            chunk_size=100
        )

        cols = ['n_major', 'major_base', 'minor_base', 'outgroup_bases']

        assert_frame_equal(
            anc1.configs.sort_values(by=cols).reset_index(drop=True).sort_index(axis=1),
            anc2.configs.sort_values(by=cols).reset_index(drop=True).sort_index(axis=1)
        )

    def test_from_data_chunked(self):
        """
        Test that the from_data function produces the expected results.
        """
        anc = OutgroupAncestralAlleleAnnotation.from_data(
            n_major=[13, 15, 17, 11],
            major_bases=['A', 'C', 'G', 'T'],
            minor_bases=['C', 'G', 'T', 'A'],
            outgroup_bases=[['A', 'C'], ['G', 'G'], ['G', 'G'], ['A', 'A']],
            n_ingroups=20
        )

        cols = ['n_major', 'major_base', 'minor_base', 'outgroup_bases', 'sites', 'multiplicity']

        self.assertDictEqual(anc.configs[cols].to_dict(), {
            'major_base': {0: 0, 1: 1, 2: 2, 3: 3},
            'minor_base': {0: 1, 1: 2, 2: 3, 3: 0},
            'outgroup_bases': {0: (0, 1), 1: (2, 2), 2: (2, 2), 3: (0, 0)},
            'n_major': {0: 13, 1: 15, 2: 17, 3: 11},
            'sites': {0: [0], 1: [1], 2: [2], 3: [3]},
            'multiplicity': {0: 1, 1: 1, 2: 1, 3: 1},
        })
