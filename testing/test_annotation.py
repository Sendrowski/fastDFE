import itertools
import logging
from unittest.mock import MagicMock, PropertyMock, patch, Mock

import numpy as np
import pandas as pd
from cyvcf2 import Variant
from matplotlib import pyplot as plt
from matplotlib.collections import PathCollection
from numpy import testing
from pandas.testing import assert_frame_equal

from fastdfe.annotation import _ESTSFSAncestralAnnotation, base_indices, SiteConfig, SiteInfo
from fastdfe.io_handlers import count_sites, GFFHandler, get_called_bases, DummyVariant
from testing import prioritize_installed_packages

# logging.getLogger('fastdfe').setLevel(logging.DEBUG)

prioritize_installed_packages()

from testing import TestCase
import pytest

import fastdfe as fd


class AnnotationTestCase(TestCase):
    """
    Test the annotators.
    """

    @staticmethod
    def test_maximum_parsimony_annotation_annotate_site():
        """
        Test the maximum parsimony annotation for a single site.
        """
        # test data
        cases = [
            dict(
                ingroups=["sample1", "sample2"],
                samples=["sample1", "sample2", "sample3"],
                gt_bases=np.array(["A/A", "A/G", "G/G"]),
                expected="A"
            ),
            dict(
                ingroups=["sample1", "sample2"],
                samples=["sample1", "sample2", "sample3"],
                gt_bases=np.array(["./.", "C/C", "C/C"]),
                expected="C"
            ),
            dict(
                ingroups=["sample1", "sample2"],
                samples=["sample1", "sample2", "sample3"],
                gt_bases=np.array(["T/T", "./.", "./."]),
                expected="T"
            ),
            dict(
                ingroups=["sample1", "sample2"],
                samples=["sample1", "sample2", "sample3"],
                gt_bases=np.array(["./.", "./.", "./."]),
                expected="."
            )
        ]

        # mock Annotator
        mock_annotator = MagicMock()
        type(mock_annotator).info_ancestral = PropertyMock(return_value="AA")

        for test_case in cases:
            annotation = fd.MaximumParsimonyAncestralAnnotation(samples=test_case["ingroups"])
            annotation.handler = mock_annotator

            # mock the VCF reader with samples
            mock_reader = MagicMock()
            mock_reader.samples = test_case["samples"]
            annotation.reader = mock_reader

            # create ingroup mask
            annotation.samples_mask = np.isin(annotation.reader.samples, annotation.samples)

            # mock variant with a dictionary for INFO
            mock_variant = MagicMock()
            mock_variant.gt_bases = test_case["gt_bases"]
            mock_variant.INFO = {}

            # run the method
            annotation.annotate_site(mock_variant)

            # check if the result matches the expectation
            assert mock_variant.INFO[annotation.handler.info_ancestral] == test_case["expected"]

    def test_maximum_parsimony_annotation_get_ancestral(self):
        """
        Test the _get_ancestral method.
        """
        ann = fd.MaximumParsimonyAncestralAnnotation

        # check that reference is used for dummy variants
        self.assertEqual('A', ann._get_ancestral(DummyVariant(ref='A', pos=1, chrom="1"), mask=np.array([])))

        # check that only allele is used for monomorphic sites
        variant = Mock(spec=Variant)
        variant.is_snp = False
        variant.REF = 'A'
        variant.gt_bases = np.array(['G/G', 'A/A'])
        self.assertEqual('G', ann._get_ancestral(variant, mask=np.array([True, False])))

        # check that only allele is used for non-SNPs only if it is a valid base
        variant = Mock(spec=Variant)
        variant.is_snp = False
        variant.REF = 'A'
        variant.gt_bases = np.array(['GAT/GAT', 'A/A'])
        self.assertEqual('.', ann._get_ancestral(variant, mask=np.array([True, False])))

        # check that only allele is used for non-SNPs only if reference is a valid base
        variant = Mock(spec=Variant)
        variant.is_snp = False
        variant.REF = 'GAT'
        variant.gt_bases = np.array(['A/A', 'A/A'])
        self.assertEqual('.', ann._get_ancestral(variant, mask=np.array([True, False])))

        # make sure we can pass site without any ingroup calls
        variant = Mock(spec=Variant)
        variant.is_snp = False
        variant.gt_bases = np.array(['./.', './.'])
        self.assertEqual('.', ann._get_ancestral(variant, mask=np.array([False, True])))

        # check that the major allele is used for biallelic sites
        variant = Mock(spec=Variant)
        variant.is_snp = True
        variant.gt_bases = np.array(['A/A', 'G/G', 'G/G'])
        self.assertEqual('G', ann._get_ancestral(variant, mask=np.array([True, True, True])))

    def test_maximum_parsimony_annotation_betula(self):
        """
        Test the maximum parsimony annotator with the betula dataset.
        """
        ann = fd.Annotator(
            vcf='resources/genome/betula/biallelic.polarized.vcf.gz',
            output='scratch/test_maximum_parsimony_annotation.vcf',
            annotations=[fd.MaximumParsimonyAncestralAnnotation()],
            max_sites=10000
        )

        ann.annotate()

        fd.Parser(ann.output, n=20, info_ancestral='BB').parse().plot()

        # assert number of sites is the same
        self.assertEqual(10000, count_sites(ann.output))

    @staticmethod
    def test_degeneracy_annotation_human_test_genome():
        """
        Test the degeneracy annotator on a small human genome.
        """
        deg = fd.DegeneracyAnnotation()

        vcf = "resources/genome/sapiens/chr21_test.vcf.gz"

        ann = fd.Annotator(
            vcf=vcf,
            output='scratch/test_degeneracy_annotation_human_test_genome.vcf',
            fasta="http://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/chr21.fa.gz",
            gff="resources/genome/sapiens/hg38.sorted.gtf.gz",
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
        deg = fd.DegeneracyAnnotation()

        vcf = "resources/genome/sapiens/chr21_test.vcf.gz"

        ann = fd.Annotator(
            vcf=vcf,
            output='scratch/test_degeneracy_annotation_human_test_genome_remote_fasta_gzipped.vcf',
            fasta="http://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/chr21.fa.gz",
            gff="resources/genome/sapiens/hg38.sorted.gtf.gz",
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

        ann = fd.Annotator(
            vcf=vcf,
            output='scratch/test_degeneracy_annotation_betula_subset.vcf',
            fasta="resources/genome/betula/genome.subset.20.fasta",
            gff="resources/genome/betula/genome.gff.gz",
            annotations=[
                fd.DegeneracyAnnotation()
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
        ann = fd.Annotator(
            vcf="resources/genome/betula/biallelic.subset.10000.vcf.gz",
            fasta="resources/genome/betula/genome.subset.20.fasta",
            gff="resources/genome/betula/genome.gff.gz",
            output='scratch/test_annotator_load_vcf_from_url.vcf',
            annotations=[
                fd.DegeneracyAnnotation()
            ],
        )

        ann.annotate()

        # assert number of sites is the same
        assert ann.n_sites == 10000
        assert count_sites(ann.output) == 10000

    @staticmethod
    def test_compare_synonymy_annotation_with_vep_betula():
        """
        Compare the synonymy annotation with VEP for the betula dataset.
        """
        syn = fd.SynonymyAnnotation()

        ann = fd.Annotator(
            vcf="resources/genome/betula/biallelic.subset.10000.vcf.gz",
            output='scratch/test_compare_synonymy_annotation_with_vep_betula.vcf',
            fasta="resources/genome/betula/genome.subset.20.fasta",
            gff="resources/genome/betula/genome.gff.gz",
            annotations=[syn],
        )

        ann.annotate()

        assert syn.n_vep_comparisons == 3566
        assert len(syn.vep_mismatches) == 0

    @pytest.mark.slow
    def test_compare_synonymy_annotation_with_vep_human_chr21(self):
        """
        Compare the synonymy annotation with VEP for human chromosome 21.
        """
        syn = fd.SynonymyAnnotation()

        ann = fd.Annotator(
            vcf="snakemake/results/vcf/sapiens/chr21.vep.vcf.gz",
            output="scratch/test_compare_synonymy_annotation_with_vep_human_chr21.vcf",
            fasta="resources/genome/sapiens/chr21.fasta",
            gff="resources/genome/sapiens/chr21.sorted.gff3",
            aliases=dict(chr21=['21']),
            annotations=[syn]
        )

        ann.annotate()

        assert syn.n_vep_comparisons == 9804
        assert len(syn.vep_mismatches) == 42

    @pytest.mark.slow
    def test_compare_synonymy_annotation_with_snpeff_human_chr21(self):
        """
        Compare the synonymy annotation with snpEff for human chromosome 21.
        """
        syn = fd.SynonymyAnnotation()

        ann = fd.Annotator(
            vcf="snakemake/results/vcf/sapiens/chr21.snpeff.vcf.gz",
            output="scratch/test_compare_synonymy_annotation_with_snpeff_human_chr21.vcf",
            fasta="resources/genome/sapiens/chr21.fasta",
            gff="resources/genome/sapiens/hg38.sorted.gtf.gz",
            aliases=dict(chr21=['21']),
            annotations=[syn]
        )

        ann.annotate()

        assert syn.n_snpeff_comparisons == 11240
        assert len(syn.snpeff_mismatches) == 233

    @pytest.mark.slow
    def test_compare_synonymy_annotation_with_vep_hgdp_chr21(self):
        """
        Compare the synonymy annotation with VEP for HGDP chromosome 21.
        """
        syn = fd.SynonymyAnnotation()

        ann = fd.Annotator(
            vcf="snakemake/results/vcf/hgdp/21/opts.vep.vcf.gz",
            output="scratch/test_compare_synonymy_annotation_with_vep_hgdp_chr21.vcf",
            fasta="snakemake/results/fasta/hgdp/21.fasta.gz",
            gff="snakemake/results/gff/hgdp/21.gff3.gz",
            aliases=dict(chr21=['21']),
            annotations=[syn]
        )

        ann.annotate()

        assert syn.n_vep_comparisons == 6010
        assert len(syn.vep_mismatches) == 40

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
            degeneracy = fd.DegeneracyAnnotation._get_degeneracy(codon, pos)
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
                synonymy = fd.SynonymyAnnotation.is_synonymous(codon1, codon2)
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

    def test_count_target_sites_betula_gff_all_contigs(self):
        """
        Test the count_target_sites function on the Betula gff file.
        """
        gff = "resources/genome/betula/genome.gff.gz"

        target_sites = fd.Annotation.count_target_sites(gff)

        self.assertEqual(3273, len(target_sites))
        self.assertEqual(337816, target_sites['Contig0'])

    def test_count_target_sites_betula_gff_some_contigs(self):
        """
        Test the count_target_sites function on the Betula gff file.
        """
        gff = "resources/genome/betula/genome.gff.gz"

        target_sites = fd.Annotation.count_target_sites(gff, contigs=['Contig0', 'Contig10'])

        self.assertEqual(2, len(target_sites))
        self.assertEqual(337816, target_sites['Contig0'])
        self.assertEqual(4992, target_sites['Contig10'])

    def test_aliases_empty_dict(self):
        """
        Test aliases when no aliases are given.
        """
        contig_aliases = {}
        expected_mappings = {}
        expected_aliases_expanded = {}

        mappings, aliases_expanded = fd.FileHandler._expand_aliases(contig_aliases)

        self.assertEqual(mappings, expected_mappings)
        self.assertEqual(aliases_expanded, expected_aliases_expanded)

    def test_aliases_single_contig(self):
        """
        Test aliases when a single contig is given.
        """
        contig_aliases = {'chr1': ['1']}
        expected_mappings = {'chr1': 'chr1', '1': 'chr1'}
        expected_aliases_expanded = {'chr1': ['1', 'chr1']}

        mappings, aliases_expanded = fd.FileHandler._expand_aliases(contig_aliases)

        self.assertEqual(mappings, expected_mappings)
        self.assertEqual(aliases_expanded, expected_aliases_expanded)

    def test_aliases_multiple_contigs(self):
        """
        Test aliases when multiple contigs are given.
        """
        contig_aliases = {'chr1': ['1'], 'chr2': ['2']}
        expected_mappings = {'chr1': 'chr1', '1': 'chr1', 'chr2': 'chr2', '2': 'chr2'}
        expected_aliases_expanded = {'chr1': ['1', 'chr1'], 'chr2': ['2', 'chr2']}

        mappings, aliases_expanded = fd.FileHandler._expand_aliases(contig_aliases)

        self.assertEqual(mappings, expected_mappings)
        self.assertEqual(aliases_expanded, expected_aliases_expanded)

    def test_aliases_multiple_aliases(self):
        """
        Test aliases when multiple aliases are given for a single contig.
        """
        contig_aliases = {'chr1': ['1', 'one']}
        expected_mappings = {'chr1': 'chr1', '1': 'chr1', 'one': 'chr1'}
        expected_aliases_expanded = {'chr1': ['1', 'one', 'chr1']}

        mappings, aliases_expanded = fd.FileHandler._expand_aliases(contig_aliases)

        self.assertEqual(mappings, expected_mappings)
        self.assertEqual(aliases_expanded, expected_aliases_expanded)


class MaximumLikelihoodAncestralAnnotationTest(TestCase):
    """
    Test the MaximumLikelihoodAncestralAnnotation class.
    """

    @staticmethod
    def compare_with_est_sfs(
            anc: fd.MaximumLikelihoodAncestralAnnotation
    ) -> (_ESTSFSAncestralAnnotation, pd.DataFrame):
        """
        Compare the results of the MaximumLikelihoodAncestralAnnotation class with the results of EST-SFS.
        """
        est_sfs = _ESTSFSAncestralAnnotation(anc)

        if anc.prior is None:
            binary = "resources/EST-SFS/cmake-build-debug/EST_SFS_no_prior"
        else:
            binary = "resources/EST-SFS/cmake-build-debug/EST_SFS_with_prior"

        est_sfs.infer(binary=binary)

        # get site information
        site_info = pd.DataFrame(anc.get_inferred_site_info())

        # prefix columns
        site_info.columns = ['native.' + col for col in site_info.columns]

        # add EST-SFS results to site information
        site_info['est_sfs.config'] = est_sfs.probs.config
        site_info['est_sfs.p_major_ancestral'] = est_sfs.probs.prob

        return est_sfs, site_info

    def test_negative_lower_bounds_raises_value_error(self):
        """
        Test that a ValueError is raised when the lower bound of a parameter is negative.
        """
        with self.assertRaises(ValueError):
            fd.MaximumLikelihoodAncestralAnnotation(
                outgroups=["ERR2103730", "ERR2103731"],
                n_runs=3,
                n_ingroups=5,
                model=fd.JCSubstitutionModel(bounds=dict(k=(-1, 1)))
            )

    def test_get_p_tree_one_outgroup(self):
        """
        Test the get_p_tree function with one outgroup.
        """
        params = dict(
            k=2,
            K0=0.5
        )

        model = fd.K2SubstitutionModel()

        for base, outgroup_base in itertools.product(range(4), range(4)):
            p = fd.MaximumLikelihoodAncestralAnnotation.get_p_tree(
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

            p_expected = model._get_prob(
                b1=base,
                b2=outgroup_base,
                i=0,
                params=params
            )

            self.assertEqual(p, p_expected)

    def test_get_p_tree_two_outgroups(self):
        """
        Test the get_p_tree function with two outgroups.
        """
        params = dict(
            k=2,
            K0=0.5,
            K1=0.25,
            K2=0.125
        )

        model = fd.K2SubstitutionModel()

        for base, outgroup_base1, outgroup_base2, internal_node in itertools.product(range(4), repeat=4):
            p = fd.MaximumLikelihoodAncestralAnnotation.get_p_tree(
                base=base,
                n_outgroups=2,
                internal_nodes=[internal_node],
                outgroup_bases=[outgroup_base1, outgroup_base2],
                params=params,
                model=model
            )

            p_expected = (model._get_prob(base, internal_node, 0, params) *
                          model._get_prob(internal_node, outgroup_base1, 1, params) *
                          model._get_prob(internal_node, outgroup_base2, 2, params))

            self.assertEqual(p, p_expected)

    def test_get_p_site_three_outgroups(self):
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

        model = fd.K2SubstitutionModel()

        for base, out1, out2, out3, int_node1, int_node2 in itertools.product(range(4), repeat=6):
            p = fd.MaximumLikelihoodAncestralAnnotation.get_p_tree(
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

            p_expected = (model._get_prob(base, int_node1, 0, params) *
                          model._get_prob(int_node1, out1, 1, params) *
                          model._get_prob(int_node1, int_node2, 2, params) *
                          model._get_prob(int_node2, out2, 3, params) *
                          model._get_prob(int_node2, out3, 4, params))

            self.assertEqual(p, p_expected)

    def test_fixed_params_different_branch_rates(self):
        """
        Test the MaximumLikelihoodAncestralAnnotation class with fixed parameters and different branch rates.
        """
        anc = fd.MaximumLikelihoodAncestralAnnotation.from_est_sfs(
            file="resources/EST-SFS/test-data.txt",
            n_runs=10,
            parallelize=False,
            model=fd.K2SubstitutionModel(fixed_params=dict(k=2, K0=0.5, K2=0.125))
        )

        anc.infer()

        self.assertEqual(anc.params_mle['k'], 2)
        self.assertEqual(anc.params_mle['K0'], 0.5)
        self.assertEqual(anc.params_mle['K2'], 0.125)
        self.assertNotEqual(anc.params_mle['K1'], 0.5)
        self.assertNotEqual(anc.params_mle['K3'], 0.125)

    def test_fixed_params_same_branch_rates(self):
        """
        Test the MaximumLikelihoodAncestralAnnotation class with fixed parameters and same branch rates.
        """
        anc = fd.MaximumLikelihoodAncestralAnnotation.from_est_sfs(
            file="resources/EST-SFS/test-data.txt",
            n_runs=10,
            parallelize=True,
            model=fd.K2SubstitutionModel(fixed_params=dict(k=2, K=0.5), pool_branch_rates=True)
        )

        anc.infer()

        self.assertEqual(anc.params_mle['k'], 2)
        self.assertEqual(anc.params_mle['K'], 0.5)

    @staticmethod
    def test_expected_ancestral_alleles_fixed_branch_rate_inferred_site_info_kingman_prior():
        """
        Test the expected ancestral alleles with fixed branch rates and inferred site information.
        """
        configs = [
            dict(n_major=15, major_base='A', minor_base='C', outgroup_bases=['A', '.', '.'], ancestral_expected='A'),
            dict(n_major=15, major_base='G', minor_base=None, outgroup_bases=['G', '.', '.'], ancestral_expected='G'),
            dict(n_major=15, major_base='C', minor_base='A', outgroup_bases=['.', '.', '.'], ancestral_expected='C'),
            dict(n_major=15, major_base='A', minor_base='T', outgroup_bases=['A', 'T', '.'], ancestral_expected='A'),
            dict(n_major=15, major_base='A', minor_base='T', outgroup_bases=['T', 'A', '.'], ancestral_expected='A'),
            dict(n_major=15, major_base='T', minor_base='C', outgroup_bases=['T', 'C', 'C'], ancestral_expected='T'),
            dict(n_major=15, major_base='T', minor_base='C', outgroup_bases=['G', '.', '.'], ancestral_expected='T'),
            dict(n_major=15, major_base='T', minor_base='C', outgroup_bases=['C', 'T', '.'], ancestral_expected='T'),
            dict(n_major=15, major_base='T', minor_base='C', outgroup_bases=['T', 'C', 'C'], ancestral_expected='T'),
            dict(n_major=15, major_base='G', minor_base=None, outgroup_bases=['A', 'C', 'T'], ancestral_expected='G'),
            dict(n_major=15, major_base=None, minor_base=None, outgroup_bases=['A', 'C', 'T'], ancestral_expected='.'),
            dict(n_major=15, major_base=None, minor_base=None, outgroup_bases=['.', '.', '.'], ancestral_expected='.'),

            # this is counterintuitive, but we can only consider bases that are present in the ingroup
            dict(n_major=15, major_base='A', minor_base=None, outgroup_bases=['T', 'T', '.'], ancestral_expected='A'),
            dict(n_major=15, major_base='A', minor_base=None, outgroup_bases=['T', 'T', 'T'], ancestral_expected='A'),

            # this works because 'T' is present in the ingroup
            dict(n_major=15, major_base='A', minor_base='T', outgroup_bases=['T', 'T', 'T'], ancestral_expected='T'),
            dict(n_major=20, major_base='A', minor_base='T', outgroup_bases=['T', 'T', 'T'], ancestral_expected='T'),

            # this again doesn't work because 'T' is not present in the ingroup
            dict(n_major=15, major_base='A', minor_base='G', outgroup_bases=['T', 'T', 'T'], ancestral_expected='A'),
        ]

        anc = fd.MaximumLikelihoodAncestralAnnotation.from_data(
            n_major=[c['n_major'] for c in configs],
            major_base=[c['major_base'] for c in configs],
            minor_base=[c['minor_base'] for c in configs],
            outgroup_bases=[c['outgroup_bases'] for c in configs],
            n_ingroups=20,
            prior=fd.KingmanPolarizationPrior(),
            model=fd.JCSubstitutionModel(pool_branch_rates=True, fixed_params=dict(K=0.1)),
            parallelize=False
        )

        anc.infer()

        summary = pd.DataFrame(anc.get_inferred_site_info())

        summary['expected'] = [c['ancestral_expected'] for c in configs]

        testing.assert_array_equal(summary.major_ancestral.values, summary.expected.values)

    def test_raises_error_when_zero_outgroups_given(self):
        """
        Test that an error is raised when zero outgroups are given.
        """
        with self.assertRaises(ValueError):
            fd.MaximumLikelihoodAncestralAnnotation(
                outgroups=[],
                n_ingroups=10
            )

    def test_outgroup_not_found_raises_error(self):
        """
        Test that an error is raised when an outgroup is not found.
        """
        with self.assertRaises(ValueError) as context:
            anc = fd.MaximumLikelihoodAncestralAnnotation(
                outgroups=["ERR2103730", "blabla"],
                n_ingroups=10
            )

            ann = fd.Annotator(
                vcf="resources/genome/betula/all.with_outgroups.subset.10000.vcf.gz",
                output='scratch/test_outgroup_not_found_raises_error.vcf',
                annotations=[anc],
                max_sites=10
            )

            ann.annotate()

        # Print the caught error message
        print("Caught error: " + str(context.exception))

    def test_more_ingroups_than_ingroup_samples_raises_warning(self):
        """
        Test that a warning is raised when more ingroups are specified than ingroup samples are present.
        """
        with self.assertLogs(level="WARNING", logger=logging.getLogger('fastdfe')):
            anc = fd.MaximumLikelihoodAncestralAnnotation(
                ingroups=["ASP04", "ASP05"],
                outgroups=["ERR2103730", "ERR2103731"],
                n_ingroups=5
            )

            ann = fd.Annotator(
                vcf="resources/genome/betula/all.with_outgroups.subset.10000.vcf.gz",
                output='scratch/test_fewer_ingroups_than_ingroup_samples_raises_warning.vcf',
                annotations=[anc],
                max_sites=10
            )

            ann.annotate()

    @staticmethod
    def test_explicitly_specified_present_samples_raises_no_error():
        """
        Test that an error is raised when an outgroup is not found.
        """
        anc = fd.MaximumLikelihoodAncestralAnnotation(
            ingroups=["ASP04", "ASP05"],
            outgroups=["ERR2103730", "ERR2103731"],
            n_ingroups=2
        )

        ann = fd.Annotator(
            vcf="resources/genome/betula/all.with_outgroups.subset.10000.vcf.gz",
            output='scratch/test_explicitly_specified_present_samples_raises_no_error.vcf',
            annotations=[anc],
            max_sites=10
        )

        ann.annotate()

    def test_exclude_ingroups_no_implicit_ingroups(self):
        """
        Test that the exclude parameter works.
        """
        anc = fd.MaximumLikelihoodAncestralAnnotation(
            exclude=["ASP04", "ASP05", "ASP06"],
            outgroups=["ERR2103730", "ERR2103731"],
            n_ingroups=2,
        )

        ann = fd.Annotator(
            vcf="resources/genome/betula/all.with_outgroups.subset.10000.vcf.gz",
            output='scratch/test_explicitly_specified_present_samples_raises_no_error.vcf',
            annotations=[anc],
            max_sites=10
        )

        ann.annotate()

        self.assertEqual(anc._ingroup_mask.sum(), 374)

    def test_exclude_ingroups_no_explicit_ingroups(self):
        """
        Test that the exclude parameter works.
        """
        anc = fd.MaximumLikelihoodAncestralAnnotation(
            ingroups=["ASP01", "ASP02", "ASP03", "ASP04", "ASP05", "ASP06"],
            exclude=["ASP04", "ASP05", "ASP06"],
            outgroups=["ERR2103730", "ERR2103731"],
            n_ingroups=2,
        )

        ann = fd.Annotator(
            vcf="resources/genome/betula/all.with_outgroups.subset.10000.vcf.gz",
            output='scratch/test_explicitly_specified_present_samples_raises_no_error.vcf',
            annotations=[anc],
            max_sites=10
        )

        ann.annotate()

        self.assertEqual(anc._ingroup_mask.sum(), 3)

    def test_get_likelihood_outgroup_ancestral_allele_annotation(self):
        """
        Test the get_likelihood function.
        """
        anc = fd.MaximumLikelihoodAncestralAnnotation(
            outgroups=["ERR2103730", "ERR2103731"],
            n_ingroups=10,
            prior=None
        )

        ann = fd.Annotator(
            vcf="resources/genome/betula/all.with_outgroups.subset.10000.vcf.gz",
            output='scratch/test_get_likelihood_outgroup_ancestral_allele_annotation.vcf',
            annotations=[anc],
            max_sites=10000
        )

        ann.annotate()

        self.assertEqual(anc.evaluate_likelihood(anc.params_mle), anc.likelihood)

    @pytest.mark.slow
    def test_run_inference_full_betula_dataset(self):
        """
        Test the get_likelihood function.
        """
        anc = fd.MaximumLikelihoodAncestralAnnotation(
            outgroups=["ERR2103730", "ERR2103731"],
            n_ingroups=10,
            prior=None,
            model=fd.K2SubstitutionModel(),
            parallelize=True
        )

        ann = fd.Annotator(
            vcf="resources/genome/betula/biallelic.with_outgroups.vcf.gz",
            output='scratch/test_get_likelihood_full_betula_dataset.vcf',
            annotations=[anc],
            max_sites=100000
        )

        ann.annotate()

        anc.to_est_sfs("resources/EST-SFS/test-betula-biallelic-100000.txt")

        self.assertEqual(anc.evaluate_likelihood(anc.params_mle), anc.likelihood)

    @staticmethod
    def test_from_est_sfs_est_sfs_sample_dataset():
        """
        Test the from_est_sfs function with the est-sfs sample dataset.
        """
        anc = fd.MaximumLikelihoodAncestralAnnotation.from_est_sfs(
            file="resources/EST-SFS/test-data.txt",
            model=fd.JCSubstitutionModel(),
            n_runs=10,
            prior=fd.AdaptivePolarizationPrior(),
            parallelize=True
        )

        # evaluate a ML estimates of EST-SFS
        ll = anc.evaluate_likelihood(dict(
            K0=0.000000,
            K1=0.061141,
            K2=0.000000,
            K3=0.019841,
            K4=0.019863
        ))

        pass

    def test_from_est_sfs_input_valid_probs(self):
        """
        Test that the probabilities are between 0 and 1 when using a prior.
        """
        anc = fd.MaximumLikelihoodAncestralAnnotation.from_est_sfs(
            file="resources/EST-SFS/test-data.txt",
            model=fd.JCSubstitutionModel(pool_branch_rates=True),
            n_runs=10,
            prior=fd.KingmanPolarizationPrior(),
            parallelize=False
        )

        anc.infer()

        self.assertTrue(anc.configs.p_major_ancestral.between(0, 1).all())

    @staticmethod
    def test_from_est_sfs_varied_chunk_sizes():
        """
        Test that the chunked and non-chunked versions of the from_est_sfs function return the same
        results across multiple chunk sizes.
        """
        test_sizes = [1, 2, 5, 11, 30]
        reference_chunk_size = 100

        # Create a DataFrame using the reference_chunk_size
        anc_reference = fd.MaximumLikelihoodAncestralAnnotation.from_est_sfs(
            file="resources/EST-SFS/test-data.txt",
            chunk_size=reference_chunk_size
        )

        df_reference = anc_reference.configs.sort_values(
            by=['n_major', 'major_base', 'minor_base', 'outgroup_bases']
        ).reset_index(drop=True).sort_index(axis=1)

        for chunk_size in test_sizes:
            anc_test = fd.MaximumLikelihoodAncestralAnnotation.from_est_sfs(
                file="resources/EST-SFS/test-data.txt",
                chunk_size=chunk_size
            )

            df_test = anc_test.configs.sort_values(
                by=['n_major', 'major_base', 'minor_base', 'outgroup_bases']
            ).reset_index(drop=True).sort_index(axis=1)

            assert_frame_equal(df_reference, df_test, check_dtype=False)

    def test_to_est_sfs_test_data_no_poly_allelic(self):
        """
        Test the to_est_sfs function.
        """
        file_in = "resources/EST-SFS/test-data-no-poly-allelic.txt"
        file_out = "scratch/test_to_est_sfs.txt"

        anc = fd.MaximumLikelihoodAncestralAnnotation.from_est_sfs(
            file=file_in,
            model=fd.JCSubstitutionModel(pool_branch_rates=True),
            n_runs=10,
            prior=fd.AdaptivePolarizationPrior(),
            parallelize=False
        )

        anc.to_est_sfs(file_out)

        # compare files
        with open(file_in, 'r') as f1, open(file_out, 'r') as f2:
            for line1, line2 in zip(f1, f2):
                self.assertEqual(line1.strip(), line2.strip())

    def test_to_est_sfs_betula_biallelic(self):
        """
        Test the to_est_sfs function.
        """
        file_in = "resources/EST-SFS/test-betula-biallelic-10000.txt"
        file_out = "scratch/test_to_est_sfs_betula_biallelic_10000.txt"

        anc = fd.MaximumLikelihoodAncestralAnnotation.from_est_sfs(
            file=file_in,
            model=fd.JCSubstitutionModel(pool_branch_rates=True),
            n_runs=10,
            prior=fd.AdaptivePolarizationPrior(),
            parallelize=False
        )

        anc.to_est_sfs(file_out)

        # compare files
        with open(file_in, 'r') as f1, open(file_out, 'r') as f2:
            for i, (line1, line2) in enumerate(zip(f1, f2)):
                self.assertEqual(line1.strip(), line2.strip())

    def test_parallelize_unequal_likelihoods(self):
        """
        Test that the parallelize function works correctly when the likelihoods are not equal.
        """
        anc = fd.MaximumLikelihoodAncestralAnnotation.from_est_sfs(
            file="resources/EST-SFS/test-data.txt",
            n_runs=10,
            parallelize=True
        )

        anc.infer()

        self.assertFalse(np.all(anc.likelihoods[0] == anc.likelihoods))

    def test_from_data_chunked(self):
        """
        Test that the from_data function produces the expected results.
        """
        anc = fd.MaximumLikelihoodAncestralAnnotation.from_data(
            n_major=[13, 15, 17, 11],
            major_base=['A', 'C', 'G', 'T'],
            minor_base=['C', 'G', 'T', 'A'],
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

    def test_upper_bounds_larger_than_lower_bounds_raises_value_error(self):
        """
        Test that a ValueError is raised when the lower bound of a parameter is negative.
        """
        with self.assertRaises(ValueError) as context:
            fd.JCSubstitutionModel(bounds=dict(K=(10, 9)))

        # Print the caught error message
        print("Caught error: " + str(context.exception))

    def test_zero_lower_bound_raises_value_error(self):
        """
        Test that a ValueError is raised when the lower bound of a parameter is zero.
        """
        with self.assertRaises(ValueError) as context:
            fd.JCSubstitutionModel(bounds=dict(K=(0, 9)))

        # Print the caught error message
        print("Caught error: " + str(context.exception))

    @pytest.mark.slow
    def test_papio_thorough_two_outgroups(self):
        """
        Test the MLEAncestralAlleleAnnotation class on the Papio vcf file.
        """
        samples = pd.read_csv("resources/genome/papio/metadata.csv")

        anc = fd.MaximumLikelihoodAncestralAnnotation(
            ingroups=list(samples[samples.C_origin == 'Anubis, Tanzania']['PGDP_ID']),
            outgroups=[
                samples[samples.Species == 'kindae'].iloc[0].PGDP_ID.replace('Sci_', ''),
                samples[samples.Species == 'gelada'].iloc[0].PGDP_ID.replace('Sci_', '')
            ],
            n_runs=10,
            n_ingroups=10,
            parallelize=True,
            prior=fd.KingmanPolarizationPrior(),
            confidence_threshold=0,
            max_sites=10000
        )

        ann = fd.Annotator(
            vcf="resources/genome/papio/output.filtered.snps.chr1.removed.AB.pass.vep.vcf.gz",
            output='scratch/test_papio_thorough_two_outgroups.vcf',
            annotations=[anc],
            max_sites=10000
        )

        ann.annotate()

        self.assertTrue(anc.is_monotonic())

        pass

    @pytest.mark.slow
    def test_papio_thorough_three_outgroups(self):
        """
        Test the MLEAncestralAlleleAnnotation class on the Papio vcf file.
        """
        samples = pd.read_csv("resources/genome/papio/metadata.csv")

        anc = fd.MaximumLikelihoodAncestralAnnotation(
            ingroups=list(samples[samples.C_origin == 'Anubis, Tanzania']['PGDP_ID']),
            outgroups=[
                samples[samples.C_origin == 'Anubis, Ethiopia'].iloc[0].PGDP_ID,
                samples[samples.Species == 'kindae'].iloc[0].PGDP_ID.replace('Sci_', ''),
                samples[samples.Species == 'gelada'].iloc[0].PGDP_ID.replace('Sci_', '')
            ],
            n_runs=10,
            n_ingroups=10,
            parallelize=True,
            prior=fd.KingmanPolarizationPrior(),
            confidence_threshold=0,
            max_sites=10000
        )

        ann = fd.Annotator(
            vcf="resources/genome/papio/output.filtered.snps.chr1.removed.AB.pass.vep.vcf.gz",
            output='scratch/test_papio_thorough_three_outgroups.vcf',
            annotations=[anc],
            max_sites=10000
        )

        # mismatches mostly occur where we are not very confident in the ancestral allele
        ann.annotate()

        anc2, site_info = self.compare_with_est_sfs(anc)

        diff_params = np.array(list(anc2.params_mle.values())) / np.array(list(anc.params_mle.values()))

        # mle estimates are very similar
        self.assertTrue(np.all(((0.6 < diff_params) & (diff_params < 1.5)) | (diff_params == 0)))

        self.assertTrue(anc.is_monotonic())

        pass

    @pytest.mark.slow
    def test_papio_thorough_three_outgroups_adaptive_prior(self):
        """
        Test the MLEAncestralAlleleAnnotation class on the Papio vcf file.
        """
        samples = pd.read_csv("resources/genome/papio/metadata.csv")

        anc = fd.MaximumLikelihoodAncestralAnnotation(
            ingroups=list(samples[samples.C_origin == 'Anubis, Tanzania']['PGDP_ID']),
            outgroups=[
                samples[samples.C_origin == 'Anubis, Ethiopia'].iloc[0].PGDP_ID,
                samples[samples.Species == 'kindae'].iloc[0].PGDP_ID.replace('Sci_', ''),
                samples[samples.Species == 'gelada'].iloc[0].PGDP_ID.replace('Sci_', '')
            ],
            n_runs=10,
            n_ingroups=10,
            parallelize=True,
            prior=fd.AdaptivePolarizationPrior(),
            confidence_threshold=0,
            max_sites=10000
        )

        ann = fd.Annotator(
            vcf="resources/genome/papio/output.filtered.snps.chr1.removed.AB.pass.vep.vcf.gz",
            output='scratch/test_papio_thorough_three_outgroups.vcf',
            annotations=[anc],
            max_sites=10000
        )

        # mismatches mostly occur where we are not very confident in the ancestral allele
        ann.annotate()

        self.assertTrue(anc.is_monotonic())

        pass

    @pytest.mark.skip("Too slow")
    def test_papio_thorough_four_outgroups(self):
        """
        Test the MLEAncestralAlleleAnnotation class on the Papio vcf file.
        """
        samples = pd.read_csv("resources/genome/papio/metadata.csv")

        anc = fd.MaximumLikelihoodAncestralAnnotation(
            ingroups=list(samples[samples.C_origin == 'Anubis, Tanzania']['PGDP_ID']),
            outgroups=[
                samples[samples.C_origin == 'Anubis, Ethiopia'].iloc[0].PGDP_ID,
                samples[samples.Species == 'hamadryas'].iloc[0].PGDP_ID.replace('Sci_', ''),
                samples[samples.Species == 'kindae'].iloc[0].PGDP_ID.replace('Sci_', ''),
                samples[samples.Species == 'gelada'].iloc[0].PGDP_ID.replace('Sci_', '')
            ],
            n_runs=10,
            n_ingroups=10,
            parallelize=True,
            prior=fd.KingmanPolarizationPrior(),
            confidence_threshold=0,
            max_sites=10000
        )

        ann = fd.Annotator(
            vcf="resources/genome/papio/output.filtered.snps.chr1.removed.AB.pass.vep.vcf.gz",
            output='scratch/test_papio_thorough_four_outgroups.vcf',
            annotations=[anc],
            max_sites=10000
        )

        # mismatches mostly occur where we are not very confident in the ancestral allele
        ann.annotate()

        self.assertTrue(anc.is_monotonic())

        pass

    @pytest.mark.slow
    def test_pendula_thorough_two_outgroups(self):
        """
        Test the MLEAncestralAlleleAnnotation class on the Betula pendula vcf file.
        """
        anc = fd.MaximumLikelihoodAncestralAnnotation(
            outgroups=["ERR2103730", "ERR2103731"],
            n_runs=50,
            n_ingroups=10,
            parallelize=True,
            prior=fd.KingmanPolarizationPrior()
        )

        ann = fd.Annotator(
            vcf="resources/genome/betula/all.with_outgroups.vcf.gz",
            output='scratch/test_pendula_thorough.vcf',
            annotations=[anc],
            max_sites=100000
        )

        ann.annotate()

        anc.evaluate_likelihood(anc.params_mle)

        anc.plot_likelihoods()

        # the two outgroup have a very similar divergence from the ingroup
        # self.assertTrue(anc._is_monotonic())

        pass

    @pytest.mark.slow
    def test_pendula_thorough_one_outgroup(self):
        """
        Test the MLEAncestralAlleleAnnotation class on the Betula pendula vcf file.
        """
        anc = fd.MaximumLikelihoodAncestralAnnotation(
            outgroups=["ERR2103730"],
            n_runs=50,
            n_ingroups=10,
            parallelize=True,
            prior=fd.KingmanPolarizationPrior()
        )

        ann = fd.Annotator(
            vcf="resources/genome/betula/all.with_outgroups.vcf.gz",
            output='scratch/test_pendula_thorough.vcf',
            annotations=[anc],
            max_sites=100000
        )

        ann.annotate()

        anc.evaluate_likelihood(anc.params_mle)

        anc.plot_likelihoods()

        self.assertTrue(anc.is_monotonic())

        pass

    @staticmethod
    def test_pendula_K2_model():
        """
        Test the MLEAncestralAlleleAnnotation class on the Betula pendula vcf file.
        """
        anc = fd.MaximumLikelihoodAncestralAnnotation(
            outgroups=["ERR2103730", "ERR2103731"],
            n_runs=10,
            n_ingroups=5,
            model=fd.K2SubstitutionModel(),
            prior=fd.KingmanPolarizationPrior()
        )

        ann = fd.Annotator(
            vcf="resources/genome/betula/all.with_outgroups.subset.10000.vcf.gz",
            output='scratch/test_pendula_use_prior_K2_model.vcf',
            annotations=[anc],
            max_sites=1000
        )

        ann.annotate()

        pass

    @staticmethod
    def test_pendula_JC_model():
        """
        Test the MLEAncestralAlleleAnnotation class on the Betula pendula vcf file.
        """
        anc = fd.MaximumLikelihoodAncestralAnnotation(
            outgroups=["ERR2103730", "ERR2103731"],
            n_runs=3,
            n_ingroups=5,
            model=fd.JCSubstitutionModel(),
            prior=fd.KingmanPolarizationPrior()
        )

        ann = fd.Annotator(
            vcf="resources/genome/betula/all.with_outgroups.subset.10000.vcf.gz",
            output='scratch/test_pendula_use_prior_JC_model.vcf',
            annotations=[anc],
            max_sites=1000
        )

        ann.annotate()

        pass

    @staticmethod
    def test_pendula_no_prior():
        """
        Test the MLEAncestralAlleleAnnotation class on the Betula pendula vcf file.
        """
        anc = fd.MaximumLikelihoodAncestralAnnotation(
            outgroups=["ERR2103730", "ERR2103731"],
            n_runs=3,
            n_ingroups=5,
            prior=None
        )

        ann = fd.Annotator(
            vcf="resources/genome/betula/all.with_outgroups.subset.10000.vcf.gz",
            output='scratch/test_pendula_not_use_prior.vcf',
            annotations=[anc],
            max_sites=1000
        )

        ann.annotate()

    @staticmethod
    def test_priors_betula_dataset():
        """
        Test the EST-SFS wrapper.
        """
        annotations = []
        priors = [fd.KingmanPolarizationPrior(), fd.AdaptivePolarizationPrior()]

        fig, ax = plt.subplots(1)

        for i, prior in enumerate(priors):
            anc = fd.MaximumLikelihoodAncestralAnnotation.from_est_sfs(
                file="resources/EST-SFS/test-betula-biallelic-10000.txt",
                model=fd.K2SubstitutionModel(),
                n_runs=10,
                prior=prior,
                parallelize=True
            )

            anc.infer()

            # touch p_polarization to make sure it is calculated
            _ = anc.p_polarization

            anc.prior.plot(ax=ax, show=False)

            annotations.append(anc)

        # set legend labels
        ax.legend([p.__class__.__name__ for p in priors])

        # set alpha
        for child in ax.get_children():
            if isinstance(child, PathCollection):
                child.set_alpha(0.5)

        plt.show()

        pass

    @staticmethod
    @pytest.mark.slow
    def test_compute_priors_from_betula_vcf_odd_even_number_of_ingroups():
        """
        Test the EST-SFS wrapper.
        """
        annotators = []
        priors = [fd.KingmanPolarizationPrior(), fd.AdaptivePolarizationPrior()]
        n_ingroups = [10, 11]

        fig, ax = plt.subplots(4, figsize=(10, 10))

        for i, (prior, n_ingroup) in enumerate(itertools.product(priors, n_ingroups)):
            anc = fd.MaximumLikelihoodAncestralAnnotation(
                outgroups=["ERR2103730", "ERR2103731"],
                n_ingroups=n_ingroup,
                prior=prior
            )

            ann = fd.Annotator(
                vcf="resources/genome/betula/all.with_outgroups.vcf.gz",
                output='scratch/test_fewer_ingroups_than_ingroup_samples_raises_error_{i}.vcf',
                max_sites=100000,
                annotations=[anc]
            )

            ann.annotate()

            anc.prior.plot(ax=ax[i], show=False)

            # set title
            ax[i].set_title(f"n_ingroups = {n_ingroup}, prior = {prior.__class__.__name__}")

            annotators.append(anc)

        plt.show()

        pass

    def test_compare_with_est_sfs_betula(self):
        """
        Compare MLE params and site probabilities with EST-SFS using the betula dataset.
        """
        annotators = []

        cases = [
            dict(
                prior=None,
                model=fd.JCSubstitutionModel(),
                tol_params=0.1,
                tol_sites=0.02,
                outgroups=["ERR2103730"],
                vcf="resources/genome/betula/all.with_outgroups.vcf.gz"
            ),
            dict(
                prior=None,
                model=fd.JCSubstitutionModel(),
                tol_params=0.6,
                tol_sites=0.02,
                outgroups=["ERR2103730", "ERR2103731"],
                vcf="resources/genome/betula/all.with_outgroups.vcf.gz"
            ),
            dict(
                prior=None,
                model=fd.JCSubstitutionModel(),
                tol_params=0.4,
                tol_sites=0.03,
                outgroups=["ERR2103730", "ERR2103731"],
                vcf="resources/genome/betula/biallelic.with_outgroups.vcf.gz"
            ),
            dict(
                prior=None,
                model=fd.K2SubstitutionModel(),
                tol_params=0.4,
                tol_sites=0.03,
                outgroups=["ERR2103730", "ERR2103731"],
                vcf="resources/genome/betula/biallelic.with_outgroups.vcf.gz"
            ),
            dict(
                prior=None,
                model=fd.JCSubstitutionModel(),
                tol_params=0.1,
                tol_sites=0.04,
                outgroups=["ERR2103730"],
                vcf="resources/genome/betula/biallelic.with_outgroups.vcf.gz"
            ),
            dict(
                prior=None,
                model=fd.K2SubstitutionModel(),
                tol_params=0.3,
                tol_sites=0.04,
                outgroups=["ERR2103730"],
                vcf="resources/genome/betula/biallelic.with_outgroups.vcf.gz"
            )
        ]

        max_sites = 10000
        n_ingroups = 11

        for i, case in enumerate(cases):
            anc = fd.MaximumLikelihoodAncestralAnnotation(
                outgroups=case['outgroups'],
                n_ingroups=n_ingroups,
                prior=case['prior'],
                model=case['model'],
                max_sites=max_sites
            )

            ann = fd.Annotator(
                vcf=case['vcf'],
                output='scratch/dummy.vcf',
                annotations=[anc],
                max_sites=max_sites
            )

            # set up to infer branch rates
            ann._setup()

            anc2, site_info = self.compare_with_est_sfs(anc)

            params_mle = np.array([[anc.params_mle[k], anc2.params_mle[k]] for k in anc.params_mle])

            diff_params = np.abs(params_mle[:, 0] - params_mle[:, 1]) / params_mle[:, 1]
            diff_params_max = diff_params.max()

            self.assertTrue(diff_params_max < case['tol_params'])

            # exclude sites where the major is fixed in the ingroup subsample
            site_info = site_info[site_info['native.n_major'] != n_ingroups]

            diff_sites = np.abs(site_info['native.p_major_ancestral'] - site_info['est_sfs.p_major_ancestral'])
            diff_sites_mean = diff_sites.mean()

            # these discrepancies are quite high, but for the sites for which the estimates differ the most,
            # EST-SFS's probabilities seem quite off. Even after double-checking I could not an explanation
            # for this and EST-SFS is a bit of a black box as its code is so poorly documented.
            self.assertTrue(diff_sites_mean < case['tol_sites'])

            annotators.append(anc)

        pass

    def test_compare_with_est_sfs_test_set(self):
        """
        Compare MLE params and site probabilities with EST-SFS using the EST-SFS test dataset.
        """
        for model in [fd.JCSubstitutionModel(), fd.K2SubstitutionModel()]:
            anc = fd.MaximumLikelihoodAncestralAnnotation.from_est_sfs(
                file="resources/EST-SFS/test-data-no-poly-allelic.txt",
                model=model,
                n_runs=10,
                prior=None,
                parallelize=True
            )

            anc.infer()

            est_sfs, site_info = self.compare_with_est_sfs(anc)

            params_native = anc.params_mle
            params_wrapper = est_sfs.params_mle

            likelihoods_native = anc.likelihoods
            likelihoods_wrapper = est_sfs.likelihoods

            testing.assert_almost_equal(
                list(params_native.values()),
                list(params_wrapper.values()),
                decimal=3
            )

            testing.assert_almost_equal(
                site_info['native.p_major_ancestral'].values,
                site_info['est_sfs.p_major_ancestral'].values,
                decimal=5
            )

        pass

    @staticmethod
    def test_get_outgroup_bases():
        """
        Test the _get_outgroup_bases function.
        """
        genotypes = np.array(["A|T", "./T", "C|G", ".|.", "T/T", "A|.", "N|G", "A|N"])
        n_outgroups = 8

        result = fd.MaximumLikelihoodAncestralAnnotation._get_outgroup_bases(genotypes, n_outgroups, minor_base='A')
        np.testing.assert_array_equal(result, np.array(['A', 'T', 'C', 'N', 'T', 'A', 'G', 'A']))

        result = fd.MaximumLikelihoodAncestralAnnotation._get_outgroup_bases(genotypes, n_outgroups, minor_base='T')
        np.testing.assert_array_equal(result, np.array(['T', 'T', 'C', 'N', 'T', 'A', 'G', 'A']))

    def test_get_base_index(self):
        """
        Test the get_base_index function.
        """
        bases = np.array(['A', 'C', 'G', 'T', 'N'])
        expected = np.array([0, 1, 2, 3, -1])

        result = fd.MaximumLikelihoodAncestralAnnotation.get_base_index(bases)

        np.testing.assert_array_equal(result, expected)

        for base in bases:
            self.assertEqual(
                fd.MaximumLikelihoodAncestralAnnotation.get_base_index(base),
                ['A', 'C', 'G', 'T'].index(base) if base in ['A', 'C', 'G', 'T'] else -1
            )

    def test_get_base_string(self):
        """
        Test the get_base_string function.
        """
        bases = np.array([0, 1, 2, 3, -1])
        expected = np.array(['A', 'C', 'G', 'T', '.'])

        result = fd.MaximumLikelihoodAncestralAnnotation.get_base_string(bases)

        np.testing.assert_array_equal(result, expected)

        for base in bases:
            self.assertEqual(
                fd.MaximumLikelihoodAncestralAnnotation.get_base_string(base),
                ['A', 'C', 'G', 'T', '.'][base]
            )

    def test_ad_hoc_annotation(self):
        """
        Test the ad hoc annotation.
        """
        configs = [
            dict(n_major=15, major_base='A', minor_base='C', outgroup_bases=['A'], ancestral_expected='A'),
            dict(n_major=15, major_base='G', minor_base=None, outgroup_bases=['G'], ancestral_expected='G'),
            dict(n_major=15, major_base='C', minor_base='A', outgroup_bases=[], ancestral_expected='C'),
            dict(n_major=15, major_base='A', minor_base='T', outgroup_bases=['A', 'T'], ancestral_expected='A'),
            dict(n_major=15, major_base='T', minor_base='C', outgroup_bases=['T', 'C', 'C'], ancestral_expected='C'),
            dict(n_major=15, major_base='T', minor_base='C', outgroup_bases=['G'], ancestral_expected='T'),
            dict(n_major=15, major_base='G', minor_base=None, outgroup_bases=['A', 'C', 'T'], ancestral_expected='G'),
            dict(n_major=15, major_base=None, minor_base=None, outgroup_bases=['A', 'C', 'T'], ancestral_expected='A'),
            dict(n_major=15, major_base='A', minor_base=None, outgroup_bases=['T', 'T', 'T'], ancestral_expected='T'),
            dict(n_major=15, major_base='A', minor_base=None, outgroup_bases=['T', 'T'], ancestral_expected='T'),
            dict(n_major=15, major_base='A', minor_base='T', outgroup_bases=['T', 'T', 'T'], ancestral_expected='T'),
            dict(n_major=15, major_base='A', minor_base='G', outgroup_bases=['T', 'T', 'T'], ancestral_expected='T'),
        ]

        for i, config in enumerate(configs):
            site_config = fd.annotation.SiteConfig(
                n_major=config['n_major'],
                major_base=fd.MaximumLikelihoodAncestralAnnotation.get_base_index(config['major_base']),
                minor_base=fd.MaximumLikelihoodAncestralAnnotation.get_base_index(config['minor_base']),
                outgroup_bases=fd.MaximumLikelihoodAncestralAnnotation.get_base_index(
                    np.array(config['outgroup_bases'])
                )
            )

            site_info = fd.annotation._AdHocAncestralAnnotation._get_site_information(site_config)

            self.assertEqual(site_info['ancestral_base'], config['ancestral_expected'])

    def test_p_config(self):
        """
        Test the p_config function.
        """
        configs = [
            dict(n_major=15, major_base='T', minor_base='C', outgroup_bases=['C', 'T'], p_larger='equal'),
            dict(n_major=15, major_base='T', minor_base='C', outgroup_bases=['.', '.'], p_larger='equal'),
            dict(n_major=15, major_base='A', minor_base='C', outgroup_bases=['A'], p_larger='major'),
            dict(n_major=15, major_base='G', minor_base=None, outgroup_bases=['G'], p_larger='major'),
            dict(n_major=15, major_base='C', minor_base='A', outgroup_bases=[], p_larger='equal'),
            dict(n_major=15, major_base='A', minor_base='T', outgroup_bases=['A', 'T'], p_larger='major'),
            dict(n_major=15, major_base='A', minor_base='T', outgroup_bases=['T', 'A'], p_larger='equal'),
            dict(n_major=15, major_base='T', minor_base='C', outgroup_bases=['T', 'C', 'C'], p_larger='major'),
            dict(n_major=15, major_base='T', minor_base='C', outgroup_bases=['C', 'T', 'C'], p_larger='minor'),
            dict(n_major=15, major_base='T', minor_base='C', outgroup_bases=['G'], p_larger='equal'),
            dict(n_major=15, major_base='G', minor_base=None, outgroup_bases=['A', 'C', 'T'], p_larger='major'),
            dict(n_major=15, major_base=None, minor_base=None, outgroup_bases=['A', 'C', 'T'], p_larger='equal'),
            dict(n_major=15, major_base='A', minor_base=None, outgroup_bases=['T', 'T', 'T'], p_larger='major'),
            dict(n_major=15, major_base='A', minor_base=None, outgroup_bases=['T', 'T'], p_larger='major'),
            dict(n_major=15, major_base='A', minor_base='T', outgroup_bases=['T', 'T', 'T'], p_larger='minor'),
            dict(n_major=15, major_base='A', minor_base='G', outgroup_bases=['T', 'T', 'T'], p_larger='equal'),
            dict(n_major=15, major_base='G', minor_base='C', outgroup_bases=['.', 'C'], p_larger='minor'),
        ]

        for i, config in enumerate(configs):
            probs = {}

            for name, focal_base in zip(['major', 'minor'],
                                        [fd.annotation.BaseType.MAJOR, fd.annotation.BaseType.MINOR]):
                probs[name] = fd.MaximumLikelihoodAncestralAnnotation.get_p_config(
                    config=fd.annotation.SiteConfig(
                        n_major=config['n_major'],
                        major_base=fd.MaximumLikelihoodAncestralAnnotation.get_base_index(config['major_base']),
                        minor_base=fd.MaximumLikelihoodAncestralAnnotation.get_base_index(config['minor_base']),
                        outgroup_bases=fd.MaximumLikelihoodAncestralAnnotation.get_base_index(
                            np.array(config['outgroup_bases'])
                        )
                    ),
                    base_type=focal_base,
                    params=dict(K=0.2),
                    model=fd.JCSubstitutionModel(pool_branch_rates=True)
                )

            if config['p_larger'] == 'major':
                self.assertGreater(probs['major'], probs['minor'])
            elif config['p_larger'] == 'minor':
                self.assertLess(probs['major'], probs['minor'])
            elif config['p_larger'] == 'equal':
                self.assertAlmostEqual(probs['major'], probs['minor'], places=10)

        pass

    @staticmethod
    def test_ancestral_first_inner_node():
        """
        Test the p_config function.
        """
        configs = pd.DataFrame([
            dict(n_major=15, major_base='T', minor_base='C', outgroup_bases=['C', 'T', 'C'], expected=['C']),
            dict(n_major=15, major_base='T', minor_base='C', outgroup_bases=['C', 'T', 'T'], expected=['C']),
            dict(n_major=15, major_base='T', minor_base='C', outgroup_bases=['C', 'T', 'T', 'T', 'T'], expected=['C']),
            dict(n_major=15, major_base='T', minor_base='C', outgroup_bases=['C', 'T'], expected=['C', 'T']),
            dict(n_major=15, major_base='T', minor_base='C', outgroup_bases=['.', '.'], expected=['C', 'T']),
            dict(n_major=15, major_base='A', minor_base='C', outgroup_bases=['A', '.'], expected=['A']),
            dict(n_major=15, major_base='C', minor_base='A', outgroup_bases=['A', '.'], expected=['A']),
            dict(n_major=15, major_base='A', minor_base='C', outgroup_bases=['.', 'A'], expected=['A']),
            dict(n_major=15, major_base='C', minor_base='A', outgroup_bases=['.', 'A'], expected=['A']),
            dict(n_major=15, major_base='C', minor_base='A', outgroup_bases=['C', 'A'], expected=['A', 'C']),
            dict(n_major=15, major_base='A', minor_base='C', outgroup_bases=['C', 'A'], expected=['A', 'C']),
            dict(n_major=19, major_base='C', minor_base='A', outgroup_bases=['.', 'A'], expected=['A']),
        ])

        for i, config in configs.iterrows():
            site = SiteConfig(
                n_major=config['n_major'],
                major_base=fd.MaximumLikelihoodAncestralAnnotation.get_base_index(config['major_base']),
                minor_base=fd.MaximumLikelihoodAncestralAnnotation.get_base_index(config['minor_base']),
                outgroup_bases=fd.MaximumLikelihoodAncestralAnnotation.get_base_index(
                    np.array(config['outgroup_bases'])
                )
            )

            anc = fd.MaximumLikelihoodAncestralAnnotation.from_data(
                n_major=[site.n_major],
                major_base=[site.major_base],
                minor_base=[site.minor_base],
                outgroup_bases=[[0 for _ in site.outgroup_bases]],
                n_ingroups=20,
                prior=fd.KingmanPolarizationPrior(),
                model=fd.JCSubstitutionModel(pool_branch_rates=True, fixed_params=dict(K=0.2)),
                pass_indices=True
            )

            # dummy inference, all parameters are fixed
            anc.infer()

            site_info = anc._get_site_info(site)

            probs = np.array(list(site_info.p_bases_first_node.values()))
            is_max = np.where(np.isclose(probs, np.max(probs)))[0]

            max_bases = np.array(list(site_info.p_bases_first_node.keys()))[is_max]

            np.testing.assert_array_equal(sorted(max_bases), sorted(config['expected']))

            pass

    def test_infer_variants_without_setup_raises_runtime_error(self):
        """
        Test that an error is raised when the infer function is called without setup.
        """
        with self.assertRaises(RuntimeError):
            anc = fd.MaximumLikelihoodAncestralAnnotation(
                outgroups=["ERR2103730", "ERR2103731"],
                n_ingroups=10,
                prior=None
            )

            anc.infer()

    def test_annotation_parse_variant(self):
        """
        Test that the annotation parses a mocked variant correctly.
        """
        # create a mocked annotation
        anc = fd.MaximumLikelihoodAncestralAnnotation(
            outgroups=["ERR2103730", "ERR2103731"],
            ingroups=["ASP04", "ASP05", "ASP06", "ASP07"],
            n_ingroups=8,
            prior=None
        )

        # create a mocked variant
        variant = Mock(spec=Variant)
        variant.REF = "A"
        variant.ALT = ["T"]
        variant.is_snp = True
        anc._prepare_masks(["ASP04", "ASP05", "ASP06", "ASP07", "ERR2103730", "ERR2103731"])
        variant.gt_bases = np.array(["A|A", "T|T", "A|T", "T|T", "T|T", "T|T"])

        # parse the mocked variant
        site = anc._parse_variant(variant)

        self.assertEqual(site.n_major, 5)
        self.assertEqual(site.major_base, base_indices['T'])
        self.assertEqual(site.minor_base, base_indices['A'])

    def test_annotation_parse_variant_minor_allele_zero_frequency_but_contained_in_ingroup(self):
        """
        Test that the annotation parses a mocked variant correctly when the minor allele has zero frequency but is
        contained in the ingroup.
        """
        # create a mocked annotation
        anc = fd.MaximumLikelihoodAncestralAnnotation(
            outgroups=["ERR2103730"],
            ingroups=["ASP04", "ASP05", "ASP06", "ASP07"],
            n_ingroups=4,
            prior=None
        )

        # create a mocked variant
        variant = Mock(spec=Variant)
        variant.REF = "A"
        variant.ALT = ["T"]
        variant.is_snp = True
        anc._prepare_masks(["ASP04", "ASP05", "ASP06", "ASP07", "ERR2103730"])
        variant.gt_bases = np.array(["A|T", "T|T", "T|T", "T|T", "T|A"])

        # parse the mocked variant
        site = anc._parse_variant(variant)

        self.assertEqual(site.n_major, 4)
        self.assertEqual(site.major_base, base_indices['T'])
        self.assertEqual(site.minor_base, base_indices['A'])

    def test_annotation_parse_variant_minor_allele_zero_frequency_but_contained_in_outgroup(self):
        """
        Test that the annotation parses a mocked variant correctly when the minor allele has zero frequency but is
        contained in the outgroup only.
        """
        # create a mocked annotation
        anc = fd.MaximumLikelihoodAncestralAnnotation(
            outgroups=["ERR2103730"],
            ingroups=["ASP04", "ASP05", "ASP06", "ASP07"],
            n_ingroups=4,
            prior=None
        )

        # create a mocked variant
        variant = Mock(spec=Variant)
        variant.REF = "A"
        variant.ALT = ["T"]
        variant.is_snp = True
        anc._prepare_masks(["ASP04", "ASP05", "ASP06", "ASP07", "ERR2103730"])
        variant.gt_bases = np.array(["T|T", "T|T", "T|T", "T|T", "T|A"])

        # parse the mocked variant
        site = anc._parse_variant(variant)

        self.assertEqual(site.n_major, 4)
        self.assertEqual(site.major_base, base_indices['T'])
        self.assertEqual(site.minor_base, base_indices['A'])

    @staticmethod
    def test_plot_tree_no_outgroup():
        """
        Test the plot_tree function.
        """
        SiteInfo(
            n_major=15,
            major_base='T',
            minor_base='C',
            outgroup_bases=[],
            rate_params=dict()
        ).plot_tree()

    @staticmethod
    def test_plot_tree_one_outgroup():
        """
        Test the plot_tree function.
        """
        SiteInfo(
            n_major=15,
            major_base='T',
            minor_base='C',
            outgroup_bases=['C'],
            rate_params=dict(K0=0.05, K1=0.3)
        ).plot_tree()

    @staticmethod
    def test_plot_tree_two_outgroups():
        """
        Test the plot_tree function.
        """
        SiteInfo(
            n_major=15,
            major_base='T',
            minor_base='C',
            outgroup_bases=['C', 'T'],
            rate_params=dict(K0=0.05, K1=0.3, K2=0.1)
        ).plot_tree()

    @staticmethod
    def test_plot_tree_three_outgroups():
        """
        Test the plot_tree function.
        """
        SiteInfo(
            n_major=15,
            major_base='T',
            minor_base='C',
            outgroup_bases=['C', 'T', 'C'],
            rate_params=dict(K0=0.05, K1=0.3, K2=0.1, K3=0.4, K4=0.2)
        ).plot_tree()

    @staticmethod
    def test_plot_tree_three_outgroups_pooled_rate_params():
        """
        Test the plot_tree function.
        """
        SiteInfo(
            n_major=15,
            major_base='T',
            minor_base='C',
            outgroup_bases=['C', 'T', 'C'],
            rate_params=dict(K=0.05)
        ).plot_tree()

    def test_maximum_likelihood_annotation_annotate_site(self):
        """
        Test the maximum parsimony annotation for a single site.
        """
        # test cases
        cases = [
            # minor allele 'G' but supported by outgroup
            dict(
                ingroups=["sample1", "sample2"],
                outgroups=["sample3"],
                samples=["sample1", "sample2", "sample3"],
                gt_bases=np.array(["A/A", "A/G", "G/G"]),
                expected="G",
                confidence_threshold=0,
                n_ingroups=4,
                prior=None
            ),
            # minor allele 'G', supported by outgroup, but not with enough confidence
            dict(
                ingroups=["sample1", "sample2"],
                outgroups=["sample3"],
                samples=["sample1", "sample2", "sample3"],
                gt_bases=np.array(["A/A", "A/G", "G/G"]),
                expected=".",
                confidence_threshold=1,
                n_ingroups=4,
                prior=None
            ),
            # minor allele 'G', supported by outgroup, but not enough ingroup samples
            dict(
                ingroups=["sample1", "sample2"],
                outgroups=["sample3"],
                samples=["sample1", "sample2", "sample3"],
                gt_bases=np.array(["A/A", "A/G", "G/G"]),
                expected=".",
                confidence_threshold=0,
                n_ingroups=5,
                prior=None
            ),
            # monomorphic site with outgroup information
            dict(
                ingroups=["sample1", "sample2"],
                outgroups=["sample3"],
                samples=["sample1", "sample2", "sample3"],
                gt_bases=np.array(["./.", "C/C", "C/C"]),
                expected="C",
                n_ingroups=2,
                confidence_threshold=0,
                prior=None
            ),
            # monomorphic site with no outgroup information
            dict(
                ingroups=["sample1", "sample2"],
                outgroups=["sample3"],
                samples=["sample1", "sample2", "sample3"],
                gt_bases=np.array(["T/T", "./.", "./."]),
                expected="T",
                n_ingroups=2,
                confidence_threshold=0,
                prior=None
            ),
            # site without any calls
            dict(
                ingroups=["sample1", "sample2"],
                outgroups=["sample3"],
                samples=["sample1", "sample2", "sample3"],
                gt_bases=np.array(["./.", "./.", "./."]),
                expected=".",
                n_ingroups=2,
                confidence_threshold=0,
                prior=None
            ),
            # poly-allelic site
            dict(
                ingroups=["sample1", "sample2"],
                outgroups=["sample3"],
                samples=["sample1", "sample2", "sample3"],
                gt_bases=np.array(["A/A", "./.", "G/T"]),
                expected=".",
                n_ingroups=2,
                confidence_threshold=0,
                prior=None
            ),
            # poly-allelic site
            dict(
                ingroups=["sample1", "sample2"],
                outgroups=["sample3"],
                samples=["sample1", "sample2", "sample3"],
                gt_bases=np.array(["A/T", "G/.", "T/T"]),
                expected=".",
                n_ingroups=3,
                confidence_threshold=0,
                prior=None
            ),
        ]

        # mock Annotator
        mock_annotator = MagicMock()
        type(mock_annotator).info_ancestral = PropertyMock(return_value="AA")

        for i, test_case in enumerate(cases):
            # create with dummy input and just fix branch rates
            ann = fd.MaximumLikelihoodAncestralAnnotation.from_data(
                n_major=[0],
                major_base=['A'],
                minor_base=['T'],
                outgroup_bases=[['A']],
                n_ingroups=test_case["n_ingroups"],
                model=fd.JCSubstitutionModel(pool_branch_rates=True, fixed_params=dict(K=0.1)),
                prior=test_case["prior"],
                confidence_threshold=test_case["confidence_threshold"]
            )

            ann.outgroups = test_case["outgroups"]
            ann.ingroups = test_case["ingroups"]
            ann.handler = mock_annotator

            # prepare masks
            ann._prepare_masks(test_case["samples"])

            # infer branch rates (which has no effect)
            ann.infer()

            # mock variant with a dictionary for INFO
            variant = MagicMock()
            variant.gt_bases = test_case["gt_bases"]
            variant.INFO = {}
            variant.is_snp = len(set(get_called_bases(variant.gt_bases))) > 1
            variant.REF = 'A'  # doesn't matter what the reference is, but it needs to be a valid base

            # run the method
            ann.annotate_site(variant)

            # check if the result matches the expectation
            self.assertEqual(variant.INFO[ann.handler.info_ancestral], test_case["expected"])

    def test_is_confident(self):
        """
        Test the is_confident method.
        """
        self.assertEqual(True, fd.MaximumLikelihoodAncestralAnnotation._is_confident(0, 0.1))
        self.assertEqual(True, fd.MaximumLikelihoodAncestralAnnotation._is_confident(0, 0.5))
        self.assertEqual(True, fd.MaximumLikelihoodAncestralAnnotation._is_confident(0, 0.9))

        self.assertEqual(False, fd.MaximumLikelihoodAncestralAnnotation._is_confident(1, 0.1))
        self.assertEqual(False, fd.MaximumLikelihoodAncestralAnnotation._is_confident(1, 0.5))
        self.assertEqual(False, fd.MaximumLikelihoodAncestralAnnotation._is_confident(1, 0.9))

        self.assertEqual(True, fd.MaximumLikelihoodAncestralAnnotation._is_confident(0.5, 0.1))
        self.assertEqual(True, fd.MaximumLikelihoodAncestralAnnotation._is_confident(0.5, 0.2))
        self.assertEqual(False, fd.MaximumLikelihoodAncestralAnnotation._is_confident(0.5, 0.3))
        self.assertEqual(False, fd.MaximumLikelihoodAncestralAnnotation._is_confident(0.5, 0.5))
        self.assertEqual(False, fd.MaximumLikelihoodAncestralAnnotation._is_confident(0.5, 0.7))
        self.assertEqual(True, fd.MaximumLikelihoodAncestralAnnotation._is_confident(0.5, 0.8))
        self.assertEqual(True, fd.MaximumLikelihoodAncestralAnnotation._is_confident(0.5, 0.9))

    def test_get_outgroup_divergence_one_outgroup(self):
        """
        Test the get_outgroup_divergence method with one outgroup.
        """
        # create a mocked annotation
        anc = fd.MaximumLikelihoodAncestralAnnotation(
            outgroups=["outgroup1"],
            ingroups=["ingroup1", "ingroup2", "ingroup3", "ingroup4"],
            n_ingroups=8,
            prior=None
        )

        anc.params_mle = {
            'K0': 0.05
        }

        testing.assert_array_equal([0.05], anc.get_outgroup_divergence())

    def test_get_outgroup_divergence_two_outgroups(self):
        """
        Test the get_outgroup_divergence method with two outgroups.
        """
        # create a mocked annotation
        anc = fd.MaximumLikelihoodAncestralAnnotation(
            outgroups=["outgroup1", "outgroup2"],
            ingroups=["ingroup1", "ingroup2", "ingroup3", "ingroup4"],
            n_ingroups=8,
            prior=None
        )

        anc.params_mle = {
            'K0': 0.05,
            'K1': 0.1,
            'K2': 0.2
        }

        testing.assert_array_almost_equal([0.15, 0.25], anc.get_outgroup_divergence())

    def test_get_outgroup_divergence_three_outgroups(self):
        """
        Test the get_outgroup_divergence method with three outgroups.
        """
        # create a mocked annotation
        anc = fd.MaximumLikelihoodAncestralAnnotation(
            outgroups=["outgroup1", "outgroup2", "outgroup3"],
            ingroups=["ingroup1", "ingroup2", "ingroup3", "ingroup4"],
            n_ingroups=8,
            prior=None
        )

        anc.params_mle = {
            'K0': 0.05,
            'K1': 0.1,
            'K2': 0.13,
            'K3': 0.3,
            'K4': 0.4
        }

        testing.assert_array_almost_equal([0.15, 0.48, 0.58], anc.get_outgroup_divergence())

    @staticmethod
    @pytest.mark.slow
    def test_betula_different_confidence_thresholds():
        """
        Test the SFS for different ancestral allele annotation confidence thresholds.
        """
        spectra = {}

        for threshold in [0, 0.5, 0.9]:
            p = fd.Parser(
                vcf="resources/genome/betula/biallelic.with_outgroups.vcf.gz",
                n=10,
                max_sites=10000,
                annotations=[
                    fd.MaximumLikelihoodAncestralAnnotation(
                        outgroups=["ERR2103730"],
                        exclude=["ERR2103731"],
                        n_ingroups=20,
                        confidence_threshold=threshold,
                    )
                ]
            )

            sfs = p.parse()

            spectra[str(threshold)] = sfs.all

        # we have fewer derived alleles for higher confidence thresholds
        fd.Spectra.from_spectra(spectra).plot()

        pass

    @pytest.mark.slow
    def test_papio_sfs_for_different_number_of_outgroups(self):
        """
        Test the SFS for different numbers of outgroups.
        """
        spectra, parsers = {}, {}

        samples = pd.read_csv("resources/genome/papio/metadata.csv")

        outgroups = [
            # samples[samples.C_origin == 'Anubis, Ethiopia'].iloc[0].PGDP_ID,
            samples[samples.Species == 'hamadryas'].iloc[0].PGDP_ID.replace('Sci_', ''),
            samples[samples.Species == 'kindae'].iloc[0].PGDP_ID.replace('Sci_', ''),
            samples[samples.Species == 'gelada'].iloc[0].PGDP_ID.replace('Sci_', '')
        ]

        for n_outgroups in [1, 2, 3]:
            p = fd.Parser(
                vcf="resources/genome/papio/output.filtered.snps.chr1.removed.AB.pass.vep.vcf.gz",
                n=8,
                max_sites=10000,
                annotations=[
                    fd.MaximumLikelihoodAncestralAnnotation(
                        ingroups=list(samples[samples.C_origin == 'Anubis, Tanzania']['PGDP_ID']),
                        outgroups=outgroups[-n_outgroups:],
                        n_runs=10,
                        n_ingroups=10,
                        parallelize=True,
                        prior=fd.KingmanPolarizationPrior(),
                        confidence_threshold=0
                    )
                ]
            )

            sfs = p.parse()

            spectra[f"{n_outgroups}_outgroups"] = sfs.all
            parsers[f"{n_outgroups}_outgroups"] = p

        s = fd.Spectra.from_spectra(spectra)
        s.plot()

        # make sure total number of sites is the same
        self.assertTrue(np.all(s.n_sites == s.n_sites[0]))

        # The number of polymorphic sites is very similar as it should
        # If only one outgroup is used, there is a slight over-representation of high-frequency derived alleles
        pass

    @pytest.mark.slow
    def test_papio_sfs_for_different_subsample_size(self):
        """
        Test the SFS for different numbers of outgroups.
        """
        spectra, parsers = {}, {}

        samples = pd.read_csv("resources/genome/papio/metadata.csv")

        for n_ingroups in [5, 10, 20, 40]:
            p = fd.Parser(
                vcf="resources/genome/papio/output.filtered.snps.chr1.removed.AB.pass.vep.vcf.gz",
                n=8,
                max_sites=10000,
                annotations=[
                    fd.MaximumLikelihoodAncestralAnnotation(
                        ingroups=list(samples[samples.C_origin == 'Anubis, Tanzania']['PGDP_ID']),
                        outgroups=[
                            samples[samples.Species == 'hamadryas'].iloc[0].PGDP_ID.replace('Sci_', ''),
                            samples[samples.Species == 'kindae'].iloc[0].PGDP_ID.replace('Sci_', ''),
                            samples[samples.Species == 'gelada'].iloc[0].PGDP_ID.replace('Sci_', '')
                        ],
                        n_runs=10,
                        n_ingroups=n_ingroups,
                        parallelize=True,
                        prior=fd.KingmanPolarizationPrior(),
                        confidence_threshold=0,
                    )
                ]
            )

            sfs = p.parse()

            spectra[f"{n_ingroups}_ingroups"] = sfs.all
            parsers[f"{n_ingroups}_ingroups"] = p

        # looks rather similar really
        fd.Spectra.from_spectra(spectra).plot()

        pass

    @pytest.mark.slow
    def test_papio_sfs_for_different_priors(self):
        """
        Test the SFS for different numbers of outgroups.
        """
        spectra, parsers = {}, {}

        samples = pd.read_csv("resources/genome/papio/metadata.csv")

        priors = {
            'none': None,
            'Kingman': fd.KingmanPolarizationPrior(),
            'Adaptive': fd.AdaptivePolarizationPrior(),
        }

        for name, prior in priors.items():
            p = fd.Parser(
                vcf="resources/genome/papio/output.filtered.snps.chr1.removed.AB.pass.vep.vcf.gz",
                n=8,
                max_sites=10000,
                annotations=[
                    fd.MaximumLikelihoodAncestralAnnotation(
                        ingroups=list(samples[samples.C_origin == 'Anubis, Tanzania']['PGDP_ID']),
                        outgroups=[
                            samples[samples.Species == 'hamadryas'].iloc[0].PGDP_ID.replace('Sci_', ''),
                            samples[samples.Species == 'kindae'].iloc[0].PGDP_ID.replace('Sci_', ''),
                            # samples[samples.Species == 'gelada'].iloc[0].PGDP_ID.replace('Sci_', '')
                        ],
                        n_runs=10,
                        n_ingroups=20,
                        parallelize=True,
                        prior=prior,
                        confidence_threshold=0,
                    )
                ]
            )

            sfs = p.parse()

            spectra[name] = sfs.all
            parsers[name] = p

        # slightly more high-frequency derived alleles without prior, which is expected
        fd.Spectra.from_spectra(spectra).plot()

        pass

    @pytest.mark.slow
    def test_betula_biallelic_vs_monomorphic_compare_rates(self):
        """
        Test the SFS for different numbers of outgroups.
        """
        spectra, parsers = {}, {}

        vcfs = {
            'biallelic': "resources/genome/betula/biallelic.with_outgroups.vcf.gz",
            'all': "resources/genome/betula/all.with_outgroups.vcf.gz"
        }

        for key, vcf in vcfs.items():
            p = fd.Parser(
                vcf=vcf,
                n=10,
                annotations=[
                    fd.MaximumLikelihoodAncestralAnnotation(
                        outgroups=["ERR2103730"],
                        exclude=["ERR2103731"],
                        n_ingroups=20,
                        confidence_threshold=0,
                        prior=fd.KingmanPolarizationPrior()
                    )
                ]
            )

            sfs = p.parse()

            spectra[key] = sfs.all
            parsers[key] = p

        # extremely similar
        fd.Spectra.from_spectra(spectra).plot()

        self.assertEqual(
            parsers['all'].annotations[0].params_mle,
            parsers['biallelic'].annotations[0].params_mle
        )
