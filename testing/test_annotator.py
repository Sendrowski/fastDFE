import logging
from unittest.mock import patch, Mock

import pandas as pd

from fastdfe.vcf import count_sites, VCFHandler
from testing import prioritize_installed_packages

prioritize_installed_packages()

from unittest import TestCase
import pytest

from fastdfe import Annotator, MaximumParsimonyAnnotation, Parser, DegeneracyAnnotation, SynonymyAnnotation

logging.getLogger('fastdfe').setLevel(logging.INFO)


class AnnotatorTestCase(TestCase):
    """
    Test the annotators.
    """

    vcf_file = 'resources/genome/betula/biallelic.subset.10000.vcf.gz'
    fasta_file = 'resources/genome/betula/genome.subset.20.fasta'
    gff_file = 'resources/genome/betula/genome.gff.gz'

    def test_maximum_parsimony_annotation(self):
        """
        Test the maximum parsimony annotator.
        """
        ann = Annotator(
            vcf=self.vcf_file,
            output='scratch/test_maximum_parsimony_annotator.vcf',
            annotations=[MaximumParsimonyAnnotation()],
            info_ancestral='BB'
        )

        ann.annotate()

        Parser(self.vcf_file, 20, info_ancestral='BB').parse().plot(title="Original")
        Parser(ann.output, 20, info_ancestral='BB').parse().plot(title="Annotated")

        # assert number of sites is the same
        assert count_sites(self.vcf_file) == count_sites(ann.output)

    def test_degeneracy_annotation_human_test_genome(self):
        """
        Test the degeneracy annotator on a small human genome.
        """
        deg = DegeneracyAnnotation(
            fasta_file="https://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/chr21.fa.gz",
            gff_file="resources/genome/sapiens/hg38.gtf"
        )

        vcf = "resources/genome/sapiens/chr21_test.vcf.gz"

        ann = Annotator(
            vcf=vcf,
            output='scratch/test_degeneracy_annotation.vcf',
            annotations=[deg],
        )

        ann.annotate()

        # assert number of sites is the same
        assert count_sites(vcf) == count_sites(ann.output)

        # assert number of annotated sites and total number of sites
        assert deg.n_annotated == 7
        assert ann.n_sites == 1517

    def test_degeneracy_annotation_human_test_genome_remote_fasta_gzipped(self):
        """
        Test the degeneracy annotator on a small human genome with a remote gzipped fasta file.
        """
        deg = DegeneracyAnnotation(
            fasta_file="https://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/chr21.fa.gz",
            gff_file="resources/genome/sapiens/hg38.gtf"
        )

        vcf = "resources/genome/sapiens/chr21_test.vcf.gz"

        ann = Annotator(
            vcf=vcf,
            output='scratch/test_degeneracy_annotation.vcf',
            annotations=[deg],
        )

        ann.annotate()

        # assert number of annotated sites and total number of sites
        assert deg.n_annotated == 7
        assert ann.n_sites == 1517

    def test_degeneracy_annotation_betula_subset(self):
        """
        Test the degeneracy annotator.
        """
        vcf = "resources/genome/betula/all.subset.100000.vcf.gz"

        ann = Annotator(
            vcf=vcf,
            output='scratch/test_degeneracy_annotation.vcf',
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

    def test_annotator_load_vcf_from_url(self):
        """
        Test the annotator loading a VCF from a URL.
        """
        ann = Annotator(
            vcf="https://github.com/Sendrowski/fastDFE/blob/master/resources/genome/"
                "betula/biallelic.subset.10000.vcf.gz?raw=true",
            output='scratch/test_degeneracy_annotation.vcf',
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

    def test_compare_synonymy_annotation_with_vep_betula(self):
        """
        Compare the synonymy annotation with VEP.
        """
        syn = SynonymyAnnotation(
            fasta_file="resources/genome/betula/genome.subset.20.fasta",
            gff_file="resources/genome/betula/genome.gff.gz"
        )

        ann = Annotator(
            vcf="resources/genome/betula/biallelic.subset.10000.vcf.gz",
            output='scratch/test_compare_synonymy_annotation_with_vep.vcf',
            annotations=[syn],
        )

        ann.annotate()

        assert syn.n_vep_comparisons == 3566
        assert len(syn.vep_mismatches) == 0

    @pytest.mark.slow
    def test_compare_synonymy_annotation_with_vep_human_chr21(self):
        """
        Compare the synonymy annotation with VEP.
        """
        syn = SynonymyAnnotation(
            fasta_file="resources/genome/sapiens/chr21.fasta",
            gff_file="resources/genome/sapiens/hg38.sorted.gtf.gz"
        )

        ann = Annotator(
            vcf="snakemake/results/vcf/sapiens/chr21.vep.vcf.gz",
            output="test_compare_synonymy_annotation_with_vep_human_chr21.vcf",
            annotations=[syn]
        )

        ann.annotate()

        pass

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
