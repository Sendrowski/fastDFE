import logging

import pytest

from testing import prioritize_installed_packages

prioritize_installed_packages()

from unittest import TestCase

import dadi
import numpy as np

from fastdfe import DegeneracyStratification, BaseTransitionStratification, TransitionTransversionStratification, \
    BaseContextStratification, AncestralBaseStratification, Parser, DegeneracyAnnotation, MaximumParsimonyAnnotation, \
    CodingSequenceFiltration, BaseInference

logging.getLogger('fastdfe').setLevel(logging.INFO)


class ParserTestCase(TestCase):
    """
    Test the inference.
    """

    def test_compare_sfs_with_data(self):
        """
        Compare the sfs from dadi with the one from the data.
        """
        p = Parser(vcf='resources/genome/betula/biallelic.subset.10000.vcf.gz', n=20, stratifications=[], seed=2)

        sfs = p.parse().all

        sfs.plot()

        data_dict = dadi.Misc.make_data_dict_vcf(
            'resources/genome/betula/biallelic.subset.10000.vcf.gz',
            'resources/genome/betula/pops_dadi.txt'
        )

        sfs2 = dadi.Spectrum.from_data_dict(data_dict, ['pop0'], [20], polarized=True)

        diff_rel = np.abs(sfs.to_numpy() - sfs2.data) / sfs2.data

        # assert total number of sites
        assert sfs.data.sum() == 10000 - p.n_skipped

        # check that the sum of the sfs is the same
        # for some reason dadi skips two sites
        # self.assertAlmostEqual(sfs.data.sum(), sfs2.data.sum())

        # this is better for large VCF files
        assert np.max(diff_rel) < 0.8

    def test_degeneracy_stratification(self):
        """
        Test the degeneracy stratification.
        """
        p = Parser(
            vcf='resources/genome/betula/biallelic.subset.10000.vcf.gz',
            n=20,
            stratifications=[DegeneracyStratification()]
        )

        sfs = p.parse()

        sfs.plot()

        # assert total number of sites
        assert sfs.all.data.sum() == 10000 - p.n_skipped

    def test_base_transition_stratification(self):
        """
        Test the base transition stratification.
        """
        p = Parser(
            vcf='resources/genome/betula/all.with_outgroups.subset.10000.vcf.gz',
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
            vcf='resources/genome/betula/all.with_outgroups.subset.10000.vcf.gz',
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
            vcf='resources/genome/betula/biallelic.subset.10000.vcf.gz',
            n=20,
            stratifications=[BaseContextStratification(fasta_file='resources/genome/betula/genome.subset.20.fasta')]
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
            vcf='resources/genome/betula/biallelic.subset.10000.vcf.gz',
            n=20,
            stratifications=[AncestralBaseStratification()]
        )

        sfs = p.parse()

        sfs.plot()

        # assert total number of sites
        assert sfs.all.data.sum() == 10000 - p.n_skipped

    def test_parse_vcf_chr21(self):
        """
        Parse the VCF file using a remote fasta
        :return:
        """
        deg = DegeneracyAnnotation(
            fasta_file="http://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/chr21.fa.gz",
            gff_file="resources/genome/sapiens/hg38.sorted.gtf.gz"
        )

        aa = MaximumParsimonyAnnotation()

        f = CodingSequenceFiltration(
            gff_file="resources/genome/sapiens/hg38.sorted.gtf.gz"
        )

        p = Parser(
            vcf="resources/genome/sapiens/chr21_test.vcf.gz",
            n=20,
            ignore_not_polarized=True,
            annotations=[deg, aa],
            filtrations=[f],
            max_sites=100000
        )

        sfs = p.parse()

        assert sfs.all.data.sum() == 6

    def test_parse_betula_vcf_biallelic_adjust_mutational_target_sites(self):
        """
        Parse the VCF file of Betula spp.
        """
        p = Parser(
            vcf="resources/genome/betula/biallelic.subset.10000.vcf.gz",
            n=20,
            n_target_sites=1000000,
            annotations=[
                DegeneracyAnnotation(
                    fasta_file="resources/genome/betula/genome.subset.20.fasta",
                    gff_file="resources/genome/betula/genome.gff.gz"
                ),
                MaximumParsimonyAnnotation()
            ],
            filtrations=[
                CodingSequenceFiltration(
                    gff_file="resources/genome/betula/genome.gff.gz"
                )
            ]
        )

        sfs = p.parse()

        assert sfs.all.data.sum() == p.n_target_sites

    def test_parse_betula_vcf(self):
        """
        Parse the VCF file of Betula spp.
        """
        p = Parser(
            vcf="resources/genome/betula/all.subset.100000.vcf.gz",
            n=20,
            annotations=[
                DegeneracyAnnotation(
                    fasta_file="resources/genome/betula/genome.subset.20.fasta",
                    gff_file="resources/genome/betula/genome.gff.gz"
                ),
                MaximumParsimonyAnnotation()
            ],
            filtrations=[
                CodingSequenceFiltration(
                    gff_file="resources/genome/betula/genome.gff.gz"
                )
            ],
            max_sites=100000
        )

        sfs = p.parse()

        pass

    @pytest.mark.slow
    def test_parse_betula_complete_vcf_including_monomrphic(self):
        """
        Parse the VCF file of Betula spp.
        """
        p = Parser(
            vcf="resources/genome/betula/all.vcf.gz",
            n=20,
            annotations=[
                DegeneracyAnnotation(
                    fasta_file="resources/genome/betula/genome.fasta",
                    gff_file="resources/genome/betula/genome.gff.gz"
                ),
                MaximumParsimonyAnnotation()
            ],
            filtrations=[
                CodingSequenceFiltration(
                    gff_file="resources/genome/betula/genome.gff.gz"
                )
            ],
        )

        sfs = p.parse()

        sfs.plot()

    def test_parse_human_chr22_from_online_resources_and_perform_inference(self):
        """
        Parse the VCF file using remote files.
        """
        p = Parser(
            vcf="http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/1000_genomes_project/release/"
                "20181203_biallelic_SNV/ALL.chr21.shapeit2_integrated_v1a.GRCh38.20181129.phased.vcf.gz",
            n=10,
            annotations=[
                DegeneracyAnnotation(
                    fasta_file="http://ftp.ensembl.org/pub/release-109/fasta/homo_sapiens/"
                               "dna/Homo_sapiens.GRCh38.dna.chromosome.21.fa.gz",
                    gff_file="http://ftp.ensembl.org/pub/release-109/gff3/homo_sapiens/"
                             "Homo_sapiens.GRCh38.109.chromosome.21.gff3.gz",
                    aliases=dict(chr21=['21'])
                ),
                MaximumParsimonyAnnotation()
            ],
            filtrations=[
                CodingSequenceFiltration(
                    gff_file="http://ftp.ensembl.org/pub/release-109/gff3/homo_sapiens/"
                             "Homo_sapiens.GRCh38.109.chromosome.21.gff3.gz",
                    aliases=dict(chr21=['21'])
                )
            ],
            max_sites=100000
        )

        sfs = p.parse()

        sfs.plot()
