import logging
from unittest.mock import Mock

import pytest

from testing import prioritize_installed_packages

prioritize_installed_packages()

from testing import TestCase

import dadi
import numpy as np

import fastdfe as fd


class ParserTestCase(TestCase):
    """
    Test parser.
    """

    @staticmethod
    def test_compare_sfs_with_dadi():
        """
        Compare the sfs from dadi with the one from the data.
        """
        p = fd.Parser(vcf='resources/genome/betula/biallelic.subset.10000.vcf.gz', n=20, stratifications=[], seed=2)

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

    @staticmethod
    def test_degeneracy_stratification():
        """
        Test the degeneracy stratification.
        """
        p = fd.Parser(
            vcf='resources/genome/betula/biallelic.subset.10000.vcf.gz',
            n=20,
            stratifications=[fd.DegeneracyStratification()]
        )

        sfs = p.parse()

        sfs.plot()

        # assert total number of sites
        assert sfs.all.data.sum() == 10000 - p.n_skipped

        # assert that all types are a subset of the stratification
        assert set(sfs.types).issubset(set(p.stratifications[0].get_types()))

    @staticmethod
    def test_contig_stratification():
        """
        Test the degeneracy stratification.
        """
        p = fd.Parser(
            vcf='resources/genome/betula/biallelic.subset.10000.vcf.gz',
            n=20,
            stratifications=[fd.ContigStratification()]
        )

        sfs = p.parse()

        sfs.plot()

        # assert total number of sites
        assert sfs.all.data.sum() == 10000 - p.n_skipped

        # assert that all types are a subset of the stratification
        assert set(sfs.types).issubset(set(p.stratifications[0].get_types()))

    @staticmethod
    def test_chunked_stratification():
        """
        Test the degeneracy stratification.
        """
        n_chunks = 7
        s = fd.ChunkedStratification(n_chunks=n_chunks)

        p = fd.Parser(
            vcf='resources/genome/betula/biallelic.subset.10000.vcf.gz',
            n=20,
            stratifications=[s]
        )

        sfs = p.parse()

        sfs.plot()

        # assert total number of sites
        assert sfs.all.data.sum() == 10000

        assert len(sfs.types) == n_chunks

        # assert that all types are a subset of the stratification
        assert set(sfs.types).issubset(set(s.get_types()))

        assert sum(s.chunk_sizes) == 10000

        assert (sfs.data.sum() == s.chunk_sizes).all()

    @pytest.mark.slow
    def test_vep_stratification(self):
        """
        Test the synonymy stratification.
        """
        p = fd.Parser(
            vcf='snakemake/results/vcf/sapiens/chr21.vep.vcf.gz',
            n=20,
            stratifications=[fd.VEPStratification()]
        )

        sfs = p.parse()

        sfs.plot()

        # assert that all types are a subset of the stratification
        assert set(sfs.types).issubset(set(p.stratifications[0].get_types()))

    @pytest.mark.slow
    def test_snpeff_stratification(self):
        """
        Test the synonymy stratification.
        """
        p = fd.Parser(
            vcf='snakemake/results/vcf/sapiens/chr21.snpeff.vcf.gz',
            n=20,
            stratifications=[fd.SnpEffStratification()]
        )

        sfs = p.parse()

        sfs.plot()

        # assert that all types are a subset of the stratification
        assert set(sfs.types).issubset(set(p.stratifications[0].get_types()))

    @staticmethod
    def test_base_transition_stratification():
        """
        Test the base transition stratification.
        """
        p = fd.Parser(
            vcf='resources/genome/betula/all.with_outgroups.subset.10000.vcf.gz',
            n=20,
            stratifications=[fd.BaseTransitionStratification()]
        )

        sfs = p.parse()

        sfs.plot()

        # assert total number of sites
        assert sfs.all.data.sum() == 10000 - p.n_skipped

        # assert that all types are a subset of the stratification
        assert set(sfs.types).issubset(set(p.stratifications[0].get_types()))

    @staticmethod
    def test_transition_transversion_stratification():
        """
        Test the transition transversion stratification.
        """
        p = fd.Parser(
            vcf='resources/genome/betula/all.with_outgroups.subset.10000.vcf.gz',
            n=20,
            stratifications=[fd.TransitionTransversionStratification()]
        )

        sfs = p.parse()

        sfs.plot()

        # assert total number of sites
        assert sfs.all.data.sum() == 10000 - p.n_skipped

        # assert that all types are a subset of the stratification
        assert set(sfs.types).issubset(set(p.stratifications[0].get_types()))

    @staticmethod
    def test_base_context_stratification():
        """
        Test the base context stratification.
        """
        p = fd.Parser(
            vcf='resources/genome/betula/biallelic.subset.10000.vcf.gz',
            n=20,
            stratifications=[fd.BaseContextStratification(fasta='resources/genome/betula/genome.subset.20.fasta')]
        )

        sfs = p.parse()

        sfs.plot()

        # assert total number of sites
        assert sfs.all.data.sum() == 10000 - p.n_skipped

        # assert that all types are a subset of the stratification
        assert set(sfs.types).issubset(set(p.stratifications[0].get_types()))

    @staticmethod
    def test_reference_base_stratification():
        """
        Test the reference base stratification.
        """
        p = fd.Parser(
            vcf='resources/genome/betula/biallelic.subset.10000.vcf.gz',
            n=20,
            stratifications=[fd.AncestralBaseStratification()]
        )

        sfs = p.parse()

        sfs.plot()

        # assert total number of sites
        assert sfs.all.data.sum() == 10000 - p.n_skipped

        # assert that all types are a subset of the stratification
        assert set(sfs.types).issubset(set(p.stratifications[0].get_types()))

    @staticmethod
    def test_parse_vcf_chr21_test():
        """
        Parse human chr21 test VCF file.
        """
        p = fd.Parser(
            vcf="resources/genome/sapiens/chr21_test.vcf.gz",
            gff="resources/genome/sapiens/hg38.sorted.gtf.gz",
            fasta="http://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/chr21.fa.gz",
            n=20,
            skip_non_polarized=True,
            annotations=[
                fd.DegeneracyAnnotation(),
                fd.MaximumParsimonyAncestralAnnotation()
            ],
            filtrations=[
                fd.CodingSequenceFiltration()
            ],
            stratifications=[fd.DegeneracyStratification()],
            max_sites=100000
        )

        sfs = p.parse()

        assert sfs.all.data.sum() == 6

    @staticmethod
    def test_parse_vcf_chr21():
        """
        Parse human chr21 VCF file.

        TODO Still looks like we mostly have slightly deleterious mutations.
        """
        p = fd.Parser(
            vcf="resources/genome/sapiens/chr21.vcf.gz",
            gff="resources/genome/sapiens/hg38.sorted.gtf.gz",
            fasta="http://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/chr21.fa.gz",
            n=10,
            target_site_counter=fd.TargetSiteCounter(
                n_samples=100000,
                n_target_sites=fd.Annotation.count_target_sites(
                    "resources/genome/sapiens/hg38.sorted.gtf.gz"
                )['chr21']
            ),
            skip_non_polarized=True,
            annotations=[
                fd.DegeneracyAnnotation(),
                fd.MaximumParsimonyAncestralAnnotation()
            ],
            filtrations=[
                fd.CodingSequenceFiltration()
            ],
            stratifications=[fd.DegeneracyStratification()]
        )

        sfs = p.parse()

        pass

    def test_parse_betula_vcf_biallelic_infer_monomorphic(self):
        """
        Parse the VCF file of Betula spp.
        """

        p = fd.Parser(
            vcf="resources/genome/betula/biallelic.vcf.gz",
            fasta="resources/genome/betula/genome.subset.20.fasta",
            gff="resources/genome/betula/genome.gff.gz",
            target_site_counter=fd.TargetSiteCounter(
                n_samples=1000000,
                n_target_sites=100000
            ),
            n=20,
            max_sites=10000,
            annotations=[
                fd.DegeneracyAnnotation( ),
                fd.MaximumParsimonyAncestralAnnotation()
            ],
            filtrations=[
                fd.CodingSequenceFiltration()
            ],
            stratifications=[fd.DegeneracyStratification()]
        )

        sfs = p.parse()

        self.assertEqual(sfs.n_sites.sum(), 100000)

        # assert fixed number of target sites
        self.assertAlmostEqual(sfs['neutral'].n_sites, 19369.006191, places=6)
        self.assertAlmostEqual(sfs['selected'].n_sites, 80630.993809, places=6)

    def test_filter_out_all_raises_warning(self):
        """
        Test that filtering out all sites logs a warning.
        """
        p = fd.Parser(
            vcf="resources/genome/betula/biallelic.subset.10000.vcf.gz",
            n=20,
            filtrations=[fd.AllFiltration()]
        )

        with self.assertLogs(level="WARNING", logger=logging.getLogger('fastdfe')):
            p.parse()

    @staticmethod
    def test_parser_no_stratifications():
        """
        Test that filtering out all sites logs a warning.
        """
        p = fd.Parser(
            vcf="resources/genome/betula/biallelic.subset.10000.vcf.gz",
            n=20,
            stratifications=[]
        )

        sfs = p.parse()

        assert 'all' in sfs.types

    @staticmethod
    def test_parse_betula_vcf():
        """
        Parse the VCF file of Betula spp.
        """
        p = fd.Parser(
            vcf="resources/genome/betula/all.subset.100000.vcf.gz",
            fasta="resources/genome/betula/genome.subset.20.fasta",
            gff="resources/genome/betula/genome.gff.gz",
            n=20,
            annotations=[
                fd.DegeneracyAnnotation(),
                fd.MaximumParsimonyAncestralAnnotation()
            ],
            filtrations=[
                fd.CodingSequenceFiltration()
            ],
            stratifications=[fd.DegeneracyStratification()]
        )

        sfs = p.parse()

        pass

    @pytest.mark.slow
    def test_parse_betula_complete_vcf_biallelic(self):
        """
        Parse the VCF file of Betula spp.
        """
        p = fd.Parser(
            vcf="resources/genome/betula/biallelic.vcf.gz",
            fasta="resources/genome/betula/genome.fasta",
            gff="resources/genome/betula/genome.gff.gz",
            n=10,
            annotations=[
                fd.SynonymyAnnotation()
            ],
            filtrations=[
                fd.CodingSequenceFiltration()
            ],
            stratifications=[fd.SynonymyStratification()]
        )

        sfs = p.parse()

        sfs.plot()

    @pytest.mark.slow
    def test_parse_betula_compare_monomorphic_vcf_with_inferred_monomorphic_betula_ingroups(self):
        """
        Parse the VCF file of Betula spp.

        TODO we get different ratios of neutral to selected sites here.
        The same happens with biallelic.with_outgroups.vcf.gz which was directly derived from
        all.with_outgroups.vcf.gz by filtering out all sites that are not biallelic.
        """
        p = fd.Parser(
            vcf="resources/genome/betula/all.vcf.gz",
            fasta="resources/genome/betula/genome.fasta",
            gff="resources/genome/betula/genome.gff.gz",
            target_site_counter=None,
            n=20,
            annotations=[
                fd.DegeneracyAnnotation()
            ],
            filtrations=[
                fd.CodingSequenceFiltration()
            ],
            stratifications=[fd.DegeneracyStratification()]
        )

        sfs = p.parse()

        p2 = fd.Parser(
            vcf="resources/genome/betula/biallelic.vcf.gz",
            fasta="resources/genome/betula/genome.fasta",
            gff="resources/genome/betula/genome.gff.gz",
            target_site_counter=fd.TargetSiteCounter(
                n_samples=1000000,
                n_target_sites=sfs.n_sites.sum()
            ),
            n=20,
            annotations=[
                fd.DegeneracyAnnotation()
            ],
            filtrations=[
                fd.CodingSequenceFiltration()
            ],
            stratifications=[fd.DegeneracyStratification()]
        )

        sfs2 = p2.parse()

        infs = []
        for spectra in [sfs, sfs2]:
            inf = fd.BaseInference(
                sfs_neut=spectra['neutral'],
                sfs_sel=spectra['selected'],
                do_bootstrap=True,
                model=fd.DiscreteFractionalParametrization(),
            )

            inf.run()

            infs.append(inf)

        fd.Inference.plot_discretized(infs, labels=['monomorphic', 'inferred'])

    @pytest.mark.slow
    def test_parse_betula_complete_vcf_including_monomorphic(self):
        """
        Parse the VCF file of Betula spp.
        """
        p = fd.Parser(
            vcf="resources/genome/betula/all.vcf.gz",
            fasta="resources/genome/betula/genome.fasta",
            gff="resources/genome/betula/genome.gff.gz",
            n=20,
            annotations=[
                fd.DegeneracyAnnotation()
            ],
            filtrations=[
                fd.CodingSequenceFiltration()
            ],
            stratifications=[fd.DegeneracyStratification()]
        )

        sfs = p.parse()

        sfs.plot()

        pass

    @staticmethod
    def test_parse_human_chr21_from_online_resources():
        """
        Parse the VCF file using remote files.
        """
        p = fd.Parser(
            vcf="http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/1000_genomes_project/release/"
                "20181203_biallelic_SNV/ALL.chr21.shapeit2_integrated_v1a.GRCh38.20181129.phased.vcf.gz",
            fasta="http://ftp.ensembl.org/pub/release-109/fasta/homo_sapiens/"
                       "dna/Homo_sapiens.GRCh38.dna.chromosome.21.fa.gz",
            gff="http://ftp.ensembl.org/pub/release-109/gff3/homo_sapiens/"
                     "Homo_sapiens.GRCh38.109.chromosome.21.gff3.gz",
            aliases=dict(chr21=['21']),
            n=10,
            annotations=[
                fd.DegeneracyAnnotation(),
                fd.MaximumParsimonyAncestralAnnotation()
            ],
            filtrations=[
                fd.CodingSequenceFiltration()
            ],
            stratifications=[fd.DegeneracyStratification()],
            max_sites=100000
        )

        sfs = p.parse()

        sfs.plot()

    def test_target_site_counter_betula(self):
        """
        Test whether the monomorphic site counter works on the Betula data.
        """
        p = fd.Parser(
            vcf="resources/genome/betula/biallelic.vcf.gz",
            fasta="resources/genome/betula/genome.fasta",
            gff="resources/genome/betula/genome.gff.gz",
            max_sites=10000,
            n=10,
            target_site_counter=fd.TargetSiteCounter(
                n_target_sites=100000,
                n_samples=10000
            ),
            annotations=[
                fd.DegeneracyAnnotation()
            ],
            stratifications=[fd.DegeneracyStratification()]
        )

        # set log level to DEBUG
        p.target_site_counter.logger.setLevel(logging.DEBUG)

        sfs = p.parse()

        # make sure that the sum of the target sites is correct
        self.assertEqual(sfs.n_sites.sum(), p.target_site_counter.n_target_sites)

        # assert that 3 contigs were parsed
        self.assertEqual(3, len(p._positions))

        # make sure we also consider 3 contigs for the target site counter
        self.assertEqual(3, len(p.target_site_counter.count_contig_sizes()))

    def test_target_site_counter_update_target_sites_target_sites_lower_than_polymorphic_raises_warning(self):
        """
        Test updating the target sites for different spectra.
        """
        c = fd.TargetSiteCounter(
            n_target_sites=1000,
            n_samples=10000
        )

        with self.assertLogs(level="WARNING", logger=logging.getLogger('fastdfe.TargetSiteCounter')) as warning:
            c._update_target_sites(fd.Spectra(dict(
                # an SFS, decreasing sequence
                neutral=[177130, 997, 441, 228, 156, 117, 114, 83, 105, 109, 652],
                selected=[797939, 1329, 499, 265, 162, 104, 117, 90, 94, 119, 794]
            )))

            print(warning[1][0])

    def test_target_site_counter_update_target_sites_target_sites_no_monomorphic_raises_warning(self):
        """
        Test updating the target sites for different spectra.
        """
        c = fd.TargetSiteCounter(
            n_target_sites=100000,
            n_samples=10000
        )

        with self.assertLogs(level="WARNING", logger=logging.getLogger('fastdfe.TargetSiteCounter')) as warning:
            c._update_target_sites(fd.Spectra(dict(
                # an SFS, decreasing sequence
                neutral=[0, 997, 441, 228, 156, 117, 114, 83, 105, 109, 652],
                selected=[0, 1329, 499, 265, 162, 104, 117, 90, 94, 119, 794]
            )))

            print(warning[1][0])

    def test_target_site_counter_update_target_sites_sum_coincides_with_given_target_sites(self):
        """
        Test updating the target sites for different spectra.
        """
        c = fd.TargetSiteCounter(
            n_target_sites=100000,
            n_samples=10000
        )

        sfs1 = fd.Spectra(dict(
            # an SFS, decreasing sequence
            neutral=[177130, 997, 441, 228, 156, 117, 114, 83, 105, 109, 652],
            selected=[797939, 1329, 499, 265, 162, 104, 117, 90, 94, 119, 794]
        ))

        sfs2 = c._update_target_sites(sfs1)

        # make sure that the sum of the target sites is the same
        self.assertEqual(sfs2.n_sites.sum(), 100000)

        # make sure ratio of neutral to selected is the same
        self.assertEqual(
            sfs1.data.loc[0, 'neutral'] / sfs1.data.loc[0, 'selected'],
            sfs2.data.loc[0, 'neutral'] / sfs2.data.loc[0, 'selected']
        )

    def test_target_site_counter_update_target_sites_more_entries_sum_coincides_with_given_target_sites(self):
        """
        Test updating the target sites for different spectra.
        """
        c = fd.TargetSiteCounter(
            n_target_sites=100000,
            n_samples=10000
        )

        sfs1 = fd.Spectra({
            'type1.neutral': [177130, 997, 441, 228, 156, 117],
            'type1.selected': [797939, 1329, 499, 265, 162, 104],
            'type2.neutral': [144430, 114, 83, 105, 109, 652],
            'type2.selected': [797939, 117, 90, 94, 119, 794]
        })

        sfs2 = c._update_target_sites(sfs1)

        # make sure that the sum of the target sites is the same
        self.assertEqual(sfs2.n_sites.sum(), 100000)

    @pytest.mark.slow
    def test_betula_biallelic_dfe_for_different_n_target_sites(self):
        """
        Test the DFE estimation for different numbers of target sites.
        """
        n_target_sites = [1000, 10000, 100000, 1000000, 10000000]

        parsers = []  # parsers
        spectra = fd.Spectra({})  # spectra
        inferences = []  # inferences

        for i, n in enumerate(n_target_sites):
            p = fd.Parser(
                vcf="resources/genome/betula/biallelic.vcf.gz",
                fasta="resources/genome/betula/genome.fasta",
                gff="resources/genome/betula/genome.gff.gz",
                max_sites=10000,
                n=10,
                target_site_counter=fd.TargetSiteCounter(
                    n_target_sites=n,
                    n_samples=100000
                ),
                annotations=[
                    fd.DegeneracyAnnotation()
                ],
                stratifications=[fd.DegeneracyStratification()]
            )

            sfs = p.parse()

            inf = fd.BaseInference(
                sfs_neut=sfs['neutral'],
                sfs_sel=sfs['selected'],
                do_bootstrap=True
            )

            inf.run()

            parsers.append(p)
            inferences.append(inf)

            spectra += sfs.prefix(str(n))

        spectra.plot()

        fd.Inference.plot_discretized(inferences, labels=list(map(str, n_target_sites)))

    @pytest.mark.slow
    def test_betula_compare_dfe_across_different_samples_sizes_n(self):
        """
        Test the DFE estimation for different sample sizes.
        """
        sample_sizes = [5, 10, 15, 20, 25, 30]

        parsers = []  # parsers
        spectra = fd.Spectra({})  # spectra
        inferences = []  # inferences

        for i, n in enumerate(sample_sizes):
            p = fd.Parser(
                vcf="resources/genome/betula/all.vcf.gz",
                fasta="resources/genome/betula/genome.fasta",
                gff="resources/genome/betula/genome.gff.gz",
                # max_sites=1000000,
                n=n,
                annotations=[
                    fd.DegeneracyAnnotation()
                ],
                stratifications=[fd.DegeneracyStratification()]
            )

            sfs = p.parse()

            inf = fd.BaseInference(
                sfs_neut=sfs['neutral'],
                sfs_sel=sfs['selected'],
                do_bootstrap=True,
                model=fd.DiscreteFractionalParametrization()
            )

            inf.run()

            parsers.append(p)
            inferences.append(inf)

            spectra += sfs.prefix(str(n))

        spectra.plot(use_subplots=True)

        fd.Inference.plot_discretized(inferences, labels=[f"n={n}" for n in sample_sizes])
        fd.Inference.plot_inferred_parameters(inferences, labels=[f"n={n}" for n in sample_sizes], scale='lin')

    @pytest.mark.slow
    def test_human_chr1_compare_dfe_across_different_samples_sizes_n(self):
        """
        Test the DFE estimation for different sample sizes.
        """
        sample_sizes = [5, 10, 15, 20, 25, 30]

        parsers = []  # parsers
        spectra = fd.Spectra({})  # spectra
        inferences = []  # inferences

        for i, n in enumerate(sample_sizes):
            p = fd.Parser(
                vcf="http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/1000_genomes_project/release/"
                    "20181203_biallelic_SNV/ALL.chr1.shapeit2_integrated_v1a.GRCh38.20181129.phased.vcf.gz",
                fasta="http://ftp.ensembl.org/pub/release-109/fasta/homo_sapiens/"
                           "dna/Homo_sapiens.GRCh38.dna.chromosome.1.fa.gz",
                gff="http://ftp.ensembl.org/pub/release-109/gff3/homo_sapiens/"
                         "Homo_sapiens.GRCh38.109.chromosome.1.gff3.gz",
                aliases=dict(chr1=['1']),
                n=n,
                target_site_counter=fd.TargetSiteCounter(
                    n_samples=1000000,
                    n_target_sites=fd.Annotation.count_target_sites(
                        file="http://ftp.ensembl.org/pub/release-109/gff3/homo_sapiens/"
                             "Homo_sapiens.GRCh38.109.chromosome.1.gff3.gz"
                    )['1']
                ),
                annotations=[
                    fd.DegeneracyAnnotation()
                ],
                filtrations=[
                    fd.CodingSequenceFiltration()
                ],
                stratifications=[fd.DegeneracyStratification()],
                info_ancestral='AA_ensembl'
            )

            sfs = p.parse()

            inf = fd.BaseInference(
                sfs_neut=sfs['neutral'],
                sfs_sel=sfs['selected'],
                do_bootstrap=True,
                model=fd.DiscreteFractionalParametrization()
            )

            inf.run()

            parsers.append(p)
            inferences.append(inf)

            spectra += sfs.prefix(str(n))

        spectra.plot(use_subplots=True)

        fd.Inference.plot_discretized(inferences, labels=[f"n={n}" for n in sample_sizes])
        fd.Inference.plot_inferred_parameters(inferences, labels=[f"n={n}" for n in sample_sizes], scale='lin')

        pass

    def test_parser_betula_include_samples(self):
        """
        Test that the parser includes only the samples that are given in the include_samples parameter.
        """
        p = fd.Parser(
            vcf="resources/genome/betula/biallelic.subset.10000.vcf.gz",
            n=20,
            include_samples=['ASP01', 'ASP02', 'ASP03']
        )

        p._setup()

        self.assertEqual(np.sum(p._samples_mask), 3)

    def test_parser_betula_include_all_samples(self):
        """
        Test that the parser includes all samples if the include_samples parameter is not given.
        """
        p = fd.Parser(
            vcf="resources/genome/betula/biallelic.subset.10000.vcf.gz",
            n=20
        )

        p._setup()

        self.assertEqual(np.sum(p._samples_mask), 377)

    def test_parser_betula_exclude_two_samples(self):
        """
        Test that the parser excludes the samples that are given in the exclude_samples parameter.
        """
        p = fd.Parser(
            vcf="resources/genome/betula/biallelic.subset.10000.vcf.gz",
            n=20,
            exclude_samples=['ASP01', 'ASP02']
        )

        p._setup()

        self.assertEqual(np.sum(p._samples_mask), 375)

    def test_parser_betula_include_exclude(self):
        """
        Test that both include and exclude samples work together.
        """
        p = fd.Parser(
            vcf="resources/genome/betula/biallelic.subset.10000.vcf.gz",
            n=20,
            include_samples=['ASP01', 'ASP02', 'ASP03'],
            exclude_samples=['ASP02']
        )

        p._setup()

        self.assertEqual(np.sum(p._samples_mask), 2)
