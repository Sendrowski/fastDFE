import logging

import dadi
import numpy as np
import pandas as pd
import pytest

import fastdfe as fd
from fastdfe.io_handlers import get_called_bases
from testing import TestCase


class ParserTestCase(TestCase):
    """
    Test parser.
    """

    @staticmethod
    def test_parse_sfs_compare_subsample_modes():
        """
        Compare the sfs from dadi with the one from the data.
        """
        p1 = fd.Parser(
            vcf='resources/genome/betula/biallelic.polarized.subset.10000.vcf.gz',
            n=20,
            stratifications=[],
            subsample_mode='probabilistic',
            max_sites=10000
        )

        sfs = p1.parse().all

        p2 = fd.Parser(
            vcf='resources/genome/betula/biallelic.polarized.subset.10000.vcf.gz',
            n=20,
            stratifications=[],
            subsample_mode='random',
            max_sites=10000,
            seed=2
        )

        sfs2 = p2.parse().all

        fd.Spectra.from_spectra(dict(
            probabilistic=sfs,
            random=sfs2
        )).plot()

        diff_rel = np.abs(sfs.data - sfs2.data) / sfs2.data

        assert diff_rel[sfs2.data != 0].max() < 0.5

    @staticmethod
    def test_parse_sfs_compare_probabilistic_with_dadi():
        """
        Compare the sfs from dadi with the one from the data.
        """
        p1 = fd.Parser(
            vcf='resources/genome/betula/biallelic.polarized.subset.10000.vcf.gz',
            n=20,
            stratifications=[],
            subsample_mode='probabilistic',
            max_sites=10000
        )

        sfs = p1.parse().all

        data_dict = dadi.Misc.make_data_dict_vcf(
            'resources/genome/betula/biallelic.polarized.subset.10000.vcf.gz',
            'resources/genome/betula/pops_dadi.txt'
        )

        sfs2 = dadi.Spectrum.from_data_dict(data_dict, ['pop0'], [20], polarized=True)

        diff_rel = np.abs(sfs.data - sfs2.data) / sfs2.data

        assert diff_rel.max() < 1e-12

    @staticmethod
    @pytest.mark.slow
    def test_compare_sfs_with_dadi_full_set():
        """
        Compare the sfs from dadi with the one from the data.
        """
        p = fd.Parser(
            vcf='resources/genome/betula/biallelic.polarized.vcf.gz',
            n=20,
            stratifications=[],
            seed=2
        )

        sfs = p.parse().all

        data_dict = dadi.Misc.make_data_dict_vcf(
            'resources/genome/betula/biallelic.polarized.vcf.gz',
            'resources/genome/betula/pops_dadi.txt'
        )

        sfs_dadi = dadi.Spectrum.from_data_dict(data_dict, ['pop0'], [20], polarized=True)

        diff_rel = np.abs(sfs.to_numpy() - sfs_dadi.data) / sfs_dadi.data

        s = fd.Spectra(dict(
            native=sfs.to_numpy(),
            dadi=sfs_dadi.data
        ))

        s.plot()

        # rather similar given number of sites and subsamples
        assert diff_rel[sfs_dadi.data != 0].max() < 0.1

    @staticmethod
    def test_degeneracy_stratification():
        """
        Test the degeneracy stratification.
        """
        p = fd.Parser(
            vcf='resources/genome/betula/biallelic.polarized.subset.10000.vcf.gz',
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
            vcf='resources/genome/betula/biallelic.polarized.subset.10000.vcf.gz',
            n=20,
            stratifications=[fd.ContigStratification()]
        )

        sfs = p.parse()

        sfs.plot()

        # assert total number of sites
        assert np.round(sfs.all.data.sum()) == 10000 - p.n_skipped

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
            vcf='resources/genome/betula/biallelic.polarized.subset.10000.vcf.gz',
            n=20,
            stratifications=[s]
        )

        sfs = p.parse()

        sfs.plot()

        # assert total number of sites
        assert np.round(sfs.all.data.sum()) == 10000 - p.n_skipped

        assert s.n_valid == 10000 - p.n_skipped

        assert len(sfs.types) == n_chunks

        # assert that all types are a subset of the stratification
        assert set(sfs.types).issubset(set(s.get_types()))

    @pytest.mark.slow
    def test_vep_stratification(self):
        """
        Test the synonymy stratification against VEP for human chr21.
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
        Test the synonymy stratification against SNPEFF for human chr21.
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
            vcf='resources/genome/betula/all.polarized.subset.10000.vcf.gz',
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
            vcf='resources/genome/betula/all.polarized.subset.10000.vcf.gz',
            n=20,
            stratifications=[fd.TransitionTransversionStratification()]
        )

        sfs = p.parse()

        sfs.plot()

        # assert total number of sites
        assert np.round(sfs.all.data.sum()) == 10000 - p.n_skipped

        # assert that all types are a subset of the stratification
        assert set(sfs.types).issubset(set(p.stratifications[0].get_types()))

    @staticmethod
    def test_base_context_stratification():
        """
        Test the base context stratification.
        """
        p = fd.Parser(
            vcf='resources/genome/betula/biallelic.polarized.subset.10000.vcf.gz',
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
            vcf='resources/genome/betula/biallelic.polarized.subset.10000.vcf.gz',
            n=20,
            stratifications=[fd.AncestralBaseStratification()]
        )

        sfs = p.parse()

        sfs.plot()

        # assert total number of sites
        assert np.round(sfs.all.data.sum()) == 10000 - p.n_skipped

        # assert that all types are a subset of the stratification
        assert set(sfs.types).issubset(set(p.stratifications[0].get_types()))

    def test_parse_vcf_chr21_test(self):
        """
        Parse human chr21 test VCF file.
        """
        p = fd.Parser(
            vcf="resources/genome/sapiens/chr21_test.vcf.gz",
            gff="resources/genome/sapiens/hg38.sorted.gtf.gz",
            fasta="http://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/chr21.fa.gz",
            n=20,
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

        self.assertEqual(np.round(sfs.all.data.sum()), 6)

    def test_parse_betula_vcf_biallelic_infer_monomorphic_subset(self):
        """
        Parse a subset of the VCF file of Betula spp.
        """

        p = fd.Parser(
            vcf="resources/genome/betula/biallelic.polarized.subset.10000.vcf.gz",
            fasta="resources/genome/betula/genome.subset.20.fasta",
            gff="resources/genome/betula/genome.gff.gz",
            target_site_counter=fd.TargetSiteCounter(
                n_samples=1000000,
                n_target_sites=100000
            ),
            n=20,
            max_sites=10000,
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

        self.assertEqual(sfs.n_sites.sum(), 100000)

        # assert fixed number of target sites
        # self.assertAlmostEqual(sfs['neutral'].n_sites, 18897.233850, places=5)
        # self.assertAlmostEqual(sfs['selected'].n_sites, 81102.766149, places=5)

    def test_filter_out_all_raises_warning(self):
        """
        Test that filtering out all sites logs a warning.
        """
        p = fd.Parser(
            vcf="resources/genome/betula/biallelic.polarized.subset.10000.vcf.gz",
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
            vcf="resources/genome/betula/biallelic.polarized.subset.10000.vcf.gz",
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
            vcf="resources/genome/betula/all.polarized.subset.10000.vcf.gz",
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
    def test_parse_betula_complete_vcf_biallelic_synonymy(self):
        """
        Parse the VCF file of Betula spp.
        """
        p = fd.Parser(
            vcf="resources/genome/betula/biallelic.polarized.vcf.gz",
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
    def test_parse_betula_complete_from_remote(self):
        """
        Parse the VCF file of Betula spp. using remote files.
        """
        parsers = []
        spectra = []

        for outgroups in [["ERR2103730"], ["ERR2103730", "ERR2103731"]]:
            p = fd.Parser(
                vcf="https://github.com/Sendrowski/fastDFE/blob/dev/resources/"
                    "genome/betula/biallelic.polarized.subset.50000.vcf.gz?raw=true",
                fasta="https://github.com/Sendrowski/fastDFE/blob/dev/resources/"
                      "genome/betula/genome.subset.1000.fasta.gz?raw=true",
                gff="https://github.com/Sendrowski/fastDFE/blob/dev/resources/"
                    "genome/betula/genome.gff.gz?raw=true",
                n=10,
                subsample_mode='random',
                annotations=[
                    fd.DegeneracyAnnotation(),
                    fd.MaximumLikelihoodAncestralAnnotation(
                        n_ingroups=10,
                        outgroups=outgroups,
                        subsample_mode='random'
                    )
                ],
                filtrations=[
                    fd.CodingSequenceFiltration(),
                    fd.ExistingOutgroupFiltration(outgroups=outgroups)
                ],
                stratifications=[fd.DegeneracyStratification()],
                max_sites=10000
            )

            sfs = p.parse()

            sfs.plot()

            parsers.append(p)
            spectra.append(sfs)

        # noinspection all
        # using two outgroups produces bad results as the two outgroups are too close to each other
        site_info = dict(
            one_outgroup=pd.DataFrame(parsers[0].annotations[1].get_inferred_site_info()),
            two_outgroups=pd.DataFrame(parsers[1].annotations[1].get_inferred_site_info())
        )

        pass

    @pytest.mark.slow
    def test_parse_betula_compare_monomorphic_vcf_with_inferred_monomorphic_betula(self):
        """
        Parse the VCF file of Betula spp.
        """
        p = fd.Parser(
            vcf="resources/genome/betula/all.polarized.vcf.gz",
            fasta="resources/genome/betula/genome.fasta",
            gff="resources/genome/betula/genome.gff.gz",
            target_site_counter=None,
            # max_sites=100000,
            n=20,
            annotations=[fd.DegeneracyAnnotation()],
            filtrations=[fd.CodingSequenceFiltration()],
            stratifications=[fd.DegeneracyStratification()]
        )

        sfs = p.parse()

        p2 = fd.Parser(
            vcf="resources/genome/betula/biallelic.polarized.vcf.gz",
            fasta="resources/genome/betula/genome.fasta",
            gff="resources/genome/betula/genome.gff.gz",
            target_site_counter=fd.TargetSiteCounter(
                n_samples=100000,
                n_target_sites=sfs.n_sites.sum()
            ),
            # max_sites=100000,
            n=20,
            annotations=[fd.DegeneracyAnnotation()],
            filtrations=[fd.CodingSequenceFiltration()],
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

        # the ratio of neutral to selected sites should be the same
        # but is about 0.225 for monomorphic VCF and 0.29 for inferred monomorphic sites
        fd.Inference.plot_discretized(infs, labels=['monomorphic', 'inferred'])

        pass

    @pytest.mark.slow
    def test_parse_betula_compare_monomorphic_vcf_with_inferred_monomorphic_betula_same_vcf(self):
        """
        Parse the VCF file of Betula spp.
        """
        p = fd.Parser(
            vcf="resources/genome/betula/all.polarized.vcf.gz",
            fasta="resources/genome/betula/genome.fasta",
            gff="resources/genome/betula/genome.gff.gz",
            target_site_counter=None,
            max_sites=100000,
            n=20,
            annotations=[fd.DegeneracyAnnotation()],
            filtrations=[fd.CodingSequenceFiltration()],
            stratifications=[fd.DegeneracyStratification()]
        )

        sfs = p.parse()

        p2 = fd.Parser(
            vcf="resources/genome/betula/all.polarized.vcf.gz",
            fasta="resources/genome/betula/genome.fasta",
            gff="resources/genome/betula/genome.gff.gz",
            target_site_counter=fd.TargetSiteCounter(
                n_samples=1000000,
                n_target_sites=sfs.n_sites.sum()
            ),
            max_sites=100000,
            n=20,
            annotations=[fd.DegeneracyAnnotation()],
            filtrations=[fd.SNPFiltration(), fd.CodingSequenceFiltration()],
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
                parallelize=True
            )

            inf.run()

            infs.append(inf)

        fd.Inference.plot_discretized(infs, labels=['monomorphic', 'inferred'])

        # calculate ratio of neutral to selected sites
        r1 = sfs['neutral'].data[0] / sfs['selected'].data[0]
        r2 = sfs2['neutral'].data[0] / sfs2['selected'].data[0]

        # make sure that the ratio is similar
        self.assertTrue(abs(r1 - r2) < 0.01)

        pass

    @pytest.mark.slow
    def test_parse_betula_complete_vcf_including_monomorphic(self):
        """
        Parse the VCF file of Betula spp.
        """
        p = fd.Parser(
            vcf="resources/genome/betula/all.polarized.vcf.gz",
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
    @pytest.mark.slow
    def test_parse_human_chr21_from_online_resources():
        """
        Parse the VCF for human chr21 file using remote files.
        """
        # parse selected and neutral SFS from human chromosome 1
        p = fd.Parser(
            vcf="https://ngs.sanger.ac.uk/production/hgdp/hgdp_wgs.20190516/"
                "hgdp_wgs.20190516.full.chr21.vcf.gz",
            fasta="http://ftp.ensembl.org/pub/release-109/fasta/homo_sapiens/"
                  "dna/Homo_sapiens.GRCh38.dna.chromosome.21.fa.gz",
            gff="http://ftp.ensembl.org/pub/release-109/gff3/homo_sapiens/"
                "Homo_sapiens.GRCh38.109.chromosome.21.gff3.gz",
            aliases=dict(chr21=['21']),
            n=8,
            target_site_counter=fd.TargetSiteCounter(
                n_samples=1000000,
                n_target_sites=fd.Annotation.count_target_sites(
                    "http://ftp.ensembl.org/pub/release-109/gff3/homo_sapiens/"
                    "Homo_sapiens.GRCh38.109.chromosome.21.gff3.gz"
                )['21']
            ),
            annotations=[
                fd.DegeneracyAnnotation()
            ],
            filtrations=[
                fd.SNPFiltration(),
                fd.CodingSequenceFiltration()
            ],
            stratifications=[fd.DegeneracyStratification()],
            info_ancestral='AA_ensembl',
            skip_non_polarized=True
        )

        sfs = p.parse()

        sfs.plot()

        pass

    def test_target_site_counter_betula(self):
        """
        Test whether the monomorphic site counter works on the Betula data.
        """
        p = fd.Parser(
            vcf="resources/genome/betula/biallelic.polarized.subset.10000.vcf.gz",
            fasta="resources/genome/betula/genome.subset.20.fasta",
            gff="resources/genome/betula/genome.gff.gz",
            max_sites=10000,
            n=10,
            target_site_counter=fd.TargetSiteCounter(
                n_target_sites=40000,
                n_samples=10000
            ),
            annotations=[
                fd.DegeneracyAnnotation()
            ],
            stratifications=[fd.DegeneracyStratification()]
        )

        # set log level to DEBUG
        p.target_site_counter._logger.setLevel(logging.DEBUG)

        sfs = p.parse()

        # make sure that the sum of the target sites is correct
        self.assertEqual(sfs.n_sites.sum(), p.target_site_counter.n_target_sites)

        # assert that 3 contigs were parsed
        self.assertEqual(3, len(p._contig_bounds))

    def test_target_site_counter_update_target_sites_target_sites_lower_than_polymorphic_raises_warning(self):
        """
        Test updating the target sites for different spectra.
        """
        c = fd.TargetSiteCounter(
            n_target_sites=1000,
            n_samples=10000
        )

        # assign a polymorphic SFS to the target site counter
        c._sfs_polymorphic = fd.Spectra(dict(
            neutral=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            selected=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ))

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

        # assign a polymorphic SFS to the target site counter
        c._sfs_polymorphic = fd.Spectra(dict(
            neutral=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            selected=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ))

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

        # assign a polymorphic SFS to the target site counter
        c._sfs_polymorphic = fd.Spectra(dict(
            neutral=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            selected=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ))

        sfs1 = fd.Spectra(dict(
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

        # assign a polymorphic SFS to the target site counter
        c._sfs_polymorphic = fd.Spectra({
            'type1.neutral': [0, 0, 0, 0, 0, 0],
            'type1.selected': [0, 0, 0, 0, 0, 0],
            'type2.neutral': [0, 0, 0, 0, 0, 0],
            'type2.selected': [0, 0, 0, 0, 0, 0]
        })

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
                vcf="resources/genome/betula/biallelic.polarized.subset.10000.vcf.gz",
                fasta="resources/genome/betula/genome.subset.20.fasta",
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
                stratifications=[fd.DegeneracyStratification()],
                filtrations=[fd.SNPFiltration()]
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

        spectra.plot()

        # very similar results for all n_target_sites
        fd.Inference.plot_discretized(inferences, labels=list(map(str, n_target_sites)))

        self.assertTrue((np.array([inf.bootstraps.mean() for inf in inferences]).var(axis=0) < 1e-1).all())

        pass

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
                vcf="resources/genome/betula/all.polarized.vcf.gz",
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

    @pytest.mark.skip(reason="takes too long")
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
                vcf="https://ngs.sanger.ac.uk/production/hgdp/hgdp_wgs.20190516/"
                    "hgdp_wgs.20190516.full.chr1.vcf.gz",
                fasta="http://ftp.ensembl.org/pub/release-109/fasta/homo_sapiens/"
                      "dna/Homo_sapiens.GRCh38.dna.chromosome.1.fa.gz",
                gff="http://ftp.ensembl.org/pub/release-109/gff3/homo_sapiens/"
                    "Homo_sapiens.GRCh38.109.chromosome.1.gff3.gz",
                aliases=dict(chr1=['1']),
                n=n,
                target_site_counter=fd.TargetSiteCounter(
                    n_samples=100000,
                    n_target_sites=fd.Annotation.count_target_sites(
                        file="http://ftp.ensembl.org/pub/release-109/gff3/homo_sapiens/"
                             "Homo_sapiens.GRCh38.109.chromosome.1.gff3.gz"
                    )['1']
                ),
                annotations=[
                    fd.DegeneracyAnnotation()
                ],
                filtrations=[
                    fd.CodingSequenceFiltration(),
                    fd.SNPFiltration()
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

    @staticmethod
    def test_get_called_genotypes():
        """
        Test the get_called_genotypes function.
        """
        result = get_called_bases(["A|T", "C/T", ".|G"])

        expected = np.array(['A', 'T', 'C', 'T', 'G'])

        np.testing.assert_array_equal(result, expected)

    @staticmethod
    def test_manuscript_example():
        """
        Test the example from the manuscript.
        """
        basepath = ("https://github.com/Sendrowski/fastDFE/"
                    "blob/master/resources/genome/betula/")

        # instantiate parser
        p = fd.Parser(
            n=8,  # SFS sample size
            vcf=(basepath + "biallelic.with_outgroups."
                            "subset.50000.vcf.gz?raw=true"),
            fasta=basepath + "genome.subset.1000.fasta.gz?raw=true",
            gff=basepath + "genome.gff.gz?raw=true",
            target_site_counter=fd.TargetSiteCounter(
                n_target_sites=350000  # total number of target sites
            ),
            annotations=[
                fd.DegeneracyAnnotation(),  # determine degeneracy
                fd.MaximumLikelihoodAncestralAnnotation(
                    outgroups=["ERR2103730"]  # use one outgroup
                )
            ],
            stratifications=[fd.DegeneracyStratification()]
        )

        # obtain SFS
        spectra: fd.Spectra = p.parse()

        spectra.plot()
