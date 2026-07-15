"""
Parser -> DFE inference integration tests.

These exercise the VCF-to-SFS parser (provided by sfsutils and re-exported from fastdfe)
feeding into DFE inference. They need the full genome fixtures and are part of the slow tier.
Relocated here from the parser tests when SFS parsing was factored out into sfsutils.
"""
import pytest

import fastdfe as fd
from testing import TestCase


class ParserDFETestCase(TestCase):
    """
    Parser + BaseInference integration tests.
    """

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
