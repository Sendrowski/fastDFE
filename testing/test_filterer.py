import logging

from fastdfe.vcf import count_sites
from testing import prioritize_installed_packages

prioritize_installed_packages()
from unittest.mock import Mock

from unittest import TestCase

from fastdfe import Filterer, PolyAllelicFiltration, SNPFiltration, SNVFiltration

logging.getLogger('fastdfe').setLevel(logging.INFO)


class FiltererTestCase(TestCase):
    """
    Test the filter.
    """
    def test_filter_snp_filtration(self):
        """
        Test the SNP filtration.
        """
        f = Filterer(
            vcf="resources/genome/betula/biallelic.subset.10000.vcf.gz",
            output='scratch/test_maximum_parsimony_annotator.vcf',
            filtrations=[SNPFiltration()],
        )

        f.filter()

        # assert no sites were filtered
        assert f.n_filtered == 2

        # assert number of sites is the same
        assert count_sites(f.vcf) == count_sites(f.output) + f.n_filtered

    def test_filter_no_poly_allelic_filtration(self):
        """
        Test the no poly-allelic filtration.
        """
        f = Filterer(
            vcf="resources/genome/betula/biallelic.subset.10000.vcf.gz",
            output='scratch/test_maximum_parsimony_annotator.vcf',
            filtrations=[PolyAllelicFiltration()],
        )

        f.filter()

        # assert no sites were filtered
        assert f.n_filtered == 0

        # assert number of sites is the same
        assert count_sites(f.vcf) == count_sites(f.output)

    def test_annotator_load_vcf_from_url(self):
        """
        Test the annotator loading a VCF from a URL.
        """
        f = Filterer(
            vcf="resources/genome/betula/biallelic.subset.10000.vcf.gz",
            output='scratch/test_degeneracy_annotation.vcf',
            filtrations=[PolyAllelicFiltration()]
        )

        f.filter()

        # assert number of sites is the same
        assert f.n_sites == 10000
        assert f.n_filtered == 0

    def test_snp_filtration(self):
        """
        Test the SNP filtration.
        """
        f = SNPFiltration()

        assert not f.filter_site(variant=Mock(is_snp=False))
        assert f.filter_site(variant=Mock(is_snp=True))

    def test_snv_filtration(self):
        """
        Test the SNV filtration.
        """
        f = SNVFiltration()

        assert not f.filter_site(variant=Mock(REF='AG'))
        assert f.filter_site(variant=Mock(REF='A'))

    def test_no_poly_allelic_filtration(self):
        """
        Test the no poly-allelic filtration.
        """
        f = PolyAllelicFiltration()

        assert not f.filter_site(variant=Mock(ALT=['T', 'G']))
        assert not f.filter_site(variant=Mock(ALT=['T', 'G', 'C']))
        assert f.filter_site(variant=Mock(ALT=['T']))
        assert f.filter_site(variant=Mock(ALT=[]))
