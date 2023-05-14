import logging

from cyvcf2 import Variant

from fastdfe.vcf import count_sites
from testing import prioritize_installed_packages

prioritize_installed_packages()
from unittest.mock import Mock

from unittest import TestCase

from fastdfe import Filterer, NoPolyAllelicFiltration, SNPFiltration

logging.getLogger('fastdfe').setLevel(logging.DEBUG)


class FilterTestCase(TestCase):
    """
    Test the filter.
    """

    vcf_file = 'resources/genome/betula/biallelic.subset.10000.vcf.gz'

    def test_filter_snp_filtration(self):
        """
        Test the SNP filtration.
        """
        f = Filterer(
            vcf=self.vcf_file,
            output='scratch/test_maximum_parsimony_annotator.vcf',
            filtrations=[SNPFiltration()],
        )

        f.filter()

        # assert no sites were filtered
        assert f.n_filtered == 0

        # assert number of sites is the same
        assert count_sites(self.vcf_file) == count_sites(f.output)

    def test_filter_no_poly_allelic_filtration(self):
        """
        Test the no poly-allelic filtration.
        """
        f = Filterer(
            vcf=self.vcf_file,
            output='scratch/test_maximum_parsimony_annotator.vcf',
            filtrations=[NoPolyAllelicFiltration()],
        )

        f.filter()

        # assert no sites were filtered
        assert f.n_filtered == 0

        # assert number of sites is the same
        assert count_sites(self.vcf_file) == count_sites(f.output)

    def test_annotator_load_vcf_from_url(self):
        """
        Test the annotator loading a VCF from a URL.
        """
        f = Filterer(
            vcf="https://github.com/Sendrowski/fastDFE/blob/master/resources/genome/betula/biallelic.subset.10000.vcf.gz?raw=true",
            output='scratch/test_degeneracy_annotation.vcf',
            filtrations=[NoPolyAllelicFiltration()]
        )

        f.filter()

        # assert number of sites is the same
        assert f.n_sites == 4304
        assert f.n_filtered == 0

    def test_snp_filtration(self):
        """
        Test the SNP filtration.
        """
        f = SNPFiltration()

        assert not f.filter_site(variant=Mock(is_snp=False))
        assert f.filter_site(variant=Mock(is_snp=True))

    def test_no_poly_allelic_filtration(self):
        """
        Test the no poly-allelic filtration.
        """
        f = NoPolyAllelicFiltration()

        assert not f.filter_site(variant=Mock(ALT=['T', 'G']))
        assert not f.filter_site(variant=Mock(ALT=['T', 'G', 'C']))
        assert f.filter_site(variant=Mock(ALT=['T']))
        assert f.filter_site(variant=Mock(ALT=[]))
