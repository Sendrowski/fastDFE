import numpy as np
from cyvcf2 import Variant

from fastdfe.vcf import count_sites
from testing import prioritize_installed_packages

prioritize_installed_packages()
from unittest.mock import Mock

from testing import TestCase

from fastdfe import Filterer, PolyAllelicFiltration, SNPFiltration, SNVFiltration, DeviantOutgroupFiltration


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
            output='scratch/test_filter_snp_filtration.vcf',
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
            output='scratch/test_filter_no_poly_allelic_filtration.vcf',
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
            output='scratch/test_annotator_load_vcf_from_url.vcf',
            filtrations=[PolyAllelicFiltration()]
        )

        f.filter()

        # assert number of sites is the same
        assert f.n_sites == 10000
        assert f.n_filtered == 0

    def test_deviant_outgroup_filtration(self):
        """
        Test the annotator loading a VCF from a URL.
        """
        f = Filterer(
            vcf="resources/genome/betula/all.with_outgroups.subset.10000.vcf.gz",
            output='scratch/test_deviant_outgroup_filtration.vcf',
            filtrations=[DeviantOutgroupFiltration(outgroups=["ERR2103730", "ERR2103731"])]
        )

        f.filter()

        # assert number of sites is the same
        assert f.n_sites == 10000
        assert f.n_filtered == 510

    def test_deviant_outgroup_filtration_single_site(self):
        mock_variant = Mock(spec=Variant)

        # test case 1: variant is not a SNP
        mock_variant.is_snp = False
        filter_obj = DeviantOutgroupFiltration(outgroups=['outgroup1'], ingroups=['ingroup1'])
        filter_obj.samples = np.array(['ingroup1', 'outgroup1'])
        filter_obj.create_masks()
        assert filter_obj.filter_site(mock_variant)  # expect True as the variant is not SNP

        # test case 2: variant is a SNP, strict mode is enabled and no outgroup sample is present
        mock_variant.is_snp = True
        filter_obj = DeviantOutgroupFiltration(outgroups=['outgroup1'], ingroups=['ingroup1'], strict_mode=True)
        filter_obj.samples = np.array(['ingroup1', 'outgroup1'])
        filter_obj.create_masks()
        mock_variant.gt_bases = np.array(['A/A', './.'])
        assert not filter_obj.filter_site(mock_variant)  # expect False as no outgroup sample is present

        # test case 3: variant is a SNP, strict mode is disabled and no outgroup sample is present
        filter_obj = DeviantOutgroupFiltration(outgroups=['outgroup1'], ingroups=['ingroup1'], strict_mode=False)
        filter_obj.samples = np.array(['ingroup1', 'outgroup1'])
        filter_obj.create_masks()
        mock_variant.gt_bases = np.array(['A/A', './.'])
        assert filter_obj.filter_site(mock_variant)  # # expect True as strict mode off and no outgroup sample present

        # test case 4: variant is an SNP and outgroup base is different from ingroup base
        filter_obj = DeviantOutgroupFiltration(outgroups=['outgroup1'], ingroups=['ingroup1'])
        filter_obj.samples = np.array(['ingroup1', 'outgroup1'])
        filter_obj.create_masks()
        mock_variant.gt_bases = np.array(['A/A', 'T/T'])
        assert not filter_obj.filter_site(mock_variant)  # expect False as outgroup base is different from ingroup base

        # test case 5: variant is a SNP and outgroup base is same as ingroup base
        filter_obj = DeviantOutgroupFiltration(outgroups=['outgroup1'], ingroups=['ingroup1'])
        filter_obj.samples = np.array(['ingroup1', 'outgroup1'])
        filter_obj.create_masks()
        mock_variant.gt_bases = np.array(['A/A', 'A/A'])
        assert filter_obj.filter_site(mock_variant)  # expect True as outgroup base is same as ingroup base

        # test case 6: multiple ingroups and outgroups with matching major bases
        mock_variant = Mock(spec=Variant)
        mock_variant.is_snp = True
        filter_obj = DeviantOutgroupFiltration(outgroups=['outgroup1', 'outgroup2'], ingroups=['ingroup1', 'ingroup2'])
        filter_obj.samples = np.array(['ingroup1', 'ingroup2', 'outgroup1', 'outgroup2'])
        filter_obj.create_masks()
        mock_variant.gt_bases = np.array(['A/A', 'A/T', 'A/A', 'A/G'])
        assert filter_obj.filter_site(mock_variant)  # expect True as major base 'A' is common in ingroup and outgroup

        # test case 7: multiple ingroups and outgroups with differing major bases
        mock_variant = Mock(spec=Variant)
        mock_variant.is_snp = True
        filter_obj = DeviantOutgroupFiltration(outgroups=['outgroup1', 'outgroup2'], ingroups=['ingroup1', 'ingroup2'])
        filter_obj.samples = np.array(['ingroup1', 'ingroup2', 'outgroup1', 'outgroup2'])
        filter_obj.create_masks()
        mock_variant.gt_bases = np.array(['A/A', 'A/T', 'T/T', 'T/G'])
        assert not filter_obj.filter_site(mock_variant)  # expect False as major base 'A' in ingroup and 'T' in outgroup

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

        assert not f.filter_site(variant=Mock(REF='AG', ALT=['A', 'G']))
        assert f.filter_site(variant=Mock(REF='A', ALT=['G']))
        assert f.filter_site(variant=Mock(REF='A', ALT=['G', 'C']))
        assert not f.filter_site(variant=Mock(REF='A', ALT=['GA']))
        assert f.filter_site(variant=Mock(REF='A', ALT=['G', 'C', 'T']))

    def test_no_poly_allelic_filtration(self):
        """
        Test the no poly-allelic filtration.
        """
        f = PolyAllelicFiltration()

        assert not f.filter_site(variant=Mock(ALT=['T', 'G']))
        assert not f.filter_site(variant=Mock(ALT=['T', 'G', 'C']))
        assert f.filter_site(variant=Mock(ALT=['T']))
        assert f.filter_site(variant=Mock(ALT=[]))
