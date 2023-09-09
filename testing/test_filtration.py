import numpy as np
from cyvcf2 import Variant

from fastdfe.io_handlers import count_sites
from testing import prioritize_installed_packages

prioritize_installed_packages()
from unittest.mock import Mock

from testing import TestCase

import fastdfe as fd


class FiltrationTestCase(TestCase):
    """
    Test the filter.
    """

    @staticmethod
    def test_filter_snp_filtration():
        """
        Test the SNP filtration.
        """
        f = fd.Filterer(
            vcf="resources/genome/betula/biallelic.subset.10000.vcf.gz",
            output='scratch/test_filter_snp_filtration.vcf',
            filtrations=[fd.SNPFiltration()],
        )

        f.filter()

        # assert no sites were filtered
        assert f.n_filtered == 2

        # assert number of sites is the same
        assert count_sites(f.vcf) == count_sites(f.output) + f.n_filtered

    @staticmethod
    def test_filter_no_poly_allelic_filtration():
        """
        Test the no poly-allelic filtration.
        """
        f = fd.Filterer(
            vcf="resources/genome/betula/biallelic.subset.10000.vcf.gz",
            output='scratch/test_filter_no_poly_allelic_filtration.vcf',
            filtrations=[fd.PolyAllelicFiltration()],
        )

        f.filter()

        # assert no sites were filtered
        assert f.n_filtered == 0

        # assert number of sites is the same
        assert count_sites(f.vcf) == count_sites(f.output)

    @staticmethod
    def test_annotator_load_vcf_from_url():
        """
        Test the annotator loading a VCF from a URL.
        """
        f = fd.Filterer(
            vcf="resources/genome/betula/biallelic.subset.10000.vcf.gz",
            output='scratch/test_annotator_load_vcf_from_url.vcf',
            filtrations=[fd.PolyAllelicFiltration()]
        )

        f.filter()

        # assert number of sites is the same
        assert f.n_sites == 10000
        assert f.n_filtered == 0

    @staticmethod
    def test_deviant_outgroup_filtration():
        """
        Test the annotator loading a VCF from a URL.
        """
        f = fd.Filterer(
            vcf="resources/genome/betula/all.with_outgroups.subset.10000.vcf.gz",
            output='scratch/test_deviant_outgroup_filtration.vcf',
            filtrations=[fd.DeviantOutgroupFiltration(outgroups=["ERR2103730", "ERR2103731"])]
        )

        f.filter()

        # assert number of sites is the same
        assert f.n_sites == 10000
        assert f.n_filtered == 510

    @staticmethod
    def test_deviant_outgroup_filtration_single_site():
        """
        Test the annotator loading a VCF from a URL.
        """
        mock_variant = Mock(spec=Variant)

        # test case 1: variant is not a SNP
        mock_variant.is_snp = False
        filter_obj = fd.DeviantOutgroupFiltration(outgroups=['outgroup1'], ingroups=['ingroup1'])
        filter_obj.samples = np.array(['ingroup1', 'outgroup1'])
        filter_obj._create_masks()
        assert filter_obj.filter_site(mock_variant)  # expect True as the variant is not SNP

        # test case 2: variant is an SNP, strict mode is enabled and no outgroup sample is present
        mock_variant.is_snp = True
        filter_obj = fd.DeviantOutgroupFiltration(outgroups=['outgroup1'], ingroups=['ingroup1'], strict_mode=True)
        filter_obj.samples = np.array(['ingroup1', 'outgroup1'])
        filter_obj._create_masks()
        mock_variant.gt_bases = np.array(['A/A', './.'])
        assert not filter_obj.filter_site(mock_variant)  # expect False as no outgroup sample is present

        # test case 3: variant is an SNP, strict mode is disabled and no outgroup sample is present
        filter_obj = fd.DeviantOutgroupFiltration(outgroups=['outgroup1'], ingroups=['ingroup1'], strict_mode=False)
        filter_obj.samples = np.array(['ingroup1', 'outgroup1'])
        filter_obj._create_masks()
        mock_variant.gt_bases = np.array(['A/A', './.'])
        assert filter_obj.filter_site(mock_variant)  # # expect True as strict mode off and no outgroup sample present

        # test case 4: variant is an SNP and outgroup base is different from ingroup base
        filter_obj = fd.DeviantOutgroupFiltration(outgroups=['outgroup1'], ingroups=['ingroup1'])
        filter_obj.samples = np.array(['ingroup1', 'outgroup1'])
        filter_obj._create_masks()
        mock_variant.gt_bases = np.array(['A/A', 'T/T'])
        assert not filter_obj.filter_site(mock_variant)  # expect False as outgroup base is different from ingroup base

        # test case 5: variant is an SNP and outgroup base is same as ingroup base
        filter_obj = fd.DeviantOutgroupFiltration(outgroups=['outgroup1'], ingroups=['ingroup1'])
        filter_obj.samples = np.array(['ingroup1', 'outgroup1'])
        filter_obj._create_masks()
        mock_variant.gt_bases = np.array(['A/A', 'A/A'])
        assert filter_obj.filter_site(mock_variant)  # expect True as outgroup base is same as ingroup base

        # test case 6: multiple ingroups and outgroups with matching major bases
        mock_variant = Mock(spec=Variant)
        mock_variant.is_snp = True
        filter_obj = fd.DeviantOutgroupFiltration(outgroups=['outgroup1', 'outgroup2'],
                                                  ingroups=['ingroup1', 'ingroup2'])
        filter_obj.samples = np.array(['ingroup1', 'ingroup2', 'outgroup1', 'outgroup2'])
        filter_obj._create_masks()
        mock_variant.gt_bases = np.array(['A/A', 'A/T', 'A/A', 'A/G'])
        assert filter_obj.filter_site(mock_variant)  # expect True as major base 'A' is common in ingroup and outgroup

        # test case 7: multiple ingroups and outgroups with differing major bases
        mock_variant = Mock(spec=Variant)
        mock_variant.is_snp = True
        filter_obj = fd.DeviantOutgroupFiltration(outgroups=['outgroup1', 'outgroup2'],
                                                  ingroups=['ingroup1', 'ingroup2'])
        filter_obj.samples = np.array(['ingroup1', 'ingroup2', 'outgroup1', 'outgroup2'])
        filter_obj._create_masks()
        mock_variant.gt_bases = np.array(['A/A', 'A/T', 'T/T', 'T/G'])
        assert not filter_obj.filter_site(mock_variant)  # expect False as major base 'A' in ingroup and 'T' in outgroup

        # test case 8: make sure we retain monoallelic sites if retain_monomorphic is True
        mock_variant = Mock(spec=Variant)
        mock_variant.is_snp = False
        filter_obj = fd.DeviantOutgroupFiltration(outgroups=['outgroup1', 'outgroup2'],
                                                  ingroups=['ingroup1', 'ingroup2'], retain_monomorphic=True)
        filter_obj.samples = np.array(['ingroup1', 'ingroup2', 'outgroup1', 'outgroup2'])
        filter_obj._create_masks()
        mock_variant.gt_bases = np.array(['A/A', 'A/A', 'T/T', 'T/T'])
        assert filter_obj.filter_site(mock_variant)  # expect True as major base 'A' in ingroup and 'T' in outgroup

        # test case 9: make sure we don't retain monoallelic sites if retain_monomorphic is False
        mock_variant = Mock(spec=Variant)
        mock_variant.is_snp = False
        filter_obj = fd.DeviantOutgroupFiltration(outgroups=['outgroup1', 'outgroup2'],
                                                  ingroups=['ingroup1', 'ingroup2'], retain_monomorphic=False)
        filter_obj.samples = np.array(['ingroup1', 'ingroup2', 'outgroup1', 'outgroup2'])
        filter_obj._create_masks()
        mock_variant.gt_bases = np.array(['A/A', 'A/A', 'T/T', 'T/T'])
        assert not filter_obj.filter_site(mock_variant)  # expect False as major base 'A' in ingroup and 'T' in outgroup

    @staticmethod
    def test_existing_outgroup_filtration_single_site():
        """
        Test the existing outgroup filtration.
        """
        # test case 1: variants has one fully defined outgroup sample
        mock_variant = Mock(spec=Variant)
        filter_obj = fd.ExistingOutgroupFiltration(outgroups=['outgroup1'])
        filter_obj.samples = np.array(['ingroup1', 'outgroup1'])
        filter_obj._create_mask()
        mock_variant.gt_bases = np.array(['A/A', 'T/T'])
        assert filter_obj.filter_site(mock_variant)

        # test case 2: variants has one missing outgroup sample
        mock_variant = Mock(spec=Variant)
        filter_obj = fd.ExistingOutgroupFiltration(outgroups=['outgroup1'])
        filter_obj.samples = np.array(['ingroup1', 'outgroup1'])
        filter_obj._create_mask()
        mock_variant.gt_bases = np.array(['A/A', './.'])
        assert not filter_obj.filter_site(mock_variant)

        # test case 3: variants has one fully defined outgroup sample and one missing outgroup sample
        mock_variant = Mock(spec=Variant)
        filter_obj = fd.ExistingOutgroupFiltration(outgroups=['outgroup1', 'outgroup2'])
        filter_obj.samples = np.array(['outgroup1', 'ingroup1', 'outgroup2'])
        filter_obj._create_mask()
        mock_variant.gt_bases = np.array(['./.', 'A/A', 'T/T'])
        assert not filter_obj.filter_site(mock_variant)

        # test case 4: variants has one outgroup sample with one missing allele
        mock_variant = Mock(spec=Variant)
        filter_obj = fd.ExistingOutgroupFiltration(outgroups=['outgroup1'])
        filter_obj.samples = np.array(['ingroup1', 'outgroup1'])
        filter_obj._create_mask()
        mock_variant.gt_bases = np.array(['A/A', 'T/.'])
        assert filter_obj.filter_site(mock_variant)

        # test case 5: variants has three outgroup samples with one missing allele each
        mock_variant = Mock(spec=Variant)
        filter_obj = fd.ExistingOutgroupFiltration(outgroups=['outgroup1', 'outgroup2', 'outgroup3'])
        filter_obj.samples = np.array(['ingroup1', 'outgroup1', 'outgroup2', 'outgroup3'])
        filter_obj._create_mask()
        mock_variant.gt_bases = np.array(['A/A', 'T/.', 'T/.', 'T/.'])
        assert filter_obj.filter_site(mock_variant)

    @staticmethod
    def test_snp_filtration():
        """
        Test the SNP filtration.
        """
        f = fd.SNPFiltration()

        assert not f.filter_site(variant=Mock(is_snp=False))
        assert f.filter_site(variant=Mock(is_snp=True))

    @staticmethod
    def test_snv_filtration():
        """
        Test the SNV filtration.
        """
        f = fd.SNVFiltration()

        assert not f.filter_site(variant=Mock(REF='AG', ALT=['A', 'G']))
        assert f.filter_site(variant=Mock(REF='A', ALT=['G']))
        assert f.filter_site(variant=Mock(REF='A', ALT=['G', 'C']))
        assert not f.filter_site(variant=Mock(REF='A', ALT=['GA']))
        assert f.filter_site(variant=Mock(REF='A', ALT=['G', 'C', 'T']))

    @staticmethod
    def test_no_poly_allelic_filtration():
        """
        Test the no poly-allelic filtration.
        """
        f = fd.PolyAllelicFiltration()

        assert not f.filter_site(variant=Mock(ALT=['T', 'G']))
        assert not f.filter_site(variant=Mock(ALT=['T', 'G', 'C']))
        assert f.filter_site(variant=Mock(ALT=['T']))
        assert f.filter_site(variant=Mock(ALT=[]))

    def test_coding_sequence_filtration_raises_error_if_no_fasta_given(self):
        """
        Test the coding sequence filtration.
        """
        with self.assertRaises(ValueError) as error:
            f = fd.Filterer(
                vcf="resources/genome/betula/biallelic.subset.10000.vcf.gz",
                output='scratch/test_coding_sequence_filtration.vcf',
                filtrations=[fd.CodingSequenceFiltration()],
            )

            f.filter()

            print(error)

    def test_coding_sequence_filtration(self):
        """
        Test the coding sequence filtration.
        """
        f = fd.Filterer(
            vcf="resources/genome/betula/biallelic.subset.10000.vcf.gz",
            output='scratch/test_coding_sequence_filtration.vcf',
            gff="resources/genome/betula/genome.gff",
            filtrations=[fd.CodingSequenceFiltration()],
        )

        f.filter()

        # assert no sites were filtered
        assert f.n_filtered == 6434

        # assert number of sites is the same
        assert count_sites(f.vcf) - f.n_filtered == count_sites(f.output)
