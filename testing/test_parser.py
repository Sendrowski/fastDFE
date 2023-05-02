from testing import prioritize_installed_packages

prioritize_installed_packages()

from unittest import TestCase

import dadi
import numpy as np

import fastdfe
from fastdfe import DegeneracyStratification


class InferenceTestCase(TestCase):
    """
    TODO test stratifications.
    """
    vcf_file = 'resources/genome/betula/biallelic.vcf.gz'
    pop_file = 'resources/genome/betula/pops_dadi.txt'

    def test_compare_sfs_with_data(self):
        """
        Compare the sfs from dadi with the one from the data.
        """
        p = fastdfe.Parser(vcf_file=self.vcf_file, n=20, stratifications=[])

        sfs = p.parse().all

        data_dict = dadi.Misc.make_data_dict_vcf(self.vcf_file, self.pop_file)

        sfs2 = dadi.Spectrum.from_data_dict(data_dict, ['pop0'], [20], polarized=True)

        # this can be improved but the values are rather equal
        assert np.max(np.abs(sfs.to_numpy() - sfs2.data) / sfs2.data) < 0.2

    def test_degeneracy_stratification(self):
        """
        Test the degeneracy stratification.
        """
        p = fastdfe.Parser(vcf_file=self.vcf_file, n=20, stratifications=[DegeneracyStratification()])

        sfs = p.parse()

        assert sfs['neutral'].to_list() == [22776.0, 4075.0, 1413.0, 803.0, 532.0, 365.0, 282.0, 180.0, 136.0, 125.0,
                                            112.0, 93.0, 94.0, 111.0, 130.0, 130.0, 131.0, 109.0, 84.0, 43.0, 15.0]
        assert sfs['selected'].to_list() == [47250.0, 6100.0, 1876.0, 1036.0, 684.0, 438.0, 321.0, 232.0, 173.0, 138.0,
                                             105.0, 112.0, 125.0, 140.0, 145.0, 161.0, 154.0, 144.0, 98.0, 37.0, 17.0]
