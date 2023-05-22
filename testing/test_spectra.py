from testing import prioritize_installed_packages

prioritize_installed_packages()

from unittest import TestCase

import dadi
import numpy as np
import pandas as pd
from numpy import testing
from fastdfe import spectrum, Spectra, Spectrum


class SpectraTestCase(TestCase):
    n = 20

    show_plots = True

    def test_multiply_by_scalar(self):
        s = Spectra(dict(
            all=np.arange(20),
            sub=np.arange(3, 23)
        )) * 4

        testing.assert_array_equal(np.arange(20) * 4, s[['all']].to_list()[0])
        testing.assert_array_equal(np.arange(3, 23) * 4, s[['sub']].to_list()[0])

    def test_restore_from_list(self):
        s = Spectra(dict(
            all=np.arange(20),
            sub=np.arange(3, 23)
        )) * 4

        s2 = s.from_list(s.to_list(), s.types)

        testing.assert_array_equal(s.to_numpy(), s2.to_numpy())

    def test_multiply_by_list(self):
        s = Spectra(dict(
            all=np.arange(20),
            sub=np.arange(3, 23)
        )) * [2, 4]

        testing.assert_array_equal(np.arange(20) * 2, s[['all']].to_list()[0])
        testing.assert_array_equal(np.arange(3, 23) * 4, s[['sub']].to_list()[0])

    def test_create_from_spectrum_dict(self):
        s = Spectra.from_spectra(dict(
            all=spectrum.standard_kingman(self.n) * 4,
            sub=spectrum.standard_kingman(self.n) * 2
        ))

        testing.assert_array_equal((spectrum.standard_kingman(self.n) * 4).to_list(), s['all'].to_list())
        testing.assert_array_equal((spectrum.standard_kingman(self.n) * 2).to_list(), s['sub'].to_list())

    def test_create_from_spectrum(self):
        s = Spectra.from_spectrum(spectrum.standard_kingman(self.n))

        testing.assert_array_equal(spectrum.standard_kingman(self.n).to_list(), s['all'].to_list())
        testing.assert_equal(self.n, s.n)
        testing.assert_equal(1, s.k)

    def test_create_from_array_pass_targets_div(self):
        data = spectrum.standard_kingman(self.n)
        data.data[-1] = 6543
        s = Spectra.from_list([data.to_list()], types=['all'])

        testing.assert_array_equal(data.data, s['all'].data)
        testing.assert_equal(s.n, self.n)
        testing.assert_equal(1, s.k)

    def test_create_from_dataframe(self):
        df = pd.DataFrame(dict(
            all=(spectrum.standard_kingman(self.n) * 4).to_list(),
            sub=(spectrum.standard_kingman(self.n) * 2).to_list()
        ))

        s = Spectra.from_dataframe(df)

        pd.testing.assert_frame_equal(df, s.data)

    def test_restore_from_file(self):
        s = Spectra.from_spectra(dict(
            all=spectrum.standard_kingman(self.n) * 4,
            sub=spectrum.standard_kingman(self.n) * 2
        ))

        out = "scratch/test_restore_from_file.csv"
        s.to_file(out)
        s2 = s.from_file(out)

        pd.testing.assert_frame_equal(s.data, s2.data)

    def test_restore_from_dict(self):
        s = Spectra.from_spectra(dict(
            all=spectrum.standard_kingman(self.n) * 4,
            sub=spectrum.standard_kingman(self.n) * 2
        ))

        s2 = s.from_dict(s.to_dict())

        pd.testing.assert_frame_equal(s.data, s2.data)

    def test_normalize(self):
        s = Spectra.from_spectra(dict(
            all=spectrum.standard_kingman(self.n),
            sub=spectrum.standard_kingman(self.n)
        )) * [2, 4]

        s = s.normalize()

        testing.assert_almost_equal(1, np.sum(s['all'].to_list()[:-1]))
        testing.assert_almost_equal(1, np.sum(s['sub'].to_list()[:-1]))

    def test_properties(self):
        s = Spectra.from_spectra(dict(
            all=spectrum.standard_kingman(self.n),
            sub=spectrum.standard_kingman(self.n)
        )) * [2, 4]

        testing.assert_equal(2, s.k)
        testing.assert_equal(self.n, s.n)
        testing.assert_array_equal(['all', 'sub'], s.types)

    def test_from_polydfe(self):
        data = np.arange(self.n) * 10
        n_sites = 100000000
        n_div = 12345

        s = Spectrum.from_polydfe(data, n_sites=n_sites, n_div=n_div)

        testing.assert_equal(s.to_list()[0], n_sites - np.sum(data) - n_div)
        testing.assert_equal(s.n_monomorphic, n_sites - np.sum(data))
        testing.assert_equal(s.n_polymorphic, np.sum(data))
        testing.assert_equal(s.n_div, n_div)
        testing.assert_equal(s.n_sites, n_sites)

    def test_wattersons_estimator(self):
        data = [3434, 6346, 234, 4342, 55, 525, 24, 56, 2, 42, 4]
        testing.assert_equal(Spectrum(data).theta, dadi.Spectrum(data).Watterson_theta() / sum(data))

    def test_plot_spectrum(self):
        (spectrum.standard_kingman(self.n) * 4).plot(show=self.show_plots)

    def test_plot_spectra(self):
        Spectra.from_spectra(dict(
            all=spectrum.standard_kingman(self.n) * 4,
            sub=spectrum.standard_kingman(self.n) * 2
        )).plot(show=self.show_plots)

    def test_plot_spectra_use_subplots(self):
        Spectra.from_spectra(dict(
            all=spectrum.standard_kingman(self.n) * 4,
            sub=spectrum.standard_kingman(self.n) * 2
        )).plot(show=self.show_plots, use_subplots=True)

    def test_select_wildcard(self):
        s = Spectra.from_spectra(dict(
            all=spectrum.standard_kingman(self.n),
            all2=spectrum.standard_kingman(self.n),
            sub=spectrum.standard_kingman(self.n)
        ))

        assert len(s['all*']) == 2
        assert isinstance(s['sub*'], Spectrum)
        assert isinstance(s[['sub*']], Spectra)

    def test_merge_level(self):
        s = Spectra.from_spectra({
            'bla.foo.bar': Spectrum([1000, 3, 4, 1]),
            'test1.te.x': Spectrum([100, 4, 7, 6]),
            'test2.te.x': Spectrum([55, 2, 1, 7]),
        }).merge_groups([1, 2])

        expected = Spectra.from_spectra({
            'foo.bar': Spectrum([1000, 3, 4, 1]),
            'te.x': Spectrum([155, 6, 8, 13]),
        })

        pd.testing.assert_frame_equal(s.data, expected.data)

    def test_add_spectra(self):
        s = Spectra.from_spectra({
            'bla.foo.bar': Spectrum([1000, 3, 4, 1]),
            'test1.te.x': Spectrum([100, 4, 7, 6]),
            'test2.foo.x': Spectrum([55, 2, 1, 7]),
        }) + Spectra.from_spectra({
            'bla.foo.bar': Spectrum([543, 7, 1, 2]),
            'test1.te.x': Spectrum([200, 4, 10, 6]),
            'test2.test2.x': Spectrum([55, 2, 1, 7]),
        })

        expected = Spectra.from_spectra({
            'bla.foo.bar': Spectrum([1543, 10, 5, 3]),
            'test1.te.x': Spectrum([300, 8, 17, 12]),
            'test2.foo.x': Spectrum([55, 2, 1, 7]),
            'test2.test2.x': Spectrum([55, 2, 1, 7]),
        })

        pd.testing.assert_frame_equal(s.data, expected.data)

    def test_rename_spectra(self):
        # create spectra with two subtypes and two types
        spectra = Spectra.from_spectra({
            "subtype1.type1": Spectrum.standard_kingman(10) * 1,
            "subtype1.type2": Spectrum.standard_kingman(10) * 2,
            "subtype2.type1": Spectrum.standard_kingman(10) * 3,
        })

        spectra = spectra.prefix("subsub")

        assert spectra.types == ["subsub.subtype1.type1", "subsub.subtype1.type2", "subsub.subtype2.type1"]

    def test_fold_spectrum(self):
        """
        Test that the folding of spectra works as expected
        """
        spectra = [
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            [2, 1, 4, 4, 325, 5, 213, 515]
        ]

        for s in spectra:
            np.testing.assert_array_equal(Spectrum(s).fold().data, dadi.Spectrum(s).fold().data)

    def test_spectrum_is_folded(self):
        """
        Test that the is_folded property works as expected
        """
        data = [
            dict(s=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], folded=False),
            dict(s=[1, 1, 0, 0], folded=True),
            dict(s=[1, 1, 5, 0, 0], folded=True),
            dict(s=[0, 0, 0, 0, 1], folded=False),
        ]

        for d in data:
            self.assertEqual(Spectrum(d['s']).is_folded(), d['folded'])

    def test_folding_folded_spectrum_has_no_effect(self):
        """
        Test that folding a folded spectrum has no effect
        """
        data = [
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            [1, 1, 0, 0],
            [1, 1, 5, 0, 0],
            [0, 0, 0, 0, 1],
        ]

        for s in data:
            np.testing.assert_array_equal(Spectrum(s).fold().fold().data, Spectrum(s).fold().data)

    def test_fold_spectra(self):
        """
        Test that the folding of spectra works as expected
        """
        s = Spectra.from_spectra(dict(
            type1=Spectrum([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
            type2=Spectrum([1, 2, 3, 4, 5, 6, 7, 8, 9, 65])
        ))

        s_folded = s.fold()

        for t in s.types:
            np.testing.assert_array_equal(s[t].fold().data, s_folded[t].data)

        assert not np.array(list(s.is_folded().values())).all()
        assert np.array(list(s_folded.is_folded().values())).all()

    def test_has_dots(self):
        """
        Test that the has_dots() method works as expected
        """
        assert Spectra.from_spectra(dict(
            type1=Spectrum([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
            type2=Spectrum([1, 2, 3, 4, 5, 6, 7, 8, 9, 65])
        )).has_dots() is False

        assert Spectra.from_spectra({
            'test.foo': Spectrum([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        }).has_dots() is True

    def test_replace_dots(self):
        """
        Test that the replace_dots() method works as expected
        """
        s = Spectra.from_spectra({
            'test.foo': Spectrum([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        }).replace_dots()

        assert s.has_dots() is False
        assert s.types == ['test_foo']
