import dadi
import numpy as np
import pandas as pd
from numpy import testing

from fastdfe import spectrum, Spectra, Spectrum
from testing import TestCase


class SpectraTestCase(TestCase):
    """
    Test the Spectrum and Spectra classes.
    """
    n = 20

    @staticmethod
    def test_multiply_by_scalar():
        """
        Test that multiplying a spectra by a scalar works as expected.
        """
        s = Spectra(dict(
            all=np.arange(20),
            sub=np.arange(3, 23)
        )) * 4

        testing.assert_array_equal(np.arange(20) * 4, s[['all']].to_list()[0])
        testing.assert_array_equal(np.arange(3, 23) * 4, s[['sub']].to_list()[0])

    @staticmethod
    def test_restore_from_list():
        """
        Test that restoring a spectra from a list works as expected.
        """
        s = Spectra(dict(
            all=np.arange(20),
            sub=np.arange(3, 23)
        )) * 4

        s2 = s.from_list(s.to_list(), s.types)

        testing.assert_array_equal(s.to_numpy(), s2.to_numpy())

    @staticmethod
    def test_multiply_by_list():
        """
        Test that multiplying a spectra by a list works as expected.
        """
        s = Spectra(dict(
            all=np.arange(20),
            sub=np.arange(3, 23)
        )) * [2, 4]

        testing.assert_array_equal(np.arange(20) * 2, s[['all']].to_list()[0])
        testing.assert_array_equal(np.arange(3, 23) * 4, s[['sub']].to_list()[0])

    def test_kingman_sfs(self):
        """
        Test that the kingman sfs is as expected.
        """
        np.testing.assert_array_equal(
            Spectrum.standard_kingman(10, n_monomorphic=100).data,
            [100, 1, 0.5, 1 / 3, 1 / 4, 1 / 5, 1 / 6, 1 / 7, 1 / 8, 1 / 9, 0]
        )

    def test_create_from_spectrum_dict(self):
        """
        Test that creating a spectra from a dict of spectra works as expected.
        """
        s = Spectra.from_spectra(dict(
            all=spectrum.standard_kingman(self.n) * 4,
            sub=spectrum.standard_kingman(self.n) * 2
        ))

        testing.assert_array_equal((spectrum.standard_kingman(self.n) * 4).to_list(), s['all'].to_list())
        testing.assert_array_equal((spectrum.standard_kingman(self.n) * 2).to_list(), s['sub'].to_list())

    def test_create_from_spectrum(self):
        """
        Test that creating a spectra from a single spectrum works as expected.
        """
        s = Spectra.from_spectrum(spectrum.standard_kingman(self.n))

        testing.assert_array_equal(spectrum.standard_kingman(self.n).to_list(), s['all'].to_list())
        testing.assert_equal(self.n, s.n)
        testing.assert_equal(1, s.k)

    def test_create_from_array_pass_targets_div(self):
        """
        Test that creating a spectra from an array works as expected.
        """
        data = spectrum.standard_kingman(self.n)
        data.data[-1] = 6543
        s = Spectra.from_list([data.to_list()], types=['all'])

        testing.assert_array_equal(data.data, s['all'].data)
        testing.assert_equal(s.n, self.n)
        testing.assert_equal(1, s.k)

    def test_create_from_dataframe(self):
        """
        Test that creating a spectra from a dataframe works as expected.
        """
        df = pd.DataFrame(dict(
            all=(spectrum.standard_kingman(self.n) * 4).to_list(),
            sub=(spectrum.standard_kingman(self.n) * 2).to_list()
        ))

        s = Spectra.from_dataframe(df)

        pd.testing.assert_frame_equal(df, s.data)

    def test_spectra_restore_from_file(self):
        """
        Test that restoring a spectra from a file works as expected.
        """
        s = Spectra.from_spectra(dict(
            all=spectrum.standard_kingman(self.n) * 4,
            sub=spectrum.standard_kingman(self.n) * 2
        ))

        out = "scratch/test_spectra_restore_from_file.csv"
        s.to_file(out)
        s2 = s.from_file(out)

        pd.testing.assert_frame_equal(s.data, s2.data)

    def test_spectrum_restore_from_file(self):
        """
        Test that restoring a spectrum from a file works as expected.
        """
        s = spectrum.standard_kingman(self.n)

        out = "scratch/test_spectrum_restore_from_file.csv"
        s.to_file(out)
        s2 = Spectrum.from_file(out)

        testing.assert_array_almost_equal(s.data, s2.data, decimal=15)

    def test_restore_from_dict(self):
        """
        Test that restoring a spectra from a dict works as expected.
        """
        s = Spectra.from_spectra(dict(
            all=spectrum.standard_kingman(self.n) * 4,
            sub=spectrum.standard_kingman(self.n) * 2
        ))

        s2 = s.from_dict(s.to_dict())

        pd.testing.assert_frame_equal(s.data, s2.data)

    def test_normalize(self):
        """
        Test that normalizing a spectra works as expected.
        """
        s = Spectra.from_spectra(dict(
            all=spectrum.standard_kingman(self.n),
            sub=spectrum.standard_kingman(self.n)
        )) * [2, 4]

        s = s.normalize()

        testing.assert_almost_equal(1, np.sum(s['all'].to_list()[:-1]))
        testing.assert_almost_equal(1, np.sum(s['sub'].to_list()[:-1]))

    def test_properties(self):
        """
        Test that the properties of a spectra work as expected.
        """
        s = Spectra.from_spectra(dict(
            all=spectrum.standard_kingman(self.n),
            sub=spectrum.standard_kingman(self.n)
        )) * [2, 4]

        testing.assert_equal(2, s.k)
        testing.assert_equal(self.n, s.n)
        testing.assert_array_equal(['all', 'sub'], s.types)

    def test_from_polydfe(self):
        """
        Test that the from_polydfe method works as expected.
        """
        data = np.arange(self.n) * 10
        n_sites = 100000000
        n_div = 12345

        s = Spectrum.from_polydfe(data, n_sites=n_sites, n_div=n_div)

        testing.assert_equal(s.to_list()[0], n_sites - np.sum(data) - n_div)
        testing.assert_equal(s.n_monomorphic, n_sites - np.sum(data))
        testing.assert_equal(s.n_polymorphic, np.sum(data))
        testing.assert_equal(s.n_div, n_div)
        testing.assert_equal(s.n_sites, n_sites)

    @staticmethod
    def test_wattersons_estimator():
        """
        Test that the wattersons_estimator method works as expected.
        """
        data = [3434, 6346, 234, 4342, 55, 525, 24, 56, 2, 42, 4]
        testing.assert_equal(Spectrum(data).theta, dadi.Spectrum(data).Watterson_theta() / sum(data))

    def test_plot_spectrum(self):
        """
        Test that the plot_spectrum method works as expected.
        """
        (spectrum.standard_kingman(self.n) * 4).plot()

    def test_plot_spectrum_log_scale(self):
        """
        Test that the plot_spectrum method works as expected when using a log scale.
        """
        (spectrum.standard_kingman(self.n) * 4).plot(log_scale=True)

    def test_plot_spectra(self):
        """
        Test that the plot_spectra method works as expected.
        """
        Spectra.from_spectra(dict(
            all=spectrum.standard_kingman(self.n) * 4,
            sub=spectrum.standard_kingman(self.n) * 2
        )).plot()

    def test_plot_spectra_log_scale(self):
        """
        Test that the plot_spectra method works as expected when using a log scale.
        """
        Spectra.from_spectra(dict(
            all=spectrum.standard_kingman(self.n) * 4,
            sub=spectrum.standard_kingman(self.n) * 2
        )).plot(log_scale=True)

    def test_plot_spectra_use_subplots(self):
        """
        Test that the plot_spectra method works as expected.
        """
        Spectra.from_spectra(dict(
            all=spectrum.standard_kingman(self.n) * 4,
            sub=spectrum.standard_kingman(self.n) * 2
        )).plot(use_subplots=True)

    def test_plot_spectra_use_subplots_log_scale(self):
        """
        Test that the plot_spectra method works as expected when using a log scale.
        """
        Spectra.from_spectra(dict(
            all=spectrum.standard_kingman(self.n) * 4,
            sub=spectrum.standard_kingman(self.n) * 2
        )).plot(use_subplots=True, log_scale=True)

    def test_select_wildcard(self):
        """
        Test that the select method works as expected.
        """
        s = Spectra.from_spectra(dict(
            all=spectrum.standard_kingman(self.n),
            all2=spectrum.standard_kingman(self.n),
            sub=spectrum.standard_kingman(self.n)
        ))

        assert len(s['all.*']) == 2
        assert isinstance(s['sub.*'], Spectrum)
        assert isinstance(s[['sub.*']], Spectra)

    def test_select_no_regex(self):
        """
        Test that the select method works as expected when not using regex.
        """
        s = Spectra.from_spectra(dict(
            all=spectrum.standard_kingman(self.n),
            all2=spectrum.standard_kingman(self.n),
            sub=spectrum.standard_kingman(self.n)
        ))

        assert isinstance(s.__getitem__('all', use_regex=False), Spectrum)
        assert isinstance(s.__getitem__(['sub', 'all'], use_regex=False), Spectra)
        assert len(s.__getitem__(['sub', 'all'], use_regex=False)) == 2
        assert isinstance(s.__getitem__(['sub'], use_regex=False), Spectra)
        assert len(s.__getitem__(['sub'], use_regex=False)) == 1

    @staticmethod
    def test_merge_level():
        """
        Test that the merge_level method works as expected.
        """
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

    @staticmethod
    def test_add_spectra():
        """
        Test that the add_spectra method works as expected.
        """
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

    @staticmethod
    def test_rename_spectra():
        """
        Test that the rename_spectra method works as expected.
        """
        # create spectra with two subtypes and two types
        spectra = Spectra.from_spectra({
            "subtype1.type1": Spectrum.standard_kingman(10) * 1,
            "subtype1.type2": Spectrum.standard_kingman(10) * 2,
            "subtype2.type1": Spectrum.standard_kingman(10) * 3,
        })

        spectra = spectra.prefix("subsub")

        assert spectra.types == ["subsub.subtype1.type1", "subsub.subtype1.type2", "subsub.subtype2.type1"]

    @staticmethod
    def test_fold_spectrum():
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

    @staticmethod
    def test_folding_folded_spectrum_has_no_effect():
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

    @staticmethod
    def test_fold_spectra():
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

    @staticmethod
    def test_has_dots():
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

    @staticmethod
    def test_replace_dots():
        """
        Test that the replace_dots() method works as expected
        """
        s = Spectra.from_spectra({
            'test.foo': Spectrum([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        }).replace_dots()

        assert s.has_dots() is False
        assert s.types == ['test_foo']

    @staticmethod
    def test_drop_empty():
        """
        Test that the drop_empty() method works as expected
        """
        assert Spectra.from_spectra(dict(
            type1=Spectrum([1, 0, 0, 0, 0, 0, 0, 0, 0, 10]),
            type2=Spectrum([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            type3=Spectrum([1, 2, 3, 4, 5, 6, 7, 8, 9, 65]),
        )).drop_empty().types == ['type1', 'type3']

    @staticmethod
    def test_drop_zero_entries():
        """
        Test that the drop_zero_entries() method works as expected
        """
        assert Spectra.from_spectra(dict(
            type1=Spectrum([1, 0, 0, 0, 0, 0, 0, 0, 0, 10]),
            type2=Spectrum([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            type3=Spectrum([1, 2, 3, 4, 5, 6, 7, 8, 9, 65]),
            type4=Spectrum([1, 2, 0, 4, 5, 6, 7, 8, 9, 65]),
        )).drop_zero_entries().types == ['type3']

    @staticmethod
    def test_drop_sparse():
        """
        Test that the drop_zero_entries() method works as expected
        """
        spectra = Spectra.from_spectra(dict(
            type1=Spectrum([1, 0, 0, 0, 0, 0, 0, 0, 0, 10]),
            type2=Spectrum([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            type3=Spectrum([1, 2, 3, 4, 5, 6, 7, 8, 9, 65]),
            type4=Spectrum([1, 2, 0, 4, 5, 6, 7, 8, 9, 65]),
        ))

        assert spectra.drop_sparse(n_polymorphic=0).types == ['type3', 'type4']
        assert spectra.drop_sparse(n_polymorphic=1).types == ['type3', 'type4']
        assert spectra.drop_sparse(n_polymorphic=40).types == ['type3', 'type4']
        assert spectra.drop_sparse(n_polymorphic=41).types == ['type3']
        assert spectra.drop_sparse(n_polymorphic=44).types == []

    @staticmethod
    def test_add_spectrum():
        """
        Test whether two spectrum objects can be added.
        """
        testing.assert_array_equal((Spectrum([1, 2, 3]) + Spectrum([3, 4, 5])).data, Spectrum([4, 6, 8]).data)

    @staticmethod
    def test_subtract_spectrum():
        """
        Test whether two spectrum objects can be subtracted.
        """
        testing.assert_array_equal((Spectrum([3, 4, 5]) - Spectrum([1, 2, 3])).data, [2, 2, 2])

    @staticmethod
    def test_normalize_spectrum():
        """
        Test whether two spectrum objects can be normalized.
        """
        testing.assert_array_equal(Spectrum([10, 1, 2, 3, 10]).normalize().data, [10, 1 / 6, 1 / 3, 1 / 2, 10])

    @staticmethod
    def test_spectra_n_monomorphic():
        """
        Test whether the n_monomorphic property works as expected.
        """
        spectra = Spectra.from_spectra(dict(
            type1=Spectrum([1, 0, 0, 0, 0, 0, 0, 0, 0, 10]),
            type2=Spectrum([1, 2, 0, 4, 5, 6, 7, 8, 9, 65]),
        ))

        assert np.all(spectra.n_monomorphic == pd.Series(dict(type1=11, type2=66)))

    def test_reorder_levels(self):
        """
        Test whether the reorder_levels method works as expected.
        """
        spectra = Spectra({
            "type1.subtype1.subsubtype1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "type1.subtype1.subsubtype2": [1, 4, 12, 4, 5, 6, 7, 8, 9, 10],
            "type1.subtype2.subsubtype1": [1, 7, 7, 4, 5, 6, 7, 8, 9, 10],
            "type1.subtype2.subsubtype2": [1, 8, 3, 4, 5, 6, 7, 8, 9, 10],
            "type2.subtype1.subsubtype1": [1, 0, 1, 4, 5, 6, 7, 8, 9, 10],
            "type2.subtype1.subsubtype2": [1, 6, 3, 4, 5, 6, 7, 8, 9, 10],
        })

        spectra2 = spectra.reorder_levels([1, 0, 2])

        self.assertEqual(spectra2.types, [
            "subtype1.type1.subsubtype1",
            "subtype1.type1.subsubtype2",
            "subtype2.type1.subsubtype1",
            "subtype2.type1.subsubtype2",
            "subtype1.type2.subsubtype1",
            "subtype1.type2.subsubtype2",
        ])

        testing.assert_array_equal(spectra2.to_list(), spectra.reorder_levels([1, 0, 2]).to_list())

        testing.assert_array_equal(
            list(spectra2["subtype1.type1.subsubtype1"]),
            list(spectra["type1.subtype1.subsubtype1"])
        )

    def test_subsample_spectrum_random(self):
        """
        Test whether the subsample method works as expected.
        """
        kingman = Spectrum.standard_kingman(30) * 10000

        sub = kingman.subsample(10, mode='random')

        s = Spectra.from_spectra(dict(
            actual=Spectrum.standard_kingman(10) * 10000,
            subsampled=sub
        ))

        s.plot()

        self.assertEqual(10, sub.n)

        diff_rel = np.abs(s['actual'].polymorphic - s['subsampled'].polymorphic) / s['actual'].polymorphic

        self.assertTrue(np.all(diff_rel < 0.1))

    def test_subsample_folded_spectrum(self):
        """
        Test whether the subsample method works as expected.
        """
        kingman = (Spectrum.standard_kingman(30) * 10000).fold()

        sub = kingman.subsample(10, mode='probabilistic')

        truth = (Spectrum.standard_kingman(10) * 10000).fold()

        self.assertTrue(np.all(np.abs(truth.data[1:] - sub.data[1:]) < 1e-12))

    def test_subsample_spectrum_probabilistic_vs_random_folded(self):
        """
        Test random vs probabilistic subsampling.
        """
        kingman = (Spectrum.standard_kingman(30) * 10000).fold()

        sub_random = kingman.subsample(10, mode='random')
        sub_probabilistic = kingman.subsample(10, mode='probabilistic')

        s = Spectra.from_spectra(dict(
            actual=(Spectrum.standard_kingman(10) * 10000).fold(),
            random=sub_random,
            probabilistic=sub_probabilistic
        ))

        s.plot()

        diff_rel = np.abs(s['actual'].polymorphic - s['probabilistic'].polymorphic) / s['probabilistic'].polymorphic

        self.assertTrue(np.all(diff_rel[:5] < 1e-15))

    def test_subsample_spectrum_probabilistic_vs_random_unfolded(self):
        """
        Test random vs probabilistic subsampling.
        """
        kingman = Spectrum.standard_kingman(30) * 10000

        sub_random = kingman.subsample(10, mode='random')
        sub_probabilistic = kingman.subsample(10, mode='probabilistic')

        s = Spectra.from_spectra(dict(
            actual=Spectrum.standard_kingman(10) * 10000,
            random=sub_random,
            probabilistic=sub_probabilistic
        ))

        s.plot()

        diff_rel = np.abs(s['actual'].polymorphic - s['probabilistic'].polymorphic) / s['probabilistic'].polymorphic

        self.assertTrue(np.all(diff_rel < 1e-15))

    def test_larger_subsample_size_raises_value_error(self):
        """
        Test whether the subsample method raises a ValueError when the requested subsample size is larger than the
        original sample size.
        """
        self.assertRaises(ValueError, Spectrum.standard_kingman(30).subsample, 40)

    @staticmethod
    def test_subsample_spectra():
        """
        Test whether the subsample method works as expected.
        """
        spectra = Spectra({
            "type1.subtype1.subsubtype1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "type1.subtype1.subsubtype2": [1, 4, 12, 4, 5, 6, 7, 8, 9, 10],
        })

        spectra.subsample(7, mode='random').plot()

        # make sure n_sites is the same
        testing.assert_array_equal(spectra.subsample(7).n_sites, spectra.n_sites)

    def test_has_divergence(self):
        """
        Test whether the has_divergence method works as expected.
        """
        s1 = Spectrum([1, 2, 3, 0])
        s2 = Spectrum([1, 2, 3, 4])

        self.assertFalse(s1.has_div)
        self.assertTrue(s2.has_div)

        self.assertTrue(Spectra.from_spectra({
            "type1": s1,
            "type2": s2
        }).has_div.any())

        self.assertFalse(Spectra.from_spectra({
            "type1": s1,
            "type2": s1
        }).has_div.all())

    @staticmethod
    def test_copy_spectrum():
        """
        Test whether the copy method works as expected.
        """
        s = Spectrum([1, 2, 3, 4, 5])

        s2 = s.copy()

        testing.assert_array_equal(s.data, s2.data)

    @staticmethod
    def test_spectrum_power():
        """
        Test whether the power method works as expected.
        """
        s = Spectrum([1, 2, 3, 4, 5])

        testing.assert_array_equal((s ** 2).data, [1, 4, 9, 16, 25])

    @staticmethod
    def test_spectrum_divide():
        """
        Test whether the divide method works as expected.
        """
        s = Spectrum([1, 2, 3, 4, 5])

        testing.assert_array_equal((s / 2).data, [0.5, 1, 1.5, 2, 2.5])

    @staticmethod
    def test_spectrum_floor_divide():
        """
        Test whether the floor divide method works as expected.
        """
        s = Spectrum([1, 2, 3, 4, 5])

        testing.assert_array_equal((s // 2).data, [0, 1, 1, 2, 2])

    @staticmethod
    def test_spectra_to_dataframe():
        """
        Test whether the to_dataframe method works as expected.
        """
        s = Spectra.from_spectra(dict(
            all=spectrum.standard_kingman(10),
            sub=spectrum.standard_kingman(10) * 2
        ))

        df = s.to_dataframe()

        pd.testing.assert_frame_equal(df, s.data)

    def test_spectra_floor_divide(self):
        """
        Test whether the floor divide method works as expected.
        """
        s = Spectra.from_spectra(dict(
            all=spectrum.standard_kingman(10),
            sub=spectrum.standard_kingman(10) * 2
        ))

        s2 = s // 2

        testing.assert_array_equal(s2['all'].data, s['all'].data // 2)
        testing.assert_array_equal(s2['sub'].data, s['sub'].data // 2)

    def test_misidentify_value_error(self):
        """
        Test whether the misidentify method raises a ValueError when the rate is not between 0 and 1.
        """
        sfs = Spectrum([0, 1, 2])

        with self.assertRaises(ValueError):
            sfs.misidentify(-0.1)

        with self.assertRaises(ValueError):
            sfs.misidentify(1.5)

    def test_misidentify_expected_sfs_even_n(self):
        """
        Test whether the misidentify method works as expected for even n.
        """
        sfs = Spectrum([177595, 922, 363, 210, 164, 162, 712])

        np.testing.assert_array_equal(
            sfs.misidentify(0.1).data,
            [177595, 922 * 0.9 + 162 * 0.1, 363 * 0.9 + 164 * 0.1, 210,
             164 * 0.9 + 363 * 0.1, 162 * 0.9 + 922 * 0.1, 712]
        )

    def test_misidentify_expected_sfs_odd_n(self):
        """
        Test whether the misidentify method works as expected for odd n.
        """
        sfs = Spectrum([177749, 889, 347, 214, 189, 739])

        np.testing.assert_array_equal(
            sfs.misidentify(0.1).data,
            [177749, 889 * 0.9 + 189 * 0.1, 347 * 0.9 + 214 * 0.1,
             214 * 0.9 + 347 * 0.1, 189 * 0.9 + 889 * 0.1, 739]
        )

    def test_misidentify_epislon_0(self):
        """
        Test whether the misidentify method works as expected for epsilon = 0.
        """
        sfs = Spectrum([177595, 922, 363, 210, 164, 162, 712])

        np.testing.assert_array_equal(
            sfs.misidentify(0).data,
            sfs.data
        )

    def test_misidentify_epislon_1(self):
        """
        Test whether the misidentify method works as expected for epsilon = 1.
        """
        sfs = Spectrum([177595, 922, 363, 210, 164, 162, 712])

        np.testing.assert_array_equal(
            sfs.misidentify(1).data,
            [sfs.data[0]] + sfs.data[1:sfs.n][::-1].tolist() + [sfs.data[-1]]
        )
