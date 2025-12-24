from unittest.mock import patch

import numpy as np
import pytest
from matplotlib import pyplot as plt
from scipy.special import hyp1f1
from tqdm import tqdm

import fastdfe as fd
from fastdfe import GammaExpParametrization
from fastdfe.discretization import Discretization
from testing import TestCase


class DiscretizationTestCase(TestCase):
    """
    Test the Discretization class.
    """
    n = 20

    @staticmethod
    def diff(a, b):
        """
        Difference.
        """
        return a - b

    def diff_rel(self, a, b):
        """
        Relative difference.
        """
        return self.diff(a, b) / a

    def diff_max_abs(self, a, b):
        """
        Maximum absolute difference.
        """
        return np.max(np.abs(self.diff(a, b)))

    def diff_rel_max_abs(self, a, b):
        """
        Maximum absolute relative difference.
        """
        return np.max(np.abs(self.diff_rel(a, b)))

    def test_dfe_to_sfs_linearization_midpoint_vs_quad(self):
        """
        Check for discrepancies between the linearized DFE to SFS
        transformation using midpoint integration vs scipy's quad.
        """
        opts = dict(
            n=self.n,
            h=0.5,
            intervals_ben=(1.0e-5, 100, 100),
            intervals_del=(-10000, -1.0e-5, 100)
        )

        d_midpoint = Discretization(**opts, integration_mode='midpoint')
        d_quad = Discretization(**opts, integration_mode='quad')

        P_quad = d_midpoint.get_counts(h=0.5)
        P_lin = d_quad.get_counts(h=0.5)

        # P_diff_rel = self.diff_rel(P_quad, P_lin)
        diff = self.diff_rel_max_abs(P_quad, P_lin)

        # this is rather large but is due to points near 0
        # for which we have rather small values (on the order of 1e-17)
        assert diff < 1

        model = GammaExpParametrization()
        params = GammaExpParametrization.x0 | {'h': 0.5}

        sfs_quad = d_quad.model_selection_sfs(model, params)
        sfs_mid = d_midpoint.model_selection_sfs(model, params)

        diff = self.diff_rel_max_abs(sfs_quad, sfs_mid)

        assert diff < 5e-3

    def test_sfs_counts_linearized_vs_quad(self):
        """
        Compare precomputed linearization using scipy's quad with ad hoc integration.
        """
        model = GammaExpParametrization()
        params = GammaExpParametrization.x0 | {'h': 0.5}

        opts = dict(
            n=self.n,
            integration_mode='quad',
            intervals_ben=(1.0e-5, 100, 100),
            intervals_del=(-5000, -1.0e-5, 100)
        )

        sfs_adhoc = Discretization(**opts, linearized=False).model_selection_sfs(model, params)
        sfs_cached = Discretization(**opts, linearized=True).model_selection_sfs(model, params)

        diff = np.max(np.abs((sfs_cached - sfs_adhoc) / sfs_cached))

        assert diff < 4e-3

    @pytest.mark.skip(reason="np.float128 not available on osx-arm64")
    def test_compare_H_with_regularized(self):
        """
        Compare H with its regularized equivalent that
        use the limit as S -> 0 for small S.
        Note, behaviour similar for other values of x.
        """
        s = np.logspace(np.log10(1e-20), np.log10(1e-10), 10000)
        fig, axs = plt.subplots(nrows=2)
        fig.tight_layout(pad=4)

        for ax in axs:
            ax.plot(np.abs(s), Discretization.get_counts_high_precision(0.95, s), label='H(s)')
            ax.plot(np.abs(s), Discretization.get_counts_high_precision_regularized(0.95, s), label='$H_{reg}(s)$')

            s = -s

            ax.set_xlabel('S' if s[0] > 0 else '-S')
            ax.set_ylabel('H')
            ax.set_xscale('log')
            ax.set_title('sojourn times')

        plt.legend()
        plt.show()

    def test_allele_counts_large_negative_S(self):
        """
        Plot divergence vs S.
        """
        d = Discretization(
            n=self.n,
            linearized=True
        )

        S = np.linspace(-100, -10000, 100)
        k = np.arange(1, self.n)
        I = np.ones((S.shape[0], k.shape[0]))
        c1 = d.get_counts_semidominant_unregularized(S[:, None] * I, k[None, :] * I)
        c2 = d.get_counts_semidominant_large_negative_S(S[:, None] * I, k[None, :] * I)

        diff = self.diff_rel_max_abs(c1, c2)

        assert diff < 1e-15

    def test_hyp1f1_mpmath_vs_scipy(self):
        """
        Compare mpmath and scipy's implementation of the confluent hypergeometric function.
        """
        s = np.linspace(-500, 500, 1000)
        k = np.arange(1, self.n)
        I = np.ones((s.shape[0], k.shape[0]))

        S = s[:, None] * I
        K = k[None, :] * I

        y_scipy = hyp1f1(K, self.n, S)
        y_mpmath = Discretization.hyp1f1(K, self.n, S)

        assert self.diff_rel_max_abs(y_scipy, y_mpmath) < 1e-11

    def test_allele_counts_large_positive_S(self):
        """
        Plot divergence vs S.
        """
        d = Discretization(
            n=self.n,
            linearized=True
        )

        S = np.linspace(900000, 1000000, 100)
        k = np.arange(1, self.n)
        I = np.ones((S.shape[0], k.shape[0]))
        c1 = d.get_counts_semidominant_unregularized(S[:, None] * I, k[None, :] * I)
        c2 = d.get_counts_semidominant_large_positive_S(S[:, None] * I, k[None, :] * I)

        diff = self.diff_rel_max_abs(c1, c2)

        # still relatively large, but we can now evaluate large S
        # using mpmath
        assert diff < 1e-4

    def test_plot_allele_count_regularized(self):
        """
        Plot allele count for different values of k.
        """
        d = Discretization(
            n=self.n,
            intervals_ben=(1.0e-15, 100, 1000),
            intervals_del=(-1000000, -1.0e-15, 1000)
        )

        fig, (ax1, ax2) = plt.subplots(nrows=2)
        fig.tight_layout(pad=4)

        for k in range(1, self.n):
            ax2.plot(np.arange(d.s[d.s != 0].shape[0]),
                     d.get_counts_semidominant_unregularized(d.s[d.s != 0], k * np.ones_like(d.s[d.s != 0])),
                     label=f"k = {k}")
            ax1.set_title('default')

        for k in range(1, self.n):
            ax1.plot(np.arange(d.s.shape[0]), d.get_counts_semidominant_regularized(d.s, k * np.ones_like(d.s)),
                     label=f"k = {k}")
            ax1.set_title('regularized')

        plt.legend(prop=dict(size=5))
        plt.show()

    def test_plot_H_fixed_with_regularized_negative(self):
        """
        Plot H_fixed with and without regularization for negative values of S.
        """

        d = Discretization(
            n=self.n,
            intervals_ben=(1.0e-15, 100, 0),
            intervals_del=(-1000000, -1.0e-15, 1000)
        )

        plt.plot(np.arange(len(d.s)), d.get_counts_fixed_semidominant_regularized(d.s), alpha=0.5, label='regularized')
        plt.plot(np.arange(len(d.s)), d.get_counts_fixed_semidominant(d.s), alpha=0.5, label='normal')

        plt.legend()
        plt.show()

    def test_plot_H_fixed_with_regularized_positive(self):
        """
        Plot H_fixed with and without regularization for positive values of S.
        """

        d = Discretization(
            n=self.n,
            intervals_ben=(1.0e-15, 1, 1000),
            intervals_del=(-1000000, -1.0e-15, 0)
        )

        plt.plot(np.arange(len(d.s)), d.get_counts_fixed_semidominant_regularized(d.s), alpha=0.5,
                 label='positive regularized')
        plt.plot(np.arange(len(d.s)), d.get_counts_fixed_semidominant(d.s), alpha=0.5, label='positive normal')

        plt.legend()
        plt.show()

    def test_compare_with_semidominant(self):
        """
        Compare regularized allele counts with those obtained
        from H_h for semi-dominant case (h = 0.5).
        """
        m = 5
        diff = np.zeros((3, m, 201))
        for i, n in enumerate([6, 20, 100]):
            d = Discretization(
                n=n,
                intervals_ben=(1.0e-5, 1.0e4, 100),
                intervals_del=(-1.0e+8, -1.0e-5, 100)
            )

            for j, k in tqdm(enumerate(np.linspace(1, n - 1, m).astype(int)), total=m):
                ys1 = d._get_counts_dominant(n=d.n, k=k, S=d.s, h=0.5)
                ys2 = d.get_counts_semidominant_regularized(k=np.full_like(d.s, k), S=d.s)

                diff[i, j] = (np.abs(ys1 - ys2) / (ys2 + 1e-10))

        diff_max = np.max(diff, axis=(1, 2))
        diff_mean = np.mean(diff, axis=(1, 2))
        self.assertLess(diff.mean(), 0.004)
        self.assertTrue((diff_max < np.array([0.02, 0.06, 0.3])).all())
        pass

    def test_cache_fixed_h_not_semidominant(self):
        """
        Make sure that caching of dominance-specific allele counts invokes the correct methods.
        """
        d = Discretization(
            n=10,
            h=0.2,
            intervals_ben=(1e-5, 1e4, 100),
            intervals_del=(-1e8, -1e-5, 100),
            intervals_h=(0.0, 1.0, 5),
            parallelize=False
        )

        with patch.object(d, "get_counts_dominant", wraps=d.get_counts_dominant) as spy:
            d.precompute()
            spy.assert_called()

    def test_cache_fixed_h_semidominant(self):
        """
        Make sure that caching of semidominant allele counts invokes the correct methods.
        """
        d = Discretization(
            n=10,
            h=0.5,
            intervals_ben=(1e-5, 1e4, 100),
            intervals_del=(-1e8, -1e-5, 100),
            intervals_h=(0.0, 1.0, 5),
            parallelize=False
        )

        with patch.object(d, "get_counts_semidominant", wraps=d.get_counts_semidominant) as spy:
            d.precompute()
            spy.assert_called()

    def test_cache_dominance(self):
        """
        Test caching of dominance-specific allele counts.
        """
        d = Discretization(
            n=10,
            h=None,
            intervals_ben=(1.0e-5, 1.0e4, 100),
            intervals_del=(-1.0e+8, -1.0e-5, 100),
            intervals_h=(0.0, 1.0, 5)
        )

        d.precompute()

        np.testing.assert_array_equal(d.grid_h, [0.0, 0.25, 0.5, 0.75, 1.0])

        T = d.get_counts_semidominant()
        T2 = d.get_counts(h=0.5)
        T3 = d.get_counts(h=0.5 - 1e-5)
        T4 = d.get_counts(h=0.5 + 1e-5)

        diff = (np.abs(T2 - T) / (T + 1e-10))
        diff_max = np.max(diff)
        diff_mean = np.mean(diff)

        self.assertLess(diff_max, 0.03)
        self.assertLess(diff_mean, 1e-3)
        self.assertLess((np.abs(T2 - T3) / (T2 + 1e-10)).max(), 0.01)
        self.assertLess((np.abs(T2 - T4) / (T2 + 1e-10)).max(), 0.01)

    def test_plot_over_h(self):
        """
        Plot allele counts over h for different h callbacks.
        """
        for callback in [lambda k, S: np.full_like(S, k), lambda k, S: np.full_like(S, np.sqrt(k))]:
            d = Discretization(
                n=10,
                h=None,
                h_callback=callback,
                intervals_ben=(1, 1000, 5),
                intervals_del=(-1.0e+8, -1, 5),
                intervals_h=(0, 1, 10)
            )

            hs = np.linspace(0, 1, 101)
            counts = np.array([d.get_counts(h=h) for h in hs]).transpose(1, 2, 0)

            fig, ax = plt.subplots(
                nrows=d.n // 3,
                ncols=3,
                figsize=(14, 10)
            )
            ax = ax.flatten()

            for k in range(d.n - 1):
                for j in range(d.s.shape[0]):
                    ax[k].plot(hs, counts[k, j], alpha=0.7, label=f"S={d.s[j]:.2f}", linewidth=2)
                ax[k].set_ylabel("SFS count")
                ax[k].set_xlabel("h")
                ax[k].set_yscale("log")
                ax[k].set_title(f"k={k + 1}")

            ax[0].legend(prop=dict(size=8))
            fig.tight_layout()
            plt.show()

    def test_default_h_callback(self):
        """
        Make sure default results in single fixed h of 0.5.
        """
        d = Discretization(
            n=10,
            h_callback=lambda k, S: np.full_like(S, k),
            h=0.5,
            intervals_h=(0.0, 1.0, 6),
            intervals_ben=(1.0e-5, 1.0e4, 100),
            intervals_del=(-1e8, -1.0e-5, 100)
        )

        self.assertEqual(d.grid_h, None)

    def test_custom_h_fixed_callback_fixed_h(self):
        """
        Make sure passing fixed h callback and fixed h results in correct grid.
        """
        d = Discretization(
            n=10,
            h=0.5,
            h_callback=lambda k, S: np.full_like(S, k - 0.1),
            intervals_ben=(1.0e-5, 1.0e4, 100),
            intervals_del=(-1e8, -1.0e-5, 100)
        )

        self.assertEqual(d.grid_h, None)

    def test_custom_h_fixed_callback_variable_h(self):
        """
        Make sure fixed h callback with variable h results in default grid.
        """
        d = Discretization(
            n=10,
            h=None,
            h_callback=lambda k, S: np.full_like(S, k - 0.1),
            intervals_h=(0.0, 1.0, 6),
            intervals_ben=(1.0e-5, 1.0e4, 100),
            intervals_del=(-1e8, -1.0e-5, 100),
        )

        np.testing.assert_array_equal(d.grid_h, np.linspace(*d.intervals_h))

        # test interpolation
        counts1 = d.get_counts(h=0.5)
        counts2 = d.get_counts(h=0.6)
        np.testing.assert_array_almost_equal((counts1 + counts2) / 2, d.get_counts(h=0.55))

    def test_custom_h_variable_callback_fixed_h(self):
        """
        Make sure variable h callback with fixed h results in correct grid and interpolation.
        """
        d = Discretization(
            n=10,
            h=0.5,
            h_callback=lambda k, S: 0.4 * np.exp(-k * abs(S)),
            intervals_h=(0.0, 1.0, 6),
            intervals_ben=(1.0e-5, 1.0e4, 100),
            intervals_del=(-1e8, -1.0e-5, 100)
        )

        self.assertEqual(d.grid_h, None)

    def test_interpolation_weights_variable_h_fixed_callback(self):
        """
        Make sure variable h callback with fixed h results in correct grid and interpolation.
        """
        d = Discretization(
            n=10,
            h=None,
            h_callback=lambda k, S: 0.4 * np.exp(-k * abs(S)),
            intervals_h=(0.0, 1.0, 6),
            intervals_ben=(1.0e-5, 1.0e4, 100),
            intervals_del=(-1e8, -1.0e-5, 100)
        )

        np.testing.assert_array_equal(d.grid_h, np.linspace(*d.intervals_h))

        i, w = d.get_interpolation_weights(h=0.5)
        hs = d.map_h(0.5, d.s)
        np.testing.assert_array_almost_equal(i + w, hs * 5 + 1)
        x = np.arange(len(d.s))

        plt.plot(x, i + w, label='index + weight')
        plt.plot(x, w, label='weights')
        plt.plot(x, i, label='indices')
        plt.plot(x, hs, label='h callback')
        plt.title("Interpolation indices + weights over S")
        plt.xlabel("S")
        plt.ylabel("Index + weight")
        plt.legend()
        plt.show()

        pass

    def test_plot_custom_h_callback(self):
        """
        Plot allele counts over h for a custom h callback.
        """
        h = lambda k, S: 0.4 * np.exp(-k * abs(S))

        plt_s = np.logspace(-5, 2, 100)
        plt_k = [0.1, 1, 10, 100]
        plt.figure(figsize=(10, 6))
        for k in plt_k:
            hs = h(k, plt_s)
            plt.plot(plt_s, hs, label=f'k={k}')
        plt.xscale('log')
        plt.xlabel('S')
        plt.ylabel('h')
        plt.title('Custom h callback over S for different k')
        plt.legend()
        plt.show()

    def test_fixed_h_vs_interpolated_h(self):
        """
        Compare allele counts obtained using fixed h(S) vs interpolated h.
        """
        k = 1

        d1 = Discretization(
            n=10,
            h=0.5,
            h_callback=lambda k, S: 0.4 * np.exp(-k * abs(S)),
            intervals_h=(0.0, 1.0, 6),
            intervals_ben=(1.0e-5, 1.0e4, 100),
            intervals_del=(-1e8, -1.0e-5, 100)
        )

        c1 = d1.get_counts(0.5)

        d2 = Discretization(
            n=10,
            h=None,
            h_callback=d1.h_callback,
            intervals_h=d1.intervals_h,
            intervals_ben=d1.intervals_ben,
            intervals_del=d1.intervals_del
        )

        c2 = d2.get_counts(0.5)

        diff = self.diff_rel_max_abs(c1, c2)
        self.assertLess(diff, 0.03)

    def test_compare_fixed_counts_with_semidominant(self):
        """
        Compare Discretization.get_counts_fixed_dominant and Discretization.get_counts_fixed_regularized.
        """
        d = Discretization(
            n=10,
            h=None,
            intervals_ben=(1.0e-5, 1.0e4, 100),
            intervals_del=(-1e8, -1.0e-5, 100)
        )

        y1 = fd.discretization.Discretization.get_counts_fixed_dominant(d.s, 0.5)
        y2 = fd.discretization.Discretization.get_counts_fixed_semidominant_regularized(d.s)

        diff = np.abs(y1 - y2) / (y2 + 1e-10)
        self.assertLess(diff.max(), 0.001)

    def test_fixed_counts_varying_h(self):
        """
        Test Discretization.get_counts_fixed_dominant with varying h.
        """
        d = Discretization(
            n=10,
            h=None,
            intervals_ben=(1.0e-5, 1.0e4, 100),
            intervals_del=(-1e8, -1.0e-5, 100),
        )

        rng = np.random.default_rng(0)
        hs = rng.uniform(0, 1, size=d.s.shape[0])

        y_var = fd.discretization.Discretization.get_counts_fixed_dominant(d.s, hs)

        # reference: two fixed extremes
        y_h0 = fd.discretization.Discretization.get_counts_fixed_dominant(d.s, 0.0)
        y_h1 = fd.discretization.Discretization.get_counts_fixed_dominant(d.s, 1.0)

        # bounds: per-s dominance interpolation
        assert np.all(y_var >= np.minimum(y_h0, y_h1))
        assert np.all(y_var <= np.maximum(y_h0, y_h1))

        x = np.arange(d.s.shape[0])

        plt.plot(x, y_h0, label="h=0", alpha=0.7)
        plt.plot(x, y_h1, label="h=1", alpha=0.7)
        plt.scatter(x, y_var, c=hs, cmap="viridis", s=10, label="varying h")

        plt.yscale("log")
        plt.colorbar(label="h")
        plt.ylim(1e-4, 1e5)
        plt.legend()
        plt.show()


    def test_plot_counts_fixed(self):
        """
        Plot allele counts for fixed h.
        """
        d = Discretization(
            n=10,
            h=None,
            intervals_ben=(1.0e-5, 1.0e4, 100),
            intervals_del=(-1e8, -1.0e-5, 100),
        )

        hs = np.round(np.linspace(0, 1, 6), 2)

        # cache all results at once
        counts = {
            h: fd.discretization.Discretization.get_counts_fixed_dominant(d.s, h)
            for h in hs
        }

        # make sure counts are larger for larger h on beneficial side and vice versa
        Y = np.stack([counts[h] for h in hs], axis=0)
        assert np.all(np.diff(Y[:, d.s > 0], axis=0) >= 0)
        assert np.all(np.diff(Y[:, d.s < 0], axis=0) <= 0)

        # plotting
        x = np.arange(len(d.s))
        for h, y in counts.items():
            plt.plot(x, y, label=f"h={h}")

        plt.legend()
        plt.yscale("log")
        plt.ylim(1e-4, 1e5)
        plt.show()
