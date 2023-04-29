from unittest import TestCase

import numpy as np
from matplotlib import pyplot as plt
from scipy.special import hyp1f1

from fastdfe import GammaExpParametrization, discretization
from fastdfe.discretization import Discretization
from fastdfe.discretization import H, H_regularized
from fastdfe.discretization import H_fixed, H_fixed_regularized


class DiscretizationTestCase(TestCase):
    n = 20

    show_plots = True

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

    def show_plot(self):
        if self.show_plots:
            plt.show()

    def test_dfe_to_sfs_linearization_midpoint_vs_quad(self):
        """
        Check for discrepancies between the linearized DFE to SFS
        transformation using midpoint integration vs scipy's quad.
        """
        opts = dict(
            n=self.n,
            intervals_ben=(1.0e-5, 100, 100),
            intervals_del=(-10000, -1.0e-5, 100)
        )

        d_midpoint = Discretization(**opts, integration_mode='midpoint')
        d_quad = Discretization(**opts, integration_mode='quad')

        P_quad = d_midpoint.dfe_to_sfs
        P_lin = d_quad.dfe_to_sfs

        # P_diff_rel = self.diff_rel(P_quad, P_lin)
        diff = self.diff_rel_max_abs(P_quad, P_lin)

        # this is rather large but is due to points near 0
        # for which we have rather small values (on the order of 1e-17)
        assert diff < 1

        model = GammaExpParametrization()
        params = GammaExpParametrization.x0

        sfs_quad = d_quad.model_selection_sfs(model, params)
        sfs_mid = d_midpoint.model_selection_sfs(model, params)

        diff = self.diff_rel_max_abs(sfs_quad, sfs_mid)

        assert diff < 5e-3

    def test_sfs_counts_linearized_vs_quad(self):
        """
        Compare precomputed linearization using scipy's quad with ad hoc integration.
        TODO do this for different DFEs.
        """
        model = GammaExpParametrization()
        params = GammaExpParametrization.x0

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
            ax.plot(np.abs(s), H(0.95, s), label='H(s)')
            ax.plot(np.abs(s), H_regularized(0.95, s), label='$H_{reg}(s)$')

            s = -s

            ax.set_xlabel('S' if s[0] > 0 else '-S')
            ax.set_ylabel('H')
            ax.set_xscale('log')
            ax.set_title('sojourn times')

        plt.legend()
        self.show_plot()

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
        c1 = d.get_allele_count(S[:, None] * I, k[None, :] * I)
        c2 = d.get_allele_count_large_negative_S(S[:, None] * I, k[None, :] * I)

        assert self.diff_rel_max_abs(c1, c2) < 1e-15

    def test_hyp1f1_mpmath_vs_scipy(self):
        """
        Compare mpmath and scipy's implementation of the confluent hypergeometric function.
        """
        s = np.linspace(-500, 500, 1000)
        k = np.arange(1, self.n)
        I = np.ones((s.shape[0], k.shape[0]))

        S = s[:, None] * I
        K = k[None, :] * I

        scipy = hyp1f1(K, self.n, S)
        mpmath = discretization.hyp1f1(K, self.n, S)

        assert self.diff_rel_max_abs(scipy, mpmath) < 1e-14

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
        c1 = d.get_allele_count(S[:, None] * I, k[None, :] * I)
        c2 = d.get_allele_count_large_positive_S(S[:, None] * I, k[None, :] * I)

        diff = self.diff_rel_max_abs(c1, c2)

        # still relatively large, but we can now evaluate large S
        # using mpmath
        assert diff < 1e-4

    def test_plot_allele_count_regularized(self):
        d = Discretization(
            n=self.n,
            intervals_ben=(1.0e-15, 100, 1000),
            intervals_del=(-1000000, -1.0e-15, 1000)
        )

        fig, (ax1, ax2) = plt.subplots(nrows=2)
        fig.tight_layout(pad=4)

        for k in range(1, self.n):
            ax2.plot(np.arange(d.s[d.s != 0].shape[0]),
                     d.get_allele_count(d.s[d.s != 0], k * np.ones_like(d.s[d.s != 0])), label=f"k = {k}")
            ax1.set_title('default')

        for k in range(1, self.n):
            ax1.plot(np.arange(d.s.shape[0]), d.get_allele_count_regularized(d.s, k * np.ones_like(d.s)),
                     label=f"k = {k}")
            ax1.set_title('regularized')

        plt.legend(prop=dict(size=5))
        self.show_plot()

    def test_plot_H_fixed_with_regularized_negative(self):

        d = Discretization(
            n=self.n,
            intervals_ben=(1.0e-15, 100, 0),
            intervals_del=(-1000000, -1.0e-15, 1000)
        )

        plt.plot(np.arange(len(d.s)), H_fixed_regularized(d.s), alpha=0.5, label='regularized')
        plt.plot(np.arange(len(d.s)), H_fixed(d.s), alpha=0.5, label='normal')

        plt.legend()
        self.show_plot()

    def test_plot_H_fixed_with_regularized_positive(self):

        d = Discretization(
            n=self.n,
            intervals_ben=(1.0e-15, 1, 1000),
            intervals_del=(-1000000, -1.0e-15, 0)
        )

        plt.plot(np.arange(len(d.s)), H_fixed_regularized(d.s), alpha=0.5,
                 label='positive regularized')
        plt.plot(np.arange(len(d.s)), H_fixed(d.s), alpha=0.5, label='positive normal')

        plt.legend()
        self.show_plot()
