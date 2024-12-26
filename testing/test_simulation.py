"""
Test the Simulation class.
"""
import numpy as np
from matplotlib import pyplot as plt
import pytest
import fastdfe as fd
from testing import TestCase


class SimulationTestCase(TestCase):
    """
    Test the Simulation class.
    """

    def test_get_neutral_sfs(self):
        """
        Test the get_neutral_sfs method.
        """
        theta = 1e-4
        n_sites = 1e8
        n = 10
        sfs = fd.Simulation.get_neutral_sfs(theta=theta, n_sites=n_sites, n=n)

        self.assertAlmostEqual(sfs.n_sites, n_sites)
        self.assertAlmostEqual(sfs.n, n)
        self.assertAlmostEqual(sfs.theta, theta)

        np.testing.assert_array_almost_equal(sfs.data[1:-1], [theta * n_sites / i for i in range(1, 10)])

        self.assertAlmostEqual(sfs.data[0], n_sites - sum(sfs.data[1:]))
        self.assertAlmostEqual(sfs.data[-1], 0)

    def test_get_neutral_sfs_with_demography(self):
        """
        Test the get_neutral_sfs method with demography.
        """
        theta = 1e-4
        n_sites = 1e8
        n = 10
        r = [0.98, 1.21, 0.87, 1.43, 0.92, 1.32, 0.99, 1.12, 0.95]
        sfs = fd.Simulation.get_neutral_sfs(theta=theta, n_sites=n_sites, n=n, r=r)

        self.assertAlmostEqual(sfs.n_sites, n_sites)
        self.assertAlmostEqual(sfs.n, n)

        # nuisance parameters affect mutation rate
        self.assertAlmostEqual(sfs.theta, theta, delta=1e-5)

        np.testing.assert_array_almost_equal(sfs.data[1:-1], [theta * n_sites / i * r[i - 1] for i in range(1, 10)])

        self.assertAlmostEqual(sfs.data[0], n_sites - sum(sfs.data[1:]))
        self.assertAlmostEqual(sfs.data[-1], 0)

    def test_get_neutral_sfs_wrong_length_r(self):
        """
        Test the get_neutral_sfs method with wrong length r.
        """
        theta = 1e-4
        n_sites = 1e8
        n = 10
        r = [1] * 10

        with self.assertRaises(ValueError) as e:
            fd.Simulation.get_neutral_sfs(theta=theta, n_sites=n_sites, n=n, r=r)

        print(e.exception)

    def test_recover_result_full_dfe_no_demography(self):
        """
        Test that the simulated result can be recovered by inference.
        """
        # very deleterious DFEs were difficult to recover
        sim = fd.Simulation(
            sfs_neut=fd.Simulation.get_neutral_sfs(n=20, n_sites=1e8, theta=1e-4),
            params=dict(S_d=-300, b=1, p_b=0.05, S_b=0.1),
            eps=0,
        )

        sfs_sel = sim.run()

        inf = fd.BaseInference(
            sfs_sel=sfs_sel,
            sfs_neut=sim.sfs_neut,
            discretization=sim.discretization,
            do_bootstrap=True,
            n_bootstraps=100,
            parallelize=True
        )

        inf.run()

        self.assertAlmostEqual(sim.n_sites, inf.sfs_sel.n_sites)
        self.assertAlmostEqual(sim.n_sites, inf.sfs_neut.n_sites)

        self.assertAlmostEqual(sim.theta, inf.sfs_neut.theta)

        for key in sim.params:
            self.assertAlmostEqual(inf.params_mle[key], sim.params[key], delta=np.abs(sim.params[key]) / 1000)

        self.assertAlmostEqual(inf.bootstraps.mean()['S_d'], sim.params['S_d'], delta=sim.params['S_d'] / -100)
        self.assertAlmostEqual(inf.bootstraps.mean()['b'], sim.params['b'], delta=sim.params['b'] / 50)
        self.assertAlmostEqual(inf.bootstraps.mean()['p_b'], sim.params['p_b'], delta=sim.params['p_b'] / 20)
        self.assertAlmostEqual(inf.bootstraps.mean()['S_b'], sim.params['S_b'], delta=sim.params['S_b'] / 1.5)

        self.assertAlmostEqual(inf.bootstraps.mean()['eps'], 0, delta=1e-3)

    def test_recover_result_full_dfe_with_demography(self):
        """
        Test that the simulated result can be recovered by inference.
        """
        sim = fd.Simulation(
            sfs_neut=fd.Simulation.get_neutral_sfs(
                n=20,
                n_sites=1e8,
                theta=1e-4,
                r=[0.98, 1.21, 0.87, 1.43, 0.92, 1.32, 0.99,
                   1.12, 0.95, 1.21, 0.87, 1.43, 0.92,
                   0.99, 1.12, 0.95, 1.21, 0.87, 1.43]
            ),
            params=dict(S_d=-300, b=1, p_b=0.05, S_b=0.1),
            eps=0
        )

        sfs_sel = sim.run()

        inf = fd.BaseInference(
            sfs_sel=sfs_sel,
            sfs_neut=sim.sfs_neut,
            discretization=sim.discretization,
            do_bootstrap=True,
            n_bootstraps=100,
            parallelize=True
        )

        inf.run()

        self.assertAlmostEqual(sim.n_sites, inf.sfs_sel.n_sites, delta=1e-6)
        self.assertAlmostEqual(sim.n_sites, inf.sfs_neut.n_sites)

        self.assertAlmostEqual(sim.theta, inf.sfs_neut.theta)

        for key in sim.params:
            self.assertAlmostEqual(inf.params_mle[key], sim.params[key], delta=np.abs(sim.params[key]) / 100)

        self.assertAlmostEqual(inf.bootstraps.mean()['S_d'], sim.params['S_d'], delta=sim.params['S_d'] / -100)
        self.assertAlmostEqual(inf.bootstraps.mean()['b'], sim.params['b'], delta=sim.params['b'] / 50)
        self.assertAlmostEqual(inf.bootstraps.mean()['p_b'], sim.params['p_b'], delta=sim.params['p_b'] / 20)
        self.assertAlmostEqual(inf.bootstraps.mean()['S_b'], sim.params['S_b'], delta=sim.params['S_b'] / 1.5)

        self.assertAlmostEqual(inf.bootstraps.mean()['eps'], 0, delta=1e-3)

    def test_recover_result_deleterious_dfe_no_demography(self):
        """
        Test that the simulated result can be recovered by inference.
        """
        sim = fd.Simulation(
            sfs_neut=fd.Simulation.get_neutral_sfs(n=20, n_sites=1e8, theta=1e-4),
            params=dict(S_d=-300, b=1, p_b=0, S_b=0.1),
            eps=0
        )

        sfs_sel = sim.run()

        inf = fd.BaseInference(
            sfs_sel=sfs_sel,
            sfs_neut=sim.sfs_neut,
            discretization=sim.discretization,
            fixed_params=dict(all=dict(p_b=0, S_b=0.1)),
            do_bootstrap=True,
            n_bootstraps=100,
            parallelize=True
        )

        inf.run()

        self.assertAlmostEqual(sim.n_sites, inf.sfs_sel.n_sites)
        self.assertAlmostEqual(sim.n_sites, inf.sfs_neut.n_sites)

        self.assertAlmostEqual(sim.theta, inf.sfs_neut.theta)

        for key in sim.params:
            self.assertAlmostEqual(inf.params_mle[key], sim.params[key], delta=np.abs(sim.params[key]) / 1000)

        self.assertAlmostEqual(inf.bootstraps.mean()['S_d'], sim.params['S_d'], delta=sim.params['S_d'] / -100)
        self.assertAlmostEqual(inf.bootstraps.mean()['b'], sim.params['b'], delta=sim.params['b'] / 100)
        self.assertAlmostEqual(inf.bootstraps.mean()['eps'], 0, delta=1e-3)

        self.assertAlmostEqual(inf.bootstraps.mean()['p_b'], 0)
        self.assertAlmostEqual(inf.bootstraps.mean()['S_b'], 0.1)

    def test_usage_example_simulation_class(self):
        """
        Test the usage example of the Simulation class.
        """
        import fastdfe as fd

        # create simulation object by specifying neutral SFS and DFE
        sim = fd.Simulation(
            sfs_neut=fd.Simulation.get_neutral_sfs(n=20, n_sites=1e8, theta=1e-4),
            params=dict(S_d=-300, b=0.3, p_b=0.1, S_b=0.1),
            model=fd.GammaExpParametrization()
        )

        # perform the simulation
        sfs_sel = sim.run()

        # plot SFS
        sfs_sel.plot()

    def test_parameter_out_of_bounds(self):
        """
        Test that the simulation of a neutral DFE with the Wright-Fisher model is correct.
        """
        with self.assertRaises(ValueError) as e:
            fd.Simulation(
                sfs_neut=fd.Simulation.get_neutral_sfs(n=10, n_sites=1e8, theta=1e-4),
                params=dict(S_d=0, b=1, p_b=0, S_b=1),
            )

        print(e.exception)

    def test_wright_fisher_simulation_sample_cdf(self):
        """
        Test that the sample_cdf method of the Wright-Fisher model is correct.
        """
        sim = fd.simulation.WrightFisherSimulation(
            sfs_neut=fd.Simulation.get_neutral_sfs(n=20, n_sites=1e8, theta=1e-4),
            params=dict(S_d=-300, b=10, p_b=0.1, S_b=50),
        )

        samples = sim._sample_cdf(n=1000)

        ax = plt.gca()
        ax.hist(samples, bins=100, density=True, alpha=0.5)
        x = np.linspace(-1000, 300, 1000)
        ax.plot(x, sim.model.get_pdf(**sim.params)(x), color='red')
        ax.set_xlim(-1000, 300)
        plt.show()

        self.assertAlmostEqual(samples.mean(), -300 * 0.9 + 50 * 0.1, delta=10)

    @pytest.mark.skip(reason="takes too long for reasonable values")
    def test_simulation_against_wright_fisher_neutral(self):
        """
        Test that the simulation of a neutral DFE with the Wright-Fisher model is correct.
        """
        sim = fd.simulation.WrightFisherSimulation(
            sfs_neut=fd.Simulation.get_neutral_sfs(n=10, n_sites=1e8, theta=1e-6),
            params=dict(S_d=-1e-100, b=1, p_b=0, S_b=1),
            eps=0,
            n_generations=1000,
            pop_size=100
        )

        sfs_sel = sim.run()

        sfs_sel.plot()

        self.assertAlmostEqual(sfs_sel.n_sites, sim.sfs_neut.n_sites)
        self.assertAlmostEqual(sfs_sel.n, sim.sfs_neut.n)
        self.assertAlmostEqual(sfs_sel.theta, sim.sfs_neut.theta)

        fd.Spectra(dict(neutral=sim.sfs_neut, selected=sfs_sel)).plot()

        np.testing.assert_array_almost_equal(sfs_sel.data, sim.sfs_neut.data)

    @pytest.mark.skip(reason="takes too long for reasonable values")
    def test_simulation_against_wright_fisher_deleterious(self):
        """
        Test that the simulation of a neutral DFE with the Wright-Fisher model is correct.
        """
        fd.logger.setLevel('DEBUG')

        sim = fd.simulation.WrightFisherSimulation(
            sfs_neut=fd.Simulation.get_neutral_sfs(n=10, n_sites=1e8, theta=1e-6),
            params=dict(S_d=-100, b=1, p_b=0, S_b=1),
            eps=0,
            n_generations=100,
            pop_size=100
        )

        sfs_sel = sim.run()

        sfs_sel.plot()

