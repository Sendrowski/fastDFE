"""
Simulate an SFS given a DFE.
"""

__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"
__date__ = "2024-09-06"

from typing import Tuple, Literal, Sequence

import numpy as np

from .parametrization import _from_string
from .base_inference import BaseInference
from .discretization import Discretization
from .parametrization import Parametrization
from .spectrum import Spectrum


class Simulation:
    """
    Simulate an SFS under selection given a DFE and a neutral SFS.

    Example usage:

    ::

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

    """

    def __init__(
            self,
            params: dict = None,
            model: Parametrization | str = 'GammaExpParametrization',
            sfs_neut: Spectrum = None,
            eps: float = 0,
            intervals_del: Tuple[float, float, int] = (-1.0e+8, -1.0e-5, 1000),
            intervals_ben: Tuple[float, float, int] = (1.0e-5, 1.0e4, 1000),
            integration_mode: Literal['midpoint', 'quad'] = 'midpoint',
            linearized: bool = True,
            discretization: Discretization = None,
            parallelize: bool = True
    ):
        """
        Create a simulation object.

        :param params: Parameters for the DFE parametrization (see model). By default, the default parameters of
            ``model`` will be used.
        :param model: DFE parametrization model
        :param sfs_neut: Neutral SFS. This sfs is informative on the population sample size, population mutation rate,
            the number of sites, and demography. :func:`get_neutral_sfs` can be used to obtain a neutral SFS.
        :param eps: Ancestral misidentification error
        :param intervals_del: ``(start, stop, n_interval)`` for deleterious population-scaled
            selection coefficients. The intervals will be log10-spaced.
        :param intervals_ben: Same as ``intervals_del`` but for positive selection coefficients
        :param integration_mode: Integration mode for the DFE, ``quad`` not recommended
        :param linearized: Whether to use the linearized DFE, ``False`` not recommended
        :param parallelize: Whether to parallelize computations
        """
        #: The DFE parametrization
        self.model: Parametrization = _from_string(model)

        #: Parameters for the DFE parametrization
        self.params: dict = self.model.x0 if params is None else params

        #: Neutral SFS
        self.sfs_neut: Spectrum = sfs_neut

        #: SFS sample size
        self.n: int = self.sfs_neut.n

        #: Population mutation rate
        self.theta: float = self.sfs_neut.theta

        #: Number of sites
        self.n_sites: int = self.sfs_neut.n_sites

        #: Ancestral misidentification error
        self.eps: float = eps

        if discretization is None:
            # create discretization instance
            #: Discretization instance
            self.discretization: Discretization = Discretization(
                n=self.n,
                intervals_del=intervals_del,
                intervals_ben=intervals_ben,
                integration_mode=integration_mode,
                linearized=linearized,
                parallelize=parallelize
            )

        else:
            # otherwise assign instance
            self.discretization: Discretization = discretization

    def run(self) -> Spectrum:
        """
        Simulate an SFS given a DFE.

        :return: Simulated SFS
        """
        # obtain modelled selected SFS
        counts_modelled = self.discretization.model_selection_sfs(self.model, self.params)

        # adjust for mutation rate and mutational target size
        counts_modelled *= self.theta * self.n_sites

        # add contribution of demography and adjust polarization error
        counts_modelled = BaseInference._add_demography(self.sfs_neut, counts_modelled, eps=self.eps)

        return Spectrum([self.n_sites - sum(counts_modelled)] + list(counts_modelled) + [0])

    @staticmethod
    def get_neutral_sfs(
            theta: float,
            n_sites: int,
            n: int,
            r: Sequence[float] = None
    ) -> Spectrum:
        """
        Obtain a standard neutral SFS for a given theta and number of sites.

        :param theta: Population mutation rate
        :param n_sites: Number of sites in the simulated SFS
        :param n: Number of frequency classes in the simulated SFS
        :param r: Nuisance parameters that account for demography. An array of length `n-1` whose elements are
            multiplied element-wise with the polymorphic counts of the neutral SFS. By default, no demography effects
            are considered which is equivalent to `r = [1] * (n-1)`. Note that non-default values of `r` will affect
            the estimated population mutation rate of the resulting SFS.
        :return: Neutral SFS
        """
        if r is None:
            r = np.ones(n + 1)
        else:
            r = list(r)

            if len(r) != n - 1:
                raise ValueError(f"The length of r must be n - 1 = {n - 1}; got {len(r)}.")

            r = np.array([1] + r + [1])

        sfs: Spectrum = Spectrum.standard_kingman(n=n) * theta * n_sites

        # add demography
        sfs.data *= r

        # add monomorphic counts
        sfs.data[0] = n_sites - sfs.n_sites

        return sfs
