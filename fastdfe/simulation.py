"""
Simulate an SFS given a DFE.
"""

__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"
__date__ = "2024-09-06"

import logging
from typing import Tuple, Literal, Sequence, Callable, Optional

import numpy as np

from .base_inference import BaseInference
from .discretization import Discretization
from .parametrization import Parametrization, DFE
from .parametrization import _from_string
from sfsutils.spectrum import Spectrum, Spectra

logger = logging.getLogger('fastdfe')


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
        sim.run()

        # plot spectra
        sim.get_spectra().plot()

    """

    def __init__(
            self,
            sfs_neut: Spectrum,
            params: dict = None,
            model: Parametrization | str = 'GammaExpParametrization',
            intervals_del: Tuple[float, float, int] = (-1.0e+8, -1.0e-5, 1000),
            intervals_ben: Tuple[float, float, int] = (1.0e-5, 1.0e4, 1000),
            h_callback: Callable[[float, np.ndarray], np.ndarray] = None,
            integration_mode: Literal['midpoint', 'quad'] = 'midpoint',
            linearized: bool = True,
            n_sites_div: float = None,
            discretization: Discretization = None,
            parallelize: bool = True
    ):
        """
        Create a simulation object.

        :param sfs_neut: Neutral SFS. This sfs is informative on the population sample size, population mutation rate,
            the number of sites, and demography.
            :meth:`Simulation.get_neutral_sfs() <fastdfe.simulation.Simulation.get_neutral_sfs>` can be used to
            obtain a neutral SFS.
        :param params: Parameters of the DFE parametrization (see model) in addition to ``eps`` and dominance ``h``.
            By default, `eps=0`, `h=0.5`, the default parameters of ``model`` will be used. ``eps`` is the
            ancestral-misidentification parameter of the inference model, which enters through the nuisance
            parameters of the given neutral SFS. The simulated selected SFS thus differs from the one simulated
            for ``eps=0`` with :meth:`Spectrum.misidentify() <sfsutils.spectrum.Spectrum.misidentify>` applied.
        :param model: DFE parametrization model
        :param intervals_del: ``(start, stop, n_interval)`` for deleterious population-scaled
            selection coefficients. The intervals will be log10-spaced. Decreasing the number of intervals to ``100``
            provides nearly identical results while increasing speed, especially when precomputing across dominance
            coefficients.
        :param intervals_ben: Same as ``intervals_del`` but for positive selection coefficients. Decreasing the number
            of intervals to ``100`` provides nearly identical results while increasing speed, especially when
            precomputing across dominance coefficients.
        :param h_callback: A function mapping the scalar parameter ``h`` and the array of selection
            coefficients ``S`` to dominance coefficients of the same shape, allowing models where ``h``
            depends on ``S``. The default is ``lambda h, S: np.full_like(S, h)``, keeping ``h`` constant.
            The parameter ``h`` is taken from ``params``. The fitness of heterozygotes and mutation homozygotes
            is defined as ``1 + 2hs`` and ``1 + 2s``, respectively.
        :param integration_mode: Integration mode when computing expected SFS under semidominance.
            ``quad`` is not recommended.
        :param linearized: Whether to discretize and cache the linearized integral mapping DFE to SFS or use
            :func:`scipy.integrate.quad` in each call. ``False`` is slow, intended for validation only and
            deprecated.
        :param n_sites_div: Number of mutational target sites over which divergence is counted. When
            given, the simulated SFS carry an expected fixed-derived (divergence) class -- the
            expected number of substitutions ``theta * n_sites_div * flux`` (``flux = 1`` for neutral,
            the selected-relative substitution rate otherwise; cf.
            :meth:`~fastdfe.discretization.Discretization.get_divergence_flux`). When ``None`` (the
            default) the fixed-derived class is zero. Note that this is the *true*
            (population-level) divergence; it does not model finite-sample misattributed polymorphism.
        :param discretization: Precomputed :class:`~fastdfe.discretization.Discretization` instance. When given,
            ``intervals_del``, ``intervals_ben``, ``h_callback``, ``integration_mode``, ``linearized`` and
            ``parallelize`` are ignored.
        :param parallelize: Whether to parallelize computations
        """
        #: The DFE parametrization
        self.model: Parametrization = _from_string(model)

        #: Parameters for the DFE parametrization
        self.params: dict = dict(self.model.x0 if params is None else params)

        if 'eps' not in self.params:
            self.params['eps'] = 0

        if 'h' not in self.params:
            self.params['h'] = 0.5

        # check if parameters are within bounds
        self._check_bounds()

        #: Neutral SFS
        self.sfs_neut: Spectrum = sfs_neut

        #: Modelled SFS
        self.sfs_sel: Spectrum = None

        #: DFE instance
        self.dfe: DFE = DFE(model=self.model, params=self.params)

        #: SFS sample size
        self.n: int = self.sfs_neut.n

        #: Population mutation rate
        self.theta: float = self.sfs_neut.theta

        #: Number of sites
        self.n_sites: float = self.sfs_neut.n_sites

        #: Number of mutational target sites for divergence (``None`` -> no divergence class)
        self.n_sites_div: Optional[float] = n_sites_div

        if discretization is None:
            # create discretization instance
            #: Discretization instance
            self.discretization: Discretization = Discretization(
                n=self.n,
                h=self.params['h'],
                intervals_del=intervals_del,
                intervals_ben=intervals_ben,
                h_callback=h_callback,
                integration_mode=integration_mode,
                linearized=linearized,
                parallelize=parallelize
            )

        else:
            # otherwise assign instance
            self.discretization: Discretization = discretization

        #: Logger
        self._logger = logger.getChild('Simulation')

    def run(self) -> Spectrum:
        """
        Simulate an SFS given a DFE.

        :return: Simulated SFS
        """
        self._logger.info("Simulating SFS under selection using parameters: " + ', '.join(
            [f"{k}={v:.4g}" for k, v in self.params.items()]
        ))

        # obtain modelled selected SFS
        counts_modelled = self.discretization.model_selection_sfs(self.model, self.params)

        # adjust for mutation rate and mutational target size
        counts_modelled *= self.theta * self.n_sites

        # add contribution of demography and adjust polarization error
        counts_modelled = BaseInference._add_demography(self.sfs_neut, counts_modelled, eps=self.params['eps'])

        # a folded neutral SFS yields a folded selected SFS
        if BaseInference._is_folded(self.sfs_neut):
            counts_modelled = Spectrum.from_polymorphic(counts_modelled).fold().polymorphic

        # expected fixed-derived (divergence) class: the true number of substitutions,
        # theta * L_div * flux (flux = selected substitutions per neutral substitution)
        n_div = 0.0
        if self.n_sites_div is not None:
            flux = self.discretization.get_divergence_flux(self.model, self.params)
            n_div = self.theta * self.n_sites_div * flux

        n_monomorphic = max(self.n_sites - sum(counts_modelled) - n_div, 0)

        self.sfs_sel = Spectrum([n_monomorphic] + list(counts_modelled) + [n_div])

        return self.sfs_sel

    def _check_bounds(self):
        """
        Check if the specified parameters are within the bounds of the DFE parametrization.
        """
        for k, v in self.params.items():
            if k in self.model.bounds and not self.model.bounds[k][0] <= v <= self.model.bounds[k][1]:
                raise ValueError(
                    f"Parameter {k}={v} is out of bounds {self.model.bounds[k]} "
                    f"for model {self.model.__class__.__name__}."
                )

        if not 0 <= self.params['eps'] <= 1:
            raise ValueError(f"Ancestral misidentification parameter eps={self.params['eps']} is out of bounds [0, 1].")

        if not 0 <= self.params['h'] <= 1:
            raise ValueError(f"Dominance parameter h={self.params['h']} is out of bounds [0, 1].")

    def get_spectra(self) -> Spectra:
        """
        Get the spectra used in the simulation.

        :return: Spectra with the neutral and selected SFS. When ``n_sites_div`` was given, both
            carry their expected divergence (fixed-derived) class, which is part of the total number of sites.
        """
        sfs_neut = self.sfs_neut
        if self.n_sites_div is not None:
            # neutral divergence: theta * L_div (the neutral substitution flux is 1)
            data = list(self.sfs_neut.data)
            data[-1] = self.theta * self.n_sites_div
            data[0] = max(self.n_sites - sum(data[1:]), 0)
            sfs_neut = Spectrum(data)

        return Spectra.from_spectra({
            "neutral": sfs_neut,
            "selected": self.sfs_sel
        })

    def get_alpha(self) -> float:
        """
        Get the proportion of adaptive substitutions (alpha) under the specified DFE and parameters.

        :return: Proportion of adaptive substitutions
        """
        return self.discretization.get_alpha(self.model, self.params)

    def get_omega(self) -> float:
        """
        Get omega, the rate of non-synonymous over synonymous substitutions (dN/dS), under the
        specified DFE and parameters.

        :return: Rate of non-synonymous over synonymous substitutions (dN/dS)
        """
        return self.discretization.get_omega(self.model, self.params)

    def get_omega_a(self) -> float:
        """
        Get omega_a, the rate of adaptive non-synonymous over synonymous substitutions
        (adaptive dN/dS), under the specified DFE and parameters.

        :return: Rate of adaptive non-synonymous over synonymous substitutions (adaptive dN/dS)
        """
        return self.discretization.get_omega_a(self.model, self.params)

    @staticmethod
    def get_neutral_sfs(
            theta: float,
            n_sites: float,
            n: int,
            r: Sequence[float] = None
    ) -> Spectrum:
        """
        Obtain a standard neutral SFS for a given theta and number of sites.

        :param theta: Population mutation rate
        :param n_sites: Number of total sites
        :param n: Sample size, giving ``n + 1`` frequency classes
        :param r: Nuisance parameters that account for demography. An array of length ``n-1`` whose elements are
            multiplied element-wise with the polymorphic counts of the Kingman SFS. By default, no demography effects
            are considered which is equivalent to ``r = [1] * (n-1)``. Note that non-default values of ``r`` will also
            affect estimates of the population mutation rate.
        :return: Neutral SFS
        """
        return Spectrum.get_neutral(
            theta=theta,
            n_sites=n_sites,
            n=n,
            r=r
        )
