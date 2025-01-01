"""
Simulate an SFS given a DFE.
"""

__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"
__date__ = "2024-09-06"

import logging
from typing import Tuple, Literal, Sequence, Callable

import numpy as np
from scipy.stats import hypergeom
from tqdm import tqdm

from .base_inference import BaseInference
from .discretization import Discretization
from .parametrization import Parametrization
from .parametrization import _from_string
from .spectrum import Spectrum

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

        # check if parameters are within bounds
        self._check_bounds()

        #: Neutral SFS
        self.sfs_neut: Spectrum = sfs_neut

        #: SFS sample size
        self.n: int = self.sfs_neut.n

        #: Population mutation rate
        self.theta: float = self.sfs_neut.theta

        #: Number of sites
        self.n_sites: float = self.sfs_neut.n_sites

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

        #: Logger
        self._logger = logger.getChild('Simulation')

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

    def _check_bounds(self):
        """
        Check if the specified parameters are within the bounds of the DFE parametrization.
        """
        for k, v in self.params.items():
            if k in self.model.bounds and not self.model.bounds[k][0] <= v <= self.model.bounds[k][1]:
                raise ValueError(
                    f"Parameter {k}= {v} is not out of bounds {self.model.bounds[k]} "
                    f"for model {self.model.__class__.__name__}."
                )

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
        :param n_sites: Number of sites in the simulated SFS
        :param n: Number of frequency classes in the simulated SFS
        :param r: Nuisance parameters that account for demography. An array of length ``n-1`` whose elements are
            multiplied element-wise with the polymorphic counts of the Kingman SFS. By default, no demography effects
            are considered which is equivalent to ``r = [1] * (n-1)``. Note that non-default values of ``r`` will also
            affect estimates of the population mutation rate.
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

    def get_wright_fisher(
            self,
            pop_size: int,
            generations: int
    ) -> 'WrightFisherSimulation':
        """
        Get a Wright-Fisher simulation object.

        :param pop_size: Effective population size
        :param generations: Number of generations to simulate
        :return: Wright-Fisher simulation object
        """
        return WrightFisherSimulation(
            params=self.params,
            model=self.model,
            sfs_neut=self.sfs_neut,
            eps=self.eps,
            pop_size=pop_size,
            n_generations=generations,
        )


class WrightFisherSimulation:  # pragma: no cover
    """
    Simulate an SFS under selection given a DFE under the Wright-Fisher model.

    Example usage:

    ::

        import fastdfe as fd

        # create simulation object by specifying neutral SFS and DFE
        sim = fd.simulation.WrightFisherSimulation(
            sfs_neut=fd.Simulation.get_neutral_sfs(n=20, n_sites=1e7, theta=1e-4),
            params=dict(S_d=-300, b=0.3, p_b=0.1, S_b=0.1),
            model=fd.GammaExpParametrization(),
            pop_size=100,
            n_generations=500
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
            pop_size: int = 1000,
            n_generations: int = 100,
            n_sites: int = None,
            theta: float = None,
            parallelize: bool = True,
            n_threads: int = 100
    ):
        """
        Create a Wright-Fisher simulation object.

        :param params: Parameters for the DFE parametrization (see model). By default, the default parameters of
            ``model`` will be used.
        :param model: DFE parametrization model
        :param sfs_neut: Neutral SFS. This sfs is informative on the population sample size, population mutation rate,
            the number of sites, and demography. :func:`get_neutral_sfs` can be used to obtain a neutral SFS.
        :param eps: Ancestral misidentification error
        :param pop_size: Effective population size
        :param n_generations: Number of generations to simulate
        :param n_sites: Number of sites in the simulated SFS. If not provided, the number of sites in `sfs_neut` will
            be used.
        :param theta: Population mutation rate. If not provided, the population mutation rate in `sfs_neut` will be used.
        :param parallelize: Whether to parallelize computations
        :param n_threads: Number of threads to use for parallelization
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
        self.theta: float = self.sfs_neut.theta if theta is None else theta

        #: Number of sites
        self.n_sites: int = self.sfs_neut.n_sites if n_sites is None else n_sites

        #: Ancestral misidentification error
        self.eps: float = eps

        #: Effective population size
        self.pop_size: int = pop_size

        #: Number of generations to simulate
        self.n_generations: int = n_generations

        #: CDF of the DFE
        self.cdf: Callable[[np.ndarray], np.ndarray] = self.model.get_cdf(**self.params)

        #: Logger
        self._logger = logger.getChild('WrightFisherSimulation')

        #: Whether to parallelize computations
        self.parallelize: bool = parallelize

        #: Number of threads to use for parallelization
        self.n_threads: int = n_threads

        #: Tolerance for CDF bounds
        self.tol_bounds = 1e-6

        #: Tolerance for bisection
        self.tol_bisect = 1e-6

        #: Bounds for the CDF
        self.lower, self.upper = self._determine_cdf_bounds()

    def _determine_cdf_bounds(self) -> Tuple[float, float]:
        """
        Determine bounds for the CDF by identifying where the CDF approaches 0 and 1.

        :return: Lower and upper bounds for the CDF.
        """
        lower = -1e-4
        upper = 1e-4

        # Expand lower bound until CDF is close to 0
        while self.cdf(lower) > self.tol_bounds:
            lower *= 2

        # Expand upper bound until CDF is close to 1
        while self.cdf(upper) < 1 - self.tol_bounds:
            upper *= 2

        self._logger.debug(f"Bounded CDF between {lower} and {upper}.")

        return lower, upper

    def run(self) -> Spectrum:
        """
        Simulate an SFS using the Wright-Fisher model with per-individual, per-locus mutations and selection.

        :return: Simulated SFS.
        """
        rng = np.random.default_rng()

        # population mutation rate for all loci
        theta = int(self.sfs_neut.theta * self.n_sites / 2)

        # fixed total of mutations
        n_mut = int(self.n_generations * theta)

        self._logger.info(f'Pre-sampling selection coefficients for {n_mut} mutations.')

        # pre-sample selection coefficients for mutations
        selection_coefficients = self._sample_cdf(
            n=n_mut,
            desc="Number of bisections"
        ) / self.pop_size

        # log number of selection coefficients lower than -1
        if (n_low := (selection_coefficients < -1).sum()) > 0:
            self._logger.debug(
                f"Number of selection coefficients lower than -1: {n_low}."
            )

        # selection coefficients per individual and locus
        s = np.zeros((self.pop_size, n_mut))

        # initial allele frequencies per individual and locus
        freqs = np.zeros((self.pop_size, n_mut), dtype=np.int8)

        # pre-sample individuals for mutations across all generations
        individuals = rng.choice(self.pop_size, size=(self.n_generations, theta), replace=True)

        for g in tqdm(range(self.n_generations), desc="Simulating generations"):
            # indices for mutations
            i_muts = np.arange(g * theta, (g + 1) * theta)

            # introduce mutations under assumption of infinite sites
            freqs[individuals[g], i_muts] = 1
            s[individuals[g], i_muts] = selection_coefficients[i_muts]

            segregating = (freqs.sum(axis=0) > 0) & (freqs.sum(axis=0) < self.pop_size)
            n_segregating = segregating.sum()

            # selection
            w = 1 + s[:, segregating]
            probs = w / w.sum(axis=0)  # selection probabilities

            u = rng.random((self.pop_size, n_segregating))

            # Compute cumulative sums of probabilities along rows
            cumsum_probs = np.cumsum(probs, axis=0)

            idx = (cumsum_probs[:, None, :] < u).sum(axis=0)

            if g % 100 == 0:
                self._logger.debug(f"Number of mutations at generation {g}: {freqs.any(axis=0).sum()}")

            freqs[:, segregating] = freqs[:, segregating][idx, np.arange(n_segregating)]

        # vectorize the hypergeometric PMF calculation
        values = hypergeom.pmf(k=np.arange(self.n + 1)[:, None], M=self.pop_size, n=freqs.sum(axis=0), N=self.n)

        sfs = Spectrum(values.sum(axis=1))

        sfs.data[0] = self.n_sites - sfs.n_polymorphic
        sfs.data[-1] = 0

        return sfs

    def _sample_cdf(
            self,
            n: int,
            seed: int = None,
            tolerance: float = 1e-13,
            desc: str = "Bisection"
    ) -> np.ndarray:
        """
        Sample the cdf of the dfe with a fully vectorized bisection method.

        :param n: Number of samples to draw
        :param seed: Random seed
        :param tolerance: Tolerance for the bisection method
        :return: Array of values x such that cdf(x) = u
        """
        # generate uniform samples
        samples = np.random.default_rng(seed=seed).uniform(0, 1, n)

        # initialize bounds
        lower = np.full(n, self.lower, dtype=float)
        upper = np.full(n, self.upper, dtype=float)

        pbar = tqdm(desc=desc)

        while np.mean(upper - lower) > tolerance:
            # compute midpoint
            mid = (lower + upper) / 2.0

            # compute cdf at midpoint for all samples
            cdf_mid = self.cdf(mid)

            # update bounds vectorized
            lower_update = cdf_mid < samples

            lower = np.where(lower_update, mid, lower)
            upper = np.where(~lower_update, mid, upper)

            pbar.update(1)

        pbar.close()

        # return final approximated x values
        return (lower + upper) / 2.0
