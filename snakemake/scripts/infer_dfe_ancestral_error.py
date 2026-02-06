import fastdfe as fd
import numpy as np


def misidentify(sfs, epsilon: float) -> fd.Spectrum:
    """
    Introduce ancestral misidentification at rate epsilon. Note that monomorphic counts won't be affected.

    :param sfs: Input spectrum
    :param epsilon: Misidentification rate (0 <= epsilon <= 1)
    :return: Spectrum with misidentification applied
    """
    data = sfs.data.copy()
    n = sfs.n

    if epsilon > 0:
        flipped = epsilon * data[1:n][::-1]
        retained = (1 - epsilon) * data[1:n]
        data[1:n] = retained + flipped
    else:
        # shifting mass produces rugged spectra
        mid = (sfs.n + 1) // 2
        data[:mid] += -epsilon * data[-mid:][::-1]
        data[-mid:] -= -epsilon * data[-mid:]

    return fd.Spectrum(data)

eps = 0.05

sim = fd.Simulation(
    sfs_neut=fd.Simulation.get_neutral_sfs(n=20, n_sites=1e8, theta=1e-4),
    params=dict(S_d=-300, b=0.3, p_b=0, S_b=1, h=0.5),
    model=fd.GammaExpParametrization(bounds=dict(eps=(0, 1)))
)

sim.run()

inf = fd.BaseInference(
    sfs_neut=misidentify(sim.sfs_neut, eps),
    sfs_sel=misidentify(sim.sfs_sel, eps),
    fixed_params=dict(all=dict(h=0.5)),
    discretization=sim.discretization,
    bounds=dict(eps=(0, 1)),
    model=sim.model
)

inf.run()

fd.DFE.plot_many(
    [sim.dfe, inf.get_dfe()],
    labels=['true DFE'] + ['inferred DFE']
)

pass