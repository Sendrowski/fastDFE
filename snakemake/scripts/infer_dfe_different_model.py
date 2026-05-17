import fastdfe as fd
import numpy as np

n_bootstraps = 10

dfes = {}
for S_d in -np.logspace(4, 1, 5):
    sim = fd.Simulation(
        sfs_neut=fd.Simulation.get_neutral_sfs(n=20, n_sites=1e8, theta=1e-4),
        params=dict(S_d=S_d, b=0.3, p_b=0.1, S_b=1, h=0.5),
        model=fd.GammaExpParametrization()
    )

    sim.run()

    inf = fd.BaseInference(
        sfs_neut=sim.sfs_neut,
        sfs_sel=sim.sfs_sel,
        model=fd.DiscreteFractionalParametrization(np.array([-100000, -1000, -100, -10, -1, 0, 1000])),
        fixed_params=dict(all=dict(h=0.5)),
        discretization=sim.discretization,
        bounds=dict(eps=(0, 1)),
        n_bootstraps=n_bootstraps
    )

    inf.run()

    dfes[f"true.S_d={round(S_d)}"] = sim.dfe
    dfes[f"inferred.S_d={round(S_d)}"] = inf.get_dfe()

fd.DFE.plot_many(
    list(dfes.values()),
    labels=list(dfes.keys()),
    intervals=[-np.inf, -10, -1, 0, 1, np.inf]
)

pass