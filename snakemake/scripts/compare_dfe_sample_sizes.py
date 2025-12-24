import sys

# necessary to import fastdfe locally
sys.path.append('.')

import numpy as np
from matplotlib import pyplot as plt
import fastdfe as fd

try:
    testing = False
    n_dfes = 6
    n_runs = 10
    n_bootstraps = 100
    n_bootstrap_retries = 2
    n_fixed = 20
    intervals_del = (-1.0e+8, -1.0e-5, 1000)
    intervals_ben = (1.0e-5, 1.0e4, 1000)
    out = snakemake.output[0]
except NameError:
    # testing
    testing = True
    n_dfes = 2
    n_runs = 1
    n_bootstraps = 4
    n_bootstrap_retries = 1
    n_fixed = 20
    intervals_del = (-1.0e+8, -1.0e-5, 100)
    intervals_ben = (1.0e-5, 1.0e4, 100)
    out = "scratch/compare_dfe_accuracy.png"

fig, ax = plt.subplots(2, 2, figsize=(10.5, 6))

plt.rcParams['xtick.labelsize'] = 9
plt.rcParams["axes.prop_cycle"] = plt.cycler(color=plt.cm.viridis(np.linspace(0.15, 1, n_dfes + 1)))

params = [dict(S_d=-300, b=0.3, p_b=0, S_b=1), dict(S_d=-300, b=0.3, p_b=0.05, S_b=1)]


def sci(x):
    base, exp = f"{x:e}".split("e")
    return f"{int(np.round(float(base)))}e{int(exp)}"


for i, param in enumerate(params):

    dfes = {}
    spectra = {}

    for n in np.logspace(np.log10(6), np.log10(100), n_dfes).astype(int):
        sim = fd.Simulation(
            sfs_neut=fd.Simulation.get_neutral_sfs(n=n, n_sites=1e7, theta=1e-5),
            intervals_ben=intervals_ben,
            intervals_del=intervals_del,
            params=param,
            model=fd.GammaExpParametrization()
        )

        sim.run()

        s = sim.get_spectra()
        s.data *= 10000 / s.n_polymorphic

        inf = fd.BaseInference(
            sfs_neut=s['neutral'],
            sfs_sel=s['selected'],
            intervals_ben=intervals_ben,
            intervals_del=intervals_del,
            fixed_params=dict(all=dict(h=0.5, eps=0) | (dict(p_b=0, S_b=1) if param['p_b'] == 0 else {})),
            n_runs=n_runs,
            n_bootstraps=n_bootstraps,
            n_bootstrap_retries=n_bootstrap_retries
        )

        inf.run()

        spectra[n] = inf.get_spectra()[['neutral', 'selected']]
        dfes[n] = inf.get_dfe()

    param_str = ', '.join([f'{k}={v}' for k, v in param.items() if k in ['S_d', 'b', 'p_b', 'S_b']])
    fd.DFE.plot_many(
        [sim.dfe] + list(dfes.values()),
        labels=np.arange(len(dfes) + 1).astype(str),
        point_estimate='mean',
        # intervals=[-np.inf, -100, -10, -1, 1, np.inf],
        ax=ax[i, 0],
        title=f"SFS sample sizes $({param_str})$",
        show=False
    )

    labels = (["true DFE"] +
              [f"n={n},{'':<{3 - int(np.log10(n))}}#SNP={sci(spectra[n].all.n_polymorphic)}" for n in dfes.keys()])
    ax[i, 0].legend(ax[i, 0].get_legend_handles_labels()[0], labels, prop={"family": "monospace", "size": 8})

    dfes = {}
    spectra = {}

    for n_snps in np.logspace(3, 6, n_dfes) * 1.5:
        sim = fd.Simulation(
            sfs_neut=fd.Simulation.get_neutral_sfs(n=n_fixed, n_sites=1e7, theta=1e-5),
            intervals_ben=intervals_ben,
            intervals_del=intervals_del,
            params=param,
            model=fd.GammaExpParametrization()
        )

        sim.run()

        s = sim.get_spectra()
        s.data *= n_snps / s.n_polymorphic

        inf = fd.BaseInference(
            sfs_neut=s['neutral'],
            sfs_sel=s['selected'],
            intervals_ben=intervals_ben,
            intervals_del=intervals_del,
            fixed_params=dict(all=dict(h=0.5, eps=0) | (dict(p_b=0, S_b=1) if param['p_b'] == 0 else {})),
            n_runs=n_runs,
            n_bootstraps=n_bootstraps,
            n_bootstrap_retries=n_bootstrap_retries
        )

        inf.run()

        spectra[n_snps] = inf.get_spectra()[['neutral', 'selected']]
        dfes[n_snps] = inf.get_dfe()

    param_str = ', '.join([f'{k}={v}' for k, v in param.items() if k in ['S_d', 'b', 'p_b', 'S_b']])
    fd.DFE.plot_many(
        [sim.dfe] + list(dfes.values()),
        labels=np.arange(len(dfes) + 1).astype(str),
        point_estimate='mean',
        # intervals=[-np.inf, -100, -10, -1, 1, np.inf],
        ax=ax[i, 1],
        title=f"Number of SNPs $({param_str})$",
        show=False
    )

    labels = ["true DFE"] + [f"n={n_fixed}, #SNP={sci(spectra[theta].all.n_polymorphic)}" for theta in dfes.keys()]
    ax[i, 1].legend(ax[i, 1].get_legend_handles_labels()[0], labels, prop={"family": "monospace", "size": 8})

plt.tight_layout()
fig.savefig(out, dpi=200)

if testing:
    plt.show()

pass
