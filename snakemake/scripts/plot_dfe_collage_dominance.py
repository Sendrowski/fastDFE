"""
DFEs simulated under partial recessivity beside the DFEs inferred assuming semidominance
and inferred with the simulated dominance coefficient, one row per simulated DFE.
"""
import re
import sys

import numpy as np
from matplotlib.container import BarContainer

# necessary to import fastdfe locally
sys.path.append('.')

import matplotlib.pyplot as plt
import fastdfe as fd


try:
    testing = False
    files = snakemake.input.inferred
    files_semi = snakemake.input.semidominant
    h = snakemake.params.h
    out = snakemake.output[0]
except NameError:
    testing = True
    base = ("results/slim/n_replicate=1/n_chunks=100/g=1e4/L=1e7/mu=1e-8/r=1e-7/N=1e3/"
            "{}/n=20/dominance_0.2/unfolded/")
    dfe_params = [
        "s_b=1e-3/b=0.3/s_d=3e-1/p_b=0.00",
        "s_b=1e-3/b=0.1/s_d=3e-2/p_b=0.00",
        "s_b=1e-2/b=0.1/s_d=3e-1/p_b=0.01",
        "s_b=1e-3/b=0.3/s_d=3e-2/p_b=0.05",
    ]
    files = [base.format(p) + "summary.json" for p in dfe_params]
    files_semi = [base.format(p) + "summary.semidominant.json" for p in dfe_params]
    h = 0.2
    out = "scratch/dfe_collage_dominance.pdf"

plt.rcParams['xtick.labelsize'] = 8

intervals = [-np.inf, -100, -10, -1, 1, np.inf]
columns = [rf"simulated ($h={h}$)", r"inferred ($h=0.5$)", rf"inferred ($h={h}$)"]


def extract_params(path: str) -> dict:
    """
    Extract simulation parameters from path string.
    """
    params = {}
    for key in ["s_d", "s_b", "b", "p_b", "N", "mu"]:
        m = re.search(rf"/{key}=([0-9.eE+-]+)", path)
        if m:
            params[key] = float(m.group(1))
    return params


def format_title(params: dict) -> str:
    """
    Format DFE parameters as a panel title.
    """
    digits = dict(S_d=0, b=2, p_b=2, S_b=1, N_e=0)
    return ", ".join(f"${k}$={params[k]:.{d}f}" for k, d in digits.items() if k in params)


fig, ax = plt.subplots(len(files), 3, figsize=(10, 1.6 * len(files)), sharex=True, sharey=True)

for i, (f, f_semi) in enumerate(zip(files, files_semi)):
    result = fd.InferenceResult.from_file(f)
    result_semi = fd.InferenceResult.from_file(f_semi)
    params = extract_params(f)
    Ne = result.spectra["neutral"].theta / (4 * params["mu"])

    dfe_slim = fd.DFE(dict(
        S_d=-4 * Ne * params["s_d"],
        b=params["b"],
        p_b=params["p_b"],
        S_b=4 * Ne * params["s_b"],
        N_e=Ne
    ))

    for j, dfe in enumerate([dfe_slim, result_semi.dfe, result.dfe]):
        fd.DFE.plot_many(
            [dfe],
            labels=[""],
            intervals=intervals,
            ax=ax[i, j],
            show=False,
            title=format_title({k: v for k, v in dfe.params.items() if k in ("S_d", "b", "p_b", "S_b", "N_e")}),
        )
        ax[i, j].set_title(ax[i, j].get_title(), fontsize=8.8, pad=5)
        if ax[i, j].legend_ is not None:
            ax[i, j].legend_.remove()

        for container in ax[i, j].containers:
            if isinstance(container, BarContainer):
                for patch in container.patches:
                    patch.set_facecolor("C0")
                    patch.set_edgecolor("C0")
                    patch.set_hatch("")

        if j > 0:
            ax[i, j].set_ylabel("")
        if i < len(files) - 1:
            ax[i, j].set_xlabel("")

for j, title in enumerate(columns):
    ax[0, j].annotate(title, xy=(0.5, 1.35), xycoords="axes fraction", ha="center", va="bottom", fontsize=15.6)

fig.tight_layout(h_pad=0.78, w_pad=0.6)

fig.savefig(out)

if testing:
    plt.show()
