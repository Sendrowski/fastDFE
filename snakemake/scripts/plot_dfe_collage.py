"""
Combine DFE plots.
"""
import re
import sys
from pathlib import Path

import numpy as np

# necessary to import fastdfe locally
sys.path.append('.')

import matplotlib.pyplot as plt
import fastdfe as fd

try:
    testing = False
    files = snakemake.input
    labels = snakemake.params.labels
    out = snakemake.output[0]
except NameError:
    testing = True
    files = [
        "results/slim/n_replicate=1/n_chunks=100/g=1e4/L=1e7/mu=1e-8/r=1e-7/N=1e3/s_b=1e-3/b=0.3/s_d=3e-1/p_b=0.00/n=20/constant/unfolded/summary.json",
        "results/slim/n_replicate=1/n_chunks=100/g=1e4/L=1e7/mu=1e-8/r=1e-7/N=1e3/s_b=1e-3/b=0.1/s_d=3e-2/p_b=0.00/n=20/constant/unfolded/summary.json",
        "results/slim/n_replicate=1/n_chunks=100/g=1e4/L=1e7/mu=1e-8/r=1e-7/N=1e3/s_b=1e-2/b=0.1/s_d=3e-1/p_b=0.01/n=20/constant/unfolded/summary.json",
        "results/slim/n_replicate=1/n_chunks=100/g=1e4/L=1e7/mu=1e-8/r=1e-7/N=1e3/s_b=1e-3/b=0.3/s_d=3e-2/p_b=0.05/n=20/constant/unfolded/summary.json"
    ]
    labels = [
        "Strongly deleterious",
        "Weakly deleterious",
        "Rare beneficial",
        "Frequent beneficial",
    ]
    out = "scratch/dfe_collage.png"

plt.rcParams['xtick.labelsize'] = 8


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


def format_params(params_slim: dict, params_fd: dict) -> tuple[dict, dict]:
    """
    Format parameters for display.
    """
    trailing = dict(
        p_b=2,
        b=2,
        S_b=1,
        S_d=0,
        N_e=0
    )

    for key in params_slim:
        params_slim[key] = f"{params_slim[key]:.{trailing[key]}f}"

        if key in params_fd:

            params_fd[key] = f"{params_fd[key]:.{trailing[key]}f}"

            while params_slim[key].endswith("0") and params_fd[key].endswith("0"):
                params_slim[key] = params_slim[key][:-1]
                params_fd[key] = params_fd[key][:-1]

            params_slim[key] = params_slim[key].rstrip(".")
            params_fd[key] = params_fd[key].rstrip(".")

    return (
        ','.join(
            [f"${k}$={' ' * (len(params_fd[k]) - len(v)) if k in params_fd else ''}{v}" for k, v in params_slim.items()]),
        ','.join([f"${k}$={' ' * (len(params_slim[k]) - len(v)) if k in params_slim else ''}{v}" for k, v in
                  params_fd.items()]),
    )


fig, ax = plt.subplots(2, 2, figsize=(7, 4), sharex=True, sharey=True)
ax = ax.flatten()

for i, f in enumerate(files):
    result = fd.InferenceResult.from_file(f)
    params = extract_params(f)

    Ne = result.spectra["neutral"].theta / (4 * params["mu"])

    dfe_slim = fd.DFE(dict(
        S_d=-4 * Ne * params["s_d"],
        b=params["b"],
        p_b=params["p_b"],
        S_b=4 * Ne * params["s_b"],
        N_e=Ne
    ))

    params_slim = dfe_slim.params.copy()
    if params_slim["p_b"] == 0:
        params_slim.pop("S_b")
    params_fd = {k: v for k, v in result.dfe.params.items() if k in params_slim}

    params_slim, params_fd = format_params(params_slim, params_fd)

    fd.DFE.plot_many(
        [dfe_slim, result.dfe],
        labels=[
            f"SLiM   ({params_slim})",
            f"fastDFE({params_fd})"
        ],
        intervals=[-np.inf, -100, -10, -1, 1, np.inf],
        ax=ax[i],
        show=False,
        title=labels[i],
    )
    leg = ax[i].legend(loc='upper left', prop={"family": "monospace", "size": 6.3})
    for h in leg.legend_handles:
        h.set_hatch(None)

    ax[i].set_ylim(0, 1)

    # remove hatch patterns from bars
    for container in ax[i].containers:
        if container is not None:
            for patch in container:
                if patch is not None and hasattr(patch, "set_hatch"):
                    patch.set_hatch("")

fig.tight_layout(pad=0.05)
ax[1].set_ylabel("")
ax[3].set_ylabel("")
ax[0].set_xlabel("")
ax[1].set_xlabel("")

fig.savefig(out, dpi=300)

if testing:
    plt.show()
