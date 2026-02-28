"""
Combine DFE plots.
"""
import re
import sys
from pathlib import Path

import numpy as np
from matplotlib.container import BarContainer

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
        "results/slim/n_replicate=1/n_chunks=100/g=1e4/L=1e7/mu=1e-8/r=1e-7/N=1e3/s_b=1e-3/b=0.3/s_d=3e-1/p_b=0.00/n=20/expansion_4/unfolded/summary.json",
        "results/slim/n_replicate=1/n_chunks=100/g=1e4/L=1e7/mu=1e-8/r=1e-7/N=1e3/s_b=1e-3/b=0.3/s_d=3e-1/p_b=0.00/n=20/reduction_4/unfolded/summary.json",
        "results/slim/n_replicate=1/n_chunks=100/g=1e4/L=1e7/mu=1e-8/r=1e-7/N=1e3/s_b=1e-3/b=0.3/s_d=3e-1/p_b=0.00/n=20/bottleneck_20/unfolded/summary.json",
        "results/slim/n_replicate=1/n_chunks=100/g=1e4/L=1e7/mu=1e-8/r=1e-7/N=1e3/s_b=1e-3/b=0.3/s_d=3e-1/p_b=0.00/n=20/substructure_0.0001/unfolded/summary.json",
        "results/slim/n_replicate=1/n_chunks=100/g=1e4/L=1e7/mu=1e-8/r=1e-7/N=1e3/s_b=1e-3/b=0.3/s_d=3e-1/p_b=0.00/n=20/dominance_0.3/unfolded/summary.semidominant.json"
    ]
    labels = [
        "constant",
        "expansion",
        "reduction",
        "bottleneck",
        "substructure",
        "recessiveness",
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

fig, ax = plt.subplots(3, 2, figsize=(7, 4.5), sharex=True, dpi=400)
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
    params_fd = {k: v for k, v in result.dfe.params.items() if k in params_slim}

    fd.DFE.plot_many(
        [dfe_slim, result.dfe],
        labels=[
            f"SLiM",
            f"fastDFE"
        ],
        intervals=[-np.inf, -100, -10, -1, 1, np.inf],
        ax=ax[i],
        show=False,
        title=labels[i],
    )

    colors = ["C0", "C1"]

    bar_containers = [c for c in ax[i].containers if isinstance(c, BarContainer)]

    for j, bc in enumerate(bar_containers):
        for patch in bc.patches:
            patch.set_facecolor(colors[j])
            patch.set_edgecolor(colors[j])

    for j, h in enumerate(ax[i].legend_.legend_handles):
        h.set_facecolor(colors[j])
        h.set_edgecolor(colors[j])
        h.set_hatch(None)

    ax[i].set_ylim(0, 1)

    # remove hatch patterns from bars
    for container in ax[i].containers:
        if container is not None:
            for patch in container:
                if patch is not None and hasattr(patch, "set_hatch"):
                    patch.set_hatch("")

fig.tight_layout(pad=0.05)
for x in ax:
    x.set_ylabel("")

for x in ax[:4]:
    x.set_xlabel("")

fig.savefig(out, dpi=300)

if testing:
    plt.show()
