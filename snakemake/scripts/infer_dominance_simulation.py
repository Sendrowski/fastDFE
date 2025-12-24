import sys

import numpy as np
import pandas as pd

# necessary to import fastdfe locally
sys.path.append('.')

from matplotlib.cm import ScalarMappable
import matplotlib.gridspec as gridspec
from matplotlib import pyplot as plt
import fastdfe as fd

try:
    testing = False
    hs = np.round(np.linspace(0, 1, 11), 2)
    n = snakemake.params.n
    n_runs = 10
    n_bootstraps = 100
    n_bootstrap_retries = 2
    intervals_del = (-1.0e+8, -1.0e-5, 100)
    intervals_ben = (1.0e-5, 1.0e4, 100)
    intervals_h = (0, 1, 21)
    out = snakemake.output[0]
except NameError:
    # testing
    testing = True
    hs = np.round(np.linspace(0, 1, 11), 2)
    n = 20
    n_runs = 10
    n_bootstraps = 10
    n_bootstrap_retries = 2
    intervals_del = (-1.0e+8, -1.0e-5, 100)
    intervals_ben = (1.0e-5, 1.0e4, 100)
    intervals_h = (0, 1, 21)
    out = "compare_dfe_accuracy.png"

d = fd.discretization.Discretization(
    n=n,
    h=None,
    intervals_del=intervals_del,
    intervals_ben=intervals_ben,
    intervals_h=intervals_h
)

sims, infs = {}, {}
for h in hs:
    sims[h] = fd.Simulation(
        sfs_neut=fd.Simulation.get_neutral_sfs(n=n, n_sites=1e8, theta=1e-4),
        params=dict(S_d=-300, b=0.3, p_b=0, S_b=1, h=h),
        model=fd.GammaExpParametrization(),
        intervals_del=intervals_del,
        intervals_ben=intervals_ben,
        discretization=d
    )
    sims[h].run()

    infs[h] = fd.BaseInference(
        sfs_neut=sims[h].sfs_neut / 2,
        sfs_sel=sims[h].sfs_sel,
        intervals_h=intervals_h,
        intervals_del=intervals_del,
        intervals_ben=intervals_ben,
        discretization=d,
        fixed_params=dict(all=dict(eps=0, p_b=0, S_b=1)),
        n_runs=n_runs,
        n_bootstraps=n_bootstraps,
        n_bootstrap_retries=n_bootstrap_retries,
    )

    infs[h].run()

fig = plt.figure(figsize=(9, 5))

gs = gridspec.GridSpec(2, 3, width_ratios=[1, 1, 0.05])

# generate the 2×2 axes grid in one go
ax = np.array([[fig.add_subplot(gs[i, j]) for j in range(2)] for i in range(2)])

# colorbar axis spans all rows in last column
cax = fig.add_subplot(gs[:, 2])

plt.rcParams['xtick.labelsize'] = 9
plt.rcParams["legend.fontsize"] = 9

# global color map for all h
cmap = plt.cm.viridis
norm = plt.Normalize(min(hs), max(hs))
colors = {h: cmap(norm(h)) for h in hs}

fd.Spectra.from_spectra(
    {f"h={h}": infs[h].sfs_sel for h in hs}
).plot(ax=ax[0, 0], show=False)
ax[0, 0].set_title(
    f'Selected SFS $({", ".join([f"{k}={v}" for k, v in sims[hs[0]].params.items() if k in ["S_d", "b", "p_b"]])})$'
)
ax[0, 0].legend_.remove()

bars = ax[0, 0].patches
offset = 0

# each h
for h in hs:
    for b in bars[offset: offset + n + 1]:
        b.set_facecolor(colors[h])
        b.set_alpha(0.8)
        b.set_hatch("")
    offset += n + 1

fd.DFE.plot_many(
    [sims[hs[0]].dfe] + [inf.get_dfe() for inf in infs.values()],
    labels=['true DFE'] + ['_nolegend_' for _ in hs],
    intervals=[-np.inf, -100, -10, -1, 0],
    ax=ax[0, 1],
    show=False,
    point_estimate='mean',
    title='Inferred DFE under varying $h$'
)
leg = ax[0, 1].legend()

bars = ax[0, 1].patches
n_bins = 4
offset = 0

legend_handle = leg.get_patches()[0]
legend_handle.set_facecolor("grey")
legend_handle.set_alpha(0.9)

for b in bars[offset: offset + n_bins]:
    b.set_facecolor("grey")
    b.set_alpha(0.9)
    b.set_hatch("")
offset += n_bins

# inferred DFEs = one block per h
for h in hs:
    for b in bars[offset: offset + n_bins]:
        b.set_facecolor(colors[h])
        b.set_alpha(0.9)
        b.set_hatch("")
    offset += n_bins

data = [infs[h].bootstraps["h"].values for h in hs]

vp = ax[1, 0].violinplot(
    data,
    positions=hs,
    showmeans=True,
    showextrema=False,
    widths=0.08
)

# recolor each body
for h, body in zip(hs, vp['bodies']):
    body.set_facecolor(colors[h])
    body.set_edgecolor(colors[h])
    body.set_alpha(0.8)

# recolor the means
if 'cmeans' in vp:
    vp['cmeans'].set_color('black')

ax[1, 0].set_xlabel("true $h$")
ax[1, 0].set_ylabel("inferred $h$")
ax[1, 0].set_title("Inferred vs true $h$")

vals = []
for h in hs:
    df = infs[h].bootstraps.copy()
    df["S_d"] = df["S_d"].abs()
    df["true_h"] = h
    vals.append(df)

df_all = pd.concat(vals, ignore_index=True)

# Normalize true_h into [0,1] for colormap
norm = plt.Normalize(min(hs), max(hs))
cmap = plt.cm.viridis

sc = ax[1, 1].scatter(
    df_all["S_d"],
    df_all["h"],
    s=12,
    alpha=0.5,
    c=df_all["true_h"],
    cmap=cmap,
    norm=norm,
)

ax[1, 1].set_xscale("log")
ax[1, 1].set_xlabel("$S_d$")
ax[1, 1].set_ylabel("$h$")
ax[1, 1].set_title("Bootstrap distribution of $h$ vs $S_d$")

# add colorbar
sm = ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])

cbar = fig.colorbar(sm, cax=cax)
cbar.ax.set_ylabel("h", rotation=0, labelpad=0, va='center', fontsize=13)

plt.tight_layout(pad=0.8)

fig.savefig(out, dpi=200)

if testing:
    plt.show()
