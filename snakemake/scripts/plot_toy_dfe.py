"""
Plot toy DFE.
"""
import numpy as np
from matplotlib import pyplot as plt

import fastdfe as fd

plt.figure(figsize=(2.6, 1.5), dpi=200)

ax = plt.gca()

fd.DFE(dict(S_d=-1000, b=0.2, p_b=0.1, S_b=1)).plot(
    ax=ax, show=False,
    intervals=[-np.inf, -100, -1, 0, np.inf]
)

bars = ax.patches  # bars in order of intervals

# muted palette
muted_red = "#c65f5f"
muted_green = "#6fa38f"

# first 3 bars red, last bar green (leave others default if any)
for i, b in enumerate(bars):
    if i < 3:
        b.set_facecolor(muted_red)
    elif i == len(bars) - 1:
        b.set_facecolor(muted_green)

# aesthetics
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_linewidth(0.8)
ax.spines["bottom"].set_linewidth(0.8)


ax.tick_params(axis="both", labelsize=7, width=0.8, length=3)
ax.set_yticks([0, 0.5, 1.0])
ax.set_yticklabels(["0", "0.5", "1"])
ax.set_axisbelow(True)
ax.grid(axis="y", alpha=0.2, linewidth=0.6)
ax.set_ylabel('')
ax.set_xlabel('$S = 4 N_e s$', fontsize=8)
ax.set_title('')

plt.tight_layout()
plt.show()
