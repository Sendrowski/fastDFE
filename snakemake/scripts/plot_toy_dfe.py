"""
Plot toy DFE.
"""
import numpy as np
from matplotlib import pyplot as plt

import fastdfe as fd

plt.figure(figsize=(2.6, 1.5), dpi=200)

ax = plt.gca()

fd.DFE(fd.GammaExpParametrization().x0).plot(
    ax=ax, show=False,
    intervals=[-np.inf, -100, -1, 0, np.inf]
)

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
