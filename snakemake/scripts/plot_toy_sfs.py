"""
Plot toy DFE.
"""
import numpy as np
from matplotlib import pyplot as plt

import fastdfe as fd

plt.figure(figsize=(2.6, 1.5), dpi=200)

ax = plt.gca()

s = fd.Spectra(dict(
    neutral=fd.Spectrum.standard_kingman(7) * 0.8,
    selected=fd.Spectrum.standard_kingman(7))
)

s.plot(ax=ax, show=False)

# aesthetics
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_linewidth(0.8)
ax.spines["bottom"].set_linewidth(0.8)


ax.tick_params(axis="both", labelsize=7, width=0.8, length=3)
ax.set_yticks([])
ax.set_yticklabels([])
ax.set_axisbelow(True)
ax.grid(axis="y", alpha=0.2, linewidth=0.6)
ax.set_ylabel('')
ax.set_xlabel('allele count', fontsize=8)
ax.set_title('')

leg = ax.legend(fontsize=8, handlelength=1.2, handletextpad=0.4,
                borderpad=0.3, labelspacing=0.3)

plt.tight_layout()
plt.show()
