"""
Plot DFE used for demographic distortions figure.
"""
import numpy as np
from matplotlib import pyplot as plt

import fastdfe as fd

fig, ax = plt.subplots(figsize=(3.5, 1.7), dpi=400)

# s_b=1e-3/b=0.3/s_d=3e-1/p_b=0.00/Ne=1e3
Ne = 1e3
fd.DFE(dict(S_d=-3e-1 * 4 * Ne, S_b=1e-3 * 4 * Ne, p_b=0.00, b=0.3)).plot(
    ax=ax, show=False,
    intervals=[-np.inf, -100, -1, 0, np.inf]
)

ax.set_xlabel('$S = 4 N_e s$', fontsize=8)
ax.set_title('')
ax.set_ylabel('')

plt.tight_layout()
plt.show()
