"""
Plot DFE used for demographic distortions figure.
"""
import numpy as np
from matplotlib import pyplot as plt

import fastdfe as fd

fig, ax = plt.subplots(figsize=(4.5, 2.5), dpi=400)

# s_b=1e-3/b=0.1/s_d=3e-2/p_b=0.00/Ne=1e3
fd.DFE(dict(S_d=-3e-1 * 1e3, S_b=1e-3 * 1e3, p_b=0.00, b=0.3)).plot(
    ax=ax, show=False,
    intervals=[-np.inf, -100, -10, -1, 0, np.inf]
)

ax.set_xlabel('$S = 4 N_e s$', fontsize=8)
ax.set_title('')

plt.tight_layout()
plt.show()
