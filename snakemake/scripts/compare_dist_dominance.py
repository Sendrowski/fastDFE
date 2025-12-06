"""
Compare distance including dominance with distance excluding dominance
"""
from math import pi
from typing import Callable

import numpy as np
import scipy.special as sp
from matplotlib import pyplot as plt

from fastdfe.discretization import Discretization


def integral(S, h, x):
    """
    Indefinite integral of the function
    """
    erf = sp.erfi if S > 0 else sp.erf

    return (
            np.sqrt(pi) * np.exp(np.float128(h ** 2 * S / (1 - 2 * h))) /
            (2 * np.sqrt(4 * h - 2) * np.sqrt(np.abs(S) / 2)) *
            erf(np.sqrt(np.abs(S) / 2) * (h * (2 * x - 1) - x) / np.sqrt(h - 0.5))
    )


def f(x, S, h):
    """
    Allele frequency distribution for scaled selection coefficient S and dominance h
    """
    return (
            np.exp(np.float128(2 * S * h * x + S * (1 - 2 * h) * x ** 2)) /
            (x * (1 - x)) *
            (integral(S, h, 1) - integral(S, h, x)) /
            (integral(S, h, 1) - integral(S, h, 0))
    )


comp = {}
for (x, S) in [(0.1, 1)]:
    comp[(x, S)] = (f(x=x, S=S, h=0.6), Discretization.get_counts_high_precision_regularized(x=x, S=S))


def plot_surface(ax: plt.Axes, func: Callable, title: str):
    """
    Plot the surface plot of the function

    :param ax: The axis object to plot on
    :param func: The function to plot
    :param title: Title for the subplot
    """
    # Parameters
    x_vals = np.linspace(0.01, 0.99, 100)  # allele frequencies
    S_vals = np.linspace(-100, 100, 100)  # selection coefficients

    # Compute the function values
    X, S = np.meshgrid(x_vals, S_vals)
    Z = np.array([func(x, s) for x, s in zip(np.ravel(X), np.ravel(S))])
    Z = Z.reshape(X.shape)

    # Surface plot
    surf = ax.plot_surface(X, S, Z, cmap='viridis', edgecolor='none')
    ax.set_xlabel('x')
    ax.set_ylabel('S')
    ax.set_zlabel('f')
    ax.set_title(title)

    # Color bar for surface plot
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)


# Create a figure with two subplots
fig = plt.figure(figsize=(14, 7))

# First subplot for f(x, S, h=0.6)
ax1 = fig.add_subplot(121, projection='3d')
plot_surface(ax1, lambda x, S: f(x, S, h=0.6), title="f(x, S, h=0.6)")

# Second subplot for discretization.H_regularized
ax2 = fig.add_subplot(122, projection='3d')
plot_surface(ax2, Discretization.get_counts_high_precision_regularized, title="discretization.H_regularized")

plt.tight_layout()
plt.show()

S = 20
q = np.linspace(0.01, 0.99, 100)
h_vals = np.array([0.1, 0.6, 1, 1.3])

for h in h_vals:
    plt.plot(q, f(q, S, h), label=f"h={h}")

plt.legend()
plt.xlabel("q")
plt.ylabel("f(q)")
plt.show()

pass
