"""
Reference values for ``Discretization._get_counts_dominant``, the sojourn times under general dominance, obtained
by nested adaptive quadrature. Running this script prints the values stored in
``DiscretizationTestCase.test_dominance_kernel_against_adaptive_quadrature``.

For a population-scaled selection coefficient ``S``, dominance coefficient ``h`` and
``E(q) = -2 * S * h * q - S * (1 - 2 * h) * q ** 2``, the expected count of mutations at allele count ``k`` in a
sample of size ``n`` is the integral over the frequency ``x`` of ``C(n, k) * x ** (k - 1) * (1 - x) ** (n - k - 1)``
times ``exp(-E(x)) * int_x^1 exp(E(q)) dq / int_0^1 exp(E(q)) dq``.
"""
from math import comb

import numpy as np
from scipy.integrate import quad

#: Sample size
n = 20

#: Points ``(h, S, k)`` at which the kernel is evaluated
points = [
    (0.0, -300.0, 1), (0.0, -300.0, 19), (0.0, -30.0, 10), (0.0, 1000.0, 10),
    (0.1, -300.0, 1), (0.1, -300.0, 19), (0.1, 10.0, 5),
    (0.9, -30.0, 10), (0.9, 1000.0, 19),
    (1.0, -300.0, 1), (1.0, 100.0, 15), (1.0, 1000.0, 19),
]


def exponent(q: float, S: float, h: float) -> float:
    """
    Exponent ``E(q)`` of the integrand.

    :param q: Allele frequency
    :param S: Population-scaled selection coefficient
    :param h: Dominance coefficient
    :return: The exponent
    """
    return -2 * S * h * q - S * (1 - 2 * h) * q ** 2


def get_breakpoints(a: float, b: float, S: float, h: float) -> list | None:
    """
    Points near the ends of an interval at the scales over which the integrand varies.

    :param a: Lower end
    :param b: Upper end
    :param S: Population-scaled selection coefficient
    :param h: Dominance coefficient
    :return: Sorted points within the interval, ``None`` if there are none
    """
    widths = [1 / (abs(S) * max(h, 1e-9) + 1), 1 / (np.sqrt(abs(S)) + 1), 1 / (abs(S) + 1)]
    points = {end + sign * m * w for end in (a, b) for sign in (1, -1) for w in widths for m in (1, 5, 25)}

    return sorted(p for p in points if a < p < b)[:40] or None


def get_count(h: float, S: float, k: int) -> float:
    """
    Expected count at allele count ``k``.

    :param h: Dominance coefficient
    :param S: Population-scaled selection coefficient
    :param k: Allele count
    :return: The expected count
    """
    kwargs = dict(limit=400, epsabs=0, epsrel=1e-11)

    if S <= 0:
        # E increases in q, so exp(E(q) - E(1)) does not overflow
        def inner(x: float) -> float:
            return quad(
                lambda q: np.exp(exponent(q, S, h) - exponent(1, S, h)), x, 1, points=get_breakpoints(x, 1, S, h),
                **kwargs
            )[0]

        norm = inner(0.0)

        def weight(x: float) -> float:
            return np.exp(-exponent(x, S, h)) * inner(x) / norm
    else:
        norm = quad(lambda q: np.exp(exponent(q, S, h)), 0, 1, points=get_breakpoints(0, 1, S, h), **kwargs)[0]

        def weight(x: float) -> float:
            return quad(
                lambda q: np.exp(S * (x - q) * ((1 - 2 * h) * (x + q) + 2 * h)), x, 1,
                points=get_breakpoints(x, 1, S, h), **kwargs
            )[0] / norm

    return quad(
        lambda x: comb(n, k) * x ** (k - 1) * (1 - x) ** (n - k - 1) * weight(x), 0, 1,
        points=get_breakpoints(0, 1, S, h), limit=400, epsabs=0, epsrel=1e-10
    )[0]


if __name__ == '__main__':
    for h, S, k in points:
        print(f"({h}, {S}, {k}, {get_count(h, S, k):.12e}),")
