"""Halphen's constant, the rate of best rational approximation of the exponential on the negative axis.

Gonchar and Rakhmanov proved the "1/9" conjecture in the form ``lim E_nn(exp)**(1/n) = upsilon`` with
``1/upsilon = 9.28903...``, where ``upsilon`` is the unique positive root of

    sum_{n >= 1} n upsilon**n / (1 - (-upsilon)**n) = 1/8.

Aptekarev sharpened this to ``E_nn(exp) = 2 upsilon**(n + 1/2) (1 + o(1))``. The paper conjectures that the
same rate holds for every ``phi_l``.
"""

from __future__ import annotations

import math

from scipy.optimize import brentq


def _halphen_series(upsilon: float, terms: int = 200) -> float:
    """Evaluate ``sum_{n >= 1} n upsilon**n / (1 - (-upsilon)**n)`` for ``0 < upsilon < 1``."""
    return sum(n * upsilon**n / (1.0 - (-upsilon) ** n) for n in range(1, terms + 1))


def halphen_constant() -> float:
    """Compute Halphen's constant ``upsilon = 1/9.28903...``.

    Returns:
        The unique root in ``(0, 1)`` of ``sum_{n >= 1} n upsilon**n / (1 - (-upsilon)**n) = 1/8``.
    """
    return float(brentq(lambda u: _halphen_series(u) - 0.125, 0.05, 0.5, xtol=1e-16, rtol=4 * 2.220446049250313e-16))


HALPHEN = halphen_constant()
"""Halphen's constant ``upsilon``."""

NINE_POINT_TWO_EIGHT_NINE = 1.0 / HALPHEN
"""The famous ``9.28903...``: the geometric rate at which the minimax error of ``exp`` decays with the degree."""


def asymptotic_error(degree: int) -> float:
    """Aptekarev's asymptotic minimax error ``2 upsilon**(degree + 1/2)`` for the exponential.

    Args:
        degree: Type ``(degree, degree)`` of the approximation.

    Returns:
        The leading-order minimax error on the negative real axis.
    """
    return 2.0 * math.pow(HALPHEN, degree + 0.5)
