"""Rational approximations of several phi functions that share one set of poles.

An approximation of ``phi_l`` in partial fractions induces approximations of every ``phi_{l+k}`` with the same
poles (eq. (4.2) of the paper). The poles come either from a Carathéodory-Fejér approximation (``degree``
poles, about ``9.28903**-degree`` accuracy) or from a Talbot-type contour (``degree`` nodes, about
``3.89**-degree`` accuracy but independent of the time step in the sense of eq. (5.7)).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from phi_functions.cf import cf_phi
from phi_functions.contours import CONTOURS, contour_exp
from phi_functions.rational import PartialFractions

Method = Literal["cf", "talbot", "parabola", "hyperbola"]
"""Pole sources accepted by ``method``: ``"cf"`` or a key of :data:`~phi_functions.contours.CONTOURS`."""


def common_pole_approximations(
    orders: Sequence[int],
    *,
    degree: int = 12,
    method: Method = "cf",
    shift: float | None = None,
    base_order: int = 0,
) -> list[PartialFractions]:
    """Rational approximations of ``phi_l`` for the given ``l`` that share one set of poles.

    The poles are those of an approximation of ``phi_{base_order}``; the other functions are induced by
    eq. (4.2). Accuracy is best for ``l`` close to ``base_order`` (Table 4.1 of the paper) and, when the base
    is the exponential, improves markedly with a shift of order one (Table 4.2).

    Args:
        orders: Indices ``l >= base_order`` of the phi functions.
        degree: Number of poles: the CF degree, or the number of contour nodes.
        method: ``"cf"`` for Carathéodory-Fejér poles, or a contour name (``"talbot"``, ``"parabola"``,
            ``"hyperbola"``).
        shift: Lu's shift ``s`` applied to an approximation of the exponential; defaults to ``1`` when
            ``base_order`` is ``0`` and must be ``None`` or ``0`` otherwise.
        base_order: Index of the phi function whose approximation supplies the poles. Contours require ``0``.

    Returns:
        One :class:`PartialFractions` per entry of ``orders``, all with identical ``poles``.

    Raises:
        ValueError: If an order is below ``base_order``, a shift is requested for a base other than the
            exponential, or the method is unknown.

    Examples:
        Approximations of ``exp``, ``phi_1`` and ``phi_2`` with the same eight poles:

        >>> import numpy as np
        >>> fractions = common_pole_approximations((0, 1, 2), degree=8)
        >>> len(fractions), fractions[0].degree
        (3, 8)
        >>> all(np.array_equal(r.poles, fractions[0].poles) for r in fractions)
        True
    """
    if any(order < base_order for order in orders):
        msg = f"all orders must be at least base_order={base_order}, got {list(orders)}"
        raise ValueError(msg)
    shift = _resolve_shift(shift, base_order)
    base = _base_approximation(method, degree, base_order)
    if shift != 0.0:
        base = base.shifted(shift)
    return [base.induced(order - base_order) for order in orders]


def _resolve_shift(shift: float | None, base_order: int) -> float:
    """Return Lu's shift, defaulting to ``1`` for the exponential and ``0`` otherwise.

    Raises:
        ValueError: If a non-zero shift is requested for a base other than the exponential.
    """
    if shift is None:
        shift = 1.0 if base_order == 0 else 0.0
    if shift != 0.0 and base_order != 0:
        msg = "a shift is only meaningful for an approximation of the exponential (base_order=0)"
        raise ValueError(msg)
    return shift


def _base_approximation(method: Method, degree: int, base_order: int) -> PartialFractions:
    """Return the approximation of ``phi_{base_order}`` whose poles every other function reuses.

    Raises:
        ValueError: If the method is unknown, or a contour is asked for a base other than the exponential.
    """
    if method == "cf":
        return cf_phi(degree, base_order).approximation
    if method not in CONTOURS:
        msg = f"unknown method {method!r}; choose 'cf' or one of {sorted(CONTOURS)}"
        raise ValueError(msg)
    if base_order != 0:
        msg = "contour quadrature approximates the exponential; use base_order=0"
        raise ValueError(msg)
    return contour_exp(degree, CONTOURS[method])
