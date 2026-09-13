"""Trapezoid rules on Talbot-type contours as rational approximations of the exponential.

The Cauchy integral

    exp(z) = 1/(2 pi i) * integral_Gamma exp(s) / (s - z) ds

over a Hankel contour ``Gamma`` winding around the negative real axis is parametrised as
``s = N g(theta)``, ``theta`` in ``(-pi, pi)``, and discretised by the ``N``-point trapezoid rule. The result
is a rational function of type ``(N - 1, N)`` whose poles are the quadrature nodes, so it can be applied to a
matrix with one shifted solve per node. The three contour families and their optimised parameters are from
Trefethen, Weideman and Schmelzer, *Talbot quadratures and rational approximations* (2006); the error decays
like ``rate**-N`` for ``z`` on the negative real axis.

For ``phi_l`` the paper's Theorem 5.1 gives ``phi_l(z) = 1/(2 pi i) * integral exp(s) / (s**l (s - z)) ds``,
which is exactly the trapezoid rule applied to the rational function :meth:`PartialFractions.induced` produces
from the one for ``exp``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from phi_functions.rational import PartialFractions


@dataclass(frozen=True)
class Contour:
    """A Hankel contour ``s = N g(theta)``, ``theta`` in ``(-pi, pi)``, with ``g(-theta) = conj g(theta)``.

    Attributes:
        name: Human-readable name.
        map: The function ``g``.
        derivative: The derivative ``g'``.
        rate: Asymptotic convergence factor: the trapezoid error on the negative axis is ``O(rate**-N)``.
    """

    name: str
    map: Callable[[NDArray[np.float64]], NDArray[np.complex128]]
    derivative: Callable[[NDArray[np.float64]], NDArray[np.complex128]]
    rate: float


def _cotangent_map(theta: NDArray[np.float64]) -> NDArray[np.complex128]:
    """Talbot's cotangent contour with the optimised parameters."""
    theta_cot = np.ones_like(theta) / 0.6407
    nonzero = theta != 0.0
    theta_cot[nonzero] = theta[nonzero] / np.tan(0.6407 * theta[nonzero])
    return -0.6122 + 0.5017 * theta_cot + 0.2645j * theta


def _cotangent_derivative(theta: NDArray[np.float64]) -> NDArray[np.complex128]:
    """Derivative of the cotangent contour."""
    derivative = np.zeros_like(theta)
    nonzero = theta != 0.0
    x = 0.6407 * theta[nonzero]
    derivative[nonzero] = 1.0 / np.tan(x) - x / np.sin(x) ** 2
    return 0.5017 * derivative + 0.2645j


TALBOT = Contour("talbot", _cotangent_map, _cotangent_derivative, 3.89)
"""Talbot's cotangent contour ``N (-0.6122 + 0.5017 theta cot(0.6407 theta) + 0.2645 i theta)``."""

PARABOLA = Contour(
    "parabola",
    lambda theta: 0.1309 - 0.1194 * theta**2 + 0.25j * theta,
    lambda theta: -0.2388 * theta + 0.25j * np.ones_like(theta),
    2.85,
)
"""The parabolic contour ``N (0.1309 - 0.1194 theta**2 + 0.25 i theta)``."""

HYPERBOLA = Contour(
    "hyperbola",
    lambda theta: 2.246 * (1.0 - np.sin(1.1721 - 0.3443j * theta)),
    lambda theta: 2.246 * 0.3443j * np.cos(1.1721 - 0.3443j * theta),
    3.20,
)
"""The hyperbolic contour ``N 2.246 (1 - sin(1.1721 - 0.3443 i theta))``."""

CONTOURS: dict[str, Contour] = {c.name: c for c in (TALBOT, PARABOLA, HYPERBOLA)}
"""The built-in contours by name."""


def contour_exp(nodes: int, contour: Contour | str = TALBOT) -> PartialFractions:
    """Approximate the exponential by the ``nodes``-point trapezoid rule on a Hankel contour.

    Args:
        nodes: Number of quadrature nodes ``N``; the nodes are the poles of the result, and for even ``N``
            they come in ``N/2`` conjugate pairs.
        contour: A :class:`Contour` or the name of a built-in one (``"talbot"``, ``"parabola"``,
            ``"hyperbola"``).

    Returns:
        The rational function ``i/N sum_k exp(s_k) w_k / (z - s_k)`` with ``s_k = N g(theta_k)`` and
        ``w_k = N g'(theta_k)`` at the midpoints ``theta_k`` of a uniform partition of ``(-pi, pi)``.

    Raises:
        ValueError: If ``nodes`` is not positive or the contour name is unknown.
    """
    if nodes < 1:
        msg = f"nodes must be positive, got {nodes}"
        raise ValueError(msg)
    if isinstance(contour, str):
        try:
            contour = CONTOURS[contour]
        except KeyError:
            msg = f"unknown contour {contour!r}; choose from {sorted(CONTOURS)}"
            raise ValueError(msg) from None
    # Midpoints of a uniform partition of (-pi, pi) into `nodes` cells; only those with theta > 0 are
    # computed, the rest follow by conjugation. An odd count puts one node exactly at theta = 0.
    step = 2.0 * np.pi / nodes
    upper = (np.arange(nodes // 2) + 0.5 * (1 + nodes % 2)) * step
    s_upper = nodes * contour.map(upper)
    c_upper = 1j / nodes * np.exp(s_upper) * nodes * contour.derivative(upper)
    poles = np.concatenate([s_upper.conj()[::-1], s_upper])
    residues = np.concatenate([c_upper.conj()[::-1], c_upper])
    if nodes % 2 == 1:
        zero = np.zeros(1)
        s_zero = nodes * contour.map(zero)
        c_zero = 1j / nodes * np.exp(s_zero) * nodes * contour.derivative(zero)
        poles = np.concatenate([poles[: nodes // 2], s_zero.real, poles[nodes // 2 :]])
        residues = np.concatenate([residues[: nodes // 2], c_zero.real, residues[nodes // 2 :]])
    return PartialFractions(poles, residues, 0.0)
