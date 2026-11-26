"""Evaluation of several rational functions that share their poles, with one linear solve per pole."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike, NDArray

if TYPE_CHECKING:
    from phi_functions._fractions import PartialFractions

Solve = Callable[[complex, NDArray], NDArray]
"""``solve(z, rhs)`` returns ``(A - z I)^{-1} rhs`` for the operator ``A`` being evaluated."""


def evaluate_shared_poles(
    fractions: Sequence[PartialFractions], solve: Solve, b: ArrayLike, *, real_operator: bool = True
) -> NDArray:
    """Evaluate ``r_i(A) b`` for rational functions ``r_i`` that share their poles.

    Each pole costs one linear solve regardless of how many functions are evaluated; that is the point of
    approximating all phi functions in a common set of poles. When ``A`` and ``b`` are real and the functions
    have real coefficients only poles in the closed upper half-plane are solved for, as
    ``x(conj z) = conj x(z)``. That symmetry fails for a complex ``A``, so pass ``real_operator=False`` then.

    Args:
        fractions: Rational functions with identical ``poles`` arrays.
        solve: ``solve(z, rhs)`` returns ``(A - z I)^{-1} rhs``.
        b: Right-hand side, a vector or a matrix of column vectors.
        real_operator: Whether ``A`` is real. If ``False`` every pole is solved for and the result is complex.

    Returns:
        Array of shape ``(len(fractions), *b.shape)`` holding ``r_i(A) b``.

    Raises:
        ValueError: If no functions are given or their poles differ.

    Examples:
        ``r(z) = 1 + 2 / (z - 1)`` and the function it induces, applied to a diagonal matrix with one solve:

        >>> import numpy as np
        >>> from phi_functions.rational import PartialFractions
        >>> a = np.diag([-1.0, -3.0])
        >>> r = PartialFractions(poles=[1.0], residues=[2.0], constant=1.0)
        >>> solve = lambda z, rhs: np.linalg.solve(a - z * np.eye(2), rhs)
        >>> evaluate_shared_poles([r, r.induced(1)], solve, np.ones(2))
        array([[ 0. ,  0.5],
               [-1. , -0.5]])
    """
    poles = _shared_poles(fractions)
    rhs = np.asarray(b)
    real_data = real_operator and not np.iscomplexobj(rhs) and all(r.is_real for r in fractions)

    out = np.array([r.constant * rhs for r in fractions], dtype=np.complex128)
    for j, weight in _solve_terms(fractions[0], real_data):
        x = solve(complex(poles[j]), rhs)
        for i, r in enumerate(fractions):
            contribution = r.residues[j] * x
            out[i] += weight * (contribution.real if real_data else contribution)
    return out.real if real_data else out


def _shared_poles(fractions: Sequence[PartialFractions]) -> NDArray[np.complex128]:
    """Return the poles common to all ``fractions``.

    Raises:
        ValueError: If no functions are given or their poles differ.
    """
    if not fractions:
        msg = "at least one rational function is required"
        raise ValueError(msg)
    poles = fractions[0].poles
    for other in fractions[1:]:
        if other.poles.shape != poles.shape or not np.array_equal(other.poles, poles):
            msg = "all rational functions must share the same poles"
            raise ValueError(msg)
    return poles


def _solve_terms(fraction: PartialFractions, real_data: bool) -> list[tuple[int, float]]:
    """Return the poles to solve for as ``(index, weight)`` pairs.

    For real data only the poles in the closed upper half-plane are solved for, a conjugate pair counting
    twice through the real part of its contribution; otherwise every pole is solved for once.
    """
    representatives = fraction.conjugate_representatives() if real_data else None
    if representatives is None:
        return [(j, 1.0) for j in range(fraction.degree)]
    upper, real = representatives
    return [(int(j), 2.0) for j in upper] + [(int(j), 1.0) for j in real]
