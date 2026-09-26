"""Rational functions in partial-fraction form, and the operations the paper builds on them.

A rational function of type ``(n, n)`` with simple poles is written as

    r(z) = constant + sum_j residues[j] / (z - poles[j]).

Three operations matter for exponential integrators:

* ``shifted(s)`` returns ``exp(s) r(z - s)``, which turns an approximation of ``exp`` on the negative axis
  into one on ``(-inf, s]`` (Lu's shift, eq. (4.4) of the paper);
* ``induced(k)`` returns ``sum_j residues[j] poles[j]**(-k) / (z - poles[j])``, the approximation of
  ``phi_{l+k}`` with the poles of an approximation of ``phi_l`` (eq. (4.2)), and
* ``apply(solve, b)`` evaluates ``r(A) b`` with one shifted linear solve per pole, or per conjugate pair
  when the data are real.
"""

from __future__ import annotations

import cmath
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

Solve = Callable[[complex, NDArray], NDArray]
"""``solve(z, rhs)`` returns ``(A - z I)^{-1} rhs`` for the operator ``A`` being evaluated."""


@dataclass(frozen=True)
class PartialFractions:
    """A rational function ``constant + sum_j residues[j] / (z - poles[j])`` with simple poles.

    Attributes:
        poles: The poles ``z_j``, a one-dimensional complex array.
        residues: The residues ``c_j`` at the poles, same shape as ``poles``.
        constant: The value at infinity.

    Examples:
        ``r(z) = 1 + 2 / (z + 1)``, and the function it induces one order up:

        >>> r = PartialFractions(poles=[-1.0], residues=[2.0], constant=1.0)
        >>> float(r(1.0))
        2.0
        >>> r.is_real
        True
        >>> r.induced(1).residues.real
        array([-2.])
    """

    poles: NDArray[np.complex128]
    residues: NDArray[np.complex128]
    constant: complex = 0.0

    def __post_init__(self) -> None:
        """Coerce the fields to complex arrays and check that they match."""
        poles = np.atleast_1d(np.asarray(self.poles, dtype=np.complex128))
        residues = np.atleast_1d(np.asarray(self.residues, dtype=np.complex128))
        if poles.ndim != 1 or poles.shape != residues.shape:
            msg = f"poles and residues must be 1-d and of equal length, got {poles.shape} and {residues.shape}"
            raise ValueError(msg)
        object.__setattr__(self, "poles", poles)
        object.__setattr__(self, "residues", residues)
        object.__setattr__(self, "constant", complex(self.constant))

    @property
    def degree(self) -> int:
        """Number of poles."""
        return self.poles.size

    @property
    def is_real(self) -> bool:
        """Whether the function has real coefficients, i.e. the poles are closed under conjugation."""
        return self.conjugate_representatives() is not None

    def conjugate_representatives(self, rtol: float = 1e-10) -> tuple[NDArray[np.intp], NDArray[np.intp]] | None:
        """Split the poles into conjugate pairs and real poles.

        Args:
            rtol: Relative tolerance for matching a pole with the conjugate of another.

        Returns:
            ``(upper, real)`` with the indices of the poles in the open upper half-plane and on the real
            axis, if every pole in the upper half-plane has a conjugate partner with conjugate residue, the
            real poles have real residues and the constant is real. ``None`` otherwise.
        """
        if abs(self.constant.imag) > rtol * max(1.0, abs(self.constant)):
            return None
        upper = np.flatnonzero(self.poles.imag > 0)
        real = np.flatnonzero(self.poles.imag == 0)
        lower = list(np.flatnonzero(self.poles.imag < 0))
        if upper.size != len(lower):
            return None
        if np.any(np.abs(self.residues[real].imag) > rtol * np.abs(self.residues[real])):
            return None
        for j in upper:
            pole, residue = self.poles[j].conjugate(), self.residues[j].conjugate()
            match = next(
                (
                    k
                    for k in lower
                    if abs(self.poles[k] - pole) <= rtol * abs(pole)
                    and abs(self.residues[k] - residue) <= rtol * abs(residue)
                ),
                None,
            )
            if match is None:
                return None
            lower.remove(match)
        return upper, real

    def __call__(self, z: ArrayLike) -> NDArray:
        """Evaluate the rational function at the points ``z``.

        Args:
            z: Real or complex scalar or array.

        Returns:
            ``r(z)`` elementwise; real when ``z`` is real and the function has real coefficients.
        """
        points = np.asarray(z)
        values = self.constant + np.sum(self.residues / (points.astype(np.complex128)[..., None] - self.poles), axis=-1)
        if not np.iscomplexobj(points) and self.is_real:
            return values.real
        return values

    def shifted(self, s: float) -> PartialFractions:
        """Return ``exp(s) r(z - s)``.

        For an approximation ``r`` of the exponential this is Lu's shift: ``exp(z) = exp(s) exp(z - s)``
        with the second factor approximated on the negative axis, so the result approximates ``exp`` on
        ``(-inf, s]``.

        Args:
            s: The shift.

        Returns:
            The shifted rational function.
        """
        factor = cmath.exp(s)
        return PartialFractions(self.poles + s, factor * self.residues, factor * self.constant)

    def induced(self, k: int) -> PartialFractions:
        """Return the same-pole approximation of ``phi_{l+k}`` induced by this approximation of ``phi_l``.

        This is eq. (4.2) of the paper: ``r(B_z)`` for the block matrix ``B_z = [[z, 1], [0, 0]]`` has
        ``sum_j c_j / (z_j (z - z_j))`` in its off-diagonal entry, and iterating gives
        ``sum_j c_j z_j**(-k) / (z - z_j)``.

        Args:
            k: Number of orders to step up; ``0`` returns the function itself.

        Returns:
            The induced rational function; its value at infinity is ``0`` for ``k >= 1``.

        Raises:
            ValueError: If ``k`` is negative.
        """
        if k < 0:
            msg = f"k must be non-negative, got {k}"
            raise ValueError(msg)
        if k == 0:
            return self
        return PartialFractions(self.poles, self.residues * self.poles ** (-k), 0.0)

    def apply(self, solve: Solve, b: ArrayLike) -> NDArray:
        """Evaluate ``r(A) b`` with one shifted linear solve per pole.

        Args:
            solve: ``solve(z, rhs)`` returns ``(A - z I)^{-1} rhs``.
            b: Right-hand side, a vector or a matrix of column vectors.

        Returns:
            ``r(A) b``, real when ``b`` is real and the function has real coefficients.
        """
        return evaluate_shared_poles([self], solve, b)[0]


def evaluate_shared_poles(fractions: Sequence[PartialFractions], solve: Solve, b: ArrayLike) -> NDArray:
    """Evaluate ``r_i(A) b`` for rational functions ``r_i`` that share their poles.

    Each pole costs one linear solve regardless of how many functions are evaluated; that is the point of
    approximating all phi functions in a common set of poles. When ``b`` is real and the functions have
    real coefficients only poles in the closed upper half-plane are solved for, as ``x(conj z) = conj x(z)``.

    Args:
        fractions: Rational functions with identical ``poles`` arrays.
        solve: ``solve(z, rhs)`` returns ``(A - z I)^{-1} rhs``.
        b: Right-hand side, a vector or a matrix of column vectors.

    Returns:
        Array of shape ``(len(fractions), *b.shape)`` holding ``r_i(A) b``.

    Raises:
        ValueError: If no functions are given or their poles differ.

    Examples:
        ``r(z) = 1 + 2 / (z - 1)`` and the function it induces, applied to a diagonal matrix with one solve:

        >>> import numpy as np
        >>> a = np.diag([-1.0, -3.0])
        >>> r = PartialFractions(poles=[1.0], residues=[2.0], constant=1.0)
        >>> solve = lambda z, rhs: np.linalg.solve(a - z * np.eye(2), rhs)
        >>> evaluate_shared_poles([r, r.induced(1)], solve, np.ones(2))
        array([[ 0. ,  0.5],
               [-1. , -0.5]])
    """
    poles = _shared_poles(fractions)
    rhs = np.asarray(b)
    real_data = not np.iscomplexobj(rhs) and all(r.is_real for r in fractions)

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


def symmetrized(poles: ArrayLike, residues: ArrayLike, constant: complex = 0.0) -> PartialFractions:
    """Build a real-coefficient rational function from pole data that is conjugate-symmetric up to rounding.

    Poles are paired with their nearest conjugate partner and both members of a pair are replaced by the
    average; an unpaired pole (there is one when the degree is odd) is moved onto the real axis. The result
    satisfies :attr:`PartialFractions.is_real` exactly, which lets :func:`evaluate_shared_poles` halve the
    number of linear solves.

    Args:
        poles: Poles, symmetric about the real axis up to rounding, with at most one real pole.
        residues: Residues at those poles.
        constant: Value at infinity; its imaginary part is discarded.

    Returns:
        The symmetrized rational function, poles sorted by imaginary part.

    Raises:
        ValueError: If the poles cannot be paired.

    Examples:
        A conjugate pair perturbed by rounding becomes exactly conjugate:

        >>> r = symmetrized([1 + 2j, 1 - 2j + 1e-13], [1 + 1j, 1 - 1j + 1e-13j], 0.5)
        >>> r.is_real
        True
        >>> r.poles
        array([1.-2.j, 1.+2.j])
    """
    p = np.atleast_1d(np.asarray(poles, dtype=np.complex128))
    c = np.atleast_1d(np.asarray(residues, dtype=np.complex128))
    order = np.argsort(np.abs(p.imag))
    n_real = p.size % 2
    real_idx, rest = order[:n_real], order[n_real:]
    upper = [j for j in rest if p[j].imag > 0]
    lower = [j for j in rest if p[j].imag <= 0]
    if len(upper) != len(lower):
        msg = "poles are not symmetric about the real axis"
        raise ValueError(msg)
    new_poles = [complex(p[j].real) for j in real_idx]
    new_residues = [complex(c[j].real) for j in real_idx]
    for j in upper:
        k = min(lower, key=lambda k: abs(p[k] - p[j].conjugate()))
        lower.remove(k)
        pole = 0.5 * (p[j] + p[k].conjugate())
        residue = 0.5 * (c[j] + c[k].conjugate())
        new_poles += [pole, pole.conjugate()]
        new_residues += [residue, residue.conjugate()]
    sort = np.argsort(np.asarray(new_poles).imag)
    return PartialFractions(np.asarray(new_poles)[sort], np.asarray(new_residues)[sort], complex(constant).real)
