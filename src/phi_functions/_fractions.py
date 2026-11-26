"""The ``PartialFractions`` type: a rational function with simple poles in partial-fraction form."""

from __future__ import annotations

import cmath
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from phi_functions._shared import Solve, evaluate_shared_poles


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
        upper = np.flatnonzero(self.poles.imag > 0)
        real = np.flatnonzero(self.poles.imag == 0)
        lower = np.flatnonzero(self.poles.imag < 0)
        if upper.size != lower.size or not self._real_scalars(real, rtol):
            return None
        pairs = _pair_conjugates(self.poles, upper, lower)
        if not all(self._conjugate_pair(j, k, rtol) for j, k in pairs):
            return None
        return upper, real

    def _real_scalars(self, real: NDArray[np.intp], rtol: float) -> bool:
        """Whether the constant and the residues at the poles ``real`` are real up to ``rtol``."""
        residues = self.residues[real]
        real_residues = not np.any(np.abs(residues.imag) > rtol * np.abs(residues))
        return real_residues and abs(self.constant.imag) <= rtol * max(1.0, abs(self.constant))

    def _conjugate_pair(self, j: int, k: int, rtol: float) -> bool:
        """Whether pole and residue ``k`` are the conjugates of pole and residue ``j`` up to ``rtol``."""
        pole, residue = self.poles[j].conjugate(), self.residues[j].conjugate()
        close_pole = abs(self.poles[k] - pole) <= rtol * abs(pole)
        return close_pole and abs(self.residues[k] - residue) <= rtol * abs(residue)

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

    def apply(self, solve: Solve, b: ArrayLike, *, real_operator: bool = True) -> NDArray:
        """Evaluate ``r(A) b`` with one shifted linear solve per pole.

        Args:
            solve: ``solve(z, rhs)`` returns ``(A - z I)^{-1} rhs``.
            b: Right-hand side, a vector or a matrix of column vectors.
            real_operator: Whether ``A`` is real, see :func:`evaluate_shared_poles`.

        Returns:
            ``r(A) b``, real when ``A`` and ``b`` are real and the function has real coefficients.
        """
        return evaluate_shared_poles([self], solve, b, real_operator=real_operator)[0]


def _pair_conjugates(
    poles: NDArray[np.complex128], upper: Sequence[int] | NDArray[np.intp], lower: Sequence[int] | NDArray[np.intp]
) -> list[tuple[int, int]]:
    """Pair each pole in ``upper`` with the remaining pole in ``lower`` nearest its conjugate.

    Raises:
        ValueError: If ``upper`` and ``lower`` differ in length.
    """
    if len(upper) != len(lower):
        msg = "poles are not symmetric about the real axis"
        raise ValueError(msg)
    remaining = [int(k) for k in lower]
    pairs = []
    for j in upper:
        k = min(remaining, key=lambda k: abs(poles[k] - poles[j].conjugate()))
        remaining.remove(k)
        pairs.append((int(j), k))
    return pairs
