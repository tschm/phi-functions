"""Repair of pole data that is conjugate-symmetric only up to rounding."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from phi_functions._fractions import PartialFractions, _pair_conjugates


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
    real_idx, upper, lower = _split_half_planes(p)
    new_poles = list(p[real_idx].real.astype(np.complex128))
    new_residues = list(c[real_idx].real.astype(np.complex128))
    for j, k in _pair_conjugates(p, upper, lower):
        pole = 0.5 * (p[j] + p[k].conjugate())
        residue = 0.5 * (c[j] + c[k].conjugate())
        new_poles += [pole, pole.conjugate()]
        new_residues += [residue, residue.conjugate()]
    sort = np.argsort(np.asarray(new_poles).imag)
    return PartialFractions(np.asarray(new_poles)[sort], np.asarray(new_residues)[sort], complex(constant).real)


def _split_half_planes(p: NDArray[np.complex128]) -> tuple[NDArray[np.intp], NDArray[np.intp], NDArray[np.intp]]:
    """Split pole indices into the real pole, the upper and the lower half-plane, up to rounding.

    When the number of poles is odd, the pole nearest the real axis counts as the real one; every other pole
    belongs to the upper half-plane if its imaginary part is positive and to the lower one otherwise.
    """
    order = np.argsort(np.abs(p.imag))
    n_real = p.size % 2
    rest = order[n_real:]
    return order[:n_real], rest[p[rest].imag > 0], rest[p[rest].imag <= 0]
