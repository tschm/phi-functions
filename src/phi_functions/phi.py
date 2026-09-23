"""Stable evaluation of the phi functions of exponential integrators.

The phi functions are the entire functions

    phi_0(z) = exp(z),
    phi_l(z) = (phi_{l-1}(z) - phi_{l-1}(0)) / z = (phi_{l-1}(z) - 1/(l-1)!) / z,   l >= 1,

so ``phi_1(z) = (exp(z) - 1)/z`` and ``phi_2(z) = (exp(z) - 1 - z)/z**2``. Equivalently

    phi_l(z) = sum_{k >= l} z**(k-l) / k!  =  1/(l-1)! * integral_0^1 exp((1 - t) z) t**(l-1) dt.

The recurrence loses digits to cancellation whenever ``|z|**l / l!`` is small compared with ``exp(Re z)``.
This module evaluates the integral representation with a Gauss-Legendre rule for moderate ``|z|`` and uses
the recurrence only where it is stable, so every point of the closed left half-plane is evaluated to
working precision.
"""

from __future__ import annotations

import math

import numpy as np
import scipy.linalg
from numpy.typing import ArrayLike, NDArray

QUADRATURE_RADIUS = 15.0
"""Points with ``|z|`` up to this radius are evaluated by quadrature, all others by the recurrence."""

# Gauss-Legendre rule on [0, 1]. 48 nodes integrate polynomials of degree 95 exactly, and the Taylor tail
# of exp((1 - t) z) beyond that degree is below 1e-35 for |z| <= QUADRATURE_RADIUS.
_GAUSS_NODES, _GAUSS_WEIGHTS = np.polynomial.legendre.leggauss(48)
_NODES = 0.5 * (_GAUSS_NODES + 1.0)
_WEIGHTS = 0.5 * _GAUSS_WEIGHTS


def phi(order: int, z: ArrayLike) -> NDArray[np.float64] | NDArray[np.complex128]:
    """Evaluate ``phi_order`` at the points ``z``.

    Args:
        order: Index ``l >= 0`` of the phi function; ``phi_0`` is the exponential.
        z: Real or complex scalar or array.

    Returns:
        ``phi_order(z)`` elementwise. Real input gives real output, complex input gives complex output.

    Raises:
        ValueError: If ``order`` is negative.

    Examples:
        >>> import numpy as np
        >>> float(phi(1, 0.0))
        1.0
        >>> bool(np.isclose(phi(1, -1.0), 1 - np.exp(-1.0)))
        True
        >>> phi(2, np.array([0.0, -1e-8])).round(12)
        array([0.5, 0.5])
    """
    if order < 0:
        msg = f"order must be non-negative, got {order}"
        raise ValueError(msg)
    points = np.asarray(z)
    dtype = np.complex128 if np.iscomplexobj(points) else np.float64
    flat = np.atleast_1d(points).astype(dtype).ravel()
    if order == 0:
        values = np.exp(flat)
    else:
        values = np.empty_like(flat)
        near = np.abs(flat) <= QUADRATURE_RADIUS
        values[near] = _phi_quadrature(order, flat[near])
        values[~near] = _phi_recurrence(order, flat[~near])
    return values.reshape(points.shape)


def _phi_quadrature(order: int, z: NDArray) -> NDArray:
    """Evaluate the integral representation of ``phi_order`` by Gauss-Legendre quadrature."""
    kernel = np.exp(np.multiply.outer(z, 1.0 - _NODES))
    weights = _WEIGHTS * _NODES ** (order - 1) / math.factorial(order - 1)
    return kernel @ weights


def _phi_recurrence(order: int, z: NDArray) -> NDArray:
    """Evaluate ``phi_order`` from the exponential by the defining recurrence."""
    value = np.exp(z)
    for k in range(order):
        value = (value - 1.0 / math.factorial(k)) / z
    return value


def phi_matrix(order: int, a: ArrayLike) -> NDArray:
    """Evaluate ``phi_order(A)`` for a small dense square matrix ``A``.

    The exponential of the block upper bidiagonal matrix with ``A`` in the leading block, identity blocks on
    the superdiagonal and ``order + 1`` block rows has ``phi_order(A)`` as its top-right block. This is
    Saad's identity iterated ``order`` times; it costs a dense exponential of dimension ``(order + 1) n``
    and is meant as a reference, not as the fast path (see :mod:`phi_functions.matrix` for that).

    Args:
        order: Index ``l >= 0`` of the phi function.
        a: Square matrix.

    Returns:
        The matrix ``phi_order(A)``.

    Raises:
        ValueError: If ``order`` is negative or ``a`` is not square.

    Examples:
        For a diagonal matrix ``phi_order`` acts on the diagonal entries:

        >>> import numpy as np
        >>> a = np.diag([0.0, -1.0])
        >>> bool(np.allclose(phi_matrix(1, a), np.diag(phi(1, [0.0, -1.0]))))
        True
    """
    if order < 0:
        msg = f"order must be non-negative, got {order}"
        raise ValueError(msg)
    matrix = np.asarray(a)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        msg = f"expected a square matrix, got shape {matrix.shape}"
        raise ValueError(msg)
    n = matrix.shape[0]
    if order == 0:
        return scipy.linalg.expm(matrix)
    dtype = np.result_type(matrix.dtype, np.float64)
    augmented = np.zeros(((order + 1) * n, (order + 1) * n), dtype=dtype)
    augmented[:n, :n] = matrix
    for k in range(order):
        augmented[k * n : (k + 1) * n, (k + 1) * n : (k + 2) * n] = np.eye(n)
    return scipy.linalg.expm(augmented)[:n, order * n :]
