"""Evaluate ``phi_l(h A) b`` for several ``l`` at once, with a common set of poles.

Exponential integrators need ``exp(h A) b`` and ``phi_l(h A) b`` for a few ``l`` at every step. With a
rational approximation in partial fractions, each is a sum of solutions of shifted systems
``(h A - z_j I) x_j = b``. Approximating all phi functions with the *same* poles (eq. (4.2) of the paper)
means the shifted systems are solved once and every ``phi_l(h A) b`` is a different linear combination of the
same ``x_j``. For real data the poles come in conjugate pairs and only half of the systems are solved.

The poles come either from a Carathéodory-Fejér approximation (``degree`` poles, about
``9.28903**-degree`` accuracy) or from a Talbot-type contour (``degree`` nodes, about
``3.89**-degree`` accuracy but independent of the time step in the sense of eq. (5.7)).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
from numpy.typing import ArrayLike, NDArray

from phi_functions.cf import cf_phi
from phi_functions.contours import CONTOURS, contour_exp
from phi_functions.rational import PartialFractions, evaluate_shared_poles

Factorize = Callable[[complex], Callable[[NDArray], NDArray]]
"""``factorize(z)`` returns a function ``rhs -> (h A - z I)^{-1} rhs`` for the operator being evaluated."""


def common_pole_approximations(
    orders: Sequence[int],
    *,
    degree: int = 12,
    method: str = "cf",
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
    """
    if any(order < base_order for order in orders):
        msg = f"all orders must be at least base_order={base_order}, got {list(orders)}"
        raise ValueError(msg)
    if shift is None:
        shift = 1.0 if base_order == 0 else 0.0
    if shift != 0.0 and base_order != 0:
        msg = "a shift is only meaningful for an approximation of the exponential (base_order=0)"
        raise ValueError(msg)

    if method == "cf":
        base = cf_phi(degree, base_order).approximation
    elif method in CONTOURS:
        if base_order != 0:
            msg = "contour quadrature approximates the exponential; use base_order=0"
            raise ValueError(msg)
        base = contour_exp(degree, CONTOURS[method])
    else:
        msg = f"unknown method {method!r}; choose 'cf' or one of {sorted(CONTOURS)}"
        raise ValueError(msg)
    if shift != 0.0:
        base = base.shifted(shift)
    return [base.induced(order - base_order) for order in orders]


def shifted_factorizer(a: ArrayLike | scipy.sparse.sparray | scipy.sparse.spmatrix, *, h: float = 1.0) -> Factorize:
    """Return ``factorize(z)`` computing ``(h A - z I)^{-1}`` by a dense or sparse LU decomposition.

    Args:
        a: Square matrix, dense or SciPy sparse.
        h: Time step multiplying ``A``.

    Returns:
        A function mapping a shift ``z`` to a solver ``rhs -> (h A - z I)^{-1} rhs``.

    Raises:
        ValueError: If ``a`` is not square.
    """
    if isinstance(a, (scipy.sparse.sparray, scipy.sparse.spmatrix)):
        matrix = scipy.sparse.csc_matrix(a, dtype=np.complex128)
        n = matrix.shape[0]
        if matrix.shape != (n, n):
            msg = f"expected a square matrix, got shape {matrix.shape}"
            raise ValueError(msg)
        scaled = h * matrix
        identity = scipy.sparse.identity(n, dtype=np.complex128, format="csc")

        def factorize_sparse(z: complex) -> Callable[[NDArray], NDArray]:
            """Sparse LU of ``h A - z I``."""
            lu = scipy.sparse.linalg.splu(scaled - z * identity)
            return lu.solve

        return factorize_sparse

    dense = np.asarray(a)
    if dense.ndim != 2 or dense.shape[0] != dense.shape[1]:
        msg = f"expected a square matrix, got shape {dense.shape}"
        raise ValueError(msg)
    scaled_dense = h * dense.astype(np.complex128)
    eye = np.eye(dense.shape[0])

    def factorize_dense(z: complex) -> Callable[[NDArray], NDArray]:
        """Dense LU of ``h A - z I``."""
        lu = scipy.linalg.lu_factor(scaled_dense - z * eye)
        return lambda rhs: scipy.linalg.lu_solve(lu, rhs)

    return factorize_dense


class PhiSolver:
    """Evaluate ``phi_l(h A) b`` for a fixed operator and a fixed set of orders, reusing the factorizations.

    Each shifted matrix ``h A - z_j I`` is factorized once, on first use, so calling the solver repeatedly -
    once per time step of an exponential integrator - costs only triangular solves.
    """

    def __init__(
        self,
        a: ArrayLike | scipy.sparse.sparray | scipy.sparse.spmatrix,
        orders: Sequence[int] = (0, 1, 2, 3),
        *,
        h: float = 1.0,
        degree: int = 12,
        method: str = "cf",
        shift: float | None = None,
        base_order: int = 0,
        factorize: Factorize | None = None,
    ) -> None:
        """Set up the approximations and the factorizer.

        Args:
            a: Square matrix ``A``, dense or SciPy sparse; ignored when ``factorize`` is given.
            orders: Indices ``l`` of the phi functions to evaluate.
            h: Time step; the functions are evaluated at ``h A``.
            degree: Number of poles, see :func:`common_pole_approximations`.
            method: ``"cf"`` or a contour name, see :func:`common_pole_approximations`.
            shift: Lu's shift, see :func:`common_pole_approximations`.
            base_order: Whose poles to use, see :func:`common_pole_approximations`.
            factorize: Custom ``factorize(z)`` returning ``rhs -> (h A - z I)^{-1} rhs``; use it for
                operators that are not matrices or for solvers of your own.
        """
        self.orders = tuple(orders)
        self.fractions = common_pole_approximations(
            self.orders, degree=degree, method=method, shift=shift, base_order=base_order
        )
        self._factorize = factorize if factorize is not None else shifted_factorizer(a, h=h)
        self._solvers: dict[complex, Callable[[NDArray], NDArray]] = {}

    @property
    def poles(self) -> NDArray[np.complex128]:
        """The common poles; each costs one factorization."""
        return self.fractions[0].poles

    def _solve(self, z: complex, rhs: NDArray) -> NDArray:
        """Solve ``(h A - z I) x = rhs``, factorizing on first use."""
        solver = self._solvers.get(z)
        if solver is None:
            solver = self._solvers[z] = self._factorize(z)
        return solver(rhs)

    def __call__(self, b: ArrayLike) -> NDArray:
        """Evaluate ``phi_l(h A) b`` for every ``l`` in ``orders``.

        Args:
            b: Right-hand side, a vector or a matrix of column vectors.

        Returns:
            Array of shape ``(len(orders), *b.shape)``.
        """
        return evaluate_shared_poles(self.fractions, self._solve, b)


def phi_matvec(
    a: ArrayLike | scipy.sparse.sparray | scipy.sparse.spmatrix,
    b: ArrayLike,
    orders: Sequence[int] = (0, 1, 2, 3),
    *,
    h: float = 1.0,
    degree: int = 12,
    method: str = "cf",
    shift: float | None = None,
    base_order: int = 0,
    factorize: Factorize | None = None,
) -> NDArray:
    """Evaluate ``phi_l(h A) b`` for the given orders in one go.

    Convenience wrapper around :class:`PhiSolver` for a single right-hand side. Build a :class:`PhiSolver`
    directly when the same operator is applied repeatedly.

    Args:
        a: Square matrix ``A``, dense or SciPy sparse.
        b: Right-hand side.
        orders: Indices ``l`` of the phi functions.
        h: Time step, see :class:`PhiSolver`.
        degree: Number of poles, see :class:`PhiSolver`.
        method: ``"cf"`` or a contour name, see :class:`PhiSolver`.
        shift: Lu's shift, see :class:`PhiSolver`.
        base_order: Whose poles to use, see :class:`PhiSolver`.
        factorize: Custom factorizer, see :class:`PhiSolver`.

    Returns:
        Array of shape ``(len(orders), *b.shape)`` with ``phi_l(h A) b`` for each ``l``.
    """
    solver = PhiSolver(
        a, orders, h=h, degree=degree, method=method, shift=shift, base_order=base_order, factorize=factorize
    )
    return solver(b)
