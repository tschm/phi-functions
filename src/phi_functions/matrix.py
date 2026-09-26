"""Evaluate ``phi_l(h A) b`` for several ``l`` at once, with a common set of poles.

Exponential integrators need ``exp(h A) b`` and ``phi_l(h A) b`` for a few ``l`` at every step. With a
rational approximation in partial fractions, each is a sum of solutions of shifted systems
``(h A - z_j I) x_j = b``. Approximating all phi functions with the *same* poles (eq. (4.2) of the paper)
means the shifted systems are solved once and every ``phi_l(h A) b`` is a different linear combination of the
same ``x_j``. For real data the poles come in conjugate pairs and only half of the systems are solved.

The common poles come from :func:`phi_functions.poles.common_pole_approximations`, either from a
Carathéodory-Fejér approximation or from a Talbot-type contour.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
from numpy.typing import ArrayLike, NDArray

from phi_functions.poles import Method, common_pole_approximations
from phi_functions.rational import evaluate_shared_poles

Factorize = Callable[[complex], Callable[[NDArray], NDArray]]
"""``factorize(z)`` returns a function ``rhs -> (h A - z I)^{-1} rhs`` for the operator being evaluated."""


def shifted_factorizer(a: ArrayLike | scipy.sparse.sparray | scipy.sparse.spmatrix, *, h: float = 1.0) -> Factorize:
    """Return ``factorize(z)`` computing ``(h A - z I)^{-1}`` by a dense or sparse LU decomposition.

    Args:
        a: Square matrix, dense or SciPy sparse.
        h: Time step multiplying ``A``.

    Returns:
        A function mapping a shift ``z`` to a solver ``rhs -> (h A - z I)^{-1} rhs``.

    Raises:
        ValueError: If ``a`` is not square.

    Examples:
        For a diagonal matrix the shifted solve divides by ``a_ii - z``:

        >>> import numpy as np
        >>> solve = shifted_factorizer(np.diag([-1.0, -2.0]))(1j)
        >>> solve(np.ones(2))
        array([-0.5+0.5j, -0.4+0.2j])
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

    Examples:
        For ``A = -I`` every ``phi_l(A) b`` is ``phi_l(-1) b``:

        >>> import numpy as np
        >>> solver = PhiSolver(-np.eye(3), orders=(0, 1), degree=12)
        >>> e0, e1 = solver(np.ones(3))
        >>> bool(np.allclose(e0, np.exp(-1.0))), bool(np.allclose(e1, 1 - np.exp(-1.0)))
        (True, True)
        >>> solver.poles.size
        12
    """

    def __init__(
        self,
        a: ArrayLike | scipy.sparse.sparray | scipy.sparse.spmatrix,
        orders: Sequence[int] = (0, 1, 2, 3),
        *,
        h: float = 1.0,
        degree: int = 12,
        method: Method = "cf",
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
    method: Method = "cf",
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

    Examples:
        For ``A = -I`` the rows are ``exp(-1) b`` and ``phi_1(-1) b = (1 - exp(-1)) b``:

        >>> import numpy as np
        >>> phi_matvec(-np.eye(2), np.ones(2), orders=(0, 1)).round(6)
        array([[0.367879, 0.367879],
               [0.632121, 0.632121]])
    """
    solver = PhiSolver(
        a, orders, h=h, degree=degree, method=method, shift=shift, base_order=base_order, factorize=factorize
    )
    return solver(b)
