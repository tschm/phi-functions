"""Tests for evaluating phi_l(h A) b with common poles."""

from typing import get_args

import numpy as np
import pytest
import scipy.sparse

from phi_functions.contours import CONTOURS
from phi_functions.matrix import PhiSolver, common_pole_approximations, phi_matvec, shifted_factorizer
from phi_functions.phi import phi_matrix
from phi_functions.poles import Method


@pytest.fixture(scope="module")
def laplacian() -> np.ndarray:
    """Second-order finite differences for the Laplacian on 40 interior points of the unit interval."""
    n = 40
    a = -2.0 * np.eye(n) + np.eye(n, k=1) + np.eye(n, k=-1)
    return a * (n + 1) ** 2


@pytest.fixture(scope="module")
def rhs() -> np.ndarray:
    """A smooth right-hand side."""
    return np.sin(np.pi * np.arange(1, 41) / 41)


@pytest.fixture(scope="module")
def reference(laplacian, rhs) -> np.ndarray:
    """phi_l(h A) b for l = 0 .. 3 via the dense augmented exponential."""
    return np.array([phi_matrix(order, 0.01 * laplacian) @ rhs for order in range(4)])


def relative_errors(result, reference):
    """Relative maximum error of each row."""
    return np.max(np.abs(result - reference), axis=-1) / np.max(np.abs(reference), axis=-1)


@pytest.mark.parametrize(("method", "degree", "tolerance"), [("cf", 12, 1e-8), ("talbot", 24, 1e-9), ("cf", 6, 1e-3)])
def test_phi_matvec_dense(method, degree, tolerance, laplacian, rhs, reference):
    """All four phi functions of h A applied to b, from one set of poles."""
    result = phi_matvec(laplacian, rhs, (0, 1, 2, 3), h=0.01, degree=degree, method=method)
    assert result.dtype == np.float64
    assert result.shape == (4, 40)
    assert np.all(relative_errors(result, reference) < tolerance)


def test_phi_matvec_sparse_matches_dense(laplacian, rhs, reference):
    """A SciPy sparse matrix goes through splu and gives the same answer."""
    sparse = scipy.sparse.csr_matrix(laplacian)
    result = phi_matvec(sparse, rhs, (0, 1, 2, 3), h=0.01)
    assert np.all(relative_errors(result, reference) < 1e-8)


def test_complex_data_solve_every_pole(laplacian, rhs, reference):
    """Complex right-hand sides use all poles and give a complex result."""
    result = phi_matvec(laplacian, rhs.astype(complex), (0, 1, 2, 3), h=0.01)
    assert result.dtype == np.complex128
    assert np.all(relative_errors(result, reference) < 1e-8)


def test_solver_factorizes_each_pole_once(laplacian, rhs):
    """Repeated calls reuse the factorizations; real data need one per conjugate pair."""
    inner = shifted_factorizer(laplacian, h=0.01)
    shifts = []

    def counting(z):
        """Record every factorization request."""
        shifts.append(z)
        return inner(z)

    solver = PhiSolver(laplacian, (0, 1, 2), h=0.01, degree=8, factorize=counting)
    first = solver(rhs)
    second = solver(np.column_stack([rhs, 2 * rhs]))
    assert len(shifts) == 4
    assert first.shape == (3, 40)
    assert second.shape == (3, 40, 2)
    np.testing.assert_allclose(second[..., 1], 2 * first, rtol=1e-12)
    assert solver.poles.shape == (8,)
    assert solver.orders == (0, 1, 2)


def test_exponential_euler_step(laplacian):
    """One exponential Euler step u1 = exp(hA) u0 + h phi_1(hA) g agrees with the dense formula."""
    n = laplacian.shape[0]
    x = np.arange(1, n + 1) / (n + 1)
    u0 = np.sin(np.pi * x)
    g = u0 - u0**3
    h = 0.02
    solver = PhiSolver(laplacian, (0, 1), h=h, degree=10)
    e0, e1 = solver(np.column_stack([u0, g])).transpose(0, 2, 1)
    u1 = e0[0] + h * e1[1]
    expected = phi_matrix(0, h * laplacian) @ u0 + h * phi_matrix(1, h * laplacian) @ g
    np.testing.assert_allclose(u1, expected, rtol=1e-9, atol=1e-12)


def test_common_pole_approximations_share_poles():
    """The approximations for all orders have identical poles and the requested base."""
    fractions = common_pole_approximations((1, 2, 4), degree=8, base_order=1)
    for fraction in fractions[1:]:
        np.testing.assert_array_equal(fraction.poles, fractions[0].poles)
    assert fractions[0].constant != 0
    assert fractions[1].constant == 0


def test_common_pole_approximations_validate():
    """Orders below the base, shifts off the exponential, unknown methods and contours with a base are refused."""
    with pytest.raises(ValueError, match="at least base_order"):
        common_pole_approximations((0, 1), base_order=1)
    with pytest.raises(ValueError, match="only meaningful"):
        common_pole_approximations((1, 2), base_order=1, shift=1.0)
    with pytest.raises(ValueError, match="unknown method"):
        common_pole_approximations((0,), method="pade")
    with pytest.raises(ValueError, match="base_order=0"):
        common_pole_approximations((1,), method="talbot", base_order=1)


def test_method_literal_matches_contours():
    """The ``Method`` alias lists exactly ``"cf"`` and the registered contours."""
    assert set(get_args(Method)) == {"cf", *CONTOURS}


def test_shifted_factorizer_rejects_non_square():
    """Both the dense and the sparse path check the shape."""
    with pytest.raises(ValueError, match="square"):
        shifted_factorizer(np.ones((2, 3)))
    with pytest.raises(ValueError, match="square"):
        shifted_factorizer(scipy.sparse.csr_matrix(np.ones((2, 3))))
