"""Tests for the stable evaluation of the phi functions."""

import math

import numpy as np
import pytest

from phi_functions.phi import QUADRATURE_RADIUS, phi, phi_matrix


def test_phi_zero_is_the_exponential():
    """phi_0 is exp."""
    z = np.array([-3.0, 0.0, 2.5])
    np.testing.assert_allclose(phi(0, z), np.exp(z), rtol=1e-15)


@pytest.mark.parametrize("order", [1, 2, 3, 4, 7])
def test_value_at_the_origin(order):
    """phi_l(0) = 1/l!, the point where the recurrence is useless."""
    assert phi(order, 0.0) == pytest.approx(1.0 / math.factorial(order), rel=1e-15)


def test_phi_one_closed_form():
    """phi_1(z) = (exp(z) - 1)/z away from the origin."""
    z = np.array([-4.0, -1.0, 1.0, 20.0])
    np.testing.assert_allclose(phi(1, z), (np.exp(z) - 1.0) / z, rtol=1e-14)


@pytest.mark.parametrize("order", [1, 2, 3, 4, 5])
def test_against_augmented_matrix_exponential(order):
    """Quadrature and recurrence agree with expm of the augmented matrix to near machine precision."""
    points = np.array(
        [
            1e-9,
            0.3,
            -0.7,
            2.0,
            -5.0,
            -14.9,
            14.9,
            -15.1,
            15.1,
            -40.0,
            -1e3,
            0.3 + 0.2j,
            -0.7 - 3j,
            2.0 + 14j,
            -15.1 - 2j,
            -40.0 + 30j,
        ]
    )
    reference = np.array([phi_matrix(order, np.array([[z]]))[0, 0] for z in points])
    np.testing.assert_allclose(phi(order, points), reference, rtol=1e-12)


@pytest.mark.parametrize("order", [1, 3])
def test_continuity_across_the_quadrature_radius(order):
    """The switch between quadrature and recurrence is invisible."""
    inside = phi(order, -QUADRATURE_RADIUS * (1 - 1e-12))
    outside = phi(order, -QUADRATURE_RADIUS * (1 + 1e-12))
    assert inside == pytest.approx(outside, rel=1e-10)


def test_real_in_real_out_and_shape():
    """Real input gives a real array of the input's shape; complex input a complex one; scalars give 0-d arrays."""
    z = np.linspace(-3, 1, 12).reshape(3, 4)
    out = phi(2, z)
    assert out.shape == (3, 4)
    assert out.dtype == np.float64
    assert phi(2, z + 1j).dtype == np.complex128
    assert phi(2, -1.0).shape == ()


def test_negative_order_rejected():
    """Orders below zero are meaningless."""
    with pytest.raises(ValueError, match="non-negative"):
        phi(-1, 0.0)
    with pytest.raises(ValueError, match="non-negative"):
        phi_matrix(-1, np.eye(2))


def test_phi_matrix_diagonalisable():
    """phi_l(A) = V phi_l(D) V^{-1} for a diagonalisable A."""
    rng = np.random.default_rng(0)
    v = rng.standard_normal((4, 4))
    d = np.array([-3.0, -1.0, -0.1, 0.5])
    a = v @ np.diag(d) @ np.linalg.inv(v)
    for order in (0, 1, 2, 3):
        expected = v @ np.diag(phi(order, d)) @ np.linalg.inv(v)
        np.testing.assert_allclose(phi_matrix(order, a), expected, rtol=1e-11, atol=1e-13)


def test_phi_matrix_rejects_non_square():
    """Only square matrices have functions."""
    with pytest.raises(ValueError, match="square"):
        phi_matrix(1, np.ones((2, 3)))
