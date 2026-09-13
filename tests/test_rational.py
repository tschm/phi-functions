"""Tests for the partial-fraction rational functions."""

import numpy as np
import pytest

from phi_functions.rational import PartialFractions, evaluate_shared_poles, symmetrized


@pytest.fixture
def real_fraction() -> PartialFractions:
    """A real-coefficient function with one conjugate pair and one real pole."""
    return PartialFractions([1 + 2j, 1 - 2j, -3.0], [0.5 - 1j, 0.5 + 1j, 2.0], 0.25)


def test_evaluation_matches_the_formula(real_fraction):
    """r(z) = constant + sum c_j / (z - z_j)."""
    z = np.array([0.0, 1.5, -7.0])
    expected = 0.25 + (0.5 - 1j) / (z - (1 + 2j)) + (0.5 + 1j) / (z - (1 - 2j)) + 2.0 / (z + 3.0)
    values = real_fraction(z)
    assert values.dtype == np.float64
    np.testing.assert_allclose(values, expected.real, rtol=1e-14)
    assert real_fraction(1j).dtype == np.complex128


def test_is_real_detects_conjugate_symmetry(real_fraction):
    """Conjugate-closed poles with conjugate residues and a real constant have real coefficients."""
    assert real_fraction.is_real
    upper, real = real_fraction.conjugate_representatives()
    assert list(upper) == [0]
    assert list(real) == [2]
    assert not PartialFractions([1 + 2j, 1 - 2j], [0.5 - 1j, 0.5 - 1j]).is_real
    assert not PartialFractions([1 + 2j], [1.0]).is_real
    assert not PartialFractions([-3.0], [1j]).is_real
    assert not PartialFractions([-3.0], [1.0], 1j).is_real


def test_shifted_is_exp_s_times_r_of_z_minus_s(real_fraction):
    """shifted(s)(z) = exp(s) r(z - s)."""
    z = np.array([-2.0, 0.3, 4.0])
    np.testing.assert_allclose(real_fraction.shifted(1.5)(z), np.exp(1.5) * real_fraction(z - 1.5), rtol=1e-14)


def test_induced_divides_residues_by_powers_of_the_poles(real_fraction):
    """induced(k) has residues c_j z_j^{-k} and vanishes at infinity; induced(0) is the function itself."""
    induced = real_fraction.induced(2)
    np.testing.assert_array_equal(induced.poles, real_fraction.poles)
    np.testing.assert_allclose(induced.residues, real_fraction.residues / real_fraction.poles**2)
    assert induced.constant == 0
    assert induced.is_real
    assert real_fraction.induced(0) is real_fraction
    with pytest.raises(ValueError, match="non-negative"):
        real_fraction.induced(-1)


def test_induced_is_the_off_diagonal_entry_of_r_of_the_block_matrix(real_fraction):
    """Proposition 4.1 of the paper: r([[z, 1], [0, 0]]) has induced(1)(z) in its (1, 2) entry."""
    z = 0.7
    block = np.array([[z, 1.0], [0.0, 0.0]])
    value = real_fraction.constant * np.eye(2) + sum(
        c * np.linalg.inv(block - p * np.eye(2))
        for p, c in zip(real_fraction.poles, real_fraction.residues, strict=True)
    )
    assert value[0, 1] == pytest.approx(real_fraction.induced(1)(z), rel=1e-13)


def test_apply_matches_dense_evaluation(real_fraction):
    """r(A) b through shifted solves equals the dense r(A) b."""
    rng = np.random.default_rng(1)
    a = rng.standard_normal((5, 5))
    b = rng.standard_normal(5)
    eye = np.eye(5)
    dense = real_fraction.constant * eye + sum(
        c * np.linalg.inv(a - p * eye) for p, c in zip(real_fraction.poles, real_fraction.residues, strict=True)
    )
    calls = []

    def solve(z, rhs):
        """Solve the shifted system and record the shift."""
        calls.append(z)
        return np.linalg.solve(a - z * eye, rhs)

    result = real_fraction.apply(solve, b)
    assert result.dtype == np.float64
    np.testing.assert_allclose(result, (dense @ b).real, rtol=1e-12)
    assert len(calls) == 2, "one solve per conjugate pair plus one per real pole"

    complex_result = real_fraction.apply(solve, b.astype(complex))
    assert complex_result.dtype == np.complex128
    np.testing.assert_allclose(complex_result, dense @ b, rtol=1e-12)
    assert len(calls) == 5, "complex data solve every pole"


def test_shared_poles_require_identical_poles(real_fraction):
    """The common-pole evaluation refuses functions with different poles, and needs at least one function."""
    other = PartialFractions(real_fraction.poles + 1, real_fraction.residues)

    def solve(z, rhs):
        """Scalar operator A = 1."""
        return rhs / (1.0 - z)

    with pytest.raises(ValueError, match="same poles"):
        evaluate_shared_poles([real_fraction, other], solve, np.ones(3))
    with pytest.raises(ValueError, match="at least one"):
        evaluate_shared_poles([], solve, np.ones(3))


def test_shared_poles_evaluate_several_functions_with_one_solve_per_pole(real_fraction):
    """Two functions with common poles cost the same number of solves as one."""
    induced = real_fraction.induced(1)
    calls = []

    def solve(z, rhs):
        """Scalar operator A = 2, recording the shifts."""
        calls.append(z)
        return rhs / (2.0 - z)

    values = evaluate_shared_poles([real_fraction, induced], solve, np.ones(2))
    assert values.shape == (2, 2)
    np.testing.assert_allclose(values[0], real_fraction(2.0), rtol=1e-14)
    np.testing.assert_allclose(values[1], induced(2.0), rtol=1e-14)
    assert len(calls) == 2


def test_symmetrized_repairs_rounding():
    """Poles that are conjugate up to rounding become exactly conjugate; an odd one lands on the real axis."""
    poles = np.array([1 + 2j, 1 - 2j + 1e-14, -3.0 + 1e-15j])
    residues = np.array([0.5 - 1j, 0.5 + 1j - 1e-14j, 2.0 + 1e-15j])
    fraction = symmetrized(poles, residues, 0.25 + 1e-15j)
    assert fraction.is_real
    assert fraction.poles[1].imag == 0
    assert fraction.poles[0] == fraction.poles[2].conjugate()
    assert fraction.constant == 0.25


def test_symmetrized_rejects_asymmetric_poles():
    """Two poles in the upper half-plane and none below cannot be paired."""
    with pytest.raises(ValueError, match="symmetric"):
        symmetrized([1j, 2j], [1.0, 1.0])


def test_mismatched_shapes_rejected():
    """Poles and residues must be one-dimensional and of equal length."""
    with pytest.raises(ValueError, match="equal length"):
        PartialFractions([1j, -1j], [1.0])
