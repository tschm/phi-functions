"""Property-based tests for the identities ``PartialFractions`` promises for any poles and residues.

The poles are drawn in the right half-plane with real part at least one, and every evaluation point lies
on the closed negative real axis, so no point comes closer than one to a pole and the identities can be
checked to a tight tolerance.
"""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from phi_functions.rational import PartialFractions

pytestmark = pytest.mark.property

coefficients = st.floats(min_value=-3.0, max_value=3.0)
real_parts = st.floats(min_value=1.0, max_value=5.0)
points = st.lists(st.floats(min_value=-20.0, max_value=0.0), min_size=1, max_size=5).map(np.array)


@st.composite
def real_fractions(draw: st.DrawFn) -> PartialFractions:
    """A rational function with real coefficients: conjugate pole pairs, real poles and a real constant."""
    pairs = draw(st.integers(min_value=0, max_value=3))
    singles = draw(st.integers(min_value=0 if pairs else 1, max_value=3))
    poles: list[complex] = []
    residues: list[complex] = []
    for _ in range(pairs):
        pole = complex(draw(real_parts), draw(st.floats(min_value=0.5, max_value=5.0)))
        residue = complex(draw(coefficients), draw(coefficients))
        poles += [pole, pole.conjugate()]
        residues += [residue, residue.conjugate()]
    for _ in range(singles):
        poles.append(draw(real_parts))
        residues.append(draw(coefficients))
    return PartialFractions(poles, residues, draw(coefficients))


def scale(r: PartialFractions) -> float:
    """An upper bound on ``|r(z)|`` for ``z`` at distance at least one from every pole."""
    return abs(r.constant) + float(np.sum(np.abs(r.residues)))


@settings(deadline=None)
@given(r=real_fractions(), z=points, s=st.floats(min_value=0.0, max_value=2.0))
def test_shifted_is_exp_s_times_r_of_z_minus_s(r, z, s):
    """shifted(s)(z) = exp(s) r(z - s)."""
    np.testing.assert_allclose(r.shifted(s)(z), np.exp(s) * r(z - s), rtol=1e-12, atol=1e-12 * np.exp(s) * scale(r))


@settings(deadline=None)
@given(r=real_fractions(), z=st.floats(min_value=-20.0, max_value=0.0))
def test_induced_is_the_off_diagonal_entry_of_r_of_the_block_matrix(r, z):
    """Proposition 4.1 of the paper: r([[z, 1], [0, 0]]) has induced(1)(z) in its (1, 2) entry."""
    block = np.array([[z, 1.0], [0.0, 0.0]])
    value = r.constant * np.eye(2) + sum(
        c * np.linalg.inv(block - p * np.eye(2)) for p, c in zip(r.poles, r.residues, strict=True)
    )
    np.testing.assert_allclose(value[0, 1], r.induced(1)(z), rtol=1e-12, atol=1e-12 * scale(r))


@settings(deadline=None)
@given(r=real_fractions(), z=points)
def test_real_coefficients_give_real_values_on_the_real_axis(r, z):
    """A function with real coefficients is real on the real axis, and returned as a real array."""
    values = r(z)
    assert r.is_real
    assert values.dtype == np.float64
    complex_values = r(z.astype(np.complex128))
    np.testing.assert_allclose(complex_values.imag, 0.0, atol=1e-12 * scale(r))
    np.testing.assert_allclose(values, complex_values.real, rtol=1e-12, atol=1e-12 * scale(r))


@settings(deadline=None)
@given(r=real_fractions(), n=st.integers(min_value=1, max_value=6), seed=st.integers(min_value=0, max_value=2**32 - 1))
def test_apply_with_real_data_matches_dense_evaluation(r, n, seed):
    """apply(solve, b) with half the solves equals constant b + sum c_j (A - z_j I)^{-1} b, and is real."""
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    a = -(q * rng.uniform(0.0, 10.0, n)) @ q.T
    b = rng.standard_normal(n)

    def solve(z: complex, rhs: np.ndarray) -> np.ndarray:
        """Return (A - z I)^{-1} rhs."""
        return np.linalg.solve(a - z * np.eye(n), rhs)

    dense = r.constant * b + sum(c * solve(p, b) for p, c in zip(r.poles, r.residues, strict=True))
    result = r.apply(solve, b)
    assert result.dtype == np.float64
    tolerance = 1e-10 * scale(r) * np.linalg.norm(b)
    np.testing.assert_allclose(dense.imag, 0.0, atol=tolerance)
    np.testing.assert_allclose(result, dense.real, rtol=1e-10, atol=tolerance)
