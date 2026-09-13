"""Tests for the Carathéodory-Fejér construction, against the tables of the paper."""

import numpy as np
import pytest

from phi_functions.cf import cf_exp, cf_phi
from phi_functions.halphen import asymptotic_error
from phi_functions.phi import phi

# Table 4.1 of the paper: maximal error of the type (n, n) CF approximation to phi_l on the negative axis.
TABLE_4_1_DIAGONAL = {
    6: [1.0e-06, 8.5e-08, 7.0e-09, 5.6e-10],
    8: [1.2e-08, 7.5e-10, 4.8e-11, 3.0e-12],
    10: [1.4e-10, 7.1e-12, 3.7e-13, 1.9e-14],
    12: [1.6e-12, 6.8e-14, 4.3e-15, 5.6e-16],
}

# Table 4.1 off the diagonal: error of phi_l approximated with the poles of the CF approximant of phi_k, k < l.
TABLE_4_1_INDUCED = {
    (6, 0): [1.0e-06, 5.3e-05, 4.6e-04, 1.6e-03],
    (6, 1): [8.5e-08, 4.0e-06, 3.1e-05],
    (6, 2): [7.0e-09, 2.9e-07],
    (8, 0): [1.2e-08, 8.0e-07, 9.1e-06, 4.2e-05],
    (10, 1): [7.1e-12, 5.6e-10, 7.3e-09],
    (12, 0): [1.6e-12, 1.6e-10, 2.6e-09, 1.8e-08],
}

# Table 4.2: exp with a shift s = 1, poles shared by phi_0 .. phi_3.
TABLE_4_2_SHIFT_ONE = {
    6: [2.7e-06, 1.1e-05, 2.4e-05, 4.4e-05],
    8: [3.2e-08, 1.5e-07, 3.8e-07, 6.6e-07],
    10: [3.7e-10, 1.7e-09, 6.9e-09, 1.0e-08],
    12: [4.3e-12, 3.0e-11, 5.3e-11, 2.3e-10],
}

ROUNDOFF = 5e-16


def sup_error(fraction, order, x):
    """Maximal absolute error of a rational function against phi_order on the grid x."""
    return float(np.max(np.abs(fraction(x) - phi(order, x))))


@pytest.mark.parametrize("degree", sorted(TABLE_4_1_DIAGONAL))
@pytest.mark.parametrize("order", [0, 1, 2, 3])
def test_table_4_1_diagonal(degree, order, negative_axis):
    """The CF error matches the paper's Table 4.1 and is neither worse nor suspiciously better."""
    result = cf_phi(degree, order)
    error = sup_error(result.approximation, order, negative_axis)
    expected = TABLE_4_1_DIAGONAL[degree][order]
    assert error <= 1.3 * expected + ROUNDOFF
    assert error >= 0.7 * expected


@pytest.mark.parametrize("degree", [6, 8, 10])
@pytest.mark.parametrize("order", [0, 1, 2, 3])
def test_error_estimate_from_singular_values(degree, order, negative_axis):
    """Twice the (n+1)-st singular value predicts the actual maximal error to a few per cent."""
    result = cf_phi(degree, order)
    assert result.degree == degree
    assert result.order == order
    error = sup_error(result.approximation, order, negative_axis)
    assert result.error_estimate == pytest.approx(error, rel=0.05)
    assert result.singular_values.shape == (75,)


@pytest.mark.parametrize("degree", [6, 8, 10, 12])
def test_exponential_error_follows_aptekarev(degree, negative_axis):
    """E_nn(exp) = 2 upsilon^(n + 1/2) (1 + o(1)) already at small n."""
    error = sup_error(cf_exp(degree).approximation, 0, negative_axis)
    assert error == pytest.approx(asymptotic_error(degree), rel=0.03)


@pytest.mark.parametrize(("degree", "base"), sorted(TABLE_4_1_INDUCED))
def test_table_4_1_common_poles(degree, base, negative_axis):
    """Higher phi functions in the poles of a lower one lose accuracy exactly as Table 4.1 reports."""
    base_fraction = cf_phi(degree, base).approximation
    for k, expected in enumerate(TABLE_4_1_INDUCED[degree, base]):
        error = sup_error(base_fraction.induced(k), base + k, negative_axis)
        assert error <= 1.3 * expected + ROUNDOFF, (degree, base, k)


@pytest.mark.parametrize("degree", sorted(TABLE_4_2_SHIFT_ONE))
def test_table_4_2_shift_one(degree, negative_axis):
    """A shift of one improves the induced approximations, matching Table 4.2."""
    shifted = cf_exp(degree).approximation.shifted(1.0)
    for order, expected in enumerate(TABLE_4_2_SHIFT_ONE[degree]):
        error = sup_error(shifted.induced(order), order, negative_axis)
        assert error <= 1.3 * expected, (degree, order)


def test_approximant_is_real_with_conjugate_pairs():
    """Even degrees give n/2 conjugate pairs, odd degrees add one real pole; all poles are distinct."""
    even = cf_exp(6).approximation
    odd = cf_phi(5, 1).approximation
    assert even.is_real
    assert odd.is_real
    assert len(even.conjugate_representatives()[0]) == 3
    upper, real = odd.conjugate_representatives()
    assert (len(upper), len(real)) == (2, 1)
    assert len(np.unique(even.poles)) == 6
    assert len(np.unique(odd.poles)) == 5


def test_odd_degree_accuracy(negative_axis):
    """Odd degrees fit between their even neighbours."""
    error = sup_error(cf_exp(7).approximation, 0, negative_axis)
    assert TABLE_4_1_DIAGONAL[8][0] < error < TABLE_4_1_DIAGONAL[6][0]


def test_too_few_chebyshev_terms_for_the_degree():
    """When the singular value is at noise level the poles cannot be located and the construction says so."""
    with pytest.raises(ValueError, match="poles outside the unit disc"):
        cf_phi(40, 0, chebyshev_terms=45)


def test_invalid_parameters():
    """Degree, order and the FFT sizes are validated."""
    with pytest.raises(ValueError, match="degree"):
        cf_phi(0)
    with pytest.raises(ValueError, match="non-negative"):
        cf_phi(4, -1)
    with pytest.raises(ValueError, match="chebyshev_terms"):
        cf_phi(4, 0, chebyshev_terms=3)
