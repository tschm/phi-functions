"""Tests for Halphen's constant."""

import pytest

from phi_functions.halphen import HALPHEN, NINE_POINT_TWO_EIGHT_NINE, asymptotic_error, halphen_constant


def test_one_over_upsilon_is_nine_point_two_eight_nine():
    """1/upsilon = 9.28903..., the constant of the "1/9" conjecture."""
    assert round(1.0 / halphen_constant(), 5) == 9.28903
    assert pytest.approx(9.2890254919, rel=1e-10) == NINE_POINT_TWO_EIGHT_NINE
    assert halphen_constant() == HALPHEN


def test_asymptotic_error():
    """Aptekarev's formula 2 upsilon^(n + 1/2)."""
    assert asymptotic_error(6) == pytest.approx(2.0 * HALPHEN**6.5)
    assert asymptotic_error(6) == pytest.approx(1.02e-6, rel=0.01)
