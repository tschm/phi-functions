"""Tests for the trapezoid rules on Talbot-type contours."""

import numpy as np
import pytest

from phi_functions.contours import CONTOURS, HYPERBOLA, PARABOLA, TALBOT, contour_exp
from phi_functions.phi import phi


@pytest.mark.parametrize("contour", [TALBOT, PARABOLA, HYPERBOLA])
@pytest.mark.parametrize("nodes", [8, 12, 16, 24])
def test_geometric_convergence(contour, nodes, negative_axis):
    """The error on the negative axis decays like rate**-N for each contour family."""
    fraction = contour_exp(nodes, contour)
    error = np.max(np.abs(fraction(negative_axis) - np.exp(negative_axis)))
    assert error <= 3.0 * contour.rate**-nodes + 1e-13
    assert error >= 0.3 * contour.rate**-nodes


def test_names_resolve_to_the_same_contours():
    """A contour can be requested by name."""
    for name, contour in CONTOURS.items():
        np.testing.assert_array_equal(contour_exp(10, name).poles, contour_exp(10, contour).poles)
    with pytest.raises(ValueError, match="unknown contour"):
        contour_exp(10, "circle")
    with pytest.raises(ValueError, match="positive"):
        contour_exp(0)


@pytest.mark.parametrize("nodes", [7, 24, 25])
def test_nodes_are_conjugate_symmetric(nodes):
    """Even counts give N/2 conjugate pairs, odd counts one more node on the real axis; the count is exact."""
    fraction = contour_exp(nodes)
    assert fraction.degree == nodes
    assert fraction.is_real
    upper, real = fraction.conjugate_representatives()
    assert len(upper) == nodes // 2
    assert len(real) == nodes % 2


def test_odd_node_count_is_as_accurate(negative_axis):
    """The node at theta = 0 is handled like every other node."""
    error = np.max(np.abs(contour_exp(25)(negative_axis) - np.exp(negative_axis)))
    assert error < 1e-13


def test_twenty_four_nodes_reach_machine_precision(negative_axis):
    """Talbot's contour with 24 nodes, 12 conjugate pairs, gives exp to about 1e-14 on the negative axis."""
    error = np.max(np.abs(contour_exp(24)(negative_axis) - np.exp(negative_axis)))
    assert error < 1e-13


def test_induced_phi_functions_match_figure_5_2(negative_axis):
    """Theorem 5.1: the rule induced on phi_l keeps exponential accuracy, as in Fig. 5.2 (shift s = 1)."""
    shifted = contour_exp(24).shifted(1.0)
    bounds = [2e-13, 1e-12, 2e-12, 1e-11]
    for order, bound in enumerate(bounds):
        error = np.max(np.abs(shifted.induced(order)(negative_axis) - phi(order, negative_axis)))
        assert error < bound, order


def test_contour_winds_around_the_negative_axis():
    """Nodes run from the lower to the upper half-plane and pass to the right of the origin."""
    fraction = contour_exp(16)
    assert np.all(np.diff(fraction.poles.imag) > 0)
    assert np.all(fraction.poles.real[fraction.poles.imag**2 < 1.0] > 0) or fraction.poles.real[7] > 0
