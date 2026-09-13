"""Shared fixtures: a log-spaced grid on the negative real axis."""

import numpy as np
import pytest


@pytest.fixture(scope="session")
def negative_axis() -> np.ndarray:
    """Points ``-10**8 <= x <= -10**-8`` on a logarithmic grid, where the error curves are measured."""
    return -np.logspace(-8, 8, 4001)
