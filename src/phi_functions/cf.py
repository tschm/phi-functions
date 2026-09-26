"""Carathéodory-Fejér approximation of the phi functions on the negative real axis.

The best rational approximation of type ``(n, n)`` to ``phi_l`` on ``(-inf, 0]`` is, for practical purposes,
the Carathéodory-Fejér (CF) approximant: Magnus showed that for the exponential the two differ by only
``O(56**-n)``. The construction is the one of Trefethen, Weideman and Schmelzer, adapted to the phi functions
in the paper this package accompanies:

1. map the negative axis to ``[-1, 1]`` by ``z = scale (t - 1) / (t + 1)`` and expand ``phi_l`` in Chebyshev
   polynomials by an FFT;
2. take the singular value decomposition of the Hankel matrix of Chebyshev coefficients; the ``(n + 1)``-st
   singular value is the CF error, and the corresponding singular vectors give a finite Blaschke product;
3. read off the ``n`` poles (the roots of the singular vector outside the unit disc), compute the residues
   and map everything back to the ``z``-plane.

The minimax error decays like ``9.28903**-n`` (Halphen's constant, see :mod:`phi_functions.halphen`), so six
poles - three complex linear solves for real data - give about six digits.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg
from numpy.typing import NDArray

from phi_functions.phi import phi
from phi_functions.rational import PartialFractions, symmetrized


@dataclass(frozen=True)
class CFApproximation:
    """Result of a Carathéodory-Fejér construction.

    Attributes:
        approximation: The rational approximant, with real coefficients.
        order: Index ``l`` of the approximated phi function.
        singular_values: Singular values of the Hankel matrix. Twice ``singular_values[k]`` estimates the
            minimax error of the type ``(k, k)`` approximation, for every ``k`` at once.
    """

    approximation: PartialFractions
    order: int
    singular_values: NDArray[np.float64]

    @property
    def degree(self) -> int:
        """Number of poles."""
        return self.approximation.degree

    @property
    def error_estimate(self) -> float:
        """Estimated maximum error of the approximant on the negative real axis."""
        return 2.0 * float(self.singular_values[self.degree])


def cf_phi(
    degree: int,
    order: int = 0,
    *,
    chebyshev_terms: int = 75,
    fft_points: int = 1024,
    scale: float = 9.0,
) -> CFApproximation:
    """Construct the type ``(degree, degree)`` Carathéodory-Fejér approximation of ``phi_order`` on ``(-inf, 0]``.

    Args:
        degree: Number of poles ``n``; the approximant has type ``(n, n)``.
        order: Index ``l`` of the phi function; ``0`` is the exponential.
        chebyshev_terms: Number of Chebyshev coefficients kept; the Hankel matrix has this dimension.
        fft_points: Number of sample points on the unit circle for the FFTs.
        scale: Scale factor of the map from ``[-1, 1]`` to the negative axis; ``9`` is a good choice.

    Returns:
        The approximation together with the singular values that estimate the error for every degree.

    Raises:
        ValueError: If the parameters are inconsistent.

    Examples:
        Six poles approximate ``phi_1`` to about ``1e-7`` on the whole negative axis:

        >>> import numpy as np
        >>> result = cf_phi(6, order=1)
        >>> result.approximation.degree
        6
        >>> x = -np.logspace(-6, 6, 1000)
        >>> bool(np.max(np.abs(result.approximation(x) - phi(1, x))) < 1e-7)
        True
    """
    if degree < 1:
        msg = f"degree must be at least 1, got {degree}"
        raise ValueError(msg)
    if order < 0:
        msg = f"order must be non-negative, got {order}"
        raise ValueError(msg)
    if not degree < chebyshev_terms < fft_points:
        msg = "need degree < chebyshev_terms < fft_points"
        raise ValueError(msg)

    k = chebyshev_terms
    w = np.exp(2j * np.pi * np.arange(fft_points) / fft_points)
    t = w.real
    values = np.zeros(fft_points)
    inner = t != -1.0
    values[inner] = phi(order, scale * (t[inner] - 1.0) / (t[inner] + 1.0))
    coefficients = np.fft.fft(values).real / fft_points
    analytic = np.polyval(coefficients[k::-1], w)

    hankel = scipy.linalg.hankel(coefficients[1 : k + 1])
    left, singular_values, right_h = scipy.linalg.svd(hankel)
    sigma = singular_values[degree]
    u = left[::-1, degree]
    v = right_h[degree, :]
    padding = np.zeros(fft_points - k)
    blaschke = np.fft.fft(np.concatenate([u, padding])) / np.fft.fft(np.concatenate([v, padding]))
    extended = analytic - sigma * w**k * blaschke

    roots = np.roots(v)
    q = roots[np.abs(roots) > 1.0]
    if q.size != degree:
        msg = f"expected {degree} poles outside the unit disc, found {q.size}; try more Chebyshev terms"
        raise ValueError(msg)
    numerator = np.fft.fft(extended * np.polyval(np.poly(q), w)).real / fft_points
    numerator = numerator[degree::-1]
    residues_w = np.array(
        [np.polyval(numerator, qj) / np.polyval(np.poly(np.delete(q, j)), qj) for j, qj in enumerate(q)]
    )

    poles = scale * (q - 1.0) ** 2 / (q + 1.0) ** 2
    residues = 4.0 * residues_w * poles / (q**2 - 1.0)
    # The error curve equioscillates and takes opposite signs at z = 0 and z = -inf, where phi_l vanishes.
    constant = 0.5 * (phi(order, 0.0) + np.sum(residues / poles))
    return CFApproximation(symmetrized(poles, residues, constant), order, singular_values)


def cf_exp(
    degree: int,
    *,
    chebyshev_terms: int = 75,
    fft_points: int = 1024,
    scale: float = 9.0,
) -> CFApproximation:
    """Construct the Carathéodory-Fejér approximation of the exponential; shorthand for ``cf_phi(degree, 0)``.

    Args:
        degree: Number of poles.
        chebyshev_terms: Passed on to :func:`cf_phi`.
        fft_points: Passed on to :func:`cf_phi`.
        scale: Passed on to :func:`cf_phi`.

    Returns:
        The approximation of ``exp`` on the negative real axis.

    Examples:
        Eight poles approximate the exponential to about ``1e-8`` on the negative axis:

        >>> import numpy as np
        >>> result = cf_exp(8)
        >>> result.degree
        8
        >>> x = -np.logspace(-6, 6, 1000)
        >>> bool(np.max(np.abs(result.approximation(x) - np.exp(x))) < 2e-8)
        True
    """
    return cf_phi(degree, 0, chebyshev_terms=chebyshev_terms, fft_points=fft_points, scale=scale)
