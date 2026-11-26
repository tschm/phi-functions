"""Rational functions in partial-fraction form, and the operations the paper builds on them.

A rational function of type ``(n, n)`` with simple poles is written as

    r(z) = constant + sum_j residues[j] / (z - poles[j]).

Three operations matter for exponential integrators:

* ``shifted(s)`` returns ``exp(s) r(z - s)``, which turns an approximation of ``exp`` on the negative axis
  into one on ``(-inf, s]`` (Lu's shift, eq. (4.4) of the paper);
* ``induced(k)`` returns ``sum_j residues[j] poles[j]**(-k) / (z - poles[j])``, the approximation of
  ``phi_{l+k}`` with the poles of an approximation of ``phi_l`` (eq. (4.2)), and
* ``apply(solve, b)`` evaluates ``r(A) b`` with one shifted linear solve per pole, or per conjugate pair
  when the data are real.

The implementation lives in private modules; this module is the public entry point.
"""

from phi_functions._fractions import PartialFractions
from phi_functions._shared import Solve, evaluate_shared_poles
from phi_functions._symmetry import symmetrized

__all__ = ["PartialFractions", "Solve", "evaluate_shared_poles", "symmetrized"]
