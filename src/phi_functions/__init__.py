"""Rational approximation of the exponential and the phi functions of exponential integrators.

Implements the two methods of Schmelzer and Trefethen, *Evaluating matrix functions for exponential
integrators via Carathéodory-Fejér approximation and contour integrals* (Electron. Trans. Numer. Anal. 29,
2007): near-best rational approximations of ``phi_l`` on the negative real axis by the Carathéodory-Fejér
method, and trapezoid rules on Talbot-type contours. Both yield partial fractions, so ``phi_l(A) b`` costs
one shifted linear solve per pole, and all phi functions can share one set of poles.
"""

from phi_functions.cf import CFApproximation, cf_exp, cf_phi
from phi_functions.contours import CONTOURS, HYPERBOLA, PARABOLA, TALBOT, Contour, contour_exp
from phi_functions.halphen import HALPHEN, asymptotic_error, halphen_constant
from phi_functions.matrix import PhiSolver, phi_matvec, shifted_factorizer
from phi_functions.phi import phi, phi_matrix
from phi_functions.poles import common_pole_approximations
from phi_functions.rational import PartialFractions, evaluate_shared_poles, symmetrized

__all__ = [
    "CONTOURS",
    "HALPHEN",
    "HYPERBOLA",
    "PARABOLA",
    "TALBOT",
    "CFApproximation",
    "Contour",
    "PartialFractions",
    "PhiSolver",
    "asymptotic_error",
    "cf_exp",
    "cf_phi",
    "common_pole_approximations",
    "contour_exp",
    "evaluate_shared_poles",
    "halphen_constant",
    "phi",
    "phi_matrix",
    "phi_matvec",
    "shifted_factorizer",
    "symmetrized",
]
