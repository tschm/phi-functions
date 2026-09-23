# phi-functions

Rational approximation of the exponential and of the phi functions of exponential integrators on the
negative real axis, and fast evaluation of `phi_l(A) b` for negative semidefinite matrices `A`.

The package implements the two methods of

> T. Schmelzer and L. N. Trefethen, *Evaluating matrix functions for exponential integrators via
> Carathéodory-Fejér approximation and contour integrals*, Electron. Trans. Numer. Anal. 29 (2007), 1-18.

The phi functions are `phi_0(z) = exp(z)` and `phi_l(z) = (phi_{l-1}(z) - 1/(l-1)!) / z`, so that
`phi_1(z) = (exp(z) - 1)/z`. They appear in every exponential integrator, e.g. the exponential Euler step
`u_{n+1} = exp(hA) u_n + h phi_1(hA) g(u_n)`.

## What is in the box

| Module | Content |
| --- | --- |
| `phi_functions.phi` | `phi(l, z)`: stable evaluation for scalars and arrays. `phi_matrix(l, A)`: dense reference through an augmented matrix exponential. |
| `phi_functions.cf` | `cf_phi(n, l)`: the type `(n, n)` Carathéodory-Fejér approximation of `phi_l` on `(-inf, 0]`, indistinguishable in practice from the best approximation. The singular values estimate the error for every degree at once. |
| `phi_functions.contours` | `contour_exp(N, contour)`: trapezoid rules on Talbot's cotangent contour, a parabola or a hyperbola, as rational functions with the quadrature nodes as poles. |
| `phi_functions.rational` | `PartialFractions`: poles, residues and value at infinity, with `shifted(s)` (Lu's shift), `induced(k)` (the same poles reused for `phi_{l+k}`) and `apply(solve, b)` (one shifted solve per pole, half of them for real data). |
| `phi_functions.matrix` | `PhiSolver` and `phi_matvec`: `phi_l(hA) b` for several `l` from one set of poles, with dense or sparse LU factorizations that are computed once and reused across time steps. |
| `phi_functions.halphen` | Halphen's constant `1/9.28903...`, the geometric rate at which the errors decay. |

## Usage

Near-best rational approximation of `phi_1` with six poles, accurate to about `1e-7` on the negative axis:

```python
import numpy as np
from phi_functions import cf_phi, phi

result = cf_phi(6, order=1)
r = result.approximation  # PartialFractions with 3 conjugate pairs of poles
x = -np.logspace(-6, 6, 1000)
print(f"{np.max(np.abs(r(x) - phi(1, x))):.2e} {result.error_estimate:.2e}")
```

```result
8.45e-08 8.45e-08
```

All phi functions of a matrix at once, sharing the poles of a shifted approximation of the exponential
(Table 4.2 of the paper). Each conjugate pair of poles costs one complex LU factorization; the factorizations
are cached in the solver, so calling it once per time step costs only triangular solves:

```python
import numpy as np
from phi_functions import PhiSolver, phi_matrix

n = 200
a = (n + 1) ** 2 * (-2 * np.eye(n) + np.eye(n, k=1) + np.eye(n, k=-1))  # 1-d Laplacian
solver = PhiSolver(a, orders=(0, 1, 2, 3), h=0.01, degree=12)  # 6 complex factorizations
u = np.sin(np.pi * np.arange(1, n + 1) / (n + 1))
e0, e1, e2, e3 = solver(u)  # exp(hA) u, phi_1(hA) u, ...

reference = phi_matrix(1, 0.01 * a) @ u  # dense augmented exponential
print(np.max(np.abs(e1 - reference)) < 1e-10 * np.max(np.abs(reference)))
```

```result
True
```

`a` may be a SciPy sparse matrix, in which case `splu` is used, or you can pass your own
`factorize(z) -> solve` for operators that are not matrices. With `method="talbot"` the poles are the nodes
of a 24-point Talbot contour instead, which converges more slowly (about `3.89**-N` versus
`9.29**-n`) but shares its resolvents across time steps in the sense of eq. (5.7) of the paper.

## Accuracy

The test suite reproduces Tables 4.1 and 4.2 and Fig. 5.2 of the paper. Maximal errors on the negative real
axis of the type `(n, n)` CF approximations:

| `n` | `phi_0` | `phi_1` | `phi_2` | `phi_3` |
| --- | --- | --- | --- | --- |
| 6 | 1.0e-06 | 8.5e-08 | 7.0e-09 | 5.6e-10 |
| 8 | 1.2e-08 | 7.5e-10 | 4.8e-11 | 3.0e-12 |
| 10 | 1.4e-10 | 7.1e-12 | 3.7e-13 | 1.9e-14 |
| 12 | 1.6e-12 | 7.4e-14 | 4.1e-15 | 7.9e-16 |

With the poles of the shifted (`s = 1`) exponential shared by all four functions the errors for `n = 12` are
`4.3e-12`, `2.9e-11`, `5.3e-11` and `2.3e-10`.

## Development

The development tooling (CI, pre-commit hooks, `ruff.toml`, `pytest.ini` and the `Makefile`) is synced from
the [rhiza](https://github.com/jebel-quant/rhiza) template; `.rhiza/template.yml` pins the version and
`make update` pulls a newer one. The gates CI runs are available locally:

```bash
make install        # create the virtual environment and sync dependencies
make fmt            # pre-commit hooks: ruff, markdownlint, bandit, interrogate, ...
make typecheck      # ty over src/
make deps           # deptry: unused or missing dependencies
make docs-coverage  # docstring coverage with interrogate
make security       # bandit security scan
make license        # fail on copyleft licences among the dependencies
make rhiza-test     # repository checks, including the doctests and the README examples above
make test           # the test suite with its coverage gate
make all            # every gate, as CI does
```

`make help` lists every task.
