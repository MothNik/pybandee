# 📜✍️ pybandee Changelog

## v0.0a2

📆 March 06, 2025

### 🎉 New Features

- implemented Numba `jit`-compiled functions to handle pentadiagonal matrices in
  `.penta.numba`, namely
  - `.ptrans1_factorize` for a standard LU-decomposition of the matrix `A` without
    pivoting
  - `.ptrans1_solve_single_rhs` for solving the system `A @ x = b` from the
    LU-decomposition of `A`
  - `.ptrans1_slogdet` for computing the log-determinant of the matrix `A` from its
    LU-decomposition
  - `.ptrans1_symmetric_inverse_central_penta_bands` for computing
    the central pentadiagonal part of `inv(A)` from the LU-decomposition of `A`
