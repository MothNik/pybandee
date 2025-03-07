# 📜✍️ pybandee Changelog

## v0.0a3

📆 March 07, 2025

### 🎉 New Features

- all `.penta.numba_funcs.ptrans1_*` now have a small input validation (Array shapes) to
  avoid out-of-bounds errors

### 💥⛓️‍💥 Breaking Changes

- renamed the `.penta.numba` module to `.penta.numba_funcs` to avoid confusion when
  doing something like `from pybandee.penta import numba` because this collides with
  the actual `numba` package
- all `.penta.numba_funcs.ptrans1_*` functions now return a tuple `(result, info)` where
  `result` is the actual result of the computation and `info` is an integer that
  indicates the success of the computation (`info == 0` means success and
  `info != 0` means failure)
- `.penta.numba_funcs.ptrans1_factorize` now explicitly returns the factorization and
  the `info` and allows for either overwriting the input matrix or not via the
  `overwrite_matrix` flag, i.e.,

  ```python
  # old
  factorization = matrix.copy()
  info = ptrans1_factorize(matrix=factorization)  # ← overwrites factorization

  # new
  factorization, info = ptrans1_factorize(
      matrix=matrix,
      overwrite_matrix=False,  # ← can overwrite factorization (default is False)
  )
  ```

- `.penta.numba_funcs.ptrans1_solve_single_rhs` now explicitly returns the solution and
  the `info` and allows for either overwriting the input RHS vector or not via the
  `overwrite_rhs` flag, i.e.,

  ```python
  # old
  solution = rhs.copy()
  info = ptrans1_solve_single_rhs(
      factorization=factorization,
      rhs=solution,  # ← overwrites solution
  )

  # new
  solution, info = ptrans1_solve_single_rhs(
      factorization=factorization,
      rhs=rhs,
      overwrite_rhs=False,  # ← can overwrite solution (default is False)
  )
  ```

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
