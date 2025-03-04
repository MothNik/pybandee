# 🧮 pybandee ⚡️

**⚠⚠⚠ Important Note ⚠⚠⚠**

This package is currently only **an experimental alpha version** that is missing

- Python interfaces with input validation
- documentation
- tests that cover all possible edge cases
- citation of the original papers

**⚠⚠⚠ Important Note ⚠⚠⚠**

A Python package specialised in factorising and solving banded matrices. Currently,
the package supports the following functionalities:

- pentadiagonal factorisation and solving
- computation of the log-determinant and inverse elements of a pentadiagonal matrix

# ☁️➡️📦 Installation

While this package can be installed and run without the optional dependency `numba`
as

```bash
pip install pybandee
```

It is recommended to install the package together with `numba` because the low-level
style code will be significantly faster. This can be done with

```bash
pip install pybandee["fast"]
```

## 5️⃣ Pentadiagonal Matrices

**⚠ Currently, only the Numba low-level implementation is available ⚠**<br>
**Please install the package with `pip install pybandee["fast"]` to get the performance benefits**

For a well-conditioned pentadiagonal matrix consisting of

- 2 sub-diagonals,
- 1 main diagonal, and
- 2 super-diagonals

a simple $\mathbf{L}\mathbf{U}$ factorisation can be computed using the _PTRANS-I_
algorithm. This can be solved efficiently and allows for the computation of additional
properties such as the log-determinant and inverse elements.

For the `numba` low-level implementations, the pentadiagonal matrix

$$
\mathbf{A} = \begin{bmatrix}
c_{0} & d_{0} & e_{0} & 0 & 0 & 0 & 0 & 0 \\
b_{1} & c_{1} & d_{1} & e_{1} & 0 & 0 & 0 & 0 \\
a_{2} & b_{2} & c_{2} & d_{2} & e_{2} & 0 & 0 & 0 \\
0 & a_{3} & b_{3} & c_{3} & d_{3} & e_{3} & 0 & 0 \\
0 & 0 & a_{4} & b_{4} & c_{4} & d_{4} & e_{4} & 0 \\
0 & 0 & 0 & a_{5} & b_{5} & c_{5} & d_{5} & e_{5} \\
0 & 0 & 0 & 0 & a_{6} & b_{6} & c_{6} & d_{6} \\
0 & 0 & 0 & 0 & 0 & a_{7} & b_{7} & c_{7} \\
\end{bmatrix}
$$

needs to be stored in the row-major banded storage format as

```python
a = np.array(
    [   #  sub2   sub1   main   sup1   sup2
        [     *,     *,    c0,    d0,    e0     ],
        [     *,    b1,    c1,    d1,    e1     ],
        [    a2,    b2,    c2,    d2,    e2     ],
        [    a3,    b3,    c3,    d3,    e3     ],
        [    a4,    b4,    c4,    d4,    e4     ],
        [    a5,    b5,    c5,    d5,    e5     ],
        [    a6,    b6,    c6,    d6,     *     ],
        [    a7,    b7,    c7,     *,     *     ],
    ],
    order="C",  # ← this is important
)
```

where the entries marked with `*` are not used and can be any value.

A linear system can be solved like

```python
 === Imports ===

import numpy as np

from pybandee.penta import numba as jit_penta

# === Setup ===

np.random.seed(0)

# a symmetric positive definite pentadiagonal matrix is created in row-major banded
# storage format
lhs_matrix = np.zeros(shape=(50, 5), dtype=np.float64)
column = np.random.rand(lhs_matrix.shape[0] - 2)
lhs_matrix[2:, 0] = column.copy()
lhs_matrix[0:-2, 4] = column.copy()
column = np.random.rand(lhs_matrix.shape[0] - 1)
lhs_matrix[1:, 1] = column.copy()
lhs_matrix[0:-1, 3] = column.copy()
column = np.random.rand(lhs_matrix.shape[0])
lhs_matrix[:, 2] = column.copy() + 2.0  # to ensure positive definiteness
lhs_matrix_original = lhs_matrix.copy()

rhs_vector = np.random.rand(lhs_matrix.shape[0])  # ← will be overwritten by the solve
rhs_vector_original = rhs_vector.copy()

# === Factorisation and Solve ===

info = jit_penta.ptrans1_factorize(matrix=lhs_matrix)
assert info == 0  # ← factorization successful

jit_penta.ptrans1_solve_single_rhs(
    factorization=lhs_matrix,  # ← is now the factorization
    rhs=rhs_vector,  # ← will become the solution
)

# === Comparison against Numpy ===

lhs_matrix_dense = np.zeros(
    shape=(lhs_matrix_original.shape[0], lhs_matrix_original.shape[0]),
    dtype=np.float64,
)
lhs_matrix_dense += np.diag(lhs_matrix_original[2::, 0], k=-2)
lhs_matrix_dense += np.diag(lhs_matrix_original[1::, 1], k=-1)
lhs_matrix_dense += np.diag(lhs_matrix_original[:, 2])
lhs_matrix_dense += np.diag(lhs_matrix_original[:-1, 3], k=1)
lhs_matrix_dense += np.diag(lhs_matrix_original[:-2, 4], k=2)

assert np.allclose(
    np.linalg.solve(lhs_matrix_dense, rhs_vector_original),
    rhs_vector,
)
```

Further properties can be computed such as

- the log-determinant of the matrix

```python
# === Log-Determinant ===

sign, logabsdet = jit_penta.ptrans1_slogdet(factorization=lhs_matrix)
numpy_sign, numpy_logabsdet = np.linalg.slogdet(lhs_matrix_dense)

assert np.isclose(sign, numpy_sign)
assert np.isclose(logabsdet, numpy_logabsdet)
```

- the central pentadiagonal band of the inverse in the same layout as the original
  matrix (**symmetric matrices only**)

```python
# === Central pentadiagonal band of the inverse ===

inverse_band = jit_penta.ptrans1_symmetric_inverse_central_penta_bands(
    factorization=lhs_matrix
)

numpy_inverse = np.linalg.inv(lhs_matrix_dense)
numpy_inverse_band = np.zeros_like(inverse_band)

numpy_inverse_band[2::, 0] = np.diag(numpy_inverse, k=-2)
numpy_inverse_band[1::, 1] = np.diag(numpy_inverse, k=-1)
numpy_inverse_band[:, 2] = np.diag(numpy_inverse)
numpy_inverse_band[:-1, 3] = np.diag(numpy_inverse, k=1)
numpy_inverse_band[:-2, 4] = np.diag(numpy_inverse, k=2)

assert np.allclose(inverse_band, numpy_inverse_band)
```

All of the functions can be invoked in a `numba.jit`-decorated function to get the
maximum performance benefits ⚡️
