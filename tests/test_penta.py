"""
This test suite implements all tests for the module :mod:`pybandee.penta`

"""

# === Imports ===

import numpy as np
from scipy.linalg import solve_banded

from pybandee import penta

# === Tests ===


def test_pentadiagonal_solve() -> None:

    np.random.seed(0)

    for num_rows in (5, 12, 13):
        a = np.random.rand(5, num_rows)
        a_og = a.copy()
        b = np.random.rand(a.shape[1])

        x_lapack = solve_banded(
            l_and_u=(2, 2),
            ab=a,
            b=b,
        )

        x_penta = penta.penta_solve(
            lhs_matrix=a,
            matrix_format="lapack_general_banded",
            rhs=b,
            lhs_overwrite=True,
            rhs_overwrite=True,
        )

        assert np.array_equal(a, a_og)
        assert np.allclose(x_penta.solution, x_lapack)
        assert np.allclose(b, x_lapack)

    return
