"""
This test suite implements all tests for the module :mod:`pybandee.penta`

"""

# === Imports ===

import numpy as np
from scipy.linalg import solve_banded

from pybandee.penta import numba as penta

# === Tests ===


def test_pentadiagonal_solve() -> None:

    np.random.seed(0)

    for num_rows in (5, 12, 13):
        a = np.random.rand(5, num_rows)
        b = np.random.rand(a.shape[1])

        # Computation of the reference solution
        x_lapack = solve_banded(
            l_and_u=(2, 2),
            ab=a,
            b=b,
        )

        # Casting to the row-major banded format
        factorization = np.empty(shape=(a.shape[1], 5))

        factorization[0:2, 0] = 0.0
        factorization[0, 1] = 0.0
        factorization[num_rows - 1, 3] = 0.0
        factorization[num_rows - 2 :, 4] = 0.0

        factorization[2:, 0] = a[4, 0 : num_rows - 2]
        factorization[1:, 1] = a[3, 0 : num_rows - 1]
        factorization[:, 2] = a[2, :]
        factorization[: num_rows - 1, 3] = a[1, 1:]
        factorization[: num_rows - 2, 4] = a[0, 2:]

        # Factorization
        info = penta.ptrans1_factorize(matrix=factorization)
        assert info == 0

        # Solution
        x_penta = b.copy()
        penta.ptrans1_solve_single_rhs(
            factorization=factorization,
            rhs=x_penta,
        )

        assert np.allclose(x_penta, x_lapack)

    return


def test_pentadiagonal_slogdet() -> None:

    np.random.seed(0)

    for num_rows in (5, 12, 13):

        penta_matrix = np.random.rand(num_rows, 5)
        penta_matrix[::, 2] += 2.0
        dense_matrix = np.zeros((penta_matrix.shape[0], penta_matrix.shape[0]))

        dense_matrix += np.diag(penta_matrix[2:, 0], k=-2)
        dense_matrix += np.diag(penta_matrix[1:, 1], k=-1)
        dense_matrix += np.diag(penta_matrix[:, 2])
        dense_matrix += np.diag(penta_matrix[:-1, 3], k=1)
        dense_matrix += np.diag(penta_matrix[:-2, 4], k=2)

        sloget_dense = np.linalg.slogdet(dense_matrix)

        factorization = penta_matrix.copy()
        info = penta.ptrans1_factorize(matrix=factorization)
        assert info == 0

        sign, logabsdet = penta.ptrans1_slogdet(factorization=factorization)

        assert np.isclose(sign, sloget_dense.sign)
        assert np.isclose(logabsdet, sloget_dense.logabsdet)

    return


def test_pentadiagonal_symmetric_inverse_central_penta_bands() -> None:

    np.random.seed(0)

    for num_rows in (5, 12, 13, 100, 101):

        penta_matrix = np.zeros(shape=(num_rows, 5))
        dense_matrix = np.zeros(shape=(num_rows, num_rows))

        vect = np.random.rand(num_rows - 2)
        penta_matrix[2:, 0] = vect.copy()
        penta_matrix[0:-2, 4] = vect.copy()
        dense_matrix += np.diag(vect, k=-2)
        dense_matrix += np.diag(vect, k=2)
        vect = np.random.rand(num_rows - 1)
        penta_matrix[1:, 1] = vect.copy()
        penta_matrix[0:-1, 3] = vect.copy()
        dense_matrix += np.diag(vect, k=-1)
        dense_matrix += np.diag(vect, k=1)
        vect = 2.0 + np.random.rand(num_rows)
        penta_matrix[:, 2] = vect.copy()
        dense_matrix += np.diag(vect)

        penta.ptrans1_factorize(matrix=penta_matrix)
        central_inverse = penta.ptrans1_symmetric_inverse_central_penta_bands(
            factorization=penta_matrix
        )

        dense_inverse = np.linalg.inv(dense_matrix)

        assert np.allclose(
            np.diagonal(dense_inverse, offset=-2),
            central_inverse[2:, 0],
        )
        assert np.allclose(
            np.diagonal(dense_inverse, offset=-1),
            central_inverse[1:, 1],
        )
        assert np.allclose(
            np.diagonal(dense_inverse),
            central_inverse[:, 2],
        )
        assert np.allclose(
            np.diagonal(dense_inverse, offset=1),
            central_inverse[:-1, 3],
        )
        assert np.allclose(
            np.diagonal(dense_inverse, offset=2),
            central_inverse[:-2, 4],
        )

    return
