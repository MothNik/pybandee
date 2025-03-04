"""
Module :mod:`penta._ptrans1`

This module implements the PTRANS-I algorithm for the factorisation and solving of
pentadiagonal linear systems as described in [1]_.

It relies heavily on ``numba`` and all its functions can be called in other
``jit``-compiled functions.

"""

# === Imports ===

from typing import Tuple

import numpy as np
from numpy.typing import NDArray

from ..._utils import jit

# === Functions ===


@jit(
    "int64(float64[:, ::1])",
    nopython=True,
    cache=True,
)
def ptrans1_factorize(matrix: NDArray[np.float64]) -> int:
    """
    Factorises a pentadiagonal matrix using the PTRANS-I algorithm. Please refer to the
    Notes section for details.

    Parameters
    ----------
    matrix : :obj:`numpy.ndarray` of shape (m, 5) and dtype ``numpy.float64``
        The pentadiagonal matrix to be factorised that will be overwritten with the
        factors.
        It has to be ordered in row-major format (C-order).

    Returns
    -------
    info : :obj:`int`
        The return code of the factorisation.
        - ``0``: successful factorisation
        - ``i >= 1``: when the ``i``-th row lead to a zero division, i.e., the matrix
            is ill-conditioned

    Notes
    -----
    The pentadiagonal matrix ``A`` which is stored in row compressed format

    ```
       # sub 2      sub 1       main         sup 1     sup 2
    [[    *           *          d_0          a_0       b_0      ]
     [    *          c_1         d_1          a_1       b_1      ]
     [    e_2        c_2         d_2          a_2       b_2      ]
                                ...
     [    e_i        c_i         d_i          a_i       b_i      ]
                                ...
     [    e_{n-3}    c_{n-3}     d_{n-3}      a_{n-3}   b_{n-3}  ]
     [    e_{n-2}    c_{n-2}     d_{n-2}      a_{n-2}     *      ]
     [    e_{n-1}    c_{n-1}     d_{n-1}      *           *      ]]
    ```

    is factorised as ``A = L @ U`` like

    ```
    [[   *           *          mu_0        al_0        be_0      ]
     [   *          ga_1        mu_1        al_1        be_1      ]
     [  e_2         ga_2        mu_2        al_2        be_2      ]
                                ...
     [  e_i         ga_i        mu_i        al_i        be_i  ]
                                ...
     [  e_{n-3}     ga_{n-3}    mu_{n-3}    al_{n-3}    be_{n-3}  ]
     [  e_{n-2}     ga_{n-2}    mu_{n-2}    al_{n-2}      *       ]
     [  e_{n-1}     ga_{n-1}    mu_{n-1}      *           *       ]]
    ```

    where

    - the right two columns (``al`` "alpha" and ``be`` "beta") are the factors of the
        unit upper tridiagonal triangular transformation matrix ``U``
    - the left three columns (``e``, ``ga`` "gamma", and ``mu`` "mu") are the factors of
        the lower triangular matrix ``L``


    """

    num_rows = matrix.shape[0]

    # first row
    mu_i = matrix[0, 2]
    if mu_i == 0.0:
        return 1

    al_i_minus_2 = matrix[0, 3] / mu_i
    be_i_minus_2 = matrix[0, 4] / mu_i

    matrix[0, 0] = 0.0
    matrix[0, 1] = 0.0
    matrix[0, 2] = mu_i
    matrix[0, 3] = al_i_minus_2
    matrix[0, 4] = be_i_minus_2

    # second row
    ga_i = matrix[1, 1]
    mu_i = matrix[1, 2] - al_i_minus_2 * ga_i
    if mu_i == 0.0:
        return 2

    al_i_minus_1 = (matrix[1, 3] - be_i_minus_2 * ga_i) / mu_i
    be_i_minus_1 = matrix[1, 4] / mu_i

    matrix[1, 0] = 0.0
    matrix[1, 1] = ga_i
    matrix[1, 2] = mu_i
    matrix[1, 3] = al_i_minus_1
    matrix[1, 4] = be_i_minus_1

    # Central rows
    # NOTE: this loop is manually unrolled by a factor of 2
    split_row_index = 2 + 2 * ((num_rows - 4) // 2)
    for row_index in range(2, split_row_index, 2):
        e_i = matrix[row_index, 0]
        ga_i = matrix[row_index, 1] - al_i_minus_2 * e_i
        mu_i = matrix[row_index, 2] - be_i_minus_2 * e_i - al_i_minus_1 * ga_i
        if mu_i == 0.0:
            return row_index + 1

        al_i_minus_2 = al_i_minus_1
        al_i_minus_1 = (matrix[row_index, 3] - be_i_minus_1 * ga_i) / mu_i

        be_i_minus_2 = be_i_minus_1
        be_i_minus_1 = matrix[row_index, 4] / mu_i

        matrix[row_index, 1] = ga_i
        matrix[row_index, 2] = mu_i
        matrix[row_index, 3] = al_i_minus_1
        matrix[row_index, 4] = be_i_minus_1

        e_i = matrix[row_index + 1, 0]
        ga_i = matrix[row_index + 1, 1] - al_i_minus_2 * e_i
        mu_i = matrix[row_index + 1, 2] - be_i_minus_2 * e_i - al_i_minus_1 * ga_i
        if mu_i == 0.0:
            return row_index + 2

        al_i_minus_2 = al_i_minus_1
        al_i_minus_1 = (matrix[row_index + 1, 3] - be_i_minus_1 * ga_i) / mu_i

        be_i_minus_2 = be_i_minus_1
        be_i_minus_1 = matrix[row_index + 1, 4] / mu_i

        matrix[row_index + 1, 1] = ga_i
        matrix[row_index + 1, 2] = mu_i
        matrix[row_index + 1, 3] = al_i_minus_1
        matrix[row_index + 1, 4] = be_i_minus_1

    for row_index in range(split_row_index, num_rows - 2):
        e_i = matrix[row_index, 0]
        ga_i = matrix[row_index, 1] - al_i_minus_2 * e_i
        mu_i = matrix[row_index, 2] - be_i_minus_2 * e_i - al_i_minus_1 * ga_i
        if mu_i == 0.0:
            return row_index + 1

        al_i_minus_2 = al_i_minus_1
        al_i_minus_1 = (matrix[row_index, 3] - be_i_minus_1 * ga_i) / mu_i

        be_i_minus_2 = be_i_minus_1
        be_i_minus_1 = matrix[row_index, 4] / mu_i

        matrix[row_index, 1] = ga_i
        matrix[row_index, 2] = mu_i
        matrix[row_index, 3] = al_i_minus_1
        matrix[row_index, 4] = be_i_minus_1

    # second last row
    e_i = matrix[num_rows - 2, 0]
    ga_i = matrix[num_rows - 2, 1] - al_i_minus_2 * e_i
    mu_i = matrix[num_rows - 2, 2] - be_i_minus_2 * e_i - al_i_minus_1 * ga_i
    if mu_i == 0.0:
        return num_rows - 1

    al_i_minus_2 = al_i_minus_1
    al_i_minus_1 = (matrix[num_rows - 2, 3] - be_i_minus_1 * ga_i) / mu_i

    be_i_minus_2 = be_i_minus_1

    matrix[num_rows - 2, 1] = ga_i
    matrix[num_rows - 2, 2] = mu_i
    matrix[num_rows - 2, 3] = al_i_minus_1
    matrix[num_rows - 2, 4] = 0.0

    # last row
    e_i = matrix[num_rows - 1, 0]
    ga_i = matrix[num_rows - 1, 1] - al_i_minus_2 * e_i
    mu_i = matrix[num_rows - 1, 2] - be_i_minus_2 * e_i - al_i_minus_1 * ga_i
    if mu_i == 0.0:
        return num_rows

    matrix[num_rows - 1, 1] = ga_i
    matrix[num_rows - 1, 2] = mu_i
    matrix[num_rows - 1, 3] = 0.0
    matrix[num_rows - 1, 4] = 0.0

    return 0


@jit(
    "Tuple((float64, float64))(float64[:, ::1])",
    nopython=True,
    cache=True,
)
def ptrans1_slogdet(factorization: NDArray[np.float64]) -> Tuple[float, float]:
    """
    Computes the sign and the natural logarithm of the determinant of a pentadiagonal
    matrix using the factors obtained from the PTRANS-I algorithm.

    It is simple given as the sum of the natural logarithms of the ``mu``-factors in the
    factorised matrix (after sign correction).

    Parameters
    ----------
    factorization : :obj:`numpy.ndarray` of shape (m, 5) and dtype ``numpy.float64``
        The factorisation of the pentadiagonal matrix to compute the determinant of.
        It has to be ordered in row-major format (C-order).
    is_factorized : :obj:`bool`
        Whether the matrix is already factorised (``True``) or not (``False``).

    Returns
    -------
    sign : :obj:`float`
        The sign of the determinant of the matrix.
    logabsdet : :obj:`float`
        The natural logarithm of the absolute determinant of the matrix.

    If the determinant is zero, then sign will be ``0.0`` and ``logabsdet`` will be
    ``-inf``. In all cases, the determinant is equal to ``sign * np.exp(logabsdet)``.

    """

    mus = np.ascontiguousarray(factorization[:, 2])

    # for an even number of negative ``mu``-factors, the sign is positive
    sign = 1.0 if np.count_nonzero(mus < 0) % 2 == 0 else -1.0

    # the log determinant is the sum of the natural logarithms of the absolute
    # ``mu``-factors
    logabsdet = np.log(np.abs(mus)).sum()

    # if there was an exact zero ``mu``-factor (resulting in a negative infinite
    # log determinant), the sign has to be set to zero according to the NumPy
    # convention
    if np.isneginf(logabsdet):
        return 0.0, -np.inf

    return sign, logabsdet


def ptrans1_symmetric_inverse_central_penta_bands(
    factorization: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Computes the central pentadiagonal part of the inverse of a SYMMETRIC pentadiagonal
    matrix using its factors obtained from the PTRANS-I algorithm.

    Rather than forming the full inverse (which is dense), only the part with the same
    sparsity pattern as the original matrix is computed, which is way more efficient.

    Parameters
    ----------
    factorization : :obj:`numpy.ndarray` of shape (m, 5) and dtype ``numpy.float64``
        The factorisation of the pentadiagonal matrix to compute the inverse of.
        It has to be ordered in row-major format (C-order).

    Returns
    -------
    inverse_penta_part : :obj:`numpy.ndarray` of shape (m, 5) and dtype ``numpy.float64``
        The central part pentadiagonal part of the inverse of the original matrix.

    """  # noqa: E501

    num_rows = factorization.shape[0]
    inverse_penta_part = np.zeros(shape=(num_rows, 5), dtype=np.float64)
    mu_values = np.ascontiguousarray(factorization[:, 2])

    # this algorithms starts from the lower right and moves its way up to the upper left
    # corner of the matrix
    # for this, the diagonal matrix W = diag(1 / mu), the strictly upper triangular
    # matrix T = I - U are required

    # last row
    # NOTE: the current matrix view to work on is of shape (3, 5)
    # NOTE: rows are indicated by letters a, b, and c because only 3 rows need to be
    #       stored in memory at the same time
    # NOTE: columns are indicated by numbers -2, -1, +0, +1, and +2 as an offset of the
    #       main diagonal (m is minus, p is plus)

    # last row
    w_c_c = 1.0 / mu_values[num_rows - 1]
    t_b_p1 = -factorization[num_rows - 2, 3]

    x_c_c = w_c_c
    x_b_p1 = w_c_c * t_b_p1

    inverse_penta_part[num_rows - 1, 4] = 0.0
    inverse_penta_part[num_rows - 1, 3] = 0.0
    inverse_penta_part[num_rows - 1, 2] = x_c_c
    inverse_penta_part[num_rows - 1, 1] = x_b_p1
    # the first column cannot be computed yet

    # second last row
    w_b_b = 1.0 / mu_values[num_rows - 2]
    x_b_b = w_b_b + t_b_p1 * x_b_p1

    inverse_penta_part[num_rows - 2, 4] = 0.0
    inverse_penta_part[num_rows - 2, 3] = x_b_p1
    inverse_penta_part[num_rows - 2, 2] = x_b_b

    # central rows
    for row_index in range(num_rows - 3, 0, -1):
        t_a_p2 = -factorization[row_index, 4]
        t_a_p1 = -factorization[row_index, 3]
        w_a_a = 1.0 / mu_values[row_index]

        x_a_p2 = t_a_p1 * x_b_p1 + t_a_p2 * x_c_c
        x_a_p1 = t_a_p1 * x_b_b + t_a_p2 * x_b_p1
        x_a_a = w_a_a + t_a_p1 * x_a_p1 + t_a_p2 * x_a_p2

        inverse_penta_part[row_index + 2, 0] = x_a_p2
        inverse_penta_part[row_index + 1, 1] = x_a_p1
        inverse_penta_part[row_index, 4] = x_a_p2
        inverse_penta_part[row_index, 3] = x_a_p1
        inverse_penta_part[row_index, 2] = x_a_a

        x_c_c = x_b_b
        x_b_p1 = x_a_p1
        x_b_b = x_a_a

    # first row
    t_a_p2 = -factorization[0, 4]
    t_a_p1 = -factorization[0, 3]
    w_a_a = 1.0 / mu_values[0]

    x_a_p2 = t_a_p1 * x_b_p1 + t_a_p2 * x_c_c
    x_a_p1 = t_a_p1 * x_b_b + t_a_p2 * x_b_p1
    x_a_a = w_a_a + t_a_p1 * x_a_p1 + t_a_p2 * x_a_p2

    inverse_penta_part[2, 0] = x_a_p2
    inverse_penta_part[1, 1] = x_a_p1
    inverse_penta_part[0, 4] = x_a_p2
    inverse_penta_part[0, 3] = x_a_p1
    inverse_penta_part[0, 2] = x_a_a
    inverse_penta_part[0, 1] = 0.0
    inverse_penta_part[0, 0] = 0.0

    return inverse_penta_part


@jit(
    "void(float64[:, ::1], float64[::1])",
    nopython=True,
    cache=True,
)
def ptrans1_solve_single_rhs(
    factorization: NDArray[np.float64],
    rhs: NDArray[np.float64],
) -> None:
    """
    Solves a pentadiagonal linear system using the factors obtained from the PTRANS-I
    algorithm when only a single right-hand side is given.

    Parameters
    ----------
    factorization : :obj:`numpy.ndarray` of shape (m, 5) and dtype ``numpy.float64``
        The factorisation of the pentadiagonal matrix to solve the linear system with.
        It has to be ordered in row-major format (C-order).
    rhs : :obj:`numpy.ndarray` of shape (m,) and dtype ``numpy.float64``
        The right-hand side of the linear system that will be overwritten with the
        solution.

    """

    num_rows = factorization.shape[0]

    # --- Transformation ---

    # the right hand side is transformed into the vector ``z`` by the lower triangular
    # matrix

    # first row
    z_i_minus_2 = rhs[0] / factorization[0, 2]
    rhs[0] = z_i_minus_2

    # second row
    z_i_minus_1 = (rhs[1] - factorization[1, 1] * z_i_minus_2) / factorization[1, 2]
    rhs[1] = z_i_minus_1

    # central rows
    # NOTE: this loop is manually unrolled by a factor of 2
    split_row_index = 2 + 2 * ((num_rows - 3) // 2)
    for row_index in range(2, split_row_index, 2):
        z_i = (
            rhs[row_index]
            - factorization[row_index, 0] * z_i_minus_2
            - factorization[row_index, 1] * z_i_minus_1
        ) / factorization[row_index, 2]
        z_i_minus_2 = z_i_minus_1
        z_i_minus_1 = z_i

        rhs[row_index] = z_i

        z_i = (
            rhs[row_index + 1]
            - factorization[row_index + 1, 0] * z_i_minus_2
            - factorization[row_index + 1, 1] * z_i_minus_1
        ) / factorization[row_index + 1, 2]
        z_i_minus_2 = z_i_minus_1
        z_i_minus_1 = z_i

        rhs[row_index + 1] = z_i

    for row_index in range(split_row_index, num_rows - 1):
        z_i = (
            rhs[row_index]
            - factorization[row_index, 0] * z_i_minus_2
            - factorization[row_index, 1] * z_i_minus_1
        ) / factorization[row_index, 2]
        z_i_minus_2 = z_i_minus_1
        z_i_minus_1 = z_i

        rhs[row_index] = z_i

    # last row
    # NOTE: this is done separately to avoid the memoized z-values to be overwritten
    #       because the backward substitution will now loop over them in reverse order
    z_i = (
        rhs[num_rows - 1]
        - factorization[num_rows - 1, 0] * z_i_minus_2
        - factorization[num_rows - 1, 1] * z_i_minus_1
    ) / factorization[num_rows - 1, 2]

    rhs[num_rows - 1] = z_i

    # --- Backward substitution ---

    z_i_minus_1 -= factorization[num_rows - 2, 3] * z_i
    rhs[num_rows - 2] = z_i_minus_1

    # central rows
    # NOTE: this loop is also manually unrolled by a factor of 2
    split_row_index = num_rows - 3 - 2 * ((num_rows - 3) // 2)
    for row_index in range(num_rows - 3, split_row_index, -2):
        rhs[row_index] -= (
            factorization[row_index, 3] * z_i_minus_1
            + factorization[row_index, 4] * z_i
        )
        z_i = z_i_minus_1
        z_i_minus_1 = rhs[row_index]

        rhs[row_index - 1] -= (
            factorization[row_index - 1, 3] * z_i_minus_1
            + factorization[row_index - 1, 4] * z_i
        )
        z_i = z_i_minus_1
        z_i_minus_1 = rhs[row_index - 1]

    for row_index in range(split_row_index, 0, -1):
        rhs[row_index] -= (
            factorization[row_index, 3] * z_i_minus_1
            + factorization[row_index, 4] * z_i
        )

        z_i = z_i_minus_1
        z_i_minus_1 = rhs[row_index]

    # last row
    rhs[0] -= factorization[0, 3] * z_i_minus_1 + factorization[0, 4] * z_i

    return


if __name__ == "__main__":

    from scipy.linalg import solve_triangular

    np.set_printoptions(precision=6, suppress=True)

    np.random.seed(0)
    test = np.empty(shape=(7, 5))
    vect = np.random.rand(test.shape[0] - 2)
    test[2::, 0] = vect.copy()
    test[0:-2, 4] = vect.copy()
    vect = np.random.rand(test.shape[0] - 1)
    test[1::, 1] = vect.copy()
    test[0:-1, 3] = vect.copy()
    test[::, 2] = 2.0 + np.random.rand(test.shape[0])
    test_dense = np.zeros((test.shape[0], test.shape[0]))

    test_dense += np.diag(test[2:, 0], k=-2)
    test_dense += np.diag(test[1:, 1], k=-1)
    test_dense += np.diag(test[:, 2])
    test_dense += np.diag(test[:-1, 3], k=1)
    test_dense += np.diag(test[:-2, 4], k=2)

    print(np.round(test, 2))
    print(np.round(test_dense, 2))

    print(np.linalg.slogdet(test_dense))

    fact = test.copy()
    ptrans1_factorize(fact)
    print(ptrans1_slogdet(fact), end="\n\n")

    l_mat = np.zeros((test.shape[0], test.shape[0]))
    l_mat += np.diag(fact[2:, 0], k=-2)
    l_mat += np.diag(fact[1:, 1], k=-1)
    l_mat += np.diag(fact[:, 2])

    u_mat = np.zeros((test.shape[0], test.shape[0]))
    u_mat += np.diag(np.ones(test.shape[0]), k=0)
    u_mat += np.diag(fact[:-1, 3], k=1)
    u_mat += np.diag(fact[:-2, 4], k=2)

    assert np.allclose(test_dense, l_mat @ u_mat)

    b_vect = np.random.rand(test.shape[0])
    x_penta = b_vect.copy()
    x_dense = np.linalg.solve(test_dense, b_vect)

    print("L-solve:\n", solve_triangular(l_mat, b_vect, lower=True), end="\n\n")

    ptrans1_solve_single_rhs(fact, x_penta)

    print(x_dense, end="\n\n")
    print(x_penta)

    assert np.allclose(x_dense, x_penta)

    test_inv = np.linalg.inv(test_dense)
    uinv_dinv = solve_triangular(
        u_mat,
        np.diag(1.0 / fact[::, 2]),
        lower=False,
    )
    u_minus_eye = np.eye(test.shape[0]) - u_mat
    print(u_minus_eye @ test_inv, end="\n----\n")
    eye_minus_l = np.eye(test.shape[0]) - l_mat / fact[::, 2][np.newaxis, ::]
    test_inv_rec = test_inv @ eye_minus_l

    test_penta_inv = ptrans1_symmetric_inverse_central_penta_bands(fact)

    print("\n----\n")
    print(test_inv, end="\n----\n")
    # print(uinv_dinv, end="\n----\n")
    # print(1.0 / fact[::, 2], end="\n----\n")
    # print(eye_minus_l, end="\n----\n")
    # print(test_inv_rec, end="\n----\n")
    print(test_penta_inv)

    raise KeyboardInterrupt()

    print("\n\nTest inverse diagonal")
    print(np.diag(np.linalg.inv(test_dense)), end="\n\n")
    print(
        np.diag(
            solve_triangular(
                u_mat,
                np.diag(1.0 / fact[::, 2]),
                lower=False,
            )
        ),
        end="\n\n",
    )

    from time import perf_counter_ns

    fact = test.copy()
    b_vect = np.random.rand(test.shape[0])
    start_time = perf_counter_ns()
    ptrans1_factorize(fact)
    print(ptrans1_slogdet(fact))
    end_time = perf_counter_ns()

    print(f"Factorization lapsed time: {(end_time - start_time) / 1e3:.0f} mus")

    start_time = perf_counter_ns()
    ptrans1_solve_single_rhs(fact, b_vect)
    end_time = perf_counter_ns()

    print(f"Solve elapsed time: {(end_time - start_time) / 1e3:.0f} mus")
