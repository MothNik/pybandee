"""
Module :mod:`penta._ptrans1`

This module implements the PTRANS-I algorithm for the factorisation and solving of
pentadiagonal linear systems as described in [1]_.

It relies heavily on ``numba`` and all function can be called in other ``jit``-compiled
functions.

"""

# === Imports ===

from typing import Tuple

import numpy as np
from numpy.typing import NDArray

from pybandee._utils import jit

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
    [[    *           *          d_0          a_0       b_i      ]
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
    for row_index in range(2, num_rows - 2):
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
    "Tuple((float64, float64))(float64[:, ::1], bool)",
    nopython=True,
    cache=True,
)
def ptrans1_slogdet(
    matrix: NDArray[np.float64],
    is_factorized: bool,
) -> Tuple[float, float]:
    """
    Computes the sign and the natural logarithm of the determinant of a pentadiagonal
    matrix using the factors obtained from the PTRANS-I algorithm.

    It is simple given as the sum of the natural logarithms of the ``mu``-factors in the
    factorised matrix (after sign correction).

    Parameters
    ----------
    matrix : :obj:`numpy.ndarray` of shape (m, 5) and dtype ``numpy.float64``
        The pentadiagonal matrix or its factorisation to compute the log determinant
        from.
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

    if not is_factorized:
        ptrans1_factorize(matrix=matrix)

    mus = np.ascontiguousarray(matrix[:, 2])

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


def ptrans1_inverse_main_diagonal(
    matrix: NDArray[np.float64],
    is_factorized: bool,
) -> NDArray[np.float64]:
    """
    Computes the main diagonal of the inverse of a pentadiagonal matrix using the
    factors obtained from the PTRANS-I algorithm.
    This turns out to be very simple as the main diagonal of the inverse is just the
    reciprocal of the ``mu``-factors in the factorised matrix.

    Parameters
    ----------
    matrix : :obj:`numpy.ndarray` of shape (m, 5) and dtype ``numpy.float64``
        The pentadiagonal matrix or its factorisation to compute the main diagonal of
        the inverse from.
        It has to be ordered in row-major format (C-order).
    is_factorized : :obj:`bool`
        Whether the matrix is already factorised (``True``) or not (``False``).

    Returns
    -------
    inverse_main_diagonal : :obj:`numpy.ndarray` of shape (m,) and dtype ``numpy.float64``
        The main diagonal of the inverse of the matrix.

    References
    ----------
    .. [1] Fasllija E., Parallel computation of the diagonal of the inverse of a sparse
           matrix, 2017, pp. 8-9

    """

    if not is_factorized:
        ptrans1_factorize(matrix=matrix)

    return np.reciprocal(np.ascontiguousarray(matrix[:, 2]))


test = np.random.rand(3, 3)
test = np.tril(test)
print(1.0 / np.diag(test))

print(np.diag(np.linalg.inv(test)))

raise KeyboardInterrupt()

test = np.random.rand(100, 5)
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
print(ptrans1_slogdet(fact, True))
print(ptrans1_slogdet(test.copy(), False))

print(np.diag(np.linalg.inv(test_dense)))
print(ptrans1_inverse_main_diagonal(fact, True))
print(np.allclose(ptrans1_inverse_main_diagonal(fact, True), np.diag(np.linalg.inv(test_dense))))

raise KeyboardInterrupt()

from time import perf_counter_ns

fact = test.copy()
start_time = perf_counter_ns()
ptrans1_factorize(fact)
print(ptrans1_slogdet(fact, True))
end_time = perf_counter_ns()

print(f"Elapsed time: {(end_time - start_time) / 1e3:.0f} mus")


