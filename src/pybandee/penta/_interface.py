"""
Module :mod:`penta._interface`

This module implements implements the user-facing interface for the pentadiagonal matrix
functions. It provides the following functions:

- factorisation
- solving
- computing the log determinant

"""

# === Imports ===

import numpy as np

from .._utils import (
    LinearSystemSolution,
    MatrixFactorization,
    RealNumericArrayLike,
    get_validated_real_numeric_1d_array_like,
)
from ._penta_utils import (
    PentaDiagonalMatrixFormat,
    convert_to_validated_penta,
    is_data_linked,
)
from ._ptrans1 import ptrans1_factorize, ptrans1_solve_single_rhs

# === Functions ===


# TODO: the smallest possible pentadiagonal matrix is 3x3 and not 5x5
def penta_factorize(
    matrix: RealNumericArrayLike,
    matrix_format: PentaDiagonalMatrixFormat = "penta_row",
    overwrite: bool = False,
    check_finite: bool = True,
) -> MatrixFactorization:
    """
    Factorises a pentadiagonal matrix using the PTRANS-I algorithm, i.e., the matrix
    ``A`` is factorised into ``L`` and ``U`` such that ``A = L @ U``.
    It is only recommended for well-conditioned matrices since no pivoting is performed.

    Parameters
    ----------
    matrix : ArrayLike of shape (m, n)
        The matrix to factorise whose storage format is given by ``matrix_format``.
        To be truly pentadiagonal, it has to have at least 5 rows and columns when
        converted to the dense format.
        Its data type is internally promoted to ``numpy.float64``.
    matrix_format : {``"penta_row"``, ``"lapack_general_banded"``, ``"dense"``}, default=``"penta_row"``
        The storage format of the input matrix. Any format other than ``"penta_row"``
        will require a conversion since the underlying algorithms are tailored for the
        pentadiagonal row-major format. The shape of ``matrix`` is given in brackets
        where ``p`` is the number of rows/columns of the dense equivalent of ``matrix``.

        - ``"penta_row"``: the pentadiagonal row-major format (C-order) which requires
            no conversion and is thus the most efficient format (``shape=(p, 5)``)
        - ``"lapack_general_banded"``: the LAPACK general banded matrix format as
            expected by the function :func:`scipy.linalg.solve_banded`
            (``shape=(5, p)``)
        - ``"dense"``: the dense matrix format (``shape=(p, p)``)

        Please refer to the Notes section for details.

    overwrite : :obj:`bool`, default=``False``
        Whether to overwrite ``matrix`` with the factorised matrix (``True``) or not
        (``False``).
        Note that this is only relevant if ``matrix_format="penta_row"`` and no other
        type conversion is performed. Applicability is checked on the fly making sure
        that overwriting is still performed despite ``overwrite=False`` in case the link
        between the input and the converted matrix is broken.
    check_finite : :obj:`bool`, default=``True``
        Whether to check that the input matrix contains only finite values (``True``) or
        not (``False``).
        While disabling will speed up the computation, the results will be corrupted
        by the presence of ``nan`` or ``inf`` values. Yet, the algorithm will always
        terminate since its execution steps are not relying on previous computations.

    Returns
    -------
    A :obj:`namedtuple` with the following fields:

    factorization : :obj:`numpy.ndarray` of shape (p, 5) and dtype ``numpy.float64``
        The factorised matrix. Please refer to the Notes section for details.
    method : :obj:`str`
        The method used for factorisation.

    Raises
    ------
    TypeError
        If ``matrix`` or ``matrix_format`` are not of expected type.
    ValueError
        If ``matrix`` does contains non-finite values and ``check_finite=True``.
    ValueError
        If ``matrix`` cannot be converted to the ``matrix_format="penta_row"`` format.
    LinAlgError
        If ``matrix`` cannot be factorised.

    Notes
    -----
    The ``(p, p) = (7, 7)`` dense square pentadiagonal matrix
    (``"matrix_format="penta_row"``) with the zero entries marked by ``*``

    ```
    [[   a00 a01 a02  *   *   *   *    ]
     [   a10 a11 a12 a13  *   *   *    ]
     [   a20 a21 a22 a23 a24  *   *    ]
     [    *  a31 a32 a33 a34 a35  *    ]
     [    *   *  a42 a43 a44 a45 a46   ]
     [    *   *   *  a53 a54 a55 a56   ]
     [    *   *   *   *  a64 a65 a66   ]]
    ```

    can also be stored in either the LAPACK general banded format
    (``"matrix_format="lapack_general_banded"``):

    ```
    [[    *   *  a02 a13 a24 a35 a46   ]   # super-diagonal 2
     [    *  a01 a12 a23 a34 a45 a56   ]   # super-diagonal 1
     [   a00 a11 a22 a33 a44 a55 a66   ]   # main diagonal
     [   a10 a21 a32 a43 a54 a65  *    ]   # sub-diagonal 1
     [   a20 a31 a42 a53 a64  *   *    ]]  # sub-diagonal 2
    ```

    or the pentadiagonal row-major format (``"matrix_format="penta_row"``):

    ```
    #    (1) (2) (3) (4) (5)
    [[    *   *  a00 a01 a02   ]   # row 0
     [    *  a10 a11 a12 a13   ]   # row 1
     [   a20 a21 a22 a23 a24   ]   # ...
     [   a31 a32 a33 a34 a35   ]
     [   a42 a43 a44 a45 a46   ]
     [   a53 a54 a55 a56  *    ]   # row p-2
     [   a64 a65 a66  *   *    ]]  # row p-1

    # (1) sub-diagonal 2
    # (2) sub-diagonal 1
    # (3) main diagonal
    # (4) super-diagonal 1
    # (5) super-diagonal 2
    ```

    Starting from this representation, the factorised matrix ``A = L @ U`` is returned
    in the same format as

    ```
    [[    *   *  l00 u01 u02   ]
     [    *  l10 l11 u12 u13   ]
     [   l20 l21 l22 u23 u24   ]
     [   l31 l32 l33 u34 u35   ]
     [   l42 l43 l44 u45 u46   ]
     [   l53 l54 l55 u56  *    ]
     [   l64 l65 l66  *   *    ]]
    ```

    for the lower triangular matrix ``L``

    ```
    [[   l00  *   *   *   *   *   *    ]
     [   l10 l11  *   *   *   *   *    ]
     [   l20 l21 l22  *   *   *   *    ]
     [    *  l31 l32 l33  *   *   *    ]
     [    *   *  l42 l43 l44  *   *    ]
     [    *   *   *  l53 l54 l55  *    ]
     [    *   *   *   *  l64 l65 l66   ]]
    ```

    and the unit (main diagonal of ones) upper triangular matrix ``U``

    ```
    [[    1  u01 u02  *   *   *   *    ]
     [    *   1  u12 u13  *   *   *    ]
     [    *   *   1  u23 u24  *   *    ]
     [    *   *   *   1  u34 u35  *    ]
     [    *   *   *   *   1  u45 u46   ]
     [    *   *   *   *   *   1  u56   ]
     [    *   *   *   *   *   *   1    ]]
    ```

    """  # noqa: E501

    # --- Input Validation and Conversion ---

    # NOTE: ``matrix`` or its copy will be overwritten with the factorisation, thus the
    #       variable name is chosen to reflect this
    factorization, overwrite = convert_to_validated_penta(
        matrix=matrix,
        matrix_name="matrix",
        matrix_format=matrix_format,
        overwrite=overwrite,
        check_finite=check_finite,
    )

    # --- Factorisation ---

    if not overwrite:
        factorization = factorization.copy()

    info = ptrans1_factorize(matrix=factorization)

    if info == 0:
        return MatrixFactorization(
            factorization=factorization,
            method="prtrans1",
        )

    if info > 0:
        raise np.linalg.LinAlgError(
            f"Factorisation failed due to a zero-division at row index {info - 1}. "
            f"The matrix is ill-conditioned for the PTRANS-I algorithm."
        )

    raise AssertionError(f"Unexpected error during factorisation, got {info=}")


# TODO: add matrix ``rhs`` support
def penta_solve(
    lhs_matrix: RealNumericArrayLike,
    rhs: RealNumericArrayLike,
    matrix_format: PentaDiagonalMatrixFormat = "penta_row",
    lhs_overwrite: bool = False,
    rhs_overwrite: bool = False,
    check_finite: bool = True,
) -> LinearSystemSolution:
    """
    Solves a linear system of equations with a pentadiagonal matrix using the PTRANS-I
    algorithm, i.e., the system ``A @ x = b`` is solved for ``x``.
    It is only recommended for well-conditioned matrices since no pivoting is performed.

    Parameters
    ----------
    lhs_matrix : ArrayLike of shape (m, n)
        The left-hand side matrix ``A`` of the linear system whose storage format is
        given by ``matrix_format``. To be truly pentadiagonal, it has to have at least
        5 rows and columns when converted to the dense format. Its data type is
        internally promoted to ``numpy.float64``.
    rhs : ArrayLike of shape (m,)
        The right-hand side vector ``b`` of the linear system. Its data type is
        internally promoted to ``numpy.float64``.
    matrix_format : {``"penta_row"``, ``"lapack_general_banded"``, ``"dense"``}, default=``"penta_row"``
        The storage format of ``lhs_matrix``. Any format other than ``"penta_row"`` will
        require a conversion since the underlying algorithms are tailored for the
        pentadiagonal row-major format. The shape of ``lhs_matrix`` is given in brackets
        where ``p`` is the number of rows/columns of the dense equivalent of
        ``lhs_matrix``.

        - ``"penta_row"``: the pentadiagonal row-major format (C-order) which requires
            no conversion and is thus the most efficient format (``shape=(p, 5)``)
        - ``"lapack_general_banded"``: the LAPACK general banded matrix format as
            expected by the function :func:`scipy.linalg.solve_banded`
            (``shape=(5, p)``)
        - ``"dense"``: the dense matrix format (``shape=(p, p)``)

        Please refer to the Notes section for details.

    lhs_overwrite, rhs_overwrite : :obj:`bool`, default=``False``
        Whether to overwrite ``lhs_matrix`` (``True``) or ``rhs`` (``True``) with the
        factorised matrix and solution, respectively, or not (``False``).
        These are only relevant if no type conversion is performed. Applicability is
        checked on the fly making sure that overwriting is still performed despite
        ``lhs_overwrite=False`` or ``rhs_overwrite=False`` in case the link between the
        input and the converted data is broken.
    check_finite : :obj:`bool`, default=``True``
        Whether to check that the input contains only finite values (``True``) or not
        (``False``).
        While disabling will speed up the computation, the results will be corrupted
        by the presence of ``nan`` or ``inf`` values. Yet, the algorithm will always
        terminate since its execution steps are not relying on previous computations.

    Returns
    -------
    A namedtuple with the following fields:

    solution : :obj:`numpy.ndarray` of shape (m,) and dtype ``numpy.float64``
        The solution to the linear system.
    factorization : :obj:`numpy.ndarray` of shape (p, 5) and dtype ``numpy.float64``
        The factorised matrix. Please refer to the Notes section for details.
    method : :obj:`str`
        The method used for factorisation.

    Raises
    ------
    TypeError
        If ``lhs_matrix``, ``rhs``, or ``matrix_format`` are not of expected type.
    ValueError
        If ``lhs_matrix`` or ``rhs`` do not contain finite values and
        ``check_finite=True``.
    ValueError
        If ``lhs_matrix`` cannot be converted to the ``matrix_format="penta_row"``
        format.
    LinAlgError
        If the linear system cannot be solved.

    """  # noqa: E501

    # --- Input Validation and Conversion ---

    # NOTE: ``lhs`` or its copy will be overwritten with the factorisation, thus the
    #       variable name is chosen to reflect this
    lhs_factorization, lhs_overwrite = convert_to_validated_penta(
        matrix=lhs_matrix,
        matrix_name="lhs_matrix",
        matrix_format=matrix_format,
        overwrite=lhs_overwrite,
        check_finite=check_finite,
    )

    # NOTE: ``rhs`` or its copy will be overwritten with the solution, thus the variable
    #       name is chosen to reflect this
    rhs_solution = get_validated_real_numeric_1d_array_like(
        value=rhs,
        name="rhs",
        min_size=lhs_factorization.shape[0],
        max_size=lhs_factorization.shape[0],
        enforce_finite=check_finite,
        output_dtype=np.float64,
    )

    if not rhs_overwrite:
        rhs_overwrite = not is_data_linked(
            array=rhs_solution,
            original=rhs,
        )

    # --- Factorisation and Solve ---

    # if required, copies are made
    if not lhs_overwrite:
        lhs_factorization = lhs_factorization.copy()

    if not rhs_overwrite:
        rhs_solution = rhs_solution.copy()

    info = ptrans1_factorize(matrix=lhs_factorization)

    if info == 0:
        ptrans1_solve_single_rhs(
            factorization=lhs_factorization,
            rhs=rhs_solution,
        )

        return LinearSystemSolution(
            solution=rhs_solution,
            factorization=lhs_factorization,
            method="ptrans1",
        )

    if info > 0:
        raise np.linalg.LinAlgError(
            f"Factorisation failed due to a zero-division at row index {-info - 1}. "
            f"The matrix is ill-conditioned for the PTRANS-I algorithm.\n"
            f"Solving step was not attempted."
        )

    raise AssertionError(f"Unexpected error during factorisation, got {info=}")


if __name__ == "__main__":
    from scipy.linalg import solve_banded

    np.random.seed(0)

    a = np.random.rand(5, 13)
    a_og = a.copy()
    b = np.random.rand(a.shape[1])

    print("LAPACK")
    x_lapack = solve_banded(
        l_and_u=(2, 2),
        ab=a,
        b=b,
    )

    print("PENTA")
    x_penta = penta_solve(
        lhs_matrix=a,
        matrix_format="lapack_general_banded",
        rhs=b,
        lhs_overwrite=True,
        rhs_overwrite=True,
    )

    assert np.array_equal(a, a_og)
    assert np.allclose(x_penta.solution, x_lapack)
    assert np.allclose(b, x_lapack)
