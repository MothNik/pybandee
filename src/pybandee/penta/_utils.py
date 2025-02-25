"""
Module :mod:`penta._utils`

This module implements implements utility functions to convert matrices to pentadiagonal
row-major format.

"""

# === Imports ===

from typing import Literal, Tuple

import numpy as np
from numpy.typing import NDArray

from pybandee._utils import (
    RealNumericArrayLike,
    get_validated_real_numeric_2d_array_like,
    is_data_linked,
    jit,
)

# === Typing ===

PentaDiagonalMatrixFormat = Literal[
    "penta_row",
    "lapack_general_banded",
    "dense",
]

# === Functions ===


def validate_penta(matrix: NDArray[np.float64]) -> None:
    """
    Validates a matrix in pentadiagonal row-major format.

    Parameters
    ----------
    matrix : :obj:`numpy.ndarray` of shape (n, 5)
        The matrix to validate.

    Raises
    ------
    ValueError
        If the matrix does not adhere to the pentadiagonal row-major format.

    """

    num_rows, num_cols = matrix.shape

    if num_cols != 5:
        raise ValueError(
            f"Expected pentadiagonal row-major matrix to have 5 columns but got matrix "
            f"of shape {matrix.shape}."
        )

    if num_rows < 5:
        raise ValueError(
            f"Expected pentadiagonal row-major matrix to have at least 5 rows but got "
            f"matrix of shape {matrix.shape}."
        )

    return


@jit(
    "Tuple((float64[:, ::1], int64))(float64[:, ::1])",
    nopython=True,
    cache=True,
)
def convert_lapack_general_banded_to_penta(
    matrix: NDArray[np.float64],
) -> Tuple[NDArray[np.float64], int]:
    """
    Converts a matrix in LAPACK general banded matrix format to the pentadiagonal
    row-major format (C-order).

    Parameters
    ----------
    matrix : :obj:`numpy.ndarray` of shape (5, n)
        The matrix to convert.

    Returns
    -------
    penta_matrix : :obj:`numpy.ndarray` of shape (n, 5)
        The pentadiagonal matrix in row-major format.
    info : :obj:`int`
        The return code of the conversion.
        - ``0``: successful conversion
        - ``-1``: wrong row count (expected ``5``)
        - ``-2``: wrong column count (expected ``>= 5``)

    """

    # --- Input Validation ---

    num_rows, num_cols = matrix.shape

    if num_rows != 5:
        return (
            np.empty(shape=(5, num_cols), dtype=np.float64),
            -1,
        )

    if num_cols < 5:
        return (
            np.empty(shape=(5, num_cols), dtype=np.float64),
            -2,
        )

    # --- Conversion ---

    penta_matrix = np.empty(shape=(num_cols, 5), dtype=np.float64)

    penta_matrix[0, 0:2] = 0.0
    penta_matrix[1, 0] = 0.0
    penta_matrix[num_cols - 2, 4:5] = 0.0
    penta_matrix[num_cols - 1, 3:5] = 0.0

    penta_matrix[2:, 0] = matrix[4, 0 : num_cols - 2]
    penta_matrix[1:, 1] = matrix[3, 0 : num_cols - 1]
    penta_matrix[:, 2] = matrix[2, ::]
    penta_matrix[0 : num_cols - 1, 3] = matrix[1, 1:]
    penta_matrix[0 : num_cols - 2, 4] = matrix[0, 2:]

    return (
        penta_matrix,
        0,
    )


def _raise_error_lapack_general_banded_to_penta(
    matrix_shape: Tuple[int, int],
    info: int,
) -> None:
    """
    Raises an error based on the return code of the conversion function
    :func:`convert_lapack_general_banded_to_penta`.

    Parameters
    ----------
    matrix_shape : (:obj:`int`, :obj:`int`)
        The shape of the matrix.
    info : :obj:`int`
        The return code of the conversion function.

    Raises
    ------
    ValueError
        If the conversion failed.

    """

    if info >= 0:
        return

    if info == -1:
        raise ValueError(
            f"Expected matrix in LAPACK general banded format to have exactly 5 rows "
            f"but got matrix of shape {matrix_shape}."
        )

    if info == -2:
        raise ValueError(
            f"Expected matrix in LAPACK general banded format to have at least 5 "
            f"columns but got matrix of shape {matrix_shape}."
        )

    raise AssertionError(
        f"Unexpected return code {info} for conversion function "
        f"'convert_lapack_general_banded_to_penta'."
    )


@jit(
    "Tuple((float64[:, ::1], int64))(float64[:, ::1])",
    nopython=True,
    cache=True,
)
def convert_dense_to_penta(
    matrix: NDArray[np.float64],
) -> Tuple[NDArray[np.float64], int]:
    """
    Converts a dense square pentadiagonal matrix to the pentadiagonal row-major format.

    Parameters
    ----------
    matrix : :obj:`numpy.ndarray` of shape (p, p)
        The matrix to convert.

    Returns
    -------
    penta_matrix : :obj:`numpy.ndarray` of shape (p, 5)
        The pentadiagonal matrix in row-major format.
    info : :obj:`int`
        The return code of the conversion.
        - ``0``: successful conversion
        - ``-1``: wrong row count (expected ``>= 5``)
        - ``-2``: wrong column count (expected ``>= 5``)
        - ``-3``: matrix is not square

    """

    # --- Input Validation ---

    num_rows, num_cols = matrix.shape

    if num_rows < 5:
        return (
            np.empty(shape=(5, num_cols), dtype=np.float64),
            -1,
        )

    if num_cols < 5:
        return (
            np.empty(shape=(5, num_cols), dtype=np.float64),
            -2,
        )

    if num_rows != num_cols:
        return (
            np.empty(shape=(5, num_cols), dtype=np.float64),
            -3,
        )

    # --- Conversion ---

    penta_matrix = np.empty(shape=(num_rows, 5), dtype=np.float64)

    penta_matrix[0, 0:2] = 0.0
    penta_matrix[1, 0] = 0.0
    penta_matrix[num_rows - 2, 4:5] = 0.0
    penta_matrix[num_rows - 1, 3:5] = 0.0

    penta_matrix[2:, 0] = np.diag(matrix, k=-2)
    penta_matrix[1:, 1] = np.diag(matrix, k=-1)
    penta_matrix[:, 2] = np.diag(matrix)
    penta_matrix[0 : num_rows - 1, 3] = np.diag(matrix, k=1)
    penta_matrix[0 : num_rows - 2, 4] = np.diag(matrix, k=2)

    return (
        penta_matrix,
        0,
    )


def _raise_error_dense_to_penta(
    matrix_shape: Tuple[int, int],
    info: int,
) -> None:
    """
    Raises an error based on the return code of the conversion function
    :func:`convert_dense_to_penta`.

    Parameters
    ----------
    matrix_shape : (:obj:`int`, :obj:`int`)
        The shape of the matrix.
    info : :obj:`int`
        The return code of the conversion function.

    Raises
    ------
    ValueError
        If the conversion failed.

    """

    if info >= 0:
        return

    if info == -1:
        raise ValueError(
            f"Expected dense matrix to have at least 5 rows but got matrix of shape "
            f"{matrix_shape}."
        )

    if info == -2:
        raise ValueError(
            f"Expected dense matrix to have at least 5 columns but got matrix of shape "
            f"{matrix_shape}."
        )

    if info == -3:
        raise ValueError(
            f"Expected dense matrix to be square but got matrix of shape "
            f"{matrix_shape}."
        )

    raise AssertionError(
        f"Unexpected return code {info} for conversion function "
        f"'convert_dense_to_penta'."
    )


def convert_to_validated_penta(
    matrix: RealNumericArrayLike,
    matrix_name: str,
    matrix_format: PentaDiagonalMatrixFormat,
    overwrite: bool,
    check_finite: bool,
) -> Tuple[NDArray[np.float64], bool]:
    """
    Validated a matrix and converts it to the pentadiagonal row-major format.

    Parameters
    ----------
    matrix : Array-like of shape (n, m) and dtype ``numpy.float64``
        The matrix to factorise whose storage format is given by ``matrix_format``.
        To be truly pentadiagonal, it has to have at least 5 rows and columns when
        converted to the dense format.
        Its data type is internally promoted to ``numpy.float64``.
    matrix_name : :obj:`str`
        The name of the matrix.
        This is used when raising errors.
    matrix_format : {``"penta_row"``, ``"lapack_general_banded"``, "``dense``"}
        The storage format of ``matrix``. Any format other than ``"penta_row"`` will
        require a conversion since the underlying algorithms are tailored for the
        pentadiagonal row-major format. The shape of ``matrix`` is given in brackets
        where ``p`` is the number of rows/columns of the dense equivalent of
        ``lhs_matrix``.

        - ``"penta_row"``: the pentadiagonal row-major format (C-order) which requires
            no conversion and is thus the most efficient format (``shape=(p, 5)``)
        - ``"lapack_general_banded"``: the LAPACK general banded matrix format as
            expected by the function :func:`scipy.linalg.solve_banded` (``shape=(5, p)``)
        - ``"dense"``: the dense matrix format (``shape=(p, p)``)

        Please refer to the Notes section for details.
    overwrite : :obj:`bool`
        Whether ``matrix`` should be overwritten with the factorised matrix
        (``True``) or not (``False`).
        See the Returns section for details.
    check_finite : :obj:`bool`, default=``True``
        Whether to check that the input matrix contains only finite values (``True``) or
        not (``False``).

    Returns
    -------
    penta_matrix : :obj:`numpy.ndarray` of shape (p, 5)
        The pentadiagonal matrix in row-major format.
    overwrite_updated : :obj:`bool`
        Whether ``matrix`` can be overwritten with the factorised matrix without
        violating the input requirements given by ``overwrite`` (``True``) or not
        (``False``).
        So, if ``overwrite=False`` and the conversion breaks the link between the input
        and output matrices, this will be set to ``True``.

    Raises
    ------
    ValueError
        If ``matrix_format`` is not one of the expected values.
    ValueError
        If the conversion failed.

    """  # noqa: E501

    # --- Input Validation ---

    converted_matrix = get_validated_real_numeric_2d_array_like(
        value=matrix,
        name=matrix_name,
        rows_min_num=None,  # NOTE: checks will be performed by the converter
        rows_max_num=None,
        columns_max_num=None,
        columns_min_num=None,
        enforce_finite=check_finite,
        output_dtype=np.float64,
    )

    if not isinstance(matrix_format, str):
        matrix_format_type_str = f"{type(matrix_format)}"[7:-1]
        raise TypeError(
            f"Expected 'matrix_format' to be of type 'str', but got "
            f"'{matrix_format_type_str}'."
        )

    # --- Conversion ---

    matrix_format = matrix_format.lower()  # type: ignore

    if matrix_format != "penta_row":
        try:
            converter, converter_error_checker = {
                "lapack_general_banded": (
                    convert_lapack_general_banded_to_penta,
                    _raise_error_lapack_general_banded_to_penta,
                ),
                "dense": (
                    convert_dense_to_penta,
                    _raise_error_dense_to_penta,
                ),
            }[matrix_format]

        except KeyError:
            raise ValueError(
                f"Expected 'matrix_format' to be one of 'penta_row', "
                f"'lapack_general_banded', or 'dense' but got '{matrix_format}'."
            )

        converted_matrix, info = converter(matrix=converted_matrix)
        converter_error_checker(matrix_shape=matrix.shape, info=info)  # type: ignore

    # if overwrite is disabled, but the original data is not linked to the converted
    # data, then the original data can be safely overwritten
    if not overwrite:
        overwrite = not is_data_linked(
            array=converted_matrix,
            original=matrix,
        )

    # NOTE: this validation covers both the "penta_row" format and any other format
    #       to be sure that the conversion functions work correctly
    validate_penta(matrix=converted_matrix)

    return converted_matrix, overwrite


if __name__ == "__main__":

    from _ptrans1 import ptrans1_factorize, ptrans1_solve_single_rhs
    from scipy.linalg import solve_banded

    np.random.seed(0)
    test = np.random.rand(5, 6)
    test[2, ::] += 2.0
    b_vect = np.random.rand(test.shape[1])
    print(test, end="\n\n")

    dense = np.zeros(shape=(test.shape[1], test.shape[1]), dtype=np.float64)
    dense += np.diag(test[4, :-2], k=-2)
    dense += np.diag(test[3, :-1], k=-1)
    dense += np.diag(test[2, ::])
    dense += np.diag(test[1, 1:], k=1)
    dense += np.diag(test[0, 2:], k=2)

    print(dense, end="\n\n")

    penta, info = convert_lapack_general_banded_to_penta(test)
    print(test, end="\n\n")
    print(penta)

    penta_from_dense, info = convert_dense_to_penta(dense)
    print(penta_from_dense)
    assert np.allclose(penta, penta_from_dense, equal_nan=True)

    x_scipy = solve_banded((2, 2), test, b_vect)
    fact = penta.copy()
    ptrans1_factorize(fact)
    x_penta = b_vect.copy()
    ptrans1_solve_single_rhs(fact, x_penta)

    print(x_scipy)
    print(x_penta)
    assert np.allclose(x_scipy, x_penta)
