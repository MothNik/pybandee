"""
Mod :mod:`_utils._models`

This module provides models used across the ``pybandee`` package.

"""

# === Imports ===

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import bandwidth

from ._custom_types import Integer, RealNumericArrayLike
from ._miscellaneous import is_data_linked, warn_verbose
from ._validate import get_validated_integer, get_validated_real_numeric_2d_array_like

# === Typing ===

LAndUCounts = Union[
    Integer,
    Tuple[Integer, Integer],
    List[Integer],
    None,
]

# === Auxiliary Functions ===


def get_validated_l_and_u(
    matrix: NDArray[np.float64],
    matrix_dense_shape: Tuple[int, int],
    l_and_u: LAndUCounts = None,
) -> tuple[int, int]:
    """
    Validates the number of sub-diagonals and super-diagonals in a banded matrix.

    Parameters
    ----------
    matrix : :obj:`numpy.ndarray` of shape (m, n) and dtype ``numpy.float64``
        The matrix to be converted to a banded matrix.
    matrix_dense_shape : (:obj:`int`, :obj:`int`)
        The shape of the dense version of ``matrix``.
    l_and_u : :obj:`int` or (:obj:`int`, :obj:`int`) or [:obj:`int`] or ``None``, default=``None``
        The number of sub-diagonals and super-diagonals in the matrix, respectively.
        Both have to be non-negative integers ``>= 0``.
        If only a single integer is provided, it is assumed that the matrix has the
        same number of sub-diagonals and super-diagonals.
        If ``None``, the function will determine the bandwidth automatically.

    Returns
    -------
    l_and_u : (:obj:`int`, :obj:`int`)
        The number of sub-diagonals and super-diagonals in the matrix.

    Raises
    ------
    TypeError
        If ``l_and_u`` is not of expected type.
    ValueError
        If the number of sub-diagonals and super-diagonals is not valid.

    """

    # --- Input Validation ---

    if l_and_u is None:
        return bandwidth(matrix)

    l_and_u_internal = l_and_u
    if not isinstance(l_and_u_internal, (tuple, list)):
        l_and_u_internal = (l_and_u_internal, l_and_u_internal)

    if len(l_and_u_internal) != 2:
        raise ValueError(
            f"Expected the number of sub-diagonals and super-diagonals to be "
            f"specified as one or two integers, but got {len(l_and_u_internal)} values."
        )

    l_and_u_internal = tuple(  # type: ignore
        get_validated_integer(
            value=value,
            name=f"l_and_u[{index}] (={name})",
            min_value=0,
            min_inclusive=True,
        )
        for index, (value, name) in enumerate(zip(l_and_u_internal, ("l", "u")))
    )

    # --- Check against the Matrix Shape ---

    # column number check
    num_required_columns = sum(l_and_u_internal) + 1

    if matrix_dense_shape != num_required_columns:
        raise ValueError(
            f"Expected the dense version of matrix to have {num_required_columns} "
            f"columns because it has {l_and_u_internal[0]} sub-diagonals, 1 main "
            f"diagonal, and {l_and_u_internal[1]} super-diagonals, but got a matrix "
            f"with dense shape {matrix_dense_shape}."
        )

    # row number check
    # the every diagonal requires one more row to be stored, e.g., 2 sub-diagonals
    # means that there have to be 3 rows which results in a dense lower triangle
    num_required_rows = max(
        l_and_u_internal[0] + 1,  # sub-diagonals
        l_and_u_internal[1] + 1,  # super-diagonals
    )

    if matrix_dense_shape[0] < num_required_rows:
        raise ValueError(
            f"Expected the dense version of matrix to have at least "
            f"{num_required_rows} rows (maximum of l + 1 = {l_and_u_internal[0] + 1} "
            f"and u + 1 = {l_and_u_internal[1] + 1}) to store the sub-diagonals and "
            f"super-diagonals, but got a matrix with dense shape {matrix_dense_shape}."
        )

    return l_and_u_internal  # type: ignore


# === Models ===


@dataclass
class GeneralBandedMatrix:
    """
    Base class for banded matrices that serves solely as a data container and cannot
    perform any operations by itself.
    It is not meant to be instantiated directly but to be created from one of its
    static methods.

    Parameters
    ----------
    data : :obj:`numpy.ndarray` of shape (sum(l_and_u) + 1, n) and dtype ``numpy.float64``
        The banded matrix in row major format.
        Please refer to the Notes section for details.
    l_and_u : (:obj:`int`, :obj:`int`)
        The number of sub-diagonals and super-diagonals in the matrix, respectively.
    enable_overwrite : :obj:`bool`, default=``False``
        Whether to allow overwriting ``matrix`` by routines that work on it (``True``)
        or not (``False``).
    is_linked_to_original_data : :obj:`bool` or ``None``, default=``True``
        Whether the banded matrix is linked to the original data as a view (``True``) or
        not (``False``).
        If it is not evaluated or required, it can be set to ``None``.
    warn_link_to_original_data : :obj:`bool`, default=``True``
        Whether to issue a performance warning when the ``overwrite_allowed`` property
        is called and ``is_linked_to_original_data`` is ``None`` because in this case
        the overwrite operation cannot be allowed in this case even if it might be
        beneficial.

    """  # noqa: E501

    data: NDArray[np.float64]
    l_and_u: Tuple[int, int]

    enable_overwrite: bool = field(default=False)
    is_linked_to_original_data: Optional[bool] = field(default=None)
    warn_link_to_original_data: bool = field(default=True)

    shape: Tuple[int, int] = field(init=False)
    main_diagonal_column_index: int = field(init=False)

    # --- Initialisation ---

    def __post_init__(self) -> None:

        self.shape = (self.data.shape[0], self.data.shape[1])
        self.main_diagonal_column_index = self.l_and_u[0]

        return

    # --- Properties ---

    @property
    def is_diagonal(self) -> bool:
        """
        Whether the matrix is a diagonal matrix (``True``) or not (``False``).

        """

        return self.l_and_u == (0, 0)

    @property
    def is_pentadiagonal(self) -> bool:
        """
        Whether the matrix is a pentadiagonal matrix (``True``) or not (``False``).

        """

        return self.l_and_u >= (0, 0) and self.l_and_u <= (2, 2)

    @property
    def is_upper_triangular(self) -> bool:
        """
        Whether the matrix is an upper triangular matrix (``True``) or not (``False``).

        """

        return self.l_and_u[0] == 0

    @property
    def is_lower_triangular(self) -> bool:
        """
        Whether the matrix is a lower triangular matrix (``True``) or not (``False``).

        """

        return self.l_and_u[1] == 0

    @property
    def overwrite_allowed(self) -> bool:
        """
        Checks whether an overwrite operation is allowed and returns the result.
        Even if the overwrite operation might not be allowed by the ``overwrite``
        argument alone, it can still be allowed if the matrix is not linked to the
        original data.

        Returns
        -------
        overwrite_allowed : :obj:`bool`
            Whether the overwrite operation is allowed (``True``) or not (``False``).

        """

        if not self.enable_overwrite:
            return False

        if self.is_linked_to_original_data is None:
            warn_verbose(
                message=(
                    "The overwrite operation cannot be allowed because the "
                    "`is_linked_to_original_data` property was not specified.\n"
                    "To silence this warning, set the 'is_linked_to_original_data' "
                    "attribute to 'False'."
                ),
                category=RuntimeWarning,
                issue_warning=self.warn_link_to_original_data,
            )

            return False

        return not self.is_linked_to_original_data

    # --- Static Methods ---

    @staticmethod
    def from_dense(
        matrix: RealNumericArrayLike,
        l_and_u: Optional[Tuple[Integer, Integer]] = None,
        check_finite: bool = True,
    ) -> "GeneralBandedMatrix":
        """
        Creates a banded matrix from a dense matrix.

        Parameters
        ----------
        matrix : :obj:`numpy.ndarray` of shape (n, n) and dtype ``numpy.float64``
            The dense matrix to be converted to a banded matrix.
            Its data type is internally promoted to ``numpy.float64``.
        l_and_u : (:obj:`int`, :obj:`int`) or ``None``, default=``None``
            The number of sub-diagonals and super-diagonals in the matrix, respectively.
            If ``None``, the function will determine the bandwidth automatically. In
            this case, it is advisable for ``matrix`` to have an either C- or
        check_finite : :obj:`bool`, default=``True``
            Whether to check the matrix for containing only finite values (``True``)
            or not (``False``).

        Returns
        -------
        banded_matrix : :obj:`GeneralBandedMatrix`
            The banded matrix.

        Raises
        ------
        ValueError
            If ``matrix`` or ``l_and_u`` is not of expected type.
        ValueError
            If ``matrix`` is empty.
        ValueError
            If the number of sub-diagonals and super-diagonals is not valid.
        ValueError
            If ``matrix``contains non-finite values and ``check_finite=True``.

        """

        # --- Input Validation ---

        data = get_validated_real_numeric_2d_array_like(
            value=matrix,
            name="matrix",
            enforce_finite=check_finite,
            output_dtype=np.float64,
        )

        if data.shape[0] != data.shape[1]:
            raise ValueError(
                f"Expected a square matrix, but got a matrix of shape {data.shape}."
            )

        l_and_u = get_validated_l_and_u(
            matrix=data,
            matrix_dense_shape=data.shape,  # type: ignore
            l_and_u=l_and_u,
        )

        # --- Banded Matrix ---

        new_data = np.empty(
            shape=(sum(l_and_u) + 1, data.shape[1]),
            dtype=np.float64,
        )

        # subdiagonals
        for column_index, offset in enumerate(range(-l_and_u[0], 0)):
            new_data[-offset::, column_index] = data.diagonal(offset=offset).copy()

        # main diagonal
        new_data[:, l_and_u[0]] = data.diagonal(offset=0).copy()

        # superdiagonals
        for offset in range(1, l_and_u[1] + 1):
            new_data[0:-offset, l_and_u[0] + offset] = data.diagonal(
                offset=offset
            ).copy()

        return GeneralBandedMatrix(
            data=new_data,
            l_and_u=l_and_u,
        )
