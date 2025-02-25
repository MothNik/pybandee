"""
Module :mod:`_utils._miscellaneous`

This module provides a collection of miscellaneous utility functions that are used
across the ``pybandee`` package.

"""

# === Imports ===

import re
from typing import Union
from warnings import warn

import numpy as np

from ._custom_types import ArrayLike, RealNumeric

# === Functions ===


def split_class_name_to_readable(obj: object) -> str:
    """
    Splits a class name into a more readable format by inserting spaces between
    capital letters.

    Parameters
    ----------
    obj : object
        The object whose class name is to be split.

    Returns
    -------
    readable_class_name : :obj:`str`
        The class name with spaces inserted between capital letters.

    """

    return re.sub(r"(?<!^)(?=[A-Z])", " ", obj.__class__.__name__)


def warn_verbose(
    message: str,
    category: type,
    issue_warning: bool,
) -> None:
    """
    Issues a warning if requested.

    Parameters
    ----------
    message : :obj:`str`
        The warning message.
    category : :obj:`type`
        The warning category.
    issue_warning : :obj:`bool`
        Whether to issue the warning (``True``) or not (``False``).

    """  # noqa: E501

    if issue_warning:
        warn(message, category)
        return

    return


def is_data_linked(
    array: np.ndarray,
    original: Union[RealNumeric, ArrayLike],
) -> bool:
    """
    Strictly checks for ``array`` not sharing any data with ``original``, under the
    assumption that ``array = atleast_{x}d(original)`` followed by a potential type
    conversion.
    If ``array`` is a view of ``original``, this function returns ``False``.

    Was copied from the SciPy utility function ``scipy.linalg._misc._datacopied``, but
    the name and the docstring were adapted to make them clearer. Besides, the check for
    scalar ``original``s was added.

    """

    if np.isscalar(original):
        return False
    if array is original:
        return True
    if not isinstance(original, np.ndarray) and hasattr(original, "__array__"):
        return original.__array__().dtype is array.dtype  # type: ignore

    return array.base is not None
