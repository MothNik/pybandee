"""
Module :mod:`_utils._custom_types`

This module implements custom types that are used throughout the package to allow for
more readable code and better type checking.

"""

# === Setup ===

__all__ = [
    "Numpy1DArrayFloat64",
    "Numpy2DArrayFloat64",
]

# === Imports ===

from typing import Literal, Tuple

import numpy as np

# === Custom Types ===

Numpy1DArrayFloat64 = np.ndarray[Tuple[int], np.dtype[np.float64]]
Numpy2DArrayFloat64 = np.ndarray[Tuple[int, int], np.dtype[np.float64]]
NumpyPentaDiagonalArrayFloat64 = np.ndarray[
    Tuple[int, Literal[5]], np.dtype[np.float64]
]
