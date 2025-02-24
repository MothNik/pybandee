"""
Mod :mod:`_utils._custom_types`

This module provides type definitions used across the ``pybandee`` package.

"""

# === Imports ===

from collections import namedtuple
from typing import Union

import numpy as np
from numpy.typing import ArrayLike

# === Type Definitions ===

# a real numeric value
RealNumeric = Union[int, float, np.integer, np.floating]
Integer = Union[int, np.integer]

# a real numeric Arraylike
RealNumericArrayLike = Union[RealNumeric, ArrayLike]

# matrix factorisations
MatrixFactorization = namedtuple(
    typename="MatrixFactorization",
    field_names=(
        "factorization",
        "method",
    ),
)

# linear system solutions
LinearSystemSolution = namedtuple(
    typename="LinearSystemSolution",
    field_names=(
        "solution",
        "factorization",
        "method",
    ),
)
