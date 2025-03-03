"""
Module :mod:`penta`

This module provides utilities for working with pentadiagonal matrices, namely

- factorisation
- solving
- computation of their log determinant

For accessing the Numba-compatible low-level functions, the submodule :mod:`low_level`
is available.

"""

# === Imports ===

from ._interface import (  # noqa: F401
    penta_factorize,
    penta_slogdet,
    penta_slogdet_from_factorization,
    penta_solve,
)
from ._penta_utils import convert_to_validated_penta # noqa: F401
from ._ptrans1 import (  # noqa: F401
    ptrans1_factorize,
    ptrans1_slogdet,
    ptrans1_solve_single_rhs,
    ptrans1_symmetric_inverse_central_penta_bands,
)
