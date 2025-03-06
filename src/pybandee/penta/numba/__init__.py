"""
Module :mod:`penta.numba`

This module provides all the Numba ``jit``-compatible utilities for working with
pentadiagonal matrices, namely

- factorisation
- solving
- computation of their log determinant
- computation of the central pentadiagonal bands of the inverse (symmetric matrices
    only)


"""

# === Imports ===

from ._ptrans1 import (  # noqa: F401
    ptrans1_factorize,
    ptrans1_slogdet,
    ptrans1_solve_single_rhs,
    ptrans1_symmetric_inverse_central_penta_bands,
)
