"""
Package ``pybandee``

A Python package specialised in factorising and solving banded matrices. Currently,
the package supports the following functionalities:

- pentadiagonal factorisation and solving
- computation of the log-determinant and inverse elements of a pentadiagonal matrix

"""

# === Imports ===

import os as _os

# === Package Metadata ===

_AUTHOR_FILE_PATH = _os.path.join(_os.path.dirname(__file__), "AUTHORS.txt")
_VERSION_FILE_PATH = _os.path.join(_os.path.dirname(__file__), "VERSION.txt")

with open(_AUTHOR_FILE_PATH, "r") as author_file:
    __author__ = author_file.read().strip()

with open(_VERSION_FILE_PATH, "r") as version_file:
    __version__ = version_file.read().strip()
