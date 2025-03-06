"""
Configuration file for ``pytest``.

This handles

- the command line option to deactivate Numba ``jit``-compilation so that the coverage
    tests can be run properly

"""

# === Imports ===

import os
from enum import Enum

# === Models ===

# NOTE: the following code is copied from src/pybandee/_utils/numba_helpers.py
#       to avoid an import of the package before the environment variable is set

# an Enum that specifies the possible actions that can be taken regarding Numba
# ``jit``-compilation


class NumbaJitActions(Enum):
    """
    Specifies the possible actions that can be taken regarding Numba
    ``jit``-compilation.

    """

    NORMAL = "0"
    DEACTIVATE = "1"


# === Constants ===

# the runtime argument that is used to specify that Numba ``jit``-compilation should
# take no effect
NUMBA_NO_JIT_ARGV = "--no-jit"

# the environment variable that is used to specify that Numba ``jit``-compilation should
# take no effect
NUMBA_NO_JIT_ENV_KEY = "CUSTOM_NUMBA_NO_JIT"

# === Functions ===


def pytest_addoption(parser):
    """
    Adds the command line option to deactivate Numba ``jit``-compilation.

    """

    parser.addoption(
        NUMBA_NO_JIT_ARGV,
        action="store_true",
        help="Disable Numba JIT compilation",
    )


def pytest_configure(config):
    """
    Configures the runtime environment based on the command line option.

    """

    if config.getoption(NUMBA_NO_JIT_ARGV):
        os.environ[NUMBA_NO_JIT_ENV_KEY] = NumbaJitActions.DEACTIVATE.value
