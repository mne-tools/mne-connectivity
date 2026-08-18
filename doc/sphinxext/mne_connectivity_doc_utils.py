"""Helpers for the MNE-Connectivity documentation build."""

# Authors: The MNE-Connectivity developers
# SPDX-License-Identifier: BSD-3-Clause

import os
import warnings
from contextlib import suppress


def reset_modules(gallery_conf, fname, when):
    """Reset the state each example is executed with."""
    # Sphinx-Gallery runs examples in worker processes that never execute conf.py, so
    # anything conf.py sets at import time has to be set again here
    os.environ["_MNE_BUILDING_DOC"] = "true"
    _limit_blas_threads(gallery_conf["parallel"] or 1)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            import pyvista
    except Exception:
        return
    pyvista.OFF_SCREEN = False
    pyvista.BUILDING_GALLERY = True


def _limit_blas_threads(n_workers):
    """Keep concurrently executing examples from oversubscribing the CPUs."""
    if os.getenv("OMP_NUM_THREADS") is not None:
        return
    import numpy  # noqa: F401  threadpool_limits only sees already-loaded libraries
    from threadpoolctl import threadpool_limits

    max_threads = (os.cpu_count() or 2) // 2  # number of physical cores
    # suppress e.g. AttributeError raised by older versions of OpenBLAS
    with suppress(Exception):
        threadpool_limits(max(max_threads // n_workers, 1), user_api="blas")
