"""Helpers for the MNE-Connectivity documentation build."""

# Authors: The MNE-Connectivity developers
# SPDX-License-Identifier: BSD-3-Clause

import os
import warnings


def reset_modules(gallery_conf, fname, when):
    """Reset the state each example is executed with."""
    # Sphinx-Gallery runs examples in worker processes that never execute conf.py, so
    # anything conf.py sets at import time has to be set again here
    os.environ["_MNE_BUILDING_DOC"] = "true"
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            import pyvista
    except Exception:
        return
    pyvista.OFF_SCREEN = False
    pyvista.BUILDING_GALLERY = True
