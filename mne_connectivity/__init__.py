"""Connectivity for MEG, EEG and iEEG data."""

# Authors: Adam Li <ali39@jhu.edu>
#          Eric Larson <larson.eric.d@gmail.com>
#          Britta Westner <britta.wstnr@gmail.com>
#
# License: BSD (3-clause)

from ._version import __version__  # noqa: F401
from .base import (
    Connectivity,
    EpochConnectivity,
    EpochSpectralConnectivity,
    EpochSpectroTemporalConnectivity,
    EpochTemporalConnectivity,
    SpectralConnectivity,
    SpectroTemporalConnectivity,
    TemporalConnectivity,
)
from .datasets import make_signals_in_freq_bands
from .effective import phase_slope_index
from .envelope import envelope_correlation, symmetric_orth
from .io import read_connectivity
from .spectral import spectral_connectivity_epochs, spectral_connectivity_time
from .utils import (
    check_indices,
    degree,
    seed_target_indices,
    seed_target_multivariate_indices,
)
from .vector_ar import select_order, vector_auto_regression
