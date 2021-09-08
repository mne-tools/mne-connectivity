"""Connectivity for MEG, EEG and iEEG data."""

# Authors: Adam Li <ali39@jhu.edu>
#          Eric Larson <larson.eric.d@gmail.com>
#          Britta Westner <britta.wstnr@gmail.com>
#
# License: BSD (3-clause)

__version__ = '0.2'

from .base import (Connectivity, EpochConnectivity, EpochSpectralConnectivity,
                   EpochSpectroTemporalConnectivity, EpochTemporalConnectivity,
                   SpectralConnectivity, SpectroTemporalConnectivity,
                   TemporalConnectivity)
from .effective import phase_slope_index
from .envelope import envelope_correlation
from .io import read_connectivity
from .spectral import spectral_connectivity
from .var import vector_auto_regression
from .utils import check_indices, degree, seed_target_indices
