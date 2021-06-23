"""Connectivity MEG, EEG, iEEG data processing."""

# Authors: Adam Li <ali39@jhu.edu>
#          Eric Larson <larson.eric.d@gmail.com>
#          Britta Westner <britta.wstnr@gmail.com>
#
# License: BSD (3-clause)

__version__ = '0.1'

from .envelope import envelope_correlation
from .effective import phase_slope_index
from .spectral import spectral_connectivity
from .utils import seed_target_indices, degree, check_indices
