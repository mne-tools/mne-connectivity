"""Connectivity MEG, EEG, iEEG data processing."""

# Authors: Adam Li <ali39@jhu.edu>
#          Eric Larson <>
#          Britta Westner <>
#
# License: BSD (3-clause)

__version__ = "0.1.dev0"

from .envelope import envelope_correlation
from .effective import phase_slope_index
from .spectral import spectral_connectivity

__all__ = ['__version__']
