"""Connectivity for MEG, EEG and iEEG data."""

# Authors: Adam Li <ali39@jhu.edu>
#          Eric Larson <larson.eric.d@gmail.com>
#          Britta Westner <britta.wstnr@gmail.com>
#
# License: BSD (3-clause)

__version__ = '0.5.0.dev0'

from .base import (Connectivity, EpochConnectivity, EpochSpectralConnectivity,
                   EpochSpectroTemporalConnectivity, EpochTemporalConnectivity,
                   SpectralConnectivity, SpectroTemporalConnectivity,
                   TemporalConnectivity, MultivariateSpectralConnectivity,
                   MultivariateSpectroTemporalConnectivity)
from .effective import phase_slope_index
from .envelope import envelope_correlation, symmetric_orth
from .io import read_connectivity
from .spectral import (spectral_connectivity_time, spectral_connectivity_epochs,
                       multivariate_spectral_connectivity_epochs)
from .vector_ar import vector_auto_regression, select_order
from .utils import (check_indices, degree, seed_target_indices,
                    multivariate_seed_target_indices)
