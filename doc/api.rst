###
API
###

This is the reference for classes (``CamelCase`` names) and functions
(``underscore_case`` names) of MNE-Connectivity, grouped thematically by analysis
stage. MNE-Connectivity also provides connectivity data structure
containers for different types of connectivity data, which are
described below.

:py:mod:`mne_connectivity`:

.. automodule:: mne_connectivity
   :no-members:
   :no-inherited-members:

Most-used classes
=================

.. currentmodule:: mne_connectivity

.. autosummary::
   :toctree: generated/

   TemporalConnectivity
   SpectralConnectivity
   SpectroTemporalConnectivity
   EpochTemporalConnectivity
   EpochSpectralConnectivity
   EpochSpectroTemporalConnectivity

Connectivity functions
======================

These functions compute connectivity and return
one of the Connectivity data structure classes
listed above.

.. currentmodule:: mne_connectivity

.. autosummary::
   :toctree: generated/

   envelope_correlation
   phase_slope_index
   spectral_connectivity

Post-processing on connectivity
===============================

.. currentmodule:: mne_connectivity

.. autosummary::
   :toctree: generated/

   degree
   seed_target_indices
   check_indices