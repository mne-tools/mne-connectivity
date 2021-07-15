###
API
###

:py:mod:`mne_connectivity`:

.. automodule:: mne_connectivity
   :no-members:
   :no-inherited-members:

This is the application programming interface (API) reference
for classes (``CamelCase`` names) and functions
(``underscore_case`` names) of MNE-Connectivity, grouped thematically by analysis
stage. The data structure classes contain different types of connectivity data
and are described below.

Most-used classes
=================

.. currentmodule:: mne_connectivity

.. autosummary::
   :toctree: generated/

   Connectivity
   TemporalConnectivity
   SpectralConnectivity
   SpectroTemporalConnectivity
   EpochConnectivity
   EpochTemporalConnectivity
   EpochSpectralConnectivity
   EpochSpectroTemporalConnectivity

Connectivity functions
======================

These functions compute connectivity and return
one of the Connectivity data structure classes
listed above.

.. autosummary::
   :toctree: generated/

   envelope_correlation
   phase_slope_index
   spectral_connectivity
   var

Reading functions
=================

.. autosummary::
   :toctree: generated/

   read_connectivity


Statistics
==========

.. autosummary::
   :toctree: generated/

   stats.portmanteau

Post-processing on connectivity
===============================

.. autosummary::
   :toctree: generated/

   degree
   seed_target_indices
   check_indices