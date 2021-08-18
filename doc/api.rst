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
   vector_auto_regression

Reading functions
=================

.. autosummary::
   :toctree: generated/

   read_connectivity

Pre-processing on connectivity
==============================

.. autosummary::
   :toctree: generated/

   symmetric_orth

Post-processing on connectivity
===============================

.. autosummary::
   :toctree: generated/

   degree
   seed_target_indices
   check_indices

Visualization functions
=======================

.. currentmodule:: mne_connectivity.viz

.. autosummary::
   :toctree: generated/

   plot_sensors_connectivity
   plot_connectivity_circle