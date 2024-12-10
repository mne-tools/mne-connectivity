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
listed above. All these functions work with MNE-Python's ``Epochs`` class,
which is the recommended input to these functions. However, they also work
on numpy array inputs.

.. autosummary::
   :toctree: generated/

   envelope_correlation
   phase_slope_index
   vector_auto_regression
   spectral_connectivity_epochs
   spectral_connectivity_time

Decoding classes
================

These classes fit filters which decompose data into discrete sources of
connectivity, amplifying the signal-to-noise ratio of these interactions.

.. currentmodule:: mne_connectivity.decoding

.. autosummary::
   :toctree: generated/

   CoherencyDecomposition

Reading functions
=================

.. currentmodule:: mne_connectivity

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
   seed_target_multivariate_indices
   check_indices
   select_order

Visualization functions
=======================

.. currentmodule:: mne_connectivity.viz

.. autosummary::
   :toctree: generated/

   plot_sensors_connectivity
   plot_connectivity_circle

Dataset functions
=================

.. currentmodule:: mne_connectivity

.. autosummary::
   :toctree: generated/

   make_signals_in_freq_bands
   make_surrogate_data