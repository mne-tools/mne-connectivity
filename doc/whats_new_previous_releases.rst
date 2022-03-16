:orphan:

.. _whats_new_in_previous_releases:

.. currentmodule:: mne_connectivity

What was new in previous releases?
==================================

Version 0.3 (2022-03-01)
------------------------

This version has bug fixes minor improvements in certain functions. A big change
is the renaming of functions ``spectral_connectivity`` to ``spectral_connectivity_epochs``,
which makes it explicit that the function operates over Epochs, rather then time.
Importantly, we also provide a conda installation now.

Enhancements
~~~~~~~~~~~~

- Adding symmetric orthogonalization via :func:`mne_connectivity.symmetric_orth`, by `Eric Larson`_ (:gh:`36`)
- Improved RAM usage for :func:`mne_connectivity.vector_auto_regression` by leveraging code from ``statsmodels``, by `Adam Li`_ (:gh:`46`)
- Added :func:`mne_connectivity.select_order` for helping to select VAR order using information criterion, by `Adam Li`_ (:gh:`46`)
- All connectivity functions retain ``events``, ``event_id`` and ``metadata`` from `mne.Epochs` objects as input and is stored as part of the connectivity object, by `Adam Li`_ (:gh:`58`)
- Add spectral connectivity over time function :func:`mne_connectivity.spectral_connectivity_time`, by `Adam Li`_ (:gh:`67`)
- Add conda installation, by `Adam Li`_ and `Richard Höchenberger`_ (:gh:`81`)

Bug
~~~

- Fixed bug when saving connectivity with ``n_jobs`` greater than 1 from :func:`mne_connectivity.spectral_connectivity_epochs`, by `Adam Li`_ (:gh:`43`)
- Fixed bug to allow saving complex data connectivity, by `Adam Li`_ (:gh:`43`)
- Fixed bug to keep label orientation upright in :func:`mne_connectivity.viz.plot_connectivity_circle`, by `Alexander Kroner`_ (:gh:`60`)

API
~~~

- Added ``h5netcdf`` as a requirement for saving connectivity data, by `Adam Li`_ (:gh:`43`)
- Changed keyword argument ``model_order`` in :func:`mne_connectivity.vector_auto_regression` to ``lags`` to more align with statsmodels API, by `Adam Li`_ (:gh:`47`)
- Add ``pandas`` as a requirement for dealing with metadata associated from the original Epochs file, by `Adam Li`_ (:gh:`58`)
- Rename ``mne_connectivity.spectral_connectivity`` to :func:`mne_connectivity.spectral_connectivity_epochs`, by `Adam Li`_ (:gh:`69`)

Authors
~~~~~~~
People who contributed to this release (in alphabetical order):

* `Adam Li`_
* `Alexander Kroner`_
* `Eric Larson`_
* `Richard Höchenberger`_

.. _changes_0_2:

Version 0.2 (2021-09-07)
------------------------

In this version, we return the relevant Connectivity class from each of the
connectivity estimation functions. These internally use ``xarray`` to represent
the connectivity data. One can easily get the v0.1 numpy array by doing
``conn.get_data()``, which will get exactly the same output as one got in v0.1
running any of the connectivity functions.

Changelog
~~~~~~~~~

- Adding `Connectivity`, `TemporalConnectivity`, `SpectralConnectivity` and `SpectroTemporalConnectivity` as a data structure to hold connectivity data, by `Adam Li`_ (:gh:`6`)
- Adding `EpochConnectivity`, `EpochTemporalConnectivity`, `EpochSpectralConnectivity` and `EpochSpectroTemporalConnectivity` as a data structure to hold connectivity data over Epochs, by `Adam Li`_ (:gh:`6`)
- ``indices`` argument in Connectivity classes can now be ``symmetric``, allowing for memory-efficient storage of symmetric connectivity, by `Adam Li`_ (:gh:`20`)
- New function ``save`` in Connectivity classes along with :func:`read_connectivity` can now be used to write and read Connectivity data as netCDF files, by `Adam Li`_ (:gh:`20`)
- New function :func:`vector_auto_regression` to compute dynamic connectivity vector auto-regressive (VAR) model, by `Adam Li`_ (:gh:`23`)

API
~~~

- :func:`envelope_correlation`, ``spectral_connectivity``, and :func:`phase_slope_index` all return ``_Connectivity`` containers now, by `Adam Li`_ (:gh:`6`)
- Added ``xarray`` as a dependency where all connectivity containers are now underlying xarrays, by `Adam Li`_ (:gh:`6`)
- The ``combine`` argument in :func:`envelope_correlation` was removed, and now all Epoch Connectivity classes have a ``combine`` class function, by `Adam Li`_ (:gh:`20`)

Authors
~~~~~~~

People who contributed to this release (in alphabetical order):

* `Adam Li`_
* `Eric Larson`_
* `Britta Westner`_

.. _changes_0_1:

Version 0.1 (2021-06-25)
------------------------

Changes when mne-connectivity was part of MNE-Python
----------------------------------------------------

In July, 2021, ``mne.connectivity`` submodule was ported over from the MNE-Python 
repo into this repository, ``mne-connectivity`` as v0.1. Starting v0.24 of MNE-Python, that sub-module 
will be deprecated and development will move over into this repository. Starting v0.25 of MNE-Python,
``mne.connectivity`` will completely be removed.

Authors
~~~~~~~

People who contributed to this release (in alphabetical order):

* `Adam Li`_

.. include:: authors.inc
