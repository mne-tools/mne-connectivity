:orphan:

.. _whats_new_in_previous_releases:

.. currentmodule:: mne_connectivity

What was new in previous releases?
==================================

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
