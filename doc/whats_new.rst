:orphan:

.. _whats_new:


What's new?
===========

Here we list a changelog of MNE-connectivity.

.. contents:: Contents
   :local:
   :depth: 3

.. currentmodule:: mne_connectivity

.. _current:

Current
-------

Changelog
~~~~~~~~~

- Adding `Connectivity`, `TemporalConnectivity`, `SpectralConnectivity` and `SpectroTemporalConnectivity` as a data structure to hold connectivity data, by `Adam Li`_ (:gh:`6`)
- Adding `EpochConnectivity`, `EpochTemporalConnectivity`, `EpochSpectralConnectivity` and `EpochSpectroTemporalConnectivity` as a data structure to hold connectivity data over Epochs, by `Adam Li`_ (:gh:`6`)
- ``indices`` argument in Connectivity classes can now be ``symmetric``, allowing for memory-efficient storage of symmetric connectivity, by `Adam Li`_ (:gh:`20`)
- New function ``save`` in Connectivity classes along with :func:`read_connectivity` can now be used to write and read Connectivity data as netCDF files, by `Adam Li`_ (:gh:`20`)

Bug
~~~

API
~~~

- :func:`envelope_correlation`, :func:`spectral_connectivity`, and :func:`phase_slope_index` all return ``_Connectivity`` containers now, by `Adam Li`_ (:gh:`6`)
- Added ``xarray`` as a dependency where all connectivity containers are now underlying xarrays, by `Adam Li`_ (:gh:`6`)
- The ``combine`` argument in :func:`envelope_correlation` was removed, and now all Epoch Connectivity classes have a ``combine`` class function, by `Adam Li`_ (:gh:`20`)


Changes when mne-connectivity was part of MNE-Python
----------------------------------------------------

In July, 2021, ``mne.connectivity`` submodule was ported over from the MNE-Python 
repo into this repository, ``mne-connectivity``. Starting v0.24 of MNE-Python, that sub-module 
will be deprecated and development will move over into this repository. Starting v0.25 of MNE-Python,
``mne.connectivity`` will completely be removed.

Authors
~~~~~~~

* Adam Li

.. include:: authors.inc
