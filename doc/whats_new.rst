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

- Adding `TemporalConnectivity`, `SpectralConnectivity` and `SpectroTemporalConnectivity` as a data structure to hold connectivity data, by `Adam Li`_ (:gh:`6`)
- Adding `EpochTemporalConnectivity`, `EpochSpectralConnectivity` and `EpochSpectroTemporalConnectivity` as a data structure to hold connectivity data over Epochs, by `Adam Li`_ (:gh:`6`)

Bug
~~~

API
~~~

- :func:`envelope_correlation`, :func:`spectral_connectivity`, and :func:`phase_slope_index` all return ``_Connectivity`` containers now, by `Adam Li`_ (:gh:`6`)
- Added ``xarray`` as a dependency where all connectivity containers are now underlying xarrays, by `Adam Li`_ (:gh:`6`)


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
