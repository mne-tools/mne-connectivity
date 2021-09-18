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

Version 0.3 (Unreleased)
------------------------

...

Enhancements
~~~~~~~~~~~~

- Adding symmetric orthogonalization via :func:`mne_connectivity.symmetric_orth`, by `Eric Larson`_ (:gh:`36`)

Bug
~~~

- Fixed bug when saving connectivity with ``n_jobs`` greater than 1 from :func:`mne_connectivity.spectral_connectivity`, by `Adam Li`_ (:gh:`43`)
- Fixed bug to allow saving complex data connectivity, by `Adam Li`_ (:gh:`43`)

API
~~~

- Added ``h5netcdf`` as a requirement for saving connectivity data, by `Adam Li`_ (:gh:`43`).

Authors
~~~~~~~

* `Adam Li`_
* `Eric Larson`_

:doc:`Find out what was new in previous releases <whats_new_previous_releases>`

.. include:: authors.inc
