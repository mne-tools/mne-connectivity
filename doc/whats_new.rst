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
- Improved RAM usage for :func:`mne_connectivity.vector_auto_regression` by leveraging code from ``statsmodels``, by `Adam Li`_ (:gh:`46`)
- Added :func:`mne_connectivity.select_order` for helping to select VAR order using information criterion, by `Adam Li`_ (:gh:`46`)
- All connectivity functions retain ``events``, ``event_id`` and ``metadata`` from `mne.Epochs` objects as input and is stored as part of the connectivity object, by `Adam Li`_ (:gh:`58`)

Bug
~~~

- Fixed bug when saving connectivity with ``n_jobs`` greater than 1 from :func:`mne_connectivity.spectral_connectivity`, by `Adam Li`_ (:gh:`43`)
- Fixed bug to allow saving complex data connectivity, by `Adam Li`_ (:gh:`43`)
- Fixed bug to keep label orientation upright in :func:`mne_connectivity.viz.plot_connectivity_circle`, by `Alexander Kroner`_ (:gh:`60`)

API
~~~

- Added ``h5netcdf`` as a requirement for saving connectivity data, by `Adam Li`_ (:gh:`43`)
- Changed keyword argument ``model_order`` in :func:`mne_connectivity.vector_auto_regression` to ``lags`` to more align with statsmodels API, by `Adam Li`_ (:gh:`47`)
- Add ``pandas`` as a requirement for dealing with metadata associated from the original Epochs file, by `Adam Li`_ (:gh:`58`)


Authors
~~~~~~~

* `Adam Li`_
* `Eric Larson`_
* `Alexander Kroner`_

:doc:`Find out what was new in previous releases <whats_new_previous_releases>`

.. include:: authors.inc
