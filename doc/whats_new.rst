.. _whats_new:


What's new?
===========

Here we list a changelog of MNE-connectivity.

.. contents:: Contents
   :local:
   :depth: 3

.. currentmodule:: mne_connectivity

.. _current:

Version 0.8 (in dev)
--------------------

Minimum supported Python version is now 3.10.

Enhancements
~~~~~~~~~~~~

- Add a new :class:`~mne_connectivity.decoding.CoherencyDecomposition` class for decomposing connectivity sources using multivariate coherency-based methods, by `Thomas Binns`_ (:pr:`193`).
- Add new plotting methods :meth:`CoherencyDecomposition.plot_filters() <mne_connectivity.decoding.CoherencyDecomposition.plot_filters>` and :meth:`CoherencyDecomposition.plot_patterns() <mne_connectivity.decoding.CoherencyDecomposition.plot_patterns>` for visualising the decomposed connectivity sources, by `Thomas Binns`_ (:pr:`208`).
- Add support for computing multiple components of multivariate connectivity in the :func:`~mne_connectivity.spectral_connectivity_epochs` and :func:`~mne_connectivity.spectral_connectivity_time` functions and :class:`~mne_connectivity.decoding.CoherencyDecomposition` class, and add support for storing data with a components dimension in all :class:`~mne_connectivity.Connectivity` classes, by `Thomas Binns`_ and `Eric Larson`_ (:pr:`213`).
- Add support for :class:`mne.time_frequency.EpochsSpectrum` objects to be passed as data to the :func:`~mne_connectivity.spectral_connectivity_epochs` function, by `Thomas Binns`_ and `Eric Larson`_ (:pr:`220`).
- Update the cross-references for relevant functions and classes and make data types more explicit throughout the documentation, by `Thomas Binns`_ (:pr:`214`).

Bug
~~~

- Improve the documentation of the ``fmin`` and ``cwt_freqs`` parameters in the :func:`~mne_connectivity.spectral_connectivity_epochs` function, by `Richard Köhler`_ and `Daniel McCloy`_ (:pr:`242`).

API
~~~

- Add a new ``min_distance`` parameter to the :func:`~mne_connectivity.viz.plot_sensors_connectivity` function which offers greater control over the minimum distance required between sensors to plot a connection between them, by `Thomas Binns`_ and `Eric Larson`_ (:pr:`221`).

Authors
~~~~~~~

* `Thomas Binns`_
* `Richard Köhler`_
* `Adam Li`_
* `Marijn van Vliet`_
* `Eric Larson`_
* `Daniel McCloy`_

:doc:`Find out what was new in previous releases <whats_new_previous_releases>`

.. include:: authors.inc
