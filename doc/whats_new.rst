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

Version 0.7
-----------

This version adds a new multivariate connectivity method to the ``spectral`` connectivity analysis functions,
an example explaining when certain connectivity methods should be used, and a function for simulating connectivity.
Compute efficiency has also been improved in some cases, including a bug fix for parallel processing.

Enhancements
~~~~~~~~~~~~

- Add support for a new multivariate connectivity method (canonical coherence; ``cacoh``) in :func:`mne_connectivity.spectral_connectivity_epochs` and :func:`mne_connectivity.spectral_connectivity_time`, by `Thomas Binns`_ and `Mohammad Orabe`_ and `Mina Jamshidi`_ (:pr:`163`).
- Add a new :ref:`example <ex-compare-cohy-methods>` for comparing use-cases of different coherency-based methods, by `Thomas Binns`_ and `Mohammad Orabe`_ (:pr:`163`).
- Add a new function :func:`mne_connectivity.make_signals_in_freq_bands` for simulating connectivity between signals, by `Thomas Binns`_ and `Adam Li`_ (:pr:`173`).

Bug
~~~

- Fix bug where ``n_jobs=-1`` would not be converted to the CPU count in :func:`mne_connectivity.spectral_connectivity_epochs`, by `Thomas Binns`_ (:pr:`177`).

Authors
~~~~~~~

* `Thomas Binns`_
* `Mohammad Orabe`_
* `Mina Jamshidi`_
* `Adam Li`_
* `Eric Larson`_
* `Daniel McCloy`_


:doc:`Find out what was new in previous releases <whats_new_previous_releases>`

.. include:: authors.inc
