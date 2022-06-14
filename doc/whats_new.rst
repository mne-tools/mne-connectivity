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

Version 0.4 (Unreleased)
------------------------

...

Enhancements
~~~~~~~~~~~~

- Add ``node_height`` to :func:`mne_connectivity.viz.plot_connectivity_circle` and enable passing a polar ``ax``, by `Alex Rockhill`_ :gh:`88`
- Add directed phase lag index (dPLI) as a method in :func:`mne_connectivity.spectral_connectivity_epochs` with a corresponding example by `Kenji Marshall`_ (:gh:`79`)

Bug
~~~

- Fix the output of :func:`mne_connectivity.spectral_connectivity_epochs` when ``faverage=True``, allowing one to save the Connectivity object, by `Adam Li`_ and `Szonja Weigl`_ (:gh:`91`)
- Fix the incompatibility of dimensions of frequencies in the creation of ``EpochSpectroTemporalConnectivity`` object in :func:`mne_connectivity.spectral_connectivity_time` by providing the frequencies of interest into the object, rather than the frequencies used in the time-frequency decomposition by `Adam Li`_ and `Sezan Mert`_ (:gh:`98`)

API
~~~

- 

Authors
~~~~~~~

* `Kenji Marshall`_
* `Adam Li`_
* `Alex Rockhill`_
* `Szonja Weigl`_
* `Sezan Mert`_

:doc:`Find out what was new in previous releases <whats_new_previous_releases>`

.. include:: authors.inc
