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

Version 0.5 (Unreleased)
------------------------

This version has major changes in :func:`mne_connectivity.spectral_connectivity_time`. Several bugs are fixed, and the
function now computes static connectivity over time, as opposed to static connectivity over trials computed by  :func:`mne_connectivity.spectral_connectivity_epochs`.

Enhancements
~~~~~~~~~~~~

- Add the ``PLI`` and ``wPLI`` methods in :func:`mne_connectivity.spectral_connectivity_time` by `Santeri Ruuskanen`_ (:gh:`104`).
- Improve the documentation of :func:`mne_connectivity.spectral_connectivity_time` by `Santeri Ruuskanen`_ (:gh:`104`).
- Add the option to average connectivity across epochs and frequencies in :func:`mne_connectivity.spectral_connectivity_time` by `Santeri Ruuskanen`_ (:gh:`104`).
- Select multitaper frequencies automatically in :func:`mne_connectivity.spectral_connectivity_time` similarly to :func:`mne_connectivity.spectral_connectivity_epochs` by `Santeri Ruuskanen`_ (:gh:`104`).

Bug
~~~

- When using the ``multitaper`` mode in :func:`mne_connectivity.spectral_connectivity_time`, average CSD over tapers instead of the complex signal by `Santeri Ruuskanen`_ (:gh:`104`).
- Average over time when computing connectivity measures in :func:`mne_connectivity.spectral_connectivity_time` by `Santeri Ruuskanen`_ (:gh:`104`).
- Fix support for multiple connectivity methods in calls to :func:`mne_connectivity.spectral_connectivity_time` by `Santeri Ruuskanen`_ (:gh:`104`).
- Fix bug with the ``indices`` parameter in :func:`mne_connectivity.spectral_connectivity_time`, the behavior is now as expected by `Santeri Ruuskanen`_ (:gh:`104`).
- Fix bug with parallel computation in :func:`mne_connectivity.spectral_connectivity_time`, add instructions for memory mapping in doc by `Santeri Ruuskanen`_ (:gh:`104`).

API
~~~

- Streamline the API of :func:`mne_connectivity.spectral_connectivity_time` with :func:`mne_connectivity.spectral_connectivity_epochs` by `Santeri Ruuskanen`_ (:gh:`104`).

Authors
~~~~~~~

* `Santeri Ruuskanen`_

:doc:`Find out what was new in previous releases <whats_new_previous_releases>`

.. include:: authors.inc
