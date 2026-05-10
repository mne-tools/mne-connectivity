"""
==================================================================================
Determine the significance of connectivity estimates against baseline connectivity
==================================================================================

This example demonstrates how surrogate data can be generated to assess whether
connectivity estimates are significantly greater than baseline.
"""

# Author: Thomas S. Binns <t.s.binns@outlook.com>
# License: BSD (3-clause)
# sphinx_gallery_thumbnail_number = 3

# %%

import matplotlib.pyplot as plt
import mne
import numpy as np
from mne.datasets import somato

from mne_connectivity import (
    make_surrogate_evoked_data,
    make_surrogate_resting_data,
    spectral_connectivity_epochs,
)

########################################################################################
# Background
# ----------
#
# When performing connectivity analyses, we often want to know whether the results we
# observe reflect genuine interactions between signals. We can assess this by performing
# statistical tests between our connectivity estimates and a 'baseline' level of
# connectivity. However, due to factors such as background noise and sample
# size-dependent biases (see e.g., :footcite:t:`VinckEtAl2010`), it is often not
# appropriate to treat 0 as this baseline. Therefore, we need a way to estimate the
# baseline level of connectivity.
#
# One approach is to manipulate the original data in such a way that the covariance
# structure is destroyed, creating surrogate data. Connectivity estimates from the
# original and surrogate data can then be compared to determine whether the original
# data contains significant interactions.
#
# Such surrogate data can be easily generated in MNE using the
# :func:`~mne_connectivity.make_surrogate_resting_data` :footcite:`PellegriniEtAl2023`
# and :func:`~mne_connectivity.make_surrogate_evoked_data` :footcite:`AruEtAl2015`
# functions. In this example, we will demonstrate how surrogate data can be created, and
# how you can use this to assess the statistical significance of your connectivity
# estimates.

########################################################################################
# Loading the data
# ----------------
#
# We start by loading from the :ref:`somato-dataset` dataset, MEG data showing
# event-related activity in response to somatosensory stimuli. We construct epochs
# around these events in the time window [-1.5, 1.0] seconds.

# %%

# Load data
data_path = somato.data_path()
raw_fname = data_path / "sub-01" / "meg" / "sub-01_task-somato_meg.fif"
raw = mne.io.read_raw_fif(raw_fname)
events = mne.find_events(raw, stim_channel="STI 014")

# Pre-processing
raw.pick("grad").load_data()  # focus on gradiometers
raw.filter(1, 35)
raw, events = raw.resample(sfreq=100, events=events)  # reduce compute time

# Construct epochs around events
epochs = mne.Epochs(
    raw, events, event_id=1, tmin=-1.5, tmax=1.0, baseline=(-0.5, 0), preload=True
)
epochs = epochs[:30]  # select a subset of epochs to speed up computation

########################################################################################
# Assessing connectivity in non-evoked data
# -----------------------------------------
#
# We will first demonstrate how connectivity can be assessed from non-evoked data. In
# this example, we use data from the pre-trial period of [-1.5, -0.5] seconds. We
# compute time-resolved Fourier coefficients of the data using the
# :meth:`~mne.Epochs.compute_tfr` method with ``output="complex"``.
#
# Next, we pass these coefficients to
# :func:`~mne_connectivity.spectral_connectivity_epochs` to compute connectivity using
# the imaginary part of coherency (``imcoh``). Our indices specify that connectivity
# between all unique pairs of channels should be computed.

# %%

# Compute time-resolved Fourier coefficients for pre-trial data
freqs = np.arange(4, 23, 2)
tfr_kwargs = dict(
    freqs=freqs, n_cycles=freqs / 2.0, method="morlet", decim=3, output="complex"
)
pretrial_coeffs = epochs.compute_tfr(tmin=None, tmax=-0.5, **tfr_kwargs)

# Compute connectivity for pre-trial data
indices = np.tril_indices(epochs.info["nchan"], k=-1)  # all unique connections
pretrial_con = spectral_connectivity_epochs(
    pretrial_coeffs, method="imcoh", indices=indices
)
pretrial_con = np.abs(pretrial_con.get_data()).mean(
    axis=(0, 2)  # average connections and timepoints
)

########################################################################################
# Next, we generate the surrogate data by passing the coefficients into the
# :func:`~mne_connectivity.make_surrogate_resting_data` function. This approach
# destroys the data's covariance structure by randomly shuffling the order of epochs,
# independently for each channel. To get a reliable estimate of the baseline
# connectivity, we perform this shuffling procedure :math:`\text{n}_{\text{shuffle}}`
# times, producing :math:`\text{n}_{\text{shuffle}}` surrogate datasets. We can then
# iterate over these shuffles and compute the connectivity for each one.

# %%

# Generate pre-trial surrogate data
n_shuffles = 100  # recommended is >= 1,000; limited here to reduce compute time
pretrial_surrogates = make_surrogate_resting_data(
    pretrial_coeffs, n_shuffles=n_shuffles, rng_seed=42
)

# Compute connectivity for pre-trial surrogate data
pretrial_surrogate_con = []
for shuffle_i, surrogate in enumerate(pretrial_surrogates, 1):
    print(f"Computing connectivity for shuffle {shuffle_i} of {n_shuffles}")
    surrogate_con = spectral_connectivity_epochs(
        surrogate, method="imcoh", indices=indices, verbose=False
    )
    pretrial_surrogate_con.append(
        np.abs(surrogate_con.get_data()).mean(axis=(0, 2))
    )  # average connections and timepoints
pretrial_surrogate_con = np.array(pretrial_surrogate_con)

########################################################################################
# We can plot the connectivity of the pre-trial data against the surrogate data,
# averaged over all shuffles. This shows a strong degree of coupling in the alpha band
# (~8-12 Hz), with weaker coupling in the lower range of the beta band (~13-20 Hz). A
# simple visual inspection shows that connectivity in the alpha and beta bands are above
# the baseline level of connectivity estimated from the surrogate data. However, we need
# to confirm this statistically.

# %%

# Plot pre-trial actual vs. surrogate connectivity
fig, ax = plt.subplots(1, 1)
ax.plot(freqs, pretrial_surrogate_con.mean(axis=0), linestyle="--", label="Surrogate")
ax.plot(freqs, pretrial_con, label="Original")
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Connectivity (A.U.)")
ax.set_title("All-to-all connectivity | Pre-trial ")
ax.legend()

########################################################################################
# Assessing the statistical significance of our connectivity estimates can be done with
# the following simple procedure :footcite:`PellegriniEtAl2023`
#
# :math:`p=\LARGE{\frac{\Sigma_{s=1}^Sc_s}{S}}` ,
#
# :math:`c_s=\{1\text{ if }\text{Con}\leq\text{Con}_{\text{s}}\text{ },\text{ }0
# \text{ if otherwise }` ,
#
# where: :math:`p` is our p-value; :math:`s` is a given shuffle iteration of :math:`S`
# total shuffles; and :math:`c` is a binary indicator of whether the true connectivity,
# :math:`\text{Con}`, is greater than the surrogate connectivity,
# :math:`\text{Con}_{\text{s}}`, for a given shuffle.
#
# Note that for connectivity methods which produce negative scores (e.g., imaginary part
# of coherency, time-reversed Granger causality, etc...), you should take the absolute
# values before testing. Similar adjustments should be made for methods that produce
# scores centred around non-zero values (e.g., 0.5 for directed phase lag index).
#
# Below, we determine the statistical significance of connectivity in the lower beta
# band. We simplify this by averaging over all connections and corresponding frequency
# bins. We could of course also test the significance of each connection, each frequency
# bin, or other frequency bands such as the alpha band. Naturally, any tests involving
# multiple connections, frequencies, and/or times should be corrected for multiple
# comparisons.
#
# The test confirms our visual inspection, showing that connectivity in the lower beta
# band is significantly above the baseline level of connectivity at an alpha of 0.05,
# which we can take as evidence of genuine interactions in this frequency band.

# %%

# Find indices of lower beta frequencies
beta_freqs = np.where((freqs >= 13) & (freqs <= 20))[0]

# Compute lower beta connectivity for pre-trial data
beta_con_pretrial = pretrial_con[beta_freqs].mean()

# Compute lower beta connectivity for surrogate data
beta_con_pretrial_surrogate = pretrial_surrogate_con[:, beta_freqs].mean(axis=1)

# Compute p-value for pre-trial lower beta coupling
p_val = np.sum(beta_con_pretrial <= beta_con_pretrial_surrogate) / n_shuffles
print(f"P = {p_val:.2f}")

########################################################################################
# Assessing connectivity in evoked data
# -------------------------------------
#
# When generating surrogate data, it is important to distinguish non-evoked data (e.g.,
# resting-state, pre/inter-trial data) from evoked data (where a stimulus is presented
# or an action performed at a set time during each epoch). Critically, evoked data
# contains a temporal structure that is consistent across epochs, and thus shuffling
# epochs across channels (as is done in
# :func:`~mne_connectivity.make_surrogate_resting_data`) will fail to adequately disrupt
# the covariance structure.
#
# Any connectivity estimates will therefore overestimate the baseline connectivity in
# your data, increasing the likelihood of type II errors (see the Notes section of
# :func:`~mne_connectivity.make_surrogate_resting_data` for more information, and see
# the section :ref:`inappropriate-surrogate-data` for a demonstration).
#
# In cases where you want to assess connectivity in evoked data, you can use the
# alternative :func:`~mne_connectivity.make_surrogate_evoked_data` function. This
# approach involves cutting the time series at a random point and reversing the cut
# portion, independently for each epoch and channel.
#
# .. admonition:: Supported data types
#
#    While :func:`~mne_connectivity.make_surrogate_resting_data` supports
#    :class:`~mne.Epochs`, :class:`~mne.time_frequency.EpochsSpectrum`, and
#    :class:`~mne.time_frequency.EpochsTFR` objects,
#    :func:`~mne_connectivity.make_surrogate_evoked_data` data does not support
#    :class:`~mne.time_frequency.EpochsSpectrum` objects, as there is no time dimension
#    to manipulate.
#
#    If you want to compare the significance of connectivity estimates for evoked data
#    derived from an :class:`~mne.time_frequency.EpochsSpectrum` object, you can first
#    generate surrogates from the :class:`~mne.Epochs` data, and then compute the
#    :class:`~mne.time_frequency.EpochsSpectrum` representations from these.
#
# Again, there is pronounced alpha coupling (stronger than in the pre-trial data) and
# weaker beta coupling, both of which appear to be above the baseline level of
# connectivity.

# %%

# Compute time-resolved Fourier coefficients for post-stimulus data
poststim_coeffs = epochs.compute_tfr(tmin=0.0, tmax=None, **tfr_kwargs)

# Compute connectivity for post-stimulus data
poststim_con = spectral_connectivity_epochs(
    poststim_coeffs, method="imcoh", indices=indices, verbose=False
)
poststim_con = np.abs(poststim_con.get_data()).mean(
    axis=(0, 2)  # average connections and timepoints
)

# Generate post-stimulus surrogate data
poststim_surrogates = make_surrogate_evoked_data(
    poststim_coeffs, n_shuffles=n_shuffles, rng_seed=42
)

# Compute connectivity for post-stimulus surrogate data
poststim_surrogate_con = []
for shuffle_i, surrogate in enumerate(poststim_surrogates, 1):
    print(f"Computing connectivity for shuffle {shuffle_i} of {n_shuffles}")
    surrogate_con = spectral_connectivity_epochs(
        surrogate, method="imcoh", indices=indices, verbose=False
    )
    poststim_surrogate_con.append(
        np.abs(surrogate_con.get_data()).mean(axis=(0, 2))
    )  # average connections and timepoints
poststim_surrogate_con = np.array(poststim_surrogate_con)

# Plot post-stimulus actual vs. surrogate connectivity
fig, ax = plt.subplots(1, 1)
ax.plot(freqs, poststim_surrogate_con.mean(axis=0), linestyle="--", label="Surrogate")
ax.plot(freqs, poststim_con, label="Original")
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Connectivity (A.U.)")
ax.set_title("All-to-all connectivity | Post-stimulus")
ax.legend()

########################################################################################
# This is also confirmed by statistical testing, with connectivity in the lower beta
# band being significantly above the baseline level of connectivity.

# %%

# Compute lower beta connectivity for post-stimulus data
beta_con_poststim = poststim_con[beta_freqs].mean()

# Compute lower beta connectivity for surrogate data
beta_con_poststim_surrogate = poststim_surrogate_con[:, beta_freqs].mean(axis=0)

# Compute p-value for post-stimulus lower beta coupling
p_val = np.sum(beta_con_poststim <= beta_con_poststim_surrogate) / n_shuffles
print(f"P = {p_val:.2f}")

########################################################################################
# .. _inappropriate-surrogate-data:
#
# Generating surrogate connectivity from inappropriate data
# ---------------------------------------------------------
# We discussed above how surrogates generated by
# :func:`~mne_connectivity.make_surrogate_resting_data` from evoked data risk
# overestimating the degree of baseline connectivity. We demonstrate this below by
# generating surrogates from the post-stimulus data using this inappropriate method.

# %%

# Generate surrogates from evoked data using wrong function
bad_poststim_surrogates = make_surrogate_resting_data(
    poststim_coeffs, n_shuffles=n_shuffles, rng_seed=44
)

# Compute connectivity for bad evoked surrogate data
bad_surrogate_con = []
for shuffle_i, surrogate in enumerate(bad_poststim_surrogates, 1):
    print(f"Computing connectivity for shuffle {shuffle_i} of {n_shuffles}")
    surrogate_con = spectral_connectivity_epochs(
        surrogate, method="imcoh", indices=indices, verbose=False
    )
    bad_surrogate_con.append(
        np.abs(surrogate_con.get_data()).mean(axis=(0, 2))
    )  # average connections and timepoints
bad_surrogate_con = np.array(bad_surrogate_con)

########################################################################################
# Plotting the post-stimulus connectivity against the estimates from both sets of
# surrogate data, we see that the surrogate data from the inappropriate method
# (:func:`~mne_connectivity.make_surrogate_resting_data`) greatly overestimates the
# baseline connectivity in the alpha band.
#
# Although in this case the alpha connectivity was still far above the baseline from the
# evoked surrogates, this will not always be the case, and you can see how this risks
# false negative assessments that connectivity is not significantly different from
# baseline. Surrogate connectivity estimates from evoked data should therefore always be
# generated using the :func:`~mne_connectivity.make_surrogate_evoked_data` function.

# %%

# Plot post-stimulus actual vs. surrogate connectivity
fig, ax = plt.subplots(1, 1)
ax.plot(
    freqs,
    poststim_surrogate_con.mean(axis=0),
    linestyle="--",
    label="Surrogate\n(make_surrogate_evoked_data)",
)
ax.plot(
    freqs,
    bad_surrogate_con.mean(axis=0),
    color="C3",
    linestyle="--",
    label="Surrogate\n(make_surrogate_resting_data)",
)
ax.plot(freqs, poststim_con, color="C1", label="Original")
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Connectivity (A.U.)")
ax.set_title("All-to-all connectivity | Post-stimulus")
ax.legend()

########################################################################################
# Assessing connectivity on a group-level
# ---------------------------------------
#
# While our focus here has been on assessing the significance of connectivity on a
# single recording-level, we may also want to determine whether group-level connectivity
# estimates are significantly different from baseline. For this, we can generate
# surrogates and estimate connectivity alongside the original signals for each piece of
# data.
#
# There are multiple ways to assess the statistical significance. For example, we can
# compute p-values for each piece of data using the approach above and combine them for
# the nested data (e.g., across recordings, subjects, etc...) using Stouffer's method
# :footcite:`DowdingHaufe2018`.
#
# Alternatively, we could take the average of the surrogate connectivity estimates
# across all shuffles for each piece of data and compare them to the original
# connectivity estimates in a paired test. The :mod:`scipy.stats` and :mod:`mne.stats`
# modules have many such tools for testing this, e.g., :func:`scipy.stats.ttest_1samp`,
# :func:`mne.stats.permutation_t_test`, etc...
#
# Altogether, surrogate connectivity estimates are a powerful tool for assessing the
# significance of connectivity estimates, both on a single recording- and group-level.

########################################################################################
# References
# ----------
# .. footbibliography::

# %%
