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

from multiprocessing import cpu_count

import matplotlib.pyplot as plt
import mne
import numpy as np
from mne.datasets import somato

from mne_connectivity import make_surrogate_data, spectral_connectivity_epochs

n_jobs = cpu_count() // 4

########################################################################################
# Background
# ----------
#
# When performing connectivity analyses, we often want to know whether the results we
# observe reflect genuine interactions between signals. We can assess this by performing
# statistical tests between our connectivity estimates and a 'baseline' level of
# connectivity. However, due to factors such as background noise and sample
# size-dependent biases (see e.g. :footcite:`VinckEtAl2010`), it is often not
# appropriate to treat 0 as this baseline. Therefore, we need a way to estimate the
# baseline level of connectivity.
#
# One approach is to manipulate the original data in such a way that the covariance
# structure is destroyed, creating surrogate data. Connectivity estimates from the
# original and surrogate data can then be compared to determine whether the original
# data contains significant interactions.
#
# Such surrogate data can be easily generated in MNE using the
# :func:`~mne_connectivity.make_surrogate_data` function, which shuffles epoched data
# independently across channels :footcite:`PellegriniEtAl2023` (see the Notes section of
# the function for more information). In this example, we will demonstrate how surrogate
# data can be created, and how you can use this to assess the statistical significance
# of your connectivity estimates.

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
# compute Fourier coefficients of the data using the :meth:`~mne.Epochs.compute_psd`
# method with ``output="complex"`` (note that this requires ``mne >= 1.8``).
#
# Next, we pass these coefficients to
# :func:`~mne_connectivity.spectral_connectivity_epochs` to compute connectivity using
# the imaginary part of coherency (``imcoh``). Our indices specify that connectivity
# should be computed between all pairs of channels.

# %%

# Compute Fourier coefficients for pre-trial data
fmin, fmax = 3, 23
pretrial_coeffs = epochs.compute_psd(
    fmin=fmin, fmax=fmax, tmin=None, tmax=-0.5, output="complex"
)
freqs = pretrial_coeffs.freqs

# Compute connectivity for pre-trial data
indices = np.tril_indices(epochs.info["nchan"], k=-1)  # all-to-all connectivity
pretrial_con = spectral_connectivity_epochs(
    pretrial_coeffs, method="imcoh", indices=indices
)

########################################################################################
# Next, we generate the surrogate data by passing the Fourier coefficients into the
# :func:`~mne_connectivity.make_surrogate_data` function. To get a reliable estimate of
# the baseline connectivity, we perform this shuffling procedure
# :math:`\text{n}_{\text{shuffle}}` times, producing :math:`\text{n}_{\text{shuffle}}`
# surrogate datasets. We can then iterate over these shuffles and compute the
# connectivity for each one.

# %%

# Generate surrogate data
n_shuffles = 100  # recommended is >= 1,000; limited here to reduce compute time
pretrial_surrogates = make_surrogate_data(
    pretrial_coeffs, n_shuffles=n_shuffles, rng_seed=44
)

# Compute connectivity for surrogate data
surrogate_con = []
for shuffle_i, surrogate in enumerate(pretrial_surrogates):
    print(f"Computing connectivity for shuffle {shuffle_i+1} of {n_shuffles}")
    surrogate_con.append(
        spectral_connectivity_epochs(
            surrogate, method="imcoh", indices=indices, n_jobs=n_jobs, verbose=False
        )
    )

########################################################################################
# We can plot the all-to-all connectivity of the pre-trial data against the surrogate
# data, averaged over all shuffles. This shows a strong degree of coupling in the alpha
# band (~8-12 Hz), with weaker coupling in the lower range of the beta band (~13-20 Hz).
# A simple visual inspection shows that connectivity in the alpha and beta bands are
# above the baseline level of connectivity estimated from the surrogate data. However,
# we need to confirm this statistically.

# %%

# Plot pre-trial vs. surrogate connectivity
fig, ax = plt.subplots(1, 1)
ax.plot(
    freqs,
    np.abs([surrogate.get_data() for surrogate in surrogate_con]).mean(axis=(0, 1)),
    linestyle="--",
    label="Surrogate",
)
ax.plot(freqs, np.abs(pretrial_con.get_data()).mean(axis=0), label="Original")
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
# band using an alpha of 0.05. Naturally, any tests involving multiple connections,
# frequencies, and/or times should be corrected for multiple comparisons. Here however,
# we average over all connections and frequencies.
#
# The test confirms our visual inspection, showing that connectivity in the lower beta
# band is significantly above the baseline level of connectivity, which we can take as
# evidence of genuine interactions in this frequency band.

# %%

# Find indices of lower beta frequencies
beta_freqs = np.where((freqs >= 13) & (freqs <= 20))[0]

# Compute lower beta connectivity for pre-trial data (average connections and freqs)
beta_con_pretrial = np.abs(pretrial_con.get_data()[:, beta_freqs]).mean(axis=(0, 1))

# Compute lower beta connectivity for surrogate data (average connections and freqs)
beta_con_surrogate = np.abs(
    [surrogate.get_data()[:, beta_freqs] for surrogate in surrogate_con]
).mean(axis=(1, 2))

# Compute p-value for pre-trial lower beta coupling
alpha = 0.05
p_val = np.sum(beta_con_pretrial <= beta_con_surrogate) / n_shuffles
print(f"P < {alpha}") if p_val < alpha else print(f"P > {alpha}")

########################################################################################
# Assessing connectivity in evoked data
# -------------------------------------
#
# When generating surrogate data, it is important to distinguish non-evoked data (e.g.,
# resting-state, pre/inter-trial data) from evoked data (where a stimulus is presented
# or an action performed at a set time during each epoch). Critically, evoked data
# contains a temporal structure that is consistent across epochs, and thus shuffling
# epochs across channels will fail to adequately disrupt the covariance structure.
#
# Any connectivity estimates will therefore overestimate the baseline connectivity in
# your data, increasing the likelihood of type II errors (see the Notes section of
# :func:`~mne_connectivity.make_surrogate_data` for more information, and see the final
# section of this example for a demonstration).
#
# **In cases where you want to assess connectivity in evoked data, you can use
# surrogates generated from non-evoked data (of the same subject).** Here we do just
# that, comparing connectivity estimates from the pre-trial surrogates to the evoked,
# post-stimulus response ([0, 1] second).
#
# Again, there is pronounced alpha coupling (stronger than in the pre-trial data) and
# weaker beta coupling, both of which appear to be above the baseline level of
# connectivity.

# %%

# Compute Fourier coefficients for post-stimulus data
poststim_coeffs = epochs.compute_psd(
    fmin=fmin, fmax=fmax, tmin=0, tmax=None, output="complex"
)

# Compute connectivity for post-stimulus data
poststim_con = spectral_connectivity_epochs(
    poststim_coeffs, method="imcoh", indices=indices
)

# Plot post-stimulus vs. (pre-trial) surrogate connectivity
fig, ax = plt.subplots(1, 1)
ax.plot(
    freqs,
    np.abs([surrogate.get_data() for surrogate in surrogate_con]).mean(axis=(0, 1)),
    linestyle="--",
    label="Surrogate",
)
ax.plot(freqs, np.abs(poststim_con.get_data()).mean(axis=0), label="Original")
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Connectivity (A.U.)")
ax.set_title("All-to-all connectivity | Post-stimulus")
ax.legend()

########################################################################################
# This is also confirmed by statistical testing, with connectivity in the lower beta
# band being significantly above the baseline level of connectivity. Thus, using
# surrogate connectivity estimates from non-evoked data provides a reliable baseline for
# assessing connectivity in evoked data.

# %%

# Compute lower beta connectivity for post-stimulus data (average connections and freqs)
beta_con_poststim = np.abs(poststim_con.get_data()[:, beta_freqs]).mean(axis=(0, 1))

# Compute p-value for post-stimulus lower beta coupling
p_val = np.sum(beta_con_poststim <= beta_con_surrogate) / n_shuffles
print(f"P < {alpha}") if p_val < alpha else print(f"P > {alpha}")

########################################################################################
# Generating surrogate connectivity from inappropriate data
# ---------------------------------------------------------
#
# We discussed above how surrogates generated from evoked data risk overestimating the
# degree of baseline connectivity. We demonstrate this below by generating surrogates
# from the post-stimulus data.

# %%

# Generate surrogates from evoked data
poststim_surrogates = make_surrogate_data(
    poststim_coeffs, n_shuffles=n_shuffles, rng_seed=44
)

# Compute connectivity for evoked surrogate data
bad_surrogate_con = []
for shuffle_i, surrogate in enumerate(poststim_surrogates):
    print(f"Computing connectivity for shuffle {shuffle_i+1} of {n_shuffles}")
    bad_surrogate_con.append(
        spectral_connectivity_epochs(
            surrogate, method="imcoh", indices=indices, n_jobs=n_jobs, verbose=False
        )
    )

########################################################################################
# Plotting the post-stimulus connectivity against the estimates from the non-evoked and
# evoked surrogate data, we see that the evoked surrogate data greatly overestimates the
# baseline connectivity in the alpha band.
#
# Although in this case the alpha connectivity was still far above the baseline from the
# evoked surrogates, this will not always be the case, and you can see how this risks
# false negative assessments that connectivity is not significantly different from
# baseline.

# %%

# Plot post-stimulus vs. evoked and non-evoked surrogate connectivity
fig, ax = plt.subplots(1, 1)
ax.plot(
    freqs,
    np.abs([surrogate.get_data() for surrogate in surrogate_con]).mean(axis=(0, 1)),
    linestyle="--",
    label="Surrogate (pre-stimulus)",
)
ax.plot(
    freqs,
    np.abs([surrogate.get_data() for surrogate in bad_surrogate_con]).mean(axis=(0, 1)),
    color="C3",
    linestyle="--",
    label="Surrogate (post-stimulus)",
)
ax.plot(
    freqs, np.abs(poststim_con.get_data()).mean(axis=0), color="C1", label="Original"
)
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
# Therefore, surrogate connectivity estimates are a powerful tool for assessing the
# significance of connectivity estimates, both on a single recording- and group-level.

########################################################################################
# References
# ----------
# .. footbibliography::
