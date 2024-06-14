"""
==============================================================
Multivariate decomposition for efficient connectivity analysis
==============================================================

This example demonstrates how the tools in the decoding module can be used to
decompose data into the most relevant components of connectivity and used for
a computationally efficient multivariate analysis of connectivity, such as in
brain-computer interface (BCI) applications.
"""

# Author: Thomas S. Binns <t.s.binns@outlook.com>
# License: BSD (3-clause)
# sphinx_gallery_thumbnail_number = 2

# %%

import time

import mne
import numpy as np
from matplotlib import pyplot as plt
from mne import make_fixed_length_epochs
from mne.datasets.fieldtrip_cmc import data_path

from mne_connectivity import (
    make_signals_in_freq_bands,
    seed_target_indices,
    spectral_connectivity_epochs,
)
from mne_connectivity.decoding import CoherencyDecomposition

########################################################################################
# Background
# ----------
#
# Multivariate forms of signal analysis allow you to simultaneously consider
# the activity of multiple signals. In the case of connectivity, the
# interaction between multiple sensors can be analysed at once and the strongest
# components of this interaction captured in a lower-dimensional set of connectivity
# spectra. This approach brings not only practical benefits (e.g. easier
# interpretability of results from the dimensionality reduction), but can also offer
# methodological improvements (e.g. enhanced signal-to-noise ratio and reduced bias).
#
# Coherency-based methods are popular approaches for analysing connectivity, capturing
# correlation between signals in the frequency domain. Various coherency-based
# multivariate methods exist, including: canonical coherency (CaCoh; multivariate
# measure of coherency/coherence); and maximised imaginary coherency (MIC; multivariate
# measure of the imaginary part of coherency).
#
# These methods are described in detail in the following examples:
#  - comparison of coherency-based methods - :doc:`../compare_coherency_methods`
#  - CaCoh - :doc:`../cacoh`
#  - MIC - :doc:`../mic_mim`
#
# The CaCoh and MIC methods work by finding spatial filters that decompose the data into
# components of connectivity, and applying them to the data. With the implementations
# offered in :func:`~mne_connectivity.spectral_connectivity_epochs` and
# :func:`~mne_connectivity.spectral_connectivity_time`, the filters are fit for each
# frequency separately, and the filters are only applied to the same data they are fit
# on.
#
# Unfortunately, fitting filters for each frequency bin can be computationally
# expensive, which may prohibit the use of these techniques, e.g. in real-time BCI
# setups where the rapid analysis of data is paramount, or even in offline analyses
# with huge datasets.
#
# These issues are addressed by the
# :class:`~mne_connectivity.decoding.CoherencyDecomposition` class of the decoding
# module. Here, the filters are fit for a given frequency band collectively (not each
# frequency bin!) and are stored, allowing them to be applied to the same data they were
# fit on (e.g. for offline analyses of huge datasets) or to new data (e.g. for online
# analyses of streamed data).
#
# In this example, we show how the tools of the decoding module compare to the standard
# ``spectral_connectivity_...()`` functions in terms of their run time, and their
# ability to decompose data into connectivity components.

########################################################################################
# Case 1: Fitting to and transforming different data
# --------------------------------------------------
#
# We start by simulating some connectivity between two groups of signals at 15-20 Hz as
# 60 two-second-long epochs. To demonstrate the approach of fitting filters to one set
# of data and applying to another set of data, we will treat the first 30 epochs as the
# data on which we train the filters, and the last 30 epochs as the data we transform.
# We will use the CaCoh method, since zero time-lag interactions are not present (See
# :doc:`../compare_coherency_methods` for more information).

# %%

N_SEEDS = 10
N_TARGETS = 15

FMIN = 15
FMAX = 20

N_EPOCHS = 60

epochs = make_signals_in_freq_bands(
    n_seeds=N_SEEDS,
    n_targets=N_TARGETS,
    freq_band=(FMIN, FMAX),
    n_epochs=N_EPOCHS,
    n_times=200,
    sfreq=100,
    snr=0.2,
    rng_seed=44,
)

indices = (np.arange(N_SEEDS), np.arange(N_TARGETS) + N_SEEDS)

########################################################################################
# First, we use the standard CaCoh approach in
# :func:`~mne_connectivity.spectral_connectivity_epochs` to visualise the connectivity
# in the first 30 epochs. We also plot bivariate coherence to demonstrate the
# signal-to-noise enhancements this multivariate approach offers. As expected, we see a
# peak in connectivity at 15-20 Hz decomposed by the spatial filters.

# %%

# Connectivity profile of first 30 epochs (filters fit to these epochs)
con_cacoh_first = spectral_connectivity_epochs(
    epochs[: N_EPOCHS // 2],
    method="cacoh",
    indices=([indices[0]], [indices[1]]),
    fmin=5,
    fmax=35,
    rank=([3], [3]),
)
ax = plt.subplot(111)
ax.plot(con_cacoh_first.freqs, np.abs(con_cacoh_first.get_data()[0]), label="CaCoh")

# Connectivity profile of first 30 epochs (no filters)
con_coh_first = spectral_connectivity_epochs(
    epochs[: N_EPOCHS // 2],
    method="coh",
    indices=seed_target_indices(indices[0], indices[1]),
    fmin=5,
    fmax=35,
)
ax.plot(con_coh_first.freqs, np.mean(con_coh_first.get_data(), axis=0), label="Coh")
ax.axvspan(FMIN, FMAX, color="grey", alpha=0.2, label="Fitted freq. band")
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Connectivity (A.U.)")
ax.set_title("Epochs 0-30")
plt.legend()
plt.show()

########################################################################################
# The goal of the decoding module approach is to use the information from the first 30
# epochs to fit the filters, and then use these filters to extract the same components
# from the last 30 epochs.
#
# For this, we instantiate the
# :class:`~mne_connectivity.decoding.CoherencyDecomposition` class with: the
# information about the data being fit/transformed (using an :class:`~mne.Info` object);
# the type of connectivity we want to decompose (here CaCoh); the frequency band of the
# components we want to decompose (here 15-20 Hz); and the channel indices of the seeds
# and targets.
#
# Next, we call the :meth:`~mne_connectivity.decoding.CoherencyDecomposition.fit`
# method, passing in the first 30 epochs of data we want to fit the filters to. Once the
# filters are fit, we can apply them to the last 30 epochs using the
# :meth:`~mne_connectivity.decoding.CoherencyDecomposition.transform` method.
#
# The transformed data has shape ``(epochs x components*2 x times)``, where the new
# 'channels' are organised as the seed components, then target components. For
# convenience, the
# :meth:`~mne_connectivity.decoding.CoherencyDecomposition.get_transformed_indices`
# method can be used to get the ``indices`` of the transformed data for use in the
# ``spectral_connectivity_...()`` functions.

# %%

# Fit filters to first 30 epochs
cacoh = CoherencyDecomposition(
    info=epochs.info,
    method="cacoh",
    indices=indices,
    mode="multitaper",
    fmin=FMIN,
    fmax=FMAX,
    rank=(3, 3),
)
cacoh.fit(epochs[: N_EPOCHS // 2].get_data())

# Use filters to transform data from last 30 epochs
epochs_transformed = cacoh.transform(epochs[N_EPOCHS // 2 :].get_data())
indices_transformed = cacoh.get_transformed_indices()

########################################################################################
# We can now visualise the connectivity in the last 30 epochs of the transformed data,
# which for reference we will compare to connectivity in the last 30 epochs using
# filters fit to the data itself, as well as bivariate coherence to again demonstrate
# the signal-to-noise enhancements the multivariate approach offers.
#
# To compute connectivity of the transformed data, it is simply a case of passing to the
# ``spectral_connectivity_...()`` functions: the transformed data; the indices
# returned from
# :meth:`~mne_connectivity.decoding.CoherencyDecomposition.get_transformed_indices`; and
# the corresponding bivariate method (``"coh"`` and ``"cohy"`` for CaCoh; ``"imcoh"``
# for MIC).
#
# As you can see, the connectivity profile of the transformed data using filters fit on
# the first 30 epochs is very similar to the connectivity profile when using filters fit
# on the last 30 epochs. This shows that the filters are generalisable, able to extract
# the same components of connectivity which they were trained on from new data.

# %%

# Connectivity profile of last 30 epochs (filters fit to these epochs)
con_cacoh_last = spectral_connectivity_epochs(
    epochs[N_EPOCHS // 2 :],
    method="cacoh",
    indices=([indices[0]], [indices[1]]),
    fmin=5,
    fmax=35,
    rank=([3], [3]),
)
ax = plt.subplot(111)
ax.plot(
    con_cacoh_last.freqs,
    np.abs(con_cacoh_last.get_data()[0]),
    label="CaCoh (filters trained\non epochs 30-60)",
)

# Connectivity profile of last 30 epochs (no filters)
con_coh_last = spectral_connectivity_epochs(
    epochs[N_EPOCHS // 2 :],
    method="coh",
    indices=seed_target_indices(indices[0], indices[1]),
    fmin=5,
    fmax=35,
)
ax.plot(
    con_coh_last.freqs, np.mean(np.abs(con_coh_last.get_data()), axis=0), label="Coh"
)

# Connectivity profile of last 30 epochs (filters fit to first 30 epochs)
con_cacoh_last_from_first = spectral_connectivity_epochs(
    epochs_transformed,
    method="coh",
    indices=indices_transformed,
    fmin=5,
    fmax=35,
    sfreq=epochs.info["sfreq"],
)
ax.plot(
    con_cacoh_last_from_first.freqs,
    np.abs(con_cacoh_last_from_first.get_data()[0]),
    label="CaCoh (filters trained\non epochs 0-30)",
)
ax.axvspan(FMIN, FMAX, color="grey", alpha=0.2, label="Fitted freq. band")
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Connectivity (A.U.)")
ax.set_title("Epochs 30-60")
plt.legend()
plt.show()

########################################################################################
# In addition to assessing the validity of the approach, we can also look at the time
# taken to run the analysis. Below we present a scenario resembling an online sliding
# window approach typical of a BCI system. We consider the first 30 epochs to be the
# training data that the filters should be fit to, and the last 30 epochs to be the
# windows of data that the filters should be applied to, transforming and computing the
# connectivity of each window (epoch) of data sequentially.
#
# Doing so, we see that once the filters have been fit, it takes only a few milliseconds
# to transform each window of data and compute its connectivity.

# %%

cacoh = CoherencyDecomposition(
    info=epochs.info,
    method="cacoh",
    indices=indices,
    mode="multitaper",
    fmin=FMIN,
    fmax=FMAX,
    rank=(3, 3),
)

# Time fitting of filters
start_fit = time.time()
cacoh.fit(epochs[: N_EPOCHS // 2].get_data())
fit_duration = (time.time() - start_fit) * 1000

# Time transforming data of each epoch iteratively
start_transform = time.time()
for epoch in epochs[N_EPOCHS // 2 :]:
    epoch_transformed = cacoh.transform(epoch)
    spectral_connectivity_epochs(
        np.expand_dims(epoch_transformed, axis=0),
        method="coh",
        indices=indices_transformed,
        fmin=5,
        fmax=35,
        sfreq=epochs.info["sfreq"],
    )
transform_duration = (time.time() - start_transform) * 1000

# %%

print(f"Time to fit filters: {fit_duration:.0f} ms")
print(f"Time to transform data and compute connectivity: {transform_duration:.0f} ms")
print(f"Total time: {fit_duration + transform_duration:.0f} ms")

print(
    "\nTime to transform data and compute connectivity per epoch (window): ",
    f"{transform_duration/(N_EPOCHS//2):.0f} ms",
)

########################################################################################
# In contrast, here we follow the same sequential window approach, but fit filters to
# each window separately rather than using a pre-computed set. Naturally, the process of
# fitting and transforming the data for each window is considerably slower.
#
# Furthermore, given the noisy nature of single windows of data, there is a risk of
# overfitting the filters to this noise as opposed to the genuine interaction(s) of
# interest. This risk is mitigated by performing the initial filter fitting on a larger
# set of data.

# %%

# Time fitting and transforming data of each epoch iteratively
start_fit_transform = time.time()
for epoch in epochs[N_EPOCHS // 2 :]:
    spectral_connectivity_epochs(
        np.expand_dims(epoch, axis=0),
        method="cacoh",
        indices=([indices[0]], [indices[1]]),
        fmin=5,
        fmax=35,
        sfreq=epochs.info["sfreq"],
        rank=([3], [3]),
    )
fit_transform_duration = (time.time() - start_fit_transform) * 1000

# %%

print(
    f"Time to fit, transform, and compute connectivity: {fit_transform_duration:.0f} ms"
)

print(
    "\nTime to fit, transform, and compute connectivity per epoch (window): ",
    f"{fit_transform_duration/(N_EPOCHS//2):.0f} ms",
)

########################################################################################
# As a side note, it is important to consider that a multivariate approach may be as
# fast or even faster than a bivariate approach, depending on the number of connections
# and degree of rank subspace projection being performed.

# %%

# Time transforming data of each epoch iteratively
start = time.time()
for epoch in epochs[N_EPOCHS // 2 :]:
    spectral_connectivity_epochs(
        np.expand_dims(epoch, axis=0),
        method="coh",
        indices=seed_target_indices(indices[0], indices[1]),
        fmin=5,
        fmax=35,
        sfreq=epochs.info["sfreq"],
    )
duration = (time.time() - start) * 1000

# %%

print(f"Time to compute connectivity: {duration:.0f} ms")

print(
    "\nTime to compute connectivity per epoch (window): ",
    f"{duration/(N_EPOCHS//2):.0f} ms",
)

########################################################################################
# Case 2: Fitting to and transforming the same data
# -------------------------------------------------
#
# As mentioned above, the decoding module classes can also be used to transform the same
# data the filters are fit to. This is a similar process to that of the
# ``spectral_connectivity_...()`` functions, but with the increased efficiency of
# fitting filters to a single frequency band as opposed to each frequency bin.
#
# To demonstrate this approach, we will load some example MEG data and divide it into
# two-second-long epochs. We designate the left hemisphere sensors as the seeds and the
# right hemisphere sensors as the targets. Since this is sensor-space data, we will use
# the MIC method to analyse connectivity given its resilience to zero time-lag
# interactions (See :doc:`../compare_coherency_methods` for more information).

# %%

raw = mne.io.read_raw_ctf(data_path() / "SubjectCMC.ds")
raw.pick("mag")
raw.crop(50.0, 110.0).load_data()
raw.notch_filter(50)
raw.resample(100)

epochs = make_fixed_length_epochs(raw, duration=2.0).load_data()

# left hemisphere sensors
seeds = [idx for idx, ch_info in enumerate(epochs.info["chs"]) if ch_info["loc"][0] < 0]
# right hemisphere sensors
targets = [
    idx for idx, ch_info in enumerate(epochs.info["chs"]) if ch_info["loc"][0] > 0
]

########################################################################################
# There are two equivalent options for fitting and transforming the same data: 1)
# passing the data to the :meth:`~mne_connectivity.decoding.CoherencyDecomposition.fit`
# and :meth:`~mne_connectivity.decoding.CoherencyDecomposition.transform` methods
# sequentially; or 2) using the combined
# :meth:`~mne_connectivity.decoding.CoherencyDecomposition.fit_transform` method.
#
# We use the latter approach below, fitting the filters to the 15-20 Hz band and using
# the ``"imcoh"`` method in the call to the ``spectral_connectivity_...()`` functions.
# Plotting the results, we see a peak in connectivity at 15-20 Hz.

# %%

mic = CoherencyDecomposition(
    info=epochs.info,
    method="mic",
    indices=(seeds, targets),
    mode="multitaper",
    fmin=FMIN,
    fmax=FMAX,
    rank=(3, 3),
)

start = time.time()
epochs_transformed = mic.fit_transform(epochs.get_data())

con_mic_class = spectral_connectivity_epochs(
    epochs_transformed,
    method="imcoh",
    indices=mic.get_transformed_indices(),
    fmin=5,
    fmax=30,
    sfreq=epochs.info["sfreq"],
)
class_duration = time.time() - start

ax = plt.subplot(111)
ax.plot(
    con_mic_class.freqs,
    np.abs(con_mic_class.get_data()[0]),
    color="C2",
    label="MIC (decomposition\nclass)",
)
ax.axvspan(FMIN, FMAX, color="grey", alpha=0.2, label="Fitted freq. band")
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Connectivity (A.U.)")
plt.legend()
plt.show()

########################################################################################
# For comparison, we can also use the standard approach of the
# ``spectral_connectivity_...()`` functions, which shows a very similar connectivity
# profile in the 15-20 Hz frequency range (but not identical due to band- vs. bin-wise
# filter fitting approaches). Bivariate coherence is again shown to demonstrate the
# signal-to-noise enhancements the multivariate approach offers.

# %%

start = time.time()
con_mic_func = spectral_connectivity_epochs(
    epochs,
    method="mic",
    indices=([seeds], [targets]),
    fmin=5,
    fmax=30,
    rank=([3], [3]),
)
func_duration = time.time() - start

con_imcoh = spectral_connectivity_epochs(
    epochs,
    method="imcoh",
    indices=seed_target_indices(seeds, targets),
    fmin=5,
    fmax=30,
    rank=([3], [3]),
)

ax = plt.subplot(111)
ax.plot(
    con_mic_func.freqs,
    np.abs(con_mic_func.get_data()[0]),
    label="MIC (standard\nfunction)",
)
ax.plot(
    con_imcoh.freqs,
    np.mean(np.abs(con_imcoh.get_data()), axis=0),
    label="ImCoh",
)
ax.plot(
    con_mic_class.freqs,
    np.abs(con_mic_class.get_data()[0]),
    label="MIC (decomposition\nclass)",
)
ax.axvspan(FMIN, FMAX, color="grey", alpha=0.2, label="Fitted freq. band")
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Connectivity (A.U.)")
plt.legend()
plt.show()

########################################################################################
# As with the previous example, we can also compare the time taken to run the analyses.
# Here we see that the decomposition class is much faster than the
# ``spectral_connectivity_...()`` functions, thanks to the fact that the filters are fit
# to the entire frequency band and not each frequency bin.

# %%

print(
    "Time to fit, transform, and compute connectivity (decomposition class): "
    f"{class_duration:.2f} s"
)
print(
    f"Time to fit, transform, and compute connectivity (standard function): "
    f"{func_duration:.2f} s"
)

########################################################################################
# Limitations
# -----------
# Finally, it is important to discuss a key limitation of the decoding module approach:
# the need to define a specific frequency band. Defining this band requires some
# existing knowledge about your data or the oscillatory activity you are studying. This
# insight may come from a pilot study where a frequency band of interest was identified,
# a canonical frequency band defined in the literature, etc... In contrast, by fitting
# filters to each frequency bin, the standard ``spectral_connectivity_...()`` functions
# are more flexible.
#
# Additionally, by applying filters fit on one set of data to another, you are assuming
# that the connectivity components the filters are designed to extract are consistent
# across the two sets of data. However, this may not be the case if you are applying the
# filters to data from a distinct functional state where the spatial distribution of the
# components differs. Again, by fitting filters to each new set of data passed in, the
# standard ``spectral_connectivity_...()`` functions are more flexible, extracting
# whatever connectivity components are present in that data.
#
# On these points, we note that the ``spectral_connectivity_...()`` functions complement
# the decoding module classes well, offering a tool by which to explore your data to:
# identify possible frequency bands of interest; and identify the spatial distributions
# of connectivity components to determine if they are consistent across different
# portions of the data.
#
# Ultimately, there are distinct advantages and disadvantages to both approaches, and
# one may be more suitable than the other depending on your use case.

# %%
