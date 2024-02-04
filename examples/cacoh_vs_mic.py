"""
=====================================
Comparison of coherency-based methods
=====================================

This example demonstrates how canonical coherency (CaCoh)
:footcite:`VidaurreEtAl2019` - a multivariate method based on coherency - can
be used to compute connectivity between whole sets of sensors, alongside
spatial patterns of the connectivity.
"""

# Authors: Thomas S. Binns <t.s.binns@outlook.com>
#          Mohammad Orabe <orabe.mhd@gmail.com>
# License: BSD (3-clause)

# %%
import numpy as np
from matplotlib import pyplot as plt

import mne
from mne_connectivity import seed_target_indices, spectral_connectivity_epochs

# %%


def simulate_connectivity(
    n_seeds: int,
    n_targets: int,
    freq_band: tuple[int, int],
    n_epochs: int,
    n_times: int,
    sfreq: int,
    snr: float,
    connection_delay: int,
    rng_seed: int | None = None,
) -> np.ndarray:
    """Simulates signals interacting in a given frequency band.

    Parameters
    ----------
    n_seeds : int
        Number of seed channels to simulate.

    n_targets : int
        Number of target channels to simulate.

    freq_band : tuple of int, int
        Frequency band where the connectivity should be simulated, where the
        first entry corresponds to the lower frequency, and the second entry to
        the higher frequency.

    n_epochs : int
        Number of epochs in the simulated data.

    n_times : int
        Number of timepoints each epoch of the simulated data.

    sfreq : int
        Sampling frequency of the simulated data, in Hz.

    snr : float
        Signal-to-noise ratio of the simulated data.

    connection_delay :
        Number of timepoints for the delay of connectivity between the seeds
        and targets. If > 0, the target data is a delayed form of the seed data
        by this many timepoints.

    rng_seed : int | None (default None)
        Seed to use for the random number generator. If `None`, no seed is
        specified.

    Returns
    -------
    data : numpy.ndarray
        The simulated data stored in an array. The channels are arranged
        according to seeds, then targets.
    """
    if rng_seed is not None:
        np.random.seed(rng_seed)

    n_channels = n_seeds + n_targets
    trans_bandwidth = 1  # Hz

    # simulate signal source at desired frequency band
    signal = np.random.randn(1, n_epochs * n_times + connection_delay)
    signal = mne.filter.filter_data(
        data=signal,
        sfreq=sfreq,
        l_freq=freq_band[0],
        h_freq=freq_band[1],
        l_trans_bandwidth=trans_bandwidth,
        h_trans_bandwidth=trans_bandwidth,
        fir_design="firwin2",
        verbose=False,
    )

    # simulate noise for each channel
    noise = np.random.randn(n_channels, n_epochs * n_times + connection_delay)

    # create data by projecting signal into noise
    data = (signal * snr) + (noise * (1 - snr))

    # shift target data by desired delay
    if connection_delay > 0:
        # shift target data
        data[n_seeds:, connection_delay:] = data[n_seeds:, : n_epochs * n_times]
        # remove extra time
        data = data[:, : n_epochs * n_times]

    # reshape data into epochs
    data = data.reshape(n_channels, n_epochs, n_times)
    data = data.transpose((1, 0, 2))  # (epochs x channels x times)

    return data


# %%

# Define simulation parameters
n_seeds = 3
n_targets = 3
n_channels = n_seeds + n_targets
n_epochs = 10
n_times = 200  # samples
sfreq = 100  # Hz
snr = 0.7
rng_seed = 44

# Generate simulated data
data_delay = simulate_connectivity(
    n_seeds=n_seeds,
    n_targets=n_targets,
    freq_band=(10, 12),  # 10-12 Hz interaction
    n_epochs=n_epochs,
    n_times=n_times,
    sfreq=sfreq,
    snr=snr,
    connection_delay=2,  # samples
    rng_seed=42,
)

data_no_delay = simulate_connectivity(
    n_seeds=n_seeds,
    n_targets=n_targets,
    freq_band=(23, 25),  # 23-25 Hz interaction
    n_epochs=n_epochs,
    n_times=n_times,
    sfreq=sfreq,
    snr=snr,
    connection_delay=0,  # samples
    rng_seed=44,
)

# Combine data into a single array
data = np.concatenate((data_delay, data_no_delay), axis=1)

# %%

# Generate connectivity indices
seeds = np.concatenate(
    (np.arange(n_seeds), np.arange(n_channels, n_seeds + n_channels))
)
targets = np.concatenate(
    (np.arange(n_seeds, n_channels), np.arange(n_channels + n_seeds, n_channels * 2))
)

bivar_indices = (seeds, targets)
multivar_indices = ([seeds], [targets])

# Compute CaCoh & MIC
(cacoh, mic) = spectral_connectivity_epochs(
    data,
    method=["cacoh", "mic"],
    indices=multivar_indices,
    sfreq=sfreq,
    fmin=3,
    fmax=35,
)

# %%

fig, axis = plt.subplots(1, 1)
axis.plot(cacoh.freqs, np.abs(cacoh.get_data()[0]), linewidth=2, label="CaCoh")
axis.plot(
    mic.freqs, np.abs(mic.get_data()[0]), linewidth=2, label="MIC", linestyle="--"
)
axis.set_xlabel("Frequency (Hz)")
axis.set_ylabel("Connectivity (A.U.)")
axis.annotate("Non-zero\ntime lag\ninteraction", xy=(13, 0.85))
axis.annotate("Zero\ntime lag\ninteraction", xy=(27, 0.85))
axis.legend(loc="upper left")
fig.suptitle("CaCoh vs. MIC\nNon-zero & zero time lags")

# %%

# Compute Coh & ImCoh
(coh, imcoh) = spectral_connectivity_epochs(
    data,
    method=["coh", "imcoh"],
    indices=bivar_indices,
    sfreq=sfreq,
    fmin=3,
    fmax=35,
)

coh_mean = np.mean(coh.get_data(), axis=0)
imcoh_mean = np.mean(np.abs(imcoh.get_data()), axis=0)

coh_mean_subbed = coh_mean - np.mean(coh_mean)
imcoh_mean_subbed = imcoh_mean - np.mean(imcoh_mean)

fig, axis = plt.subplots(1, 1)
axis.plot(coh.freqs, coh_mean_subbed, linewidth=2, label="Coh")
axis.plot(imcoh.freqs, imcoh_mean_subbed, linewidth=2, label="ImCoh", linestyle="--")
axis.set_xlabel("Frequency (Hz)")
axis.set_ylabel("Mean-corrected connectivity (A.U.)")
axis.annotate("Non-zero\ntime lag\ninteraction", xy=(13, 0.25))
axis.annotate("Zero\ntime lag\ninteraction", xy=(25, 0.25))
axis.legend(loc="upper left")
fig.suptitle("Coh vs. ImCoh\nNon-zero & zero time lags")

# %%

# Generate simulated data
data_10_12 = simulate_connectivity(
    n_seeds=n_seeds,
    n_targets=n_targets,
    freq_band=(10, 12),  # 10-12 Hz interaction
    n_epochs=n_epochs,
    n_times=n_times,
    sfreq=sfreq,
    snr=snr,
    connection_delay=1,  # samples
    rng_seed=42,
)

data_23_25 = simulate_connectivity(
    n_seeds=n_seeds,
    n_targets=n_targets,
    freq_band=(23, 25),  # 10-12 Hz interaction
    n_epochs=n_epochs,
    n_times=n_times,
    sfreq=sfreq,
    snr=snr,
    connection_delay=1,  # samples
    rng_seed=44,
)

# Combine data into a single array
data = np.concatenate((data_10_12, data_23_25), axis=1)

# Compute CaCoh & MIC
(cacoh, mic) = spectral_connectivity_epochs(
    data,
    method=["cacoh", "mic"],
    indices=multivar_indices,
    sfreq=sfreq,
    fmin=3,
    fmax=35,
)

fig, axis = plt.subplots(1, 1)
axis.plot(cacoh.freqs, np.abs(cacoh.get_data()[0]), linewidth=2, label="CaCoh")
axis.plot(
    mic.freqs, np.abs(mic.get_data()[0]), linewidth=2, label="MIC", linestyle="--"
)
axis.set_xlabel("Frequency (Hz)")
axis.set_ylabel("Connectivity (A.U.)")
axis.annotate("45°\ninteraction", xy=(12.5, 0.9))
axis.annotate("90°\ninteraction", xy=(26.5, 0.9))
axis.legend(loc="upper left")
fig.suptitle("CaCoh vs. MIC\n45° & 90° interactions")

# %%

# Compute Coh & ImCoh
(coh, imcoh) = spectral_connectivity_epochs(
    data,
    method=["coh", "imcoh"],
    indices=bivar_indices,
    sfreq=sfreq,
    fmin=3,
    fmax=35,
)

coh_mean = np.mean(coh.get_data(), axis=0)
imcoh_mean = np.mean(np.abs(imcoh.get_data()), axis=0)
coh_mean_subbed = coh_mean - np.mean(coh_mean)
imcoh_mean_subbed = imcoh_mean - np.mean(imcoh_mean)

fig, axis = plt.subplots(1, 1)
axis.plot(coh.freqs, coh_mean_subbed, linewidth=2, label="Coh")
axis.plot(imcoh.freqs, imcoh_mean_subbed, linewidth=2, label="ImCoh", linestyle="--")
axis.set_xlabel("Frequency (Hz)")
axis.set_ylabel("Mean-corrected connectivity (A.U.)")
axis.annotate("45°\ninteraction", xy=(12, 0.25))
axis.annotate("90°\ninteraction", xy=(26.5, 0.25))
axis.legend(loc="upper left")
fig.suptitle("Coh vs. ImCoh\n45° & 90° interactions")

# %%
