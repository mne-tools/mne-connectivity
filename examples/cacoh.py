"""
====================================================
Compute multivariate measure of (absolute) coherence
====================================================

This example showcases the application of the Canonical Coherence (CaCoh) method, as detailed :footcite: `VidaurreEtAl2019`, for detecting neural synchronization in multivariate signal spaces.

The method maximizes the absolute value of the coherence between two sets of multivariate spaces directly in the frequency domain. For each frequency bin two spatial filters are computed in order to maximize the coherence between the projected components.
"""

# Authors: Mohammad Orabe <orabe.mhd@gmail.com>
#          Thomas S. Binns <t.s.binns@outlook.com>
#
# License: BSD (3-clause)

# %%
import numpy as np
from matplotlib import pyplot as plt

import mne
import mne_connectivity

###############################################################################
# Background
# ----------
#
# Multivariate connectivity methods have emerged as a sophisticated approach to
# understand the complex interplay between multiple brain regions simultaneously. These
# methods transcend the limitations of bivariate analyses, which focus on pairwise
# interactions, by offering a holistic view of brain network dynamics. However,
# challenges such as volume conduction and source mixing—where signals from distinct
# neural sources blend or appear falsely synchronized due to the conductive properties
# of brain tissue, complicate the interpretation of connectivity data. While other
# multivariate methods like [``MIC``](https://mne.tools/mne-connectivity/stable/auto_examples/mic_mim.html) by (:footcite:`EwaldEtAl2012`) specifically exclude
# zero time-lag interactions to avoid volume conduction artifacts, assuming these to be
# non-physiological, the Canonical Coherence (Cacoh) method can capture and analyze
# interactions between signals with both zero and non-zero time-lag.
#
# This capability allows Cacoh to  identify neural interactions that occur
# simultaneously, offering insights into connectivity that may be overlooked by other
# methods. The Cacoh method utilizes spatial filters to isolate true neural
# interactions. This approach not only simplifies the interpretation of complex neural
# interactions but also potentially enhances the signal-to-noise ratio, offering
# methodological advantages such as reduced bias from source mixing.
#
# CaCoh maximizes the absolute value of the coherence between the two multivariate
# spaces directly in the frequency domain, where for each frequency bin, two spatial
# filters are computed in order to maximize the coherence between the projected
# components.


###############################################################################
# Data Simulation
# ---------------
#
# The CaCoh method can be used to investigate the synchronization between two
# modalities. For instance it can be applied to optimize (in a non-invasive way) the
# cortico-muscular coherence between a central set of electroencephalographic (EEG)
# sensors and a peripheral set of electromyographic (EMG) electrodes, where each
# subspaces is multivariate CaCoh is capable of taking into account the fact that
# cortico-spinal interactions are multivariate in nature not only on the cortical level
# but also at the level of the spinal cord, where multiple afferent and efferent
# processes occur.
#
# CaCoh extends beyond analyzing cortico-muscular interactions. It is also applicable
# in various EEG/MEG/LFP research scenarios, such as studying the interactions between
# cortical (EEG/MEG) and subcortical activities, or in examining intracortical local
# field potentials (LFP).
#
# In this demo script, we will generates synthetic EEG signals. Let's define a function
# that enables the study of both zero-lag and non-zero-lag interactions by adjusting
# the parameter `connection_delay`.


# %%
def simulate_connectivity(
    n_seeds: int,
    n_targets: int,
    freq_band: tuple[int, int],
    n_epochs: int,
    n_times: int,
    sfreq: int,
    snr: float,
    connection_delay,
    rng_seed: int | None = None,
) -> mne.Epochs:
    """Simulates signals interacting in a given frequency band.

    Parameters
    ----------
    n_seeds : int
        Number of seed channels to simulate.

    n_targets : int
        Number of target channels to simulate.

    freq_band : tuple of int, int
        Frequency band where the connectivity should be simulated, where the first entry corresponds
        to the lower frequency, and the second entry to the higher frequency.

    n_epochs : int
        Number of epochs in the simulated data.

    n_times : int
        Number of timepoints each epoch of the simulated data.

    sfreq : int
        Sampling frequency of the simulated data, in Hz.

    snr : float
        Signal-to-noise ratio of the simulated data.

    connection_delay :
        Number of timepoints for the delay of connectivity between the seeds and targets. If > 0,
        the target data is a delayed form of the seed data by this many timepoints.

    rng_seed : int | None (default None)
        Seed to use for the random number generator. If `None`, no seed is specified.

    Returns
    -------
    epochs : mne.Epochs
        The simulated data stored in an Epochs object. The channels are arranged according to seeds,
        then targets.
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

    # store data in an MNE Epochs object
    ch_names = [f"{ch_i}_{freq_band[0]}_{freq_band[1]}" for ch_i in range(n_channels)]
    info = mne.create_info(
        ch_names=ch_names, sfreq=sfreq, ch_types="eeg", verbose=False
    )
    epochs = mne.EpochsArray(data=data, info=info, verbose=False)

    return epochs


# %%
def plot_absolute_coherency(conn_data, label):
    """Plot the absolute value of coherency across frequencies"""
    _, axis = plt.subplots()
    axis.plot(
        conn_data.freqs, np.abs(conn_data.get_data()[0]), linewidth=2, label=label
    )
    axis.set_xlabel("Frequency (Hz)")
    axis.set_ylabel("Absolute connectivity (A.U.)")
    plt.title("CaCoh")
    plt.legend(loc="upper right")
    plt.show()


# %%
# Set parameters
n_epochs = 10
n_times = 200
sfreq = 100
snr = 0.7
freq_bands = {
    "theta": [4.0, 8],
    "alpha": [8.0, 12],
    "beta": [12.0, 25],
    "Gamma": [30.0, 45.0],
}

n_seeds = 4
n_targets = 3
indices = ([np.arange(n_seeds)], [n_seeds + np.arange(n_targets)])

# %%we will generates synthetic EEG signals
# First we will simulate a small dataset that consists of 3 synthetic EEG sensors
# designed as seed channels and 4 synthetic EEG sensors designed as target channels
# Then we will consider two cases; one with zero- and one with non zero time-lag to
# explore the CaCoh of each frequency bin. The seed data is initially generated as
# noise. The target data are a band-pass filtered version of the seed channels.

# %%
# Case 1: Zero time-lag interactions.
#
# In our first scenario, we explore connectivity dynamics without any temporal
# separation between the seed and target channels, setting the connectivity delay to
# zero. This configuration will allow us to investigate instantaneous interactions,
# simulating conditions where neural signals are synchronized without time lag.
delay = 0

# Generate simulated data
con_data = simulate_connectivity(
    n_seeds=n_seeds,
    n_targets=n_targets,
    freq_band=freq_bands["beta"],
    n_epochs=n_epochs,
    n_times=n_times,
    sfreq=sfreq,
    snr=snr,
    connection_delay=delay,
    rng_seed=42,
)

# %%
# Compute the multivariate connectivity using the CaCoh method.
con = mne_connectivity.spectral_connectivity_epochs(
    con_data, indices=indices, method="cacoh"
)

# %%
# Plot the absolute coherence value for each frequency bin.
plot_absolute_coherency(con, "Zero-lag interaction")

# We observe a significant peak in the beta frequency band, indicating a high level of
# coherence. This suggests a strong synchronization between the seed and target
# channels within that frequency range. One might assume that such synchronization
# could be due to genuine neural connectivity. However, without phase lag, it's also
# possible that this result could stem from volume conduction or common reference
# artifacts, rather than direct physiological interactions.
#
# In the context of EEG analysis, volume conduction is a prevalent concern; yet, for
# example for LFP recordings, the spatial resolution is higher, and signals are less
# likely to be confounded by this phenomenon. Therefore, when interpreting such a peak
# in LFP data, one could be more confident that it reflects true neural interactions.


# %%
# Case 2: Non-zero time-lag interactions.
#
# For the exploration of non-zero time-lag interactions, we adjust the simulation to
# include a delay of 10 timepoints between the seed and target signals. This will model
# the temporal delays in neural communication.

delay = 10

# Generate new simulated data
con_data = simulate_connectivity(
    n_seeds=n_seeds,
    n_targets=n_targets,
    freq_band=freq_bands["beta"],
    n_epochs=n_epochs,
    n_times=n_times,
    sfreq=sfreq,
    snr=snr,
    connection_delay=delay,
    rng_seed=42,
)

# %%
# Compute the multivariate connectivity for the new simulated data.
con = mne_connectivity.spectral_connectivity_epochs(
    con_data, indices=indices, method="cacoh"
)

# %%
# Plot the absolute coherence value for each frequency bin.
plot_absolute_coherency(con, "Non-zero-lag interaction")

# We can see the coherence across frequencies with a notable peak also in the beta
# band, but the coherence values are overall a bit lower than in the zero-lag scenario.
# This illustrates the temporal delay introduced between seed and target signals,
# simulating a more realistic scenario where neuronal communications involve
# transmission delays (such as synaptic or axonal delays).


###############################################################################
# Theoretical description of canonical Coherence (CaCoh)
#
# In methematical terms, the Canonical Coherence (CaCoh) method aims to maximize the
# coherence between two signal spaces, :math:`A` and :math:`B`, each
# of dimension :math:`\(N_A\)` and :math:`\(N_B\)` respectively. In a practical
# scenario, :math:`A` might represent signals from EMG sensors, and :math:`B` from EEG
# sensors. The primary goal of CaCoh is to find real-valued linear combinations of
# signals from these spaces that maximize coherence at a specific frequency.
#
# This maximization is formulated as (Eq. of 8 in :footcite: `VidaurreEtAl2019`):
#
# :math:`\[ CaCoh = \lambda(\Phi)=\frac{\mathbf{a}^T \mathbf{D}(\Phi) \mathbf{b}}{\sqrt
# {\mathbf{a}^T \mathbf{a} \cdot \mathbf{b}^T \mathbf{b}}} \]`
#
# where :math:`\(\mathbf{D}(\Phi, \a, \b) = \mathbf{C}_{AA}^{-1/2} \mathbf{C}_{AB, \Phi}
# ^R \mathbf{C}_{BB}^{-1/2}\)`. Here, :math:`\(\mathbf{C}_{AB, \Phi}^R\)` denotes the
# real part of the cross-spectrum, while :math:`\(\mathbf{C}_{AA}\)` and :math:`\
# :math:`\(\beta\)`, respectively.
#
# The method inherently assumes instantaneous mixing of the signals, which justifies
# focusing on the real parts of the cross-spectral matrices. The complex Hermitian
# nature (where a square matrix is equal to its own conjugate transpose) of these
# matrices means that their imaginary components do not contribute to the maximization
# process and are thus typically set to zero.
#
# The analytical resolution of CaCoh leads to an eigenvalue problem (Eq. 12 of
# VidaurreEtAl2019):
#
# :math:`\[ \mathbf{D}(\Phi)^T \mathbf{D}(\Phi) \mathbf{b} = \lambda \mathbf{b} \]`
# :math:`\[ \mathbf{D}(\Phi) \mathbf{D}(\Phi)^T \mathbf{a} = \lambda \mathbf{a} \]`
#
# where :math:`\(\mathbf{a}\)` and :math:`\(\mathbf{b}\)` are the eigenvectors derived
# from the respective spaces :math:`\(\alpha\)` and :math:`\(\beta\)`.
# :math:`\(\lambda\)`, the maximal eigenvalue, represents the maximal CaCoh. The
# numerical estimation of the phase of coherence, where its absolute value is maximal,
# is achieved through a nonlinear search, emphasizing the method's robustness in
# identifying the most coherent signal combinations across different modalities.
#
# To provide insights into the locations of sources influencing connectivity, spatial
# patterns can be obtained through spatial filters. To identify the topographies
# corresponding to the spatial filters :math:`\alpha` and :math:`\beta`, the filters
# are multiplied by their respective real part of the cross-spectral matrix, as follows
# (Eq. 14 of :footcite: `VidaurreEtAl2019`):

# For :math:`\alpha`, calculate: :math:`t_{\boldsymbol{\alpha}} = \mathbf{C}_{A A}^R
# \boldsymbol{\alpha}`
# For :math:`\beta`, calculate: :math:`t_{\boldsymbol{\beta}} = \mathbf{C}_{B B}^R
# \boldsymbol{\beta}`

# These topographies represent the patterns of the sources with maximum coherence. The
# time courses of CaCoh components directly indicate the activity of neuronal sources.
# The spatial patterns, stored under the connectivity class's 'attrs['patterns']',
# assign a frequency-specific value to each seed and target channel. For simulated
# data, our focus is on coherence analysis without visualizing spatial patterns.
# An example for the visualization for the spatial patterns can be similarly
# accomplished using a the [``MIC``](https://mne.tools/mne-connectivity/stable/auto_examples/mic_mim.html) method (:footcite:`EwaldEtAl2012`).

###############################################################################
# Overfitting
# -----------
# The concern regarding overfitting arises when the spatial filters :math:`\(\alpha\)`
# and :math:`\(\beta\)`, designed to maximize Canonical Coherence (CaCoh), overly adapt
# to the specific dataset, compromising their applicability to new data. This is
# particularly relevant in high-dimensional datasets. To mitigate this, dimensionality
# reduction via Singular Value Decomposition (SVD) is applied to the real part of the
# cross-spectra in the spaces \(A\) and \(B\) before computing the spatial filters
# (Eqs. 14 & 15 of [1]). This process involves selecting singular vectors that preserve
# most of the data's information, ensuring that the derived filters are both effective
# and generalizable.
#
# The dimensionality of data can be controlled using the ``rank`` parameter, which by
# default assumes data is of full rank and does not reduce dimensionality. To
# accurately reflect the data's structure and avoid bias, it's important to choose a
# rank based on the expected number of significant components. This selection helps
# standardize connectivity estimates across different recordings, even when the number
# of channels varies. Note that this does not refer to the number of seeds and targets
# within a connection being identical, rather to the number of seeds and targets across
# connections.
#
# In the following example, we will create two datasets with a larger number of seeds
# and targets. In the first dataet we apply the dimensionality reduction approach to
# only the first component in our rank subspace. We aim to compare the effects on
# connectivity patterns with the second dataset.
#
# The result indicate that essential connectivity patterns are preserved even after
# dimensionality reduction, implying that much of the connectivity information in the
# additional components may be redundant. This approach underscores the efficiency of
# focusing analysis on the most significant data dimensions.

# %%
n_seeds = 15
n_targets = 10
indices = ([np.arange(n_seeds)], [n_seeds + np.arange(n_targets)])
delay = 10

con_data = simulate_connectivity(
    n_seeds=n_seeds,
    n_targets=n_targets,
    freq_band=freq_bands["beta"],
    n_epochs=n_epochs,
    n_times=n_times,
    sfreq=sfreq,
    snr=snr,
    connection_delay=delay,
    rng_seed=42,
)

con_data_red = simulate_connectivity(
    n_seeds=n_seeds,
    n_targets=n_targets,
    freq_band=freq_bands["beta"],
    n_epochs=n_epochs,
    n_times=n_times,
    sfreq=sfreq,
    snr=snr,
    connection_delay=delay,
    rng_seed=42,
)

# %%
# Compute the multivariate connectivity using the CaCoh method.
con = mne_connectivity.spectral_connectivity_epochs(
    con_data, indices=indices, method="cacoh"
)

con_red = mne_connectivity.spectral_connectivity_epochs(
    con_data, indices=indices, method="cacoh", rank=([1], [1])
)

# subtract mean of scores for comparison
con_meansub = con.get_data()[0] - con.get_data()[0].mean()
con_red_meansub = con_red.get_data()[0] - con_red.get_data()[0].mean()

# no. channels equal with and without projecting to rank subspace for patterns
assert (
    np.array(con_red.attrs["patterns"])[0, 0].shape[0]
    == np.array(con_red.attrs["patterns"])[0, 0].shape[0]
)
assert (
    np.array(con.attrs["patterns"])[1, 0].shape[0]
    == np.array(con.attrs["patterns"])[1, 0].shape[0]
)

_, axis = plt.subplots()
axis.plot(con.freqs, con_meansub, linewidth=2, label="Standard cacoh")
axis.plot(
    con_red.freqs,
    con_red_meansub,
    linewidth=2,
    label="Rank subspace (1) cacoh",
)
axis.set_xlabel("Frequency (Hz)")
axis.set_ylabel("Absolute connectivity (A.U.)")
plt.title("CaCoh")
plt.legend(loc="upper right")
plt.show()

# In the case that your data is not full rank and rank is left as None, an automatic
# rank computation is performed and an appropriate degree of dimensionality reduction
# will be enforced. The rank of the data is determined by computing the singular values
# of the data and finding those within a factor of :math: `1e-6` relative to the
# largest singular value.

# Whilst unlikely, there may be scenarios in which this threshold may be too lenient.
# In these cases, you should inspect the singular values of your data to identify an
# appropriate degree of dimensionality reduction to perform, which you can then specify
# manually using the ``rank`` argument. The code below shows one possible approach for
# finding an appropriate rank of close-to-singular data with a more conservative
# threshold.

# %%
# gets the singular values of the data
s = np.linalg.svd(con.get_data(), compute_uv=False)
# finds how many singular values are 'close' to the largest singular value
rank = np.count_nonzero(s >= s[0] * 1e-4)  # 1e-4 is the 'closeness' criteria
print(rank)

###############################################################################
# Advantages and disadvantages
#
# In EEG data analysis, zero-lag interactions are typically viewed with suspicion
# because they often indicate volume conduction rather than genuine physiological
# interactions. Volume conduction is a phenomenon where electrical currents from active
# neurons spread passively through the brain tissue and skull to reach the scalp, where
# they are recorded. This can make spatially distinct but electrically active areas
# appear to be synchronously active, creating artificial coherence at zero lag.
# However, it is possible that some zero-lag interactions are real, especially if the
# neural sources are physically close to each other or if there is a common driver
# influencing multiple regions simultaneously.
#
# CaCoh, by design, does not specifically distinguish between zero-lag interactions
# that are physiological and those that are artifacts of volume conduction. Its main
# purpose is to identify patterns of maximal coherence across multiple channels or
# conditions. However, because it does not exclude zero-lag interactions, it might not
# inherently differentiate between true connectivity and volume conduction effects.
#
# Nevertheless, in the context of LFP signals, which typically represent local field
# potentials recorded from electrodes implanted in the brain, the concern for volume
# conduction is less pronounced compared to EEG because LFP signals are less influenced
# by the spread of electrical activity through the scalp and skull. In this domain, the
# CaCoh method still operates and can potentially capture true zero-lag interactions
# that are physiological in nature. The method could distinguish between true zero-lag
# interactions and those resulting from volume conduction if the spatial resolution is
# high enough to separate the sources of the signals, which is often the case with LFPs
# due to their proximity to the neural sources.
#
# On the other hand, the presence of a non-zero lag is often indicative of genuine
# physiological interactions, as it suggests a time course for signal transmission
# across neural pathways. This is especially pertinent in EEG/MEG and LFP analyses.
# CaCoh capture those interactions and help to understand the dynamics of these
# time-lagged connections.
#

###############################################################################
# References
# ----------
# .. footbibliography::

# %%
