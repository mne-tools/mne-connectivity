"""
.. _ex-compare-cohy-methods:

=====================================
Comparison of coherency-based methods
=====================================

This example demonstrates the distinct forms of information captured by
coherency-based connectivity methods, and highlights the scenarios in which
these different methods should be applied.
"""

# Authors: Thomas S. Binns <t.s.binns@outlook.com>
#          Mohammad Orabe <orabe.mhd@gmail.com>
# License: BSD (3-clause)

# %%
import numpy as np
from matplotlib import pyplot as plt

from mne_connectivity import (
    make_signals_in_freq_bands,
    seed_target_indices,
    spectral_connectivity_epochs,
)

###############################################################################
# An introduction to coherency-based connectivity methods
# -------------------------------------------------------
#
# MNE-Connectivity supports several methods based on coherency. These are:
#
# - coherency (Cohy)
# - coherence (Coh; absolute coherency)
# - imaginary part of coherency (ImCoh)
# - canonical coherency (CaCoh)
# - maximised imaginary part of coherency (MIC)
# - multivariate interaction measure (MIM)
#
# |
#
# All of these methods centre on Cohy, a complex-valued estimate of the
# correlation between signals in the frequency domain. It is an undirected
# measure of connectivity, being invariant to the direction of information flow
# between signals.
#
# The common approach for handling these complex-valued coherency scores is to
# either take their absolute values (Coh) or their imaginary values (ImCoh
# :footcite:`NolteEtAl2004`).
#
# In addition to these traditional bivariate connectivity measures (i.e.
# between two signals), advanced multivariate measures (i.e. between groups of
# signals) have also been developed based on Cohy (CaCoh
# :footcite:`VidaurreEtAl2019`; can take the absolute value for a multivariate
# form of Coh; see :doc:`cacoh`) or ImCoh (MIC & MIM :footcite:`EwaldEtAl2012`;
# see :doc:`mic_mim`).
#
# Despite their similarities, there are distinct scenarios in which these
# different methods are most appropriate, as we will show in this example.

###############################################################################
# Zero and non-zero time-lag interactions
# ---------------------------------------
#
# The key difference between Cohy/Coh and ImCoh is how information about zero
# time-lag interactions is captured.
#
# We generally assume that communication within the brain involves some delay
# in the flow of information (i.e. a non-zero time-lag). This reflects the time
# taken for: the propagation of action potentials along axons; the release of
# neurotransmitters from presynaptic terminals and binding to receptors on
# postsynaptic terminals; etc...
#
# In contrast, interactions with no delay (i.e. a zero time-lag) are often
# considered to reflect non-physiological activity, such as volume conduction
# - the propagation of electrical activity through the brain's conductive
# tissue from a single source to multiple electrodes simultaneously
# :footcite:`NolteEtAl2004`. Such interactions therefore do not reflect
# genuine, physiological communication between brain regions. Naturally,
# having a method that can discard spurious zero time-lag connectivity
# estimates is very desirable.
#
# **Note:** Not all zero time-lag interactions are necessarily
# non-physiological :footcite:`ViriyopaseEtAl2012`.
#
# To demonstrate the differences in how Cohy/Coh and ImCoh handle zero time-lag
# interactions, we simulate two sets of data with:
#
# 1. A non-zero time-lag interaction at 10-12 Hz.
# 2. A zero time-lag interaction at 23-25 Hz.

# %%

# Generate simulated data
data_delay = make_signals_in_freq_bands(
    n_seeds=3,
    n_targets=3,
    freq_band=(10, 12),  # 10-12 Hz interaction
    connection_delay=2,  # samples; non-zero time-lag
    rng_seed=42,
)

data_no_delay = make_signals_in_freq_bands(
    n_seeds=3,
    n_targets=3,
    freq_band=(23, 25),  # 23-25 Hz interaction
    connection_delay=0,  # samples; zero time-lag
    rng_seed=44,
)

# Combine data into a single object
data = data_delay.add_channels([data_no_delay])

###############################################################################
# We compute the connectivity of these simulated signals using CaCoh (a
# multivariate form of Cohy/Coh) and MIC (a multivariate form of ImCoh).

# %%

# Generate connectivity indices
seeds = [0, 1, 2, 6, 7, 8]
targets = [3, 4, 5, 9, 10, 11]
bivar_indices = seed_target_indices(seeds, targets)
multivar_indices = ([seeds], [targets])

# Compute CaCoh & MIC
(cacoh, mic) = spectral_connectivity_epochs(
    data, method=["cacoh", "mic"], indices=multivar_indices, sfreq=100, fmin=3, fmax=35
)

###############################################################################
# As you can see, both CaCoh and MIC capture the non-zero time-lag interaction
# at 10-12 Hz, however only CaCoh captures the zero time-lag interaction at
# 23-25 Hz.

# %%

# Plot CaCoh & MIC
fig, axis = plt.subplots(1, 1)
axis.plot(cacoh.freqs, np.abs(cacoh.get_data()[0]), linewidth=2, label="CaCoh")
axis.plot(
    mic.freqs, np.abs(mic.get_data()[0]), linewidth=2, label="MIC", linestyle="--"
)
axis.set_xlabel("Frequency (Hz)")
axis.set_ylabel("Connectivity (A.U.)")
axis.annotate("Non-zero\ntime-lag\ninteraction", xy=(13.5, 0.85))
axis.annotate("Zero\ntime-lag\ninteraction", xy=(26.5, 0.85))
axis.legend(loc="upper left")
fig.suptitle("CaCoh vs. MIC\nNon-zero & zero time-lags")


# %%


def plot_connectivity_circle():
    """Plot a circle with radius 1, real and imag. axes, and angles marked."""
    fig, axis = plt.subplots(1, 1)
    t = np.linspace(0, 2 * np.pi, 100)
    axis.plot(np.cos(t), np.sin(t), color="k", linewidth=0.1)
    axis.plot([-1, 1], [0, 0], color="k", linestyle="--")
    axis.plot([0, 0], [-1, 1], color="k", linestyle="--")
    axis.axis("off")

    fontdict = {"fontsize": 10}
    qpi = np.pi / 4
    axis.text(1, 0, " 0°", ha="left", va="center", fontdict=fontdict)
    axis.text(np.pi / 4, np.pi / 4, "45°", ha="center", va="center", fontdict=fontdict)
    axis.text(0, 1, "90°", ha="center", va="bottom", fontdict=fontdict)
    axis.text(-qpi, qpi, "135°", ha="center", va="center", fontdict=fontdict)
    axis.text(-1, 0, "180°", ha="right", va="center", fontdict=fontdict)
    axis.text(-qpi, -qpi, "-135°", ha="center", va="center", fontdict=fontdict)
    axis.text(0, -1, "-90°", ha="center", va="top", fontdict=fontdict)
    axis.text(qpi, -qpi, "-45°", ha="center", va="center", fontdict=fontdict)

    fontdict = {"fontsize": 12}
    axis.text(1.15, 0, " Real", ha="left", va="center", fontdict=fontdict)
    axis.text(0, 1.15, "Imaginary", ha="center", va="bottom", fontdict=fontdict)
    axis.text(0, 0, "0 ", ha="right", va="top", fontdict=fontdict)
    axis.text(-1, 0, "-1", ha="left", va="top", fontdict=fontdict)
    axis.text(1, 0, "+1", ha="right", va="top", fontdict=fontdict)
    axis.text(0, -1, "-1 ", ha="right", va="bottom", fontdict=fontdict)
    axis.text(0, 1, "+1 ", ha="right", va="top", fontdict=fontdict)

    axis.set_aspect("equal")

    return fig, axis


###############################################################################
# The different interactions (not) captured by CaCoh and MIC can be understood
# by visualising the complex values of the interactions.

# %%

# Get complex connectivity values at frequency bands
freqs = cacoh.freqs
cacoh_10_12 = np.mean(cacoh.get_data()[0, freqs.index(10) : freqs.index(12) + 1])
cacoh_23_25 = np.mean(cacoh.get_data()[0, freqs.index(23) : freqs.index(25) + 1])

# Plot complex connectivity values
fig, axis = plot_connectivity_circle()
axis.quiver(
    0,
    0,
    np.real(cacoh_10_12),
    np.imag(cacoh_10_12),
    units="xy",
    scale=1,
    linewidth=2,
    color="C2",
    label="10-12 Hz",
)
axis.quiver(
    0,
    0,
    np.real(cacoh_23_25),
    np.imag(cacoh_23_25),
    units="xy",
    scale=1,
    linewidth=2,
    color="C3",
    label="23-25 Hz",
    zorder=99,
)
axis.legend(loc="upper right", bbox_to_anchor=[1.1, 1.1])

###############################################################################
# Above, we plot the complex-valued CaCoh scores for the 10-12 Hz and 23-25 Hz
# interactions as vectors with origin :math:`(0, 0)` bound within a circle of
# radius 1 (reflecting the fact that coherency scores span the set of complex
# values in the range :math:`[-1, 1]`).
#
# The circumference of the circle spans the range :math:`(-\pi, \pi]`. The real
# axis corresponds to vectors with angles of 0° (:math:`0\pi`; positive
# values) or 180° (:math:`\pi`; negative values). The imaginary axis
# corresponds to vectors with angles of 90° (:math:`\frac{1}{2}\pi`; positive
# values) or -90° (:math:`-\frac{1}{2}\pi`; negative values).
#
# Zero time-lag interactions have angles of 0° and 180° (i.e. no phase
# difference), corresponding to a non-zero real component, but a zero-valued
# imaginary component. We see this nicely for the 23-25 Hz interaction, which
# has an angle of ~0°. Taking the absolute CaCoh value shows us the magnitude
# of this interaction to be ~0.9. However, first projecting this information to
# the imaginary axis gives us a magnitude of ~0.
#
# In contrast, non-zero time-lag interactions do not lie on the real axis (i.e.
# a phase difference), corresponding to non-zero real and imaginary components.
# We see this nicely for the 10-12 Hz interaction, which has an angle of ~-75°.
# Taking the absolute CaCoh value shows us the magnitude of this interaction to
# be ~0.9, which is also seen when first projecting this information to the
# imaginary axis.
#
# This distinction is why connectivity methods utilising information from both
# real and imaginary components (Cohy, Coh, CaCoh) capture both zero and
# non-zero time-lag interactions, whereas methods using only the imaginary
# component (ImCoh, MIC, MIM) capture only non-zero time-lag interactions.
#
# The ability to capture these different interactions is not a feature specific
# to multivariate connectivity methods, as shown below for the bivariate
# methods Coh and ImCoh.

# %%

# Compute Coh & ImCoh
(coh, imcoh) = spectral_connectivity_epochs(
    data, method=["coh", "imcoh"], indices=bivar_indices, sfreq=100, fmin=3, fmax=35
)

coh_mean = np.mean(coh.get_data(), axis=0)
imcoh_mean = np.mean(np.abs(imcoh.get_data()), axis=0)

coh_mean_subbed = coh_mean - np.mean(coh_mean)
imcoh_mean_subbed = imcoh_mean - np.mean(imcoh_mean)

# Plot Coh & ImCoh
fig, axis = plt.subplots(1, 1)
axis.plot(coh.freqs, coh_mean_subbed, linewidth=2, label="Coh")
axis.plot(imcoh.freqs, imcoh_mean_subbed, linewidth=2, label="ImCoh", linestyle="--")
axis.set_xlabel("Frequency (Hz)")
axis.set_ylabel("Mean-corrected connectivity (A.U.)")
axis.annotate("Non-zero\ntime-lag\ninteraction", xy=(13, 0.25))
axis.annotate("Zero\ntime-lag\ninteraction", xy=(25, 0.25))
axis.legend(loc="upper left")
fig.suptitle("Coh vs. ImCoh\nNon-zero & zero time-lags")

###############################################################################
# When different coherency-based methods are most appropriate
# -----------------------------------------------------------
#
# With this information, we can define situations under which these different
# approaches are most appropriate.
#
# |
#
# **In situations where non-physiological zero time-lag interactions are
# assumed, methods based on only the imaginary part of coherency (ImCoh, MIC,
# MIM) should be used.** Examples of situations include:
#
# - Connectivity between channels of a single modality.
# - Connectivity between channels of different modalities where the same
#   reference is used.
#
# Note that this applies not only to sensor-space signals, but also to
# source-space signals where remnants of these non-physiological interactions
# may remain even after source reconstruction.
#
# |
#
# **In situations where non-physiological zero time-lag interactions are not
# assumed, methods based on real and imaginary parts of coherency (Cohy, Coh,
# CaCoh) should be used.** An example includes:
#
# - Connectivity between channels of different modalities where different
#   references are used.
#
# |
#
# Equally, when there are no non-physiological zero time-lag interactions, one
# should not use methods based on only the imaginary part of coherency. There
# are two key reasons:
#
# **1. Discarding physiological zero time-lag interactions**
#
# First, not all zero time-lag interactions are non-physiological
# :footcite:`ViriyopaseEtAl2012`. Accordingly, methods based on only the
# imaginary part of coherency may lead to information about genuine
# connectivity being lost.
#
# In situations where non-physiological zero time-lag interactions are present,
# the potential loss of physiological information is generally acceptable to
# avoid spurious connectivity estimates. However, unnecessarily discarding this
# information can of course be detrimental.
#
# **2. Biasing interactions based on the angle of interaction**
#
# Depending on their angles, two non-zero time-lag interactions may have the
# same magnitude in the complex plane, but different magnitudes when projected
# to the imaginary axis.
#
# This is demonstrated below, where we simulate 2 interactions with non-zero
# time-lags at 10-12 Hz and 23-25 Hz. Computing the connectivity, we see how
# both interactions have a similar magnitude (~0.9), but different angles
# (~-45° for 10-12 Hz; ~-90° for 23-25 Hz).

# %%

# Generate simulated data
data_10_12 = make_signals_in_freq_bands(
    n_seeds=3,
    n_targets=3,
    freq_band=(10, 12),  # 10-12 Hz interaction
    connection_delay=1,  # samples
    rng_seed=40,
)

data_23_25 = make_signals_in_freq_bands(
    n_seeds=3,
    n_targets=3,
    freq_band=(23, 25),  # 23-25 Hz interaction
    connection_delay=1,  # samples
    rng_seed=42,
)

# Combine data into a single array
data = data_10_12.add_channels([data_23_25])

# Compute CaCoh & MIC
(cacoh, mic) = spectral_connectivity_epochs(
    data, method=["cacoh", "mic"], indices=multivar_indices, sfreq=100, fmin=3, fmax=35
)

# Get complex connectivity values at frequency bands
freqs = cacoh.freqs
cacoh_10_12 = np.mean(cacoh.get_data()[0, freqs.index(10) : freqs.index(12) + 1])
cacoh_23_25 = np.mean(cacoh.get_data()[0, freqs.index(23) : freqs.index(25) + 1])

# Plot complex connectivity values
fig, axis = plot_connectivity_circle()
axis.quiver(
    0,
    0,
    np.real(cacoh_10_12),
    np.imag(cacoh_10_12),
    units="xy",
    scale=1,
    linewidth=2,
    color="C2",
    label="10-12 Hz",
)
axis.quiver(
    0,
    0,
    np.real(cacoh_23_25),
    np.imag(cacoh_23_25),
    units="xy",
    scale=1,
    linewidth=2,
    color="C3",
    label="23-25 Hz",
    zorder=99,
)
axis.legend(loc="upper right", bbox_to_anchor=[1.1, 1.1])

###############################################################################
# Plotting the connectivity values for CaCoh and MIC, we see how the 10-12 Hz
# and 23-25 Hz interactions have a similar magnitude for CaCoh, whereas the MIC
# scores for the 10-12 Hz interaction are lower than for the 23-25 Hz
# interaction.
#
# This difference reflects the fact that as the angle of interaction deviates
# from :math:`\pm` 90°, less information will be represented in the imaginary
# part of coherency. Accordingly, considering only the imaginary part of
# coherency can bias connectivity estimates based on the angle of interaction.

# %%

# Plot CaCoh & MIC
fig, axis = plt.subplots(1, 1)
axis.plot(cacoh.freqs, np.abs(cacoh.get_data()[0]), linewidth=2, label="CaCoh")
axis.plot(
    mic.freqs, np.abs(mic.get_data()[0]), linewidth=2, label="MIC", linestyle="--"
)
axis.set_xlabel("Frequency (Hz)")
axis.set_ylabel("Connectivity (A.U.)")
axis.annotate("$\\pm$45°\ninteraction", xy=(12.5, 0.9))
axis.annotate("$\\pm$90°\ninteraction", xy=(26.5, 0.9))
axis.legend(loc="upper left")
fig.suptitle("CaCoh vs. MIC\n$\\pm$45° & $\\pm$90° interactions")

###############################################################################
# In situations where non-physiological zero time-lag interactions are present,
# this phase angle-dependent bias is generally acceptable to avoid spurious
# connectivity estimates. However in situations where non-physiological zero
# time-lag interactions are not present, such a bias is clearly problematic.
#
# |
#
# Again, these considerations are not specific to multivariate methods, as
# shown below with Coh and ImCoh.

# %%

# Compute Coh & ImCoh
(coh, imcoh) = spectral_connectivity_epochs(
    data, method=["coh", "imcoh"], indices=bivar_indices, sfreq=100, fmin=3, fmax=35
)

coh_mean = np.mean(coh.get_data(), axis=0)
imcoh_mean = np.mean(np.abs(imcoh.get_data()), axis=0)

coh_mean_subbed = coh_mean - np.mean(coh_mean)
imcoh_mean_subbed = imcoh_mean - np.mean(imcoh_mean)

# Plot Coh & ImCoh
fig, axis = plt.subplots(1, 1)
axis.plot(coh.freqs, coh_mean_subbed, linewidth=2, label="Coh")
axis.plot(imcoh.freqs, imcoh_mean_subbed, linewidth=2, label="ImCoh", linestyle="--")
axis.set_xlabel("Frequency (Hz)")
axis.set_ylabel("Mean-corrected connectivity (A.U.)")
axis.annotate("$\\pm$45°\ninteraction", xy=(13, 0.25))
axis.annotate("$\\pm$90°\ninteraction", xy=(26.5, 0.25))
axis.legend(loc="upper left")
fig.suptitle("Coh vs. ImCoh\n$\\pm$45° & $\\pm$90° interactions")

###############################################################################
# Bivariate vs. multivariate coherency methods
# --------------------------------------------
#
# As we have seen, coherency-based methods can be bivariate (Cohy, Coh, ImCoh)
# and multivariate (CaCoh, MIC, MIM). Whilst both forms capture the same
# information, there are several benefits to using multivariate methods when
# investigating connectivity between many signals.
#
# The multivariate methods can be used to capture the most relevant
# interactions between two groups of signals, representing this information in
# the component, rather than signal space.
#
# The dimensionality reduction associated with these methods offers: a much
# easier interpretation of the results; a higher signal-to-noise ratio compared
# to e.g. averaging bivariate connectivity estimates across multiple pairs of
# signals; and even reduced bias in what information is captured
# :footcite:`EwaldEtAl2012`.
#
# Furthermore, despite the dimensionality reduction of multivariate methods it
# is still possible to investigate the topographies of connectivity, with
# spatial patterns of connectivity being returned alongside the connectivity
# values themselves :footcite:`HaufeEtAl2014`.
#
# More information about the multivariate coherency-based methods can be found
# in the following examples:
#
# - CaCoh - :doc:`cacoh`
# - MIC & MIM - :doc:`mic_mim`

###############################################################################
# Alternative approaches to computing connectivity
# ------------------------------------------------
#
# Coherency-based methods are only some of the many approaches available in
# MNE-Connectivity for studying interactions between signals. Other
# non-directed measures include those based on the phase-lag index
# :footcite:`StamEtAl2007,VinckEtAl2011` (see also :doc:`dpli_wpli_pli`) and
# phase locking value :footcite:`LachauxEtAl1999,BrunaEtAl2018`.
#
# Furthermore, directed measures of connectivity which determine the direction
# of information flow are also available, including a variant of the phase-lag
# index :footcite:`StamEtAl2012` (see also :doc:`dpli_wpli_pli`), the phase
# slope index :footcite:`NolteEtAl2008` (see also
# :func:`mne_connectivity.phase_slope_index`), and Granger causality
# :footcite:`BarnettSeth2015,WinklerEtAl2016` (see also
# :doc:`granger_causality`).

###############################################################################
# Conclusion
# ----------
#
# Altogether, there are clear scenarios in which different coherency-based
# methods are appropriate.
#
# Methods based on the imaginary part of coherency alone (ImCoh, MIC, MIM)
# should be used when non-physiological zero time-lag interactions are present.
#
# In contrast, methods based on the real and imaginary parts of coherency
# (Cohy, Coh, CaCoh) should be used when non-physiological zero time-lag
# interactions are absent.

###############################################################################
# References
# ----------
# .. footbibliography::

# %%
