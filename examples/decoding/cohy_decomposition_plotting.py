"""
==============================================================
Visualising spatial contributions to multivariate connectivity
==============================================================

This example demonstrates how the spatial filters and patterns of connectivity obtained
from the decomposition tools in the decoding module can be visualised and interpreted.
"""

# Author: Thomas S. Binns <t.s.binns@outlook.com>
# License: BSD (3-clause)

# %%

import mne
from mne import make_fixed_length_epochs
from mne.datasets.fieldtrip_cmc import data_path

from mne_connectivity import CoherencyDecomposition

########################################################################################
# Background
# ----------
# Multivariate forms of signal analysis allow you to simultaneously consider the
# activity of multiple signals. In the case of connectivity, the interaction between
# multiple sensors can be analysed at once and the strongest components of this
# interaction captured in a lower-dimensional set of connectivity spectra. This approach
# brings not only practical benefits (e.g. easier interpretability of results from the
# dimensionality reduction), but can also offer methodological improvements (e.g.
# enhanced signal-to-noise ratio and reduced bias).
#
# Coherency-based methods are popular approaches for analysing connectivity, capturing
# correlations between signals in the frequency domain. Various coherency-based
# multivariate methods exist, including: canonical coherency (CaCoh; multivariate
# measure of coherency/coherence) :footcite:`VidaurreEtAl2019`; and maximised imaginary
# coherency (MIC; multivariate measure of the imaginary part of coherency)
# :footcite:`EwaldEtAl2012`.
#
# These methods are described in detail in the following examples:
#  - comparison of coherency-based methods - :doc:`../compare_coherency_methods`
#  - CaCoh - :doc:`../cacoh`
#  - MIC - :doc:`../mic_mim`
#
# The CaCoh and MIC methods work by finding spatial filters that decompose the data into
# components of connectivity, and applying them to the data. Connectivity can then be
# computed on this transformed data (see :doc:`cohy_decomposition` for more
# information).
#
# However, in addition to the connectivity scores, useful insights about the data can be
# gained by visualising the topographies of the spatial filters and their corresponding
# spatial patterns. These provide important information about the spatial distributions
# of connectivity information, and represent two complementary aspects:
#
# - The filters represent how the connectivity sources are extracted from the channel
#   data, akin to an inverse model.
# - The patterns represent how the channel data is formed by the connectivity sources,
#   akin to a forward model.
#
# This distinction is discussed further in Haufe *et al.* (2014)
# :footcite:`HaufeEtAl2014`, but in short: **the patterns should be used to interpret
# the contribution of distinct brain regions/sensors to a given component of
# connectivity**. Accordingly, keep in mind that the filters and patterns are not a
# replacement for source reconstruction, as without this the patterns will still only
# tell you about the spatial contributions of sensors, not underlying brain regions,
# to connectivity.

########################################################################################
# Generating the filters and patterns
# -----------------------------------
# We will first load some example MEG data which we will generate the spatial filters
# and patterns for, and divide it into epochs.

# %%

# Load example MEG data
raw = mne.io.read_raw_ctf(data_path() / "SubjectCMC.ds")
raw.pick("mag")
raw.crop(50.0, 110.0).load_data()
raw.notch_filter(50)
raw.resample(100)

# Create epochs
epochs = make_fixed_length_epochs(raw, duration=2.0).load_data()

########################################################################################
# We designate the left hemisphere sensors as the seeds and the right hemisphere sensors
# as the targets. Since this is sensor-space data, we will use the MIC method to analyse
# connectivity, given its resilience to zero time-lag interactions (see
# :doc:`../compare_coherency_methods` for more information).

# %%

# Left hemisphere sensors
seeds = [idx for idx, ch_info in enumerate(epochs.info["chs"]) if ch_info["loc"][0] < 0]

# Right hemisphere sensors
targets = [
    idx for idx, ch_info in enumerate(epochs.info["chs"]) if ch_info["loc"][0] > 0
]

# Define indices
indices = (seeds, targets)

########################################################################################
# To fit the filters (and in turn compute the corresponding patterns), we instantiate
# the :class:`~mne_connectivity.decoding.CoherencyDecomposition` object and call the
# :meth:`~mne_connectivity.decoding.CoherencyDecomposition.fit` method. We also define
# our connectivity frequency band of interest to be 20-30 Hz. See
# :doc:`cohy_decomposition` for more information.

# %%

# Instantiate decomposition object
mic = CoherencyDecomposition(
    info=epochs.info,
    method="mic",
    indices=indices,
    mode="multitaper",
    fmin=20,
    fmax=30,
    rank=(3, 3),
)

# Fit filters & generate patterns
mic.fit(epochs.get_data())

########################################################################################
# Visualising the patterns
# ------------------------
# Visualising the patterns as topomaps can be done using the
# :meth:`~mne_connectivity.decoding.CoherencyDecomposition.plot_patterns` method.
#
# When interpreting patterns, note that the absolute value reflects the strength of the
# contribution to connectivity, and that the sign differences can be used to visualise
# the orientation of the underlying dipole sources. The spatial patterns are **not**
# bound between :math:`[-1, 1]`.
#
# Plotting the patterns for 20-30 Hz connectivity below, we find the strongest
# connectivity between the left and right hemispheres comes from centromedial left and
# frontolateral right sensors, based on the areas with the largest absolute values. As
# these patterns come from decomposition on sensor-space data, we make no assumptions
# about the underlying brain regions involved in this connectivity.

# %%

# Plot patterns
mic.plot_patterns(info=epochs.info, sensors="m.", size=2)

########################################################################################
# Visualising the filters
# -----------------------
# We can also visualise the filters as topomaps using the
# :meth:`~mne_connectivity.decoding.CoherencyDecomposition.plot_filters` method.
#
# Here we see that the filters show a similar topography to the patterns. However, this
# is not always the case, and you should never confuse the information represented by
# the filters (i.e. an inverse model) and patterns (i.e. a forward model), which can
# lead to very incorrect interpretations of the data :footcite:`HaufeEtAl2014`.

# %%

# Plot filters
mic.plot_filters(info=epochs.info, sensors="m.", size=2)

########################################################################################
# References
# ----------
# .. footbibliography::

# %%
