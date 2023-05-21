"""
================================================================
Compute multivariate measures of the imaginary part of coherency
================================================================

This example demonstrates how multivariate methods based on the imaginary part
of coherency :footcite:`EwaldEtAl2012` can be used to compute connectivity
between whole sets of sensors, and how spatial patterns of this connectivity
can be interpreted.

The methods in question are: the maximised imaginary part of coherency (MIC);
and the multivariate interaction measure (MIM; as well as its extension, the
global interaction measure, GIM).
"""

# Author: Thomas S. Binns <t.s.binns@outlook.com>
# License: BSD (3-clause)

# %%

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patheffects as pe

import mne
from mne import EvokedArray, make_fixed_length_epochs
from mne.datasets.fieldtrip_cmc import data_path
from mne_connectivity import seed_target_indices, spectral_connectivity_epochs

###############################################################################
# Background
# ----------
#
# Multivariate forms of signal analysis allow you to simultaneously consider
# the activity of multiple signals. In the case of connectivity, the
# interaction between multiple sensors can be analysed at once, producing a
# single connectivity spectrum. This approach brings not only practical
# benefits (e.g. easier interpretability of results from the dimensionality
# reduction), but can also offer methodological improvements (e.g. enhanced
# signal-to-noise ratio and reduced bias).
#
# A popular bivariate measure of connectivity is the imaginary part of
# coherency, which looks at the correlation between two signals in the
# frequency domain and is immune to spurious connectivity arising from volume
# conduction artefacts :footcite:`NolteEtAl2004`. However, depending on the
# degree of source mixing, this measure is susceptible to biased estimates of
# connectivity based on the spatial proximity of sensors
# :footcite:`EwaldEtAl2012`.
#
# To overcome this limitation, spatial filters can be used to estimate
# connectivity free from this source mixing-dependent bias, which additionally
# increases the signal-to-noise ratio and allows signals to be analysed in a
# multivariate manner :footcite:`EwaldEtAl2012`. This approach leads to the
# following methods: the maximised imaginary part of coherency (MIC); and the
# multivariate interaction measure (MIM).
#
# We start by loading some example MEG data and dividing it into
# two-second-long epochs.

# %%

raw = mne.io.read_raw_ctf(data_path() / 'SubjectCMC.ds')
raw.pick('mag')
raw.crop(50., 110.).load_data()
raw.notch_filter(50)
raw.resample(100)

epochs = make_fixed_length_epochs(raw, duration=2.0).load_data()

###############################################################################
# We will focus on connectivity between sensors over the left and right
# hemispheres, with 75 sensors in the left hemisphere designated as seeds, and
# 75 sensors in the right hemisphere designated as targets.

# %%

# left hemisphere sensors
seeds = [idx for idx, ch_info in enumerate(epochs.info['chs']) if
         ch_info['loc'][0] < 0]
# right hemisphere sensors
targets = [idx for idx, ch_info in enumerate(epochs.info['chs']) if
           ch_info['loc'][0] > 0]

# UNTIL RAGGED INDICES SUPPORTED
min_n_chs = min(len(seeds), len(targets))
seeds = seeds[:min_n_chs]
targets = targets[:min_n_chs]

multivar_indices = (np.array(seeds), np.array(targets))

seed_names = [epochs.info['ch_names'][idx] for idx in seeds]
target_names = [epochs.info['ch_names'][idx] for idx in targets]

# multivariate imaginary part of coherency
(mic, mim) = spectral_connectivity_epochs(
    epochs, method=['mic', 'mim'], indices=multivar_indices, fmin=5, fmax=30,
    rank=None)

# bivariate imaginary part of coherency (for comparison)
bivar_indices = seed_target_indices(seeds, targets)
imcoh = spectral_connectivity_epochs(
    epochs, method='imcoh', indices=bivar_indices, fmin=5, fmax=30)

###############################################################################
# By averaging across each connection between the seeds and targets, we can see
# that the bivariate measure of the imaginary part of coherency estimates a
# strong peak in connectivity between seeds and targets around 13-18 Hz, with a
# weaker peak around 27 Hz.

# %%
fig, axis = plt.subplots(1, 1)
axis.plot(imcoh.freqs, np.mean(np.abs(imcoh.get_data()), axis=0),
          linewidth=2)
axis.set_xlabel('Frequency (Hz)')
axis.set_ylabel('Absolute connectivity (A.U.)')
fig.suptitle('Imaginary part of coherency')


###############################################################################
# Maximised imaginary part of coherency (MIC)
# -------------------------------------------
#
# For MIC, a set of spatial filters are found that will maximise the estimated
# connectivity between the seed and target signals. These maximising filters
# correspond to the eigenvectors with the largest eigenvalue, derived from an
# eigendecomposition of information from the cross-spectral density (Eq. 7 of
# :footcite:`EwaldEtAl2012`):
#
# :math:`MIC=\frac{\boldsymbol{\alpha}^T \boldsymbol{E \beta}}{\parallel
# \boldsymbol{\alpha}\parallel \parallel\boldsymbol{\beta}\parallel}`,
#
# where :math:`\boldsymbol{\alpha}` and :math:`\boldsymbol{\beta}` are the
# spatial filters for the seeds and targets, respectively, and
# :math:`\boldsymbol{E}` is the imaginary part of the transformed
# cross-spectral density between the seeds and targets. All elements are
# frequency-dependent, however this is omitted for readability. MIC is bound
# between :math:`[-1, 1]` where the absolute value reflects connectivity
# strength and the sign reflects the phase angle difference between signals.
#
# In this instance, we see MIC reveal that in addition to the 13-18 Hz peak, a
# previously unobserved peak in connectivity around 9 Hz is present.
# Furthermore, the previous peak around 27 Hz is much less pronounced. This may
# indicate that the connectivity was the result of some distal interaction
# exacerbated by strong source mixing, which biased the bivariate connectivity
# estimate.

# %%

fig, axis = plt.subplots(1, 1)
axis.plot(mic.freqs, np.abs(mic.get_data()[0]), linewidth=2)
axis.set_xlabel('Frequency (Hz)')
axis.set_ylabel('Absolute connectivity (A.U.)')
fig.suptitle('Maximised imaginary part of coherency')


###############################################################################
# Furthermore, spatial patterns of connectivity can be constructed from the
# spatial filters to give a picture of the location of the sources involved in
# the connectivity. This information is stored under the `patterns` attribute
# of the connectivity class, with one value per frequency for each channel in
# the seeds and targets. As with MIC, the absolute value of the patterns
# reflect the strength, however the sign differences can be used to visualise
# the orientation of the underlying dipole sources. The spatial patterns are
# not bound between :math:`[-1, 1]`.
#
# Here, we average across the patterns in the 13-18 Hz range. Plotting the
# patterns shows that the greatest connectivity between the left and right
# hemispheres occurs for the posterolateral parietal regions of the left
# hemisphere, and the medial central regions of the right hemisphere.
#
# Using the signs of the values, we can infer the existence of a parietal
# dipole source in the left hemisphere which may account for the connectivity
# contributions seen for the posterolateral parietal regions and frontomedial
# areas (represented on the plot as a green line).

# %%

# compute average of patterns in desired frequency range
fband = [13, 18]
fband_idx = [mic.freqs.index(freq) for freq in fband]

patterns = mic.attrs["patterns"]
seed_pattern = np.mean(patterns[0][0][:, fband_idx[0]:fband_idx[1] + 1],
                       axis=1)
target_pattern = np.mean(patterns[1][0][:, fband_idx[0]:fband_idx[1] + 1],
                         axis=1)

# store the patterns for plotting
seed_info = epochs.copy().pick(seed_names).info
target_info = epochs.copy().pick(target_names).info
seed_pattern = EvokedArray(seed_pattern[:, np.newaxis], seed_info)
target_pattern = EvokedArray(target_pattern[:, np.newaxis], target_info)

# plot the patterns
fig, axes = plt.subplots(1, 4)
seed_pattern.plot_topomap(
    times=0, sensors='m.', units=dict(mag='A.U.'), cbar_fmt='%.1E',
    axes=axes[0:2], time_format='', show=False)
target_pattern.plot_topomap(
    times=0, sensors='m.', units=dict(mag='A.U.'), cbar_fmt='%.1E',
    axes=axes[2:], time_format='', show=False)
axes[0].set_position((0.1, 0.1, 0.35, 0.7))
axes[1].set_position((0.4, 0.3, 0.02, 0.3))
axes[2].set_position((0.5, 0.1, 0.35, 0.7))
axes[3].set_position((0.9, 0.3, 0.02, 0.3))
axes[0].set_title('Seed spatial pattern\n13-18 Hz')
axes[2].set_title('Target spatial pattern\n13-18 Hz')

# plot the left hemisphere dipole example
axes[0].plot(
    [-0.07, -0.035], [-0.03, -0.055], color='lime', linewidth=2,
    path_effects=[pe.Stroke(linewidth=4, foreground='k'), pe.Normal()])

plt.show()


###############################################################################
# Multivariate interaction measure (MIM)
# --------------------------------------
#
# Although it can be useful to analyse the single, largest connectivity
# component with MIC, multiple such components exist and can be examined with
# MIM. MIM can be thought of as an average of all connectivity components
# between the seeds and targets, and can be useful for an exploration of all
# available components. It is unnecessary to use the spatial filters of each
# component explicitly, and instead the desired result can be achieved from
# :math:`E` alone (Eq. 14 of :footcite:`EwaldEtAl2012`):
#
# :math:`MIM=tr(\boldsymbol{EE}^T)`,
#
# where again the frequency dependence is omitted. Unlike MIC, MIM is
# positive-valued, and is not bound below 1. Without normalisation, MIM can be
# thought of as reflecting the total interaction between the seeds and targets.
# MIM can be normalised to lie in the range :math:`[0, 1]` by dividing the
# scores by the number of channels in the seeds and targets, representing the
# total interaction *per channel*. **MIM is not normalised by default in this
# implementation**.
#
# Here we see MIM reveal the strongest connectivity component to be around 10
# Hz, with the higher frequency 13-18 Hz connectivity no longer being so
# prominent. This suggests that, across all components in the data, there may
# be more lower frequency connectivity sources than higher frequency sources.
# Thus, when combining these different components in MIM, the peak around 10 Hz
# remains, but the 13-18 Hz connectivity is diminished relative to the single,
# largest connectivity component of MIC.

# %%

fig, axis = plt.subplots(1, 1)
axis.plot(mim.freqs, mim.get_data()[0], linewidth=2)
axis.set_xlabel('Frequency (Hz)')
axis.set_ylabel('Absolute connectivity (A.U.)')
fig.suptitle('Multivariate interaction measure (non-normalised)')


###############################################################################
# Additionally, the instance where the seeds and targets are identical can be
# considered as a special case of MIM: the global interaction measure (GIM; Eq.
# 15 of :footcite:`EwaldEtAl2012`). This allows connectivity within a single
# set of signals to be estimated, and is possible as a result of the exclusion
# of zero phase lag components from the connectivity estimates. Computing GIM
# follows from Eq. 14, however since each interaction is considered twice,
# correcting the connectivity by a factor of :math:`\frac{1}{2}` is necessary
# (**the correction is performed automatically in this implementation**). Note:
# a similar principle applies in the case where MIC is computed for identical
# seeds and targets. Like MIM, GIM is also not bound below 1, but it can be
# normalised to lie in the range :math:`[0, 1]` by dividing by :math:`N/2`
# (where :math:`N` is the number of channels; **GIM is also not normalised by
# default in this implementation**).
#
# With GIM, we find a broad connectivity peak around 10 Hz, with an additional
# peak around 20 Hz. The differences observed with GIM highlight the presence
# of interactions within each hemisphere that are absent for MIC or MIM.

# %%

indices = (np.array([*seeds, *targets]), np.array([*seeds, *targets]))
gim = spectral_connectivity_epochs(
    epochs, method='mim', indices=indices, fmin=5, fmax=30, rank=None)

normalised_gim = gim.get_data()[0] / (len(indices[0]) / 2)

fig, axis = plt.subplots(1, 1)
axis.plot(gim.freqs, normalised_gim, linewidth=2)
axis.set_xlabel('Frequency (Hz)')
axis.set_ylabel('Connectivity (A.U.)')
fig.suptitle('Global interaction measure (normalised)')


###############################################################################
# Handling high-dimensional data
# ------------------------------
#
# An important issue to consider when using these multivariate methods is
# overfitting, which risks biasing connectivity estimates to maximise noise in
# the data. This risk can be reduced by performing a preliminary dimensionality
# reduction prior to estimating the connectivity with a singular value
# decomposition (Eqs. 32 & 33 of :footcite:`EwaldEtAl2012`). The degree of this
# dimensionality reduction can be specified using the ``rank`` argument, which
# by default will not perform any dimensionality reduction (assuming your data
# is full rank; see below if not). Choosing an expected rank of the data
# requires *a priori* knowledge about the number of components you expect to
# observe in the data.
#
# When comparing MIC/MIM scores across recordings, **it is highly recommended
# to estimate connectivity from the same number of channels (or equally from
# the same degree of rank subspace projection)** to avoid biases in
# connectivity estimates. Bias can be avoided by specifying a consistent rank
# subspace to project to using the ``rank`` argument, standardising your
# connectivity estimates regardless of changes in e.g. the number of channels
# across recordings. Note that this does not refer to the number of seeds and
# targets *within* a connection being identical, rather to the number of seeds
# and targets *across* connections.
#
# Here, we will project our seed and target data to only the first 25
# components of our rank subspace. Results for MIM show that the general
# spectral pattern of connectivity is retained in the rank subspace-projected
# data, suggesting that a fair degree of redundant connectivity information is
# contained in the remaining 50 components of the seed and target data. We also
# assert that the spatial patterns of MIC are returned in the original sensor
# space despite this rank subspace projection, being reconstructed using the
# products of the singular value decomposition (Eqs. 46 & 47 of
# :footcite:`EwaldEtAl2012`).

# %%

(mic_red, mim_red) = spectral_connectivity_epochs(
    epochs, method=['mic', 'mim'], indices=multivar_indices, fmin=5, fmax=30,
    rank=([25], [25]))

# subtract mean of scores for comparison
mim_red_meansub = mim_red.get_data()[0] - mim_red.get_data()[0].mean()
mim_meansub = mim.get_data()[0] - mim.get_data()[0].mean()

# compare standard and rank subspace-projected MIM
fig, axis = plt.subplots(1, 1)
axis.plot(mim_red.freqs, mim_red_meansub, linewidth=2,
          label='rank subspace (25) MIM')
axis.plot(mim.freqs, mim_meansub, linewidth=2, label='standard MIM')
axis.set_xlabel('Frequency (Hz)')
axis.set_ylabel('Mean-corrected connectivity (A.U.)')
axis.legend()
fig.suptitle('Multivariate interaction measure (non-normalised)')

# no. channels equal with and without projecting to rank subspace for patterns
assert (mic.attrs["patterns"][0][0].shape[0]
        == mic_red.attrs["patterns"][0][0].shape[0])
assert (mic.attrs["patterns"][1][0].shape[0]
        == mic_red.attrs["patterns"][1][0].shape[0])


###############################################################################
# In the case that your data is not full rank and ``rank`` is left as ``None``,
# an automatic rank computation is performed and an appropriate degree of
# dimensionality reduction will be enforced.
#
# In the case that your data is close to singular, the automatic rank
# computation may fail to detect this, and an error will be raised. In this
# case, you should inspect the singular values of your data to identify an
# appropriate degree of dimensionality reduction to perform, which you can then
# specify manually using the ``rank`` argument. The code below shows one
# possible approach for finding an appropriate rank of close-to-singular data.

# %%

# gets the singular values of the data
s = np.linalg.svd(raw.get_data(), compute_uv=False)
# finds how many singular values are "close" to the largest singular value
rank = np.count_nonzero(s >= s[0] * 1e-5)  # 1e-5 is the "closeness" criteria


###############################################################################
# Limitations
# -----------
#
# These multivariate methods offer many benefits in the form of dimensionality
# reduction, signal-to-noise ratio improvements, and invariance to
# estimate-biasing source mixing; however, no method is perfect. The immunity
# of the imaginary part of coherency to volume conduction comes from the fact
# that these artefacts have zero phase lag, and hence a zero-valued imaginary
# component. By projecting the complex-valued coherency to the imaginary axis,
# signals of a given magnitude with phase lag differences close to 90° or 270°
# see their contributions to the connectivity estimate increased relative to
# comparable signals with phase lag differences close to 0° or 180°. Therefore,
# the imaginary part of coherency is biased towards connectivity involving 90°
# and 270° phase lag difference components.
#
# Whilst this is not a limitation specific to the multivariate extension of
# this measure, these multivariate methods can introduce further bias: when
# maximising the imaginary part of coherency, components with phase lag
# differences close to 90° and 270° will likely give higher connectivity
# estimates, and so may be prioritised by the spatial filters.
#
# Such a limitation should be kept in mind when estimating connectivity using
# these methods. Possible sanity checks can involve comparing the spectral
# profiles of MIC/MIM to coherence and the imaginary part of coherency
# computed on the same data, as well as comparing to other multivariate
# measures, such as canonical coherence :footcite:`VidaurreEtAl2019`.

###############################################################################
# References
# ----------
# .. footbibliography::

# %%
