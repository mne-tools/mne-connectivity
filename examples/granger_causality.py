"""
==========================================================================
Compute directionality of connectivity with multivariate Granger causality
==========================================================================

This example demonstrates how Granger causality based on state-space models
:footcite:`BarnettSeth2015` can be used to compute directed connectivity
between sensors in a multivariate manner. Furthermore, the use of time-reversal
for improving the robustness of directed connectivity estimates to noise in the
data is discussed :footcite:`WinklerEtAl2016`.
"""

# Author: Thomas S. Binns <t.s.binns@outlook.com>
# License: BSD (3-clause)

# %%

import numpy as np
from matplotlib import pyplot as plt

import mne
from mne import make_fixed_length_epochs
from mne.datasets.fieldtrip_cmc import data_path
from mne_connectivity import spectral_connectivity_epochs

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
# Additionally, it can be of interest to examine the directionality of
# connectivity between signals, providing additional clarity to how information
# flows in a system. One such directed measure of connectivity is Granger
# causality (GC). A signal, :math:`boldsymbol{x}`, is said to Granger-cause
# another signal, :math:`boldsymbol{y}`, if information from the past of
# :math:`boldsymbol{x}` improves the prediction of the present of
# :math:`boldsymbol{y}` over the case where only information from the past of
# :math:`boldsymbol{y}` is used. Note: this of course does not mean that GC
# shows true causality between signals.
#
# The degree to which :math:`boldsymbol{x}` and :math:`boldsymbol{y}` can be
# used to predict one another can be quantified using vector autoregressive
# (VAR) models. Considering the simpler case of time domain connectivity, the
# VAR models are as follows:
#
# :math:`y_t = \sum_{k=1}^{K} a_k y_{t-k} + \xi_t^y`, :math:`Var(\xi_t^y) :=
# Sigma_y`,
#
# and :math:`boldsymbol{z}_t = \sum_{k=1}^K \boldsymbol{A}_k
# \boldsymbol{z}_{t-k} + \boldsymbol{\epsilon}_t`, :math:`\boldsymbol{\Sigma}
# := \langle \boldsymbol{\epsilon}_t \boldsymbol{\epsilon}_t^T \rangle =
# \begin{bmatrix} \Sigma_{xx} & \Sigma_{xy} \\ \Sigma_{yx} & \Sigma_{yy}
# \end{bmatrix}`,
#
# representing the reduced and full VAR models, respectively, where: :math:`K`
# is the order of the VAR model, determining the number of lags, :math:`k`,
# used; :math:`\boldsymbol{Z} := \begin{bmatrix} \boldsymbol{x} \\
# \boldsymbol{y} \end{bmatrix}`; and \xi and \boldsymbol{\epsilon} are the
# residuals of the VAR models. In this way, the information of the signals at
# time :math:`t` can be represented as a weighted form of the information from
# the previous timepoints, plus some residual information not encoded in the
# signals' past. In practice, VAR model parameters are computed from an
# autocovariance sequence generated from the time-series data using the
# Yule-Walker equations :footcite:`Whittle1963`.
#
# By comparing the residuals, or errors, of the reduced and full VAR models, we
# can therefore estimate how much :math:`\boldsymbol{x}` Granger-causes
# :math:`\boldsymbol{y}`:
#
# :math:`F_{x \rightarrow y} = ln(\frac{\Sigma_y}{\Sigma_{yy}})`,
#
# where :math:`F` is the Granger score. For example, if :math:`\boldsymbol{x}`
# contains no information about :math:`\boldsymbol{y}`, the residuals of the
# reduced and full VAR models will be identical, and
# :math:`F_{x \rightarrow y}` will naturally be 0, indicating that
# information from :math:`\boldsymbol{x}` does not flow to
# :math:`\boldsymbol{y}`. In contrast, if :math:`\boldsymbol{x}` does help to
# predict :math:`\boldsymbol{y}`, the residual of the full model will be
# smaller than that of the reduced model. :math:`\frac{\Sigma_y}{\Sigma_{yy}}`
# will therefore be greater than 1, leading to a Granger score > 0. Granger
# scores are thus bound between :math:`[0, \infty)`
#
# These same principles apply to spectral GC, which provides information about
# the directional relationships of signals for individual frequencies. The
# autocovariance sequence is instead generated from an inverse Fourier
# transform applied to the cross-spectral density of the signals, and a
# spectral transfer function is required to translate information from the VAR
# models back into the frequency domain before computing the final Granger
# scores.
#
# Barnett and Seth :footcite:`BarnettSeth2015` have defined a multivariate
# form of spectral GC based on state-space models, enabling the estimation of
# information flow between whole sets of signals simultaneously:
#
# :math:`F_{A \rarrow B}(f) = \Real ln(\frac{det(\boldsymbol{S}_{BB}(f))}
# {det(\boldsymbol{S}_{BB}(f) - \boldymbol{H}_{ba}(f)
# \boldsymbol{\Sigma}_{AA \lvert B} \boldsymbol{H}_{ba}^*(f))})`,
#
# where: :math:`A` and :math:`B` are the seeds and targets, respectively;
# :math:`f` is a given frequency; :math:`\boldsymbol{H}` is the spectral
# transfer function; :math:`\boldsymbol{\Sigma}` is the innovations form
# residuals' covariance matrix of the state-space model; :math:`\boldsymbol{S}`
# is :math:`\boldsymbol{\Sigma}` transformed by :math:`\boldsymbol{H}`; and
# :math:`\boldsymbol{\Sigma}_{IJ \lvert K} := \boldsymbol{\Sigma}_{IJ} -
# \boldsymbol{\Sigma}_{IK} \boldsymbol{\Sigma}_{KK}^{-1}
# \boldsymbol{\Sigma}_{KJ}`, representing a partial covariance matrix. The same
# principles apply as before: a numerator greater than the denominator means
# that information from the seed signals aids the prediction of activity in the
# target signals, leading to a Granger score > 0.
#
# There are several benefits to a state-space approach for computing GC:
# compared to traditional autoregressive-based approaches, the use of
# state-space models offers reduced statistical bias and increased statistical
# power; furthermore, the dimensionality reduction offered by the multivariate
# nature of the approach can aid in the interpretability and subsequent
# analysis of the results.
#
# To demonstrate the use of GC for estimating directed connectivity, we start
# by loading some example MEG data and dividing it into two-second-long epochs.

# %%

raw = mne.io.read_raw_ctf(data_path() / 'SubjectCMC.ds')
raw.pick_types(meg=True, eeg=False, ref_meg=False)
raw.crop(50., 110.).load_data()
raw.notch_filter(50)
raw.resample(100)

epochs = make_fixed_length_epochs(raw, duration=2.0).load_data()

###############################################################################
# We will focus on connectivity between sensors over the left and right
# hemispheres, with 75 sensors in the left hemisphere designated as group A,
# and 75 sensors in the right hemisphere designated as group B.

# %%

# left hemisphere sensors
signals_a = [idx for idx, ch_info in enumerate(epochs.info['chs']) if
             ch_info['loc'][0] < 0]
# right hemisphere sensors
signals_b = [idx for idx, ch_info in enumerate(epochs.info['chs']) if
             ch_info['loc'][0] > 0]

# UNTIL NEW INDICES FORMAT
min_n_chs = min(len(signals_a), len(signals_b))
signals_a = signals_a[:min_n_chs]
signals_b = signals_b[:min_n_chs]

indices_ab = (np.array(signals_a), np.array(signals_b))  # A -> B
indices_ba = (np.array(signals_b), np.array(signals_a))  # B -> A

signals_a_names = [epochs.info['ch_names'][idx] for idx in signals_a]
signals_b_names = [epochs.info['ch_names'][idx] for idx in signals_b]

# compute Granger causality
gc_ab = spectral_connectivity_epochs(
    epochs, method=['gc'], indices=indices_ab, fmin=5, fmax=30, rank=None,
    gc_n_lags=20)  # A -> B
gc_ba = spectral_connectivity_epochs(
    epochs, method=['gc'], indices=indices_ba, fmin=5, fmax=30, rank=None,
    gc_n_lags=20)  # B -> A
freqs = gc_ab.freqs


###############################################################################
# Plotting the results, we see that there is a flow of information from our
# left hemisphere (group A) to our right hemisphere (group B)...

# %%

fig = plt.figure()
plt.plot(freqs, gc_ab.get_data(), linewidth=2)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Connectivity (A.U.)')
plt.title('GC: [A -> B]')


###############################################################################
# Drivers and recievers: analysing the net direction of information flow
# ----------------------------------------------------------------------
#
# Although analysing connectivity in a given direction can be of interest,
# there may exist a bidirectional relationship between signals. In such cases,
# identifying the signals that dominate information flow (the drivers) may be
# desired. For this, we can simply subtract the Granger scores in the opposite
# direction, giving us the net GC score:
#
# :math:`F_{A \rarrow B}^(net) := F_{A \rarrow B} - F_{B \rarrow A}`.
#
# Doing so, we see that...

# %%

net_gc = gc_ab.get_data() - gc_ba.get_data()  # [A -> B] - [B -> A]

fig = plt.figure()
plt.plot(freqs, net_gc, linewidth=2)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Connectivity (A.U.)')
plt.title('Net GC: [A -> B] - [B -> A]')


###############################################################################
# Improving the robustness of connectivity estimates with time-reversal
# ---------------------------------------------------------------------
#
# One limitation of all GC methods is the risk of connectivity estimates being
# contaminated with noise. For instance, consider the case where, due to
# volume conduction, multiple sensors detect activity from the same source.
# Naturally, information recorded at these sensors mutually help predicting
# the activity of one another, leading to spurious estimates of directed
# connectivity which one may incorrectly attribute to information flow between
# different brain regions. On the other hand, even if there is no source mixing
# in the sensors, the presence of correlated noise between sensors can
# similarly bias directed connectivity estimates.
#
# To address this issue, Haufe *et al.* :footcite:`HaufeEtAl2013` propose
# contrasting causality scores obtained on the original time-series to those
# obtained on the reversed time-series. The idea behind this approach is as
# follows: if temporal order is crucial in distinguishing a driver from a
# recipient, then reversing the temporal order should reduce, if not flip, an
# estimate of directed connectivity. In practice, time-reversal is implemented
# as a transposition of the autocovariance sequence used to compute GC. Several
# studies have shown that that such an approach can reduce the degree of false
# positive connectivity estimates (even performing favourably against other
# methods such as the phase slope index) :footcite:`VinckEtAl2015` and retain
# the ability to correctly identify the net direction of information flow akin
# to net GC :footcite:`HaufeEtAl2013,WinklerEtAl2016`. This approach is termed
# time-reversed GC (TRGC):
#
# :math`\tilde{D}_{A \rarrow B}^(net) := F_{A \rarrow B}^(net) - F_{\tilde{A}
# \rarrow \tilde{B}}^(net)`,
#
# where :math:`\tilde` represents time-reversal, and:
#
# :math:`F_{\tilde{A} \rarrow \tilde{B}}^(net) := F_`F_{\tilde{A} \rarrow
# \tilde{B}} - F_`F_{\tilde{B} \rarrow \tilde{A}}`.
#
# GC on time-reversed signals can be computed simply with the
# ``method=['gc_tr']``, which will perform the time-reversal of the signals for
# the end-user. Note that **time-reversed results should only be interpreted in
# the context of net results**. In the example below, notice how the outputs
# are not used directly, but rather used to produce net scores of the
# time-reversed signals. The net scores of the time-reversed signals can then
# be subtracted from the net scores of the original signals to produce the
# final TRGC scores.

# %%

# compute GC on time-reversed signals
gc_tr_ab = spectral_connectivity_epochs(
    epochs, method=['gc_tr'], indices=indices_ab, fmin=5, fmax=30, rank=None,
    gc_n_lags=20)  # TR[A -> B]
gc_tr_ba = spectral_connectivity_epochs(
    epochs, method=['gc_tr'], indices=indices_ba, fmin=5, fmax=30, rank=None,
    gc_n_lags=20)  # TR[B -> A]

# compute net GC on time-reversed signals (TR[A -> B] - TR[B -> A])
net_gc_tr = gc_tr_ab.get_data() - gc_tr_ba.get_data()

# compute TRGC
trgc = net_gc - net_gc_tr

# plot TRGC
fig = plt.figure()
plt.plot(freqs, trgc, linewidth=2)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Connectivity (A.U.)')
plt.title('TRGC: net[A -> B] - net time-reversed[B -> A]')


###############################################################################
# Handling high-dimensional data
# ------------------------------
#
# An important issue to consider when using these multivariate methods is
# overfitting, which risks biasing connectivity estimates to noise in the data.
# This risk can be reduced by performing a preliminary dimensionality
# reduction prior to estimating the connectivity with a singular value
# decomposition. The degree of this dimensionality reduction can be specified
# using the ``rank`` argument, which by default will not perform any
# dimensionality reduction (assuming your data is full rank; see below if not).
# Choosing an expected rank of the data requires *a priori* knowledge about the
# number of components you expect to observe in the data.
#
# Here, we will be rather conservative and project our seed and target data to
# only the first 25 components of our rank subspace. Results for GC show...


###############################################################################
# In the case that your data is not full rank and ``rank`` is left as None, an
# automatic rank computation is performed and an appropriate degree of
# dimensionality reduction will be enforced.
#
# In the related case that your data is close to singular, the automatic rank
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
# References
# ----------
# .. footbibliography::
