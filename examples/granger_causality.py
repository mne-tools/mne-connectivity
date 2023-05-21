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
# causality (GC). A signal, :math:`\boldsymbol{x}`, is said to Granger-cause
# another signal, :math:`\boldsymbol{y}`, if information from the past of
# :math:`\boldsymbol{x}` improves the prediction of the present of
# :math:`\boldsymbol{y}` over the case where only information from the past of
# :math:`\boldsymbol{y}` is used. Note: GC does not make any assertions about
# the true causality between signals.
#
# The degree to which :math:`\boldsymbol{x}` and :math:`\boldsymbol{y}` can be
# used to predict one another can be quantified using vector autoregressive
# (VAR) models. Considering the simpler case of time domain connectivity, the
# VAR models are as follows:
#
# :math:`y_t = \sum_{k=1}^{K} a_k y_{t-k} + \xi_t^y` ,
# :math:`Var(\xi_t^y) := \Sigma_y` ,
#
# and :math:`\boldsymbol{z}_t = \sum_{k=1}^K \boldsymbol{A}_k
# \boldsymbol{z}_{t-k} + \boldsymbol{\epsilon}_t` ,
# :math:`\boldsymbol{\Sigma} := \langle \boldsymbol{\epsilon}_t
# \boldsymbol{\epsilon}_t^T \rangle = \begin{bmatrix} \Sigma_{xx} & \Sigma_{xy}
# \\ \Sigma_{yx} & \Sigma_{yy} \end{bmatrix}` ,
#
# representing the reduced and full VAR models, respectively, where: :math:`K`
# is the order of the VAR model, determining the number of lags, :math:`k`,
# used; :math:`\boldsymbol{Z} := \begin{bmatrix} \boldsymbol{x} \\
# \boldsymbol{y} \end{bmatrix}`; and :math:`\xi` and
# :math:`\boldsymbol{\epsilon}` are the residuals of the VAR models. In this
# way, the information of the signals at time :math:`t` can be represented as a
# weighted form of the information from the previous timepoints, plus some
# residual information not encoded in the signals' past. In practice, VAR model
# parameters are computed from an autocovariance sequence generated from the
# time-series data using the Yule-Walker equations :footcite:`Whittle1963`.
#
# By comparing the residuals, or errors, of the reduced and full VAR models, we
# can therefore estimate how much :math:`\boldsymbol{x}` Granger-causes
# :math:`\boldsymbol{y}`:
#
# :math:`F_{x \rightarrow y} = ln \Large{(\frac{\Sigma_y}{\Sigma_{yy}})}` ,
#
# where :math:`F` is the Granger score. For example, if :math:`\boldsymbol{x}`
# contains no information about :math:`\boldsymbol{y}`, the residuals of the
# reduced and full VAR models will be identical, and
# :math:`F_{x \rightarrow y}` will naturally be 0, indicating that
# information from :math:`\boldsymbol{x}` does not flow to
# :math:`\boldsymbol{y}`. In contrast, if :math:`\boldsymbol{x}` does help to
# predict :math:`\boldsymbol{y}`, the residual of the full model will be
# smaller than that of the reduced model. :math:`\Large{\frac{\Sigma_y}
# {\Sigma_{yy}}}` will therefore be greater than 1, leading to a Granger score
# > 0. Granger scores are bound between :math:`[0, \infty)`.
#
# These same principles apply to spectral GC, which provides information about
# the directionality of connectivity for individual frequencies. For spectral
# GC, the autocovariance sequence is generated from an inverse Fourier
# transform applied to the cross-spectral density of the signals. Additionally,
# a spectral transfer function is used to translate information from the VAR
# models back into the frequency domain before computing the final Granger
# scores.
#
# Barnett and Seth (2015) :footcite:`BarnettSeth2015` have defined a
# multivariate form of spectral GC based on state-space models, enabling the
# estimation of information flow between whole sets of signals simultaneously:
#
# :math:`F_{A \rightarrow B}(f) = \Re ln \Large{(\frac{
# det(\boldsymbol{S}_{BB}(f))}{det(\boldsymbol{S}_{BB}(f) -
# \boldsymbol{H}_{BA}(f) \boldsymbol{\Sigma}_{AA \lvert B}
# \boldsymbol{H}_{BA}^*(f))})}` ,
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
raw.pick('mag')
raw.crop(50., 110.).load_data()
raw.notch_filter(50)
raw.resample(100)

epochs = mne.make_fixed_length_epochs(raw, duration=2.0).load_data()

###############################################################################
# We will focus on connectivity between sensors over the parietal and occipital
# cortices, with 20 parietal sensors designated as group A, and 20 occipital
# sensors designated as group B.

# %%

# parietal sensors
signals_a = [idx for idx, ch_info in enumerate(epochs.info['chs']) if
             ch_info['ch_name'][2] == 'P']
# occipital sensors
signals_b = [idx for idx, ch_info in enumerate(epochs.info['chs']) if
             ch_info['ch_name'][2] == 'O']

# UNTIL RAGGED INDICES SUPPORTED
min_n_chs = min(len(signals_a), len(signals_b))
signals_a = signals_a[:min_n_chs]
signals_b = signals_b[:min_n_chs]

indices_ab = (np.array(signals_a), np.array(signals_b))  # A => B
indices_ba = (np.array(signals_b), np.array(signals_a))  # B => A

signals_a_names = [epochs.info['ch_names'][idx] for idx in signals_a]
signals_b_names = [epochs.info['ch_names'][idx] for idx in signals_b]

# compute Granger causality
gc_ab = spectral_connectivity_epochs(
    epochs, method=['gc'], indices=indices_ab, fmin=5, fmax=30,
    rank=(np.array([5]), np.array([5])), gc_n_lags=20)  # A => B
gc_ba = spectral_connectivity_epochs(
    epochs, method=['gc'], indices=indices_ba, fmin=5, fmax=30,
    rank=(np.array([5]), np.array([5])), gc_n_lags=20)  # B => A
freqs = gc_ab.freqs


###############################################################################
# Plotting the results, we see that there is a flow of information from our
# parietal sensors (group A) to our occipital sensors (group B) with noticeable
# peaks at ~8 Hz and ~13 Hz.

# %%

fig, axis = plt.subplots(1, 1)
axis.plot(freqs, gc_ab.get_data()[0], linewidth=2)
axis.set_xlabel('Frequency (Hz)')
axis.set_ylabel('Connectivity (A.U.)')
fig.suptitle('GC: [A => B]')


###############################################################################
# Drivers and receivers: analysing the net direction of information flow
# ----------------------------------------------------------------------
#
# Although analysing connectivity in a given direction can be of interest,
# there may exist a bidirectional relationship between signals. In such cases,
# identifying the signals that dominate information flow (the drivers) may be
# desired. For this, we can simply subtract the Granger scores in the opposite
# direction, giving us the net GC score:
#
# :math:`F_{A \rightarrow B}^{net} := F_{A \rightarrow B} -
# F_{B \rightarrow A}`.
#
# Doing so, we see that the flow of information around 8 Hz is dominant from
# parietal to occipital sensors (indicated by the positive-valued Granger
# scores). However, at 13, 18, and 24 Hz, information flow is dominant in the
# occipital to parietal direction (as shown by the negative-valued Granger
# scores).

# %%

net_gc = gc_ab.get_data() - gc_ba.get_data()  # [A => B] - [B => A]

fig, axis = plt.subplots(1, 1)
axis.plot((freqs[0], freqs[-1]), (0, 0), linewidth=2, linestyle='--',
          color='k')
axis.plot(freqs, net_gc[0], linewidth=2)
axis.set_xlabel('Frequency (Hz)')
axis.set_ylabel('Connectivity (A.U.)')
fig.suptitle('Net GC: [A => B] - [B => A]')


###############################################################################
# Improving the robustness of connectivity estimates with time-reversal
# ---------------------------------------------------------------------
#
# One limitation of GC methods is the risk of connectivity estimates being
# contaminated with noise. For instance, consider the case where, due to
# volume conduction, multiple sensors detect activity from the same source.
# Naturally, information recorded at these sensors mutually help to predict
# the activity of one another, leading to spurious estimates of directed
# connectivity which one may incorrectly attribute to information flow between
# different brain regions. On the other hand, even if there is no source
# mixing, the presence of correlated noise between sensors can similarly bias
# directed connectivity estimates.
#
# To address this issue, Haufe *et al.* (2013) :footcite:`HaufeEtAl2013`
# propose contrasting causality scores obtained on the original time-series to
# those obtained on the reversed time-series. The idea behind this approach is
# as follows: if temporal order is crucial in distinguishing a driver from a
# recipient, then reversing the temporal order should reduce, if not flip, an
# estimate of directed connectivity. In practice, time-reversal is implemented
# as a transposition of the autocovariance sequence used to compute GC. Several
# studies have shown that that such an approach can reduce the degree of
# false-positive connectivity estimates (even performing favourably against
# other methods such as the phase slope index) :footcite:`VinckEtAl2015` and
# retain the ability to correctly identify the net direction of information
# flow akin to net GC :footcite:`WinklerEtAl2016,HaufeEtAl2013`. This approach
# is termed time-reversed GC (TRGC):
#
# :math:`\tilde{D}_{A \rightarrow B}^{net} := F_{A \rightarrow B}^{net} -
# F_{\tilde{A} \rightarrow \tilde{B}}^{net}` ,
#
# where :math:`\sim` represents time-reversal, and:
#
# :math:`F_{\tilde{A} \rightarrow \tilde{B}}^{net} := F_{\tilde{A} \rightarrow
# \tilde{B}} - F_{\tilde{B} \rightarrow \tilde{A}}`.
#
# GC on time-reversed signals can be computed simply with ``method=['gc_tr']``,
# which will perform the time-reversal of the signals for the end-user. Note
# that **time-reversed results should only be interpreted in the context of net
# results**. In the example below, notice how the outputs are not used
# directly, but rather used to produce net scores of the time-reversed signals.
# The net scores of the time-reversed signals can then be subtracted from the
# net scores of the original signals to produce the final TRGC scores.

# %%

# compute GC on time-reversed signals
gc_tr_ab = spectral_connectivity_epochs(
    epochs, method=['gc_tr'], indices=indices_ab, fmin=5, fmax=30,
    rank=(np.array([5]), np.array([5])), gc_n_lags=20)  # TR[A => B]
gc_tr_ba = spectral_connectivity_epochs(
    epochs, method=['gc_tr'], indices=indices_ba, fmin=5, fmax=30,
    rank=(np.array([5]), np.array([5])), gc_n_lags=20)  # TR[B => A]

# compute net GC on time-reversed signals (TR[A => B] - TR[B => A])
net_gc_tr = gc_tr_ab.get_data() - gc_tr_ba.get_data()

# compute TRGC
trgc = net_gc - net_gc_tr

###############################################################################
# Plotting the TRGC results, there is a clear peak for information flow
# dominant in the parietal to occipital direction ~12 Hz.  Additionally, there
# is dominance of information flow from occipital to parietal sensors around
# 18-22 Hz. As with the net GC scores, this lower-higher frequency contrast
# between the directions of information flow remains present, however that is
# not to say the spectral patterns are identical (e.g. the 24 Hz negative peak
# is absent for TRGC). The absence of certain connectivity peaks for TRGC
# suggests that these estimates may have been spurious connectivity resulting
# from source mixing or correlated noise in the recordings. Altogether, the use
# of TRGC instead of net GC is generally advised.

# %%

fig, axis = plt.subplots(1, 1)
axis.plot((freqs[0], freqs[-1]), (0, 0), linewidth=2, linestyle='--',
          color='k')
axis.plot(freqs, trgc[0], linewidth=2)
axis.set_xlabel('Frequency (Hz)')
axis.set_ylabel('Connectivity (A.U.)')
fig.suptitle('TRGC: net[A => B] - net time-reversed[A => B]')


###############################################################################
# Controlling spectral smoothing with the number of lags
# ------------------------------------------------------
#
# One important parameter when computing GC is the number of lags used when
# computing the VAR model. A lower number of lags reduces the computational
# cost, but in the context of spectral GC, leads to a smoothing of Granger
# scores across frequencies. The number of lags can be specified using the
# ``gc_n_lags`` parameter. The default value is 40, however there is no correct
# number of lags to use when computing GC. Instead, you have to use your own
# best judgement of whether or not your Granger scores look overly smooth.
#
# Below is a comparison of Granger scores computed with a different number of
# lags. In the above examples we had used 20 lags, which we will compare to
# Granger scores computed with 60 lags. As you can see, the spectra of Granger
# scores computed with 60 lags is noticeably less smooth, but it does share the
# same overall pattern.

# %%

gc_ab_60 = spectral_connectivity_epochs(
    epochs, method=['gc'], indices=indices_ab, fmin=5, fmax=30,
    rank=(np.array([5]), np.array([5])), gc_n_lags=60)  # A => B

fig, axis = plt.subplots(1, 1)
axis.plot(freqs, gc_ab.get_data()[0], linewidth=2, label='20 lags')
axis.plot(freqs, gc_ab_60.get_data()[0], linewidth=2, label='60 lags')
axis.set_xlabel('Frequency (Hz)')
axis.set_ylabel('Connectivity (A.U.)')
axis.legend()
fig.suptitle('GC: [A => B]')


###############################################################################
# Handling high-dimensional data
# ------------------------------
#
# An important issue to consider when computing multivariate GC is that the
# data GC is computed on should not be rank deficient (i.e. must have full
# rank). More specifically, the autocovariance matrix must not be singular or
# close to singular.
#
# In the case that your data is not full rank and ``rank`` is left as ``None``,
# an automatic rank computation is performed and an appropriate degree of
# dimensionality reduction will be enforced. However, the automatic rank
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
# Nonethless, even in situations where you specify an appropriate rank, it is
# not guaranteed that the subsequently-computed autocovariance sequence will
# retain this non-singularity (this can depend on, e.g. the number of lags).
# Hence, you may also encounter situations where you have to specify a rank
# less than that of your data to ensure that the autocovariance sequence is
# non-singular.
#
# In the above examples, notice how a rank of 5 was given, despite there being
# 20 channels in the seeds and targets. Attempting to compute GC on the
# original data would not succeed, given that the resulting autocovariance
# sequence is singular, as the example below shows.

# %%

try:
    spectral_connectivity_epochs(
        epochs, method=['gc'], indices=indices_ab, fmin=5, fmax=30, rank=None,
        gc_n_lags=20)  # A => B
    print('Success!')
except RuntimeError as error:
    print('\nCaught the following error:\n' + repr(error))

###############################################################################
# Rigorous checks are implemented to identify any such instances which would
# otherwise cause the GC computation to produce erroneous results. You can
# therefore be confident as an end-user that these cases will be caught.


###############################################################################
# References
# ----------
# .. footbibliography::
