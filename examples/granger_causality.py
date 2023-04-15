"""
==========================================================================
Compute directionality of connectivity with multivariate Granger causality
==========================================================================

This example demonstrates how Granger causality based on state-space models
:footcite:`BarnettSeth2015` can be used to compute directed connectivity
between sensors in a multivariate manner.
"""

# Author: Thomas S. Binns <t.s.binns@outlook.com>
# License: BSD (3-clause)

# %%

import numpy as np
from matplotlib import pyplot as plt

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
# :math:``,
#
# where


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
# multivariate manner :footcite:`EwaldEtAl2012`. This leads to the following
# methods: the maximised imaginary part of coherency (MIC); and the
# multivariate interaction measure (MIM).
#
# We start by loading some example MEG data and dividing it into
# two-second-long epochs.

# %%

raw = mne.io.read_raw_ctf(data_path() / 'SubjectCMC.ds')
raw.pick_types(meg=True, eeg=False, ref_meg=False)
raw.crop(50., 110.).load_data()
raw.notch_filter(50)
raw.resample(100)

epochs = make_fixed_length_epochs(raw, duration=2.0).load_data()


###############################################################################
# Improving the robustness of connectivity estimates with time-reversal
# ---------------------------------------------------------------------
#


###############################################################################
# Handling high-dimensional data
# ------------------------------
#
# An important issue to consider when using these multivariate methods is
# overfitting, which risks biasing connectivity estimates to noise in the data.
# This risk can be reduced by performing a preliminary dimensionality
# reduction prior to estimating the connectivity with a singular value
# decomposition. The degree of this dimensionality reduction can be specified
# using the `rank` argument, which by default will not perform any
# dimensionality reduction (assuming your data is full rank; see below if not).
# Choosing an expected rank of the data requires *a priori* knowledge about the
# number of components you expect to observe in the data.
#
# Here, we will be rather conservative and project our seed and target data to
# only the first 25 components of our rank subspace. Results for GC show...


###############################################################################
# In the case that your data is not full rank and `rank` is left as None, an
# automatic rank computation is performed and an appropriate degree of
# dimensionality reduction will be enforced.
#
# In the related case that your data is close to singular, the automatic rank
# computation may fail to detect this, and an error will be raised. In this
# case, you should inspect the singular values of your data to identify an
# appropriate degree of dimensionality reduction to perform, which you can then
# specify manually using the `rank` argument. The code below shows one
# possible approach for finding an appropriate rank of close-to-singular data.

# %%

# gets the singular values of the data
s = np.linalg.svd(raw.get_data(), compute_uv=False)
# finds how many singular values are "close" to the largest singular value
rank = np.count_nonzero(s >= s[0] * 1e-5)  # 1e-5 is the "closeness" criteria


###############################################################################
# Limitations
# -----------


###############################################################################
# References
# ----------
# .. footbibliography::
