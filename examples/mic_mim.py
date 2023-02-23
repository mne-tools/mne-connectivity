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

# Authors: Thomas S. Binns <t.s.binns@outlook.com>
#
# License: BSD (3-clause)


import numpy as np

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
# multivariate manner :footcite:`EwaldEtAl2012`. This leads to the following
# methods: the maximised imaginary part of coherency (MIC); and the
# multivariate interaction measure (MIM).


# LOAD EXAMPLE DATA, VISUALISE THE CHANNELS AS A TOPOMAP, AND COMPUTE
# CONNECTIVITY HERE, THEN DISCUSS WHAT THE RESULTS MEAN BELOW; COULD ALSO
# COMPUTE REGULAR ICOH FOR COMPARISON


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
# :math:`MIC=\frac{\boldsymbol{\alpha E \beta}}{\parallel\boldsymbol{\alpha}
# \parallel \parallel\boldsymbol{\beta}\parallel}`,
#
# where :math:`\boldsymbol{\alpha}` and :math:`\boldsymbol{\beta}` are the
# spatial filters for the seeds and targets, respectively, and
# :math:`\boldsymbol{E}` is the imaginary part of the transformed
# cross-spectral density between the seeds and targets. MIC is bound between
# :math:`[-1, 1]` where the absolute value reflects connectivity strength and
# the sign reflects the phase angle.


# PLOT THE RESULTS FOR MIC (AND POSSIBLY COMPARE TO REGULAR ICOH)


# Furthermore, spatial patterns of connectivity can be constructed from the
# spatial filters to give a picture of the location of the sources involved in
# the connectivity. This information is stored under the `patterns` attribute
# of the connectivity class, with one value per frequency for each channel in
# the seeds and target. As with MIC, the absolute value of the patterns
# reflect the strength, however the sign differences can be used to visualise
# the orientation of the underlying dipole sources. The spatial patterns are
# not bound between :math:`[-1, 1]`.


# PLOT THE SPATIAL PATTERNS FOR SEEDS AND TARGETS


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
# :math:`MIM=tr(\boldsymbol{EE}^T)`.
#
# Unlike MIC, MIM is positive-valued, and is not bound below 1. MIM can be
# normalised to lie in the range :math:`[0, 1]` by dividing by :math:`N/2`
# (where :math:`N` is the number of channels in the seeds and targets). **MIM
# is not normalised by default in our implementation**. Without normalisation,
# MIM can be thought of as reflecting the total interaction between the seeds
# and targets, whereas normalised MIM reflects the total interaction *per
# channel*.


# PLOT THE RESULTS FOR MIM (COULD ALSO COMPARE NORMALISED AND NON-NORMALISED
# MIM; COULD ALSO COMPARE TO MIC)


# Additionally, the case where the seeds and targets are identical can be
# considered as a special case of MIM: the global interaction measure (GIM; Eq.
# 15 of :footcite:`EwaldEtAl2012`). This allows connectivity within a single
# set of signals to be estimated. Computing GIM follows from Eq. 14, however
# since each interaction is considered twice, correcting the connectivity by a
# factor of :math:`\frac{1}{2}` is necessary (**this correction is performed
# automatically in our implementation**). A similar principle applies in the
# case where MIC is computed for identical seeds and targets.

# COMPUTE AND PLOT GIM

###############################################################################
# Handling high-dimensional data
# ------------------------------
#
# An important issue to consider when using these multivariate methods is
# overfitting, which risks biasing connectivity estimates. This risk can be
# reduced by performing a preliminary dimensionality reduction prior to
# estimating the connectivity with a singular value decomposition (Eqs. 32 & 33
# of :footcite:`EwaldEtAl2012`). The degree of this dimensionality reduction
# can be specified using the `rank` argument, which by default will not perform
# any dimensionality reduction (assuming your data is full rank; see below).


# DEMONSTRATE HOW `RANK` CAN BE USED AND PLOT THE RESULTS; SHOW HOW PATTERNS
# STILL RETURNS INFORMATION FOR EACH CHANNEL WITH MIC


# In the above cases, the data used has been full rank. Here we artificially
# reduce the rank of the example data by repeating one of the channels (an
# extreme example, but valid nonetheless).


# CREATE SOME NON-FULL RANK DATA AND COMPUTE MIC ON IT, THEN SHOW THE RESULTS
# AND THE SHAPE OF THE SPATIAL PATTERNS


# In the case that your data is not full rank and `rank` is left as default, an
# automatic rank computation is performed anyway and an appropriate degree of
# dimensionality reduction will be enforced. Notice that the spatial patterns
# are returned for the original sensor space, which is reconstructed using the
# products of the singular value decomposition (Eqs. 46 & 47 of
# :footcite:`EwaldEtAl2012`).
#
# In the related case that your data is close to singular, the automatic rank
# computation may fail to detect this, and an error will be raised. In this
# case, you should inspect the singular values of your data to identify an
# appropriate degree of dimensionality reduction to perform, which you can then
# specify manually using the `rank` argument. The example below shows one
# possible approach for finding an appropriate rank of close-to-singular data.

# gets the singular values of the data
s = np.linalg.svd(data, compute_uv=False)
# finds how many singular values are "close" to the largest singular value
rank = np.count_nonzero(s >= s[0] * 1e-5)

###############################################################################
# Limitations
# -----------
#
# These multivariate methods offer many benefits in the form of dimensionality
# reduction, signal-to-noise ratio improvements, and invariance to estimate
# biasing source mixing; however, no method is perfect. The immunity of the
# imaginary part of coherency to volume conduction comes from the fact that
# these artefacts have zero phase lag, and hence a zero-valued imaginary
# component. By projecting the complex-valued coherency to the imaginary axis,
# a signal of a given magnitude with a phase lag of 90° or 270° would see its
# contribution to the connectivity estimate increased relative to a comparable
# signal with a phase lag close to 0° or 180°. Therefore, the imaginary part of
# coherency is biased towards connectivity involving 90° and 270° phase lag
# components.
#
# Whilst this is not a limitation specific to the multivariate extension of
# this measure, these multivariate methods can introduce further bias. When
# maximising the imaginary part of coherency, 90° and 270° phase lag components
# will likely give us higher connectivity estimates, and so will be prioritised
# by our spatial filters.
#
# Such a limitation should be kept in mind when estimating connectivity using
# these methods. Possible sanity checks can involve comparing the spectral
# profiles of MIC/MIM to coherence and the imaginary part of coherency
# computed on the same data, as well as comparing to other multivariate
# measures, such as canonical coherence :footcite:`:VidaurreEtAl2019:`.
