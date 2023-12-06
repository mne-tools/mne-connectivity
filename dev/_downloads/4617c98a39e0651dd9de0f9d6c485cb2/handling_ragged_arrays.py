"""
=========================================================
Working with ragged indices for multivariate connectivity
=========================================================

This example demonstrates how multivariate connectivity involving different
numbers of seeds and targets can be handled in MNE-Connectivity.
"""

# Author: Thomas S. Binns <t.s.binns@outlook.com>
# License: BSD (3-clause)

# %%

import numpy as np

from mne_connectivity import spectral_connectivity_epochs

###############################################################################
# Background
# ----------
#
# With multivariate connectivity, interactions between multiple signals can be
# considered together, and the number of signals designated as seeds and
# targets does not have to be equal within or across connections. Issues can
# arise from this when storing information associated with connectivity in
# arrays, as the number of entries within each dimension can vary within and
# across connections depending on the number of seeds and targets. Such arrays
# are 'ragged', and support for ragged arrays is limited in NumPy to the
# ``object`` datatype. Not only is working with ragged arrays is cumbersome,
# but saving arrays with ``dtype='object'`` is not supported by the h5netcdf
# engine used to save connectivity objects. The workaround used in
# MNE-Connectivity is to pad ragged arrays with some known values according to
# the largest number of entries in each dimension, such that there is an equal
# amount of information across and within connections for each dimension of the
# arrays.
#
# As an example, consider we have 5 channels and want to compute 2 connections:
# the first between channels in indices 0 and 1 with those in indices 2, 3,
# and 4; and the second between channels 0, 1, 2, and 3 with channel 4. The
# seed and target indices can be written as such::
#
#   seeds   = [[0, 1   ], [0, 1, 2, 3]]
#   targets = [[2, 3, 4], [4         ]]
#
# The ``indices`` parameter passed to
# :func:`~mne_connectivity.spectral_connectivity_epochs` and
# :func:`~mne_connectivity.spectral_connectivity_time` must be a tuple of
# array-likes, meaning
# that the indices can be passed as a tuple of: lists; tuples; or NumPy arrays.
# Examples of how ``indices`` can be formed are shown below::
#
#   # tuple of lists
#   ragged_indices = ([[0, 1   ], [0, 1, 2, 3]],
#                     [[2, 3, 4], [4         ]])
#
#   # tuple of tuples
#   ragged_indices = (((0, 1   ), (0, 1, 2, 3)),
#                     ((2, 3, 4), (4         )))
#
#   # tuple of arrays
#   ragged_indices = (np.array([[0, 1   ], [0, 1, 2, 3]], dtype='object'),
#                     np.array([[2, 3, 4], [4         ]], dtype='object'))
#
# **N.B. Note that since NumPy v1.19.0, dtype='object' must be specified when
# forming ragged arrays.**
#
# Just as for bivariate connectivity, the length of ``indices[0]`` and
# ``indices[1]`` is equal (i.e. the number of connections), however information
# about the multiple channel indices for each connection is stored in a nested
# array. Importantly, these indices are ragged, as the first connection will be
# computed between 2 seed and 3 target channels, and the second connection
# between 4 seed and 1 target channel(s). The connectivity functions will
# recognise the indices as being ragged, and pad them to a 'full' array by
# adding placeholder values which are masked accordingly. This makes the
# indices easier to work with, and also compatible with the engine used to save
# connectivity objects. For example, the above indices would become::
#
#   padded_indices = (np.array([[0, 1, --, --], [0,  1,  2,  3]]),
#                     np.array([[2, 3,  4, --], [4, --, --, --]]))
#
# where ``--`` are masked entries. These indices are what is stored in the
# returned connectivity objects.
#
# For the connectivity results themselves, the methods available in
# MNE-Connectivity combine information across the different channels into a
# single (time-)frequency-resolved connectivity spectrum, regardless of the
# number of seed and target channels, so ragged arrays are not a concern here.
# However, the maximised imaginary part of coherency (MIC) method also returns
# spatial patterns of connectivity, which show the contribution of each channel
# to the dimensionality-reduced connectivity estimate (explained in more detail
# in :doc:`mic_mim`). Because these patterns are returned for each channel,
# their shape can vary depending on the number of seeds and targets in each
# connection, making them ragged. To avoid this, the patterns are padded along
# the channel axis with the known and invalid entry ``np.nan``, in line with
# that applied to ``indices``. Extracting only the valid spatial patterns from
# the connectivity object is trivial, as shown below:

# %%

# create random data
data = np.random.randn(10, 5, 200)  # epochs x channels x times
sfreq = 50
ragged_indices = ([[0, 1], [0, 1, 2, 3]], [[2, 3, 4], [4]])  # seeds  # targets

# compute connectivity
con = spectral_connectivity_epochs(
    data,
    method="mic",
    indices=ragged_indices,
    sfreq=sfreq,
    fmin=10,
    fmax=30,
    verbose=False,
)
patterns = np.array(con.attrs["patterns"])
padded_indices = con.indices
n_freqs = con.get_data().shape[-1]
n_cons = len(ragged_indices[0])
max_n_chans = max(len(inds) for inds in ([*ragged_indices[0], *ragged_indices[1]]))

# show that the padded indices entries are masked
assert np.sum(padded_indices[0][0].mask) == 2  # 2 padded channels
assert np.sum(padded_indices[1][0].mask) == 1  # 1 padded channels
assert np.sum(padded_indices[0][1].mask) == 0  # 0 padded channels
assert np.sum(padded_indices[1][1].mask) == 3  # 3 padded channels

# patterns have shape [seeds/targets x cons x max channels x freqs (x times)]
assert patterns.shape == (2, n_cons, max_n_chans, n_freqs)

# show that the padded patterns entries are all np.nan
assert np.all(np.isnan(patterns[0, 0, 2:]))  # 2 padded channels
assert np.all(np.isnan(patterns[1, 0, 3:]))  # 1 padded channels
assert not np.any(np.isnan(patterns[0, 1]))  # 0 padded channels
assert np.all(np.isnan(patterns[1, 1, 1:]))  # 3 padded channels

# extract patterns for first connection using the ragged indices
seed_patterns_con1 = patterns[0, 0, : len(ragged_indices[0][0])]
target_patterns_con1 = patterns[1, 0, : len(ragged_indices[1][0])]

# extract patterns for second connection using the padded, masked indices
seed_patterns_con2 = patterns[0, 1, : padded_indices[0][1].count()]
target_patterns_con2 = patterns[1, 1, : padded_indices[1][1].count()]

# show that shapes of patterns are correct
assert seed_patterns_con1.shape == (2, n_freqs)  # channels (0, 1)
assert target_patterns_con1.shape == (3, n_freqs)  # channels (2, 3, 4)
assert seed_patterns_con2.shape == (4, n_freqs)  # channels (0, 1, 2, 3)
assert target_patterns_con2.shape == (1, n_freqs)  # channels (4)

print("Assertions completed successfully!")

# %%
