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
# arrays.
#
# Such arrays are 'ragged', and support for ragged arrays is limited in NumPy
# to the ``object`` datatype. Not only is working with ragged arrays
# cumbersome, but saving arrays with ``dtype='object'`` is not supported by the
# h5netcdf engine used to save connectivity objects.
#
# The workaround used in MNE-Connectivity is to pad ragged arrays with some
# known values according to the largest number of entries in each dimension,
# such that there is an equal amount of information across and within
# connections for each dimension of the arrays.
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
#                     ((2, 3, 4), (4,        )))
#
#   # tuple of arrays
#   ragged_indices = (np.array([[0, 1   ], [0, 1, 2, 3]], dtype='object'),
#                     np.array([[2, 3, 4], [4         ]], dtype='object'))
#
# Just as for bivariate connectivity, the length of ``indices[0]`` and
# ``indices[1]`` is equal (i.e. the number of connections), however information
# about the multiple channel indices for each connection is stored in a nested
# array.
#
# Importantly, these indices are ragged, as the first connection will be
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

########################################################################################
# Extracting data for multivariate connectivity
# ---------------------------------------------
#
# Below, we compute multivariate connectivity using the maximised imaginary part of
# coherency (MIC; see :doc:`mic_mim` for more information). Just like for bivariate
# connectivity, we can extract the connectivity in ``'raveled'`` and ``'dense'`` forms.

# %%

# Create random data
data = np.random.randn(10, 5, 200)  # epochs x channels x times
sfreq = 50
ragged_indices = (
    [[0, 1], [0, 1, 2, 3]],  # seeds
    [[2, 3, 4], [4]],  # targets
)

# Compute multivariate connectivity
con = spectral_connectivity_epochs(
    data,
    method="mic",
    indices=ragged_indices,
    sfreq=sfreq,
    fmin=10,
    fmax=30,
    verbose=False,
)

########################################################################################
# The ``'raveled'`` form directly returns the connectivity data that is stored in the
# connectivity object. That is, an array of shape ``(n_connections, ...)``, where
# ``...`` represents the remaining dimensions of the connectivity data (e.g.,
# frequencies, times).

# %%

raveled_data = con.get_data(output="raveled")
print(f"Raveled connectivity shape: {raveled_data.shape} (connections x freqs)")

########################################################################################
# In contrast, the ``'dense'`` form requires a manipulation of the data into a square
# matrix of shape ``(n_nodes, n_nodes, ...)``. Since in the context of multivariate
# connectivity, a node can be a set of multiple channels, it is not possible to treat
# each row/column of the dense matrix as the entry for a single channel (as you would
# for bivariate connectivity).
#
# Because of this, the multivariate indices are mapped into a new dense matrix space,
# which allows a square matrix to be returned. To ensure the mapping from the original
# multivariate indices to the new dense matrix space is traceable, a tuple of
# multivariate nodes in the original indices are returned. This contains the (unmasked
# form) of each node, in the position where it exists in the dense matrix space.

# %%

dense_data, multivariate_nodes = con.get_data(output="dense")
print(f"Dense connectivity shape: {dense_data.shape} (nodes x nodes x freqs)")

# This mapping is done by taking the set of channels that define each node and assigning
# them a new index based on where they appear in the original seed indices, and then
# target indices.
print(
    f"Original ragged indices: seeds {ragged_indices[0]}; targets {ragged_indices[1]}"
)
print(
    f"Multivariate nodes: {tuple(node.tolist() for node in multivariate_nodes)}; "
    f"{len(multivariate_nodes)} total"
)
mapped_indices = (np.array([0, 1]), np.array([2, 3]))
print(
    f"Mapped indices in dense matrix space: "
    f"seeds {[node.tolist() for node in mapped_indices[0]]}; "
    f"targets {[node.tolist() for node in mapped_indices[1]]}"
)
assert np.all(raveled_data == dense_data[mapped_indices[0], mapped_indices[1]])

########################################################################################
# Working with spatial patterns of connectivity from ragged indices
# -----------------------------------------------------------------
#
# The maximised imaginary part of coherency (MIC) method also returns
# spatial patterns of connectivity, which show the contribution of each channel
# to the dimensionality-reduced connectivity estimate (explained in more detail
# in :doc:`mic_mim`). Because these patterns are returned for each channel,
# their shape can vary depending on the number of seeds and targets in each
# connection, making them ragged.
#
# To avoid this, the patterns are padded along the channel axis with the known
# and invalid entry ``np.nan``, in line with that applied to ``indices``.
# Extracting only the valid spatial patterns from the connectivity object is
# trivial, as shown below:

# %%

patterns = np.array(con.attrs["patterns"])
padded_indices = con.indices
n_freqs = con.get_data().shape[-1]
n_cons = len(ragged_indices[0])
max_n_chans = max(len(inds) for inds in ([*ragged_indices[0], *ragged_indices[1]]))

# Show that the padded indices entries are masked
assert np.sum(padded_indices[0][0].mask) == 2  # 2 padded channels
assert np.sum(padded_indices[1][0].mask) == 1  # 1 padded channels
assert np.sum(padded_indices[0][1].mask) == 0  # 0 padded channels
assert np.sum(padded_indices[1][1].mask) == 3  # 3 padded channels

# Patterns have shape [seeds/targets x cons x max channels x freqs (x times)]
assert patterns.shape == (2, n_cons, max_n_chans, n_freqs)

# Show that the padded patterns entries are all np.nan
assert np.all(np.isnan(patterns[0, 0, 2:]))  # 2 padded channels
assert np.all(np.isnan(patterns[1, 0, 3:]))  # 1 padded channels
assert not np.any(np.isnan(patterns[0, 1]))  # 0 padded channels
assert np.all(np.isnan(patterns[1, 1, 1:]))  # 3 padded channels

# Extract patterns for first connection using the ragged indices
seed_patterns_con1 = patterns[0, 0, : len(ragged_indices[0][0])]
target_patterns_con1 = patterns[1, 0, : len(ragged_indices[1][0])]

# Extract patterns for second connection using the padded, masked indices
seed_patterns_con2 = patterns[0, 1, : padded_indices[0][1].count()]
target_patterns_con2 = patterns[1, 1, : padded_indices[1][1].count()]

# Show that shapes of patterns are correct
assert seed_patterns_con1.shape == (2, n_freqs)  # channels (0, 1)
assert target_patterns_con1.shape == (3, n_freqs)  # channels (2, 3, 4)
assert seed_patterns_con2.shape == (4, n_freqs)  # channels (0, 1, 2, 3)
assert target_patterns_con2.shape == (1, n_freqs)  # channels (4)

print("Assertions completed successfully!")
