"""
================================================
Export a connectivity object to a NetworkX graph
================================================

This example shows how :class:`~mne_connectivity.Connectivity` containers can be
converted to :class:`networkx.Graph` and :class:`networkx.DiGraph` objects.

We compute sensor-space spectral connectivity in the alpha band on the MNE's sample
dataset and export it to NetworkX graph objects.
"""

# Authors: Raphaël Bordas <bordasraph@gmail.com>
#          Marie-Constance Corsi <marie-constance.corsi@inria.fr>
#
# License: BSD (3-clause)

# %%

import os.path as op

import mne
import numpy as np
from mne.datasets import sample

from mne_connectivity import spectral_connectivity_epochs

########################################################################################
# Load the data and create epochs
# -------------------------------

# %%

# Set parameters
data_path = sample.data_path()
raw_fname = op.join(data_path, "MEG", "sample", "sample_audvis_filt-0-40_raw.fif")
event_fname = op.join(data_path, "MEG", "sample", "sample_audvis_filt-0-40_raw-eve.fif")

# Setup for reading the raw data
raw = mne.io.read_raw_fif(raw_fname)
events = mne.read_events(event_fname)

# To keep the example fast, we only take a handful of gradiometers spread over the
# helmet.
picks = mne.pick_types(raw.info, meg="grad", eeg=False, stim=False, exclude="bads")
picks = picks[::16]  # ~13 sensors, evenly spread over the array
raw.pick(picks)

# Create epochs for the auditory condition
event_id, tmin, tmax = {"Auditory/Left": 1}, -0.2, 0.5
epochs = mne.Epochs(
    raw,
    events,
    event_id,
    tmin,
    tmax,
    baseline=(None, 0),
    reject=dict(grad=4000e-13),
    preload=True,
)
print(epochs)

########################################################################################
# Compute connectivity
# --------------------
# We estimate connectivity using two measures:
# - the weighted phase lag index, a non-directed connectivity measure
# - and the directed phase lag index, a directed connectivity measure
#
# For both measures, we compute connectivity in the alpha band (8-13 Hz) on the
# post-stimulus window. ``faverage=True`` averages over the frequency bins, to give us
# the average connectivity in the alpha band.
#
# Because wPLI is a non-directed measure, we can save time and memory by computing only
# the lower-triangular part of the connectivity matrix (which is identical to the upper
# triangular part). For dPLI, we need to compute the full connectivity matrix.

# %%

conn_kwargs = dict(fmin=8.0, fmax=13.0, faverage=True, tmin=0.0)

tril_indices = np.tril_indices(len(epochs.ch_names), k=-1)  # exclude diagonal
wpli = spectral_connectivity_epochs(
    epochs, method="wpli2_debiased", indices=tril_indices, **conn_kwargs
)

full_indices = np.indices((len(epochs.ch_names), len(epochs.ch_names)))
full_indices = (full_indices[0].ravel(), full_indices[1].ravel())
dpli = spectral_connectivity_epochs(
    epochs, method="dpli", indices=full_indices, **conn_kwargs
)

########################################################################################
# Convert connectivity to a NetworkX graph
# ----------------------------------------
# The :meth:`~mne_connectivity.Connectivity.to_networkx` method of the connectivity
# containers allows the connectivity data to be exported to a NetworkX graph. Two types
# of graphs are supported: undirected graphs; and directed graphs. The ``directed``
# parameter of the :meth:`~mne_connectivity.Connectivity.to_networkx` method allows for
# control over which type of graph is returned.
#
# Because wPLI is non-directed, we set ``directed=False``, which returns an undirected
# :class:`~networkx.Graph`. For dPLI, we set ``directed=True``, which returns a directed
# :class:`~networkx.DiGraph`.
#
# :class:`~networkx.Graph` objects are designed for representing data from non-directed
# (symmetric) connectivity measures (e.g., coherence, PLV, wPLI, ...), while
# :class:`~networkx.DiGraph` objects are good for directed (non-symmetric) measures
# (e.g., Granger causality, dPLI, ...).

# %%

# Convert to graphs and select the single entry for the averaged alpha band
wpli_graph = wpli.to_networkx(directed=False)[0]
dpli_graph = dpli.to_networkx(directed=True)[0]

for graph in (wpli_graph, dpli_graph):
    print(f"Graph type      : {type(graph).__name__}")
    print(f"Number of nodes : {graph.number_of_nodes()}")
    print(f"Number of edges : {graph.number_of_edges()}")
    print(f"Directed        : {graph.is_directed()}\n")

########################################################################################
# In the exported graphs, the nodes are labelled with the channel names, and the edge
# weights of the nodes carry the connectivity values.
#
# You may notice that for the wPLI graph, the first connection is a self-connection of
# the first node, even though this was not a connection in the original set computed
# using the lower-triangular indices. This is because when converting from the
# underlying NumPy array to a NetworkX graph, a 2D ``[nodes, nodes]`` array is expected.
# For any missing connections, we fill these values with ``numpy.nan``, as you can see
# from the edge weight.

# %%

print("First 5 nodes:", list(wpli_graph.nodes)[:5])
print("First 5 edges:")
for u, v, w in list(wpli_graph.edges(data="weight"))[:5]:
    print(f"  {u} -- {v}: {w:.3f}")

########################################################################################
# Working with NetworkX graphs
# ----------------------------
# Once the connectivity data is exported to a graph object, NetworkX offers a wide range
# of tools for graph analysis, with an extensive set of tutorials demonstrating these
# capabilities in their `documentation <https://networkx.org/documentation/stable/>`_.
