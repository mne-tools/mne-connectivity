"""
======================================
Learn a functional connectivity graph
======================================

This example demonstrates :func:`~mne_connectivity.learn_graph`, which estimates a
sparse functional connectivity graph directly from signal smoothness
:footcite:`Kalofolias2016`, rather than from pairwise similarity/coherence computed
independently per edge. All edge weights are learned jointly, which was found to be
more robust than Phase Lag Index for short recordings and to better capture
individual-specific connectivity in real MEG data :footcite:`ShabestariEtAl2026`.

Using synthetic data with a known structure, we show how well the learned graph
recovers a ground-truth graph, how ``beta`` controls the sparsity of the learned graph,
and how a graph learned from several smooth signals recovers the geometry they live on.
"""

# Authors: Payam S. Shabestari <payam.sadeghi74@gmail.com>
#
# License: BSD (3-clause)

# %%

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from mne_connectivity import learn_graph

########################################################################################
# Learning a Known Graph from Synthetic Smooth Signals
# =====================================================
#
# To see whether :func:`~mne_connectivity.learn_graph` recovers a graph structure we
# already know, we build a sparse random graph with three communities: nodes are
# connected with probability 0.4 within a community and 0.02 across communities. We
# then generate signals that are smooth on that graph, by drawing Gaussian samples whose
# covariance is the pseudo-inverse of the graph Laplacian, so that strongly connected
# nodes have similar values.

# %%

rng = np.random.default_rng(0)
n_nodes = 30
community = np.repeat([0, 1, 2], n_nodes // 3)
p_edge = np.where(community[:, np.newaxis] == community, 0.4, 0.02)
A_true = np.triu(rng.random((n_nodes, n_nodes)) < p_edge, k=1).astype(float)
A_true += A_true.T

laplacian = np.diag(A_true.sum(axis=1)) - A_true
eigvals, eigvecs = np.linalg.eigh(laplacian)
# square root of the Laplacian pseudo-inverse, discarding the constant (zero) mode
smooth_filter = np.zeros_like(eigvals)
smooth_filter[eigvals > 1e-10] = 1.0 / np.sqrt(eigvals[eigvals > 1e-10])

n_times = 1000
noise = rng.standard_normal((n_nodes, n_times))
X = eigvecs @ (smooth_filter[:, np.newaxis] * (eigvecs.T @ noise))

triu = np.triu_indices(n_nodes, k=1)
true_edges = A_true[triu] > 0


def precision_recall(W):
    """Compare the edges of a learned graph with the ground-truth edges."""
    learned_edges = W[triu] > 0
    n_correct = np.count_nonzero(learned_edges & true_edges)
    precision = n_correct / max(np.count_nonzero(learned_edges), 1)
    recall = n_correct / np.count_nonzero(true_edges)
    return precision, recall


########################################################################################
# Now we learn a graph from ``X`` and compare it to the ground truth, using the default
# parameters (``alpha=0.5, beta=0.5``). Because the model penalizes non-smoothness, an
# edge is only given a non-zero weight if the signals at its two nodes are similar, so
# the learned graph is sparse. We summarize how well its edges match the ground truth
# with the *precision* (the fraction of learned edges that are true edges) and the
# *recall* (the fraction of true edges that were learned). Note that the learned weights
# are only defined up to an overall scale (set by ``alpha``), so we compare the edges
# rather than the weight values.

# %%

conn = learn_graph(X, verbose=False)
W_learned = conn.get_data(output="dense")
precision, recall = precision_recall(W_learned)

fig, axes = plt.subplots(1, 2, figsize=(9, 4))
axes[0].imshow(A_true, cmap="viridis")
axes[0].set_title(f"Ground truth\n({np.count_nonzero(true_edges)} edges)")
im = axes[1].imshow(W_learned, cmap="viridis")
axes[1].set_title(
    f"Learned ({np.count_nonzero(W_learned[triu])} edges)\n"
    f"precision={precision:.2f}, recall={recall:.2f}"
)
fig.colorbar(im, ax=axes[1], shrink=0.8, label="edge weight")
fig.tight_layout()

########################################################################################
# Effect of ``beta`` on Graph Sparsity
# =====================================
#
# ``alpha`` controls how strongly node degrees are pulled away from zero (preventing
# isolated nodes), while ``beta`` controls the density of the learned graph: small
# values of ``beta`` give sparse graphs, and increasing ``beta`` yields progressively
# denser graphs. Here, a sparser graph contains mostly true edges (high precision) but
# misses some of them (low recall), whereas a denser graph finds all true edges at the
# cost of including spurious ones.

# %%

betas = [0.01, 0.1, 0.5, 2.0]
fig, axes = plt.subplots(1, len(betas), figsize=(3 * len(betas), 3.4))
for ax, beta in zip(axes, betas):
    conn_beta = learn_graph(X, beta=beta, max_iter=5000, verbose=False)
    W_beta = conn_beta.get_data(output="dense")
    precision, recall = precision_recall(W_beta)
    ax.imshow(W_beta > 0, cmap="viridis")
    ax.set_title(
        f"beta={beta} ({np.count_nonzero(W_beta[triu])} edges)\n"
        f"precision={precision:.2f}, recall={recall:.2f}",
        fontsize=9,
    )
fig.tight_layout()

########################################################################################
# Learning the Geometry Behind Smooth Signals
# ===========================================
#
# In the example above, the graph was used to generate the signals. A graph can also be
# learned from signals that are smooth for another reason, for instance because they
# vary smoothly in space. Here, adapting the graph-learning tutorial that Vassilis
# Kalofolias wrote for `PyGSP <https://github.com/epfl-lts2/pygsp/pull/57>`__, we place
# nodes at random positions in a 2D square, and define four smooth functions of the
# position. Each node only "observes" the values of these four functions at its
# location; its coordinates are never given to :func:`~mne_connectivity.learn_graph`.

# %%

n_nodes = 300
coords = rng.uniform(size=(n_nodes, 2))
x, y = coords.T
signals = np.column_stack(
    [
        (-np.sin((2 - x - y) ** 2) / 2 + np.cos(y * 3)) / 2,
        np.cos((x + y) ** 2),
        ((x - 0.5) ** 2 + (y - 0.5) ** 3 + x - y) * 3,
        np.sin(3 * ((x - 0.5) ** 2 + (y - 0.5) ** 2)),
    ]
)  # shape (n_nodes, n_signals)

fig, axes = plt.subplots(1, 4, figsize=(12, 3))
for i_signal, ax in enumerate(axes):
    ax.scatter(x, y, c=signals[:, i_signal], s=12, cmap="viridis")
    ax.set_title(f"Smooth signal {i_signal + 1}")
    ax.set_aspect("equal")
    ax.set_axis_off()
fig.tight_layout()

########################################################################################
# We learn one graph from the first signal only, and one from all four signals. For a
# sparser graph, we use a small ``alpha`` rather than a small ``beta``: the sparsity of
# the learned graph is determined by the product ``alpha * beta``, whereas their ratio
# only rescales the weights :footcite:`Kalofolias2016`, and keeping ``beta`` away from
# ``0`` makes the optimization converge much faster.
#
# With a single signal, nodes with similar *values* are connected, even if they lie far
# apart (on the same level line of the function), so the learned graph does not reflect
# the layout of the nodes. With all four signals, only nodes whose values are similar
# for every signal are connected, which in practice means nodes that are close in
# space: the learned graph recovers the geometry of the nodes, with each node connected
# to a handful of its neighbors.

# %%


def plot_graph(ax, W, title):
    """Draw the edges of a learned graph at the node positions."""
    i_node, j_node = np.nonzero(np.triu(W, k=1))
    weights = W[i_node, j_node]
    segments = np.stack([coords[i_node], coords[j_node]], axis=1)
    ax.add_collection(
        LineCollection(segments, linewidths=2 * weights / weights.max(), color="k")
    )
    ax.scatter(x, y, s=6, color="tab:red", zorder=2)
    ax.set_title(f"{title}\n(average degree {2 * len(weights) / n_nodes:.1f})")
    ax.set_aspect("equal")
    ax.set_axis_off()


fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
for ax, n_signals in zip(axes, (1, 4)):
    conn_geo = learn_graph(signals[:, :n_signals], alpha=0.001, verbose=False)
    plot_graph(
        ax,
        conn_geo.get_data(output="dense"),
        f"Learned from {n_signals} signal{'s' if n_signals > 1 else ''}",
    )
fig.tight_layout()

plt.show()

########################################################################################
# References
# ----------
# .. footbibliography::
