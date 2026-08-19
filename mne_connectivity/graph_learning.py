# Authors: Payam S. Shabestari <payam.sadeghi74@gmail.com>
#
# License: BSD (3-clause)

import numpy as np
from mne import BaseEpochs
from mne._fiff.pick import _picks_to_idx
from mne.source_estimate import _BaseSourceEstimate
from mne.utils import _validate_type, logger, verbose, warn
from scipy.spatial.distance import pdist

from .base import Connectivity, EpochConnectivity
from .utils import fill_doc, parallel_loop


def _op_S(w, iu, ju, n_nodes):
    """Map an edge-weight vector to the node-degree vector (``S @ w``)."""
    return np.bincount(iu, weights=w, minlength=n_nodes) + np.bincount(
        ju, weights=w, minlength=n_nodes
    )


def _op_St(d, iu, ju):
    """Adjoint of :func:`_op_S`: node-degree vector -> edge-weight vector."""
    return d[iu] + d[ju]


def _solve_primal_dual(z, n_nodes, iu, ju, alpha, beta, max_iter, tol, step_size):
    """Solve the Kalofolias (2016) graph-learning model (their eq. 12).

    Forward-backward-forward primal-dual splitting (Combettes & Pesquet 2011;
    Komodakis & Pesquet 2014), following "Algorithm 1" of
    :footcite:`kalofolias2016learn`.
    """
    m = z.size
    w = np.zeros(m)
    d = np.zeros(n_nodes)

    if step_size is None:
        s_norm = np.sqrt(2 * (n_nodes - 1))  # exact ||S||_2 for this operator
        gamma = 1.0 / (1.0 + 2 * beta + s_norm)
    else:
        gamma = step_size

    converged = False
    n_iter = max_iter
    for it in range(max_iter):
        Sw = _op_S(w, iu, ju, n_nodes)
        St_d = _op_St(d, iu, ju)
        y = w - gamma * (2 * beta * w + St_d)
        y_bar = d + gamma * Sw

        p = np.maximum(0.0, y - 2 * gamma * z)
        p_bar = (y_bar - np.sqrt(y_bar**2 + 4 * alpha * gamma)) / 2.0

        Sp = _op_S(p, iu, ju, n_nodes)
        St_p_bar = _op_St(p_bar, iu, ju)
        q = p - gamma * (2 * beta * p + St_p_bar)
        q_bar = p_bar + gamma * Sp
        w_new = w - y + q
        d_new = d - y_bar + q_bar

        w_norm = np.linalg.norm(w)
        d_norm = np.linalg.norm(d)
        rel_w = np.linalg.norm(w_new - w) / w_norm if w_norm > 0 else np.inf
        rel_d = np.linalg.norm(d_new - d) / d_norm if d_norm > 0 else np.inf

        w, d = w_new, d_new
        if rel_w < tol and rel_d < tol:
            converged = True
            n_iter = it + 1
            break

    return w, n_iter, converged


def _learn_graph_one_epoch(
    epoch_data,
    iu,
    ju,
    n_nodes,
    metric,
    normalize,
    alpha,
    beta,
    max_iter,
    tol,
    step_size,
):
    """Learn one graph from a single ``(n_nodes, n_times)`` array."""
    z = pdist(epoch_data, metric=metric)
    if np.any(z < 0):
        raise ValueError(
            f"Pairwise distances computed with metric={metric!r} must be "
            f"non-negative for the graph-learning model to be valid, got a "
            f"negative value (min={z.min()})"
        )
    if normalize:
        z_mean = z.mean()
        if z_mean > 0:
            z = z / z_mean
    w, n_iter, converged = _solve_primal_dual(
        z, n_nodes, iu, ju, alpha, beta, max_iter, tol, step_size
    )
    if not converged:
        warn(
            f"learn_graph did not converge after {max_iter} iterations "
            f"(tol={tol}); consider increasing max_iter or tol"
        )
    return w, n_iter, converged


@verbose
@fill_doc
def learn_graph(
    data,
    names=None,
    *,
    alpha=0.5,
    beta=0.5,
    metric="sqeuclidean",
    normalize=True,
    max_iter=1000,
    tol=1e-4,
    step_size=None,
    n_jobs=1,
    verbose=None,
):
    r"""Learn a sparse graph from smooth signals.

    Estimates a symmetric, non-negative, zero-diagonal weighted adjacency
    matrix directly from signal smoothness, rather than from pairwise
    similarity/coherence computed independently per edge. All edge weights
    are learned jointly by solving

    .. math::

        \min_{w \geq 0} \; 2 w^\top z - \alpha \mathbf{1}^\top \log(Sw)
        + \beta \|w\|^2

    where ``z`` is the vector of pairwise distances between nodes (rows of
    ``data``) and ``S`` maps an edge-weight vector to the corresponding
    vector of node degrees. This is the model proposed in
    :footcite:`kalofolias2016learn`, and is the same model used on real MEG data
    (with parameters named ``sigma``, ``rho``) in
    :footcite:`shabestari2026shared`, where it was found to be more robust than
    Phase Lag Index for short recordings and to better capture
    individual-specific connectivity. The default ``alpha=0.5, beta=0.5`` is
    one of the parameter combinations swept in the robustness analysis of
    :footcite:`shabestari2026shared`. See Notes for details.

    Parameters
    ----------
    data : array_like, shape ([n_epochs,] n_nodes, n_times) | ~mne.Epochs | generator
        The data from which to learn a graph. Can be a single 2D array of
        shape ``(n_nodes, n_times)``, a 3D array of shape
        ``(n_epochs, n_nodes, n_times)``, a :class:`mne.Epochs` object, or a
        list/generator of 2D arrays or :class:`mne.SourceEstimate` /
        :class:`mne.VolSourceEstimate` objects. See Returns for how the
        output depends on the input type.
    %(names)s
        Ignored (and overridden by ``data.ch_names``) if ``data`` is an
        :class:`mne.Epochs` object.
    alpha : float
        Controls how strongly node degrees are pulled away from zero via the
        log-barrier term. Must be strictly positive. Larger values yield
        more connected (higher-degree) graphs; every node is guaranteed at
        least one edge for any ``alpha > 0``. Default ``0.5``.
    beta : float
        Controls the ``||w||^2`` penalty on edge weights. Must be
        non-negative. Larger values yield denser graphs. ``beta=0`` yields
        the sparsest graph consistent with keeping every node connected, but
        lacks a strongly-convex term and so converges markedly more slowly
        (see ``max_iter``); the default ``beta=0.5`` gives fast, reliable
        convergence while still yielding a moderately sparse graph. See
        Notes.
    metric : str | callable
        Distance metric passed to :func:`scipy.spatial.distance.pdist` to
        build the pairwise dissimilarity vector between nodes. Must yield
        only non-negative values. Default ``'sqeuclidean'`` (squared
        Euclidean distance), matching :footcite:`kalofolias2016learn` and
        :footcite:`shabestari2026shared`.
    normalize : bool
        If ``True`` (default), rescale the pairwise-distance vector by its
        mean before solving, so that the default ``alpha``/``beta`` behave
        consistently regardless of the input's physical units or amplitude
        scale. See Notes.
    max_iter : int
        Maximum number of primal-dual iterations. Default ``1000``. Sparser
        settings (small ``beta``, in particular ``beta=0``) may require more
        iterations to converge; see Notes.
    tol : float
        Relative convergence tolerance on both the edge-weight and
        node-degree updates. Default ``1e-4``.
    step_size : float | None
        Override the closed-form step size
        ``1 / (1 + 2 * beta + sqrt(2 * (n_nodes - 1)))``. Useful for matching
        the step size used by another implementation (e.g. for validation)
        or for trading off convergence speed against stability; convergence
        is only guaranteed for the default. Default ``None``.
    %(n_jobs)s
        Each epoch's graph is learned independently, so this parallelizes
        across epochs; for small numbers of nodes/iterations the joblib
        dispatch overhead may outweigh the benefit.
    %(verbose)s

    Returns
    -------
    graph : instance of Connectivity or EpochConnectivity
        The learned graph(s): a symmetric, zero-diagonal, non-negative
        weighted adjacency, stored as the upper-triangular-including-diagonal
        raveled values (diagonal entries are always exactly ``0``). A single
        :class:`~mne_connectivity.Connectivity` is returned if ``data`` was a
        bare 2D array; otherwise an
        :class:`~mne_connectivity.EpochConnectivity` is returned with one
        graph per epoch.

    See Also
    --------
    mne_connectivity.Connectivity
    mne_connectivity.EpochConnectivity

    Notes
    -----
    This implements the model of eq. 12 in :footcite:`kalofolias2016learn` (their
    ``alpha``, ``beta``), identical to the model used in
    :footcite:`shabestari2026shared` (that paper's ``sigma``, ``rho``). The
    log-barrier term on node degrees (rather than on individual edge
    weights) is what distinguishes this model from a plain Gaussian-kernel
    graph construction: it guarantees every node has at least one edge,
    without preventing individual edges from being exactly zero, so sparse
    *and* fully connected graphs can be learned simultaneously.

    Because ``z`` (the pairwise-distance vector) scales with the physical
    units/amplitude of ``data``, and ``alpha``/``beta`` interact
    multiplicatively with that scale, ``normalize=True`` divides ``z`` by its
    mean before solving so that the default ``alpha=0.5, beta=0.5`` are
    reasonable starting points regardless of the input's scale. Use
    ``normalize=False`` for a literal, unscaled port of the optimization
    problem as stated above.

    When ``beta=0``, the learned graph is the sparsest one consistent with
    keeping every node connected, and changing ``alpha`` alone only rescales
    the overall magnitude of the learned weights, not the sparsity pattern
    (see Proposition 2 of :footcite:`kalofolias2016learn`). However, ``beta=0``
    removes the only strongly-convex term in the objective, so this primal-dual
    solver converges much more slowly in that regime (often thousands of
    iterations) and the relative-change stopping criterion (``tol``) is a
    weaker guarantee of closeness to the true optimum than it is for
    ``beta > 0``. For most uses, prefer sweeping ``beta`` away from ``0`` (as
    the default does) for both faster and more reliable convergence.

    The optimization is solved per epoch using a forward-backward-forward
    primal-dual splitting with per-iteration complexity ``O(n_nodes**2)``.

    References
    ----------
    .. footbibliography::
    """
    _validate_type(alpha, "numeric", "alpha")
    _validate_type(beta, "numeric", "beta")
    if alpha <= 0:
        raise ValueError(f"alpha must be > 0, got {alpha}")
    if beta < 0:
        raise ValueError(f"beta must be >= 0, got {beta}")
    if max_iter < 1:
        raise ValueError(f"max_iter must be >= 1, got {max_iter}")
    if tol <= 0:
        raise ValueError(f"tol must be > 0, got {tol}")

    events = None
    event_id = None
    metadata = None
    picks = None

    single_epoch = isinstance(data, np.ndarray) and data.ndim == 2
    if single_epoch:
        data_iter = [data]
    elif isinstance(data, BaseEpochs):
        picks = _picks_to_idx(data.info, picks="all", exclude="bads")
        names = data.ch_names
        events = data.events
        event_id = data.event_id
        metadata = data.metadata
        data_iter = data.get_data(copy=False)
    else:
        data_iter = data

    n_nodes = None
    epochs_data = []
    for ei, epoch_data in enumerate(data_iter):
        if isinstance(epoch_data, _BaseSourceEstimate):
            epoch_data = epoch_data.data
        epoch_data = np.asarray(epoch_data)
        if epoch_data.ndim != 2:
            raise ValueError(
                f"Each entry in data must be 2D, got shape {epoch_data.shape}"
            )
        if np.iscomplexobj(epoch_data):
            raise ValueError(
                "learn_graph does not support complex-valued data, got "
                f"dtype {epoch_data.dtype}"
            )
        this_n_nodes, _ = epoch_data.shape
        if ei > 0 and this_n_nodes != n_nodes:
            raise ValueError(
                f"n_nodes mismatch between data[0] and data[{ei}], got "
                f"{n_nodes} and {this_n_nodes}"
            )
        n_nodes = this_n_nodes
        if picks is not None:
            epoch_data = epoch_data[picks]
        epochs_data.append(epoch_data)

    if n_nodes is None:
        raise RuntimeError("Passing in empty epochs/data is not allowed.")
    if n_nodes < 2:
        raise ValueError(f"learn_graph requires at least 2 nodes, got {n_nodes}")

    logger.info(
        f"Learning graph(s) for {len(epochs_data)} epoch(s), {n_nodes} nodes each"
    )

    iu, ju = np.triu_indices(n_nodes, k=1)

    parallel, my_learn_one = parallel_loop(_learn_graph_one_epoch, n_jobs=n_jobs)
    results = parallel(
        my_learn_one(
            epoch_data,
            iu,
            ju,
            n_nodes,
            metric,
            normalize,
            alpha,
            beta,
            max_iter,
            tol,
            step_size,
        )
        for epoch_data in epochs_data
    )

    n_epochs = len(results)
    triu_full = np.triu_indices(n_nodes, k=0)
    raveled_triu = np.ravel_multi_index(triu_full, dims=(n_nodes, n_nodes))

    con = np.zeros((n_epochs, n_nodes, n_nodes))
    for ei, (w, _, _) in enumerate(results):
        con[ei, iu, ju] = w
        con[ei, ju, iu] = w
    con = con.reshape(n_epochs, -1)[:, raveled_triu]

    logger.info("[learn_graph Done]")

    con_kwargs = dict(
        names=names,
        method="graph learning (Kalofolias)",
        indices="symmetric",
        n_nodes=n_nodes,
    )
    if single_epoch:
        return Connectivity(data=con[0], n_epochs_used=1, **con_kwargs)
    return EpochConnectivity(
        data=con,
        n_epochs_used=n_epochs,
        events=events,
        event_id=event_id,
        metadata=metadata,
        **con_kwargs,
    )
