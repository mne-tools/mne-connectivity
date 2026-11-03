# Authors: Payam S. Shabestari <payam.sadeghi74@gmail.com>
#
# License: BSD (3-clause)

import itertools

import numpy as np
from mne import BaseEpochs
from mne._fiff.pick import _picks_to_idx
from mne.parallel import parallel_func
from mne.source_estimate import _BaseSourceEstimate
from mne.utils import _check_option, _validate_type, logger, verbose, warn
from scipy.spatial.distance import pdist

from .base import Connectivity, EpochConnectivity
from .utils import fill_doc


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
    :footcite:`Kalofolias2016`.
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

    # return the projected primal point rather than the iterate ``w``: they
    # coincide at the fixed point, but only ``p`` has exact zeros (sparsity)
    return p, n_iter, converged


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


def _check_epoch_data(epoch_data, ei, n_nodes):
    """Check a single epoch of data and return it as a 2D array."""
    if isinstance(epoch_data, _BaseSourceEstimate):
        epoch_data = epoch_data.data
    epoch_data = np.asarray(epoch_data)
    if epoch_data.ndim != 2:
        raise ValueError(f"Each entry in data must be 2D, got shape {epoch_data.shape}")
    if np.iscomplexobj(epoch_data):
        raise ValueError(
            "learn_graph does not support complex-valued data, got "
            f"dtype {epoch_data.dtype}"
        )
    if n_nodes is not None and epoch_data.shape[0] != n_nodes:
        raise ValueError(
            f"n_nodes mismatch between data[0] and data[{ei}], got "
            f"{n_nodes} and {epoch_data.shape[0]}"
        )
    return epoch_data


def _iter_epochs(first_epoch, data_iter, n_nodes, picks):
    """Yield checked epochs one at a time, starting with the already-checked first."""
    epoch_data = first_epoch
    del first_epoch
    for ei in itertools.count(1):
        if picks is not None:
            epoch_data = epoch_data[picks]
        yield epoch_data
        epoch_data = next(data_iter, None)
        if epoch_data is None:
            return
        epoch_data = _check_epoch_data(epoch_data, ei, n_nodes)


@verbose
@fill_doc
def learn_graph(
    data,
    *,
    names=None,
    indices="lower",
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

    where :math:`z` is the vector of pairwise distances between nodes (rows of
    ``data``) and :math:`S` maps an edge-weight vector to the corresponding
    vector of node degrees. This is the model proposed in
    :footcite:`Kalofolias2016`, and is the same model used on real MEG data
    (with :math:`\alpha`, :math:`\beta` named :math:`\sigma`, :math:`\rho`) in
    :footcite:`ShabestariEtAl2026`, where it was found to be more robust than
    Phase Lag Index for short recordings and to better capture
    individual-specific connectivity. The default ``alpha=0.5, beta=0.5`` is
    one of the parameter combinations swept in the robustness analysis of
    :footcite:`ShabestariEtAl2026`. See Notes for details.

    Parameters
    ----------
    data : array_like, shape ([n_epochs,] n_nodes, n_times) | ~mne.Epochs | generator
        The data from which to learn a graph. Can be a single 2D array of
        shape ``(n_nodes, n_times)``, a 3D array of shape
        ``(n_epochs, n_nodes, n_times)``, a :class:`mne.Epochs` object, or a
        list/generator of 2D arrays or :class:`mne.SourceEstimate` /
        :class:`mne.VolSourceEstimate` objects. For :class:`mne.Epochs`, channels
        marked as bad are not used to learn the graph; they are kept as nodes, with
        their connections set to ``NaN``.
    %(names)s
        Ignored (and overridden by ``data.ch_names``) if ``data`` is an
        :class:`mne.Epochs` object.
    indices : ``'lower'`` | ``'upper'`` | ``'all'``
        Which part of the symmetric graph to store. If ``'lower'`` or ``'upper'``,
        only the lower- or upper-triangular part is stored, and the rest is filled in
        when requesting ``get_data(output='dense')``. If ``'all'``, the full matrix
        (including the zero diagonal) is stored. The graph itself is always learned
        over all pairs of nodes. Default is ``'lower'``.
    alpha : float
        Controls how strongly node degrees are pulled away from zero via the
        log-barrier term. Must be strictly positive. Larger values yield
        more connected (higher-degree) graphs; every node is guaranteed at
        least one edge for any ``alpha > 0``. Default ``0.5``.
    beta : float
        Controls the :math:`\|w\|^2` penalty on edge weights. Must be
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
        Euclidean distance), matching :footcite:`Kalofolias2016` and
        :footcite:`ShabestariEtAl2026`.
    normalize : bool
        If ``True`` (default), rescale the pairwise-distance vector by its
        mean before solving, so that the default ``alpha``/``beta`` behave
        consistently regardless of the input's physical units or amplitude
        scale. See Notes.
    max_iter : int
        Maximum number of primal-dual iterations. Default ``1000``. Sparser
        settings (small ``beta``, in particular ``beta=0``) may require more
        iterations to converge. See Notes.
    tol : float
        Relative convergence tolerance on both the edge-weight and
        node-degree updates. Default ``1e-4``.
    step_size : float | None
        Override the closed-form step size
        ``1 / (1 + 2 * beta + sqrt(2 * (n_nodes - 1)))``. Must be strictly
        positive. Useful for matching
        the step size used by another implementation (e.g. for validation)
        or for trading off convergence speed against stability. Default
        ``None``.
    %(n_jobs)s
        Each epoch's graph is learned independently, so this parallelizes
        across epochs; for small numbers of nodes/iterations the joblib
        dispatch overhead may outweigh the benefit.
    %(verbose)s

    Returns
    -------
    graph : instance of Connectivity or EpochConnectivity
        The learned graph(s): a symmetric, zero-diagonal, non-negative
        weighted adjacency, stored as the raveled values of the part of the
        matrix selected by ``indices``. A single
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
    This implements the model of eq. 12 in :footcite:`Kalofolias2016` (their
    :math:`\alpha`, :math:`\beta`), identical to the model used in
    :footcite:`ShabestariEtAl2026` (that paper's :math:`\sigma`, :math:`\rho`). The
    log-barrier term on node degrees (rather than on individual edge
    weights) is what distinguishes this model from a plain Gaussian-kernel
    graph construction: it guarantees every node has at least one edge,
    without preventing individual edges from being exactly zero, so sparse
    *and* fully connected graphs can be learned simultaneously.

    Because :math:`z` (the pairwise-distance vector) scales with the physical
    units/amplitude of ``data``, and ``alpha``/``beta`` interact
    multiplicatively with that scale, ``normalize=True`` divides :math:`z` by its
    mean before solving so that the default ``alpha=0.5, beta=0.5`` are
    reasonable starting points regardless of the input's scale. Use
    ``normalize=False`` for a literal, unscaled port of the optimization
    problem as stated above.

    When ``beta=0``, the learned graph is the sparsest one consistent with
    keeping every node connected, and changing ``alpha`` alone only rescales
    the overall magnitude of the learned weights, not the sparsity pattern
    (see Proposition 2 of :footcite:`Kalofolias2016`). However, ``beta=0``
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
    _validate_type(indices, str, "indices")
    _check_option("indices", indices, ["lower", "upper", "all"])
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
    _validate_type(step_size, ("numeric", None), "step_size")
    if step_size is not None and step_size <= 0:
        raise ValueError(f"step_size must be > 0, got {step_size}")

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

    # the first epoch determines the number of nodes; the others are validated and
    # learned one at a time, so that a generator is never loaded into memory at once
    data_iter = iter(data_iter)
    first_epoch = next(data_iter, None)
    if first_epoch is None:
        raise RuntimeError("Passing in empty epochs/data is not allowed.")
    first_epoch = _check_epoch_data(first_epoch, 0, None)
    n_nodes = first_epoch.shape[0]
    # only good channels are learned; bad channels are kept as nodes (see below)
    learned_nodes = np.arange(n_nodes) if picks is None else picks
    n_learned = len(learned_nodes)
    if n_learned < 2:
        raise ValueError(f"learn_graph requires at least 2 nodes, got {n_learned}")
    epochs_iter = _iter_epochs(first_epoch, data_iter, n_nodes, picks)
    del first_epoch

    logger.info(f"Learning graph(s) with {n_learned} nodes each")

    iu, ju = np.triu_indices(n_learned, k=1)

    parallel, my_learn_one, _ = parallel_func(
        _learn_graph_one_epoch, n_jobs, verbose=verbose
    )
    results = parallel(
        my_learn_one(
            epoch_data,
            iu,
            ju,
            n_learned,
            metric,
            normalize,
            alpha,
            beta,
            max_iter,
            tol,
            step_size,
        )
        for epoch_data in epochs_iter
    )

    # edge weights are ordered as (iu, ju) over the learned nodes; fill the symmetric
    # matrix over all nodes (connections of bad channels stay NaN), then keep the part
    # selected by indices
    con = np.full((len(results), n_nodes, n_nodes), np.nan)
    rows, cols = learned_nodes[iu], learned_nodes[ju]
    con[:, rows, cols] = [w for w, _, _ in results]
    con[:, cols, rows] = con[:, rows, cols]
    con[:, learned_nodes, learned_nodes] = 0.0
    if indices == "all":
        con = con.reshape(len(results), -1)
    else:
        if indices == "lower":
            tri = np.tril_indices(n_nodes, k=-1)
        else:  # "upper"
            tri = np.triu_indices(n_nodes, k=1)
        con = con[:, tri[0], tri[1]]

    logger.info(f"[learn_graph Done] Learned {len(results)} graph(s)")

    con_kwargs = dict(
        names=names,
        method="learn_graph",
        indices=indices,
        n_nodes=n_nodes,
    )
    if single_epoch:
        return Connectivity(data=con[0], **con_kwargs)
    return EpochConnectivity(
        data=con,
        events=events,
        event_id=event_id,
        metadata=metadata,
        **con_kwargs,
    )
