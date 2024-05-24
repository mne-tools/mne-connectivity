# Authors: Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#          Thomas S. Binns <t.s.binns@outlook.com>
#
# License: BSD (3-clause)
import numpy as np
from mne.utils import _prepare_write_metadata, logger


def parallel_loop(func, n_jobs=1, verbose=1):
    """Run loops in parallel, if joblib is available.

    Parameters
    ----------
    func : function
        function to be executed in parallel
    n_jobs : int | None
        Number of jobs. If set to None, do not attempt to use joblib.
    verbose : int
        verbosity level

    Notes
    -----
    Execution of the main script must be guarded with
    `if __name__ == '__main__':` when using parallelization.
    """
    if n_jobs:
        try:
            from joblib import Parallel, delayed
        except ImportError:
            try:
                from sklearn.externals.joblib import Parallel, delayed
            except ImportError:
                n_jobs = None

    if not n_jobs:
        if verbose:
            logger.info("running ", func, " serially")

        def par(x):
            return list(x)

    else:
        if verbose:
            logger.info("running ", func, " in parallel")
        func = delayed(func)
        par = Parallel(n_jobs=n_jobs, verbose=verbose)

    return par, func


def check_indices(indices):
    """Check indices parameter for bivariate connectivity.

    Parameters
    ----------
    indices : tuple of array of int, shape (2, n_cons)
        Tuple containing index pairs.

    Returns
    -------
    indices : tuple of array of int, shape (2, n_cons)
        The indices.

    Notes
    -----
    Indices for bivariate connectivity should be a tuple of length 2,
    containing the channel indices for the seed and target channel pairs,
    respectively. Seed and target indices should be equal-length array-likes of
    integers representing the indices of the individual channels in the data.
    """
    if not isinstance(indices, tuple) or len(indices) != 2:
        raise ValueError("indices must be a tuple of length 2")

    if len(indices[0]) != len(indices[1]):
        raise ValueError(
            "Index arrays indices[0] and indices[1] must " "have the same length"
        )

    if any(
        isinstance(inds, (np.ndarray, list, tuple))
        for inds in [*indices[0], *indices[1]]
    ):
        raise TypeError("Channel indices must be integers, not array-likes")

    return indices


def _check_multivariate_indices(indices, n_chans):
    """Check indices parameter for multivariate connectivity and mask it.

    Parameters
    ----------
    indices : tuple of array of array of int, shape (2, n_cons, variable)
        Tuple containing index sets.

    n_chans : int
        The number of channels in the data. Used when converting negative
        indices to positive indices.

    Returns
    -------
    indices : array of array of int, shape of (2, n_cons, max_n_chans)
        The padded indices as a masked array.

    Notes
    -----
    Indices for multivariate connectivity should be a tuple of length 2
    containing the channel indices for the seed and target channel sets,
    respectively. Seed and target indices should be equal-length array-likes
    representing the indices of the channel sets in the data for each
    connection. The indices for each connection should be an array-like of
    integers representing the individual channels in the data. The length of
    indices for each connection do not need to be equal. All indices within a
    connection must be unique.

    If the seed and target indices are given as lists or tuples, they will be
    converted to numpy arrays. Because the number of channels can differ across
    connections or between the seeds and targets for a given connection (i.e.
    ragged/jagged indices), the returned array will be padded out to a 'full'
    array with an invalid index (``-1``) according to the maximum number of
    channels in the seed or target of any one connection. These invalid
    entries are then masked and returned as numpy masked arrays. E.g. the
    ragged indices of shape ``(2, n_cons, variable)``::

            indices = ([[0, 1], [0, 1   ]],  # seeds
                       [[2, 3], [4, 5, 6]])  # targets

    would be padded to full arrays::

            indices = ([[0, 1, -1], [0, 1, -1]],  # seeds
                       [[2, 3, -1], [4, 5,  6]])  # targets

    to have shape ``(2, n_cons, max_n_chans)``, where ``max_n_chans = 3``. The
    invalid entries are then masked::

            indices = ([[0, 1, --], [0, 1, --]],  # seeds
                       [[2, 3, --], [4, 5,  6]])  # targets

    In case "indices" contains negative values to index channels, these will be
    converted to the corresponding positive-valued index before any masking is
    applied.

    More information on working with multivariate indices and handling
    connections where the number of seeds and targets are not equal can be
    found in the :doc:`../auto_examples/handling_ragged_arrays` example.
    """
    if not isinstance(indices, tuple) or len(indices) != 2:
        raise ValueError("indices must be a tuple of length 2")

    if len(indices[0]) != len(indices[1]):
        raise ValueError(
            "index arrays indices[0] and indices[1] must " "have the same length"
        )

    n_cons = len(indices[0])
    invalid = -1

    max_n_chans = 0
    for group_idx, group in enumerate(indices):
        for con_idx, con in enumerate(group):
            if not isinstance(con, (np.ndarray, list, tuple)):
                raise TypeError(
                    "multivariate indices must contain array-likes of channel "
                    "indices for each seed and target"
                )
            con = np.array(con)
            if len(con) != len(np.unique(con)):
                raise ValueError(
                    "multivariate indices cannot contain repeated channels "
                    "within a seed or target"
                )
            max_n_chans = max(max_n_chans, len(con))
            # convert negative to positive indices
            for chan_idx, chan in enumerate(con):
                if chan < 0:
                    if chan * -1 >= n_chans:
                        raise ValueError(
                            "a negative channel index is not present in the " "data"
                        )
                    indices[group_idx][con_idx][chan_idx] = chan % n_chans

    # pad indices to avoid ragged arrays
    padded_indices = np.full((2, n_cons, max_n_chans), invalid, dtype=np.int32)
    for con_i, (seed, target) in enumerate(zip(indices[0], indices[1])):
        padded_indices[0, con_i, : len(seed)] = seed
        padded_indices[1, con_i, : len(target)] = target

    # mask invalid indices
    masked_indices = np.ma.masked_values(padded_indices, invalid)

    return masked_indices


def seed_target_indices(seeds, targets):
    """Generate indices parameter for bivariate seed-based connectivity.

    Parameters
    ----------
    seeds : array of int | int, shape (n_unique_seeds)
        Seed indices.
    targets : array of int | int, shape (n_unique_targets)
        Indices of signals for which to compute connectivity.

    Returns
    -------
    indices : tuple of array of int, shape (2, n_cons)
        The indices parameter used for connectivity computation.

    Notes
    -----
    ``seeds`` and ``targets`` should be array-likes or integers representing
    the indices of the channel pairs in the data for each connection. ``seeds``
    and ``targets`` will be expanded such that connectivity will be computed
    between each seed and each target. E.g. the seeds and targets::

            seeds   = [0, 1]
            targets = [2, 3, 4]

    would be returned as::

            indices = (np.array([0, 0, 0, 1, 1, 1]),  # seeds
                       np.array([2, 3, 4, 2, 3, 4]))  # targets

    where the indices have been expanded to have shape ``(2, n_cons)``, where
    ``n_cons = n_unique_seeds * n_unique_targets``.
    """
    # make them arrays
    seeds = np.asarray((seeds,)).ravel()
    targets = np.asarray((targets,)).ravel()

    n_seeds = len(seeds)
    n_targets = len(targets)

    indices = (
        np.concatenate([np.tile(i, n_targets) for i in seeds]),
        np.tile(targets, n_seeds),
    )

    return indices


def seed_target_multivariate_indices(seeds, targets):
    """Generate indices parameter for multivariate seed-based connectivity.

    Parameters
    ----------
    seeds : array of array of int, shape (n_unique_seeds, variable)
        Seed indices.

    targets : array of array of int, shape (n_unique_targets, variable)
        Target indices.

    Returns
    -------
    indices : tuple of array of array of int, shape (2, n_cons, variable)
        The indices as a numpy object array.

    Notes
    -----
    ``seeds`` and ``targets`` should be array-likes representing the indices of
    the channel sets in the data for each connection. The indices for each
    connection should be an array-like of integers representing the individual
    channels in the data. The length of indices for each connection do not need
    to be equal. Furthermore, all indices within a connection must be unique.

    Because the number of channels per connection can vary, the indices are
    stored as numpy arrays with ``dtype=object``. E.g. ``seeds`` and
    ``targets``::

            seeds   = [[0]]
            targets = [[1, 2], [3, 4, 5]]

    would be returned as::

            indices = (np.array([[0   ], [0      ]], dtype=object),  # seeds
                       np.array([[1, 2], [3, 4, 5]], dtype=object))  # targets

    Even if the number of channels does not vary, the indices will still be
    stored as object arrays for compatibility.

    More information on working with multivariate indices and handling
    connections where the number of seeds and targets are not equal can be
    found in the :doc:`../auto_examples/handling_ragged_arrays` example.
    """
    array_like = (np.ndarray, list, tuple)

    if not isinstance(seeds, array_like) or not isinstance(targets, array_like):
        raise TypeError("`seeds` and `targets` must be array-like")

    for inds in [*seeds, *targets]:
        if not isinstance(inds, array_like):
            raise TypeError("`seeds` and `targets` must contain nested array-likes")
        if len(inds) != len(np.unique(inds)):
            raise ValueError("`seeds` and `targets` cannot contain repeated channels")

    indices = [[], []]
    for seed in seeds:
        for target in targets:
            indices[0].append(np.array(seed))
            indices[1].append(np.array(target))

    indices = (np.array(indices[0], dtype=object), np.array(indices[1], dtype=object))

    return indices


def degree(connectivity, threshold_prop=0.2):
    """Compute the undirected degree of a connectivity matrix.

    Parameters
    ----------
    connectivity : ndarray, shape (n_nodes, n_nodes)
        The connectivity matrix.
    threshold_prop : float
        The proportion of edges to keep in the graph before
        computing the degree. The value should be between 0
        and 1.

    Returns
    -------
    degree : ndarray, shape (n_nodes,)
        The computed degree.

    Notes
    -----
    During thresholding, the symmetry of the connectivity matrix is
    auto-detected based on :func:`numpy.allclose` of it with its transpose.
    """
    from mne_connectivity.base import BaseConnectivity

    if isinstance(connectivity, BaseConnectivity):
        connectivity = connectivity.get_data(output="dense").squeeze()

    connectivity = np.array(connectivity)
    if connectivity.ndim != 2 or connectivity.shape[0] != connectivity.shape[1]:
        raise ValueError(
            "connectivity must be have shape (n_nodes, n_nodes), "
            "got %s" % (connectivity.shape,)
        )
    n_nodes = len(connectivity)
    if np.allclose(connectivity, connectivity.T):
        split = 2.0
        connectivity[np.tril_indices(n_nodes)] = 0
    else:
        split = 1.0
    threshold_prop = float(threshold_prop)
    if not 0 < threshold_prop <= 1:
        raise ValueError(
            "threshold must be 0 <= threshold < 1, got %s" % (threshold_prop,)
        )
    degree = connectivity.ravel()  # no need to copy because np.array does
    degree[:: n_nodes + 1] = 0.0
    n_keep = int(round((degree.size - len(connectivity)) * threshold_prop / split))
    degree[np.argsort(degree)[:-n_keep]] = 0
    degree.shape = connectivity.shape
    if split == 2:
        degree += degree.T  # normally unsafe, but we know where our zeros are
    degree = np.sum(degree > 0, axis=0)
    return degree


def _prepare_xarray_mne_data_structures(conn_obj):
    """Prepare an xarray connectivity object with extra MNE data structures.

    For MNE, these are:
    - metadata -> stored as a string representation
    - event_id -> stored as two lists
    """
    # get a copy of metadata into attrs as a dictionary
    conn_obj.attrs["metadata"] = _prepare_write_metadata(conn_obj.metadata)

    # write event IDs since they are stored as a list instead
    if conn_obj.event_id is not None:
        conn_obj.attrs["event_id_keys"] = list(conn_obj.event_id.keys())
        conn_obj.attrs["event_id_vals"] = list(conn_obj.event_id.values())

    return conn_obj
