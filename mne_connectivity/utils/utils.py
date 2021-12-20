# Authors: Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)
import numpy as np

from mne.utils import logger, _prepare_write_metadata


def parallel_loop(func, n_jobs=1, verbose=1):
    """run loops in parallel, if joblib is available.

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
            logger.info('running ', func, ' serially')

        def par(x):

            return list(x)
    else:
        if verbose:
            logger.info('running ', func, ' in parallel')
        func = delayed(func)
        par = Parallel(n_jobs=n_jobs, verbose=verbose)

    return par, func


def check_indices(indices):
    """Check indices parameter.

    Parameters
    ----------
    indices : tuple of array
        Tuple of length 2 containing index pairs.

    Returns
    -------
    indices : tuple of array
        The indices.
    """
    if not isinstance(indices, tuple) or len(indices) != 2:
        raise ValueError('indices must be a tuple of length 2')

    if len(indices[0]) != len(indices[1]):
        raise ValueError('Index arrays indices[0] and indices[1] must '
                         'have the same length')

    return indices


def seed_target_indices(seeds, targets):
    """Generate indices parameter for seed based connectivity analysis.

    Parameters
    ----------
    seeds : array of int | int
        Seed indices.
    targets : array of int | int
        Indices of signals for which to compute connectivity.

    Returns
    -------
    indices : tuple of array
        The indices parameter used for connectivity computation.
    """
    # make them arrays
    seeds = np.asarray((seeds,)).ravel()
    targets = np.asarray((targets,)).ravel()

    n_seeds = len(seeds)
    n_targets = len(targets)

    indices = (np.concatenate([np.tile(i, n_targets) for i in seeds]),
               np.tile(targets, n_seeds))

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
        connectivity = connectivity.get_data(output='dense').squeeze()

    connectivity = np.array(connectivity)
    if connectivity.ndim != 2 or \
            connectivity.shape[0] != connectivity.shape[1]:
        raise ValueError('connectivity must be have shape (n_nodes, n_nodes), '
                         'got %s' % (connectivity.shape,))
    n_nodes = len(connectivity)
    if np.allclose(connectivity, connectivity.T):
        split = 2.
        connectivity[np.tril_indices(n_nodes)] = 0
    else:
        split = 1.
    threshold_prop = float(threshold_prop)
    if not 0 < threshold_prop <= 1:
        raise ValueError('threshold must be 0 <= threshold < 1, got %s'
                         % (threshold_prop,))
    degree = connectivity.ravel()  # no need to copy because np.array does
    degree[::n_nodes + 1] = 0.
    n_keep = int(round((degree.size - len(connectivity)) *
                       threshold_prop / split))
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
    conn_obj.attrs['metadata'] = _prepare_write_metadata(conn_obj.metadata)

    # write event IDs since they are stored as a list instead
    if conn_obj.event_id is not None:
        conn_obj.attrs['event_id_keys'] = list(conn_obj.event_id.keys())
        conn_obj.attrs['event_id_vals'] = list(conn_obj.event_id.values())

    return conn_obj
