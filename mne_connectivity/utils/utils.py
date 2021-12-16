# Authors: Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#          Adam Li <adam2392@gmail.com>
#
# License: BSD (3-clause)
import numpy as np

from mne import BaseEpochs
from mne.utils import logger


def map_epoch_annotations_to_epoch(dest_epoch, src_epoch):
    """Map Annotations that occur in one Epoch to another Epoch.

    Two different Epochs might occur at different time points.
    This function will map Annotations that occur in one Epoch
    setting to another Epoch taking into account their onset
    samples and window lengths.

    Parameters
    ----------
    dest_epoch : instance of Epochs | events array
        The reference Epochs that you want to match to.
    src_epoch : instance of Epochs | events array
        The source Epochs that contain Epochs you want to
        see if it overlaps at any point with ``dest_epoch``.

    Returns
    -------
    all_cases : np.ndarray of shape (n_src_epochs, n_dest_epochs)
        This is an array indicating the overlap of any source epoch
        relative to the destination epoch. An overlap is indicated
        by a ``True``, whereas if a source Epoch does not overlap
        with a destination Epoch, then the element will be ``False``.

    Notes
    -----
    This is a useful utility function to enable mapping Autoreject
    ``RejectLog`` that occurs over a set of defined Epochs to
    another Epoched data structure, such as a ``Epoch*`` connectivity
    class, which computes connectivity over Epochs.
    """
    if isinstance(dest_epoch, BaseEpochs):
        dest_events = dest_epoch.events
        dest_times = dest_epoch.times
        dest_sfreq = dest_epoch._raw_sfreq
    else:
        dest_events = dest_epoch
    if isinstance(src_epoch, BaseEpochs):
        src_events = src_epoch.events
        src_times = src_epoch.times
        src_sfreq = src_epoch._raw_sfreq
    else:
        src_events = src_epoch

    # get the sample points of the source Epochs we want
    # to map over to the destination sample points
    src_onset_sample = src_events[:, 0]
    src_epoch_tzeros = src_onset_sample / src_sfreq
    dest_onset_sample = dest_events[:, 0]
    dest_epoch_tzeros = dest_onset_sample / dest_sfreq

    # get start and stop points of every single source Epoch
    src_epoch_starts, src_epoch_stops = np.atleast_2d(
        src_epoch_tzeros) + np.atleast_2d(src_times[[0, -1]]).T

    # get start and stop points of every single destination Epoch
    dest_epoch_starts, dest_epoch_stops = np.atleast_2d(
        dest_epoch_tzeros) + np.atleast_2d(dest_times[[0, -1]]).T

    # get destination Epochs that start within the source Epoch
    src_straddles_dest_start = np.logical_and(
        np.atleast_2d(dest_epoch_starts) >= np.atleast_2d(src_epoch_starts).T,
        np.atleast_2d(dest_epoch_starts) < np.atleast_2d(src_epoch_stops).T)

    # get epochs that end within the annotations
    src_straddles_dest_end = np.logical_and(
        np.atleast_2d(dest_epoch_stops) > np.atleast_2d(src_epoch_starts).T,
        np.atleast_2d(dest_epoch_stops) <= np.atleast_2d(src_epoch_stops).T)

    # get epochs that are fully contained within annotations
    src_fully_within_dest = np.logical_and(
        np.atleast_2d(dest_epoch_starts) <= np.atleast_2d(src_epoch_starts).T,
        np.atleast_2d(dest_epoch_stops) >= np.atleast_2d(src_epoch_stops).T)

    # combine all cases to get array of shape (n_src_epochs, n_dest_epochs).
    # Nonzero entries indicate overlap between the corresponding
    # annotation (row index) and epoch (column index).
    all_cases = (src_straddles_dest_start +
                 src_straddles_dest_end +
                 src_fully_within_dest)

    return all_cases


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
    from mne_connectivity.base import _Connectivity

    if isinstance(connectivity, _Connectivity):
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
