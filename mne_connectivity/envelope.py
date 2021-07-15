# Authors: Eric Larson <larson.eric.d@gmail.com>
#          Sheraz Khan <sheraz@khansheraz.com>
#          Denis Engemann <denis.engemann@gmail.com>
#          Adam Li <adam2392@gmail.com>
#
# License: BSD (3-clause)

import numpy as np
from mne.filter import next_fast_len
from mne.source_estimate import _BaseSourceEstimate
from mne.utils import _check_option, verbose

from .base import EpochTemporalConnectivity


@verbose
def envelope_correlation(data, names=None,
                         orthogonalize="pairwise",
                         log=False, absolute=True, verbose=None):
    """Compute the envelope correlation.

    Parameters
    ----------
    data : array-like, shape=(n_epochs, n_signals, n_times) | generator
        The data from which to compute connectivity.
        The array-like object can also be a list/generator of array,
        each with shape (n_signals, n_times), or a :class:`~mne.SourceEstimate`
        object (and ``stc.data`` will be used). If it's float data,
        the Hilbert transform will be applied; if it's complex data,
        it's assumed the Hilbert has already been applied.
    names : list | array-like | None
        A list of names associated with the signals in ``data``.
        If None, will be a list of indices of the number of nodes.
    orthogonalize : 'pairwise' | False
        Whether to orthogonalize with the pairwise method or not.
        Defaults to 'pairwise'. Note that when False,
        the correlation matrix will not be returned with
        absolute values.

        .. versionadded:: 0.19
    log : bool
        If True (default False), square and take the log before orthonalizing
        envelopes or computing correlations.

        .. versionadded:: 0.22
    absolute : bool
        If True (default), then take the absolute value of correlation
        coefficients before making each epoch's correlation matrix
        symmetric (and thus before combining matrices across epochs).
        Only used when ``orthogonalize=True``.

        .. versionadded:: 0.22
    %(verbose)s

    Returns
    -------
    corr : instance of EpochConnectivity
        The pairwise orthogonal envelope correlations.
        This matrix is symmetric. The array
        will have three dimensions, the first of which is ``n_epochs``.
        The data shape would be ``(n_epochs, (n_nodes + 1) * n_nodes / 2)``

    Notes
    -----
    This function computes the power envelope correlation between
    orthogonalized signals :footcite:`HippEtAl2012,KhanEtAl2018`.

    If you would like to combine Epochs after the fact using some
    function over the Epochs axis, see the ``combine`` function from
    `EpochConnectivity` classes.

    See Also
    --------
    mne_connectivity.EpochConnectivity

    References
    ----------
    .. footbibliography::
    """
    _check_option('orthogonalize', orthogonalize, (False, 'pairwise'))
    from scipy.signal import hilbert

    corrs = list()

    # Note: This is embarassingly parallel, but the overhead of sending
    # the data to different workers is roughly the same as the gain of
    # using multiple CPUs. And we require too much GIL for prefer='threading'
    # to help.
    for ei, epoch_data in enumerate(data):
        if isinstance(epoch_data, _BaseSourceEstimate):
            epoch_data = epoch_data.data
        if epoch_data.ndim != 2:
            raise ValueError('Each entry in data must be 2D, got shape %s'
                             % (epoch_data.shape,))
        n_nodes, n_times = epoch_data.shape
        if ei > 0 and n_nodes != corrs[0].shape[0]:
            raise ValueError('n_nodes mismatch between data[0] and data[%d], '
                             'got %s and %s'
                             % (ei, n_nodes, corrs[0].shape[0]))

        # Get the complex envelope (allowing complex inputs allows people
        # to do raw.apply_hilbert if they want)
        if epoch_data.dtype in (np.float32, np.float64):
            n_fft = next_fast_len(n_times)
            epoch_data = hilbert(epoch_data, N=n_fft, axis=-1)[..., :n_times]

        if epoch_data.dtype not in (np.complex64, np.complex128):
            raise ValueError('data.dtype must be float or complex, got %s'
                             % (epoch_data.dtype,))
        data_mag = np.abs(epoch_data)
        data_conj_scaled = epoch_data.conj()
        data_conj_scaled /= data_mag
        if log:
            data_mag *= data_mag
            np.log(data_mag, out=data_mag)

        # subtract means
        data_mag_nomean = data_mag - np.mean(data_mag, axis=-1, keepdims=True)

        # compute variances using linalg.norm (square, sum, sqrt) since mean=0
        data_mag_std = np.linalg.norm(data_mag_nomean, axis=-1)
        data_mag_std[data_mag_std == 0] = 1
        corr = np.empty((n_nodes, n_nodes))

        # loop over each signal in this specific epoch
        # which is now (n_signals, n_times) and compute envelope
        for li, label_data in enumerate(epoch_data):
            if orthogonalize is False:  # the new code
                label_data_orth = data_mag[li]
                label_data_orth_std = data_mag_std[li]
            else:
                label_data_orth = (label_data * data_conj_scaled).imag
                np.abs(label_data_orth, out=label_data_orth)
                # protect against invalid value -- this will be zero
                # after (log and) mean subtraction
                label_data_orth[li] = 1.
                if log:
                    label_data_orth *= label_data_orth
                    np.log(label_data_orth, out=label_data_orth)
                label_data_orth -= np.mean(label_data_orth, axis=-1,
                                           keepdims=True)
                label_data_orth_std = np.linalg.norm(label_data_orth, axis=-1)
                label_data_orth_std[label_data_orth_std == 0] = 1

            # correlation is dot product divided by variances
            corr[li] = np.sum(label_data_orth * data_mag_nomean, axis=1)
            corr[li] /= data_mag_std
            corr[li] /= label_data_orth_std
        if orthogonalize is not False:
            # Make it symmetric (it isn't at this point)
            if absolute:
                corr = np.abs(corr)
            corr = (corr.T + corr) / 2.

        corrs.append(corr)
        del corr

    # apply function on correlation structure
    n_epochs = len(corrs)

    # ravel from 2D connectivity into 1D array
    # over all epochs
    corr = np.array([_corr.flatten() for _corr in corrs])

    # create the connectivity container
    times = None

    # create time axis
    corr = corr[..., np.newaxis]

    # only get the upper-triu indices
    triu_inds = np.triu_indices(n_nodes, k=0)
    raveled_triu_inds = np.ravel_multi_index(
        triu_inds, dims=(n_nodes, n_nodes))
    corr = corr[:, raveled_triu_inds, ...]

    conn = EpochTemporalConnectivity(
        data=corr,
        names=names,
        times=times,
        method='envelope correlation',
        indices='symmetric',
        n_epochs_used=n_epochs,
        n_nodes=n_nodes,
    )
    return conn
