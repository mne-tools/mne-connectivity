# Authors: Eric Larson <larson.eric.d@gmail.com>
#          Sheraz Khan <sheraz@khansheraz.com>
#          Denis Engemann <denis.engemann@gmail.com>
#          Adam Li <adam2392@gmail.com>
#
# License: BSD (3-clause)

import numpy as np
from mne import BaseEpochs
from mne.filter import next_fast_len
from mne.source_estimate import _BaseSourceEstimate
from mne.utils import (_check_option, verbose, logger, _validate_type, warn,
                       _ensure_int)

from .base import EpochTemporalConnectivity


@verbose
def envelope_correlation(data, names=None,
                         orthogonalize="pairwise",
                         log=False, absolute=True, verbose=None):
    """Compute the envelope correlation.

    Parameters
    ----------
    data : array-like, shape=(n_epochs, n_signals, n_times) | Epochs | generator
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
    log : bool
        If True (default False), square and take the log before orthonalizing
        envelopes or computing correlations.
    absolute : bool
        If True (default), then take the absolute value of correlation
        coefficients before making each epoch's correlation matrix
        symmetric (and thus before combining matrices across epochs).
        Only used when ``orthogonalize='pairwise'``.
    %(verbose)s

    Returns
    -------
    corr : instance of EpochConnectivity
        The pairwise orthogonal envelope correlations.
        This matrix is symmetric. The array
        will have three dimensions, the first of which is ``n_epochs``.
        The data shape would be ``(n_epochs, (n_nodes + 1) * n_nodes / 2)``.

    See Also
    --------
    mne_connectivity.EpochConnectivity

    Notes
    -----
    This function computes the power envelope correlation between
    orthogonalized signals :footcite:`HippEtAl2012,KhanEtAl2018`.

    If you would like to combine Epochs after the fact using some
    function over the Epochs axis, see the ``combine`` function from
    `EpochConnectivity` classes.

    References
    ----------
    .. footbibliography::
    """  # noqa
    _check_option('orthogonalize', orthogonalize, (False, 'pairwise'))
    from scipy.signal import hilbert

    corrs = list()

    n_nodes = None

    events = None
    event_id = None
    if isinstance(data, BaseEpochs):
        names = data.ch_names
        events = data.events
        event_id = data.event_id

        # Extract metadata from the Epochs data structure.
        # Make Annotations persist through by adding them to the metadata.
        metadata = data.metadata
        if metadata is None:
            annots_in_metadata = False
        else:
            annots_in_metadata = all(
                name not in metadata.columns for name in [
                    'annot_onset', 'annot_duration', 'annot_description'])
        if hasattr(data, 'annotations') and not annots_in_metadata:
            data.add_annotations_to_metadata(overwrite=True)
        metadata = data.metadata
        # get the actual data in numpy
        data = data.get_data()
    else:
        metadata = None

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
        if np.issubdtype(epoch_data.dtype, np.floating):
            n_fft = next_fast_len(n_times)
            epoch_data = hilbert(epoch_data, N=n_fft, axis=-1)[..., :n_times]

        if not np.iscomplexobj(epoch_data):
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
        if orthogonalize:
            # Make it symmetric (it isn't at this point)
            if absolute:
                corr = np.abs(corr)
            corr = (corr.T + corr) / 2.

        corrs.append(corr)
        del corr

    if n_nodes is None:
        raise RuntimeError('Passing in empty epochs object is not allowed.')

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
        events=events,
        event_id=event_id,
        metadata=metadata
    )
    return conn


@verbose
def symmetric_orth(data, *, n_iter=50, tol=1e-6, verbose=None):
    """Perform symmetric orthogonalization.

    Uses the method from :footcite:`ColcloughEtAl2015` to jointly
    orthogonalize the time series.

    Parameters
    ----------
    data : ndarray, shape ([n_epochs, ]n_signals, n_times) or generator
        The data to process. If a generator, it must return 2D arrays to
        process.
    n_iter : int
        The maximum number of iterations to perform.
    tol : float
        The relative tolerance for convergence.
    %(verbose)s

    Returns
    -------
    data_orth : ndarray, shape (n_epochs, n_signals, n_times) | generator
        The orthogonalized data. If ``data`` is a generator, a generator
        is returned.

    References
    ----------
    .. footbibliography::
    """
    n_iter = _ensure_int(n_iter, 'n_iter')
    if isinstance(data, np.ndarray):
        return_generator = False
        if data.ndim == 2:
            return_singleton = True
            data = [data]
        else:
            return_singleton = False
    else:
        return_generator = True
        return_singleton = False
    data_out = _gen_sym_orth(data, n_iter, tol)
    if not return_generator:
        data_out = np.array(list(data_out))
        if return_singleton:
            data_out = data_out[0]
    return data_out


def _gen_sym_orth(data, n_iter, tol):
    for ei, Z in enumerate(data):
        name = f'data[{ei}]'
        _validate_type(Z, np.ndarray, name)
        _check_option(f'{name}.ndim', Z.ndim, (2,))
        _check_option(f'{name}.dtype.kind', Z.dtype.kind, 'f')
        # implementation follows Colclough et al. 2015 (NeuroImage), quoted
        # below the paper formulation has Z of shape (m, n) with m time points
        # and n sensors, but let's reformulate to our dimensionality
        n, m = Z.shape
        if m < n:
            raise RuntimeError(
                f'Symmetric orth requires at least as many time points ({m}) '
                f'as the number of time series ({n})')
        logger.debug('Symmetric orth')
        # "starting with D(1) = I_n"
        d = np.ones(n)  # don't bother with making it full diag
        # "we then allow the vector magnitudes to vary and reduce the error ε
        # by iterating (4), (6) and (8) until convergence."
        last_err = np.inf
        power = np.linalg.norm(Z, 'fro') ** 2
        power = power or 1.
        P = None
        rank = None
        for ii in range(n_iter):
            # eq. 4: UΣVᵀ = SVD(ZD), but our Z is transposed
            U, s, Vh = np.linalg.svd(Z.T * d, full_matrices=False)
            this_rank = (s >= s[0] * 1e-6).sum()
            if rank is None:
                rank = this_rank
                # Use single precision here (should be good enough)
                if rank < len(Z):
                    warn(f'Data are rank deficient ({rank} < {len(Z)}), some '
                         'orthogonalized components will be noise')
            # eq. 6 with an added transpose
            O_ = Vh.T @ U.T
            # eq. 8 (D=diag(diag(Z^T O)))
            d = np.einsum('ij,ij->i', Z, O_)
            # eq. 2 (P=OD) with an added transpose
            O_ *= d[:, np.newaxis]
            P = O_
            err = _ep(Z, P) / power
            delta = 0 if err == 0 else (last_err - err) / err
            logger.debug(f'    {ii:2d}: ε={delta:0.2e} ({err}; r={this_rank})')
            if err == 0 or delta < tol:
                logger.debug(f'Convergence reached on iteration {ii}')
                break
            last_err = err
        else:
            warn(f'Symmetric orth did not converge for data[{ei}]')
        yield P


def _ep(Z, P):
    return np.linalg.norm(Z - P, 'fro') ** 2
