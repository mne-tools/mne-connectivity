# Authors: Adam Li <adam2392@gmail.com>
#
# License: BSD (3-clause)

import numpy as np
import xarray as xr
from mne.parallel import parallel_func
from mne.time_frequency import (tfr_array_morlet, tfr_array_multitaper)
from mne.utils import logger

from ..base import (EpochSpectroTemporalConnectivity)
from .smooth import _create_kernel, _smooth_spectra
from ..utils import check_indices, fill_doc


@fill_doc
def spectral_connectivity_time(data, names=None, method='coh', indices=None,
                               sfreq=2 * np.pi, foi=None, sm_times=.5,
                               sm_freqs=1, sm_kernel='hanning',
                               mode='cwt_morlet', mt_bandwidth=None,
                               freqs=None, n_cycles=7, decim=1,
                               block_size=None, n_jobs=1,
                               verbose=None):
    """Compute frequency- and time-frequency-domain connectivity measures.

    This method computes single-Epoch time-resolved spectral connectivity.

    The connectivity method(s) are specified using the "method" parameter.
    All methods are based on estimates of the cross- and power spectral
    densities (CSD/PSD) Sxy and Sxx, Syy.

    Parameters
    ----------
    data : Epochs
        The data from which to compute connectivity.
    %(names)s
    method : str | list of str
        Connectivity measure(s) to compute. These can be ``['coh', 'plv',
        'sxy']``. These are:

            * 'coh' : Coherence
            * 'plv' : Phase-Locking Value (PLV)
            * 'sxy' : Cross-spectrum

        By default, the coherence is used.
    indices : tuple of array | None
        Two arrays with indices of connections for which to compute
        connectivity. I.e. it is a ``(n_pairs, 2)`` array essentially.
        If None, all connections are computed.
    sfreq : float
        The sampling frequency.
    foi : array_like | None
        Extract frequencies of interest. This parameters should be an array of
        shapes (n_foi, 2) defining where each band of interest start and
        finish.
    sm_times : float
        Amount of time to consider for the temporal smoothing in seconds. By
        default, 0.5 sec smoothing is used.
    sm_freqs : int
        Number of points for frequency smoothing. By default, 1 is used which
        is equivalent to no smoothing.
    sm_kernel : {'square', 'hanning'}
        Kernel type to use. Choose either 'square' or 'hanning' (default).
    mode : str, optional
        Spectrum estimation mode can be either: 'multitaper', or
        'cwt_morlet'.
    mt_bandwidth : float | None
        The bandwidth of the multitaper windowing function in Hz.
        Only used in 'multitaper' mode.
    freqs : array
        Array of frequencies of interest for use in time-frequency
        decomposition method (specified by ``mode``).
    n_cycles : float | array of float
        Number of cycles for use in time-frequency decomposition method
        (specified by ``mode``). Fixed number or one per frequency.
    decim : int | 1
        To reduce memory usage, decimation factor after time-frequency
        decomposition. default 1 If int, returns tfr[…, ::decim]. If slice,
        returns tfr[…, decim].
    block_size : int
        How many connections to compute at once (higher numbers are faster
        but require more memory).
    n_jobs : int
        How many epochs to process in parallel.
    %(verbose)s

    Returns
    -------
    con : array | instance of Connectivity
        Computed connectivity measure(s). Either an instance of
        ``SpectralConnectivity`` or ``SpectroTemporalConnectivity``.
        The shape of each connectivity dataset is either
        (n_signals ** 2, n_freqs) mode: 'multitaper' or 'fourier'
        (n_signals ** 2, n_freqs, n_times) mode: 'cwt_morlet'
        when "indices" is None, or
        (n_con, n_freqs) mode: 'multitaper' or 'fourier'
        (n_con, n_freqs, n_times) mode: 'cwt_morlet'
        when "indices" is specified and "n_con = len(indices[0])".

    See Also
    --------
    mne_connectivity.spectral_connectivity_epochs
    mne_connectivity.SpectralConnectivity
    mne_connectivity.SpectroTemporalConnectivity

    Notes
    -----
    This function was originally implemented in ``frites`` and was
    ported over.

    .. versionadded:: 0.3
    """
    events = None
    event_id = None
    # extract data from Epochs object
    names = data.ch_names
    times = data.times  # input times for Epochs input type
    sfreq = data.info['sfreq']
    events = data.events
    event_id = data.event_id
    n_epochs, n_signals, n_times = data.get_data().shape
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
    data = data.get_data()

    # convert kernel width in time to samples
    if isinstance(sm_times, (int, float)):
        sm_times = int(np.round(sm_times * sfreq))

    # convert frequency smoothing from hz to samples
    if isinstance(sm_freqs, (int, float)):
        sm_freqs = int(np.round(max(sm_freqs, 1)))

    # temporal decimation
    if isinstance(decim, int):
        times = times[::decim]
        sm_times = int(np.round(sm_times / decim))
        sm_times = max(sm_times, 1)

    # Create smoothing kernel
    kernel = _create_kernel(sm_times, sm_freqs, kernel=sm_kernel)

    # get indices of pairs of (group) regions
    roi = names  # ch_names
    if indices is None:
        # roi_gp and roi_idx
        roi_gp, _ = roi, np.arange(len(roi)).reshape(-1, 1)

        # get pairs for directed / undirected conn
        source_idx, target_idx = np.triu_indices(len(roi_gp), k=0)
    else:
        indices_use = check_indices(indices)
        source_idx = [x[0] for x in indices_use]
        target_idx = [x[1] for x in indices_use]
        roi_gp, _ = roi, np.arange(len(roi)).reshape(-1, 1)
    n_pairs = len(source_idx)

    # frequency checking
    if freqs is not None:
        # check for single frequency
        if isinstance(freqs, (int, float)):
            freqs = [freqs]
        # array conversion
        freqs = np.asarray(freqs)
        # check order for multiple frequencies
        if len(freqs) >= 2:
            delta_f = np.diff(freqs)
            increase = np.all(delta_f > 0)
            assert increase, "Frequencies should be in increasing order"

        # frequency mean
        if foi is None:
            foi_idx = foi_s = foi_e = None
            f_vec = freqs
        else:
            _f = xr.DataArray(np.arange(len(freqs)), dims=('freqs',),
                              coords=(freqs,))
            foi_s = _f.sel(freqs=foi[:, 0], method='nearest').data
            foi_e = _f.sel(freqs=foi[:, 1], method='nearest').data
            foi_idx = np.c_[foi_s, foi_e]
            f_vec = freqs[foi_idx].mean(1)

    # build block size indices
    if isinstance(block_size, int) and (block_size > 1):
        blocks = np.array_split(np.arange(n_epochs), block_size)
    else:
        blocks = [np.arange(n_epochs)]

    n_freqs = len(f_vec)

    # compute coherence on blocks of trials
    conn = np.zeros((n_epochs, n_pairs, n_freqs, len(times)))
    logger.info('Connectivity computation...')

    # parameters to pass to the connectivity function
    call_params = dict(
        method=method, kernel=kernel, foi_idx=foi_idx,
        source_idx=source_idx, target_idx=target_idx,
        mode=mode, sfreq=sfreq, freqs=freqs, n_cycles=n_cycles,
        mt_bandwidth=mt_bandwidth,
        decim=decim, kw_cwt={}, kw_mt={}, n_jobs=n_jobs,
        verbose=verbose)

    for epoch_idx in blocks:
        # compute time-resolved spectral connectivity
        conn_tr = _spectral_connectivity(data[epoch_idx, ...], **call_params)

        # merge results
        conn[epoch_idx, ...] = np.stack(conn_tr, axis=1)

    # create a Connectivity container
    indices = 'symmetric'
    conn = EpochSpectroTemporalConnectivity(
        conn, freqs=f_vec, times=times,
        n_nodes=n_signals, names=names, indices=indices, method=method,
        spec_method=mode, events=events, event_id=event_id, metadata=metadata)

    return conn


def _spectral_connectivity(data, method, kernel, foi_idx,
                           source_idx, target_idx,
                           mode, sfreq, freqs, n_cycles, mt_bandwidth=None,
                           decim=1, kw_cwt={}, kw_mt={}, n_jobs=1,
                           verbose=False):
    """EStimate time-resolved connectivity for one epoch.

    See spectral_connectivity_epoch."""
    n_pairs = len(source_idx)

    # first compute time-frequency decomposition
    collapse = None
    if mode == 'cwt_morlet':
        out = tfr_array_morlet(
            data, sfreq, freqs, n_cycles=n_cycles, output='complex',
            decim=decim, n_jobs=n_jobs, **kw_cwt)
    elif mode == 'multitaper':
        # In case multiple values are provided for mt_bandwidth
        # the MT decomposition is done separatedly for each
        # Frequency center
        if isinstance(mt_bandwidth, (list, tuple, np.ndarray)):
            # Arrays freqs, n_cycles, mt_bandwidth should have the same size
            assert len(freqs) == len(n_cycles) == len(mt_bandwidth)
            out = []
            for f_c, n_c, mt in zip(freqs, n_cycles, mt_bandwidth):
                out += [tfr_array_multitaper(
                    data, sfreq, [f_c], n_cycles=float(n_c), time_bandwidth=mt,
                    output='complex', decim=decim, n_jobs=n_jobs, **kw_mt)]
            out = np.stack(out, axis=3).squeeze()
        elif isinstance(mt_bandwidth, (type(None), int, float)):
            out = tfr_array_multitaper(
                data, sfreq, freqs, n_cycles=n_cycles,
                time_bandwidth=mt_bandwidth, output='complex', decim=decim,
                n_jobs=n_jobs, **kw_mt)
            collapse = True
            if out.ndim == 5:  # newest MNE-Python
                collapse = -3

    # get the supported connectivity function
    conn_func = {'coh': _coh, 'plv': _plv, 'sxy': _cs}[method]

    # computes conn across trials
    # TODO: This is wrong -- it averages in the complex domain (over tapers).
    # What it *should* do is compute the conn for each taper, then average
    # (see below).
    if collapse is not None:
        out = np.mean(out, axis=collapse)
    this_conn = conn_func(out, kernel, foi_idx, source_idx, target_idx,
                          n_jobs=n_jobs, verbose=verbose, total=n_pairs)
    # This is where it should go, but the regression test fails...
    # if collapse is not None:
    #     this_conn = [c.mean(axis=collapse) for c in this_conn]
    return this_conn


###############################################################################
###############################################################################
#                               TIME-RESOLVED CORE FUNCTIONS
###############################################################################
###############################################################################

def _coh(w, kernel, foi_idx, source_idx, target_idx, n_jobs, verbose, total):
    """Pairwise coherence."""
    # auto spectra (faster that w * w.conj())
    s_auto = w.real ** 2 + w.imag ** 2

    # smooth the auto spectra
    s_auto = _smooth_spectra(s_auto, kernel)

    # define the pairwise coherence
    def pairwise_coh(w_x, w_y):
        # computes the coherence
        s_xy = w[:, w_y] * np.conj(w[:, w_x])
        s_xy = _smooth_spectra(s_xy, kernel)
        s_xx = s_auto[:, w_x]
        s_yy = s_auto[:, w_y]
        out = np.abs(s_xy) ** 2 / (s_xx * s_yy)
        # mean inside frequency sliding window (if needed)
        if isinstance(foi_idx, np.ndarray):
            return _foi_average(out, foi_idx)
        else:
            return out

    # define the function to compute in parallel
    parallel, p_fun, n_jobs = parallel_func(
        pairwise_coh, n_jobs=n_jobs, verbose=verbose, total=total)

    # compute the single trial coherence
    return parallel(p_fun(s, t) for s, t in zip(source_idx, target_idx))


def _plv(w, kernel, foi_idx, source_idx, target_idx, n_jobs, verbose, total):
    """Pairwise phase-locking value."""
    # define the pairwise plv
    def pairwise_plv(w_x, w_y):
        # computes the plv
        s_xy = w[:, w_y] * np.conj(w[:, w_x])
        # complex exponential of phase differences
        exp_dphi = s_xy / np.abs(s_xy)
        # smooth e^(-i*\delta\phi)
        exp_dphi = _smooth_spectra(exp_dphi, kernel)
        # computes plv
        out = np.abs(exp_dphi)
        # mean inside frequency sliding window (if needed)
        if isinstance(foi_idx, np.ndarray):
            return _foi_average(out, foi_idx)
        else:
            return out

    # define the function to compute in parallel
    parallel, p_fun, n_jobs = parallel_func(
        pairwise_plv, n_jobs=n_jobs, verbose=verbose, total=total)

    # compute the single trial coherence
    return parallel(p_fun(s, t) for s, t in zip(source_idx, target_idx))


def _cs(w, kernel, foi_idx, source_idx, target_idx, n_jobs, verbose, total):
    """Pairwise cross-spectra."""
    # define the pairwise cross-spectra
    def pairwise_cs(w_x, w_y):
        #  computes the cross-spectra
        out = w[:, w_x] * np.conj(w[:, w_y])
        out = _smooth_spectra(out, kernel)
        if foi_idx is not None:
            return _foi_average(out, foi_idx)
        else:
            return out

    # define the function to compute in parallel
    parallel, p_fun, n_jobs = parallel_func(
        pairwise_cs, n_jobs=n_jobs, verbose=verbose, total=total)

    # compute the single trial coherence
    return parallel(p_fun(s, t) for s, t in zip(source_idx, target_idx))


def _foi_average(conn, foi_idx):
    """Average inside frequency bands.

    The frequency dimension should be located at -2.

    Parameters
    ----------
    conn : np.ndarray
        Array of shape (..., n_freqs, n_times)
    foi_idx : array_like
        Array of indices describing frequency bounds of shape (n_foi, 2)

    Returns
    -------
    conn_f : np.ndarray
        Array of shape (..., n_foi, n_times)
    """
    # get the number of foi
    n_foi = foi_idx.shape[0]

    # get input shape and replace n_freqs with the number of foi
    sh = list(conn.shape)
    sh[-2] = n_foi

    # compute average
    conn_f = np.zeros(sh, dtype=conn.dtype)
    for n_f, (f_s, f_e) in enumerate(foi_idx):
        conn_f[..., n_f, :] = conn[..., f_s:f_e, :].mean(-2)
    return conn_f
