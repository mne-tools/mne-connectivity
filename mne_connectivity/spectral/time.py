# Authors: Adam Li <adam2392@gmail.com>
#          Santeri Ruuskanen <santeriruuskanen@gmail.com>
#
# License: BSD (3-clause)

import numpy as np
import xarray as xr
from mne.epochs import BaseEpochs
from mne.parallel import parallel_func
from mne.time_frequency import (tfr_array_morlet, tfr_array_multitaper)
from mne.utils import logger

from ..base import (SpectralConnectivity, EpochSpectralConnectivity)
from .epochs import _compute_freqs, _compute_freq_mask
from .smooth import _create_kernel, _smooth_spectra
from ..utils import check_indices, fill_doc


@fill_doc
def spectral_connectivity_time(data, names=None, method='coh', average=False,
                               indices=None, sfreq=2 * np.pi, fmin=None,
                               fmax=None, fskip=0, faverage=False, sm_times=.5,
                               sm_freqs=1, sm_kernel='hanning',
                               mode='cwt_morlet', mt_bandwidth=None,
                               cwt_freqs=None, n_cycles=7, decim=1,
                               block_size=1000, n_jobs=1, verbose=None):
    """Compute frequency- and time-frequency-domain connectivity measures.

    This method computes time-resolved connectivity measures for Epochs.

    The connectivity method(s) are specified using the "method" parameter.
    All methods are based on estimates of the cross- and power spectral
    densities (CSD/PSD) Sxy and Sxx, Syy.

    Parameters
    ----------
    data : array_like, shape (n_epochs, n_signals, n_times) | Epochs
        The data from which to compute connectivity.
    %(names)s
    method : str | list of str
        Connectivity measure(s) to compute. These can be ``['coh', 'plv',
        'sxy']``. These are:

            * 'coh' : Coherence
            * 'plv' : Phase-Locking Value (PLV)
            * 'sxy' : Cross-spectrum

        By default, the coherence is used.
    average : bool
        Average connectivity scores over Epochs. If True, output will be
        an instance of ``SpectralConnectivity`` , otherwise
        ``EpochSpectralConnectivity``. By default False.
    indices : tuple of array | None
        Two arrays with indices of connections for which to compute
        connectivity. I.e. it is a ``(n_pairs, 2)`` array essentially.
        If None, all connections are computed.
    sfreq : float
        The sampling frequency.
    fmin : float | tuple of float
        The lower frequency of interest. Multiple bands are defined using
        a tuple, e.g., (8., 20.) for two bands with 8Hz and 20Hz lower freq.
        If None the frequency corresponding to an epoch length of 5 cycles
        is used.
    fmax : float | tuple of float
        The upper frequency of interest. Multiple bands are defined using
        a tuple, e.g. (13., 30.) for two band with 13Hz and 30Hz upper freq.
    fskip : int
        Omit every "(fskip + 1)-th" frequency bin to decimate in frequency
        domain.
    faverage : bool
        Average connectivity scores for each frequency band. If True,
        the output freqs will be a list with arrays of the frequencies
        that were averaged.
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
    cwt_freqs : array
        Array of frequencies of interest for time-frequency decomposition.
        Only used in 'cwt_morlet' mode.
    n_cycles : float | array of float
        Number of cycles for use in time-frequency decomposition method
        (specified by ``mode``). Fixed number or one per frequency.
    decim : int | 1
        To reduce memory usage, decimation factor after time-frequency
        decomposition. default 1 If int, returns tfr[…, ::decim]. If slice,
        returns tfr[…, decim].
    block_size : int
        How many epochs to compute at once (higher numbers are faster
        but require more memory).
    n_jobs : int
        How many epochs to process in parallel.
    %(verbose)s

    Returns
    -------
    con : instance of Connectivity | list
        Computed connectivity measure(s). An instance of
        ``EpochSpectralConnectivity``, ``SpectralConnectivity``
        or a list of instances corresponding to connectivity measures if
        several connectivity measures are specified.
        The shape of each connectivity dataset is
        (n_epochs, n_signals, n_signals, n_freqs) when indices is None
        and (n_epochs, n_nodes, n_nodes, n_freqs) when "indices" is specified
        and "n_nodes = len(indices[0])".

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
    if isinstance(data, BaseEpochs):
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
    else:
        data = np.asarray(data)
        n_epochs, n_signals, n_times = data.shape
        times = np.arange(0, n_times)
        names = np.arange(0, n_signals)
        metadata = None

    # check that method is a list
    if isinstance(method, str):
        method = [method]
    # check that fmin and fmax are lists
    if fmin is None:
        fmin = 1
    if fmax is None:
        fmax = sfreq / 2
    fmin = np.array((fmin,), dtype=float).ravel()
    fmax = np.array((fmax,), dtype=float).ravel()
    if len(fmin) != len(fmax):
        raise ValueError('fmin and fmax must have the same length')
    if np.any(fmin > fmax):
        raise ValueError('fmax must be larger than fmin')

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
    if cwt_freqs is not None:
        # check for single frequency
        if isinstance(cwt_freqs, (int, float)):
            cwt_freqs = [cwt_freqs]
        # array conversion
        cwt_freqs = np.asarray(cwt_freqs)
        # check order for multiple frequencies
        if len(cwt_freqs) >= 2:
            delta_f = np.diff(cwt_freqs)
            increase = np.all(delta_f > 0)
            assert increase, "Frequencies should be in increasing order"

    # compute frequencies to analyze based on number of samples,
    # sampling rate, specified wavelet frequencies and mode
    freqs = _compute_freqs(n_times, sfreq, cwt_freqs, mode)

    if fmin is not None and fmax is not None:
        # compute the mask based on specified min/max and decimation factor
        freq_mask = _compute_freq_mask(freqs, fmin, fmax, fskip)

        # the frequency points where we compute connectivity
        freqs = freqs[freq_mask]

    # frequency mean
    if fmin is None or fmax is None:
        foi_idx = None
        f_vec = freqs
    else:
        _f = xr.DataArray(np.arange(len(freqs)), dims=('freqs',),
                          coords=(freqs,))
        foi_s = _f.sel(freqs=fmin, method='nearest').data
        foi_e = _f.sel(freqs=fmax, method='nearest').data
        foi_idx = np.c_[foi_s, foi_e]
        f_vec = freqs[foi_idx].mean(1)

    if faverage:
        n_freqs = len(fmin)
        out_freqs = f_vec
    else:
        n_freqs = len(freqs)
        out_freqs = freqs

    # build block size indices
    if block_size > n_epochs:
        block_size = n_epochs

    if isinstance(block_size, int) and (block_size > 1):
        n_blocks = n_epochs // block_size + n_epochs % block_size
        blocks = np.array_split(np.arange(n_epochs), n_blocks)
    else:
        blocks = [np.arange(n_epochs)]

    # compute connectivity on blocks of trials
    conn = {}
    for m in method:
        conn[m] = np.zeros((n_epochs, n_pairs, n_freqs))

    logger.info('Connectivity computation...')

    # parameters to pass to the connectivity function
    call_params = dict(
        method=method, kernel=kernel, foi_idx=foi_idx,
        source_idx=source_idx, target_idx=target_idx,
        mode=mode, sfreq=sfreq, freqs=freqs, faverage=faverage,
        n_cycles=n_cycles, mt_bandwidth=mt_bandwidth,
        decim=decim, kw_cwt={}, kw_mt={}, n_jobs=n_jobs,
        verbose=verbose)

    for epoch_idx in blocks:
        # compute time-resolved spectral connectivity
        conn_tr = _spectral_connectivity(data[epoch_idx, ...], **call_params)

        # merge results
        for m in method:
            conn[m][epoch_idx, ...] = np.stack(conn_tr[m],
                                               axis=1).squeeze(axis=-1)

    # create a Connectivity container
    indices = 'symmetric'

    if average:
        out = [SpectralConnectivity(
               conn[m].mean(axis=0), freqs=out_freqs, n_nodes=n_signals,
               names=names, indices=indices, method=method, spec_method=mode,
               events=events, event_id=event_id, metadata=metadata)
               for m in method]
    else:
        out = [EpochSpectralConnectivity(
               conn[m], freqs=out_freqs, n_nodes=n_signals, names=names,
               indices=indices, method=method, spec_method=mode, events=events,
               event_id=event_id, metadata=metadata) for m in method]

    # return the object instead of list of length one
    if len(out) == 1:
        return out[0]
    else:
        return out


def _spectral_connectivity(data, method, kernel, foi_idx,
                           source_idx, target_idx,
                           mode, sfreq, freqs, faverage, n_cycles,
                           mt_bandwidth=None, decim=1, kw_cwt={}, kw_mt={},
                           n_jobs=1, verbose=False):
    """Estimate time-resolved connectivity for one epoch.

    See spectral_connectivity_epoch."""
    n_pairs = len(source_idx)

    # first compute time-frequency decomposition
    if mode == 'cwt_morlet':
        out = tfr_array_morlet(
            data, sfreq, freqs, n_cycles=n_cycles, output='complex',
            decim=decim, n_jobs=n_jobs, **kw_cwt)
        out = np.expand_dims(out, axis=2)  # same dims with multitaper
    elif mode == 'multitaper':
        print(data.shape)
        out = tfr_array_multitaper(
            data, sfreq, freqs, n_cycles=n_cycles,
            time_bandwidth=mt_bandwidth, output='complex', decim=decim,
            n_jobs=n_jobs, **kw_mt)
    else:
        raise ValueError("Mode must be 'cwt_morlet' or 'multitaper'.")

    # compute for each required connectivity method
    this_conn = {}
    conn_func = {'coh': _coh, 'plv': _plv, 'sxy': _cs, 'pli': _pli,
                 'wpli': _wpli}
    for m in method:
        c_func = conn_func[m]
        # compute connectivity
        this_conn[m] = c_func(out, kernel, foi_idx, source_idx,
                              target_idx, n_jobs=n_jobs,
                              verbose=verbose, total=n_pairs,
                              faverage=faverage)
        # mean over tapers
        this_conn[m] = [c.mean(axis=1) for c in this_conn[m]]

    return this_conn


###############################################################################
###############################################################################
#                               TIME-RESOLVED CORE FUNCTIONS
###############################################################################
###############################################################################

def _coh(w, kernel, foi_idx, source_idx, target_idx, n_jobs, verbose, total,
         faverage):
    """Pairwise coherence.

    Input signal w is of shape (n_epochs, n_chans, n_tapers, n_freqs,
    n_times)."""
    # auto spectra (faster that w * w.conj())
    s_auto = w.real ** 2 + w.imag ** 2

    # smooth the auto spectra
    s_auto = _smooth_spectra(s_auto, kernel)

    def pairwise_coh(w_x, w_y):
        # compute coherence
        s_xy = w[:, w_y] * np.conj(w[:, w_x])
        s_xy = _smooth_spectra(s_xy, kernel)
        s_xx = s_auto[:, w_x]
        s_yy = s_auto[:, w_y]
        out = np.abs(s_xy.mean(axis=-1, keepdims=True)) / \
            np.sqrt(s_xx.mean(axis=-1, keepdims=True) *
                    s_yy.mean(axis=-1, keepdims=True))
        # mean inside frequency sliding window (if needed)
        if isinstance(foi_idx, np.ndarray) and faverage:
            return _foi_average(out, foi_idx)
        else:
            return out

    # define the function to compute in parallel
    parallel, p_fun, n_jobs = parallel_func(
        pairwise_coh, n_jobs=n_jobs, verbose=verbose, total=total)

    # compute pairwise coherence coherence
    return parallel(p_fun(s, t) for s, t in zip(source_idx, target_idx))


def _plv(w, kernel, foi_idx, source_idx, target_idx, n_jobs, verbose, total,
         faverage):
    """Pairwise phase-locking value.

    Input signal w is of shape (n_epochs, n_chans, n_tapers, n_freqs,
    n_times)."""
    # define the pairwise plv
    def pairwise_plv(w_x, w_y):
        # compute plv
        s_xy = w[:, w_y] * np.conj(w[:, w_x])
        # complex exponential of phase differences
        exp_dphi = s_xy / np.abs(s_xy)
        # smooth e^(-i*\delta\phi)
        exp_dphi = _smooth_spectra(exp_dphi, kernel)
        # mean over samples (time axis)
        exp_dphi_mean = exp_dphi.mean(axis=-1, keepdims=True)
        out = np.abs(exp_dphi_mean)
        # mean inside frequency sliding window (if needed)
        if isinstance(foi_idx, np.ndarray) and faverage:
            return _foi_average(out, foi_idx)
        else:
            return out

    # define the function to compute in parallel
    parallel, p_fun, n_jobs = parallel_func(
        pairwise_plv, n_jobs=n_jobs, verbose=verbose, total=total)

    # compute the single trial plv
    return parallel(p_fun(s, t) for s, t in zip(source_idx, target_idx))


def _pli(w, kernel, foi_idx, source_idx, target_idx, n_jobs, verbose, total,
         faverage):
    """Pairwise phase-lag index.

    Input signal w is of shape (n_epochs, n_chans, n_tapers, n_freqs,
    n_times)."""
    # define the pairwise pli
    def pairwise_pli(w_x, w_y):
        # compute cross spectrum
        s_xy = w[:, w_y] * np.conj(w[:, w_x])
        # smooth e^(-i*\delta\phi)
        s_xy = _smooth_spectra(s_xy, kernel)
        # phase lag index
        out = np.abs(np.mean(np.sign(np.imag(s_xy)),
                             axis=-1, keepdims=True))
        # mean inside frequency sliding window (if needed)
        if isinstance(foi_idx, np.ndarray) and faverage:
            return _foi_average(out, foi_idx)
        else:
            return out

    # define the function to compute in parallel
    parallel, p_fun, n_jobs = parallel_func(
        pairwise_pli, n_jobs=n_jobs, verbose=verbose, total=total)

    # compute the single trial pli
    return parallel(p_fun(s, t) for s, t in zip(source_idx, target_idx))


def _wpli(w, kernel, foi_idx, source_idx, target_idx, n_jobs, verbose, total,
          faverage):
    """Pairwise weighted phase-lag index.

    Input signal w is of shape (n_epochs, n_chans, n_tapers, n_freqs,
    n_times)."""
    # define the pairwise wpli
    def pairwise_wpli(w_x, w_y):
        # compute cross spectrum
        s_xy = w[:, w_y] * np.conj(w[:, w_x])
        # smooth
        s_xy = _smooth_spectra(s_xy, kernel)
        # magnitude of the mean of the imaginary part of the cross spectrum
        s_xy_mean_abs = np.abs(s_xy.imag.mean(axis=-1, keepdims=True))
        # mean of the magnitudes of the imaginary part of the cross spectrum
        s_xy_abs_mean = np.abs(s_xy.imag).mean(axis=-1, keepdims=True)
        out = s_xy_mean_abs / s_xy_abs_mean
        # mean inside frequency sliding window (if needed)
        if isinstance(foi_idx, np.ndarray) and faverage:
            return _foi_average(out, foi_idx)
        else:
            return out

    # define the function to compute in parallel
    parallel, p_fun, n_jobs = parallel_func(
        pairwise_wpli, n_jobs=n_jobs, verbose=verbose, total=total)

    # compute the single trial wpli
    return parallel(p_fun(s, t) for s, t in zip(source_idx, target_idx))


def _cs(w, kernel, foi_idx, source_idx, target_idx, n_jobs, verbose, total,
        faverage):
    """Pairwise cross-spectra."""
    # define the pairwise cross-spectra
    def pairwise_cs(w_x, w_y):
        #  computes the cross-spectra
        out = w[:, w_x] * np.conj(w[:, w_y])
        out = _smooth_spectra(out, kernel)
        if isinstance(foi_idx, np.ndarray) and faverage:
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
