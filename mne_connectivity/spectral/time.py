# Authors: Adam Li <adam2392@gmail.com>
#          Santeri Ruuskanen <santeriruuskanen@gmail.com>
#
# License: BSD (3-clause)

import numpy as np
import xarray as xr
from mne.epochs import BaseEpochs
from mne.parallel import parallel_func
from mne.time_frequency import (tfr_array_morlet, tfr_array_multitaper,
                                dpss_windows)
from mne.utils import (logger, warn)

from ..base import (SpectralConnectivity, EpochSpectralConnectivity)
from .epochs import _compute_freq_mask
from .smooth import _create_kernel, _smooth_spectra
from ..utils import check_indices, fill_doc


@fill_doc
def spectral_connectivity_time(data, method='coh', average=False,
                               indices=None, sfreq=None, fmin=None,
                               fmax=None, fskip=0, faverage=False, sm_times=0,
                               sm_freqs=1, sm_kernel='hanning', padding=0,
                               mode='cwt_morlet', mt_bandwidth=None,
                               freqs=None, n_cycles=7, decim=1, n_jobs=1,
                               verbose=None):
    """Compute time-frequency-domain connectivity measures.

    This function computes spectral connectivity over time from epoched data.
    The data may consist of a single epoch.

    The connectivity method(s) are specified using the ``method`` parameter.
    All methods are based on time-resolved estimates of the cross- and
    power spectral densities (CSD/PSD) Sxy and Sxx, Syy.

    Parameters
    ----------
    data : array_like, shape (n_epochs, n_signals, n_times) | Epochs
        The data from which to compute connectivity.
    method : str | list of str
        Connectivity measure(s) to compute. These can be
        ``['coh', 'plv', 'sxy', 'pli', 'wpli']``. These are:

            * 'coh' : Coherence
            * 'plv' : Phase-Locking Value (PLV)
            * 'ciplv' : Corrected imaginary Phase-Locking Value
            * 'sxy' : Cross-spectrum
            * 'pli' : Phase-Lag Index
            * 'wpli': Weighted Phase-Lag Index
    average : bool
        Average connectivity scores over epochs. If ``True``, output will be
        an instance of :class:`SpectralConnectivity`, otherwise
        :class:`EpochSpectralConnectivity`.
    indices : tuple of array_like | None
        Two arrays with indices of connections for which to compute
        connectivity. I.e. it is a ``(n_pairs, 2)`` array essentially.
        If `None`, all connections are computed.
    sfreq : float
        The sampling frequency. Required if data is not
        :class:`Epochs <mne.Epochs>`.
    fmin : float | tuple of float | None
        The lower frequency of interest. Multiple bands are defined using
        a tuple, e.g., ``(8., 20.)`` for two bands with 8 Hz and 20 Hz lower
        bounds. If `None`, the frequency corresponding to an epoch length of
        5 cycles is used.
    fmax : float | tuple of float | None
        The upper frequency of interest. Multiple bands are defined using
        a tuple, e.g. ``(13., 30.)`` for two band with 13 Hz and 30 Hz upper
        bounds. If `None`, ``sfreq/2`` is used.
    fskip : int
        Omit every ``(fskip + 1)``-th frequency bin to decimate in frequency
        domain.
    faverage : bool
        Average connectivity scores for each frequency band. If `True`,
        the output ``freqs`` will be an array of the median frequencies of each
        band.
    sm_times : float
        Amount of time to consider for the temporal smoothing in seconds.
        If zero, no temporal smoothing is applied.
    sm_freqs : int
        Number of points for frequency smoothing. By default, 1 is used which
        is equivalent to no smoothing.
    sm_kernel : {'square', 'hanning'}
        Smoothing kernel type. Choose either 'square' or 'hanning'.
    padding: float
        Amount of time to consider as padding at the beginning and end of each
        epoch in seconds.
    mode : str
        Time-frequency decomposition method. Can be either: 'multitaper', or
        'cwt_morlet'. See :func:`mne.time_frequency.tfr_array_multitaper` and
        :func:`mne.time_frequency.tfr_array_morlet` for reference.
    mt_bandwidth : float | None
        Product between the temporal window length (in seconds) and the full
        frequency bandwidth (in Hz). This product can be seen as the surface
        of the window on the time/frequency plane and controls the frequency
        bandwidth (thus the frequency resolution) and the number of good
        tapers. See :func:`mne.time_frequency.tfr_array_multitaper`
        documentation.
    freqs : array_like
        Array of frequencies of interest for time-frequency decomposition.
        Required in ``cwt_morlet`` mode. Only the frequencies within
        the range specified by ``fmin``  and ``fmax`` are used. Required if
        ``mode='cwt_morlet'``. If set when ``mode='multitaper'``, overrides the
        automatically determined frequencies of interest.
    n_cycles : float | array_like of float
        Number of cycles in the wavelet, either a fixed number or one per
        frequency. The number of cycles ``n_cycles`` and the frequencies of
        interest ``cwt_freqs`` define the temporal window length. For details,
        see :func:`mne.time_frequency.tfr_array_morlet` documentation.
    decim : int
        To reduce memory usage, decimation factor after time-frequency
        decomposition. Returns ``tfr[…, ::decim]``.
    n_jobs : int
        Number of connections to compute in parallel. Memory mapping must be
        activated. Please see the Notes section for details.
    %(verbose)s

    Returns
    -------
    con : instance of Connectivity | list
        Computed connectivity measure(s). An instance of
        :class:`EpochSpectralConnectivity`, :class:`SpectralConnectivity`
        or a list of instances corresponding to connectivity measures if
        several connectivity measures are specified.
        The shape of each connectivity dataset is
        (n_epochs, n_signals, n_signals, n_freqs) when ``indices`` is `None`
        and (n_epochs, n_nodes, n_nodes, n_freqs) when ``indices`` is specified
        and ``n_nodes = len(indices[0])``.

    See Also
    --------
    mne_connectivity.spectral_connectivity_epochs
    mne_connectivity.SpectralConnectivity
    mne_connectivity.EpochSpectralConnectivity

    Notes
    -----
    Please note that the interpretation of the measures in this function
    depends on the data and underlying assumptions and does not necessarily
    reflect a causal relationship between brain regions.

    The connectivity measures are computed over time within each epoch and
    optionally averaged over epochs. High connectivity values indicate that
    the phase coupling (interpreted as estimated connectivity) differences
    between signals stay consistent over time.

    The spectral densities can be estimated using a multitaper method with
    digital prolate spheroidal sequence (DPSS) windows, or a continuous wavelet
    transform using Morlet wavelets. The spectral estimation mode is specified
    using the ``mode`` parameter.

    When using the multitaper spectral estimation method, the
    cross-spectral density is computed separately for each taper and aggregated
    using a weighted average, where the weights correspond to the concentration
    ratios between the DPSS windows.

    By default, the connectivity between all signals is computed (only
    connections corresponding to the lower-triangular part of the
    connectivity matrix). If one is only interested in the connectivity
    between some signals, the ``indices`` parameter can be used. For example,
    to compute the connectivity between the signal with index 0 and signals
    2, 3, 4 (a total of 3 connections), one can use the following::

        indices = (np.array([0, 0, 0]),    # row indices
                   np.array([2, 3, 4]))    # col indices

        con = spectral_connectivity_time(data, method='coh',
                                         indices=indices, ...)

    In this case ``con.get_data().shape = (3, n_freqs)``. The connectivity
    scores are in the same order as defined indices.

    **Supported Connectivity Measures**

    The connectivity method(s) is specified using the ``method`` parameter. The
    following methods are supported (note: ``E[]`` denotes average over
    epochs). Multiple measures can be computed at once by using a list/tuple,
    e.g., ``['coh', 'pli']`` to compute coherence and PLI.

        'coh' : Coherence given by::

                     | E[Sxy] |
            C = ---------------------
                sqrt(E[Sxx] * E[Syy])

        'plv' : Phase-Locking Value (PLV) :footcite:`LachauxEtAl1999` given
        by::

            PLV = |E[Sxy/|Sxy|]|

        'ciplv' : Corrected imaginary PLV (icPLV)
        :footcite:`BrunaEtAl2018` given by::

                             |E[Im(Sxy/|Sxy|)]|
            ciPLV = ------------------------------------
                     sqrt(1 - |E[real(Sxy/|Sxy|)]| ** 2)

        'sxy' : Cross spectrum Sxy

        'pli' : Phase Lag Index (PLI) :footcite:`StamEtAl2007` given by::

            PLI = |E[sign(Im(Sxy))]|

        'wpli' : Weighted Phase Lag Index (WPLI) :footcite:`VinckEtAl2011`
        given by::

                      |E[Im(Sxy)]|
            WPLI = ------------------
                      E[|Im(Sxy)|]

    Parallel computation can be activated by setting the ``n_jobs`` parameter.
    Under the hood, this utilizes the ``joblib`` library. For effective
    parallelization, you should activate memory mapping in MNE-Python by
    setting ``MNE_MEMMAP_MIN_SIZE`` and ``MNE_CACHE_DIR``. Activating memory
    mapping will make ``joblib`` store arrays greater than the minimum size on
    disc, and forego direct RAM access for more efficient processing.
    For example, in your code, run

        mne.set_config('MNE_MEMMAP_MIN_SIZE', '10M')
        mne.set_config('MNE_CACHE_DIR', '/dev/shm')

    When ``MNE_MEMMAP_MIN_SIZE=None``, the underlying joblib implementation
    results in pickling and unpickling the whole array each time a pair of
    indices is accessed, which is slow, compared to memory mapping the array.

    This function is based on the ``frites.conn.conn_spec`` implementation in
    Frites.

    .. versionadded:: 0.3

    References
    ----------
    .. footbibliography::
    """
    events = None
    event_id = None
    # extract data from Epochs object
    if isinstance(data, BaseEpochs):
        names = data.ch_names
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
        names = np.arange(0, n_signals)
        metadata = None
        if sfreq is None:
            raise ValueError('Sampling frequency (sfreq) is required with '
                             'array input.')

    # check that method is a list
    if isinstance(method, str):
        method = [method]

    # check that fmin corresponds to at least 5 cycles
    dur = float(n_times) / sfreq
    five_cycle_freq = 5. / dur
    if fmin is None:
        # use the 5 cycle freq. as default
        fmin = five_cycle_freq
        logger.info(f'Fmin was not specified. Using fmin={fmin:.2f}, which '
                    'corresponds to at least five cycles.')
    else:
        if np.any(fmin < five_cycle_freq):
            warn('fmin=%0.3f Hz corresponds to %0.3f < 5 cycles '
                 'based on the epoch length %0.3f sec, need at least %0.3f '
                 'sec epochs or fmin=%0.3f. Spectrum estimate will be '
                 'unreliable.' % (np.min(fmin), dur * np.min(fmin), dur,
                                  5. / np.min(fmin), five_cycle_freq))
    if fmax is None:
        fmax = sfreq / 2
        logger.info(f'Fmax was not specified. Using fmax={fmax:.2f}, which '
                    f'corresponds to Nyquist.')

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
        sm_times = int(np.round(sm_times / decim))
        sm_times = max(sm_times, 1)

    # Create smoothing kernel
    kernel = _create_kernel(sm_times, sm_freqs, kernel=sm_kernel)

    # get indices of pairs of (group) regions
    if indices is None:
        indices_use = np.tril_indices(n_signals, k=-1)
    else:
        indices_use = check_indices(indices)
    source_idx = indices_use[0]
    target_idx = indices_use[1]
    n_pairs = len(source_idx)

    # check freqs
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

    # compute frequencies to analyze based on number of samples,
    # sampling rate, specified wavelet frequencies and mode
    freqs = _compute_freqs(n_times, sfreq, freqs, mode)

    # compute the mask based on specified min/max and decimation factor
    freq_mask = _compute_freq_mask(freqs, fmin, fmax, fskip)

    # the frequency points where we compute connectivity
    freqs = freqs[freq_mask]

    # compute central frequencies
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

    conn = dict()
    for m in method:
        conn[m] = np.zeros((n_epochs, n_pairs,  n_freqs))
    logger.info('Connectivity computation...')

    # parameters to pass to the connectivity function
    call_params = dict(
        method=method, kernel=kernel, foi_idx=foi_idx,
        source_idx=source_idx, target_idx=target_idx,
        mode=mode, sfreq=sfreq, freqs=freqs, faverage=faverage,
        n_cycles=n_cycles, mt_bandwidth=mt_bandwidth,
        decim=decim, padding=padding, kw_cwt={}, kw_mt={}, n_jobs=n_jobs,
        verbose=verbose)

    for epoch_idx in np.arange(n_epochs):
        logger.info(f'   Processing epoch {epoch_idx+1} / {n_epochs} ...')
        conn_tr = _spectral_connectivity(data[epoch_idx], **call_params)
        for m in method:
            conn[m][epoch_idx] = np.stack(conn_tr[m], axis=0)

    if indices is None:
        conn_flat = conn
        conn = dict()
        for m in method:
            this_conn = np.zeros((n_epochs, n_signals, n_signals) +
                                 conn_flat[m].shape[2:],
                                 dtype=conn_flat[m].dtype)
            this_conn[:, source_idx, target_idx] = conn_flat[m]
            this_conn = this_conn.reshape((n_epochs, n_signals ** 2,) +
                                          conn_flat[m].shape[2:])
            conn[m] = this_conn

    # create a Connectivity container
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

    logger.info('[Connectivity computation done]')

    # return the object instead of list of length one
    if len(out) == 1:
        return out[0]
    else:
        return out


def _spectral_connectivity(data, method, kernel, foi_idx,
                           source_idx, target_idx,
                           mode, sfreq, freqs, faverage, n_cycles,
                           mt_bandwidth, decim, padding, kw_cwt, kw_mt,
                           n_jobs, verbose):
    """Estimate time-resolved connectivity for one epoch.

    Data is of shape (n_channels, n_times)."""

    n_pairs = len(source_idx)
    data = np.expand_dims(data, axis=0)
    if mode == 'cwt_morlet':
        out = tfr_array_morlet(
            data, sfreq, freqs, n_cycles=n_cycles, output='complex',
            decim=decim, n_jobs=n_jobs, **kw_cwt)
        out = np.expand_dims(out, axis=2)  # same dims with multitaper
        weights = None
    elif mode == 'multitaper':
        out = tfr_array_multitaper(
            data, sfreq, freqs, n_cycles=n_cycles,
            time_bandwidth=mt_bandwidth, output='complex', decim=decim,
            n_jobs=n_jobs, **kw_mt)
        if isinstance(n_cycles, (int, float)):
            n_cycles = [n_cycles] * len(freqs)
        mt_bandwidth = mt_bandwidth if mt_bandwidth else 4
        n_tapers = int(np.floor(mt_bandwidth - 1))
        weights = np.zeros((n_tapers, len(freqs), out.shape[-1]))
        for i, (f, n_c) in enumerate(zip(freqs, n_cycles)):
            window_length = np.arange(0., n_c / float(f), 1.0 / sfreq).shape[0]
            half_nbw = mt_bandwidth / 2.
            n_tapers = int(np.floor(mt_bandwidth - 1))
            _, eigvals = dpss_windows(window_length, half_nbw, n_tapers)
            weights[:, i, :] = np.sqrt(eigvals[:, np.newaxis])
            # weights have shape (n_tapers, n_freqs, n_times)
    else:
        raise ValueError("Mode must be 'cwt_morlet' or 'multitaper'.")

    out = np.squeeze(out, axis=0)

    if padding:
        pad_idx = int(np.floor(padding * sfreq / decim))
        out = out[..., pad_idx:-pad_idx]
        weights = weights[..., pad_idx:-pad_idx]

    # compute for each connectivity method
    this_conn = {}
    conn = _parallel_con(out, method, kernel, foi_idx, source_idx, target_idx,
                         n_jobs, verbose, n_pairs, faverage, weights)
    for i, m in enumerate(method):
        this_conn[m] = [out[i] for out in conn]

    return this_conn


###############################################################################
###############################################################################
#                               TIME-RESOLVED CORE FUNCTIONS
###############################################################################
###############################################################################

def _parallel_con(w, method, kernel, foi_idx, source_idx, target_idx, n_jobs,
                  verbose, total, faverage, weights):
    """Compute spectral connectivity in parallel.

    Input signal w is of shape (n_chans, n_tapers, n_freqs, n_times)."""

    if 'coh' in method:
        # psd
        if weights is not None:
            psd = weights * w
            psd = psd * np.conj(psd)
            psd = psd.real.sum(axis=1)
            psd = psd * 2 / (weights * weights.conj()).real.sum(axis=0)
        else:
            psd = w.real ** 2 + w.imag ** 2
            psd = np.squeeze(psd, axis=1)

        # smooth
        psd = _smooth_spectra(psd, kernel)

    def pairwise_con(w_x, w_y):
        # csd
        if weights is not None:
            s_xy = np.sum(weights * w[w_x] * np.conj(weights * w[w_y]), axis=0)
            s_xy = s_xy * 2 / (weights * np.conj(weights)).real.sum(axis=0)
        else:
            s_xy = w[w_x] * np.conj(w[w_y])
            s_xy = np.squeeze(s_xy, axis=0)

        dphi = s_xy / np.abs(s_xy)
        dphi = _smooth_spectra(dphi, kernel)
        s_xy = _smooth_spectra(s_xy, kernel)
        out = []
        for m in method:
            if m == 'coh':
                s_xx = psd[w_x]
                s_yy = psd[w_y]
                coh = np.abs(s_xy.mean(axis=-1, keepdims=True)) / \
                    np.sqrt(s_xx.mean(axis=-1, keepdims=True) *
                            s_yy.mean(axis=-1, keepdims=True))
                out.append(coh)

            if m == 'plv':
                dphi_mean = dphi.mean(axis=-1, keepdims=True)
                plv = np.abs(dphi_mean)
                out.append(plv)

            if m == 'ciplv':
                rplv = np.abs(np.mean(np.real(dphi), axis=-1, keepdims=True))
                iplv = np.abs(np.mean(np.imag(dphi), axis=-1, keepdims=True))
                ciplv = iplv / (np.sqrt(1 - rplv ** 2))
                out.append(ciplv)

            if m == 'pli':
                pli = np.abs(np.mean(np.sign(np.imag(s_xy)),
                                     axis=-1, keepdims=True))
                out.append(pli)

            if m == 'wpli':
                con_num = np.abs(s_xy.imag.mean(axis=-1, keepdims=True))
                con_den = np.mean(np.abs(s_xy.imag), axis=-1, keepdims=True)
                wpli = con_num / con_den
                out.append(wpli)

            if m == 'cs':
                out.append(s_xy)

        for i, _ in enumerate(out):
            # mean inside frequency sliding window (if needed)
            if isinstance(foi_idx, np.ndarray) and faverage:
                out[i] = _foi_average(out[i], foi_idx)
            # squeeze time dimension
            out[i] = out[i].squeeze(axis=-1)

        return out

    # only show progress if verbosity level is DEBUG
    if verbose != 'DEBUG' and verbose != 'debug' and verbose != 10:
        total = None

    # define the function to compute in parallel
    parallel, p_fun, n_jobs = parallel_func(
        pairwise_con, n_jobs=n_jobs, verbose=verbose, total=total)

    return parallel(p_fun(s, t) for s, t in zip(source_idx, target_idx))


def _compute_csd(x, y, weights):
    """Compute cross spectral density of signals x and y."""
    if weights is not None:
        s_xy = np.sum(weights * x * np.conj(weights * y), axis=-3)
        s_xy = s_xy * 2 / (weights * np.conj(weights)).real.sum(axis=-3)
    else:
        s_xy = x * np.conj(y)
        s_xy = np.squeeze(s_xy, axis=-3)
    return s_xy


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
        f_e += 1 if f_s == f_e else f_e
        conn_f[..., n_f, :] = conn[..., f_s:f_e, :].mean(-2)
    return conn_f


def _compute_freqs(n_times, sfreq, freqs, mode):
    from scipy.fft import rfftfreq
    # get frequencies of interest for the different modes
    if freqs is not None:
        if any(freqs > (sfreq / 2.)):
            raise ValueError('entries in freqs cannot be '
                             'larger than Nyquist (sfreq / 2)')
        else:
            return freqs.astype(np.float64)
    if mode in ('multitaper', 'fourier'):
        # fmin fmax etc is only supported for these modes
        # decide which frequencies to keep
        return rfftfreq(n_times, 1. / sfreq)
    elif mode == 'cwt_morlet':
        # cwt_morlet mode
        if freqs is None:
            raise ValueError('define frequencies of interest using '
                             'cwt_freqs')
    else:
        raise ValueError('mode has an invalid value')
