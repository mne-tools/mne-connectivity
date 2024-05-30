# Authors: Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#          Denis A. Engemann <denis.engemann@gmail.com>
#          Adam Li <adam2392@gmail.com>
#          Thomas S. Binns <t.s.binns@outlook.com>
#
# License: BSD (3-clause)

import inspect
from functools import partial

import numpy as np
from mne.epochs import BaseEpochs
from mne.parallel import parallel_func
from mne.source_estimate import _BaseSourceEstimate
from mne.time_frequency.multitaper import (
    _compute_mt_params,
    _csd_from_mt,
    _mt_spectra,
    _psd_from_mt,
    _psd_from_mt_adaptive,
)
from mne.time_frequency.tfr import cwt, morlet
from mne.utils import _arange_div, _check_option, _time_mask, logger, verbose, warn

from ..base import SpectralConnectivity, SpectroTemporalConnectivity
from ..utils import _check_multivariate_indices, check_indices, fill_doc
from .epochs_bivariate import _CON_METHOD_MAP_BIVARIATE
from .epochs_multivariate import (
    _CON_METHOD_MAP_MULTIVARIATE,
    _check_rank_input,
    _gc_methods,
    _multivariate_methods,
)


def _compute_freqs(n_times, sfreq, cwt_freqs, mode):
    from scipy.fft import rfftfreq

    # get frequencies of interest for the different modes
    if mode in ("multitaper", "fourier"):
        # fmin fmax etc is only supported for these modes
        # decide which frequencies to keep
        freqs_all = rfftfreq(n_times, 1.0 / sfreq)
    elif mode == "cwt_morlet":
        # cwt_morlet mode
        if cwt_freqs is None:
            raise ValueError("define frequencies of interest using " "cwt_freqs")
        else:
            cwt_freqs = cwt_freqs.astype(np.float64)
        if any(cwt_freqs > (sfreq / 2.0)):
            raise ValueError(
                "entries in cwt_freqs cannot be " "larger than Nyquist (sfreq / 2)"
            )
        freqs_all = cwt_freqs
    else:
        raise ValueError("mode has an invalid value")

    return freqs_all


def _compute_freq_mask(freqs_all, fmin, fmax, fskip):
    # create a frequency mask for all bands
    freq_mask = np.zeros(len(freqs_all), dtype=bool)
    for f_lower, f_upper in zip(fmin, fmax):
        freq_mask |= (freqs_all >= f_lower) & (freqs_all <= f_upper)

    # possibly skip frequency points
    for pos in range(fskip):
        freq_mask[pos + 1 :: fskip + 1] = False
    return freq_mask


def _prepare_connectivity(
    epoch_block,
    times_in,
    tmin,
    tmax,
    fmin,
    fmax,
    sfreq,
    indices,
    method,
    mode,
    fskip,
    n_bands,
    cwt_freqs,
    faverage,
):
    """Check and precompute dimensions of results data."""
    first_epoch = epoch_block[0]

    # get the data size and time scale
    n_signals, n_times_in, times_in, warn_times = _get_and_verify_data_sizes(
        first_epoch, sfreq, times=times_in
    )

    n_times_in = len(times_in)

    if tmin is not None and tmin < times_in[0]:
        warn(
            "start time tmin=%0.2f s outside of the time scope of the data "
            "[%0.2f s, %0.2f s]" % (tmin, times_in[0], times_in[-1])
        )
    if tmax is not None and tmax > times_in[-1]:
        warn(
            "stop time tmax=%0.2f s outside of the time scope of the data "
            "[%0.2f s, %0.2f s]" % (tmax, times_in[0], times_in[-1])
        )

    mask = _time_mask(times_in, tmin, tmax, sfreq=sfreq)
    tmin_idx, tmax_idx = np.where(mask)[0][[0, -1]]
    tmax_idx += 1
    tmin_true = times_in[tmin_idx]
    tmax_true = times_in[tmax_idx - 1]  # time of last point used

    times = times_in[tmin_idx:tmax_idx]
    n_times = len(times)

    if any(this_method in _multivariate_methods for this_method in method):
        multivariate_con = True
    else:
        multivariate_con = False

    if indices is None:
        if multivariate_con:
            if any(this_method in _gc_methods for this_method in method):
                raise ValueError(
                    "indices must be specified when computing Granger "
                    "causality, as all-to-all connectivity is not supported"
                )
            else:
                logger.info("using all indices for multivariate connectivity")
                # indices expected to be a masked array, even if not ragged
                indices_use = (
                    np.arange(n_signals, dtype=int)[np.newaxis, :],
                    np.arange(n_signals, dtype=int)[np.newaxis, :],
                )
                indices_use = np.ma.masked_array(indices_use, mask=False, fill_value=-1)
        else:
            logger.info("only using indices for lower-triangular matrix")
            # only compute r for lower-triangular region
            indices_use = np.tril_indices(n_signals, -1)
    else:
        if multivariate_con:
            # pad ragged indices and mask the invalid entries
            indices_use = _check_multivariate_indices(indices, n_signals)
            if any(this_method in _gc_methods for this_method in method):
                for seed, target in zip(indices_use[0], indices_use[1]):
                    intersection = np.intersect1d(
                        seed.compressed(), target.compressed()
                    )
                    if intersection.size > 0:
                        raise ValueError(
                            "seed and target indices must not intersect when "
                            "computing Granger causality"
                        )
        else:
            indices_use = check_indices(indices)

    # number of connectivities to compute
    n_cons = len(indices_use[0])

    logger.info("    computing connectivity for %d connections" % n_cons)
    logger.info(
        "    using t=%0.3fs..%0.3fs for estimation (%d points)"
        % (tmin_true, tmax_true, n_times)
    )

    # check that fmin corresponds to at least 5 cycles
    dur = float(n_times) / sfreq
    five_cycle_freq = 5.0 / dur
    if len(fmin) == 1 and fmin[0] == -np.inf:
        # we use the 5 cycle freq. as default
        fmin = np.array([five_cycle_freq])
    else:
        if np.any(fmin < five_cycle_freq):
            warn(
                "fmin=%0.3f Hz corresponds to %0.3f < 5 cycles "
                "based on the epoch length %0.3f sec, need at least %0.3f "
                "sec epochs or fmin=%0.3f. Spectrum estimate will be "
                "unreliable."
                % (
                    np.min(fmin),
                    dur * np.min(fmin),
                    dur,
                    5.0 / np.min(fmin),
                    five_cycle_freq,
                )
            )

    # compute frequencies to analyze based on number of samples,
    # sampling rate, specified wavelet frequencies and mode
    freqs = _compute_freqs(n_times, sfreq, cwt_freqs, mode)

    # compute the mask based on specified min/max and decimation factor
    freq_mask = _compute_freq_mask(freqs, fmin, fmax, fskip)

    # the frequency points where we compute connectivity
    freqs = freqs[freq_mask]
    n_freqs = len(freqs)

    # get the freq. indices and points for each band
    freq_idx_bands = [
        np.where((freqs >= fl) & (freqs <= fu))[0] for fl, fu in zip(fmin, fmax)
    ]
    freqs_bands = [freqs[freq_idx] for freq_idx in freq_idx_bands]

    # make sure we don't have empty bands
    for i, n_f_band in enumerate([len(f) for f in freqs_bands]):
        if n_f_band == 0:
            raise ValueError(
                "There are no frequency points between "
                "%0.1fHz and %0.1fHz. Change the band "
                "specification (fmin, fmax) or the "
                "frequency resolution." % (fmin[i], fmax[i])
            )
    if n_bands == 1:
        logger.info(
            "    frequencies: %0.1fHz..%0.1fHz (%d points)"
            % (freqs_bands[0][0], freqs_bands[0][-1], n_freqs)
        )
    else:
        logger.info("    computing connectivity for the bands:")
        for i, bfreqs in enumerate(freqs_bands):
            logger.info(
                "     band %d: %0.1fHz..%0.1fHz "
                "(%d points)" % (i + 1, bfreqs[0], bfreqs[-1], len(bfreqs))
            )
    if faverage:
        logger.info("    connectivity scores will be averaged for " "each band")

    return (
        n_cons,
        times,
        n_times,
        times_in,
        n_times_in,
        tmin_idx,
        tmax_idx,
        n_freqs,
        freq_mask,
        freqs,
        freqs_bands,
        freq_idx_bands,
        n_signals,
        indices_use,
        warn_times,
    )


def _assemble_spectral_params(
    mode,
    n_times,
    mt_adaptive,
    mt_bandwidth,
    sfreq,
    mt_low_bias,
    cwt_n_cycles,
    cwt_freqs,
    freqs,
    freq_mask,
):
    """Prepare time-frequency decomposition."""
    spectral_params = dict(eigvals=None, window_fun=None, wavelets=None)
    n_tapers = None
    n_times_spectrum = 0
    if mode == "multitaper":
        window_fun, eigvals, mt_adaptive = _compute_mt_params(
            n_times, sfreq, mt_bandwidth, mt_low_bias, mt_adaptive
        )
        spectral_params.update(window_fun=window_fun, eigvals=eigvals)
    elif mode == "fourier":
        logger.info("    using FFT with a Hanning window to estimate " "spectra")
        spectral_params.update(window_fun=np.hanning(n_times), eigvals=1.0)
    elif mode == "cwt_morlet":
        logger.info("    using CWT with Morlet wavelets to estimate " "spectra")

        # reformat cwt_n_cycles if we have removed some frequencies
        # using fmin, fmax, fskip
        cwt_n_cycles = np.array((cwt_n_cycles,), dtype=float).ravel()
        if len(cwt_n_cycles) > 1:
            if len(cwt_n_cycles) != len(cwt_freqs):
                raise ValueError(
                    "cwt_n_cycles must be float or an "
                    "array with the same size as cwt_freqs"
                )
            cwt_n_cycles = cwt_n_cycles[freq_mask]

        # get the Morlet wavelets
        spectral_params.update(
            wavelets=morlet(sfreq, freqs, n_cycles=cwt_n_cycles, zero_mean=True)
        )
        n_times_spectrum = n_times
    else:
        raise ValueError("mode has an invalid value")
    return spectral_params, mt_adaptive, n_times_spectrum, n_tapers


########################################################################
# Connectivity estimators base class


class _AbstractConEstBase:
    """ABC for connectivity estimators."""

    def start_epoch(self):
        raise NotImplementedError("start_epoch method not implemented")

    def accumulate(self, con_idx, csd_xy):
        raise NotImplementedError("accumulate method not implemented")

    def combine(self, other):
        raise NotImplementedError("combine method not implemented")

    def compute_con(self, con_idx, n_epochs):
        raise NotImplementedError("compute_con method not implemented")


########################################################################


def _epoch_spectral_connectivity(
    data,
    sig_idx,
    tmin_idx,
    tmax_idx,
    sfreq,
    method,
    mode,
    window_fun,
    eigvals,
    wavelets,
    freq_mask,
    mt_adaptive,
    idx_map,
    n_cons,
    block_size,
    psd,
    accumulate_psd,
    con_method_types,
    con_methods,
    n_signals,
    n_signals_use,
    n_times,
    gc_n_lags,
    accumulate_inplace=True,
):
    """Estimate connectivity for one epoch (see spectral_connectivity)."""
    if any(this_method in _multivariate_methods for this_method in method):
        n_con_signals = n_signals_use**2
    else:
        n_con_signals = n_cons

    if wavelets is not None:
        n_times_spectrum = n_times
        n_freqs = len(wavelets)
    else:
        n_times_spectrum = 0
        n_freqs = np.sum(freq_mask)

    if not accumulate_inplace:
        # instantiate methods only for this epoch (used in parallel mode)
        con_methods = []
        for mtype in con_method_types:
            method_params = list(inspect.signature(mtype).parameters)
            if "n_signals" in method_params:
                # if it's a multivariate connectivity method
                if "n_lags" in method_params:
                    # if it's a Granger causality method
                    con_methods.append(
                        mtype(
                            n_signals_use, n_cons, n_freqs, n_times_spectrum, gc_n_lags
                        )
                    )
                else:
                    # if it's a coherence method
                    con_methods.append(
                        mtype(n_signals_use, n_cons, n_freqs, n_times_spectrum)
                    )
            else:
                con_methods.append(mtype(n_cons, n_freqs, n_times_spectrum))

    _check_option("mode", mode, ("cwt_morlet", "multitaper", "fourier"))
    if len(sig_idx) == n_signals:
        # we use all signals: use a slice for faster indexing
        sig_idx = slice(None, None)

    # compute tapered spectra
    x_t = list()
    this_psd = list()
    for this_data in data:
        if mode in ("multitaper", "fourier"):
            if isinstance(this_data, _BaseSourceEstimate):
                _mt_spectra_partial = partial(_mt_spectra, dpss=window_fun, sfreq=sfreq)
                this_x_t = this_data.transform_data(
                    _mt_spectra_partial,
                    idx=sig_idx,
                    tmin_idx=tmin_idx,
                    tmax_idx=tmax_idx,
                )
            else:
                this_x_t, _ = _mt_spectra(
                    this_data[sig_idx, tmin_idx:tmax_idx], window_fun, sfreq
                )

            if mt_adaptive:
                # compute PSD and adaptive weights
                _this_psd, weights = _psd_from_mt_adaptive(
                    this_x_t, eigvals, freq_mask, return_weights=True
                )

                # only keep freqs of interest
                this_x_t = this_x_t[:, :, freq_mask]
            else:
                # do not use adaptive weights
                this_x_t = this_x_t[:, :, freq_mask]
                if mode == "multitaper":
                    weights = np.sqrt(eigvals)[np.newaxis, :, np.newaxis]
                else:
                    # hack to so we can sum over axis=-2
                    weights = np.array([1.0])[:, None, None]

                if accumulate_psd:
                    _this_psd = _psd_from_mt(this_x_t, weights)
        else:  # mode == 'cwt_morlet'
            if isinstance(this_data, _BaseSourceEstimate):
                cwt_partial = partial(cwt, Ws=wavelets, use_fft=True, mode="same")
                this_x_t = this_data.transform_data(
                    cwt_partial, idx=sig_idx, tmin_idx=tmin_idx, tmax_idx=tmax_idx
                )
            else:
                this_x_t = cwt(
                    this_data[sig_idx, tmin_idx:tmax_idx],
                    wavelets,
                    use_fft=True,
                    mode="same",
                )
            _this_psd = (this_x_t * this_x_t.conj()).real

        x_t.append(this_x_t)
        if accumulate_psd:
            this_psd.append(_this_psd)

    x_t = np.concatenate(x_t, axis=0)
    if accumulate_psd:
        this_psd = np.concatenate(this_psd, axis=0)

    # accumulate or return psd
    if accumulate_psd:
        if accumulate_inplace:
            psd += this_psd
        else:
            psd = this_psd
    else:
        psd = None

    # tell the methods that a new epoch starts
    for method in con_methods:
        method.start_epoch()

    # accumulate connectivity scores
    if mode in ["multitaper", "fourier"]:
        for i in range(0, n_con_signals, block_size):
            n_extra = max(0, i + block_size - n_con_signals)
            con_idx = slice(i, i + block_size - n_extra)
            if mt_adaptive:
                csd = _csd_from_mt(
                    x_t[idx_map[0][con_idx]],
                    x_t[idx_map[1][con_idx]],
                    weights[idx_map[0][con_idx]],
                    weights[idx_map[1][con_idx]],
                )
            else:
                csd = _csd_from_mt(
                    x_t[idx_map[0][con_idx]], x_t[idx_map[1][con_idx]], weights, weights
                )

            for method in con_methods:
                method.accumulate(con_idx, csd)
    else:  # mode == 'cwt_morlet'  # reminder to add alternative TFR methods
        for i in range(0, n_con_signals, block_size):
            n_extra = max(0, i + block_size - n_con_signals)
            con_idx = slice(i, i + block_size - n_extra)
            # this codes can be very slow
            csd = x_t[idx_map[0][con_idx]] * x_t[idx_map[1][con_idx]].conjugate()

            for method in con_methods:
                method.accumulate(con_idx, csd)
                # future estimator types need to be explicitly handled here

    return con_methods, psd


def _get_n_epochs(epochs, n):
    """Generate lists with at most n epochs."""
    epochs_out = list()
    for epoch in epochs:
        if not isinstance(epoch, (list, tuple)):
            epoch = (epoch,)
        epochs_out.append(epoch)
        if len(epochs_out) >= n:
            yield epochs_out
            epochs_out = list()
    if 0 < len(epochs_out) < n:
        yield epochs_out


def _check_method(method):
    """Test if a method implements the required interface."""
    interface_members = [
        m[0]
        for m in inspect.getmembers(_AbstractConEstBase)
        if not m[0].startswith("_")
    ]
    method_members = [
        m[0] for m in inspect.getmembers(method) if not m[0].startswith("_")
    ]

    for member in interface_members:
        if member not in method_members:
            return False, member
    return True, None


def _get_and_verify_data_sizes(
    data, sfreq, n_signals=None, n_times=None, times=None, warn_times=True
):
    """Get and/or verify the data sizes and time scales."""
    if not isinstance(data, (list, tuple)):
        raise ValueError("data has to be a list or tuple")
    n_signals_tot = 0
    # Sometimes data can be (ndarray, SourceEstimate) groups so in the case
    # where ndarray comes first, don't use it for times
    times_inferred = False
    for this_data in data:
        this_n_signals, this_n_times = this_data.shape
        if n_times is not None:
            if this_n_times != n_times:
                raise ValueError(
                    "all input time series must have the same " "number of time points"
                )
        else:
            n_times = this_n_times
        n_signals_tot += this_n_signals

        if hasattr(this_data, "times"):
            assert isinstance(this_data, _BaseSourceEstimate)
            this_times = this_data.times
            if times is not None and not times_inferred:
                if warn_times and not np.allclose(times, this_times):
                    with np.printoptions(threshold=4, linewidth=120):
                        warn(
                            "time scales of input time series do not match:\n"
                            f"{this_times}\n{times}"
                        )
                    warn_times = False
            else:
                times = this_times
        elif times is None:
            times_inferred = True
            times = _arange_div(n_times, sfreq)

    if n_signals is not None:
        if n_signals != n_signals_tot:
            raise ValueError(
                "the number of time series has to be the same in " "each epoch"
            )
    n_signals = n_signals_tot

    return n_signals, n_times, times, warn_times


# map names to estimator types
_CON_METHOD_MAP = {**_CON_METHOD_MAP_BIVARIATE, **_CON_METHOD_MAP_MULTIVARIATE}


def _check_estimators(method):
    """Check construction of connectivity estimators."""
    n_methods = len(method)
    con_method_types = list()
    for this_method in method:
        if this_method in _CON_METHOD_MAP:
            con_method_types.append(_CON_METHOD_MAP[this_method])
        elif isinstance(this_method, str):
            raise ValueError("%s is not a valid connectivity method" % this_method)
        else:
            # support for custom class
            method_valid, msg = _check_method(this_method)
            if not method_valid:
                raise ValueError(
                    "The supplied connectivity method does "
                    "not have the method %s" % msg
                )
            con_method_types.append(this_method)

    # if none of the comp_con functions needs the PSD, we don't estimate it
    accumulate_psd = any(this_method.accumulate_psd for this_method in con_method_types)

    return con_method_types, n_methods, accumulate_psd


@verbose
@fill_doc
def spectral_connectivity_epochs(
    data,
    names=None,
    method="coh",
    indices=None,
    sfreq=None,
    mode="multitaper",
    fmin=None,
    fmax=np.inf,
    fskip=0,
    faverage=False,
    tmin=None,
    tmax=None,
    mt_bandwidth=None,
    mt_adaptive=False,
    mt_low_bias=True,
    cwt_freqs=None,
    cwt_n_cycles=7,
    gc_n_lags=40,
    rank=None,
    block_size=1000,
    n_jobs=1,
    verbose=None,
):
    r"""Compute frequency- and time-frequency-domain connectivity measures.

    The connectivity method(s) are specified using the "method" parameter.
    All methods are based on estimates of the cross- and power spectral
    densities (CSD/PSD) Sxy and Sxx, Syy.

    Parameters
    ----------
    data : array-like, shape=(n_epochs, n_signals, n_times) | Epochs
        The data from which to compute connectivity. Note that it is also
        possible to combine multiple signals by providing a list of tuples,
        e.g., data = [(arr_0, stc_0), (arr_1, stc_1), (arr_2, stc_2)],
        corresponds to 3 epochs, and arr_* could be an array with the same
        number of time points as stc_*. The array-like object can also
        be a list/generator of array, shape =(n_signals, n_times),
        or a list/generator of SourceEstimate or VolSourceEstimate objects.
    %(names)s
    method : str | list of str
        Connectivity measure(s) to compute. These can be ``['coh', 'cohy',
        'imcoh', 'cacoh', 'mic', 'mim', 'plv', 'ciplv', 'ppc', 'pli', 'dpli',
        'wpli', 'wpli2_debiased', 'gc', 'gc_tr']``. These are:

        * %(coh)s
        * %(cohy)s
        * %(imcoh)s
        * %(cacoh)s
        * %(mic)s
        * %(mim)s
        * %(plv)s
        * %(ciplv)s
        * %(ppc)s
        * %(pli)s
        * %(pli2_unbiased)s
        * %(dpli)s
        * %(wpli)s
        * %(wpli2_debiased)s
        * %(gc)s
        * %(gc_tr)s

        Multivariate methods (``['cacoh', 'mic', 'mim', 'gc', 'gc_tr']``)
        cannot be called with the other methods.
    indices : tuple of array | None
        Two arrays with indices of connections for which to compute
        connectivity. If a bivariate method is called, each array for the seeds
        and targets should contain the channel indices for each bivariate
        connection. If a multivariate method is called, each array for the
        seeds and targets should consist of nested arrays containing
        the channel indices for each multivariate connection. If ``None``,
        connections between all channels are computed, unless a Granger
        causality method is called, in which case an error is raised.
    sfreq : float
        The sampling frequency. Required if data is not
        :class:`Epochs <mne.Epochs>`.
    mode : str
        Spectrum estimation mode can be either: 'multitaper', 'fourier', or
        'cwt_morlet'.
    fmin : float | tuple of float
        The lower frequency of interest. Multiple bands are defined using
        a tuple, e.g., (8., 20.) for two bands with 8Hz and 20Hz lower freq.
    fmax : float | tuple of float
        The upper frequency of interest. Multiple bands are dedined using
        a tuple, e.g. (13., 30.) for two band with 13Hz and 30Hz upper freq.
    fskip : int
        Omit every "(fskip + 1)-th" frequency bin to decimate in frequency
        domain.
    faverage : bool
        Average connectivity scores for each frequency band. If True,
        the output freqs will be a list with arrays of the frequencies
        that were averaged.
    tmin : float | None
        Time to start connectivity estimation. Note: when "data" is an array,
        the first sample is assumed to be at time 0. For other types
        (Epochs, etc.), the time information contained in the object is used
        to compute the time indices.
    tmax : float | None
        Time to end connectivity estimation. Note: when "data" is an array,
        the first sample is assumed to be at time 0. For other types
        (Epochs, etc.), the time information contained in the object is used
        to compute the time indices.
    mt_bandwidth : float | None
        The bandwidth of the multitaper windowing function in Hz.
        Only used in 'multitaper' mode.
    mt_adaptive : bool
        Use adaptive weights to combine the tapered spectra into PSD.
        Only used in 'multitaper' mode.
    mt_low_bias : bool
        Only use tapers with more than 90 percent spectral concentration
        within bandwidth. Only used in 'multitaper' mode.
    cwt_freqs : array
        Array of frequencies of interest. Only used in 'cwt_morlet' mode.
    cwt_n_cycles : float | array of float
        Number of cycles. Fixed number or one per frequency. Only used in
        'cwt_morlet' mode.
    gc_n_lags : int
        Number of lags to use for the vector autoregressive model when
        computing Granger causality. Higher values increase computational cost,
        but reduce the degree of spectral smoothing in the results. Only used
        if ``method`` contains any of ``['gc', 'gc_tr']``.
    rank : tuple of array | None
        Two arrays with the rank to project the seed and target data to,
        respectively, using singular value decomposition. If None, the rank of
        the data is computed and projected to. Only used if ``method`` contains
        any of ``['cacoh', 'mic', 'mim', 'gc', 'gc_tr']``.
    block_size : int
        How many connections to compute at once (higher numbers are faster
        but require more memory).
    n_jobs : int
        How many samples to process in parallel.
    %(verbose)s

    Returns
    -------
    con : array | list of array
        Computed connectivity measure(s). Either an instance of
        ``SpectralConnectivity`` or ``SpectroTemporalConnectivity``.
        The shape of the connectivity result will be:

        - ``(n_cons, n_freqs)`` for multitaper or fourier modes
        - ``(n_cons, n_freqs, n_times)`` for cwt_morlet mode
        - ``n_cons = n_signals ** 2`` for bivariate methods with
          ``indices=None``
        - ``n_cons = 1`` for multivariate methods with ``indices=None``
        - ``n_cons = len(indices[0])`` for bivariate and multivariate methods
          when indices is supplied.

    See Also
    --------
    mne_connectivity.spectral_connectivity_time
    mne_connectivity.SpectralConnectivity
    mne_connectivity.SpectroTemporalConnectivity

    Notes
    -----
    Please note that the interpretation of the measures in this function
    depends on the data and underlying assumptions and does not necessarily
    reflect a causal relationship between brain regions.

    These measures are not to be interpreted over time. Each Epoch passed into
    the dataset is interpreted as an independent sample of the same
    connectivity structure. Within each Epoch, it is assumed that the spectral
    measure is stationary. The spectral measures implemented in this function
    are computed across Epochs. **Thus, spectral measures computed with only
    one Epoch will result in errorful values and spectral measures computed
    with few Epochs will be unreliable.** Please see
    ``spectral_connectivity_time`` for time-resolved connectivity estimation.

    The spectral densities can be estimated using a multitaper method with
    digital prolate spheroidal sequence (DPSS) windows, a discrete Fourier
    transform with Hanning windows, or a continuous wavelet transform using
    Morlet wavelets. The spectral estimation mode is specified using the
    "mode" parameter.

    By default, the connectivity between all signals is computed (only
    connections corresponding to the lower-triangular part of the connectivity
    matrix). If one is only interested in the connectivity between some
    signals, the "indices" parameter can be used. For example, to compute the
    connectivity between the signal with index 0 and signals "2, 3, 4" (a total
    of 3 connections) one can use the following::

        indices = (np.array([0, 0, 0]),    # row indices
                   np.array([2, 3, 4]))    # col indices

        con = spectral_connectivity_epochs(data, method='coh',
                                           indices=indices, ...)

    In this case con.get_data().shape = (3, n_freqs). The connectivity scores
    are in the same order as defined indices.

    For multivariate methods, this is handled differently. If "indices" is
    None, connectivity between all signals will be computed and a single
    connectivity spectrum will be returned (this is not possible if a Granger
    causality method is called). If "indices" is specified, seed and target
    indices for each connection should be specified as nested array-likes. For
    example, to compute the connectivity between signals (0, 1) -> (2, 3) and
    (0, 1) -> (4, 5), indices should be specified as::

        indices = (np.array([[0, 1], [0, 1]]),  # seeds
                   np.array([[2, 3], [4, 5]]))  # targets

    More information on working with multivariate indices and handling
    connections where the number of seeds and targets are not equal can be
    found in the :doc:`../auto_examples/handling_ragged_arrays` example.

    **Supported Connectivity Measures**

    The connectivity method(s) is specified using the "method" parameter. The
    following methods are supported (note: ``E[]`` denotes average over
    epochs). Multiple measures can be computed at once by using a list/tuple,
    e.g., ``['coh', 'pli']`` to compute coherence and PLI.

        'coh' : Coherence given by::

                     | E[Sxy] |
            C = ---------------------
                sqrt(E[Sxx] * E[Syy])

        'cohy' : Coherency given by::

                       E[Sxy]
            C = ---------------------
                sqrt(E[Sxx] * E[Syy])

        'imcoh' : Imaginary coherence :footcite:`NolteEtAl2004` given by::

                      Im(E[Sxy])
            C = ----------------------
                sqrt(E[Sxx] * E[Syy])

        'cacoh' : Canonical Coherency (CaCoh) :footcite:`VidaurreEtAl2019`
        given by:

            :math:`\textrm{CaCoh}=\Large{\frac{\boldsymbol{a}^T\boldsymbol{D}
            (\Phi)\boldsymbol{b}}{\sqrt{\boldsymbol{a}^T\boldsymbol{a}
            \boldsymbol{b}^T\boldsymbol{b}}}}`

            where: :math:`\boldsymbol{D}(\Phi)` is the cross-spectral density
            between seeds and targets transformed for a given phase angle
            :math:`\Phi`; and :math:`\boldsymbol{a}` and :math:`\boldsymbol{b}`
            are eigenvectors for the seeds and targets, such that
            :math:`\boldsymbol{a}^T\boldsymbol{D}(\Phi)\boldsymbol{b}`
            maximises coherency between the seeds and targets. Taking the
            absolute value of the results gives maximised coherence.

        'mic' : Maximised Imaginary part of Coherency (MIC)
        :footcite:`EwaldEtAl2012` given by:

            :math:`\textrm{MIC}=\Large{\frac{\boldsymbol{\alpha}^T
            \boldsymbol{E \beta}}{\parallel\boldsymbol{\alpha}\parallel
            \parallel\boldsymbol{\beta}\parallel}}`

            where: :math:`\boldsymbol{E}` is the imaginary part of the
            transformed cross-spectral density between seeds and targets; and
            :math:`\boldsymbol{\alpha}` and :math:`\boldsymbol{\beta}` are
            eigenvectors for the seeds and targets, such that
            :math:`\boldsymbol{\alpha}^T \boldsymbol{E \beta}` maximises the
            imaginary part of coherency between the seeds and targets.

        'mim' : Multivariate Interaction Measure (MIM)
        :footcite:`EwaldEtAl2012` given by:

            :math:`\textrm{MIM}=tr(\boldsymbol{EE}^T)`

            where :math:`\boldsymbol{E}` is the imaginary part of the
            transformed cross-spectral density between seeds and targets.

        'plv' : Phase-Locking Value (PLV) :footcite:`LachauxEtAl1999` given
        by::

            PLV = |E[Sxy/|Sxy|]|

        'ciplv' : corrected imaginary PLV (ciPLV)
        :footcite:`BrunaEtAl2018` given by::

                             |E[Im(Sxy/|Sxy|)]|
            ciPLV = ------------------------------------
                     sqrt(1 - |E[real(Sxy/|Sxy|)]| ** 2)

        'ppc' : Pairwise Phase Consistency (PPC), an unbiased estimator
        of squared PLV :footcite:`VinckEtAl2010`.

        'pli' : Phase Lag Index (PLI) :footcite:`StamEtAl2007` given by::

            PLI = |E[sign(Im(Sxy))]|

        'pli2_unbiased' : Unbiased estimator of squared PLI
        :footcite:`VinckEtAl2011`.

        'dpli' : Directed Phase Lag Index (DPLI) :footcite:`StamEtAl2012`
        given by (where H is the Heaviside function)::

            DPLI = E[H(Im(Sxy))]

        'wpli' : Weighted Phase Lag Index (WPLI) :footcite:`VinckEtAl2011`
        given by::

                      |E[Im(Sxy)]|
            WPLI = ------------------
                      E[|Im(Sxy)|]

        'wpli2_debiased' : Debiased estimator of squared WPLI
        :footcite:`VinckEtAl2011`.

        'gc' : State-space Granger Causality (GC) :footcite:`BarnettSeth2015`
        given by:

            :math:`GC = ln\Large{(\frac{\lvert\boldsymbol{S}_{tt}\rvert}{\lvert
            \boldsymbol{S}_{tt}-\boldsymbol{H}_{ts}\boldsymbol{\Sigma}_{ss
            \lvert t}\boldsymbol{H}_{ts}^*\rvert}})`

            where: :math:`s` and :math:`t` represent the seeds and targets,
            respectively; :math:`\boldsymbol{H}` is the spectral transfer
            function; :math:`\boldsymbol{\Sigma}` is the residuals matrix of
            the autoregressive model; and :math:`\boldsymbol{S}` is
            :math:`\boldsymbol{\Sigma}` transformed by :math:`\boldsymbol{H}`.

        'gc_tr' : State-space GC on time-reversed signals
        :footcite:`BarnettSeth2015,WinklerEtAl2016` given by the same equation
        as for 'gc', but where the autocovariance sequence from which the
        autoregressive model is produced is transposed to mimic the reversal of
        the original signal in time :footcite:`HaufeEtAl2012`.

    References
    ----------
    .. footbibliography::
    """
    if n_jobs != 1:
        parallel, my_epoch_spectral_connectivity, n_jobs = parallel_func(
            _epoch_spectral_connectivity, n_jobs, verbose=verbose
        )

    # format fmin and fmax and check inputs
    if fmin is None:
        fmin = -np.inf  # set it to -inf, so we can adjust it later

    fmin = np.array((fmin,), dtype=float).ravel()
    fmax = np.array((fmax,), dtype=float).ravel()
    if len(fmin) != len(fmax):
        raise ValueError("fmin and fmax must have the same length")
    if np.any(fmin > fmax):
        raise ValueError("fmax must be larger than fmin")

    n_bands = len(fmin)

    # assign names to connectivity methods
    if not isinstance(method, (list, tuple)):
        method = [method]  # make it a list so we can iterate over it

    if n_bands != 1 and any(this_method in _gc_methods for this_method in method):
        raise ValueError(
            "computing Granger causality on multiple frequency "
            "bands is not yet supported"
        )

    if any(this_method in _multivariate_methods for this_method in method):
        if not all(this_method in _multivariate_methods for this_method in method):
            raise ValueError(
                "bivariate and multivariate connectivity methods cannot be "
                "used in the same function call"
            )
        multivariate_con = True
    else:
        multivariate_con = False

    # handle connectivity estimators
    (con_method_types, n_methods, accumulate_psd) = _check_estimators(method)

    events = None
    event_id = None
    if isinstance(data, BaseEpochs):
        names = data.ch_names
        times_in = data.times  # input times for Epochs input type
        sfreq = data.info["sfreq"]

        events = data.events
        event_id = data.event_id

        # Extract metadata from the Epochs data structure.
        # Make Annotations persist through by adding them to the metadata.
        metadata = data.metadata
        if metadata is None:
            annots_in_metadata = False
        else:
            annots_in_metadata = all(
                name not in metadata.columns
                for name in ["annot_onset", "annot_duration", "annot_description"]
            )
        if hasattr(data, "annotations") and not annots_in_metadata:
            data.add_annotations_to_metadata(overwrite=True)
        metadata = data.metadata
    else:
        times_in = None
        metadata = None
        if sfreq is None:
            raise ValueError(
                "Sampling frequency (sfreq) is required with " "array input."
            )

    # loop over data; it could be a generator that returns
    # (n_signals x n_times) arrays or SourceEstimates
    epoch_idx = 0
    logger.info("Connectivity computation...")
    warn_times = True
    for epoch_block in _get_n_epochs(data, n_jobs):
        if epoch_idx == 0:
            # initialize everything times and frequencies
            (
                n_cons,
                times,
                n_times,
                times_in,
                n_times_in,
                tmin_idx,
                tmax_idx,
                n_freqs,
                freq_mask,
                freqs,
                freqs_bands,
                freq_idx_bands,
                n_signals,
                indices_use,
                warn_times,
            ) = _prepare_connectivity(
                epoch_block=epoch_block,
                times_in=times_in,
                tmin=tmin,
                tmax=tmax,
                fmin=fmin,
                fmax=fmax,
                sfreq=sfreq,
                indices=indices,
                method=method,
                mode=mode,
                fskip=fskip,
                n_bands=n_bands,
                cwt_freqs=cwt_freqs,
                faverage=faverage,
            )

            # check rank input and compute data ranks if necessary
            if multivariate_con:
                rank = _check_rank_input(rank, data, indices_use)
            else:
                rank = None
                gc_n_lags = None

            # make sure padded indices are stored in the connectivity object
            if multivariate_con and indices is not None:
                # create a copy so that `indices_use` can be modified
                indices = (indices_use[0].copy(), indices_use[1].copy())

            # get the window function, wavelets, etc for different modes
            (
                spectral_params,
                mt_adaptive,
                n_times_spectrum,
                n_tapers,
            ) = _assemble_spectral_params(
                mode=mode,
                n_times=n_times,
                mt_adaptive=mt_adaptive,
                mt_bandwidth=mt_bandwidth,
                sfreq=sfreq,
                mt_low_bias=mt_low_bias,
                cwt_n_cycles=cwt_n_cycles,
                cwt_freqs=cwt_freqs,
                freqs=freqs,
                freq_mask=freq_mask,
            )

            # unique signals for which we actually need to compute PSD etc.
            if multivariate_con:
                sig_idx = np.unique(indices_use.compressed())
                remapping = {ch_i: sig_i for sig_i, ch_i in enumerate(sig_idx)}
                remapped_inds = indices_use.copy()
                for idx in sig_idx:
                    remapped_inds[indices_use == idx] = remapping[idx]
                remapped_sig = np.unique(remapped_inds.compressed())
            else:
                sig_idx = np.unique(np.r_[indices_use[0], indices_use[1]])
            n_signals_use = len(sig_idx)

            # map indices to unique indices
            if multivariate_con:
                indices_use = remapped_inds  # use remapped seeds & targets
                idx_map = [
                    np.sort(np.repeat(remapped_sig, len(sig_idx))),
                    np.tile(remapped_sig, len(sig_idx)),
                ]
            else:
                idx_map = [np.searchsorted(sig_idx, ind) for ind in indices_use]

            # allocate space to accumulate PSD
            if accumulate_psd:
                if n_times_spectrum == 0:
                    psd_shape = (n_signals_use, n_freqs)
                else:
                    psd_shape = (n_signals_use, n_freqs, n_times_spectrum)
                psd = np.zeros(psd_shape)
            else:
                psd = None

            # create instances of the connectivity estimators
            con_methods = []
            for mtype_i, mtype in enumerate(con_method_types):
                method_params = dict(
                    n_cons=n_cons, n_freqs=n_freqs, n_times=n_times_spectrum
                )
                if method[mtype_i] in _multivariate_methods:
                    method_params.update(dict(n_signals=n_signals_use, n_jobs=n_jobs))
                    if method[mtype_i] in _gc_methods:
                        method_params.update(dict(n_lags=gc_n_lags))
                con_methods.append(mtype(**method_params))

            sep = ", "
            metrics_str = sep.join([meth.name for meth in con_methods])
            logger.info("    the following metrics will be computed: %s" % metrics_str)

        # check dimensions and time scale
        for this_epoch in epoch_block:
            _, _, _, warn_times = _get_and_verify_data_sizes(
                this_epoch,
                sfreq,
                n_signals,
                n_times_in,
                times_in,
                warn_times=warn_times,
            )

        call_params = dict(
            sig_idx=sig_idx,
            tmin_idx=tmin_idx,
            tmax_idx=tmax_idx,
            sfreq=sfreq,
            method=method,
            mode=mode,
            freq_mask=freq_mask,
            idx_map=idx_map,
            n_cons=n_cons,
            block_size=block_size,
            psd=psd,
            accumulate_psd=accumulate_psd,
            mt_adaptive=mt_adaptive,
            con_method_types=con_method_types,
            con_methods=con_methods if n_jobs == 1 else None,
            n_signals=n_signals,
            n_signals_use=n_signals_use,
            n_times=n_times,
            gc_n_lags=gc_n_lags,
            accumulate_inplace=True if n_jobs == 1 else False,
        )
        call_params.update(**spectral_params)

        if n_jobs == 1:
            # no parallel processing
            for this_epoch in epoch_block:
                logger.info(
                    "    computing cross-spectral density for epoch %d"
                    % (epoch_idx + 1)
                )
                # con methods and psd are updated inplace
                _epoch_spectral_connectivity(data=this_epoch, **call_params)
                epoch_idx += 1
        else:
            # process epochs in parallel
            logger.info(
                "    computing cross-spectral density for epochs %d..%d"
                % (epoch_idx + 1, epoch_idx + len(epoch_block))
            )

            out = parallel(
                my_epoch_spectral_connectivity(data=this_epoch, **call_params)
                for this_epoch in epoch_block
            )
            # do the accumulation
            for this_out in out:
                for _method, parallel_method in zip(con_methods, this_out[0]):
                    _method.combine(parallel_method)
                if accumulate_psd:
                    psd += this_out[1]

            epoch_idx += len(epoch_block)

    # normalize
    n_epochs = epoch_idx
    if accumulate_psd:
        psd /= n_epochs

    # compute final connectivity scores
    con = list()
    patterns = list()
    for method_i, conn_method in enumerate(con_methods):
        # future estimators will need to be handled here
        if conn_method.accumulate_psd:
            # compute scores block-wise to save memory
            for i in range(0, n_cons, block_size):
                con_idx = slice(i, i + block_size)
                psd_xx = psd[idx_map[0][con_idx]]
                psd_yy = psd[idx_map[1][con_idx]]
                conn_method.compute_con(con_idx, n_epochs, psd_xx, psd_yy)
        else:
            # compute all scores at once
            if method[method_i] in _multivariate_methods:
                conn_method.compute_con(indices_use, rank, n_epochs)
            else:
                conn_method.compute_con(slice(0, n_cons), n_epochs)

        # get the connectivity scores
        this_con = conn_method.con_scores
        this_patterns = conn_method.patterns

        if this_con.shape[0] != n_cons:
            raise RuntimeError(
                "first dimension of connectivity scores does not match the "
                "number of connections; please contact the mne-connectivity "
                "developers"
            )
        if faverage:
            if this_con.shape[1] != n_freqs:
                raise RuntimeError(
                    "second dimension of connectivity scores does not match "
                    "the number of frequencies; please contact the "
                    "mne-connectivity developers"
                )
            con_shape = (n_cons, n_bands) + this_con.shape[2:]
            this_con_bands = np.empty(con_shape, dtype=this_con.dtype)
            for band_idx in range(n_bands):
                this_con_bands[:, band_idx] = np.mean(
                    this_con[:, freq_idx_bands[band_idx]], axis=1
                )
            this_con = this_con_bands

            if this_patterns is not None:
                patterns_shape = list(this_patterns.shape)
                patterns_shape[3] = n_bands
                this_patterns_bands = np.empty(
                    patterns_shape, dtype=this_patterns.dtype
                )
                for band_idx in range(n_bands):
                    this_patterns_bands[:, :, :, band_idx] = np.mean(
                        this_patterns[:, :, :, freq_idx_bands[band_idx]], axis=3
                    )
                this_patterns = this_patterns_bands

        con.append(this_con)
        patterns.append(this_patterns)

    freqs_used = freqs
    if faverage:
        # for each band we return the frequencies that were averaged
        freqs = [np.mean(x) for x in freqs_bands]

        # make sure freq_bands is a list of equal-length lists
        # XXX: we lose information on which frequency points went into the
        # computation. If h5netcdf supports numpy objects in the future, then
        # we can change the min/max to just make it a list of lists.
        freqs_used = freqs_bands
        freqs_used = [[np.min(band), np.max(band)] for band in freqs_used]

    if indices is None and not multivariate_con:
        # return all-to-all connectivity matrices
        # raveled into a 1D array
        logger.info("    assembling connectivity matrix")
        con_flat = con
        con = list()
        for this_con_flat in con_flat:
            this_con = np.zeros(
                (n_signals, n_signals) + this_con_flat.shape[1:],
                dtype=this_con_flat.dtype,
            )
            this_con[indices_use] = this_con_flat

            # ravel 2D connectivity into a 1D array
            # while keeping other dimensions
            this_con = this_con.reshape((n_signals**2,) + this_con_flat.shape[1:])
            con.append(this_con)
    # number of nodes in the original data
    n_nodes = n_signals

    # create a list of connectivity containers
    conn_list = []
    for _con, _patterns, _method in zip(con, patterns, method):
        kwargs = dict(
            data=_con,
            patterns=_patterns,
            names=names,
            freqs=freqs,
            method=_method,
            n_nodes=n_nodes,
            spec_method=mode,
            indices=indices,
            n_epochs_used=n_epochs,
            freqs_used=freqs_used,
            times_used=times,
            n_tapers=n_tapers,
            metadata=metadata,
            events=events,
            event_id=event_id,
            rank=rank,
            n_lags=gc_n_lags if _method in _gc_methods else None,
        )
        # create the connectivity container
        if mode in ["multitaper", "fourier"]:
            klass = SpectralConnectivity
        else:
            assert mode == "cwt_morlet"
            klass = SpectroTemporalConnectivity
            kwargs.update(times=times)
        conn_list.append(klass(**kwargs))

    logger.info("[Connectivity computation done]")

    if n_methods == 1:
        # for a single method return connectivity directly
        conn_list = conn_list[0]

    return conn_list
