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
from mne.time_frequency import (
    EpochsSpectrum,
    EpochsSpectrumArray,
    EpochsTFR,
    EpochsTFRArray,
)
from mne.time_frequency.multitaper import (
    _compute_mt_params,
    _csd_from_mt,
    _mt_spectra,
    _psd_from_mt,
    _psd_from_mt_adaptive,
)
from mne.time_frequency.tfr import _tfr_from_mt, cwt, morlet
from mne.utils import _arange_div, _check_option, _time_mask, logger, verbose, warn

from ..base import SpectralConnectivity, SpectroTemporalConnectivity
from ..utils import _check_multivariate_indices, check_indices, fill_doc
from .epochs_bivariate import _CON_METHOD_MAP_BIVARIATE
from .epochs_multivariate import (
    _CON_METHOD_MAP_MULTIVARIATE,
    _check_n_components_input,
    _check_rank_input,
    _gc_methods,
    _multicomp_methods,
    _multivariate_methods,
)


def _check_times(data, sfreq, times, tmin, tmax):
    # get the data size and time scale
    n_signals, _, times_in, warn_times = _get_and_verify_data_sizes(
        data=data, sfreq=sfreq, times=times
    )
    n_times_in = len(times_in)  # XXX: Why not use times returned from above func?

    if tmin is not None and tmin < times_in[0]:
        warn(
            f"start time tmin={tmin:.2f} s outside of the time scope of the data "
            f"[{times_in[0]:.2f} s, {times_in[-1]:.2f} s]"
        )
    if tmax is not None and tmax > times_in[-1]:
        warn(
            f"stop time tmax={tmax:.2f} s outside of the time scope of the data "
            f"[{times_in[0]:.2f} s, {times_in[-1]:.2f} s]"
        )

    mask = _time_mask(times_in, tmin, tmax, sfreq=sfreq)
    tmin_idx, tmax_idx = np.where(mask)[0][[0, -1]]
    tmax_idx += 1
    tmin_true = times_in[tmin_idx]
    tmax_true = times_in[tmax_idx - 1]  # time of last point used

    times = times_in[tmin_idx:tmax_idx]
    n_times = len(times)

    logger.info(
        f"    using t={tmin_true:.3f}s..{tmax_true:.3f}s for estimation ({n_times} "
        "points)"
    )

    return (
        n_signals,
        times,
        n_times,
        times_in,
        n_times_in,
        tmin_idx,
        tmax_idx,
        warn_times,
    )


def _check_freqs(sfreq, fmin, n_times):
    # check that fmin corresponds to at least 5 cycles
    dur = float(n_times) / sfreq
    five_cycle_freq = 5.0 / dur
    if len(fmin) == 1 and fmin[0] == -np.inf:
        # we use the 5 cycle freq. as default
        fmin = np.array([five_cycle_freq])
    else:
        if np.any(fmin < five_cycle_freq):
            warn(
                f"fmin={np.min(fmin):.3f} Hz corresponds to {dur * np.min(fmin):.3f} < "
                f"5 cycles based on the epoch length {dur:.3f} sec, need at least "
                f"{5.0 / np.min(fmin):.3f} sec epochs or fmin={five_cycle_freq:.3f}. "
                "Spectrum estimate will be unreliable."
            )

    return fmin


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
            raise ValueError("define frequencies of interest using cwt_freqs")
        else:
            cwt_freqs = cwt_freqs.astype(np.float64)
        if any(cwt_freqs > (sfreq / 2.0)):
            raise ValueError(
                "entries in cwt_freqs cannot be larger than Nyquist (sfreq / 2)"
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
    freqs,
    indices,
    method,
    mode,
    fskip,
    n_bands,
    cwt_freqs,
    faverage,
    spectrum_computed,
):
    """Check and precompute dimensions of results data."""
    first_epoch = epoch_block[0]

    # Sort times
    if spectrum_computed and times_in is None:  # is a Spectrum object
        n_signals = first_epoch[0].shape[0]
        times = None
        n_times = 0
        n_times_in = 0
        tmin_idx = None
        tmax_idx = None
        warn_times = False
    else:  # data has a time dimension (time series or TFR object)
        if spectrum_computed:  # is a TFR object
            if mode == "cwt_morlet":
                first_epoch = (first_epoch[0][:, 0],)  # just take first freq
            else:  # multitaper
                first_epoch = (first_epoch[0][:, 0, 0],)  # take first taper and freq
        (
            n_signals,
            times,
            n_times,
            times_in,
            n_times_in,
            tmin_idx,
            tmax_idx,
            warn_times,
        ) = _check_times(
            data=first_epoch, sfreq=sfreq, times=times_in, tmin=tmin, tmax=tmax
        )

    # Sort freqs
    if not spectrum_computed:  # is an (ordinary) time series
        # check that fmin corresponds to at least 5 cycles
        fmin = _check_freqs(sfreq=sfreq, fmin=fmin, n_times=n_times)
        # compute frequencies to analyze based on number of samples, sampling rate,
        # specified wavelet frequencies, and mode
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
                f"There are no frequency points between {fmin[i]:.1f}Hz and "
                f"{fmax[i]:.1f}Hz. Change the band specification (fmin, fmax) or the "
                "frequency resolution."
            )
    if n_bands == 1:
        logger.info(
            f"    frequencies: {freqs_bands[0][0]:.1f}Hz..{freqs_bands[0][-1]:.1f}Hz "
            f"({n_freqs} points)"
        )
    else:
        logger.info("    computing connectivity for the bands:")
        for i, bfreqs in enumerate(freqs_bands):
            logger.info(
                f"     band {i + 1}: {bfreqs[0]:.1f}Hz..{bfreqs[-1]:.1f}Hz "
                f"({len(bfreqs)} points)"
            )
    if faverage:
        logger.info("    connectivity scores will be averaged for each band")

    # Sort indices
    multivariate_con = any(
        this_method in _multivariate_methods for this_method in method
    )

    if indices is None:
        if multivariate_con:
            if any(this_method in _gc_methods for this_method in method):
                raise ValueError(
                    "indices must be specified when computing Granger causality, as "
                    "all-to-all connectivity is not supported"
                )
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
                            "seed and target indices must not intersect when computing "
                            "Granger causality"
                        )
        else:
            indices_use = check_indices(indices)

    # number of connections to compute
    n_cons = len(indices_use[0])

    logger.info(f"    computing connectivity for {n_cons} connections")

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
    spectral_params = dict(eigvals=None, window_fun=None, wavelets=None, weights=None)
    n_tapers = None
    n_times_spectrum = 0
    is_tfr_con = False
    if mode == "multitaper":
        window_fun, eigvals, mt_adaptive = _compute_mt_params(
            n_times, sfreq, mt_bandwidth, mt_low_bias, mt_adaptive
        )
        spectral_params.update(window_fun=window_fun, eigvals=eigvals)
    elif mode == "fourier":
        logger.info("    using FFT with a Hanning window to estimate spectra")
        spectral_params.update(window_fun=np.hanning(n_times), eigvals=1.0)
    elif mode == "cwt_morlet":
        logger.info("    using CWT with Morlet wavelets to estimate spectra")

        # reformat cwt_n_cycles if we have removed some frequencies
        # using fmin, fmax, fskip
        cwt_n_cycles = np.array((cwt_n_cycles,), dtype=float).ravel()
        if len(cwt_n_cycles) > 1:
            if len(cwt_n_cycles) != len(cwt_freqs):
                raise ValueError(
                    "cwt_n_cycles must be float or an array with the same size as "
                    "cwt_freqs"
                )
            cwt_n_cycles = cwt_n_cycles[freq_mask]

        # get the Morlet wavelets
        spectral_params.update(
            wavelets=morlet(sfreq, freqs, n_cycles=cwt_n_cycles, zero_mean=True)
        )
        n_times_spectrum = n_times
        is_tfr_con = True
    else:
        raise ValueError("mode has an invalid value")
    return spectral_params, mt_adaptive, n_times_spectrum, n_tapers, is_tfr_con


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


def _compute_spectra(
    data,
    sfreq,
    mode,
    sig_idx,
    tmin_idx,
    tmax_idx,
    mt_adaptive,
    eigvals,
    wavelets,
    window_fun,
    freq_mask,
    accumulate_psd,
):
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
                    # hack to so we can sum over axis=-2 (tapers dim)
                    weights = np.ones((1, 1, 1))

                if accumulate_psd:
                    _this_psd = _psd_from_mt(this_x_t, weights)
        else:  # mode == 'cwt_morlet'
            weights = None
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

    return x_t, this_psd, weights


def _tfr_csd_from_mt(x_mt, y_mt, weights_x, weights_y):
    """Compute time-frequency CSD from tapered spectra.

    Parameters
    ----------
    x_mt : array, shape (..., n_tapers, n_freqs, n_times)
        The tapered time-frequency spectra for signals x.
    y_mt : array, shape (..., n_tapers, n_freqs, n_times)
        The tapered time-frequency spectra for signals y.
    weights_x : array, shape (n_tapers, n_freqs)
        Weights to use for combining the tapered spectra of x_mt.
    weights_y : array, shape (n_tapers, n_freqs)
        Weights to use for combining the tapered spectra of y_mt.

    Returns
    -------
    csd : array, shape (..., n_freqs, n_times)
        The CSD between x and y.
    """
    # expand weights dims to match x_mt and y_mt
    weights_x = np.expand_dims(weights_x, axis=(*np.arange(x_mt.ndim - 3), -1))
    weights_y = np.expand_dims(weights_y, axis=(*np.arange(y_mt.ndim - 3), -1))
    # compute CSD
    csd = np.sum(weights_x * x_mt * (weights_y * y_mt).conj(), axis=-3)
    denom = np.sqrt((weights_x * weights_x.conj()).real.sum(axis=-3)) * np.sqrt(
        (weights_y * weights_y.conj()).real.sum(axis=-3)
    )
    csd *= 2 / denom
    return csd


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
    weights,
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
    n_components,
    spectrum_computed,
    is_tfr_con,
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
                    # if it's a coherency-based method
                    con_methods.append(
                        mtype(
                            n_signals_use,
                            n_cons,
                            n_freqs,
                            n_times_spectrum,
                            n_components=n_components,
                        )
                    )
            else:
                con_methods.append(mtype(n_cons, n_freqs, n_times_spectrum))

    _check_option("mode", mode, ("cwt_morlet", "multitaper", "fourier"))
    if len(sig_idx) == n_signals:
        # we use all signals: use a slice for faster indexing
        sig_idx = slice(None, None)

    # compute tapered spectra
    if spectrum_computed:  # use existing spectral info
        # Select entries of interest (flexible indexing for optional tapers dim)
        if tmin_idx is not None and tmax_idx is not None:  # TFR spectra
            x_t = np.asarray(data)[:, sig_idx][..., freq_mask, tmin_idx:tmax_idx]
        else:  # normal spectra
            x_t = np.asarray(data)[:, sig_idx][..., freq_mask]
            if weights is None:  # assumes no tapers dim, i.e., for Fourier/Welch mode
                x_t = np.expand_dims(x_t, axis=2)  # CSD construction expects tapers dim
                weights = np.ones((1, 1, 1))  # assign dummy weights
        if accumulate_psd:
            if weights is not None:  # mode == 'multitaper' or 'fourier'
                if not is_tfr_con:  # normal spectra (multitaper or Fourier)
                    this_psd = _psd_from_mt(x_t, weights)
                else:  # TFR spectra (multitaper)
                    this_psd = _tfr_from_mt(x_t, weights)
            else:  # mode == 'cwt_morlet'
                this_psd = (x_t * x_t.conj()).real
    else:  # compute spectral info from scratch
        x_t, this_psd, weights = _compute_spectra(
            data=data,
            sfreq=sfreq,
            mode=mode,
            sig_idx=sig_idx,
            tmin_idx=tmin_idx,
            tmax_idx=tmax_idx,
            mt_adaptive=mt_adaptive,
            eigvals=eigvals,
            wavelets=wavelets,
            window_fun=window_fun,
            freq_mask=freq_mask,
            accumulate_psd=accumulate_psd,
        )

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
    for this_method in con_methods:
        this_method.start_epoch()

    # accumulate connectivity scores
    if mode in ["multitaper", "fourier"]:
        for i in range(0, n_con_signals, block_size):
            n_extra = max(0, i + block_size - n_con_signals)
            con_idx = slice(i, i + block_size - n_extra)
            compute_csd = _csd_from_mt if not is_tfr_con else _tfr_csd_from_mt
            if mt_adaptive:
                csd = compute_csd(
                    x_t[idx_map[0][con_idx]],
                    x_t[idx_map[1][con_idx]],
                    weights[idx_map[0][con_idx]],
                    weights[idx_map[1][con_idx]],
                )
            else:
                csd = compute_csd(
                    x_t[idx_map[0][con_idx]], x_t[idx_map[1][con_idx]], weights, weights
                )

            for this_method in con_methods:
                this_method.accumulate(con_idx, csd)
    else:  # mode == 'cwt_morlet'  # reminder to add alternative TFR methods
        for i in range(0, n_con_signals, block_size):
            n_extra = max(0, i + block_size - n_con_signals)
            con_idx = slice(i, i + block_size - n_extra)
            # this codes can be very slow
            csd = x_t[idx_map[0][con_idx]] * x_t[idx_map[1][con_idx]].conjugate()

            for this_method in con_methods:
                this_method.accumulate(con_idx, csd)
    # future estimator types need to be explicitly handled here

    return con_methods, psd


def _get_n_epochs(epochs, n):
    """Generate lists with at most n epochs."""
    epochs_out = list()
    for epoch in epochs:
        if not isinstance(epoch, list | tuple):
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
    if not isinstance(data, list | tuple):
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
                    "all input time series must have the same number of time points"
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
                "the number of time series has to be the same in each epoch"
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
            raise ValueError(f"{this_method} is not a valid connectivity method")
        else:
            # support for custom class
            method_valid, msg = _check_method(this_method)
            if not method_valid:
                raise ValueError(
                    f"The supplied connectivity method does not have the method {msg}"
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
    n_components=1,
    block_size=1000,
    n_jobs=1,
    verbose=None,
):
    r"""Compute frequency- and time-frequency-domain connectivity measures.

    The connectivity method(s) are specified using the "method" parameter. All methods
    are based on estimates of the cross- and power spectral densities (CSD/PSD) Sxy and
    Sxx, Syy.

    Parameters
    ----------
    data : array_like, shape (n_epochs, n_signals, n_times) | ~mne.Epochs | generator | ~mne.time_frequency.EpochsSpectrum | ~mne.time_frequency.EpochsTFR
        The data from which to compute connectivity. Can be epoched time series data as
        an array-like or :class:`mne.Epochs` object, or Fourier coefficients for each
        epoch as an :class:`mne.time_frequency.EpochsSpectrum` or
        :class:`mne.time_frequency.EpochsTFR` object. If time series data, the spectral
        information will be computed according to the spectral estimation mode (see the
        ``mode`` parameter). If an :class:`mne.time_frequency.EpochsSpectrum` or
        :class:`mne.time_frequency.EpochsTFR` object, existing spectral information
        will be used and the ``mode`` parameter will be ignored.

        Note that it is also possible to combine multiple time series signals by
        providing a list of tuples, e.g.: ::

            data = [(arr_0, stc_0), (arr_1, stc_1), (arr_2, stc_2)]

        which corresponds to 3 epochs where ``arr_*`` is an array with the same number
        of time points as ``stc_*``. Data can also be a list/generator of arrays, shape
        ``(n_signals, n_times)``, or a list/generator of :class:`mne.SourceEstimate` or
        :class:`mne.VolSourceEstimate` objects.

        .. versionchanged:: 0.8
           Fourier coefficients stored in an :class:`mne.time_frequency.EpochsSpectrum`
           or :class:`mne.time_frequency.EpochsTFR` object can also be passed in as
           data. Storing Fourier coefficients in
           :class:`mne.time_frequency.EpochsSpectrum` objects requires ``mne >= 1.8``.
           Storing multitaper weights in :class:`mne.time_frequency.EpochsTFR` objects
           requires ``mne >= 1.10``.
    %(names)s
    method : str | list of str
        Connectivity measure(s) to compute. These can be ``['coh', 'cohy', 'imcoh',
        'cacoh', 'mic', 'mim', 'plv', 'ciplv', 'ppc', 'pli', 'dpli', 'wpli',
        'wpli2_debiased', 'gc', 'gc_tr']``. These are:

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

        Multivariate methods (``['cacoh', 'mic', 'mim', 'gc', 'gc_tr']``) cannot be
        called with the other methods.
    indices : tuple of array_like | None
        Two array-likes with indices of connections for which to compute connectivity.
        If a bivariate method is called, each array for the seeds and targets should
        contain the channel indices for each bivariate connection. If a multivariate
        method is called, each array for the seeds and targets should consist of nested
        arrays containing the channel indices for each multivariate connection. If
        ``None``, connections between all channels are computed, unless a Granger
        causality method is called, in which case an error is raised.
    sfreq : float | None
        The sampling frequency. Required if ``data`` is an array-like.
    mode : ``'multitaper'`` | ``'fourier'`` | ``'cwt_morlet'``
        Spectrum estimation mode. Ignored if ``data`` is an
        :class:`mne.time_frequency.EpochsSpectrum` or
        :class:`mne.time_frequency.EpochsTFR` object.
    fmin : float | tuple of float | None
        The lower frequency of interest. Multiple bands are defined using a tuple, e.g.,
        (8., 20.) for two bands with 8 Hz and 20 Hz lower freq. If ``None``, the
        frequency corresponding to 5 cycles based on the epoch length is used. For
        example, with an epoch length of 1 sec, the lower frequency would be 5 / 1 sec =
        5 Hz.
    fmax : float | tuple of float
        The upper frequency of interest. Multiple bands are defined using a tuple, e.g.,
        (13., 30.) for two bands with 13 Hz and 30 Hz upper freq.
    fskip : int
        Omit every "(fskip + 1)-th" frequency bin to decimate in frequency domain.
    faverage : bool
        Average connectivity scores for each frequency band. If ``True``, the output
        freqs will be a list with arrays of the frequencies that were averaged.
    tmin : float | None
        Time to start connectivity estimation. Note: when ``data`` is an array-like, the
        first sample is assumed to be at time 0. For :class:`mne.Epochs`, the time
        information contained in the object is used to compute the time indices. Ignored
        if ``data`` is an :class:`mne.time_frequency.EpochsSpectrum` object.
    tmax : float | None
        Time to end connectivity estimation. Note: when ``data`` is an array-like, the
        first sample is assumed to be at time 0. For :class:`mne.Epochs`, the time
        information contained in the object is used to compute the time indices. Ignored
        if ``data`` is an :class:`mne.time_frequency.EpochsSpectrum` object.
    mt_bandwidth : float | None
        The bandwidth of the multitaper windowing function in Hz. Only used in
        ``'multitaper'`` mode. Ignored if ``data`` is an
        :class:`mne.time_frequency.EpochsSpectrum` or
        :class:`mne.time_frequency.EpochsTFR` object.
    mt_adaptive : bool
        Use adaptive weights to combine the tapered spectra into PSD. Only used in
        ``'multitaper'`` mode. Ignored if ``data`` is an
        :class:`mne.time_frequency.EpochsSpectrum` or
        :class:`mne.time_frequency.EpochsTFR` object.
    mt_low_bias : bool
        Only use tapers with more than 90 percent spectral concentration within
        bandwidth. Only used in ``'multitaper'`` mode. Ignored if ``data`` is an
        :class:`mne.time_frequency.EpochsSpectrum` or
        :class:`mne.time_frequency.EpochsTFR` object.
    cwt_freqs : array_like
        Array-like of frequencies of interest. Only used in ``'cwt_morlet'`` mode. Only
        the frequencies within the range specified by ``fmin`` and ``fmax`` are used.
        Ignored if ``data`` is an :class:`mne.time_frequency.EpochsSpectrum` or
        :class:`mne.time_frequency.EpochsTFR` object.
    cwt_n_cycles : float | array_like
        Number of cycles. Fixed number or one per frequency. Only used in
        ``'cwt_morlet'`` mode. Ignored if ``data`` is an
        :class:`mne.time_frequency.EpochsSpectrum` or
        :class:`mne.time_frequency.EpochsTFR` object.
    gc_n_lags : int
        Number of lags to use for the vector autoregressive model when computing Granger
        causality. Higher values increase computational cost, but reduce the degree of
        spectral smoothing in the results. Only used if ``method`` contains any of
        ``['gc', 'gc_tr']``.
    rank : tuple of array_like | None
        Two array-likes with the rank to project the seed and target data to,
        respectively, using singular value decomposition. If ``None``, the rank of the
        data is computed and projected to. Only used if ``method`` contains any of
        ``['cacoh', 'mic', 'mim', 'gc', 'gc_tr']``.
    n_components : int | None
        Number of connectivity components to extract from the data. If an int, the
        number of components must be <= the minimum rank of the seeds and targets. E.g.,
        if the seed channels had a rank of 5 and the target channels had a rank of 3,
        ``n_components`` must be <= 3. If ``None``, the number of components equal to
        the minimum rank of the seeds and targets is extracted (see the ``rank``
        parameter). Only used if ``method`` contains any of ``['cacoh', 'mic']``.

        .. versionadded:: 0.8
    block_size : int
        How many connections to compute at once (higher numbers are faster
        but require more memory).
    n_jobs : int
        How many samples to process in parallel.
    %(verbose)s

    Returns
    -------
    con : instance of SpectralConnectivity or SpectroTemporalConnectivity | list
        Computed connectivity measure(s). An instance of :class:`SpectralConnectivity`,
        :class:`SpectroTemporalConnectivity`, or a list of instances corresponding to
        connectivity measures if several connectivity measures are specified. The shape
        of the connectivity result will be:

        - ``(n_cons, n_freqs)`` for ``'multitaper'`` or ``'fourier'`` modes
        - ``(n_cons, n_freqs, n_times)`` for ``'cwt_morlet'`` mode
        - ``(n_cons, n_comps, n_freqs[, n_times])`` for valid multivariate methods if
          ``n_components > 1``
        - ``n_cons = n_signals ** 2`` for bivariate methods with ``indices=None``
        - ``n_cons = 1`` for multivariate methods with ``indices=None``
        - ``n_cons = len(indices[0])`` for bivariate and multivariate methods when
          ``indices`` is supplied

    See Also
    --------
    mne_connectivity.spectral_connectivity_time
    mne_connectivity.SpectralConnectivity
    mne_connectivity.SpectroTemporalConnectivity

    Notes
    -----
    Please note that the interpretation of the measures in this function depends on the
    data and underlying assumptions and does not necessarily reflect a causal
    relationship between brain regions.

    These measures are not to be interpreted over time. Each epoch passed into the
    dataset is interpreted as an independent sample of the same connectivity structure.
    Within each epoch, it is assumed that the spectral measure is stationary. The
    spectral measures implemented in this function are computed across epochs. **Thus,
    spectral measures computed with only one epoch will result in errorful values and
    spectral measures computed with few Epochs will be unreliable.** Please see
    :func:`~mne_connectivity.spectral_connectivity_time` for time-resolved connectivity
    estimation.

    The spectral densities can be estimated using a multitaper method with digital
    prolate spheroidal sequence (DPSS) windows, a discrete Fourier transform with
    Hanning windows, or a continuous wavelet transform using Morlet wavelets. The
    spectral estimation mode is specified using the ``mode`` parameter. Complex Welch,
    multitaper, or Morlet coefficients can also be passed in as data in the form of
    :class:`mne.time_frequency.EpochsSpectrum` or :class:`mne.time_frequency.EpochsTFR`
    objects.

    By default, the connectivity between all signals is computed (only connections
    corresponding to the lower-triangular part of the connectivity matrix). If one is
    only interested in the connectivity between some signals, the ``indices`` parameter
    can be used. For example, to compute the connectivity between the signal with index
    0 and signals "2, 3, 4" (a total of 3 connections) one can use the following::

        indices = (np.array([0, 0, 0]),    # row indices
                   np.array([2, 3, 4]))    # col indices

        con = spectral_connectivity_epochs(data, method='coh',
                                           indices=indices, ...)

    In this case ``con.get_data().shape = (3, n_freqs)``. The connectivity scores are in
    the same order as defined indices.

    For multivariate methods, this is handled differently. If ``indices`` is ``None``,
    connectivity between all signals will be computed and a single connectivity spectrum
    will be returned (this is not possible if a Granger causality method is called). If
    ``indices`` is specified, seed and target indices for each connection should be
    specified as nested array-likes. For example, to compute the connectivity between
    signals (0, 1) -> (2, 3) and (0, 1) -> (4, 5), indices should be specified as::

        indices = (np.array([[0, 1], [0, 1]]),  # seeds
                   np.array([[2, 3], [4, 5]]))  # targets

    More information on working with multivariate indices and handling connections where
    the number of seeds and targets are not equal can be found in the
    :doc:`../auto_examples/handling_ragged_arrays` example.

    **Supported Connectivity Measures**

    The connectivity method(s) is specified using the ``method`` parameter. The
    following methods are supported (note: ``E[]`` denotes average over epochs).
    Multiple measures can be computed at once by using a list/tuple, e.g., ``['coh',
    'pli']`` to compute coherence and PLI.

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

        'cacoh' : Canonical Coherency (CaCoh) :footcite:`VidaurreEtAl2019` given by:

            :math:`\textrm{CaCoh}=\Large{\frac{\boldsymbol{a}^T\boldsymbol{D}
            (\Phi)\boldsymbol{b}}{\sqrt{\boldsymbol{a}^T\boldsymbol{a}
            \boldsymbol{b}^T\boldsymbol{b}}}}`

            where: :math:`\boldsymbol{D}(\Phi)` is the cross-spectral density between
            seeds and targets transformed for a given phase angle :math:`\Phi`; and
            :math:`\boldsymbol{a}` and :math:`\boldsymbol{b}` are eigenvectors for the
            seeds and targets, such that :math:`\boldsymbol{a}^T\boldsymbol{D}(\Phi)
            \boldsymbol{b}` maximises coherency between the seeds and targets. Taking
            the absolute value of the results gives maximised coherence.

        'mic' : Maximised Imaginary part of Coherency (MIC) :footcite:`EwaldEtAl2012`
        given by:

            :math:`\textrm{MIC}=\Large{\frac{\boldsymbol{\alpha}^T
            \boldsymbol{E \beta}}{\parallel\boldsymbol{\alpha}\parallel
            \parallel\boldsymbol{\beta}\parallel}}`

            where: :math:`\boldsymbol{E}` is the imaginary part of the transformed
            cross-spectral density between seeds and targets; and
            :math:`\boldsymbol{\alpha}` and :math:`\boldsymbol{\beta}` are eigenvectors
            for the seeds and targets, such that :math:`\boldsymbol{\alpha}^T
            \boldsymbol{E \beta}` maximises the imaginary part of coherency between the
            seeds and targets.

        'mim' : Multivariate Interaction Measure (MIM) :footcite:`EwaldEtAl2012` given
        by:

            :math:`\textrm{MIM}=tr(\boldsymbol{EE}^T)`

            where :math:`\boldsymbol{E}` is the imaginary part of the transformed
            cross-spectral density between seeds and targets.

        'plv' : Phase-Locking Value (PLV) :footcite:`LachauxEtAl1999` given by::

            PLV = |E[Sxy/|Sxy|]|

        'ciplv' : corrected imaginary PLV (ciPLV) :footcite:`BrunaEtAl2018` given by::

                             |E[Im(Sxy/|Sxy|)]|
            ciPLV = ------------------------------------
                     sqrt(1 - |E[real(Sxy/|Sxy|)]| ** 2)

        'ppc' : Pairwise Phase Consistency (PPC), an unbiased estimator of squared PLV
        :footcite:`VinckEtAl2010`.

        'pli' : Phase Lag Index (PLI) :footcite:`StamEtAl2007` given by::

            PLI = |E[sign(Im(Sxy))]|

        'pli2_unbiased' : Unbiased estimator of squared PLI :footcite:`VinckEtAl2011`.

        'dpli' : Directed Phase Lag Index (DPLI) :footcite:`StamEtAl2012` given by
        (where H is the Heaviside function)::

            DPLI = E[H(Im(Sxy))]

        'wpli' : Weighted Phase Lag Index (WPLI) :footcite:`VinckEtAl2011` given by::

                      |E[Im(Sxy)]|
            WPLI = ------------------
                      E[|Im(Sxy)|]

        'wpli2_debiased' : Debiased estimator of squared WPLI :footcite:`VinckEtAl2011`.

        'gc' : State-space Granger Causality (GC) :footcite:`BarnettSeth2015` given by:

            :math:`GC = ln\Large{(\frac{\lvert\boldsymbol{S}_{tt}\rvert}{\lvert
            \boldsymbol{S}_{tt}-\boldsymbol{H}_{ts}\boldsymbol{\Sigma}_{ss
            \lvert t}\boldsymbol{H}_{ts}^*\rvert}})`

            where: :math:`s` and :math:`t` represent the seeds and targets,
            respectively; :math:`\boldsymbol{H}` is the spectral transfer function;
            :math:`\boldsymbol{\Sigma}` is the residuals matrix of the autoregressive
            model; and :math:`\boldsymbol{S}` is :math:`\boldsymbol{\Sigma}` transformed
            by :math:`\boldsymbol{H}`.

        'gc_tr' : State-space GC on time-reversed signals
        :footcite:`BarnettSeth2015,WinklerEtAl2016` given by the same equation as for
        ``'gc'``, but where the autocovariance sequence from which the autoregressive
        model is produced is transposed to mimic the reversal of the original signal in
        time :footcite:`HaufeEtAl2012`.

    References
    ----------
    .. footbibliography::
    """  # noqa: E501
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
    if not isinstance(method, list | tuple):
        method = [method]  # make it a list so we can iterate over it

    if n_bands != 1 and any(this_method in _gc_methods for this_method in method):
        raise ValueError(
            "computing Granger causality on multiple frequency bands is not yet "
            "supported"
        )

    if any(this_method in _multivariate_methods for this_method in method):
        if not all(this_method in _multivariate_methods for this_method in method):
            raise ValueError(
                "bivariate and multivariate connectivity methods cannot be used in the "
                "same function call"
            )
        multivariate_con = True
    else:
        multivariate_con = False

    # handle connectivity estimators
    (con_method_types, n_methods, accumulate_psd) = _check_estimators(method)

    times_in = None
    events = None
    event_id = None
    freqs = None
    weights = None
    metadata = None
    spectrum_computed = False
    is_tfr_con = False
    if isinstance(data, BaseEpochs | EpochsSpectrum | EpochsTFR):
        names = data.ch_names
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

        if isinstance(data, EpochsSpectrum | EpochsTFR):
            # XXX: Will need to be updated if new Spectrum/TFR methods are added
            if not np.iscomplexobj(data.get_data()):
                raise TypeError(
                    "if `data` is an EpochsSpectrum or EpochsTFR object, it must "
                    "contain complex-valued Fourier coefficients, such as that "
                    "returned from Epochs.compute_psd/tfr() with `output='complex'`"
                )
            if "segment" in data._dims:
                raise ValueError(
                    "`data` cannot contain Fourier coefficients for individual segments"
                )
            mode = data.method
            if isinstance(data, EpochsSpectrum):
                if isinstance(data, EpochsSpectrumArray):  # infer mode from dimensions
                    # Currently, actual mode doesn't matter as long as we handle tapers
                    # and their weights in the same way as for multitaper spectra
                    mode = "multitaper" if "taper" in data._dims else "fourier"
                else:  # read mode from object
                    mode = "fourier" if mode == "welch" else mode
            else:
                if isinstance(data, EpochsTFRArray):  # infer mode from dimensions
                    # Currently, actual mode doesn't matter as long as we handle tapers
                    # and their weights in the same way as for multitaper spectra
                    mode = "multitaper" if "taper" in data._dims else "morlet"
                else:
                    mode = "cwt_morlet" if mode == "morlet" else mode
                is_tfr_con = True
                times_in = data.times
            spectrum_computed = True
            freqs = data.freqs
            # Extract weights from the EpochsSpectrum/TFR object
            if not hasattr(data, "weights") or (
                data.weights is None and mode == "multitaper"
            ):
                # XXX: Remove logic when support for mne<1.10 is dropped
                raise AttributeError(
                    "weights are required for multitaper coefficients stored in "
                    "EpochsSpectrum (requires mne >= 1.8) and EpochsTFR (requires "
                    "mne >= 1.10) objects; objects saved from older versions of mne "
                    "will need to be recomputed."
                )
            if hasattr(data, "weights"):
                weights = data.weights
        else:
            times_in = data.times  # input times for Epochs input type
    elif sfreq is None:
        raise ValueError("Sampling frequency (sfreq) is required with array input.")

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
                freqs=freqs,
                indices=indices,
                method=method,
                mode=mode,
                fskip=fskip,
                n_bands=n_bands,
                cwt_freqs=cwt_freqs,
                faverage=faverage,
                spectrum_computed=spectrum_computed,
            )

            # check rank input and compute data ranks if necessary
            if multivariate_con:
                rank = _check_rank_input(rank, data, indices_use)
                n_components = _check_n_components_input(n_components, rank)
                if n_components == 1:
                    # n_components=0 means space for a components dimension is not
                    # allocated in the results, similar to how n_times_spectrum=0 is
                    # used to indicate that time is not a dimension in the results
                    n_components = 0
            else:
                rank = None
                n_components = 0
                gc_n_lags = None

            # make sure padded indices are stored in the connectivity object
            if multivariate_con and indices is not None:
                # create a copy so that `indices_use` can be modified
                indices = (indices_use[0].copy(), indices_use[1].copy())

            # get the window function, wavelets, etc for different modes
            if not spectrum_computed:
                spectral_params, mt_adaptive, n_times_spectrum, n_tapers, is_tfr_con = (
                    _assemble_spectral_params(
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
                )
            else:
                spectral_params = dict(
                    eigvals=None, window_fun=None, wavelets=None, weights=weights
                )
                n_times_spectrum = n_times  # 0 if no times
                n_tapers = None if weights is None else weights.shape[0]

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
                    if method[mtype_i] in _multicomp_methods:
                        method_params.update(dict(n_components=n_components))
                    if method[mtype_i] in _gc_methods:
                        method_params.update(dict(n_lags=gc_n_lags))
                con_methods.append(mtype(**method_params))

            sep = ", "
            metrics_str = sep.join([meth.name for meth in con_methods])
            logger.info(f"    the following metrics will be computed: {metrics_str}")

        # check dimensions and time scale
        if not spectrum_computed:
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
            n_components=n_components,
            spectrum_computed=spectrum_computed,
            is_tfr_con=is_tfr_con,
            accumulate_inplace=True if n_jobs == 1 else False,
        )
        call_params.update(**spectral_params)

        if n_jobs == 1:
            # no parallel processing
            for this_epoch in epoch_block:
                logger.info(
                    f"    computing cross-spectral density for epoch {epoch_idx + 1}"
                )
                # con methods and psd are updated inplace
                _epoch_spectral_connectivity(data=this_epoch, **call_params)
                epoch_idx += 1
        else:
            # process epochs in parallel
            logger.info(
                f"    computing cross-spectral density for epochs {epoch_idx + 1}.."
                f"{epoch_idx + len(epoch_block)}"
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
                "first dimension of connectivity scores does not match the number of "
                "connections; please contact the mne-connectivity developers"
            )
        if faverage:
            if n_components != 0 and method[method_i] in _multicomp_methods:
                this_con = np.moveaxis(this_con, 2, 1)  # make freqs the 2nd dimension
            if this_con.shape[1] != n_freqs:
                raise RuntimeError(
                    "second dimension of connectivity scores does not match the number "
                    "of frequencies; please contact the mne-connectivity developers"
                )
            con_shape = (n_cons, n_bands) + this_con.shape[2:]
            this_con_bands = np.empty(con_shape, dtype=this_con.dtype)
            for band_idx in range(n_bands):
                this_con_bands[:, band_idx] = np.mean(
                    this_con[:, freq_idx_bands[band_idx]], axis=1
                )
            this_con = this_con_bands
            if n_components != 0 and method[method_i] in _multicomp_methods:
                this_con = np.moveaxis(this_con, 1, 2)  # return comps to 2nd dimension

            if this_patterns is not None:
                if n_components != 0:
                    # make freqs the 4th dimension
                    this_patterns = np.moveaxis(this_patterns, 4, 3)
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
                if n_components != 0:
                    # return comps to 4th dimension
                    this_patterns = np.moveaxis(this_patterns, 3, 4)

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
            spec_method=(
                mode
                if not isinstance(data, EpochsSpectrum | EpochsTFR)
                else data.method
            ),
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
        if n_components and _method in _multicomp_methods:
            kwargs.update(components=np.arange(n_components) + 1)
        # create the connectivity container
        if not is_tfr_con:
            klass = SpectralConnectivity
        else:
            klass = SpectroTemporalConnectivity
            kwargs.update(times=times)
        conn_list.append(klass(**kwargs))

    logger.info("[Connectivity computation done]")

    if n_methods == 1:
        # for a single method return connectivity directly
        conn_list = conn_list[0]

    return conn_list
