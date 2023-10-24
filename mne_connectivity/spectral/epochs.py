# Authors: Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#          Denis A. Engemann <denis.engemann@gmail.com>
#          Adam Li <adam2392@gmail.com>
#          Thomas S. Binns <t.s.binns@outlook.com>
#
# License: BSD (3-clause)

from functools import partial
import inspect

import numpy as np
from mne.epochs import BaseEpochs
from mne.parallel import parallel_func
from mne.source_estimate import _BaseSourceEstimate
from mne.time_frequency.multitaper import (
    _csd_from_mt, _mt_spectra, _psd_from_mt, _psd_from_mt_adaptive)
from mne.time_frequency.tfr import cwt, morlet
from mne.time_frequency.multitaper import _compute_mt_params
from mne.utils import _arange_div, _check_option, _time_mask, logger, warn

from ..base import SpectralConnectivity, SpectroTemporalConnectivity


def _compute_freqs(n_times, sfreq, cwt_freqs, mode):
    from scipy.fft import rfftfreq
    # get frequencies of interest for the different modes
    if mode in ('multitaper', 'fourier'):
        # fmin fmax etc is only supported for these modes
        # decide which frequencies to keep
        freqs_all = rfftfreq(n_times, 1. / sfreq)
    elif mode == 'cwt_morlet':
        # cwt_morlet mode
        if cwt_freqs is None:
            raise ValueError('define frequencies of interest using '
                             'cwt_freqs')
        else:
            cwt_freqs = cwt_freqs.astype(np.float64)
        if any(cwt_freqs > (sfreq / 2.)):
            raise ValueError('entries in cwt_freqs cannot be '
                             'larger than Nyquist (sfreq / 2)')
        freqs_all = cwt_freqs
    else:
        raise ValueError('mode has an invalid value')

    return freqs_all


def _compute_freq_mask(freqs_all, fmin, fmax, fskip):
    # create a frequency mask for all bands
    freq_mask = np.zeros(len(freqs_all), dtype=bool)
    for f_lower, f_upper in zip(fmin, fmax):
        freq_mask |= ((freqs_all >= f_lower) & (freqs_all <= f_upper))

    # possibly skip frequency points
    for pos in range(fskip):
        freq_mask[pos + 1::fskip + 1] = False
    return freq_mask


def _prepare_connectivity(epoch_block, times_in, tmin, tmax, fmin, fmax, sfreq,
                          mode, fskip, n_bands, cwt_freqs, faverage):
    """Check and precompute dimensions of results data."""
    first_epoch = epoch_block[0]

    # get the data size and time scale
    n_signals, n_times_in, times_in, warn_times = _get_and_verify_data_sizes(
        first_epoch, sfreq, times=times_in)

    n_times_in = len(times_in)

    if tmin is not None and tmin < times_in[0]:
        warn('start time tmin=%0.2f s outside of the time scope of the data '
             '[%0.2f s, %0.2f s]' % (tmin, times_in[0], times_in[-1]))
    if tmax is not None and tmax > times_in[-1]:
        warn('stop time tmax=%0.2f s outside of the time scope of the data '
             '[%0.2f s, %0.2f s]' % (tmax, times_in[0], times_in[-1]))

    mask = _time_mask(times_in, tmin, tmax, sfreq=sfreq)
    tmin_idx, tmax_idx = np.where(mask)[0][[0, -1]]
    tmax_idx += 1
    tmin_true = times_in[tmin_idx]
    tmax_true = times_in[tmax_idx - 1]  # time of last point used

    times = times_in[tmin_idx:tmax_idx]
    n_times = len(times)

    logger.info('    using t=%0.3fs..%0.3fs for estimation (%d points)'
                % (tmin_true, tmax_true, n_times))

    # check that fmin corresponds to at least 5 cycles
    dur = float(n_times) / sfreq
    five_cycle_freq = 5. / dur
    if len(fmin) == 1 and fmin[0] == -np.inf:
        # we use the 5 cycle freq. as default
        fmin = np.array([five_cycle_freq])
    else:
        if np.any(fmin < five_cycle_freq):
            warn('fmin=%0.3f Hz corresponds to %0.3f < 5 cycles '
                 'based on the epoch length %0.3f sec, need at least %0.3f '
                 'sec epochs or fmin=%0.3f. Spectrum estimate will be '
                 'unreliable.' % (np.min(fmin), dur * np.min(fmin), dur,
                                  5. / np.min(fmin), five_cycle_freq))

    # compute frequencies to analyze based on number of samples,
    # sampling rate, specified wavelet frequencies and mode
    freqs = _compute_freqs(n_times, sfreq, cwt_freqs, mode)

    # compute the mask based on specified min/max and decimation factor
    freq_mask = _compute_freq_mask(freqs, fmin, fmax, fskip)

    # the frequency points where we compute connectivity
    freqs = freqs[freq_mask]
    n_freqs = len(freqs)

    # get the freq. indices and points for each band
    freq_idx_bands = [np.where((freqs >= fl) & (freqs <= fu))[0]
                      for fl, fu in zip(fmin, fmax)]
    freqs_bands = [freqs[freq_idx] for freq_idx in freq_idx_bands]

    # make sure we don't have empty bands
    for i, n_f_band in enumerate([len(f) for f in freqs_bands]):
        if n_f_band == 0:
            raise ValueError('There are no frequency points between '
                             '%0.1fHz and %0.1fHz. Change the band '
                             'specification (fmin, fmax) or the '
                             'frequency resolution.'
                             % (fmin[i], fmax[i]))
    if n_bands == 1:
        logger.info('    frequencies: %0.1fHz..%0.1fHz (%d points)'
                    % (freqs_bands[0][0], freqs_bands[0][-1],
                       n_freqs))
    else:
        logger.info('    computing connectivity for the bands:')
        for i, bfreqs in enumerate(freqs_bands):
            logger.info('     band %d: %0.1fHz..%0.1fHz '
                        '(%d points)' % (i + 1, bfreqs[0],
                                         bfreqs[-1], len(bfreqs)))
    if faverage:
        logger.info('    connectivity scores will be averaged for '
                    'each band')

    return (times, n_times, times_in, n_times_in, tmin_idx,
            tmax_idx, n_freqs, freq_mask, freqs, freqs_bands, freq_idx_bands,
            n_signals, warn_times)


def _assemble_spectral_params(mode, n_times, mt_adaptive, mt_bandwidth, sfreq,
                              mt_low_bias, cwt_n_cycles, cwt_freqs,
                              freqs, freq_mask):
    """Prepare time-frequency decomposition."""
    spectral_params = dict(
        eigvals=None, window_fun=None, wavelets=None)
    n_tapers = None
    n_times_spectrum = 0
    if mode == 'multitaper':
        window_fun, eigvals, mt_adaptive = _compute_mt_params(
            n_times, sfreq, mt_bandwidth, mt_low_bias, mt_adaptive)
        spectral_params.update(window_fun=window_fun, eigvals=eigvals)
    elif mode == 'fourier':
        logger.info('    using FFT with a Hanning window to estimate '
                    'spectra')
        spectral_params.update(window_fun=np.hanning(n_times), eigvals=1.)
    elif mode == 'cwt_morlet':
        logger.info('    using CWT with Morlet wavelets to estimate '
                    'spectra')

        # reformat cwt_n_cycles if we have removed some frequencies
        # using fmin, fmax, fskip
        cwt_n_cycles = np.array((cwt_n_cycles,), dtype=float).ravel()
        if len(cwt_n_cycles) > 1:
            if len(cwt_n_cycles) != len(cwt_freqs):
                raise ValueError('cwt_n_cycles must be float or an '
                                 'array with the same size as cwt_freqs')
            cwt_n_cycles = cwt_n_cycles[freq_mask]

        # get the Morlet wavelets
        spectral_params.update(
            wavelets=morlet(sfreq, freqs,
                            n_cycles=cwt_n_cycles, zero_mean=True))
        n_times_spectrum = n_times
    else:
        raise ValueError('mode has an invalid value')
    return spectral_params, mt_adaptive, n_times_spectrum, n_tapers


def _compute_spectral_methods_epochs(
        con_methods, epoch_block, epoch_idx, call_params, parallel,
        my_spectral_connectivity_epochs, n_jobs, n_times_in, times_in,
        warn_times
):
    """Compute CSD/PSD for spectral_connectivity_epochs... functions."""
    # check dimensions and time scale
    for this_epoch in epoch_block:
        _, _, _, warn_times = _get_and_verify_data_sizes(
            this_epoch, call_params["sfreq"], call_params["n_signals"],
            n_times_in, times_in, warn_times=warn_times)

    if n_jobs == 1:
        # no parallel processing
        for this_epoch in epoch_block:
            logger.info('    computing cross-spectral density for epoch %d'
                        % (epoch_idx + 1))
            # con methods and psd are updated inplace
            _epoch_spectral_connectivity(data=this_epoch, **call_params)
            epoch_idx += 1
    else:
        # process epochs in parallel
        logger.info(
            '    computing cross-spectral density for epochs %d..%d'
            % (epoch_idx + 1, epoch_idx + len(epoch_block)))

        out = parallel(my_spectral_connectivity_epochs(
                       data=this_epoch, **call_params)
                       for this_epoch in epoch_block)
        # do the accumulation
        for this_out in out:
            for _method, parallel_method in zip(con_methods, this_out[0]):
                _method.combine(parallel_method)
            if call_params["psd"] is not None:
                call_params["psd"] += this_out[1]

        epoch_idx += len(epoch_block)

    return epoch_idx

########################################################################
# Various connectivity estimators


class _AbstractConEstBase(object):
    """ABC for connectivity estimators."""

    def start_epoch(self):
        raise NotImplementedError('start_epoch method not implemented')

    def accumulate(self, con_idx, csd_xy):
        raise NotImplementedError('accumulate method not implemented')

    def combine(self, other):
        raise NotImplementedError('combine method not implemented')

    def compute_con(self, con_idx, n_epochs):
        raise NotImplementedError('compute_con method not implemented')

###############################################################################


_gc_methods = ['gc', 'gc_tr']


def _epoch_spectral_connectivity(data, sig_idx, tmin_idx, tmax_idx, sfreq,
                                 method, mode, window_fun, eigvals, wavelets,
                                 freq_mask, mt_adaptive, idx_map, n_cons,
                                 block_size, psd, accumulate_psd,
                                 con_method_types, con_methods, n_signals,
                                 n_signals_use, n_times, gc_n_lags,
                                 multivariate_con, accumulate_inplace=True):
    """Estimate connectivity for one epoch (see spectral_connectivity)."""
    if multivariate_con:
        n_con_signals = n_signals_use ** 2
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
            if multivariate_con:
                if "n_lags" in method_params:
                    # if it's a Granger causality method
                    con_methods.append(
                        mtype(n_signals_use, n_cons, n_freqs, n_times_spectrum,
                              gc_n_lags)
                    )
                else:
                    # if it's a coherence method
                    con_methods.append(
                        mtype(n_signals_use, n_cons, n_freqs, n_times_spectrum)
                    )
            else:
                con_methods.append(mtype(n_cons, n_freqs, n_times_spectrum))

    _check_option('mode', mode, ('cwt_morlet', 'multitaper', 'fourier'))
    if len(sig_idx) == n_signals:
        # we use all signals: use a slice for faster indexing
        sig_idx = slice(None, None)

    # compute tapered spectra
    x_t = list()
    this_psd = list()
    for this_data in data:
        if mode in ('multitaper', 'fourier'):
            if isinstance(this_data, _BaseSourceEstimate):
                _mt_spectra_partial = partial(_mt_spectra, dpss=window_fun,
                                              sfreq=sfreq)
                this_x_t = this_data.transform_data(
                    _mt_spectra_partial, idx=sig_idx, tmin_idx=tmin_idx,
                    tmax_idx=tmax_idx)
            else:
                this_x_t, _ = _mt_spectra(
                    this_data[sig_idx, tmin_idx:tmax_idx],
                    window_fun, sfreq)

            if mt_adaptive:
                # compute PSD and adaptive weights
                _this_psd, weights = _psd_from_mt_adaptive(
                    this_x_t, eigvals, freq_mask, return_weights=True)

                # only keep freqs of interest
                this_x_t = this_x_t[:, :, freq_mask]
            else:
                # do not use adaptive weights
                this_x_t = this_x_t[:, :, freq_mask]
                if mode == 'multitaper':
                    weights = np.sqrt(eigvals)[np.newaxis, :, np.newaxis]
                else:
                    # hack to so we can sum over axis=-2
                    weights = np.array([1.])[:, None, None]

                if accumulate_psd:
                    _this_psd = _psd_from_mt(this_x_t, weights)
        else:  # mode == 'cwt_morlet'
            if isinstance(this_data, _BaseSourceEstimate):
                cwt_partial = partial(cwt, Ws=wavelets, use_fft=True,
                                      mode='same')
                this_x_t = this_data.transform_data(
                    cwt_partial, idx=sig_idx, tmin_idx=tmin_idx,
                    tmax_idx=tmax_idx)
            else:
                this_x_t = cwt(this_data[sig_idx, tmin_idx:tmax_idx],
                               wavelets, use_fft=True, mode='same')
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
    if mode in ['multitaper', 'fourier']:
        for i in range(0, n_con_signals, block_size):
            n_extra = max(0, i + block_size - n_con_signals)
            con_idx = slice(i, i + block_size - n_extra)
            if mt_adaptive:
                csd = _csd_from_mt(x_t[idx_map[0][con_idx]],
                                   x_t[idx_map[1][con_idx]],
                                   weights[idx_map[0][con_idx]],
                                   weights[idx_map[1][con_idx]])
            else:
                csd = _csd_from_mt(x_t[idx_map[0][con_idx]],
                                   x_t[idx_map[1][con_idx]],
                                   weights, weights)

            for method in con_methods:
                method.accumulate(con_idx, csd)
    else:  # mode == 'cwt_morlet'  # reminder to add alternative TFR methods
        for i in range(0, n_con_signals, block_size):
            n_extra = max(0, i + block_size - n_con_signals)
            con_idx = slice(i, i + block_size - n_extra)
            # this codes can be very slow
            csd = (x_t[idx_map[0][con_idx]] *
                   x_t[idx_map[1][con_idx]].conjugate())

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
    interface_members = [m[0] for m in inspect.getmembers(_AbstractConEstBase)
                         if not m[0].startswith('_')]
    method_members = [m[0] for m in inspect.getmembers(method)
                      if not m[0].startswith('_')]

    for member in interface_members:
        if member not in method_members:
            return False, member
    return True, None


def _get_and_verify_data_sizes(data, sfreq, n_signals=None, n_times=None,
                               times=None, warn_times=True):
    """Get and/or verify the data sizes and time scales."""
    if not isinstance(data, (list, tuple)):
        raise ValueError('data has to be a list or tuple')
    n_signals_tot = 0
    # Sometimes data can be (ndarray, SourceEstimate) groups so in the case
    # where ndarray comes first, don't use it for times
    times_inferred = False
    for this_data in data:
        this_n_signals, this_n_times = this_data.shape
        if n_times is not None:
            if this_n_times != n_times:
                raise ValueError('all input time series must have the same '
                                 'number of time points')
        else:
            n_times = this_n_times
        n_signals_tot += this_n_signals

        if hasattr(this_data, 'times'):
            assert isinstance(this_data, _BaseSourceEstimate)
            this_times = this_data.times
            if times is not None and not times_inferred:
                if warn_times and not np.allclose(times, this_times):
                    with np.printoptions(threshold=4, linewidth=120):
                        warn('time scales of input time series do not match:\n'
                             f'{this_times}\n{times}')
                    warn_times = False
            else:
                times = this_times
        elif times is None:
            times_inferred = True
            times = _arange_div(n_times, sfreq)

    if n_signals is not None:
        if n_signals != n_signals_tot:
            raise ValueError('the number of time series has to be the same in '
                             'each epoch')
    n_signals = n_signals_tot

    return n_signals, n_times, times, warn_times


def _check_estimators(method, con_method_map):
    """Check construction of connectivity estimators."""
    con_method_types = list()
    for this_method in method:
        if this_method in con_method_map:
            con_method_types.append(con_method_map[this_method])
        elif isinstance(this_method, str):
            raise ValueError('%s is not a valid connectivity method' %
                             this_method)
        else:
            # support for custom class
            method_valid, msg = _check_method(this_method)
            if not method_valid:
                raise ValueError('The supplied connectivity method does '
                                 'not have the method %s' % msg)
            con_method_types.append(this_method)

    # if none of the comp_con functions needs the PSD, we don't estimate it
    accumulate_psd = any(
        this_method.accumulate_psd for this_method in con_method_types)

    return con_method_types, accumulate_psd


def _check_spectral_connectivity_epochs_settings(method, fmin, fmax, n_jobs,
                                                 verbose, con_method_map):
    """Check settings inputs for spectral_connectivity_epochs... functions."""
    if n_jobs != 1:
        parallel, my_epoch_spectral_connectivity, _ = parallel_func(
            _epoch_spectral_connectivity, n_jobs, verbose=verbose)
    else:
        parallel = None
        my_epoch_spectral_connectivity = None

    # format fmin and fmax and check inputs
    if fmin is None:
        fmin = -np.inf  # set it to -inf, so we can adjust it later

    fmin = np.array((fmin,), dtype=float).ravel()
    fmax = np.array((fmax,), dtype=float).ravel()
    if len(fmin) != len(fmax):
        raise ValueError('fmin and fmax must have the same length')
    if np.any(fmin > fmax):
        raise ValueError('fmax must be larger than fmin')
    n_bands = len(fmin)

    # assign names to connectivity methods
    if not isinstance(method, (list, tuple)):
        method = [method]  # make it a list so we can iterate over it

    # handle connectivity estimators
    con_method_types, accumulate_psd = _check_estimators(method,
                                                         con_method_map)

    return (fmin, fmax, n_bands, method, con_method_types, accumulate_psd,
            parallel, my_epoch_spectral_connectivity)


def _check_spectral_connectivity_epochs_data(data, sfreq, names):
    """Check data inputs for spectral_connectivity_epochs... functions."""
    if isinstance(data, BaseEpochs):
        names = data.ch_names
        times_in = data.times  # input times for Epochs input type
        sfreq = data.info['sfreq']

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
    else:
        events = None
        event_id = None
        times_in = None
        metadata = None
        if sfreq is None:
            raise ValueError('Sampling frequency (sfreq) is required with '
                             'array input.')

    return (names, times_in, sfreq, events, event_id, metadata)


def _store_results(
    con, patterns, method, freqs, faverage, freqs_bands, names, mode, indices,
    n_epochs, times, n_tapers, metadata, events, event_id, rank, gc_n_lags,
    n_signals
):
    """Store results in connectivity containers."""
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

    # number of nodes in the original data
    n_nodes = n_signals

    # create a list of connectivity containers
    conn_list = []
    for _con, _patterns, _method in zip(con, patterns, method):
        kwargs = dict(
            data=_con, patterns=_patterns, names=names, freqs=freqs,
            method=_method, n_nodes=n_nodes, spec_method=mode, indices=indices,
            n_epochs_used=n_epochs, freqs_used=freqs_used, times_used=times,
            n_tapers=n_tapers, metadata=metadata, events=events,
            event_id=event_id, rank=rank,
            n_lags=gc_n_lags if _method in _gc_methods else None)
        # create the connectivity container
        if mode in ['multitaper', 'fourier']:
            klass = SpectralConnectivity
        else:
            assert mode == 'cwt_morlet'
            klass = SpectroTemporalConnectivity
            kwargs.update(times=times)
        conn_list.append(klass(**kwargs))

    logger.info('[Connectivity computation done]')

    if len(method) == 1:
        # for a single method return connectivity directly
        conn_list = conn_list[0]

    return conn_list
