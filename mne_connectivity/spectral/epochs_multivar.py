# Authors: Thomas S. Binns <>
#          Mariia Mikhailenko <>
#          Tien Dung Nguyen <>
#          Veronika Shamova <>
#          Orestis Sylianou <>
#          Jeroen Habets <>
#          Richard M. Köhler <koehler.richard@charite.de>
#
# License: BSD (3-clause)

import numpy as np
from mne import BaseEpochs
from mne.parallel import parallel_func
from mne.utils import logger
from scipy import linalg as spla

from ..base import SpectralConnectivity, SpectroTemporalConnectivity
from ..utils import check_indices, fill_doc
from .epochs import (_assemble_spectral_params, _check_estimators,
                     _epoch_spectral_connectivity, _get_and_verify_data_sizes,
                     _get_n_epochs, _prepare_connectivity)


########################################################################
# Various connectivity estimators


@fill_doc
def multivar_spectral_connectivity_epochs(
    data,
    indices,
    names = None,
    method = "mic",
    sfreq = 2 * np.pi,
    mode = "multitaper",
    tmin = None,
    tmax = None,
    fmin = 0.0,
    fmax = np.inf,
    fskip = 0, 
    faverage = False, 
    cwt_freqs = None,
    mt_bandwidth = None,
    mt_adaptive = False,
    mt_low_bias = True,
    cwt_n_cycles = 7.0,
    n_seed_components = None,
    n_target_components = None,
    block_size = 1000, 
    n_jobs = 1,
    verbose = None,
):
    """Compute frequency-domain multivariate connectivity measures.

    The connectivity method(s) are specified using the "method" parameter. All
    methods are based on estimates of the cross-spectral densities (CSD) Sxy.

    Based on the "spectral_connectivity_epochs" function of the
    "mne-connectivity" package.

    PARAMETERS
    ----------
    data : BaseEpochs | array-like
    -   Data to compute connectivity on. If array-like, must have the dimensions
        [epochs x signals x timepoints].

    indices : tuple of tuple of array-like of int
    -   Two tuples of arrays with indices of connections for which to compute
        connectivity.

    names : list | None; default None
    -   Names of the channels in the data. If "data" is an Epochs object, these
        channel names will override those in the object.

    method : str | list of str; default "mic"
    -   Connectivity measure(s) to compute. These can be ['mic', 'mim', 'gc',
        'net_gc', 'trgc', 'net_trgc'].

    sfreq : float; default 6.283185307179586
    -   Sampling frequency of the data. Only used if "data" is array-like.

    mode : str; default "multitaper"
    -   Cross-spectral estimation method. Can be 'fourier', 'multitaper', or
        'cwt_wavelet'.

    t0 : float; default 0.0
    -   Time of the first sample relative to the onset of the epoch, in seconds.
        Only used if "data" is an array.

    tmin : float | None; default None
    -   The time at which to start computing connectivity, in seconds. If None,
        starts from the first sample.

    tmax : float | None; default None
    -   The time at which to stop computing connectivity, in seconds. If None,
        ends with the final sample.

    fmt_fmin : float; default 0.0
    -   Minumum frequency of interest, in Hz. Only used if "mode" is 'fourier'
        or 'multitaper'.

    fmt_fmax : float; default infinity
    -   Maximum frequency of interest, in Hz. Only used if "mode" is 'fourier'
        or 'multitaper'.

    cwt_freqs : list of float | None; default None
    -   The frequencies of interest, in Hz. If "mode" is 'cwt_morlet', this
        cannot be None. Only used if "mode" if 'cwt_morlet'.

    fmt_n_fft : int | None; default None
    -   Length of the FFT. If None, the exact number of samples between "tmin"
        and "tmax" will be used. Only used if "mode" is 'fourier' or
        'multitaper'.

    cwt_use_fft : bool; default True
    -   Whether to use FFT-based convolution to compute the wavelet transform.
        Only used if "mode" is 'cwt_morlet'.

    mt_bandwidth : float | None; default None
    -   Bandwidth of the multitaper windowing function, in Hz. Only used if
        "mode" if 'multitaper'.

    mt_adaptive : bool; default False
    -   Whether or not to use adaptive weights to combine the tapered spectra
        into the power spectral density. Only used if "mode" if 'multitaper'.

    mt_low_bias : bool; default True
    -   Whether or not to only use tapers with over 90% spectral concentration
        within the bandwidth. Only used if "mode" if 'multitaper'.

    cwt_n_cycles : float | list of float; default 7.0
    -   Number of cycles to use when constructing the Morlet wavelets. Can be a
        single number, or one per frequency. Only used if "mode" if
        'cwt_morlet'.

    cwt_decim : int | slice; default 1
    -   To redice memory usage, decimation factor during time-frequency
        decomposition. Default to 1 (no decimation). If int, uses
        tfr[..., ::"decim"]. If slice, used tfr[..., "decim"]. Only used if
        "mode" is 'cwt_morlet'.

    n_seed_components : tuple of int or None | None; default None
    -   Dimensionality reduction parameter specifying the number of seed
        components to extract from the single value decomposition of the seed
        channels' data for each connectivity node. If None, or if an individual
        entry is None, no dimensionality reduction is performed.

    n_target_components : tuple of int or None | None; default None
    -   Dimensionality reduction parameter specifying the number of target
        components to extract from the single value decomposition of the target
        channels' data for each connectivity node. If None, or if an individual
        entry is None, no dimensionality reduction is performed.

    gc_n_lags : int; default 20
    -   The number of lags to use when computing the autocovariance sequence
        from the cross-spectral density. Only used if "method" is 'gc',
        'net_gc', 'trgc', or 'net_trgc'.

    n_jobs : int; default 1
    -   Number of jobs to run in parallel when computing the cross-spectral
        density.

    verbose : bool | str | int | None; default None
    -   Whether or not to print information about the status of the connectivity
        computations. See MNE's logging information for further details.

    RETURNS
    -------
    results : SpectralConnectivity | list[SpectralConnectivity]
    -   The connectivity results as a single SpectralConnectivity object (if
        only one method is called) or a list of SpectralConnectivity objects (if
        multiple methods are called, where each object is the results for the
        corresponding entry in "method").
    """
    if n_jobs != 1:
        parallel, my_epoch_spectral_connectivity, _ = \
            parallel_func(_epoch_spectral_connectivity, n_jobs,
                          verbose=verbose)

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
    (con_method_types, n_methods, accumulate_psd,
     _) = _check_estimators(method=method, mode=mode)

    events = None
    event_id = None
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
        times_in = None
        metadata = None

    # loop over data; it could be a generator that returns
    # (n_signals x n_times) arrays or SourceEstimates
    epoch_idx = 0
    logger.info('Connectivity computation...')
    warn_times = True
    for epoch_block in _get_n_epochs(data, n_jobs):
        if epoch_idx == 0:
            # initialize everything times and frequencies
            (n_cons, times, n_times, times_in, n_times_in, tmin_idx,
             tmax_idx, n_freqs, freq_mask, freqs, freqs_bands, freq_idx_bands,
             n_signals, indices_use, warn_times) = _prepare_connectivity(
                epoch_block=epoch_block, times_in=times_in,
                tmin=tmin, tmax=tmax, fmin=fmin, fmax=fmax, sfreq=sfreq,
                indices=indices, mode=mode, fskip=fskip, n_bands=n_bands,
                cwt_freqs=cwt_freqs, faverage=faverage)

            # get the window function, wavelets, etc for different modes
            (spectral_params, mt_adaptive, n_times_spectrum,
             n_tapers) = _assemble_spectral_params(
                mode=mode, n_times=n_times, mt_adaptive=mt_adaptive,
                mt_bandwidth=mt_bandwidth, sfreq=sfreq,
                mt_low_bias=mt_low_bias, cwt_n_cycles=cwt_n_cycles,
                cwt_freqs=cwt_freqs, freqs=freqs, freq_mask=freq_mask)

            # unique signals for which we actually need to compute CSD etc.
            sig_idx = np.unique([idx for indices_group in indices for idcs in
                indices_group for idx in idcs])

            # map indices to unique indices
            idx_map = [np.searchsorted(sig_idx, idx) for idcs in indices_use for idx in idcs]

            # None of the implemented multivariate methods need PSD
            psd = None

            # create instances of the connectivity estimators
            con_methods = [mtype(n_cons, n_freqs, n_times_spectrum)
                           for mtype in con_method_types]

            sep = ', '
            metrics_str = sep.join([meth.name for meth in con_methods])
            logger.info('    the following metrics will be computed: %s'
                        % metrics_str)

        # check dimensions and time scale
        for this_epoch in epoch_block:
            _, _, _, warn_times = _get_and_verify_data_sizes(
                this_epoch, sfreq, n_signals, n_times_in, times_in,
                warn_times=warn_times)

        call_params = dict(
            sig_idx=sig_idx, tmin_idx=tmin_idx,
            tmax_idx=tmax_idx, sfreq=sfreq, mode=mode,
            freq_mask=freq_mask, idx_map=idx_map, block_size=block_size,
            psd=psd, accumulate_psd=accumulate_psd,
            mt_adaptive=mt_adaptive,
            con_method_types=con_method_types,
            con_methods=con_methods if n_jobs == 1 else None,
            n_signals=n_signals, n_times=n_times,
            accumulate_inplace=True if n_jobs == 1 else False)
        call_params.update(**spectral_params)

        if n_jobs == 1:
            # no parallel processing
            for this_epoch in epoch_block:
                logger.info('    computing connectivity for epoch %d'
                            % (epoch_idx + 1))
                # con methods and psd are updated inplace
                _epoch_spectral_connectivity(data=this_epoch, **call_params)
                epoch_idx += 1
        else:
            # process epochs in parallel
            logger.info('    computing connectivity for epochs %d..%d'
                        % (epoch_idx + 1, epoch_idx + len(epoch_block)))

            out = parallel(my_epoch_spectral_connectivity(
                           data=this_epoch, **call_params)
                           for this_epoch in epoch_block)
            # do the accumulation
            for this_out in out:
                for _method, parallel_method in zip(con_methods, this_out[0]):
                    _method.combine(parallel_method)
                if accumulate_psd:
                    psd += this_out[1]

            epoch_idx += len(epoch_block)

    n_epochs = epoch_idx

    # compute final connectivity scores
    con = list()
    for conn_method in con_methods:
        conn_method.compute_con(
            seeds, targets, n_seed_components, n_target_components, n_epochs
        )

        # get the connectivity scores
        this_con = conn_method.con_scores

        if this_con.shape[0] != n_cons:
            raise ValueError('First dimension of connectivity scores must be '
                             'the same as the number of connections')
        if faverage:
            if this_con.shape[1] != n_freqs:
                raise ValueError('2nd dimension of connectivity scores must '
                                 'be the same as the number of frequencies')
            con_shape = (n_cons, n_bands) + this_con.shape[2:]
            this_con_bands = np.empty(con_shape, dtype=this_con.dtype)
            for band_idx in range(n_bands):
                this_con_bands[:, band_idx] =\
                    np.mean(this_con[:, freq_idx_bands[band_idx]], axis=1)
            this_con = this_con_bands

        con.append(this_con)

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

    # number of nodes in the original data,
    n_nodes = n_signals

    # create a list of connectivity containers
    conn_list = []
    for _con in con:
        kwargs = dict(data=_con,
                      names=names,
                      freqs=freqs,
                      method=method,
                      n_nodes=n_nodes,
                      spec_method=mode,
                      indices=indices,
                      n_epochs_used=n_epochs,
                      freqs_used=freqs_used,
                      times_used=times,
                      n_tapers=n_tapers,
                      metadata=metadata,
                      events=events,
                      event_id=event_id
                      )
        # create the connectivity container
        if mode in ['multitaper', 'fourier']:
            klass = SpectralConnectivity
        else:
            assert mode == 'cwt_morlet'
            klass = SpectroTemporalConnectivity
            kwargs.update(times=times)
        conn_list.append(klass(**kwargs))

    logger.info('[Connectivity computation done]')

    if n_methods == 1:
        # for a single method return connectivity directly
        conn_list = conn_list[0]

    return conn_list