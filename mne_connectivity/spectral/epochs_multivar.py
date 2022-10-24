# Authors: Thomas S. Binns <thomas-samuel.binns@charite.de>
#          Tien Dung Nguyen <>
#          Richard M. Köhler <koehler.richard@charite.de>
#          Veronika Shamova <>
#          Mariia Mikhailenko <>
#          Orestis Sylianou <>
#          Jeroen Habets <>
#
# License: BSD (3-clause)

import inspect
from copy import deepcopy
import numpy as np
from mne import BaseEpochs
from mne.parallel import parallel_func
from mne.utils import logger

from ..base import SpectralConnectivity, SpectroTemporalConnectivity
from .epochs import (_assemble_spectral_params, _check_estimators,
                     _epoch_spectral_connectivity, _get_and_verify_data_sizes,
                     _get_n_epochs, _prepare_connectivity)


def multivar_spectral_connectivity_epochs(
    data, indices, names = None, method = "mic", sfreq = 2 * np.pi,
    mode = "multitaper", tmin = None, tmax = None, fmin = None, fmax = np.inf,
    fskip = 0, faverage = False, cwt_freqs = None, mt_bandwidth = None,
    mt_adaptive = False, mt_low_bias = True, cwt_n_cycles = 7,
    n_seed_components = None, n_target_components = None, gc_n_lags = 20,
    block_size = 1000, n_jobs = 1, verbose = None,
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
    (
        data, names, method, fmin, fmax, n_seed_components, n_target_components,
        parallel, my_epoch_spectral_connectivity, n_bands, con_method_types,
        events, event_id, times_in, metadata, sfreq, present_gc_methods,
        perform_svd
    ) = _sort_inputs(
        data, indices, names, method, sfreq, mode, fmin, fmax,
        n_seed_components, n_target_components, n_jobs, verbose
    )

    remaining_method_types = deepcopy(con_method_types)
    if present_gc_methods and perform_svd:
        # if singular value decomposition is being performed with Granger
        # causality, this has to be performed on the timeseries data for each
        # seed-target group separately.

        # finds the GC methods to compute
        use_method_types = [
            mtype for mtype in con_method_types if mtype.name in
            ["GC", "Net GC", "TRGC", "Net TRGC"]
        ]

        # creates an empty placeholder for non-GC connectivity results
        if use_method_types == con_method_types:
            con = []

        # performs SVD on the timeseries data for each seed-target group
        seed_target_data, n_seeds = _time_series_svd(
            data, indices, n_seed_components, n_target_components
        )

        # computes GC for each seed-target group
        n_gc_methods = len(present_gc_methods)
        svd_gc_con = [[] for x in range(n_gc_methods)]
        for gc_node_data, n_seed_comps in zip(seed_target_data, n_seeds):
            new_indices = (
                [np.arange(n_seed_comps).tolist()],
                [np.arange(n_seed_comps, gc_node_data.shape[1]).tolist()]
            )
            (
                con_methods, times, freqs_bands, freq_idx_bands, n_tapers,
                n_epochs, n_cons, n_freqs, n_signals, freqs, _
            ) = _compute_csd(
                gc_node_data, new_indices, sfreq, mode, tmin, tmax, fmin, fmax,
                fskip, faverage, cwt_freqs, mt_bandwidth, mt_adaptive,
                mt_low_bias, cwt_n_cycles, block_size, n_jobs, n_bands,
                use_method_types, parallel, my_epoch_spectral_connectivity,
                times_in, gc_n_lags
            )
            group_con, freqs_used, n_nodes = _compute_connectivity(
                con_methods, new_indices, n_seed_components,
                n_target_components, n_epochs, n_cons, faverage, n_freqs,
                n_bands, freq_idx_bands, freqs_bands, n_signals, freqs
            )
            [svd_gc_con[i].append(group_con[i]) for i in range(n_gc_methods)]
        svd_gc_con = [np.squeeze(np.array(method_con), 1) for method_con in svd_gc_con]

        # finds the methods still needing to be computed
        remaining_method_types = [
            mtype for mtype in con_method_types if mtype not in use_method_types
        ]

    if remaining_method_types:
        # if no singular value decomposition is being performed or Granger
        # causality is not being computed, the cross-spectral density can be
        # computed as normal on all channels and used for the connectivity
        # computations of each seed-target group.

        # creates an empty placeholder for SVD GC connectivity results
        if remaining_method_types == con_method_types:
            svd_gc_con = []
            use_method_types = []

        # computes connectivity
        (
            con_methods, times, freqs_bands, freq_idx_bands, n_tapers,
            n_epochs, n_cons, n_freqs, n_signals, freqs, remapped_indices
        ) = _compute_csd(
            data, indices, sfreq, mode, tmin, tmax, fmin, fmax, fskip, faverage,
            cwt_freqs, mt_bandwidth, mt_adaptive, mt_low_bias, cwt_n_cycles,
            block_size, n_jobs, n_bands, remaining_method_types, parallel,
            my_epoch_spectral_connectivity, times_in, gc_n_lags
        )
        con, freqs_used, n_nodes = _compute_connectivity(
            con_methods, remapped_indices, n_seed_components, n_target_components,
            n_epochs, n_cons, faverage, n_freqs, n_bands, freq_idx_bands,
            freqs_bands, n_signals, freqs
        )

    if svd_gc_con and con:
        # combines SVD GC and non-SVD GC results
        con.extend(svd_gc_con)
        # orders the results according to the order they were called
        methods_order = [
            *[mtype.name for mtype in use_method_types],
            *[mtype.name for mtype in remaining_method_types]
        ]
        con = [con[methods_order.index(mtype.name)] for mtype in con_method_types]
    elif svd_gc_con and not con:
        # stored SVD GC results
        con = svd_gc_con
        # finds the remapped indices of non-SVD data
        unique_indices = np.unique(sum(sum(indices, []), []))
        remapping = {ch_i: sig_i for sig_i, ch_i in enumerate(unique_indices)}
        remapped_indices = [[[remapping[idx] for idx in idcs] for idcs in
                             indices_group] for indices_group in indices]

    return _store_connectivity(
        con, method, names, freqs, n_nodes, mode, remapped_indices, n_epochs,
        freqs_used, times, n_tapers, metadata, events, event_id
    )


def _sort_inputs(
    data, indices, names, method, sfreq, mode, fmin, fmax, n_seed_components,
    n_target_components, n_jobs, verbose
):
    """Checks the format of the input parameters to the
    "multivar_spectral_connectivity_epochs" function and returns relevant
    variables."""
    # establishes parallelisation
    if n_jobs != 1:
        parallel, my_epoch_spectral_connectivity, _ = \
            parallel_func(_epoch_spectral_connectivity, n_jobs,
                          verbose=verbose)
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
    con_method_types, _, _, _ = _check_estimators(method, mode)

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

    # handle indices
    if indices is None:
        raise ValueError("indices must be specified, got `None`.")

    n_seeds = len(indices[0])
    n_targets = len(indices[1])
    if n_seeds != n_targets:
        raise ValueError(
            f"The number of seeds ({n_seeds}) and targets ({n_targets}) must  "
            "match."
        )

    for seeds, targets in zip(indices[0], indices[1]):
        if set.intersection(set(seeds), set(targets)):
            raise ValueError(
                "There are common indices present in the seeds and targets for "
                "a single connectivity index, however multivariate "
                "connectivity between shared channels is not allowed."
            )

    perform_svd = False
    if n_seed_components is None:
        n_seed_components = tuple([None] * n_seeds)
    else:
        if n_seeds != len(n_seed_components):
            raise ValueError(
            "n_seed_components must have the same length as specified seeds."
            f" Got: {len(n_seed_components)} seed components and {n_seeds}"
            "seeds."
            )
        for n_comps, chs in zip(n_seed_components, indices[0]):
            if n_comps:
                if n_comps > len(chs) and n_comps <= 0:
                    raise ValueError(
                        f"The number of components to take ({n_comps}) cannot "
                        "be greater than the number of channels in that seed "
                        f"({len(chs)}) and must be greater than 0."
                    )
                perform_svd = True

    if n_target_components is None:
        n_target_components = tuple([None] * n_targets)
    else:
        if n_targets != len(n_target_components):
            raise ValueError(
            "n_target_components must have the same length as specified"
            f" targets. Got: {len(n_target_components)} target components and"
            f" {n_targets} targets."
            )
        for n_comps, chs in zip(n_target_components, indices[1]):
            if n_comps:
                if n_comps is not None and n_comps > len(chs) and n_comps <= 0:
                    raise ValueError(
                        f"The number of components to take ({n_comps}) cannot "
                        "be greater than the number of channels in that target "
                        f"({len(chs)}) and must be greater than 0."
                    )
                perform_svd = True

    # handle Granger causality methods
    present_gc_methods = [
        con_method for con_method in method
        if con_method in ["gc", "net_gc", "trgc", "net_trgc"]
    ]

    return (
        data, names, method, fmin, fmax, n_seed_components, n_target_components,
        parallel, my_epoch_spectral_connectivity, n_bands, con_method_types,
        events, event_id, times_in, metadata, sfreq, present_gc_methods,
        perform_svd
    )

def _time_series_svd(data, indices, n_seed_components, n_target_components):
    """Performs a single value decomposition on the timeseries data for each set
    of seed-target pairs."""
    if isinstance(data, BaseEpochs):
        epochs = data.get_data(picks=data.ch_names)
    else:
        epochs = data

    seed_target_data = []
    n_seeds = []
    for seeds, targets, n_seed_comps, n_target_comps in \
        zip(indices[0], indices[1], n_seed_components, n_target_components):

        if n_seed_comps: # SVD seed data
            v_seeds = (
                np.linalg.svd(epochs[:, seeds, :], full_matrices=False)[2]
                [:, :n_seed_comps, :]
            )
        else: # use unaltered seed data
            v_seeds = epochs[:, seeds, :]
        n_seeds.append(v_seeds.shape[1])

        if n_target_comps: # SVD target data
            v_targets = (
                np.linalg.svd(epochs[:, targets, :], full_matrices=False)[2]
                [:, :n_target_comps, :]
            )
        else: # use unaltered target data
            v_targets = epochs[:, targets, :]

        seed_target_data.append(np.append(v_seeds, v_targets, axis=1))

    return seed_target_data, n_seeds


def _compute_csd(
    data, indices, sfreq, mode, tmin, tmax, fmin, fmax, fskip, faverage,
    cwt_freqs, mt_bandwidth, mt_adaptive, mt_low_bias, cwt_n_cycles, block_size,
    n_jobs, n_bands, con_method_types, parallel, my_epoch_spectral_connectivity,
    times_in, gc_n_lags
):
    """Computes the cross-spectral density of the data in preparation for the
    multivariate connectivity computations."""
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
             n_signals, _, warn_times) = _prepare_connectivity(
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

            # map indices to unique indices
            unique_indices = np.unique(sum(sum(indices, []), []))
            remapping = {ch_i: sig_i for sig_i, ch_i in
                         enumerate(unique_indices)}
            remapped_indices = [[[remapping[idx] for idx in idcs] for idcs in
                                indices_group] for indices_group in indices]

            # unique signals for which we actually need to compute CSD etc.
            sig_idx = np.unique(sum(sum(remapped_indices, []), []))
            n_signals = len(sig_idx)

            # gets seed-target indices for CSD
            idx_map = [np.repeat(sig_idx, len(sig_idx)),
                       np.tile(sig_idx, len(sig_idx))]

            # create instances of the connectivity estimators
            con_methods = []
            for mtype in con_method_types:
                if "n_lags" in list(inspect.signature(mtype).parameters):
                    con_methods.append(
                        mtype(n_signals, n_cons, n_freqs, n_times_spectrum,
                              gc_n_lags)
                    )
                else:
                    con_methods.append(
                        mtype(n_signals, n_cons, n_freqs, n_times_spectrum)
                    )

            metrics_str = ', '.join([meth.name for meth in con_methods])
            logger.info('    the following metrics will be computed: %s'
                        % metrics_str)

        # check dimensions and time scale
        for this_epoch in epoch_block:
            _, _, _, warn_times = _get_and_verify_data_sizes(
                this_epoch, sfreq, n_signals, n_times_in, times_in,
                warn_times=warn_times)

        call_params = dict(
            sig_idx=sig_idx, tmin_idx=tmin_idx, tmax_idx=tmax_idx, sfreq=sfreq,
            mode=mode, freq_mask=freq_mask, idx_map=idx_map,
            block_size=block_size, psd=None, accumulate_psd=False,
            mt_adaptive=mt_adaptive, con_method_types=con_method_types,
            con_methods=con_methods if n_jobs == 1 else None,
            n_signals=n_signals, n_times=n_times, gc_n_lags=gc_n_lags,
            accumulate_inplace=True if n_jobs == 1 else False
        )
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

            epoch_idx += len(epoch_block)

    n_epochs = epoch_idx

    return (
        con_methods, times, freqs_bands, freq_idx_bands, n_tapers, n_epochs,
        n_cons, n_freqs, n_signals, freqs, remapped_indices
    )

def _compute_connectivity(
    con_methods, indices, n_seed_components, n_target_components, n_epochs,
    n_cons, faverage, n_freqs, n_bands, freq_idx_bands, freqs_bands, n_signals,
    freqs
):
    """Computes the multivariate connectivity results."""
    # compute final connectivity scores
    con = list()
    for conn_method in con_methods:
        if conn_method.name in ["GC", "Net GC", "TRGC", "Net TRGC"]:
            conn_method.compute_con(indices[0], indices[1], n_epochs)
        else:
            conn_method.compute_con(
                indices[0], indices[1], n_seed_components, n_target_components,
                n_epochs
            )

        # get the connectivity scores
        this_con = conn_method.con_scores

        assert (this_con.shape[0] == n_cons), \
            ('The first dimension of connectivity scores does not match the '
            'number of connections. Please contact the mne-connectivity ' 
            'developers.')

        if faverage:
            assert (this_con.shape[1] == n_freqs), \
            ('The second dimension of connectivity scores does not match the '
            'number of frequencies. Please contact the mne-connectivity ' 
            'developers.')
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

    return con, freqs_used, n_nodes

def _store_connectivity(
    con, method, names, freqs, n_nodes, mode, indices, n_epochs, freqs_used,
    times, n_tapers, metadata, events, event_id
):
    """Stores multivariate connectivity results in an mne-connectivity
    object."""
    # create a list of connectivity containers
    conn_list = []
    for _con, _method in zip(con, method):
        kwargs = dict(data=_con,
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

    if len(method) == 1:
        # for a single method return connectivity directly
        conn_list = conn_list[0]

    return conn_list