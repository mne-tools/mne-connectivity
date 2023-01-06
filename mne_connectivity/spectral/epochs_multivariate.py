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
from mne.utils import logger, ProgressBar
from ..base import (
    MultivariateSpectralConnectivity, MultivariateSpectroTemporalConnectivity
)
from .epochs import (
    _assemble_spectral_params, _check_estimators, _epoch_spectral_connectivity,
    _get_and_verify_data_sizes, _get_n_epochs, _prepare_connectivity
)


def multivariate_spectral_connectivity_epochs(
    data, indices, names = None, method = "mic", sfreq = 2 * np.pi,
    mode = "multitaper", tmin = None, tmax = None, fmin = None, fmax = np.inf,
    fskip = 0, faverage = False, cwt_freqs = None, mt_bandwidth = None,
    mt_adaptive = False, mt_low_bias = True, cwt_n_cycles = 7.0,
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

    sfreq : float; default 2 * pi
    -   Sampling frequency of the data. Only used if "data" is array-like.

    mode : str; default "multitaper"
    -   Cross-spectral estimation method. Can be 'fourier', 'multitaper', or
        'cwt_wavelet'.

    tmin : float | None; default None
    -   The time at which to start computing connectivity, in seconds. If None,
        starts from the first sample.

    tmax : float | None; default None
    -   The time at which to stop computing connectivity, in seconds. If None,
        ends with the final sample.

    fmin : float; default 0.0
    -   Minumum frequency of interest, in Hz. Only used if "mode" is 'fourier'
        or 'multitaper'.

    fmax : float; default infinity
    -   Maximum frequency of interest, in Hz. Only used if "mode" is 'fourier'
        or 'multitaper'.
    
    fskip : int; default 0
    -   Omit every “(fskip + 1)-th” frequency bin to decimate in frequency
        domain.
    
    faverage : bool; default False
    -   Average connectivity scores for each frequency band. If True, the output
        freqs will be a list with arrays of the frequencies that were averaged.

    cwt_freqs : list of float | None; default None
    -   The frequencies of interest, in Hz. If "mode" is 'cwt_morlet', this
        cannot be None. Only used if "mode" if 'cwt_morlet'.

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

    n_seed_components : list of int or str or None | None; default None
    -   Dimensionality reduction parameter specifying the number of seed
        components to extract from the single value decomposition of the seed
        channels' data for each connectivity node. If an individual entry is a
        str with value "rank", the rank of the seed data will be computed and
        this number of components taken. If None, or if an individual entry is
        None, no dimensionality reduction is performed.

    n_target_components : list of int or str or None | None; default None
    -   Dimensionality reduction parameter specifying the number of seed
        components to extract from the single value decomposition of the target
        channels' data for each connectivity node. If an individual entry is a
        str with value "rank", the rank of the target data will be computed and
        this number of components taken. If None, or if an individual entry is
        None, no dimensionality reduction is performed.

    gc_n_lags : int; default 20
    -   The number of lags to use when computing the autocovariance sequence
        from the cross-spectral density. Only used if "method" is 'gc',
        'net_gc', 'trgc', or 'net_trgc'.
    
    block_size : int; default 1000
    -   How many cross-spectral density entries to compute at once (higher
        numbers are faster but require more memory).

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
        parallel, epoch_spectral_connectivity, n_bands, con_method_types,
        events, event_id, times_in, metadata, sfreq, present_gc_methods,
        perform_svd
    ) = _sort_inputs(
        data, indices, names, method, sfreq, mode, fmin, fmax,
        n_seed_components, n_target_components, n_jobs, verbose
    )

    call_params = dict(
        sfreq=sfreq, mode=mode, tmin=tmin, tmax=tmax, fmin=fmin, fmax=fmax,
        fskip=fskip, faverage=faverage, cwt_freqs=cwt_freqs,
        mt_bandwidth=mt_bandwidth, mt_adaptive=mt_adaptive,
        mt_low_bias=mt_low_bias, cwt_n_cycles=cwt_n_cycles,
        block_size=block_size, n_jobs=n_jobs, n_bands=n_bands,
        parallel=parallel, times_in=times_in, gc_n_lags=gc_n_lags,
        epoch_spectral_connectivity=epoch_spectral_connectivity
    )

    # if singular value decomposition is being performed with Granger causality,
    # this has to be performed on the timeseries data for each seed-target
    # connection separately
    if present_gc_methods and perform_svd:
        (
            gc_svd_method_types, non_gc_non_svd_method_types, gc_con, gc_topo,
            times, n_tapers, n_epochs, freqs, freqs_used, n_nodes,
            remapped_indices
        ) = _handle_gc_with_svd_connectivity(
            data=data, indices=indices, n_seed_components=n_seed_components,
            n_target_components=n_target_components, perform_svd=perform_svd,
            con_method_types=con_method_types,
            present_gc_methods=present_gc_methods, call_params=call_params
        )
        con = []
        topo = []
    else:
        non_gc_non_svd_method_types = con_method_types
        gc_svd_method_types = []
        gc_con = []
        gc_topo = []

    # if no singular value decomposition is being performed or Granger
    # causality is not being computed, the cross-spectral density can be
    # computed as normal on all channels and used for the connectivity
    # computations of each seed-target connection together
    if non_gc_non_svd_method_types:
        (
            con, topo, times, n_tapers, n_epochs, freqs, freqs_used, n_nodes,
            remapped_indices
        ) = _handle_non_gc_non_svd_connectivity(
            data=data, indices=indices, n_seed_components=n_seed_components,
            n_target_components=n_target_components,
            con_method_types=non_gc_non_svd_method_types,
            call_params=call_params
        )
    
    # combines the GC SVD and non-GC, non-SVD results together
    con, topo, remapped_indices = _collate_connectivity_results(
        con=con, gc_con=gc_con, topo=topo, gc_topo=gc_topo,
        non_gc_non_svd_method_types=non_gc_non_svd_method_types,
        gc_svd_method_types=gc_svd_method_types,
        con_method_types=con_method_types, indices=indices,
        remapped_indices=remapped_indices
    )

    return _store_connectivity(
        con, topo, method, names, freqs, n_nodes, mode, remapped_indices,
        n_epochs, freqs_used, times, n_tapers, metadata, events, event_id
    )


def _sort_inputs(
    data, indices, names, method, sfreq, mode, fmin, fmax, n_seed_components,
    n_target_components, n_jobs, verbose
):
    """Checks the format of the input parameters to the
    "multivariate_spectral_connectivity_epochs" function and returns relevant
    variables."""
    # establishes parallelisation (if n_jobs > 1, else returns the standard
    # function)
    parallel, epoch_spectral_connectivity, _ = (
        parallel_func(_epoch_spectral_connectivity, n_jobs, verbose=verbose)
    )

    fmin, fmax, n_bands = _sort_freq_inputs(fmin=fmin, fmax=fmax)

    method, con_method_types, present_gc_methods = _sort_estimator_inputs(
        method=method, mode=mode
    )

    names, times_in, sfreq, events, event_id, metadata, = _sort_data_info(
        data=data, names=names, sfreq=sfreq
    )

    n_cons = _sort_indices_inputs(indices=indices)

    perform_svd, n_seed_components, n_target_components = _sort_svd_inputs(
        data=data, indices=indices, n_cons=n_cons,
        n_seed_components=n_seed_components,
        n_target_components=n_target_components, nonzero_tol=1e-10
    )

    return (
        data, names, method, fmin, fmax, n_seed_components, n_target_components,
        parallel, epoch_spectral_connectivity, n_bands, con_method_types,
        events, event_id, times_in, metadata, sfreq, present_gc_methods,
        perform_svd
    )

def _sort_freq_inputs(fmin, fmax):
    """Formats frequency-related inputs and checks they are appropriate."""
    if fmin is None:
        fmin = -np.inf  # set it to -inf, so we can adjust it later

    fmin = np.array((fmin,), dtype=float).ravel()
    fmax = np.array((fmax,), dtype=float).ravel()
    if len(fmin) != len(fmax):
        raise ValueError('fmin and fmax must have the same length')
    if np.any(fmin > fmax):
        raise ValueError('fmax must be larger than fmin')

    n_bands = len(fmin)

    return fmin, fmax, n_bands

def _sort_estimator_inputs(method, mode):
    """Assign names to connectivity methods, check the methods and mode are
    recognised, and finds which Granger causality methods are being called."""
    if not isinstance(method, (list, tuple)):
        method = [method]  # make it a list so we can iterate over it

    con_method_types, _, _, _ = _check_estimators(method, mode)

    # find which Granger causality methods are being called
    present_gc_methods = [
        con_method for con_method in method
        if con_method in ["gc", "net_gc", "trgc", "net_trgc"]
    ]

    return method, con_method_types, present_gc_methods

def _sort_data_info(data, names, sfreq):
    """Extracts information stored in the data if it is an Epochs object,
    otherwise return this information as `None`."""
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
        events = None
        event_id = None
        metadata = None
    
    return names, times_in, sfreq, events, event_id, metadata

def _sort_indices_inputs(indices):
    """Checks that the indices are appropriate and returns the number of seeds
    ans targets in each connection."""
    if indices is None:
        raise ValueError('indices must be specified, got `None`.')

    if len(indices[0]) != len(indices[1]):
        raise ValueError(
            f'The number of seeds ({len(indices[0])}) and targets '
            f'({len(indices[1])}) must match.'
        )
    n_cons = len(indices[0])

    for seeds, targets in zip(indices[0], indices[1]):
        if not isinstance(seeds, list) or not isinstance(targets, list):
            raise TypeError(
                'Seeds and targets for each connection must be given as a list '
                'of ints.'
            )
        if (
            not all(isinstance(seed, int) for seed in seeds) or
            not all(isinstance(target, int) for target in targets)
        ):
            raise TypeError(
                'Seeds and targets for each connection must be given as a list '
                'of ints.'
            )
        if set.intersection(set(seeds), set(targets)):
            raise ValueError(
                'There are common indices present in the seeds and targets for '
                'a single connection, however multivariate connectivity '
                'between shared channels is not allowed.'
            )
    
    return n_cons

def _sort_svd_inputs(
    data, indices, n_cons, n_seed_components, n_target_components,
    nonzero_tol=1e-10
):
    """Checks that the SVD parameters are appropriate and finds the correct
    dimensionality reduction settings to use, if applicable.
    
    This involves the rank of the data being computed based its non-zero
    singular values. We use a cut-off of 1e-10 by default to determine when a
    value is non-zero, as using numpy's default cut-off is too liberal (i.e.
    low) for our purposes where we need to be stricter.
    """
    # finds if any SVD has been requested for seeds and/or targets
    perform_svd = False
    for n_components in (n_seed_components, n_target_components):
        if n_components is None:
            n_components = [None for _ in range(n_cons)]
        else:
            if not isinstance(n_components, list):
                raise TypeError(
                    'n_seed_components and n_target_components must be lists'
                )
            if n_cons != len(n_components):
                raise ValueError(
                    'n_seed_components and n_target_components must have the '
                    'same length as specified the number of connections in '
                    f'indices. Got: {len(n_components)} components and '
                    f'{n_cons} connections'
                )
            if not perform_svd and any(
                n_comps is not None for n_comps in n_components
            ):
                perform_svd = True

    # if SVD is requested, extract the data and perform subsequent checks
    if perform_svd:
        if isinstance(data, BaseEpochs):
            epochs = data.get_data(picks=data.ch_names)
        else:
            epochs = data
    
        for group_i, n_components in enumerate(
            (n_seed_components, n_target_components)
        ):
            if any(n_comps is not None for n_comps in n_components):
                index_i = 0
                for n_comps, chs in zip(n_components, indices[group_i]):
                    if isinstance(n_comps, int):
                        if n_comps > len(chs) or n_comps <= 0:
                            raise ValueError(
                                f'The number of components to take ({n_comps}) '
                                'cannot be greater than the number of channels '
                                f'in that seed/target ({len(chs)}) and must be '
                                'greater than 0'
                            )
                    elif isinstance(n_comps, str):
                        if n_comps != 'rank':
                            raise ValueError(
                                'n_seed_components and n_target_components '
                                'must be lists of `None`, an `int`, or the '
                                'string "rank"'
                            )
                        else:
                            n_components[index_i] = np.min(
                                np.linalg.matrix_rank(
                                    epochs[:, chs, :], tol=nonzero_tol
                                )
                            )
                    else:
                        raise TypeError(
                            'n_seed_components and n_target_components must be '
                            'lists of `None`, an `int`, or the string "rank"'
                        )
                    index_i += 1
    
    return perform_svd, n_seed_components, n_target_components

def _handle_gc_with_svd_connectivity(
    data, indices, n_seed_components, n_target_components, perform_svd,
    con_method_types, present_gc_methods, call_params
):
    """Computes Granger causality connectivity if SVD is being performed, in
    which case the SVD, CSD computation, and connectivity computation must be
    performed for each connection separately."""
    faverage = call_params['faverage']
    n_bands = call_params['n_bands']
    non_gc_non_svd_method_types = deepcopy(con_method_types)

    # finds the GC methods to compute
    gc_svd_method_types = [
        mtype for mtype in con_method_types if mtype.name in
        ["GC", "Net GC", "TRGC", "Net TRGC"]
    ]

    # performs SVD on the timeseries data for each connection
    seed_target_data, n_seeds = _seeds_targets_svd(
        data, indices, n_seed_components, n_target_components
    )

    # computes GC for each connection separately
    n_gc_methods = len(present_gc_methods)
    con = [[] for _ in range(n_gc_methods)]
    topo = [None for _ in range(n_gc_methods)]
    for con_data, n_seed_comps in zip(seed_target_data, n_seeds):
        new_indices = (
            [np.arange(n_seed_comps).tolist()],
            [np.arange(n_seed_comps, con_data.shape[1]).tolist()]
        )

        (
            con_methods, times, freqs_bands, freq_idx_bands, n_tapers, n_epochs,
            n_cons, n_freqs, n_signals, freqs, remapped_indices
        ) = _compute_csd(
            data=con_data, indices=new_indices,
            con_method_types=gc_svd_method_types, **call_params
        )

        this_con, _, freqs_used, n_nodes = _compute_connectivity(
            con_methods, new_indices, n_seed_components, n_target_components,
            n_epochs, n_cons, faverage, n_freqs, n_bands, freq_idx_bands,
            freqs_bands, n_signals, freqs
        )
        [con[i].append(this_con[i]) for i in range(n_gc_methods)]

    con = [np.squeeze(np.array(this_con), 1) for this_con in con]

    # finds the methods still needing to be computed
    non_gc_non_svd_method_types = [
        mtype for mtype in con_method_types if mtype not in gc_svd_method_types
    ]
    
    return (
        gc_svd_method_types, non_gc_non_svd_method_types, con, topo, times,
        n_tapers, n_epochs, freqs, freqs_used, n_nodes, remapped_indices
    )

def _seeds_targets_svd(data, indices, n_seed_components, n_target_components):
    """Performs a single value decomposition on the epoched data separately for
    the seeds and targets of each set of seed-target pairs according to the
    specified number of seed and target components. If the number of components
    for a given instance is `None`, the original data is returned."""
    if isinstance(data, BaseEpochs):
        epochs = data.get_data(picks=data.ch_names).copy()
    else:
        epochs = data.copy()

    seed_target_data = []
    n_seeds = []
    for seeds, targets, n_seed_comps, n_target_comps in \
        zip(indices[0], indices[1], n_seed_components, n_target_components):

        if n_seed_comps is not None: # SVD seed data
            seed_data = _epochs_svd(epochs[:, seeds, :], n_seed_comps)
        else: # use unaltered seed data
            seed_data = epochs[:, seeds, :]
        n_seeds.append(seed_data.shape[1])

        if n_target_comps is not None: # SVD target data
            target_data = _epochs_svd(epochs[:, targets, :], n_target_comps)
        else: # use unaltered target data
            target_data = epochs[:, targets, :]

        seed_target_data.append(np.append(seed_data, target_data, axis=1))

    return seed_target_data, n_seeds

def _epochs_svd(epochs, n_comps):
    """Performs an SVD on epoched data and selects the first k components for
    dimensionality reduction before reconstructing the data with
    (U_k @ S_k @ V_k)."""
    # mean-centre the data epoch-wise
    centred_epochs = np.array([epoch - epoch.mean() for epoch in epochs])

    # compute the SVD (transposition so that the channels are the columns of
    # each epoch)
    U, S, V = np.linalg.svd(
        centred_epochs.transpose(0, 2, 1), full_matrices=False
    )

    # take the first k components
    U_k = U[:, :, :n_comps]
    S_k = np.eye(n_comps) * S[:, np.newaxis][:, :n_comps, :n_comps]
    V_k = V[:, :n_comps, :n_comps]

    # reconstruct the dimensionality-reduced data (have to tranpose the data
    # back into [epochs x channels x timepoints])
    return (U_k @ (S_k @ V_k)).transpose(0, 2, 1)

def _handle_non_gc_non_svd_connectivity(
    data, indices, n_seed_components, n_target_components, con_method_types,
    call_params
):
    """Computes connectivity where no Granger causality with SVD is being
    performed, meaning a single CSD can be computed and the connectivity
    computations performed for all connections together."""
    faverage = call_params['faverage']
    n_bands = call_params['n_bands']

    (
        con_methods, times, freqs_bands, freq_idx_bands, n_tapers,
        n_epochs, n_cons, n_freqs, n_signals, freqs, remapped_indices
    ) = _compute_csd(
        data=data, indices=indices, con_method_types=con_method_types,
        **call_params
    )

    con, topo, freqs_used, n_nodes = _compute_connectivity(
        con_methods, remapped_indices, n_seed_components,
        n_target_components, n_epochs, n_cons, faverage, n_freqs, n_bands,
        freq_idx_bands, freqs_bands, n_signals, freqs
    )

    return (
        con, topo, times, n_tapers, n_epochs, freqs, freqs_used, n_nodes,
        remapped_indices
    )

def _compute_csd(
    data, indices, sfreq, mode, tmin, tmax, fmin, fmax, fskip, faverage,
    cwt_freqs, mt_bandwidth, mt_adaptive, mt_low_bias, cwt_n_cycles, block_size,
    n_jobs, n_bands, con_method_types, parallel, epoch_spectral_connectivity,
    times_in, gc_n_lags
):
    """Computes the cross-spectral density of the data in preparation for the
    multivariate connectivity computations."""
    logger.info('Connectivity computation...')
    warn_times = True

    (
        epoch_blocks, n_cons, times, n_times_in, n_freqs, freqs, freqs_bands,
        freq_idx_bands, n_signals, n_tapers, remapped_indices, con_methods,
        call_params
    ) = _prepare_csd_computation(
        data=data, indices=indices, sfreq=sfreq, mode=mode, tmin=tmin,
        tmax=tmax, fmin=fmin, fmax=fmax, fskip=fskip, faverage=faverage,
        cwt_freqs=cwt_freqs, mt_bandwidth=mt_bandwidth, mt_adaptive=mt_adaptive,
        mt_low_bias=mt_low_bias, cwt_n_cycles=cwt_n_cycles,
        block_size=block_size, n_jobs=n_jobs, n_bands=n_bands,
        con_method_types=con_method_types, times_in=times_in,
        gc_n_lags=gc_n_lags
    )

    # performs the CSD computation for each epoch block
    logger.info('Computing cross-spectral density from epochs')
    n_epochs = 0
    for epoch_block in ProgressBar(epoch_blocks, mesg='CSD epoch blocks'):
        # check dimensions and time scale
        for this_epoch in epoch_block:
            _, _, _, warn_times = _get_and_verify_data_sizes(
                this_epoch, sfreq, n_signals, n_times_in, times_in,
                warn_times=warn_times)
            n_epochs += 1

        # compute CSD of epochs
        epochs = parallel(
            epoch_spectral_connectivity(data=this_epoch, **call_params)
            for this_epoch in epoch_block
        )

        # unpack and accumulate CSDs of epochs in connectivity methods
        for epoch in epochs:
            for method, epoch_csd in zip(con_methods, epoch[0]):
                method.combine(epoch_csd)

    return (
        con_methods, times, freqs_bands, freq_idx_bands, n_tapers, n_epochs,
        n_cons, n_freqs, n_signals, freqs, remapped_indices
    )

def _prepare_csd_computation(
    data, indices, sfreq, mode, tmin, tmax, fmin, fmax, fskip, faverage,
    cwt_freqs, mt_bandwidth, mt_adaptive, mt_low_bias, cwt_n_cycles, block_size,
    n_jobs, n_bands, con_method_types, times_in, gc_n_lags
):
    """Collects and returns information in preparation for computing the
    cross-spectral density."""
    epoch_blocks = [epoch for epoch in _get_n_epochs(data, n_jobs)]

    # initialize everything times and frequencies
    (
        n_cons, times, n_times, times_in, n_times_in, tmin_idx, tmax_idx,
        n_freqs, freq_mask, freqs, freqs_bands, freq_idx_bands, n_signals, _,
        warn_times
    ) = _prepare_connectivity(
        epoch_block=epoch_blocks[0], times_in=times_in, tmin=tmin, tmax=tmax,
        fmin=fmin, fmax=fmax, sfreq=sfreq, indices=indices, mode=mode,
        fskip=fskip, n_bands=n_bands, cwt_freqs=cwt_freqs, faverage=faverage
    )

    # get the window function, wavelets, etc for different modes
    spectral_params, mt_adaptive, n_times_spectrum, n_tapers = (
        _assemble_spectral_params(
            mode=mode, n_times=n_times, mt_adaptive=mt_adaptive,
            mt_bandwidth=mt_bandwidth, sfreq=sfreq, mt_low_bias=mt_low_bias,
            cwt_n_cycles=cwt_n_cycles, cwt_freqs=cwt_freqs, freqs=freqs,
            freq_mask=freq_mask
        )
    )

    # get the appropriate indices for computing connectivity on
    remapped_indices, sig_idx, use_n_signals, idx_map = _sort_con_indices(
        indices=indices
    )

    # create connectivity estimators
    con_methods = _instantiate_con_estimators(
        con_method_types=con_method_types, use_n_signals=use_n_signals,
        n_cons=n_cons, n_freqs=n_freqs, n_times_spectrum=n_times_spectrum,
        gc_n_lags=gc_n_lags, n_jobs=n_jobs
    )

    # collate settings
    call_params = dict(
        sig_idx=sig_idx, tmin_idx=tmin_idx, tmax_idx=tmax_idx, sfreq=sfreq,
        mode=mode, freq_mask=freq_mask, idx_map=idx_map, block_size=block_size,
        psd=None, accumulate_psd=False, mt_adaptive=mt_adaptive,
        con_method_types=con_method_types, con_methods=None,
        n_signals=n_signals, use_n_signals=use_n_signals, n_times=n_times,
        gc_n_lags=gc_n_lags, accumulate_inplace=False
    )
    call_params.update(**spectral_params)

    return (
        epoch_blocks, n_cons, times, n_times_in, n_freqs, freqs, freqs_bands,
        freq_idx_bands, n_signals, n_tapers, remapped_indices, con_methods,
        call_params
    )

def _sort_con_indices(indices):
    """Maps indices to the unique indices, finds the signals for which the
    cross-spectra needs to be computed (and how many used signals there are),
    and gets the seed-target indixes for the cross-spectra."""
    # map indices to unique indices
    unique_indices = np.unique(np.concatenate(sum(indices, [])))
    remapping = {ch_i: sig_i for sig_i, ch_i in enumerate(unique_indices)}
    remapped_indices = tuple([
        [[remapping[idx] for idx in idcs] for idcs in indices_group]
        for indices_group in indices
    ])

    # unique signals for which we actually need to compute CSD
    sig_idx = np.unique(sum(sum(remapped_indices, []), []))
    use_n_signals = len(sig_idx)

    # gets seed-target indices for CSD
    idx_map = [np.repeat(sig_idx, len(sig_idx)), np.tile(sig_idx, len(sig_idx))]

    return remapped_indices, sig_idx, use_n_signals, idx_map

def _instantiate_con_estimators(
    con_method_types, use_n_signals, n_cons, n_freqs, n_times_spectrum,
    gc_n_lags, n_jobs
):
    """Create instances of the connectivity estimators and log the methods being
    computed."""
    con_methods = []
    for mtype in con_method_types:
        # if a GC method, provide n_lags argument
        if "n_lags" in list(inspect.signature(mtype).parameters):
            con_methods.append(
                mtype(
                    use_n_signals, n_cons, n_freqs, n_times_spectrum, gc_n_lags,
                    n_jobs
                )
            )
        else:
            con_methods.append(
                mtype(use_n_signals, n_cons, n_freqs, n_times_spectrum, n_jobs)
            )

    metrics_str = ', '.join([meth.name for meth in con_methods])
    logger.info('    the following metrics will be computed: %s' % metrics_str)
    
    return con_methods

def _compute_connectivity(
    con_methods, indices, n_seed_components, n_target_components, n_epochs,
    n_cons, faverage, n_freqs, n_bands, freq_idx_bands, freqs_bands, n_signals,
    freqs
):
    """Computes the multivariate connectivity results."""
    con = list()
    topo = list()
    for conn_method in con_methods:
        if conn_method.name in ["GC", "Net GC", "TRGC", "Net TRGC"]:
            conn_method.compute_con(indices[0], indices[1], n_epochs)
        else:
            conn_method.compute_con(
                indices[0], indices[1], n_seed_components, n_target_components,
                n_epochs
            )

        # get the connectivity and topography scores
        this_con = conn_method.con_scores
        this_topo = conn_method.topographies

        _check_correct_results_dimensions(
            this_con, this_topo, n_cons, n_freqs, conn_method.n_times
        )

        if faverage:
            this_con, this_topo = _compute_f_average(
                this_con, this_topo, n_cons, n_bands, freq_idx_bands
            )

        con.append(this_con)
        topo.append(this_topo)

    freqs_used = freqs
    if faverage:
        # for each band we return the max and min frequencies that were averaged
        freqs_used = [[np.min(band), np.max(band)] for band in freqs_bands]

    # number of nodes in the original data,
    n_nodes = n_signals

    return con, topo, freqs_used, n_nodes

def _check_correct_results_dimensions(con, topo, n_cons, n_freqs, n_times):
    """Checks that the results of the connectivity computations have the
    appropriate dimensions."""
    assert (con.shape[0] == n_cons), \
        ('The first dimension of connectivity scores does not match the '
        'number of connections. Please contact the mne-connectivity ' 
        'developers.')
    assert (con.shape[1] == n_freqs), \
        ('The second dimension of connectivity scores does not match the '
        'number of frequencies. Please contact the mne-connectivity ' 
        'developers.')
    if n_times != 0:
        assert (con.shape[2] == n_times), \
            ('The third dimension of connectivity scores does not match the '
            'number of timepoints. Please contact the mne-connectivity ' 
            'developers.')
    
    if topo is not None:
        assert (topo[0].shape[0] == n_cons and topo[1].shape[0]), \
            ('The first dimension of topographies does not match the number of '
            'connections. Please contact the mne-connectivity developers.')
        for con_i in range(n_cons):
            assert (topo[0][con_i].shape[1] == n_freqs and
                    topo[1][con_i].shape[1] == n_freqs), \
                ('The second dimension of topographies does not match the '
                'number of frequencies. Please contact the mne-connectivity '
                'developers.')
            if n_times != 0:
                assert (topo[0][con_i].shape[2] == n_times and
                        topo[1][con_i].shape[2] == n_times), \
                    ('The third dimension of topographies does not match the '
                    'number of timepoints. Please contact the mne-connectivity '
                    'developers.')

def _compute_f_average(con, topo, n_cons, n_bands, freq_idx_bands):
    """Computes the average connectivity across the frequency bands."""
    con_shape = (n_cons, n_bands) + con.shape[2:]
    con_bands = np.empty(con_shape, dtype=con.dtype)
    for band_idx in range(n_bands):
        con_bands[:, band_idx] = np.mean(
            con[:, freq_idx_bands[band_idx]], axis=1
        )

    if topo is not None:
        topo_bands = np.empty((2, n_cons), dtype=topo.dtype)
        for group_i in range(2):
            for con_i in range(n_cons):
                band_topo = [
                    np.mean(topo[group_i][con_i][freq_idx_band], axis=1)
                    for freq_idx_band in freq_idx_bands
                ]
                topo_bands[group_i][con_i] = np.array(band_topo).T
    
    return con_bands, topo_bands

def _collate_connectivity_results(
    con, gc_con, topo, gc_topo,  non_gc_non_svd_method_types,
    gc_svd_method_types, con_method_types, indices, remapped_indices
):
    """"""
    # Collate connectivity results
    if con and gc_con:
        # combines SVD GC and non-GC, non-SVD results
        con.extend(gc_con)
        topo.extend(gc_topo)
        # orders the results according to the order they were called
        methods_order = [
            *[mtype.name for mtype in gc_svd_method_types],
            *[mtype.name for mtype in non_gc_non_svd_method_types]
        ]
        con = [
            con[methods_order.index(mtype.name)] for mtype in con_method_types
        ]
        topo = [
            topo[methods_order.index(mtype.name)] for mtype in con_method_types
        ]

    elif not con and gc_con:
        # leaves SVD GC results as the only results
        con = gc_con
        topo = gc_topo
        # finds the remapped indices of non-SVD data
        unique_indices = np.unique(np.concatenate(sum(indices, [])))
        remapping = {ch_i: sig_i for sig_i, ch_i in enumerate(unique_indices)}
        remapped_indices = [[[remapping[idx] for idx in idcs] for idcs in
                             indices_group] for indices_group in indices]

    # else you only have the non-GC, non-SVD results anyway, so nothing more
    # needs to be done

    return con, topo, remapped_indices

def _store_connectivity(
    con, topo, method, names, freqs, n_nodes, mode, indices, n_epochs,
    freqs_used, times, n_tapers, metadata, events, event_id
):
    """Stores multivariate connectivity results in an mne-connectivity
    object."""
    # create a list of connectivity containers
    conn_list = []
    for _con, _topo, _method in zip(con, topo, method):
        kwargs = dict(
            data=_con, topographies=_topo, names=names, freqs=freqs,
            method=_method, n_nodes=n_nodes, spec_method=mode, indices=indices,
            n_epochs_used=n_epochs, freqs_used=freqs_used, times_used=times,
            n_tapers=n_tapers, metadata=metadata, events=events,
            event_id=event_id
        )
        # create the connectivity container
        if mode in ['multitaper', 'fourier']:
            conn_class = MultivariateSpectralConnectivity
        else:
            assert mode == 'cwt_morlet'
            conn_class = MultivariateSpectroTemporalConnectivity
            kwargs.update(times=times)
        conn_list.append(conn_class(**kwargs))

    logger.info('[Connectivity computation done]')

    if len(method) == 1:
        # for a single method return connectivity directly
        conn_list = conn_list[0]

    return conn_list