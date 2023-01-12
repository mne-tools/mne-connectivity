# Authors: Thomas Samuel Binns <t.s.binns@outlook.com>
#          Tien Dung Nguyen <>
#          Richard M. Köhler <koehler.richard@charite.de>
#
# License: BSD (3-clause)

import inspect
import copy
import numpy as np
from mne import BaseEpochs
from mne.parallel import parallel_func
from mne.utils import _arange_div, logger, ProgressBar, _time_mask
from ..base import (
    MultivariateSpectralConnectivity, MultivariateSpectroTemporalConnectivity
)
from .epochs import (
    _assemble_spectral_params, _check_estimators, _compute_freqs,
    _compute_freq_mask, _epoch_spectral_connectivity,
    _get_and_verify_data_sizes, _get_n_epochs, _prepare_connectivity
)


class _MVCSpectralEpochs():
    """Computes multivariate spectral connectivity of epoched data for the
    multivariate_spectral_connectivity_epochs function."""

    init_attrs = [
        'data', 'indices', 'names', 'method', 'sfreq', 'mode', 'tmin', 'tmax',
        'fmin', 'fmax', 'fskip', 'faverage', 'cwt_freqs', 'mt_bandwidth',
        'mt_adaptive', 'mt_low_bias', 'cwt_n_cycles', 'n_seed_components',
        'n_target_components', 'gc_n_lags', 'block_size', 'n_jobs', 'verbose'
    ]

    gc_method_aliases = ['gc', 'net_gc', 'trgc', 'net_trgc']
    gc_method_names = ['GC', 'Net GC', 'TRGC', 'Net TRGC']
    
    # possible forms of GC with information on how to use the methods
    possible_gc_forms = {
        'seeds -> targets': dict(
            flip_seeds_targets=False, reverse_time=False,
            for_methods=['GC', 'Net GC', 'TRGC', 'Net TRGC'], method_class=None
        ),
        'targets -> seeds': dict(
            flip_seeds_targets=True, reverse_time=False,
            for_methods=['Net GC', 'Net TRGC'], method_class=None
        ),
        'time-reversed[seeds -> targets]': dict(
            flip_seeds_targets=False, reverse_time=True,
            for_methods=['TRGC', 'Net TRGC'], method_class=None
        ),
        'time-reversed[targets -> seeds]': dict(
            flip_seeds_targets=True, reverse_time=True,
            for_methods=['Net TRGC'], method_class=None
        )
    }

    # possible forms of coherence with information on how to use the methods
    possible_coh_forms = {
        'MIC & MIM': dict(
            for_methods=['MIC', 'MIM'], exclude_methods=[], method_class=None
        ),
        'MIC': dict(
            for_methods=['MIC'], exclude_methods=['MIM'], method_class=None
        ),
        'MIM': dict(
            for_methods=['MIM'], exclude_methods=['MIC'], method_class=None
        )
    }

    # threshold for classifying singular values as being non-zero
    rank_nonzero_tol = 1e-10
    perform_svd = False

    # whether the requested frequencies are discontinuous (e.g. different bands)
    discontinuous_freqs = False

    # whether or not GC must be computed separately from other methods
    compute_gc_separately = False

    # storage for GC results if SVD used or freqs are discontinuous (which must
    # be computed separately from other methods)
    separate_gc_method_types = []
    separate_gc_con = []
    separate_gc_topo = []

    # storage for coherence results (and GC results if no SVD used and requested
    # frequencies are continuous)
    remaining_con = []
    remaining_topo = []

    def __init__(self, **kwargs):
        assert all(attr in self.init_attrs for attr in kwargs.keys()), (
            'Not all inputs to the _MVCSpectralEpochs class have been '
            'provided. Please contact the mne-connectivity developers.'
        )
        for name, value in kwargs.items():
            assert name in self.init_attrs, (
                'An input to the _MVCSpectralEpochs class is not recognised. '
                'Please contact the mne-connectivity developers.'
            )
            setattr(self, name, value)
        
        self._sort_inputs()
    
    def _sort_inputs(self):
        """Checks the format of the input parameters and enacts them to create
        new object attributes."""
        self._sort_parallelisation_inputs()
        self._sort_data_info()
        self._sort_estimator_inputs()
        self._sort_freq_inputs()
        self._sort_indices_inputs()
        self._sort_svd_inputs()

        if self.perform_svd or self.discontinuous_freqs:
            self.compute_gc_separately = True
    
    def _sort_parallelisation_inputs(self):
        """Establishes parallelisation of the function for computing the CSD if 
        n_jobs > 1, else uses the standard, non-parallelised function."""
        self.parallel, self._epoch_spectral_connectivity, _ = (
            parallel_func(
                _epoch_spectral_connectivity, self.n_jobs, verbose=self.verbose
            )
        )
    
    def _sort_data_info(self):
        """Extracts information stored in the data if it is an Epochs object,
        otherwise sets this information to `None`."""
        if isinstance(self.data, BaseEpochs):
            self.names = self.data.ch_names
            self.times_in = self.data.times  # input times for Epochs input type
            self.sfreq = self.data.info['sfreq']

            self.events = self.data.events
            self.event_id = self.data.event_id

            # Extract metadata from the Epochs data structure.
            # Make Annotations persist through by adding them to the metadata.
            metadata = self.data.metadata
            if metadata is None:
                self.annots_in_metadata = False
            else:
                self.annots_in_metadata = all(
                    name not in metadata.columns for name in 
                    ['annot_onset', 'annot_duration', 'annot_description']
                )
            if (
                hasattr(self.data, 'annotations') and not
                self.annots_in_metadata
            ):
                self.data.add_annotations_to_metadata(overwrite=True)
            self.metadata = self.data.metadata
        else:
            self.times_in = None
            self.events = None
            self.event_id = None
            self.metadata = None

    def _sort_estimator_inputs(self):
        """Assign names to connectivity methods, check the methods and mode are
        recognised, and finds which Granger causality methods are being
        called."""
        if not isinstance(self.method, (list, tuple)):
            self.method = [self.method]  # make it a list so we can iterate

        self.con_method_types, _, _, _ = _check_estimators(
            self.method, self.mode
        )
        metrics_str = ', '.join([meth.name for meth in self.con_method_types])
        logger.info(
            '    the following metrics will be computed: %s' % metrics_str
        )

        # find which Granger causality methods are being called
        self.present_gc_methods = [
            con_method for con_method in self.method
            if con_method in self.gc_method_aliases
        ]

        self.remaining_method_types = copy.deepcopy(self.con_method_types)

    def _sort_freq_inputs(self):
        """Formats frequency-related inputs and checks they are appropriate."""
        if self.fmin is None:
            self.fmin = -np.inf  # set it to -inf, so we can adjust it later

        self.fmin = np.array((self.fmin,), dtype=float).ravel()
        self.fmax = np.array((self.fmax,), dtype=float).ravel()
        if len(self.fmin) != len(self.fmax):
            raise ValueError('fmin and fmax must have the same length')
        if np.any(self.fmin > self.fmax):
            raise ValueError('fmax must be larger than fmin')

        self.n_bands = len(self.fmin)

        if self.present_gc_methods:
            self._check_for_discontinuous_freqs()

    def _check_for_discontinuous_freqs(self):
        """Checks whether the requested frequencies to analyse are
        discontinuous (occurs in the case that different frequency bands are
        specified in fmin and fmax, but there is a gap between the boundaries of
        each frequency band). The state-space GC method used has a
        cross-frequency relationship which would be disrupted by computing
        connectivity on a discontinuous set of frequencies, so this checks to
        see if GC needs to be computed separately on a continuous set of
        frequencies spanning from the lowest fmin and highest fmax values which
        can then be split into the specified frequency bands.
        
        A simpler check would be to set discontinuous = True if n_bands > 1,
        however it could be the case that the bands are continuous, e.g. 8-12 Hz
        and 13-20 Hz (with a freq. resolution of 1 Hz), in which case the
        frequencies are not discontinuous and GC can be computed alongside other
        methods.
        """
        n_times = self._get_n_used_times()
        # compute frequencies to analyze based on number of samples, sampling
        # rate, specified wavelet frequencies and mode
        freqs = _compute_freqs(n_times, self.sfreq, self.cwt_freqs, self.mode)
        # compute the mask based on specified min/max and decimation factor
        freq_mask = _compute_freq_mask(freqs, self.fmin, self.fmax, 0)

        # formula for finding if indices of freqs being analysed is continuous;
        # array should not contain repeats (but that should always be the
        # case for these frequency indices) and should start from 1 (we make
        # this adjustment)
        use_freqs = np.nonzero(freq_mask)[0]
        use_freqs = (use_freqs - min(use_freqs)) + 1 # need to start from 1
        if (
            sum(np.arange(1, len(use_freqs) + 1)) !=
            use_freqs[-1] * (use_freqs[-1] + 1) / 2
        ):
            assert self.n_bands != 1, (
                'Frequencies have been detected as discontinuous, yet there is '
                'only a single frequency band in the data. Please contact the '
                'mne-connectivity developers.'
            )
            self.discontinuous_freqs = True

    def _get_n_used_times(self):
        """Finds and returns the number of timepoints being examined in the
        data."""
        if self.times_in is None:
            if isinstance(self.data, BaseEpochs):
                n_times = self.data.get_data().shape[2]
            else:
                n_times = self.data.shape[2]
            times = _arange_div(n_times, self.sfreq)
        else:
            times = self.times_in

        time_mask = _time_mask(times, self.tmin, self.tmax, sfreq=self.sfreq)
        tmin_idx, tmax_idx = np.where(time_mask)[0][[0, -1]]

        return len(times[tmin_idx : tmax_idx + 1])

    def _sort_indices_inputs(self):
        """Checks that the indices are appropriate and sets the number of seeds
        and targets in each connection."""
        if self.indices is None:
            raise ValueError('indices must be specified, got `None`')

        if len(self.indices[0]) != len(self.indices[1]):
            raise ValueError(
                f'the number of seeds ({len(self.indices[0])}) and targets '
                f'({len(self.indices[1])}) must match'
            )
        self.n_cons = len(self.indices[0])

        for seeds, targets in zip(self.indices[0], self.indices[1]):
            if not isinstance(seeds, list) or not isinstance(targets, list):
                raise TypeError(
                    'seeds and targets for each connection must be given as a '
                    'list of ints'
                )
            if (
                not all(isinstance(seed, int) for seed in seeds) or
                not all(isinstance(target, int) for target in targets)
            ):
                raise TypeError(
                    'seeds and targets for each connection must be given as a '
                    'list of ints'
                )
            if set.intersection(set(seeds), set(targets)):
                raise ValueError(
                    'there are common indices present in the seeds and targets '
                    'for a single connection, however multivariate '
                    'connectivity between shared channels is not supported'
                )

    def _sort_svd_inputs(self):
        """Checks that the SVD parameters are appropriate and finds the correct
        dimensionality reduction settings to use, if applicable.
        
        This involves the rank of the data being computed based its non-zero
        singular values. We use a cut-off of 1e-10 by default to determine when
        a value is non-zero, as using numpy's default cut-off is too liberal
        (i.e. low) for our purposes where we need to be stricter.
        """
        self.n_seed_components = copy.copy(self.n_seed_components)
        self.n_target_components = copy.copy(self.n_target_components)

        # finds if any SVD has been requested for seeds and/or targets
        if self.n_seed_components is None:
            self.n_seed_components = [None for _ in range(self.n_cons)]
        if self.n_target_components is None:
            self.n_target_components = [None for _ in range(self.n_cons)]

        for n_components in (self.n_seed_components, self.n_target_components):
            if not isinstance(n_components, list):
                raise TypeError(
                    'n_seed_components and n_target_components must be lists'
                )
            if self.n_cons != len(n_components):
                raise ValueError(
                    'n_seed_components and n_target_components must have the '
                    'same length as specified the number of connections in '
                    f'indices. Got: {len(n_components)} components and '
                    f'{self.n_cons} connections'
                )
            if not self.perform_svd and any(
                n_comps is not None for n_comps in n_components
            ):
                self.perform_svd = True

        # if SVD is requested, extract the data and perform subsequent checks
        if self.perform_svd:
            if isinstance(self.data, BaseEpochs):
                epochs = self.data.get_data(picks=self.data.ch_names)
            else:
                epochs = self.data
        
            for group_i, n_components in enumerate(
                (self.n_seed_components, self.n_target_components)
            ):
                if any(n_comps is not None for n_comps in n_components):
                    index_i = 0
                    for n_comps, chs in zip(
                        n_components, self.indices[group_i]
                    ):
                        if isinstance(n_comps, int):
                            if n_comps > len(chs):
                                raise ValueError(
                                    'The number of components to take cannot '
                                    'be greater than the number of channels in '
                                    'a given seed/target'
                                )
                            if n_comps <= 0:
                                raise ValueError(
                                    'The number of components to take must be '
                                    'greater than 0'
                                )
                        elif isinstance(n_comps, str):
                            if n_comps != 'rank':
                                raise ValueError(
                                    'if n_seed_components or '
                                    'n_target_components contains a string, it '
                                    'must be the string "rank"'
                                )
                            # compute the rank of the seeds/targets for a con
                            n_components[index_i] = np.min(
                                np.linalg.matrix_rank(
                                    epochs[:, chs, :], tol=self.rank_nonzero_tol
                                )
                            )
                        elif n_comps is not None:
                            raise TypeError(
                                'n_seed_components and n_target_components '
                                'must be lists of `None`, `int`, or the string '
                                '"rank"'
                            )
                        index_i += 1

    def compute_csd_and_connectivity(self):
        """Compute the CSD of the data and derive connectivity results from
        it."""
        # if SVD is requested or the specified fbands are discontinuous, the
        # CSD (and hence connectivity) has to be computed separately for any GC
        # methods
        if self.present_gc_methods and self.compute_gc_separately:
            self._compute_separate_gc_csd_and_connectivity()

        # if GC has been computed separately, the coherence methods are computed
        # here, otherwise both GC and coherence methods are computed here
        if self.remaining_method_types:
            self._compute_remaining_csd_and_connectivity()

        # combine all connectivity results (if GC was computed seperately)
        self._collate_connectivity_results()
    
    def _compute_separate_gc_csd_and_connectivity(self):
        """Computes the CSD and connectivity for GC methods separately from
        other methods if SVD is being performed, or the requested fbands are
        discontinuous.
        
        If SVD is being performed with GC, this has to be done on the
        timeseries data for each connection separately, and so this transformed
        data cannot be used to compute the CSD for coherence-based connectivity
        methods.

        Unlike the coherence methods, the state-space GC methods used here rely
        on cross-frequency relationships, so discontinuous frequencies will mess
        up the results. Hence, GC must be computed on a continuous set of
        frequencies, and then have the requested frequency band results taken.
        """
        # finds the GC methods to compute
        self.separate_gc_method_types = [
            mtype for mtype in self.con_method_types if mtype.name in
            self.gc_method_names
        ]

        seed_target_data, n_seeds = self._seeds_targets_svd()

        # computes GC for each connection separately (no topographies for GC)
        n_gc_methods = len(self.present_gc_methods)
        self.separate_gc_con = [[] for _ in range(n_gc_methods)]
        self.separate_gc_topo = [None for _ in range(n_gc_methods)]

        for con_data, n_seed_comps in zip(seed_target_data, n_seeds):
            new_indices = (
                [np.arange(n_seed_comps).tolist()],
                [np.arange(n_seed_comps, con_data.shape[1]).tolist()]
            )

            con_methods = self._compute_csd(
                con_data, self.separate_gc_method_types, new_indices
            )

            this_con, _, = self._compute_connectivity(con_methods, new_indices)

            for method_i in range(n_gc_methods):
                self.separate_gc_con[method_i].append(this_con[method_i])

        self.separate_gc_con = [
            np.squeeze(np.array(this_con), 1) for this_con in
            self.separate_gc_con
        ]

        # finds the methods still needing to be computed
        self.remaining_method_types = [
            mtype for mtype in self.con_method_types if
            mtype not in self.separate_gc_method_types
        ]

    def _seeds_targets_svd(self):
        """SVDs the epoched data separately for the seeds and targets of each
        connection according to the specified number of seed and target
        components. If the number of components for a given instance is `None`,
        the original data is returned."""
        if isinstance(self.data, BaseEpochs):
            epochs = self.data.get_data(picks=self.data.ch_names).copy()
        else:
            epochs = self.data.copy()

        seed_target_data = []
        n_seeds = []
        for seeds, targets, n_seed_comps, n_target_comps in zip(
            self.indices[0], self.indices[1], self.n_seed_components,
            self.n_target_components
        ):
            if n_seed_comps is not None: # SVD seed data
                seed_data = self._epochs_svd(epochs[:, seeds, :], n_seed_comps)
            else: # use unaltered seed data
                seed_data = epochs[:, seeds, :]
            n_seeds.append(seed_data.shape[1])

            if n_target_comps is not None: # SVD target data
                target_data = self._epochs_svd(epochs[:, targets, :], n_target_comps)
            else: # use unaltered target data
                target_data = epochs[:, targets, :]

            seed_target_data.append(np.append(seed_data, target_data, axis=1))

        return seed_target_data, n_seeds

    def _epochs_svd(self, epochs, n_comps):
        """Performs an SVD on epoched data and selects the first k components
        for dimensionality reduction before reconstructing the data with
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

        # reconstruct the dimensionality-reduced data (have to transpose the
        # data back into [epochs x channels x timepoints])
        return (U_k @ (S_k @ V_k)).transpose(0, 2, 1)

    def _compute_remaining_csd_and_connectivity(self):
        """Computes connectivity where a single CSD can be computed and the
        connectivity computations performed for all connections together (i.e.
        anything other than GC with SVD and/or GC with discontinuous
        frequencies)."""
        con_methods = self._compute_csd(
            self.data, self.remaining_method_types, self.indices
        )

        self.remaining_con, self.remaining_topo = (
            self._compute_connectivity(con_methods, self.remapped_indices)
        )

    def _compute_csd(self, data, con_method_types, indices):
        """Computes the cross-spectral density of the data in preparation for
        the multivariate connectivity computations."""
        logger.info('Connectivity computation...')

        con_methods = self._prepare_csd_computation(
            data, con_method_types, indices
        )

        # performs the CSD computation for each epoch block
        logger.info('Computing cross-spectral density from epochs')
        self.n_epochs = 0
        for epoch_block in ProgressBar(
            self.epoch_blocks, mesg='CSD epoch blocks'
        ):
            # check dimensions and time scale
            for this_epoch in epoch_block:
                _, _, _, self.warn_times = _get_and_verify_data_sizes(
                    this_epoch, self.sfreq, self.n_signals, self.n_times_in,
                    self.times_in, warn_times=self.warn_times
                )
                self.n_epochs += 1

            # compute CSD of epochs
            epochs = self.parallel(
                self._epoch_spectral_connectivity(
                    data=this_epoch, **self.csd_call_params
                )
                for this_epoch in epoch_block
            )

            # unpack and accumulate CSDs of epochs in connectivity methods
            for epoch in epochs:
                for method, epoch_csd in zip(con_methods, epoch[0]):
                    method.combine(epoch_csd)
        
        return con_methods

    def _prepare_csd_computation(self, data, con_method_types, indices):
        """Collects and returns information in preparation for computing the
        cross-spectral density."""
        self.epoch_blocks = [
            epoch for epoch in _get_n_epochs(data, self.n_jobs)
        ]

        fmin, fmax = self._get_fmin_fmax_for_csd(con_method_types)
        (
            _, self.times, n_times, self.times_in, self.n_times_in, tmin_idx,
            tmax_idx, self.n_freqs, freq_mask, self.freqs, freqs_bands,
            freq_idx_bands, self.n_signals, _, self.warn_times
        ) = _prepare_connectivity(
            epoch_block=self.epoch_blocks[0], times_in=self.times_in,
            tmin=self.tmin, tmax=self.tmax, fmin=fmin, fmax=fmax,
            sfreq=self.sfreq, indices=self.indices, mode=self.mode,
            fskip=self.fskip, n_bands=self.n_bands, cwt_freqs=self.cwt_freqs,
            faverage=self.faverage
        )
        self._store_freq_band_info(
            con_method_types, freqs_bands, freq_idx_bands
        )

        spectral_params, mt_adaptive, self.n_times_spectrum, self.n_tapers = (
            _assemble_spectral_params(
                mode=self.mode, n_times=n_times, mt_adaptive=self.mt_adaptive,
                mt_bandwidth=self.mt_bandwidth, sfreq=self.sfreq,
                mt_low_bias=self.mt_low_bias, cwt_n_cycles=self.cwt_n_cycles,
                cwt_freqs=self.cwt_freqs, freqs=self.freqs, freq_mask=freq_mask
            )
        )

        self._sort_con_indices(indices)

        con_methods = self._instantiate_con_estimators(con_method_types)

        self.csd_call_params = dict(
            sig_idx=self.sig_idx, tmin_idx=tmin_idx, tmax_idx=tmax_idx,
            sfreq=self.sfreq, mode=self.mode, freq_mask=freq_mask,
            idx_map=self.idx_map, block_size=self.block_size, psd=None,
            accumulate_psd=False, mt_adaptive=mt_adaptive,
            con_method_types=self.con_method_types, con_methods=None,
            n_signals=self.n_signals, use_n_signals=self.use_n_signals,
            n_times=n_times, gc_n_lags=self.gc_n_lags, accumulate_inplace=False
        )
        self.csd_call_params.update(**spectral_params)

        return con_methods

    def _get_fmin_fmax_for_csd(self, con_method_types):
        """Gets fmin and fmax args to use for the CSD computation."""
        if (
            self.present_gc_methods and self.discontinuous_freqs and
            con_method_types[0] in self.separate_gc_method_types
        ):
            # compute GC on a continuous set of freqs spanning all bands of
            # interest due to the cross-freq relationship of the GC methods
            return (
                np.array((np.min(self.fmin), )),
                np.array((np.max(self.fmax), ))
            )

        # use existing fmin and fmax if GC is not being computed, or if GC is
        # being computed and the requested freq bands are not discontinuous
        return (self.fmin, self.fmax)

    def _store_freq_band_info(
        self, con_method_types, freqs_bands, freq_idx_bands
    ):
        """Ensures the frequency band information returned from the connectivity
        preparation function is correct before storing them in the object."""
        if (
            self.present_gc_methods and self.discontinuous_freqs and
            con_method_types[0] in self.separate_gc_method_types
        ):
            # compute fbands and indices as the freqs appear in fmin and fmax;
            # required as the fmin and fmax args to the connectivity preparation
            # function differ to those provided by the end user
            self.freq_idx_bands = [
                np.where((self.freqs >= fl) & (self.freqs <= fu))[0] for
                fl, fu in zip(self.fmin, self.fmax)
            ]
            self.freqs_bands = [
                self.freqs[freq_idx] for freq_idx in self.freq_idx_bands
            ]
        else:
            # use the fband arguments returned from the connectivity preparation
            # function, matching the fband args provided by the end user
            self.freq_idx_bands = freq_idx_bands
            self.freqs_bands = freqs_bands

    def _sort_con_indices(self, indices):
        """Maps indices to the unique indices, finds the signals for which the
        CSD needs to be computed (and how many used signals there are), and gets
        the seed-target indices for the CSD."""
        # map indices to unique indices
        unique_indices = np.unique(np.concatenate(sum(indices, [])))
        remapping = {ch_i: sig_i for sig_i, ch_i in enumerate(unique_indices)}
        self.remapped_indices = tuple([
            [[remapping[idx] for idx in idcs] for idcs in indices_group]
            for indices_group in indices
        ])

        # unique signals for which we actually need to compute CSD
        self.sig_idx = self._get_unique_signals(self.remapped_indices)
        self.use_n_signals = len(self.sig_idx)

        # gets seed-target indices for CSD
        self.idx_map = [
            np.repeat(
                self.sig_idx, len(self.sig_idx)),
                np.tile(self.sig_idx, len(self.sig_idx)
            )
        ]

    def _get_unique_signals(self, indices):
        """Find the unique signals in a set of indices."""
        return np.unique(sum(sum(indices, []), []))

    def _instantiate_con_estimators(self, con_method_types):
        """Create instances of the connectivity estimators and log the methods
        being computed."""
        con_methods = []
        for mtype in con_method_types:
            if "n_lags" in list(inspect.signature(mtype).parameters):
                # if a GC method, provide n_lags argument
                con_methods.append(
                    mtype(
                        self.use_n_signals, self.n_cons, self.n_freqs,
                        self.n_times_spectrum, self.gc_n_lags, self.n_jobs
                    )
                )
            else:
                # if a coherence method, do not provide n_lags argument
                con_methods.append(
                    mtype(
                        self.use_n_signals, self.n_cons, self.n_freqs,
                        self.n_times_spectrum, self.n_jobs
                    )
                )
        
        return con_methods

    def _compute_connectivity(self, con_methods, indices):
        """Computes the multivariate connectivity results."""
        con = [None for _ in range(len(con_methods))]
        topo = [None for _ in range(len(con_methods))]

        # add the GC results to con in the correct positions according to the
        # order of con_methods
        con = self._compute_gc_connectivity(con_methods, con, indices)

        # add the coherence results to con in the correct positions according to
        # the order of con_methods
        con, topo = self._compute_coh_connectivity(
            con_methods, con, topo, indices
        )

        method_i = 0
        for method_con, method_topo in zip(con, topo):
            assert method_con is not None, (
                'A connectivity result has been missed. Please contact the '
                'mne-connectivity developers.'
            )

            self._check_correct_results_dimensions(
                con_methods, method_con, method_topo
            )

            if self.faverage:
                con[method_i], topo[method_i] = self._compute_faverage(
                        con=method_con, topo=method_topo
                    )
            
            method_i += 1

        self.freqs_used = self.freqs
        if self.faverage:
            # for each band we return the frequencies that were averaged
            self.freqs = [np.mean(band) for band in self.freqs_bands]
            # return max and min frequencies that were averaged for each band
            self.freqs_used = [
                [np.min(band), np.max(band)] for band in self.freqs_bands
            ]

        # number of nodes in the original data
        self.n_nodes = self.n_signals

        return con, topo

    def _compute_gc_connectivity(self, con_methods, con, indices):
        """Computes GC connectivity.
        
        Different GC methods can rely on common information, so rather than
        re-computing this information everytime a different GC method is called,
        store this information such that it can be accessed to compute the final
        GC connectivity scores when needed.
        """
        self._get_gc_forms_to_compute(con_methods)

        if self.compute_gc_forms:
            self._compute_and_set_gc_autocov()

            gc_scores = {}
            for form_name, form_info in self.compute_gc_forms.items():
                # computes connectivity for individual GC forms
                form_info['method_class'].compute_con(
                    indices[0], indices[1], form_info['flip_seeds_targets'], 
                    form_info['reverse_time'], form_name
                )

                # assigns connectivity score to their appropriate GC forms for
                # combining into the final GC method results
                gc_scores[form_name] = form_info['method_class'].con_scores
            
            con = self._combine_gc_forms(con_methods, con, gc_scores)
        
            # remove the results for frequencies not requested by the end user
            if self.discontinuous_freqs:
                con = self._make_gc_freqs_discontinuous(con)
            
            # set n_signals to equal the number in the non-SVD data
            if self.perform_svd:
                self.n_signals = len(self._get_unique_signals(self.indices))

        return con

    def _get_gc_forms_to_compute(self, con_methods):
        """Finds the GC forms that need to be computed."""
        self.compute_gc_forms = {}
        for form_name, form_info in self.possible_gc_forms.items():
            for method in con_methods:
                if (
                    method.name in form_info['for_methods'] and
                    form_name not in self.compute_gc_forms.keys()
                ):
                    form_info.update(method_class=copy.deepcopy(method))
                    self.compute_gc_forms[form_name] = form_info

    def _compute_and_set_gc_autocov(self):
        """Computes autocovariance once and assigns it to all GC methods."""
        first_form = True
        for form_info in self.compute_gc_forms.values():
            if first_form:
                form_info['method_class'].compute_autocov(self.n_epochs)
                autocov = form_info['method_class'].autocov.copy()
                first_form = False
            else:
                form_info['method_class'].autocov = autocov

    def _combine_gc_forms(self, con_methods, con, gc_scores):
        """Combines the information from all the different GC forms so that the
        final connectivity scores for the requested GC methods are returned."""
        for method_i, method in enumerate(con_methods):
            if method.name == 'GC':
                con[method_i] = gc_scores['seeds -> targets']
            elif method.name == 'Net GC':
                con[method_i] = (
                    gc_scores['seeds -> targets'] -
                    gc_scores['targets -> seeds']
                )
            elif method.name == 'TRGC':
                con[method_i] = (
                    gc_scores['seeds -> targets'] -
                    gc_scores['time-reversed[seeds -> targets]']
                )
            elif method.name == 'Net TRGC':
                con[method_i] = (
                    (
                        gc_scores['seeds -> targets'] -
                        gc_scores['targets -> seeds']
                    ) - (
                        gc_scores['time-reversed[seeds -> targets]'] -
                        gc_scores['time-reversed[targets -> seeds]']
                    )
                )

        return con

    def _make_gc_freqs_discontinuous(self, con):
        """Remove the unrequested frequencies from the GC results so that the
        results match the frequency bands requested by the end user."""
        # find which freqs in the results are needed
        requested_freqs = np.concatenate(self.freq_idx_bands)
        freq_mask = [freq in requested_freqs for freq in range(self.n_freqs)]
        
        # exclude the unwanted freqs from the results
        for method_i, method_con in enumerate(con):
            con[method_i] = method_con[:, freq_mask]
        
        # set the frequency attrs to the correct, discontinuous values
        self.n_freqs = len(requested_freqs)
        self.freqs = self.freqs[freq_mask]

        freq_idx_bands = []
        freq_idx = 0
        for band in self.freq_idx_bands:
            freq_idx_bands.append(
                np.arange(freq_idx, freq_idx + len(band), dtype=band.dtype)
            )
            freq_idx += len(band)
        self.freq_idx_bands = freq_idx_bands

        return con

    def _compute_coh_connectivity(self, con_methods, con, topo, indices):
        """Computes MIC and MIM connectivity.
        
        MIC and MIM rely on common information, so rather than re-computing this
        information everytime a different coherence method is called, store this
        information such that it can be accessed to compute the final MIC and
        MIM connectivity scores when needed.
        """
        self._get_coh_form_to_compute(con_methods)

        if self.compute_coh_form:
            # compute connectivity for MIC and/or MIM in a single instance
            form_name = list(self.compute_coh_form.keys())[0]  # only one there
            form_info = self.compute_coh_form[form_name]
            form_info['method_class'].compute_con(
                indices[0], indices[1], self.n_seed_components,
                self.n_target_components, self.n_epochs, form_name
            )
        
            # store the MIC and/or MIM results in the right places
            for method_i, method in enumerate(con_methods):
                if method.name == "MIC":
                    con[method_i] = form_info['method_class'].mic_scores
                    topo[method_i] = form_info['method_class'].topographies
                elif method.name == "MIM":
                    con[method_i] = form_info['method_class'].mim_scores

        return con, topo

    def _get_coh_form_to_compute(self, con_methods):
        """Finds the coherence form that need to be computed."""
        method_names = [method.name for method in con_methods]
        self.compute_coh_form = {}
        for form_name, form_info in self.possible_coh_forms.items():
            if (
                all(name in method_names for name in form_info['for_methods'])
                and not any(
                    name in method_names for name in
                    form_info['exclude_methods']
                )
            ):
                coh_class = con_methods[
                    method_names.index(form_info['for_methods'][0])
                ]
                form_info.update(method_class=coh_class)
                self.compute_coh_form[form_name] = form_info
                break # only one form is possible at any one instance

    def _check_correct_results_dimensions(self, con_methods, con, topo):
        """Checks that the results of the connectivity computations have the
        appropriate dimensions."""
        n_times = con_methods[0].n_times

        assert (con.shape[0] == self.n_cons), (
            'The first dimension of connectivity scores does not match the '
            'number of connections. Please contact the mne-connectivity ' 
            'developers.'
        )

        assert (con.shape[1] == self.n_freqs), (
            'The second dimension of connectivity scores does not match the '
            'number of frequencies. Please contact the mne-connectivity ' 
            'developers.'
        )

        if n_times != 0:
            assert (con.shape[2] == n_times), (
                'The third dimension of connectivity scores does not match '
                'the number of timepoints. Please contact the mne-connectivity ' 
                'developers.'
            )
        
        if topo is not None:
            assert (topo[0].shape[0] == self.n_cons and topo[1].shape[0]), (
                'The first dimension of topographies does not match the number '
                'of connections. Please contact the mne-connectivity '
                'developers.'
            )

            for con_i in range(self.n_cons):
                assert (
                    topo[0][con_i].shape[1] == self.n_freqs and
                    topo[1][con_i].shape[1] == self.n_freqs
                ), (
                    'The second dimension of topographies does not match the '
                    'number of frequencies. Please contact the '
                    'mne-connectivity developers.'
                )

                if n_times != 0:
                    assert (
                        topo[0][con_i].shape[2] == n_times and
                        topo[1][con_i].shape[2] == n_times
                    ), (
                        'The third dimension of topographies does not match '
                        'the number of timepoints. Please contact the '
                        'mne-connectivity developers.'
                    )

    def _compute_faverage(self, con, topo):
        """Computes the average connectivity across the frequency bands."""
        con_shape = (self.n_cons, self.n_bands) + con.shape[2:]
        con_bands = np.empty(con_shape, dtype=con.dtype)
        for band_idx in range(self.n_bands):
            con_bands[:, band_idx] = np.mean(
                con[:, self.freq_idx_bands[band_idx]], axis=1
            )

        if topo is not None:
            topo_bands = np.empty((2, self.n_cons), dtype=topo.dtype)
            for group_i in range(2):
                for con_i in range(self.n_cons):
                    band_topo = [
                        np.mean(topo[group_i][con_i][:, freq_idx_band], axis=1)
                        for freq_idx_band in self.freq_idx_bands
                    ]
                    topo_bands[group_i][con_i] = np.array(band_topo).T
        else:
            topo_bands = None
        
        return con_bands, topo_bands

    def _collate_connectivity_results(self):
        """Collects the connectivity results for non-GC with SVD analysis and GC
        with SVD together according to the order in which the respective methods
        were called."""
        self.con = [*self.remaining_con, *self.separate_gc_con]
        self.topo = [*self.remaining_topo, *self.separate_gc_topo]

        if self.remaining_con and self.separate_gc_con:
            # orders the results according to the order they were called
            methods_order = [
                *[mtype.name for mtype in self.remaining_method_types],
                *[mtype.name for mtype in self.separate_gc_method_types]
            ]
            self.con = [
                self.con[methods_order.index(mtype.name)] for mtype in
                self.con_method_types
            ]
            self.topo = [
                self.topo[methods_order.index(mtype.name)] for mtype in
                self.con_method_types
            ]

        # else if only separate GC connectivity, results already in order in
        # which they were called
        # else you only have the remaining (non-GC with SVD/discontinuous
        # frequency) results, already in the order in which they were called

    def store_connectivity_results(self):
        """Stores multivariate connectivity results in mne-connectivity
        objects."""
        # create a list of connectivity containers
        self.connectivity = []
        for _con, _topo, _method in zip(self.con, self.topo, self.method):
            kwargs = dict(
                data=_con, topographies=_topo, names=self.names,
                freqs=self.freqs, method=_method, n_nodes=self.n_nodes,
                spec_method=self.mode, indices=self.indices,
                n_components=(self.n_seed_components, self.n_target_components),
                n_epochs_used=self.n_epochs, freqs_used=self.freqs_used,
                times_used=self.times, n_tapers=self.n_tapers, n_lags=None,
                metadata=self.metadata, events=self.events,
                event_id=self.event_id
            )
            if _method in ['gc', 'net_gc', 'trgc', 'net_trgc']:
                kwargs.update(n_lags=self.gc_n_lags)
            # create the connectivity container
            if self.mode in ['multitaper', 'fourier']:
                conn_class = MultivariateSpectralConnectivity
            else:
                assert self.mode == 'cwt_morlet'
                conn_class = MultivariateSpectroTemporalConnectivity
                kwargs.update(times=self.times)
            self.connectivity.append(conn_class(**kwargs))

        logger.info('[Connectivity computation done]')

        if len(self.method) == 1:
            # for a single method store the connectivity object directly
            self.connectivity = self.connectivity[0]


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
        'cwt_morlet'.

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
    """Compute frequency- and time-frequency-domain multivariate connectivity
    measures.

    The connectivity method(s) are specified using the "method" parameter.
    All methods are based on estimates of the cross-spectral density (CSD) Sxy.

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
        Connectivity measure(s) to compute. These can be ``['mic', 'mim', 'gc',
        'net_gc', 'trgc', 'net_trgc']``.
    indices : tuple of tuple of array
        Two tuples of arrays with indices of connections for which to compute
        connectivity.
    sfreq : float
        The sampling frequency.
    mode : str
        Spectrum estimation mode can be either: 'multitaper', 'fourier', or
        'cwt_morlet'.
    fmin : float | tuple of float
        The lower frequency of interest. Multiple bands are defined using
        a tuple, e.g., (8., 20.) for two bands with 8Hz and 20Hz lower freq.
        If None the frequency corresponding to an epoch length of 5 cycles
        is used.
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
        Only use tapers with more than 90%% spectral concentration within
        bandwidth. Only used in 'multitaper' mode.
    cwt_freqs : array
        Array of frequencies of interest. Only used in 'cwt_morlet' mode.
    cwt_n_cycles : float | array of float
        Number of cycles. Fixed number or one per frequency. Only used in
        'cwt_morlet' mode.
    block_size : int
        How many CSD entries to compute at once (higher numbers are faster
        but require more memory).
    n_jobs : int
        How many epochs and frequencies to process in parallel.
    %(verbose)s

    Returns
    -------
    con : array | list of array
        Computed connectivity measure(s). Either an instance of
        ``MultivariateSpectralConnectivity`` or
        ``MultivariateSpectroTemporalConnectivity``.
        The shape of each connectivity dataset is (n_con, n_freqs)
        mode: 'multitaper' or 'fourier' (n_con, n_freqs, n_times)
        mode: 'cwt_morlet' where "n_con = len(indices[0])".

    See Also
    --------
    mne_connectivity.MultivariateSpectralConnectivity
    mne_connectivity.MultivariateSpectroTemporalConnectivity

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
    one Epoch will result in errorful values.**

    The spectral densities can be estimated using a multitaper method with
    digital prolate spheroidal sequence (DPSS) windows, a discrete Fourier
    transform with Hanning windows, or a continuous wavelet transform using
    Morlet wavelets. The spectral estimation mode is specified using the
    "mode" parameter.

    By default, the connectivity between all signals is computed (only
    connections corresponding to the lower-triangular part of the
    connectivity matrix). If one is only interested in the connectivity
    between some signals, the "indices" parameter can be used. For example,
    to compute the connectivity between the signal with index 0 and signals
    "2, 3, 4" (a total of 3 connections) one can use the following::

        indices = (np.array([0, 0, 0]),    # row indices
                   np.array([2, 3, 4]))    # col indices

        con_flat = spectral_connectivity(data, method='coh',
                                         indices=indices, ...)

    In this case con_flat.shape = (3, n_freqs). The connectivity scores are
    in the same order as defined indices.

    **Supported Connectivity Measures**

    The connectivity method(s) is specified using the "method" parameter. The
    following methods are supported. Multiple measures can be computed at once
    by using a list/tuple, e.g., ``['mic', 'net_trgc']`` to compute MIC and
    Net TRGC.

        'mic' : Maximised Imaginary Coherence :footcite:`EwaldEtAl2012`
            Here, topographies of the connectivity are also computed for each
            channel, providing spatial information about the connectivity.

        'mim' : Multivariate Interaction Measure :footcite:`EwaldEtAl2012`

        'gc' : Granger causality :footcite:SETHPAPER with connectivity as::
            [seeds -> targets]

        'net_gc' : Net Granger causality with connectivity as::
            [seeds -> targets] - [targets -> seeds]

        'trgc' : Time-Reversed Granger causality with connectivity as::
            [seeds -> targets] - time-reversed[seeds -> targets]

        'net_trgc' : Net Time-Reversed Granger causality with connectivity
        as::
            ([seeds -> targets] - [targets -> seeds]) - 
            (time-reversed[seeds -> targets] - time-reversed[targets -> seeds])

        Time-reversed Granger causality methods are recommended for maximum
        robustness to noise :footcite:STEFANPAPER.

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

        'plv' : Phase-Locking Value (PLV) :footcite:`LachauxEtAl1999` given
        by::

            PLV = |E[Sxy/|Sxy|]|

        'ciplv' : corrected imaginary PLV (icPLV)
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

    References
    ----------
    .. footbibliography::
    """
    connectivity_computation = _MVCSpectralEpochs(
        data=data, indices=indices, names=names, method=method, sfreq=sfreq,
        mode=mode, tmin=tmin, tmax=tmax, fmin=fmin, fmax=fmax, fskip=fskip,
        faverage=faverage, cwt_freqs=cwt_freqs, mt_bandwidth=mt_bandwidth,
        mt_adaptive=mt_adaptive, mt_low_bias=mt_low_bias,
        cwt_n_cycles=cwt_n_cycles, n_seed_components=n_seed_components,
        n_target_components=n_target_components, gc_n_lags=gc_n_lags,
        block_size=block_size, n_jobs=n_jobs, verbose=verbose
    )

    connectivity_computation.compute_csd_and_connectivity()

    connectivity_computation.store_connectivity_results()

    return connectivity_computation.connectivity
