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
from mne.utils import logger, ProgressBar
from ..base import (
    MultivariateSpectralConnectivity, MultivariateSpectroTemporalConnectivity
)
from .epochs import (
    _assemble_spectral_params, _check_estimators, _epoch_spectral_connectivity,
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
    
    # different possible forms of GC with information on how to use the methods
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

    # different possible forms of coherence with information on how to use the
    # methods
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

    discontinuous_fbands = False

    compute_gc_separately = False

    # storage for GC with SVD results (which must be computed separately)
    gc_with_svd_method_types = []
    gc_with_svd_con = []
    gc_with_svd_topo = []

    # storage for non-GC with SVD results
    non_gc_svd_con = []
    non_gc_svd_topo = []

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
        self._sort_freq_inputs()
        self._sort_estimator_inputs()
        self._sort_data_info()
        self._sort_indices_inputs()
        self._sort_svd_inputs()
    
    def _sort_parallelisation_inputs(self):
        """Establishes parallelisation of the function for computing the CSD if 
        n_jobs > 1, else uses the standard, non-parallelised function."""
        self.parallel, self._epoch_spectral_connectivity, _ = (
            parallel_func(
                _epoch_spectral_connectivity, self.n_jobs, verbose=self.verbose
            )
        )
    
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

        self.non_gc_svd_method_types = copy.deepcopy(self.con_method_types)

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
                            if n_comps > len(chs) or n_comps <= 0:
                                raise ValueError(
                                    'The number of components to take cannot '
                                    'be greater than the number of channels in '
                                    'a given seed/target and must be greater '
                                    'than 0'
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
                        elif not isinstance(n_comps, None):
                            raise TypeError(
                                'n_seed_components and n_target_components '
                                'must be lists of `None`, `int`, or the string '
                                '"rank"'
                            )
                        index_i += 1

    def compute_csd_and_connectivity(self):
        """Compute the CSD of the data and derive connectivity results from
        it."""
        if self.present_gc_methods and self.compute_gc_separately:
            self._compute_gc_with_svd_csd_and_connectivity()

        self._compute_non_gc_with_svd_csd_and_connectivity()

        self._collate_connectivity_results()
    
    def _compute_gc_with_svd_csd_and_connectivity(self):
        """Computes the CSD and connectivity for GC methods separately from
        other methods if SVD is being performed.
        
        If SVD is being performed with GC, this has to be done on the
        timeseries data for each connection separately, and so this transformed
        data cannot be used to compute the CSD for coherence-based connectivity
        methods.
        """
        # finds the GC methods to compute
        self.gc_with_svd_method_types = [
            mtype for mtype in self.con_method_types if mtype.name in
            self.gc_method_names
        ]

        # performs SVD on the timeseries data for each connection
        seed_target_data, n_seeds = self._seeds_targets_svd()

        # computes GC for each connection separately
        n_gc_methods = len(self.present_gc_methods)
        self.gc_with_svd_con = [[] for _ in range(n_gc_methods)]
        self.gc_with_svd_topo = [None for _ in range(n_gc_methods)]
        for con_data, n_seed_comps in zip(seed_target_data, n_seeds):
            new_indices = (
                [np.arange(n_seed_comps).tolist()],
                [np.arange(n_seed_comps, con_data.shape[1]).tolist()]
            )

            self._compute_csd()

            this_con, _, = self._compute_connectivity(new_indices)

            for method_i in range(n_gc_methods):
                self.gc_with_svd_con[method_i].append(this_con[method_i])

        self.gc_with_svd_con = [
            np.squeeze(np.array(this_con), 1) for this_con in
            self.gc_with_svd_con
        ]

        # finds the methods still needing to be computed
        self.non_gc_with_svd_method_types = [
            mtype for mtype in self.con_method_types if
            mtype not in self.gc_with_svd_method_types
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
                target_data = _epochs_svd(epochs[:, targets, :], n_target_comps)
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

    def _compute_non_gc_with_svd_csd_and_connectivity(self):
        """Computes connectivity where no Granger causality with SVD is being
        performed, meaning a single CSD can be computed and the connectivity
        computations performed for all connections together."""
        self._compute_csd()

        self.non_gc_with_svd_con, self.non_gc_with_svd_topo = (
            self._compute_connectivity(self.remapped_indices)
        )

    def _compute_csd(self):
        """Computes the cross-spectral density of the data in preparation for
        the multivariate connectivity computations."""
        logger.info('Connectivity computation...')

        self._prepare_csd_computation()

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
                for method, epoch_csd in zip(self.con_methods, epoch[0]):
                    method.combine(epoch_csd)

    def _prepare_csd_computation(self):
        """Collects and returns information in preparation for computing the
        cross-spectral density."""
        self.epoch_blocks = [
            epoch for epoch in _get_n_epochs(self.data, self.n_jobs)
        ]

        # initialize everything times and frequencies
        (
            _, self.times, n_times, self.times_in, self.n_times_in, tmin_idx,
            tmax_idx, self.n_freqs, freq_mask, self.freqs, self.freqs_bands,
            self.freq_idx_bands, self.n_signals, _, self.warn_times
        ) = _prepare_connectivity(
            epoch_block=self.epoch_blocks[0], times_in=self.times_in,
            tmin=self.tmin, tmax=self.tmax, fmin=self.fmin, fmax=self.fmax,
            sfreq=self.sfreq, indices=self.indices, mode=self.mode,
            fskip=self.fskip, n_bands=self.n_bands, cwt_freqs=self.cwt_freqs,
            faverage=self.faverage
        )

        # get the window function, wavelets, etc for different modes
        spectral_params, mt_adaptive, self.n_times_spectrum, self.n_tapers = (
            _assemble_spectral_params(
                mode=self.mode, n_times=n_times, mt_adaptive=self.mt_adaptive,
                mt_bandwidth=self.mt_bandwidth, sfreq=self.sfreq,
                mt_low_bias=self.mt_low_bias, cwt_n_cycles=self.cwt_n_cycles,
                cwt_freqs=self.cwt_freqs, freqs=self.freqs, freq_mask=freq_mask
            )
        )

        # get the appropriate indices for computing connectivity on
        self._sort_con_indices()

        self._instantiate_con_estimators()

        # collate settings
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

    def _sort_con_indices(self):
        """Maps indices to the unique indices, finds the signals for which the
        CSD needs to be computed (and how many used signals there are), and gets
        the seed-target indices for the CSD."""
        # map indices to unique indices
        unique_indices = np.unique(np.concatenate(sum(self.indices, [])))
        remapping = {ch_i: sig_i for sig_i, ch_i in enumerate(unique_indices)}
        self.remapped_indices = tuple([
            [[remapping[idx] for idx in idcs] for idcs in indices_group]
            for indices_group in self.indices
        ])

        # unique signals for which we actually need to compute CSD
        self.sig_idx = np.unique(sum(sum(self.remapped_indices, []), []))
        self.use_n_signals = len(self.sig_idx)

        # gets seed-target indices for CSD
        self.idx_map = [
            np.repeat(
                self.sig_idx, len(self.sig_idx)),
                np.tile(self.sig_idx, len(self.sig_idx)
            )
        ]

    def _instantiate_con_estimators(self):
        """Create instances of the connectivity estimators and log the methods
        being computed."""
        self.con_methods = []
        for mtype in self.con_method_types:
            if "n_lags" in list(inspect.signature(mtype).parameters):
                # if a GC method, provide n_lags argument
                self.con_methods.append(
                    mtype(
                        self.use_n_signals, self.n_cons, self.n_freqs,
                        self.n_times_spectrum, self.gc_n_lags, self.n_jobs
                    )
                )
            else:
                self.con_methods.append(
                    mtype(
                        self.use_n_signals, self.n_cons, self.n_freqs,
                        self.n_times_spectrum, self.n_jobs
                    )
                )

    def _compute_connectivity(self, indices):
        """Computes the multivariate connectivity results."""
        con = [None for _ in range(len(self.con_methods))]
        topo = [None for _ in range(len(self.con_methods))]

        # add the GC results to con in the correct positions according to the
        # order of con_methods
        con = self._compute_gc_connectivity(con, indices)

        # add the coherence results to con in the correct positions according to
        # the order of con_methods
        con, topo = self._compute_coh_connectivity(con, topo, indices)

        method_i = 0
        for method_con, method_topo in zip(con, topo):
            assert method_con is not None, (
                'A connectivity result has been missed. Please contact the '
                'mne-connectivity developers.'
            )

            self._check_correct_results_dimensions(method_con, method_topo)

            if self.faverage:
                con[method_i], topo[method_i] = self._compute_faverage(
                        con=method_con, topo=method_topo,
                        freq_idx_bands=self.freq_idx_bands
                    )
            
            method_i += 1

        self.freqs_used = self.freqs
        if self.faverage:
            # return max and min frequencies that were averaged for each band
            self.freqs_used = [
                [np.min(band), np.max(band)] for band in self.freqs_bands
            ]

        # number of nodes in the original data,
        self.n_nodes = self.n_signals

        return con, topo

    def _compute_gc_connectivity(self, con, indices):
        """Computes GC connectivity.
        
        Different GC methods can rely on common information, so rather than
        re-computing this information everytime a different GC method is called,
        store this information such that it can be accessed to compute the final
        GC connectivity scores when needed.
        """
        self._get_gc_forms_to_compute()

        if self.compute_gc_forms:
            # computes autocovariance once and assigns it to all GC methods
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
            
            # combines results of individual GC forms
            con = self._combine_gc_forms(con, gc_scores)

        return con

    def _get_gc_forms_to_compute(self):
        """Finds the GC forms that need to be computed."""
        self.compute_gc_forms = {}
        for form_name, form_info in self.possible_gc_forms.items():
            for method in self.con_methods:
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

    def _combine_gc_forms(self, con, gc_scores):
        """Combines the information from all the different GC forms so that the
        final connectivity scores for the requested GC methods are returned."""
        for method_i, method in enumerate(self.con_methods):
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

    def _compute_coh_connectivity(self, con, topo, indices):
        """Computes MIC and MIM connectivity.
        
        MIC and MIM rely on common information, so rather than re-computing this
        information everytime a different coherence method is called, store this
        information such that it can be accessed to compute the final MIC and
        MIM connectivity scores when needed.
        """
        self._get_coh_form_to_compute()

        if self.compute_coh_form:
            # compute connectivity for MIC and/or MIM in a single instance
            form_name = list(self.compute_coh_form.keys())[0]  # only one there
            form_info = self.compute_coh_form[form_name]
            form_info['method_class'].compute_con(
                indices[0], indices[1], self.n_seed_components,
                self.n_target_components, self.n_epochs, form_name
            )
        
            # store the MIC and/or MIM results in the right places
            for method_i, method in enumerate(self.con_methods):
                if method.name == "MIC":
                    con[method_i] = form_info['method_class'].mic_scores
                    topo[method_i] = form_info['method_class'].topographies
                elif method.name == "MIM":
                    con[method_i] = form_info['method_class'].mim_scores

        return con, topo

    def _get_coh_form_to_compute(self):
        """Finds the coherence form that need to be computed."""
        method_names = [method.name for method in self.con_methods]
        self.compute_coh_form = {}
        for form_name, form_info in self.possible_coh_forms.items():
            if (
                all(name in method_names for name in form_info['for_methods'])
                and not any(
                    name in method_names for name in
                    form_info['exclude_methods']
                )
            ):
                coh_class = self.con_methods[
                    method_names.index(form_info['for_methods'][0])
                ]
                form_info.update(method_class=coh_class)
                self.compute_coh_form[form_name] = form_info
                break # only one form is possible at any one instance

    def _check_correct_results_dimensions(self, con, topo):
        """Checks that the results of the connectivity computations have the
        appropriate dimensions."""
        n_times = self.con_methods[0].n_times
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

    def _collate_connectivity_results(self):
        """Collects the connectivity results for non-GC with SVD analysis and GC
        with SVD together according to the order in which the respective methods
        were called."""
        # Collate connectivity results
        self.con = [*self.non_gc_with_svd_con, *self.gc_with_svd_con]
        self.topo = [*self.non_gc_with_svd_topo, *self.gc_with_svd_topo]

        if self.non_gc_with_svd_con and self.gc_with_svd_con:
            # orders the results according to the order they were called
            methods_order = [
                *[mtype.name for mtype in self.non_gc_with_svd_method_types],
                *[mtype.name for mtype in self.gc_with_svd_method_types]
            ]
            self.con = [
                self.con[methods_order.index(mtype.name)] for mtype in
                self.con_method_types
            ]
            self.topo = [
                self.topo[methods_order.index(mtype.name)] for mtype in
                self.con_method_types
            ]

        elif not self.non_gc_with_svd_con and self.gc_with_svd_con:
            # results already in order in which they were called
            # finds the remapped indices of non-SVD data
            unique_indices = np.unique(np.concatenate(sum(self.indices, [])))
            remapping = {
                ch_i: sig_i for sig_i, ch_i in enumerate(unique_indices)
            }
            self.remapped_indices = [
                [[remapping[idx] for idx in idcs] for idcs in indices_group] for
                indices_group in self.indices
            ]
        
        # else you only have the non-GC with SVD results, already in the order
        # in which they were called

    def store_connectivity_results(self):
        """Stores multivariate connectivity results in mne-connectivity
        objects."""
        # create a list of connectivity containers
        self.connectivity = []
        for _con, _topo, _method in zip(self.con, self.topo, self.method):
            kwargs = dict(
                data=_con, topographies=_topo, names=self.names,
                freqs=self.freqs, method=_method, n_nodes=self.n_nodes,
                spec_method=self.mode, indices=self.remapped_indices,
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
            self.connectivity = self.conectivity[0]

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
            gc_svd_method_types, non_gc_non_svd_method_types, gc_svd_con,
            gc_svd_topo, times, n_tapers, n_epochs, freqs, freqs_used, n_nodes,
            remapped_indices
        ) = _handle_gc_with_svd_connectivity(
            data=data, indices=indices, n_seed_components=n_seed_components,
            n_target_components=n_target_components,
            con_method_types=con_method_types,
            present_gc_methods=present_gc_methods, call_params=call_params
        )
        con = []
        topo = []
    else:
        non_gc_non_svd_method_types = con_method_types
        gc_svd_method_types = []
        gc_svd_con = []
        gc_svd_topo = []

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
        con=con, gc_svd_con=gc_svd_con, topo=topo, gc_svd_topo=gc_svd_topo,
        non_gc_non_svd_method_types=non_gc_non_svd_method_types,
        gc_svd_method_types=gc_svd_method_types,
        con_method_types=con_method_types, indices=indices,
        remapped_indices=remapped_indices
    )

    return _store_connectivity(
        con, topo, method, names, freqs, n_nodes, mode, remapped_indices,
        n_epochs, freqs_used, times, n_tapers, gc_n_lags, metadata, events,
        event_id
    )
    """


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
    metrics_str = ', '.join([meth.name for meth in con_method_types])
    logger.info('    the following metrics will be computed: %s' % metrics_str)

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
    n_seed_components = copy.copy(n_seed_components)
    n_target_components = copy.copy(n_target_components)

    # finds if any SVD has been requested for seeds and/or targets
    perform_svd = False
    if n_seed_components is None:
         n_seed_components = [None for _ in range(n_cons)]
    if n_target_components is None:
         n_target_components = [None for _ in range(n_cons)]
    for n_components in (n_seed_components, n_target_components):
        if not isinstance(n_components, list):
            raise TypeError(
                'n_seed_components and n_target_components must be lists'
            )
        if n_cons != len(n_components):
            raise ValueError(
                'n_seed_components and n_target_components must have the same '
                'length as specified the number of connections in indices. '
                f'Got: {len(n_components)} components and {n_cons} connections'
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
                                'must be lists of `None`, `int`, or the string '
                                '"rank"'
                            )
                        n_components[index_i] = np.min(
                            np.linalg.matrix_rank(
                                epochs[:, chs, :], tol=nonzero_tol
                            )
                        )
                    elif not isinstance(n_comps, None):
                        raise TypeError(
                            'n_seed_components and n_target_components must be '
                            'lists of `None`, `int`, or the string "rank"'
                        )
                    index_i += 1
    
    return perform_svd, n_seed_components, n_target_components

def _handle_gc_with_svd_connectivity(
    data, indices, n_seed_components, n_target_components, con_method_types,
    present_gc_methods, call_params
):
    """Computes Granger causality connectivity if SVD is being performed, in
    which case the SVD, CSD computation, and connectivity computation must be
    performed for each connection separately."""
    faverage = call_params['faverage']
    n_bands = call_params['n_bands']
    non_gc_non_svd_method_types = copy.deepcopy(con_method_types)

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

    # reconstruct the dimensionality-reduced data (have to transpose the data
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
    n_jobs, n_bands, con_method_types, parallel,
    epoch_spectral_connectivity, times_in, gc_n_lags
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
        if "n_lags" in list(inspect.signature(mtype).parameters):
            # if a GC method, provide n_lags argument
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
    
    return con_methods

def _compute_connectivity(
    con_methods, indices, n_seed_components, n_target_components, n_epochs,
    n_cons, faverage, n_freqs, n_bands, freq_idx_bands, freqs_bands, n_signals,
    freqs
):
    """Computes the multivariate connectivity results."""
    con = [None for _ in range(len(con_methods))]
    topo = [None for _ in range(len(con_methods))]

    # add the GC results to con in the correct positions according to the order
    # of con_methods
    con = _compute_gc_connectivity(
        con=con, con_methods=con_methods, indices=indices, n_epochs=n_epochs
    )

    # add the coherence results to con in the correct positions according to the
    # order of con_methods
    con, topo = _compute_coh_connectivity(
        con=con, topo=topo, con_methods=con_methods, indices=indices,
        n_epochs=n_epochs, n_seed_components=n_seed_components,
        n_target_components=n_target_components
    )

    method_i = 0
    for method_con, method_topo in zip(con, topo):
        assert method_con is not None, (
            'A connectivity results has been missed. Please contact the '
            'mne-connectivity developers.'
        )

        _check_correct_results_dimensions(
            method_con, method_topo, n_cons, n_freqs, con_methods[0].n_times
        )

        if faverage:
            con[method_i], topo[method_i] = _compute_faverage(
                    con=method_con, topo=method_topo, n_cons=n_cons,
                    n_bands=n_bands, freq_idx_bands=freq_idx_bands
                )
        
        method_i += 1

    freqs_used = freqs
    if faverage:
        # for each band we return the max and min frequencies that were averaged
        freqs_used = [[np.min(band), np.max(band)] for band in freqs_bands]

    # number of nodes in the original data,
    n_nodes = n_signals

    return con, topo, freqs_used, n_nodes

def _compute_gc_connectivity(con, con_methods, indices, n_epochs):
    """Computes GC connectivity.
    
    Different GC methods can rely on common information, so rather than
    re-computing this information everytime a different GC method is called,
    store this information such that it can be accessed to compute the final GC
    connectivity scores when needed.
    """
    compute_gc_forms = _get_gc_forms_to_compute(con_methods=con_methods)

    if compute_gc_forms:
        # computes autocovariance once and assigns it to all GC methods
        _compute_and_set_gc_autocov(compute_gc_forms, n_epochs)

        gc_scores = {}
        for form_name, form_info in compute_gc_forms.items():
            # computes connectivity for individual GC forms
            form_info['method_class'].compute_con(
                indices[0], indices[1], form_info['flip_seeds_targets'], 
                form_info['reverse_time'], form_name
            )

            # assigns connectivity score to their appropriate GC forms for
            # combining into the final GC method results
            gc_scores[form_name] = form_info['method_class'].con_scores
        
        # combines results of individual GC forms
        con = _combine_gc_forms(con, con_methods, gc_scores)

    return con

def _get_gc_forms_to_compute(con_methods):
    """Finds the GC forms that need to be computed."""
    # different possible forms of GC with information on how to handle indices,
    # time-direction, what GC variants this form is used for, and a place to
    # store the class where the results will be computed
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

    compute_gc_forms = {}
    for form_name, form_info in possible_gc_forms.items():
        for method in con_methods:
            if (
                method.name in form_info['for_methods'] and
                form_name not in compute_gc_forms.keys()
            ):
                form_info.update(method_class=copy.deepcopy(method))
                compute_gc_forms[form_name] = form_info
    
    return compute_gc_forms

def _compute_and_set_gc_autocov(compute_gc_forms, n_epochs):
    """Computes autocovariance once and assigns it to all GC methods."""
    first_form = True
    for form_info in compute_gc_forms.values():
        if first_form:
            form_info['method_class'].compute_autocov(n_epochs)
            autocov = form_info['method_class'].autocov.copy()
            first_form = False
        else:
            form_info['method_class'].autocov = autocov

def _combine_gc_forms(con, con_methods, gc_scores):
    """Combines the information from all the different GC forms so that the
    final connectivity scores for the requested GC methods are returned."""
    for method_i, method in enumerate(con_methods):
        if method.name == 'GC':
            con[method_i] = gc_scores['seeds -> targets']
        elif method.name == 'Net GC':
            con[method_i] = (
                gc_scores['seeds -> targets'] - gc_scores['targets -> seeds']
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

def _compute_coh_connectivity(
    con, topo, con_methods, indices, n_epochs, n_seed_components,
    n_target_components
):
    """Computes MIC and MIM connectivity.
    
    MIC and MIM rely on common information, so rather than re-computing this
    information everytime a different coherence method is called, store this
    information such that it can be accessed to compute the final MIC and MIM
    connectivity scores when needed.
    """
    compute_coh_form = _get_coh_form_to_compute(con_methods=con_methods)

    if compute_coh_form:
        # compute connectivity for MIC and/or MIM in a single instance
        form_name = list(compute_coh_form.keys())[0] # MIC & MIM, MIC, or MIM
        form_info = compute_coh_form[form_name]
        form_info['method_class'].compute_con(
            indices[0], indices[1], n_seed_components, n_target_components,
            n_epochs, form_name
        )
    
        # store the MIC and/or MIM results in the right places
        for method_i, method in enumerate(con_methods):
            if method.name == "MIC":
                con[method_i] = form_info['method_class'].mic_scores
                topo[method_i] = form_info['method_class'].topographies
            elif method.name == "MIM":
                con[method_i] = form_info['method_class'].mim_scores

    return con, topo

def _get_coh_form_to_compute(con_methods):
    """Finds the coherence form that need to be computed."""
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

    method_names = [method.name for method in con_methods]
    compute_coh_form = {}
    for form_name, form_info in possible_coh_forms.items():
        if (
            all(name in method_names for name in form_info['for_methods']) and
            not any(
                name in method_names for name in form_info['exclude_methods']
            )
        ):
            coh_class = con_methods[
                method_names.index(form_info['for_methods'][0])
            ]
            form_info.update(method_class=coh_class)
            compute_coh_form[form_name] = form_info
            break # only one form is possible at any one instance
    
    return compute_coh_form

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

def _compute_faverage(con, topo, n_cons, n_bands, freq_idx_bands):
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
    con, gc_svd_con, topo, gc_svd_topo, non_gc_non_svd_method_types,
    gc_svd_method_types, con_method_types, indices, remapped_indices
):
    """Collects the connectivity results for non-GC, non-SVD analysis and GC SVD
    together according to the order in which the respective methods were
    called."""
    # Collate connectivity results
    if con and gc_svd_con:
        # combines SVD GC and non-GC, non-SVD results
        con.extend(gc_svd_con)
        topo.extend(gc_svd_topo)
        # orders the results according to the order they were called
        methods_order = [
            *[mtype.name for mtype in non_gc_non_svd_method_types],
            *[mtype.name for mtype in gc_svd_method_types]
        ]
        con = [
            con[methods_order.index(mtype.name)] for mtype in con_method_types
        ]
        topo = [
            topo[methods_order.index(mtype.name)] for mtype in con_method_types
        ]

    elif not con and gc_svd_con:
        # leaves SVD GC results as the only results
        con = gc_svd_con
        topo = gc_svd_topo
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
    freqs_used, times, n_tapers, gc_n_lags, metadata, events, event_id
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
            n_tapers=n_tapers, n_lags=None, metadata=metadata, events=events,
            event_id=event_id
        )
        if _method in ['gc', 'net_gc', 'trgc', 'net_trgc']:
            kwargs.update(n_lags=gc_n_lags)
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