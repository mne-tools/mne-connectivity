# Authors: Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#          Denis A. Engemann <denis.engemann@gmail.com>
#          Adam Li <adam2392@gmail.com>
#
# License: BSD (3-clause)

from functools import partial
import inspect

import numpy as np
import scipy as sp
from mne import EpochsArray, compute_rank, create_info
from mne.epochs import BaseEpochs
from mne.parallel import parallel_func
from mne.source_estimate import _BaseSourceEstimate
from mne.time_frequency.multitaper import (_csd_from_mt,
                                           _mt_spectra, _psd_from_mt,
                                           _psd_from_mt_adaptive)
from mne.time_frequency.tfr import cwt, morlet
from mne.time_frequency.multitaper import _compute_mt_params
from mne.utils import (
    ProgressBar, _arange_div, _check_option, _time_mask, logger, warn, verbose)

from ..base import (SpectralConnectivity, SpectroTemporalConnectivity)
from ..utils import fill_doc, check_indices


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


def _prepare_connectivity(epoch_block, times_in, tmin, tmax,
                          fmin, fmax, sfreq, indices,
                          method, mode, fskip, n_bands,
                          cwt_freqs, faverage):
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

    if indices is None:
        if any(this_method in _multivariate_methods for this_method in method):
            if any(this_method in _gc_methods for this_method in method):
                raise ValueError(
                    'indices must be specified when computing Granger '
                    'causality, as all-to-all connectivity is not supported')
            else:
                logger.info('using all indices for multivariate connectivity')
                indices_use = (np.arange(n_signals, dtype=int),
                               np.arange(n_signals, dtype=int))
        else:
            logger.info('only using indices for lower-triangular matrix')
            # only compute r for lower-triangular region
            indices_use = np.tril_indices(n_signals, -1)
    else:
        if any(this_method in _gc_methods for this_method in method):
            if set(indices[0]).intersection(indices[1]):
                raise ValueError(
                    'seed and target indices must not intersect when computing'
                    'Granger causality')
        indices_use = check_indices(indices)

    # number of connectivities to compute
    if any(this_method in _multivariate_methods for this_method in method):
        if len(np.unique(indices_use[0])) != len(np.unique(indices_use[1])):
            raise ValueError(
                'seed and target indices cannot contain repeated channels for '
                'multivariate connectivity')
        n_cons = 1  # UNTIL NEW INDICES FORMAT
    else:
        n_cons = len(indices_use[0])

    logger.info('    computing connectivity for %d connections'
                % n_cons)
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

    return (n_cons, times, n_times, times_in, n_times_in, tmin_idx,
            tmax_idx, n_freqs, freq_mask, freqs, freqs_bands, freq_idx_bands,
            n_signals, indices_use, warn_times)


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


class _EpochMeanConEstBase(_AbstractConEstBase):
    """Base class for methods that estimate connectivity as mean epoch-wise."""

    patterns = None

    def __init__(self, n_cons, n_freqs, n_times):
        self.n_cons = n_cons
        self.n_freqs = n_freqs
        self.n_times = n_times

        if n_times == 0:
            self.csd_shape = (n_cons, n_freqs)
        else:
            self.csd_shape = (n_cons, n_freqs, n_times)

        self.con_scores = None

    def start_epoch(self):  # noqa: D401
        """Called at the start of each epoch."""
        pass  # for this type of con. method we don't do anything

    def combine(self, other):
        """Include con. accumated for some epochs in this estimate."""
        self._acc += other._acc


class _EpochMeanMultivariateConEstBase(_AbstractConEstBase):
    """Base class for mean epoch-wise multivar. con. estimation methods."""

    n_steps = None
    patterns = None

    def __init__(self, n_signals, n_cons, n_freqs, n_times, n_jobs=1):
        self.n_signals = n_signals
        self.n_cons = n_cons
        self.n_freqs = n_freqs
        self.n_times = n_times
        self.n_jobs = n_jobs

        # include time dimension, even when unused for indexing flexibility
        if n_times == 0:
            self.csd_shape = (n_signals**2, n_freqs)
            self.con_scores = np.zeros((n_cons, n_freqs, 1))
        else:
            self.csd_shape = (n_signals**2, n_freqs, n_times)
            self.con_scores = np.zeros((n_cons, n_freqs, n_times))

        # allocate space for accumulation of CSD
        self._acc = np.zeros(self.csd_shape, dtype=np.complex128)

        self._compute_n_progress_bar_steps()

    def start_epoch(self):  # noqa: D401
        """Called at the start of each epoch."""
        pass  # for this type of con. method we don't do anything

    def combine(self, other):
        """Include con. accumulated for some epochs in this estimate."""
        self._acc += other._acc

    def accumulate(self, con_idx, csd_xy):
        """Accumulate CSD for some connections."""
        self._acc[con_idx] += csd_xy

    def _compute_n_progress_bar_steps(self):
        """Calculate the number of steps to include in the progress bar."""
        self.n_steps = int(np.ceil(self.n_freqs / self.n_jobs))

    def _log_connection_number(self, con_i):
        """Log the number of the connection being computed."""
        logger.info('Computing %s for connection %i of %i'
                    % (self.name, con_i + 1, self.n_cons, ))

    def _get_block_indices(self, block_i, limit):
        """Get indices for a computation block capped by a limit."""
        indices = np.arange(block_i * self.n_jobs, (block_i + 1) * self.n_jobs)

        return indices[np.nonzero(indices < limit)]

    def reshape_csd(self):
        """Reshape CSD into a matrix of times x freqs x signals x signals."""
        if self.n_times == 0:
            return (np.reshape(self._acc, (
                self.n_signals, self.n_signals, self.n_freqs, 1)
            ).transpose(3, 2, 0, 1))

        return (np.reshape(self._acc, (
            self.n_signals, self.n_signals, self.n_freqs, self.n_times)
        ).transpose(3, 2, 0, 1))


class _CohEstBase(_EpochMeanConEstBase):
    """Base Estimator for Coherence, Coherency, Imag. Coherence."""

    accumulate_psd = True

    def __init__(self, n_cons, n_freqs, n_times):
        super(_CohEstBase, self).__init__(n_cons, n_freqs, n_times)

        # allocate space for accumulation of CSD
        self._acc = np.zeros(self.csd_shape, dtype=np.complex128)

    def accumulate(self, con_idx, csd_xy):
        """Accumulate CSD for some connections."""
        self._acc[con_idx] += csd_xy


class _CohEst(_CohEstBase):
    """Coherence Estimator."""

    name = 'Coherence'

    def compute_con(self, con_idx, n_epochs, psd_xx, psd_yy):  # lgtm
        """Compute final con. score for some connections."""
        if self.con_scores is None:
            self.con_scores = np.zeros(self.csd_shape)
        csd_mean = self._acc[con_idx] / n_epochs
        self.con_scores[con_idx] = np.abs(csd_mean) / np.sqrt(psd_xx * psd_yy)


class _CohyEst(_CohEstBase):
    """Coherency Estimator."""

    name = 'Coherency'

    def compute_con(self, con_idx, n_epochs, psd_xx, psd_yy):  # lgtm
        """Compute final con. score for some connections."""
        if self.con_scores is None:
            self.con_scores = np.zeros(self.csd_shape,
                                       dtype=np.complex128)
        csd_mean = self._acc[con_idx] / n_epochs
        self.con_scores[con_idx] = csd_mean / np.sqrt(psd_xx * psd_yy)


class _ImCohEst(_CohEstBase):
    """Imaginary Coherence Estimator."""

    name = 'Imaginary Coherence'

    def compute_con(self, con_idx, n_epochs, psd_xx, psd_yy):  # lgtm
        """Compute final con. score for some connections."""
        if self.con_scores is None:
            self.con_scores = np.zeros(self.csd_shape)
        csd_mean = self._acc[con_idx] / n_epochs
        self.con_scores[con_idx] = np.imag(csd_mean) / np.sqrt(psd_xx * psd_yy)


class _MultivariateCohEstBase(_EpochMeanMultivariateConEstBase):
    """Base estimator for multivariate imag. part of coherency methods.

    See Ewald et al. (2012). NeuroImage. DOI: 10.1016/j.neuroimage.2011.11.084
    for equation references.
    """

    name = None
    accumulate_psd = False

    def __init__(self, n_signals, n_cons, n_freqs, n_times, n_jobs=1):
        super(_MultivariateCohEstBase, self).__init__(
            n_signals, n_cons, n_freqs, n_times, n_jobs)

        if self.name == 'MIC':
            self.patterns = np.empty((2, self.n_cons), dtype=object)

    def compute_con(self, indices, ranks, n_epochs):
        """Compute multivariate imag. part of coherency between signals."""
        assert self.name in ['MIC', 'MIM'], (
            'the class name is not recognised, please contact the '
            'mne-connectivity developers')

        csd = self.reshape_csd() / n_epochs
        n_times = csd.shape[0]
        times = np.arange(n_times)
        freqs = np.arange(self.n_freqs)

        con_i = 0
        for seed_idcs, target_idcs, seed_rank, target_rank in zip(
                [indices[0]], [indices[1]], ranks[0], ranks[1]):
            self._log_connection_number(con_i)

            n_seeds = len(seed_idcs)
            con_idcs = [*seed_idcs, *target_idcs]

            C = csd[np.ix_(times, freqs, con_idcs, con_idcs)]

            # Eqs. 32 & 33
            C_bar, U_bar_aa, U_bar_bb = self._csd_svd(
                C, n_seeds, seed_rank, target_rank)

            # Eqs. 3 & 4
            E = self._compute_e(C_bar, n_seeds=U_bar_aa.shape[3])

            if self.name == 'MIC':
                self._compute_mic(
                    E, C, n_seeds, n_times, U_bar_aa, U_bar_bb, con_i)
            else:
                self._compute_mim(E, con_i)

            # Eq. 15 for MIM (same principle for MIC)
            if all(np.unique(seed_idcs) == np.unique(target_idcs)):
                self.con_scores[con_i] /= 2

            con_i += 1

        self.reshape_results()

    def _csd_svd(self, csd, n_seeds, seed_rank, target_rank):
        """Dimensionality reduction of CSD with SVD."""
        n_times = csd.shape[0]
        n_targets = csd.shape[2] - n_seeds

        C_aa = csd[..., :n_seeds, :n_seeds]
        C_ab = csd[..., :n_seeds, n_seeds:]
        C_bb = csd[..., n_seeds:, n_seeds:]
        C_ba = csd[..., n_seeds:, :n_seeds]

        # Eq. 32
        if seed_rank != n_seeds:
            U_aa = np.linalg.svd(np.real(C_aa), full_matrices=False)[0]
            U_bar_aa = U_aa[..., :seed_rank]
        else:
            U_bar_aa = np.broadcast_to(
                np.identity(n_seeds),
                (n_times, self.n_freqs) + (n_seeds, n_seeds))

        if target_rank != n_targets:
            U_bb = np.linalg.svd(np.real(C_bb), full_matrices=False)[0]
            U_bar_bb = U_bb[..., :target_rank]
        else:
            U_bar_bb = np.broadcast_to(
                np.identity(n_targets),
                (n_times, self.n_freqs) + (n_targets, n_targets))

        # Eq. 33
        C_bar_aa = np.matmul(
            U_bar_aa.transpose(0, 1, 3, 2), np.matmul(C_aa, U_bar_aa))
        C_bar_ab = np.matmul(
            U_bar_aa.transpose(0, 1, 3, 2), np.matmul(C_ab, U_bar_bb))
        C_bar_bb = np.matmul(
            U_bar_bb.transpose(0, 1, 3, 2), np.matmul(C_bb, U_bar_bb))
        C_bar_ba = np.matmul(
            U_bar_bb.transpose(0, 1, 3, 2), np.matmul(C_ba, U_bar_aa))
        C_bar = np.append(np.append(C_bar_aa, C_bar_ab, axis=3),
                          np.append(C_bar_ba, C_bar_bb, axis=3), axis=2)

        return C_bar, U_bar_aa, U_bar_bb

    def _compute_e(self, csd, n_seeds):
        """Compute E from the CSD."""
        C_r = np.real(csd)

        parallel, parallel_compute_t, _ = parallel_func(
            _mic_mim_compute_t, self.n_jobs, verbose=False)

        # imag. part of T filled when data is rank-deficient
        T = np.zeros(csd.shape, dtype=np.complex128)
        for block_i in ProgressBar(
                range(self.n_steps), mesg="frequency blocks"):
            freqs = self._get_block_indices(block_i, self.n_freqs)
            parallel(parallel_compute_t(
                C_r[:, f], T[:, f], n_seeds) for f in freqs)

        if not np.isreal(T).all() or not np.isfinite(T).all():
            raise ValueError(
                'the transformation matrix of the data must be real-valued '
                'and contain no NaN or infinity values; check that you are '
                'using full rank data or specify an appropriate rank for the '
                'seeds and targets that is less than or equal to their ranks')
        T = np.real(T)  # make T real if check passes

        # Eq. 4
        D = np.matmul(T, np.matmul(csd, T))

        # E as imag. part of D between seeds and targets
        return np.imag(D[..., :n_seeds, n_seeds:])

    def _compute_mic(self, E, C, n_seeds, n_times, U_bar_aa, U_bar_bb, con_i):
        """Compute MIC and the associated spatial patterns."""
        times = np.arange(n_times)
        freqs = np.arange(self.n_freqs)

        # Eigendecomp. to find spatial filters for seeds and targets
        w_seeds, V_seeds = np.linalg.eigh(
            np.matmul(E, E.transpose(0, 1, 3, 2)))
        w_targets, V_targets = np.linalg.eigh(
            np.matmul(E.transpose(0, 1, 3, 2), E))

        # Spatial filters with largest eigval. for seeds and targets
        alpha = V_seeds[times[:, None], freqs, :, w_seeds.argmax(axis=2)]
        beta = V_targets[times[:, None], freqs, :, w_targets.argmax(axis=2)]

        # Eq. 46; seed spatial patterns
        self.patterns[0][con_i] = (np.matmul(
            np.real(C[..., :n_seeds, :n_seeds]),
            np.matmul(U_bar_aa, np.expand_dims(alpha, axis=3))))[..., 0].T

        # Eq. 47; target spatial patterns
        self.patterns[1][con_i] = (np.matmul(
            np.real(C[..., n_seeds:, n_seeds:]),
            np.matmul(U_bar_bb, np.expand_dims(beta, axis=3))))[..., 0].T

        # Eq. 7
        self.con_scores[con_i] = (np.einsum(
            'ijk,ijk->ij', alpha, np.matmul(E, np.expand_dims(
                beta, axis=3))[..., 0]
        ) / np.linalg.norm(alpha, axis=2) * np.linalg.norm(beta, axis=2)).T

    def _compute_mim(self, E, con_i):
        """Compute MIM (a.k.a. GIM if seeds == targets)."""
        # Eq. 14
        self.con_scores[con_i] = np.matmul(
            E, E.transpose(0, 1, 3, 2)).trace(axis1=2, axis2=3).T

    def reshape_results(self):
        """Remove time dimension from results, if necessary."""
        if self.n_times == 0:
            self.con_scores = self.con_scores[..., 0]

            if self.patterns is not None:
                for group_i, patterns in enumerate(self.patterns):
                    for con_i, pattern in enumerate(patterns):
                        self.patterns[group_i][con_i] = pattern[..., 0]


def _mic_mim_compute_t(C, T, n_seeds):
    """Compute T in place for a single frequency (used for MIC and MIM)."""
    for time_i in range(C.shape[0]):
        T[time_i, :n_seeds, :n_seeds] = sp.linalg.fractional_matrix_power(
            C[time_i, :n_seeds, :n_seeds], -0.5
        )
        T[time_i, n_seeds:, n_seeds:] = sp.linalg.fractional_matrix_power(
            C[time_i, n_seeds:, n_seeds:], -0.5
        )


class _MICEst(_MultivariateCohEstBase):
    """Multivariate imaginary part of coherency (MIC) estimator."""

    name = "MIC"


class _MIMEst(_MultivariateCohEstBase):
    """Multivariate interaction measure (MIM) estimator."""

    name = "MIM"


class _PLVEst(_EpochMeanConEstBase):
    """PLV Estimator."""

    name = 'PLV'
    accumulate_psd = False

    def __init__(self, n_cons, n_freqs, n_times):
        super(_PLVEst, self).__init__(n_cons, n_freqs, n_times)

        # allocate accumulator
        self._acc = np.zeros(self.csd_shape, dtype=np.complex128)

    def accumulate(self, con_idx, csd_xy):
        """Accumulate some connections."""
        self._acc[con_idx] += csd_xy / np.abs(csd_xy)

    def compute_con(self, con_idx, n_epochs):
        """Compute final con. score for some connections."""
        if self.con_scores is None:
            self.con_scores = np.zeros(self.csd_shape)
        plv = np.abs(self._acc / n_epochs)
        self.con_scores[con_idx] = plv


class _ciPLVEst(_EpochMeanConEstBase):
    """corrected imaginary PLV Estimator."""

    name = 'ciPLV'
    accumulate_psd = False

    def __init__(self, n_cons, n_freqs, n_times):
        super(_ciPLVEst, self).__init__(n_cons, n_freqs, n_times)

        # allocate accumulator
        self._acc = np.zeros(self.csd_shape, dtype=np.complex128)

    def accumulate(self, con_idx, csd_xy):
        """Accumulate some connections."""
        self._acc[con_idx] += csd_xy / np.abs(csd_xy)

    def compute_con(self, con_idx, n_epochs):
        """Compute final con. score for some connections."""
        if self.con_scores is None:
            self.con_scores = np.zeros(self.csd_shape)
        imag_plv = np.abs(np.imag(self._acc)) / n_epochs
        real_plv = np.real(self._acc) / n_epochs
        real_plv = np.clip(real_plv, -1, 1)  # bounded from -1 to 1
        mask = (np.abs(real_plv) == 1)  # avoid division by 0
        real_plv[mask] = 0
        corrected_imag_plv = imag_plv / np.sqrt(1 - real_plv ** 2)
        self.con_scores[con_idx] = corrected_imag_plv


class _PLIEst(_EpochMeanConEstBase):
    """PLI Estimator."""

    name = 'PLI'
    accumulate_psd = False

    def __init__(self, n_cons, n_freqs, n_times):
        super(_PLIEst, self).__init__(n_cons, n_freqs, n_times)

        # allocate accumulator
        self._acc = np.zeros(self.csd_shape)

    def accumulate(self, con_idx, csd_xy):
        """Accumulate some connections."""
        self._acc[con_idx] += np.sign(np.imag(csd_xy))

    def compute_con(self, con_idx, n_epochs):
        """Compute final con. score for some connections."""
        if self.con_scores is None:
            self.con_scores = np.zeros(self.csd_shape)
        pli_mean = self._acc[con_idx] / n_epochs
        self.con_scores[con_idx] = np.abs(pli_mean)


class _PLIUnbiasedEst(_PLIEst):
    """Unbiased PLI Square Estimator."""

    name = 'Unbiased PLI Square'
    accumulate_psd = False

    def compute_con(self, con_idx, n_epochs):
        """Compute final con. score for some connections."""
        if self.con_scores is None:
            self.con_scores = np.zeros(self.csd_shape)
        pli_mean = self._acc[con_idx] / n_epochs

        # See Vinck paper Eq. (30)
        con = (n_epochs * pli_mean ** 2 - 1) / (n_epochs - 1)

        self.con_scores[con_idx] = con


class _DPLIEst(_EpochMeanConEstBase):
    """DPLI Estimator."""

    name = 'DPLI'
    accumulate_psd = False

    def __init__(self, n_cons, n_freqs, n_times):
        super(_DPLIEst, self).__init__(n_cons, n_freqs, n_times)

        # allocate accumulator
        self._acc = np.zeros(self.csd_shape)

    def accumulate(self, con_idx, csd_xy):
        """Accumulate some connections."""
        self._acc[con_idx] += np.heaviside(np.imag(csd_xy), 0.5)

    def compute_con(self, con_idx, n_epochs):
        """Compute final con. score for some connections."""
        if self.con_scores is None:
            self.con_scores = np.zeros(self.csd_shape)

        con = self._acc[con_idx] / n_epochs

        self.con_scores[con_idx] = con


class _WPLIEst(_EpochMeanConEstBase):
    """WPLI Estimator."""

    name = 'WPLI'
    accumulate_psd = False

    def __init__(self, n_cons, n_freqs, n_times):
        super(_WPLIEst, self).__init__(n_cons, n_freqs, n_times)

        # store  both imag(csd) and abs(imag(csd))
        acc_shape = (2,) + self.csd_shape
        self._acc = np.zeros(acc_shape)

    def accumulate(self, con_idx, csd_xy):
        """Accumulate some connections."""
        im_csd = np.imag(csd_xy)
        self._acc[0, con_idx] += im_csd
        self._acc[1, con_idx] += np.abs(im_csd)

    def compute_con(self, con_idx, n_epochs):
        """Compute final con. score for some connections."""
        if self.con_scores is None:
            self.con_scores = np.zeros(self.csd_shape)

        num = np.abs(self._acc[0, con_idx])
        denom = self._acc[1, con_idx]

        # handle zeros in denominator
        z_denom = np.where(denom == 0.)
        denom[z_denom] = 1.

        con = num / denom

        # where we had zeros in denominator, we set con to zero
        con[z_denom] = 0.

        self.con_scores[con_idx] = con


class _WPLIDebiasedEst(_EpochMeanConEstBase):
    """Debiased WPLI Square Estimator."""

    name = 'Debiased WPLI Square'
    accumulate_psd = False

    def __init__(self, n_cons, n_freqs, n_times):
        super(_WPLIDebiasedEst, self).__init__(n_cons, n_freqs, n_times)
        # store imag(csd), abs(imag(csd)), imag(csd)^2
        acc_shape = (3,) + self.csd_shape
        self._acc = np.zeros(acc_shape)

    def accumulate(self, con_idx, csd_xy):
        """Accumulate some connections."""
        im_csd = np.imag(csd_xy)
        self._acc[0, con_idx] += im_csd
        self._acc[1, con_idx] += np.abs(im_csd)
        self._acc[2, con_idx] += im_csd ** 2

    def compute_con(self, con_idx, n_epochs):
        """Compute final con. score for some connections."""
        if self.con_scores is None:
            self.con_scores = np.zeros(self.csd_shape)

        # note: we use the trick from fieldtrip to compute the
        # the estimate over all pairwise epoch combinations
        sum_im_csd = self._acc[0, con_idx]
        sum_abs_im_csd = self._acc[1, con_idx]
        sum_sq_im_csd = self._acc[2, con_idx]

        denom = sum_abs_im_csd ** 2 - sum_sq_im_csd

        # handle zeros in denominator
        z_denom = np.where(denom == 0.)
        denom[z_denom] = 1.

        con = (sum_im_csd ** 2 - sum_sq_im_csd) / denom

        # where we had zeros in denominator, we set con to zero
        con[z_denom] = 0.

        self.con_scores[con_idx] = con


class _PPCEst(_EpochMeanConEstBase):
    """Pairwise Phase Consistency (PPC) Estimator."""

    name = 'PPC'
    accumulate_psd = False

    def __init__(self, n_cons, n_freqs, n_times):
        super(_PPCEst, self).__init__(n_cons, n_freqs, n_times)

        # store csd / abs(csd)
        self._acc = np.zeros(self.csd_shape, dtype=np.complex128)

    def accumulate(self, con_idx, csd_xy):
        """Accumulate some connections."""
        denom = np.abs(csd_xy)
        z_denom = np.where(denom == 0.)
        denom[z_denom] = 1.
        this_acc = csd_xy / denom
        this_acc[z_denom] = 0.  # handle division by zero

        self._acc[con_idx] += this_acc

    def compute_con(self, con_idx, n_epochs):
        """Compute final con. score for some connections."""
        if self.con_scores is None:
            self.con_scores = np.zeros(self.csd_shape)

        # note: we use the trick from fieldtrip to compute the
        # the estimate over all pairwise epoch combinations
        con = ((self._acc[con_idx] * np.conj(self._acc[con_idx]) - n_epochs) /
               (n_epochs * (n_epochs - 1.)))

        self.con_scores[con_idx] = np.real(con)


class _GCEstBase(_EpochMeanMultivariateConEstBase):
    """Base multivariate state-space Granger causality estimator."""

    accumulate_psd = False

    def __init__(self, n_signals, n_cons, n_freqs, n_times, n_lags, n_jobs=1):
        super(_GCEstBase, self).__init__(
            n_signals, n_cons, n_freqs, n_times, n_jobs)

        self.freq_res = (self.n_freqs - 1) * 2
        if n_lags >= self.freq_res:
            raise ValueError(
                'the number of lags (%i) must be less than double the '
                'frequency resolution (%i)' % (n_lags, self.freq_res, ))
        self.n_lags = n_lags

    def compute_con(self, indices, ranks, n_epochs):
        """Compute multivariate state-space Granger causality."""
        assert self.name in ['GC', 'GC time-reversed'], (
            'the class name is not recognised, please contact the '
            'mne-connectivity developers')

        csd = self.reshape_csd() / n_epochs

        n_times = csd.shape[0]
        times = np.arange(n_times)
        freqs = np.arange(self.n_freqs)

        con_i = 0
        for seed_idcs, target_idcs, seed_rank, target_rank in zip(
                [indices[0]], [indices[1]], ranks[0], ranks[1]):
            self._log_connection_number(con_i)

            con_idcs = [*seed_idcs, *target_idcs]
            C = csd[np.ix_(times, freqs, con_idcs, con_idcs)]

            con_seeds = np.arange(len(seed_idcs))
            con_targets = np.arange(len(target_idcs)) + len(seed_idcs)

            C_bar = self._csd_svd(
                C, con_seeds, con_targets, seed_rank, target_rank)
            n_signals = seed_rank + target_rank
            con_seeds = np.arange(seed_rank)
            con_targets = np.arange(target_rank) + seed_rank

            autocov = self._compute_autocov(C_bar)
            if self.name == "GC time-reversed":
                autocov = autocov.transpose(0, 1, 3, 2)

            A_f, V = self._autocov_to_full_var(autocov)
            A_f_3d = np.reshape(
                A_f, (n_times, n_signals, n_signals * self.n_lags),
                order="F")
            A, K = self._full_var_to_iss(A_f_3d)

            self.con_scores[con_i] = self._iss_to_ugc(
                A, A_f_3d, K, V, con_seeds, con_targets)

            con_i += 1

        self.reshape_results()

    def _csd_svd(self, csd, seeds, targets, seed_rank, target_rank):
        """Dimensionality reduction of CSD with SVD on the covariance."""
        # sum over times and epochs to get cov. from CSD
        cov = csd.sum(axis=(0, 1))

        n_seeds = len(seeds)
        n_targets = len(targets)

        cov_aa = cov[:n_seeds, :n_seeds]
        cov_bb = cov[n_seeds:, n_seeds:]

        if seed_rank != n_seeds:
            U_aa = np.linalg.svd(np.real(cov_aa), full_matrices=False)[0]
            U_bar_aa = U_aa[:, :seed_rank]
        else:
            U_bar_aa = np.identity(n_seeds)

        if target_rank != n_targets:
            U_bb = np.linalg.svd(np.real(cov_bb), full_matrices=False)[0]
            U_bar_bb = U_bb[:, :target_rank]
        else:
            U_bar_bb = np.identity(n_targets)

        C_aa = csd[..., :n_seeds, :n_seeds]
        C_ab = csd[..., :n_seeds, n_seeds:]
        C_bb = csd[..., n_seeds:, n_seeds:]
        C_ba = csd[..., n_seeds:, :n_seeds]

        C_bar_aa = np.matmul(
            U_bar_aa.transpose(1, 0), np.matmul(C_aa, U_bar_aa))
        C_bar_ab = np.matmul(
            U_bar_aa.transpose(1, 0), np.matmul(C_ab, U_bar_bb))
        C_bar_bb = np.matmul(
            U_bar_bb.transpose(1, 0), np.matmul(C_bb, U_bar_bb))
        C_bar_ba = np.matmul(
            U_bar_bb.transpose(1, 0), np.matmul(C_ba, U_bar_aa))
        C_bar = np.append(np.append(C_bar_aa, C_bar_ab, axis=3),
                          np.append(C_bar_ba, C_bar_bb, axis=3), axis=2)

        return C_bar

    def _compute_autocov(self, csd):
        """Compute autocovariance from the CSD."""
        n_times = csd.shape[0]
        n_signals = csd.shape[2]

        circular_shifted_csd = np.concatenate(
            [np.flip(np.conj(csd[:, 1:]), axis=1), csd[:, :-1]], axis=1)
        ifft_shifted_csd = self._block_ifft(
            circular_shifted_csd, self.freq_res)
        lags_ifft_shifted_csd = np.reshape(
            ifft_shifted_csd[:, :self.n_lags + 1],
            (n_times, self.n_lags + 1, n_signals ** 2), order="F")

        signs = np.repeat([1], self.n_lags + 1).tolist()
        signs[1::2] = [x * -1 for x in signs[1::2]]
        sign_matrix = np.repeat(
            np.tile(np.array(signs), (n_signals ** 2, 1))[np.newaxis],
            n_times, axis=0).transpose(0, 2, 1)

        return np.real(np.reshape(
            sign_matrix * lags_ifft_shifted_csd,
            (n_times, self.n_lags + 1, n_signals, n_signals), order="F"))

    def _block_ifft(self, csd, n_points):
        """Compute block iFFT with n points."""
        shape = csd.shape
        csd_3d = np.reshape(
            csd, (shape[0], shape[1], shape[2] * shape[3]), order="F")

        csd_ifft = np.fft.ifft(csd_3d, n=n_points, axis=1)

        return np.reshape(csd_ifft, shape, order="F")

    def _autocov_to_full_var(self, autocov):
        """Compute full VAR model using Whittle's LWR recursion."""
        if np.any(np.linalg.det(autocov) == 0):
            raise ValueError(
                'the autocovariance matrix is singular; check if your data is '
                'rank deficient and specify an appropriate rank argument <= '
                'the rank of the seeds and targets')

        A_f, V = self._whittle_lwr_recursion(autocov)

        if not np.isfinite(A_f).all():
            raise ValueError('at least one VAR model coefficient is infinite '
                             ' or NaN; check the data you are using')

        try:
            np.linalg.cholesky(V)
        except np.linalg.LinAlgError as np_error:
            raise ValueError(
                'the covariance matrix of the redisuals is not '
                'positive-definite; check the singular values of your data '
                'and specify an appropriate rank argument <= the rank of the '
                'seeds and targets') from np_error

        return A_f, V

    def _whittle_lwr_recursion(self, G):
        """Solve Yule-Walker eqs. for full VAR params. with LWR recursion.

        See: Whittle P., 1963. Biometrika, DOI: 10.1093/biomet/50.1-2.129
        """
        # Initialise recursion
        n = G.shape[2]  # number of signals
        q = G.shape[1] - 1  # number of lags
        t = G.shape[0]  # number of times
        qn = n * q

        cov = G[:, 0, :, :]  # covariance
        G_f = np.reshape(
            G[:, 1:, :, :].transpose(0, 3, 1, 2), (t, qn, n),
            order="F")  # forward autocov
        G_b = np.reshape(
            np.flip(G[:, 1:, :, :], 1).transpose(0, 3, 2, 1), (t, n, qn),
            order="F").transpose(0, 2, 1)  # backward autocov

        A_f = np.zeros((t, n, qn))  # forward coefficients
        A_b = np.zeros((t, n, qn))  # backward coefficients

        k = 1  # model order
        r = q - k
        k_f = np.arange(k * n)  # forward indices
        k_b = np.arange(r * n, qn)  # backward indices

        A_f[:, :, k_f] = np.linalg.solve(
            cov, G_b[:, k_b, :].transpose(0, 2, 1)).transpose(0, 2, 1)
        A_b[:, :, k_b] = np.linalg.solve(
            cov, G_f[:, k_f, :].transpose(0, 2, 1)).transpose(0, 2, 1)

        # Perform recursion
        for k in np.arange(2, q + 1):
            var_A = (G_b[:, (r - 1) * n: r * n, :] -
                     np.matmul(A_f[:, :, k_f], G_b[:, k_b, :]))
            var_B = cov - np.matmul(A_b[:, :, k_b], G_b[:, k_b, :])
            AA_f = np.linalg.solve(
                var_B, var_A.transpose(0, 2, 1)).transpose(0, 2, 1)

            var_A = (G_f[:, (k - 1) * n: k * n, :] -
                     np.matmul(A_b[:, :, k_b], G_f[:, k_f, :]))
            var_B = cov - np.matmul(A_f[:, :, k_f], G_f[:, k_f, :])
            AA_b = np.linalg.solve(
                var_B, var_A.transpose(0, 2, 1)).transpose(0, 2, 1)

            A_f_previous = A_f[:, :, k_f]
            A_b_previous = A_b[:, :, k_b]

            r = q - k
            k_f = np.arange(k * n)
            k_b = np.arange(r * n, qn)

            A_f[:, :, k_f] = np.dstack(
                (A_f_previous - np.matmul(AA_f, A_b_previous), AA_f))
            A_b[:, :, k_b] = np.dstack(
                (AA_b, A_b_previous - np.matmul(AA_b, A_f_previous)))

        V = cov - np.matmul(A_f, G_f)
        A_f = np.reshape(A_f, (t, n, n, q), order="F")

        return A_f, V

    def _full_var_to_iss(self, A_f):
        """Compute innovations-form parameters for a state-space model.

        Parameters computed from a full VAR model using Aoki's method. For a
        non-moving-average full VAR model, the state-space parameter C
        (observation matrix) is identical to AF of the VAR model.

        See: Barnett, L. & Seth, A.K., 2015, Physical Review, DOI:
        10.1103/PhysRevE.91.040101.
        """
        t = A_f.shape[0]
        m = A_f.shape[1]  # number of signals
        p = A_f.shape[2] // m  # number of autoregressive lags

        I_p = np.dstack(t * [np.eye(m * p)]).transpose(2, 0, 1)
        A = np.hstack((A_f, I_p[:, : (m * p - m), :]))  # state transition
        # matrix
        K = np.hstack((
            np.dstack(t * [np.eye(m)]).transpose(2, 0, 1),
            np.zeros((t, (m * (p - 1)), m))))  # Kalman gain matrix

        return A, K

    def _iss_to_ugc(self, A, C, K, V, seeds, targets):
        """Compute unconditional GC from innovations-form state-space params.

        See: Barnett, L. & Seth, A.K., 2015, Physical Review, DOI:
        10.1103/PhysRevE.91.040101.
        """
        times = np.arange(A.shape[0])
        freqs = np.arange(self.n_freqs)
        z = np.exp(-1j * np.pi * np.linspace(0, 1, self.n_freqs))  # points
        # on a unit circle in the complex plane, one for each frequency

        H = self._iss_to_tf(A, C, K, z)  # spectral transfer function
        V_22_1 = np.linalg.cholesky(self._partial_covar(V, seeds, targets))
        HV = np.matmul(H, np.linalg.cholesky(V))
        S = np.matmul(HV, HV.conj().transpose(0, 1, 3, 2))  # Eq. 6
        S_11 = S[np.ix_(freqs, times, targets, targets)]
        HV_12 = np.matmul(H[np.ix_(freqs, times, targets, seeds)], V_22_1)
        HVH = np.matmul(HV_12, HV_12.conj().transpose(0, 1, 3, 2))

        # Eq. 11
        return np.real(
            np.log(np.linalg.det(S_11)) - np.log(np.linalg.det(S_11 - HVH)))

    def _iss_to_tf(self, A, C, K, z):
        """Compute transfer function for innovations-form state-space params.

        In the frequency domain, the back-shift operator, z, is a vector of
        points on a unit circle in the complex plane. z = e^-iw, where -pi < w
        <= pi.

        A note on efficiency: solving over the 4D time-freq. tensor is slower
        than looping over times and freqs when n_times and n_freqs high, and
        when n_times and n_freqs low, looping over times and freqs very fast
        anyway (plus tensor solving doesn't allow for parallelisation).

        See: Barnett, L. & Seth, A.K., 2015, Physical Review, DOI:
        10.1103/PhysRevE.91.040101.
        """
        t = A.shape[0]
        h = self.n_freqs
        n = C.shape[1]
        m = A.shape[1]
        I_n = np.eye(n)
        I_m = np.eye(m)
        H = np.zeros((h, t, n, n), dtype=np.complex128)

        parallel, parallel_compute_H, _ = parallel_func(
            _gc_compute_H, self.n_jobs, verbose=False
        )
        H = np.zeros((h, t, n, n), dtype=np.complex128)
        for block_i in ProgressBar(
            range(self.n_steps), mesg="frequency blocks"
        ):
            freqs = self._get_block_indices(block_i, self.n_freqs)
            H[freqs] = parallel(
                parallel_compute_H(A, C, K, z[k], I_n, I_m) for k in freqs)

        return H

    def _partial_covar(self, V, seeds, targets):
        """Compute partial covariance of a matrix.

        Given a covariance matrix V, the partial covariance matrix of V between
        indices i and j, given k (V_ij|k), is equivalent to V_ij - V_ik *
        V_kk^-1 * V_kj. In this case, i and j are seeds, and k are targets.

        See: Barnett, L. & Seth, A.K., 2015, Physical Review, DOI:
        10.1103/PhysRevE.91.040101.
        """
        times = np.arange(V.shape[0])
        W = np.linalg.solve(
            np.linalg.cholesky(V[np.ix_(times, targets, targets)]),
            V[np.ix_(times, targets, seeds)],
        )
        W = np.matmul(W.transpose(0, 2, 1), W)

        return V[np.ix_(times, seeds, seeds)] - W

    def reshape_results(self):
        """Remove time dimension from con. scores, if necessary."""
        if self.n_times == 0:
            self.con_scores = self.con_scores[:, :, 0]


def _gc_compute_H(A, C, K, z_k, I_n, I_m):
    """Compute transfer function for innovations-form state-space params.

    See: Barnett, L. & Seth, A.K., 2015, Physical Review, DOI:
    10.1103/PhysRevE.91.040101, Eq. 4.
    """
    from scipy import linalg  # is this necessary???
    H = np.zeros((A.shape[0], C.shape[1], C.shape[1]), dtype=np.complex128)
    for t in range(A.shape[0]):
        H[t] = I_n + np.matmul(
            C[t], linalg.lu_solve(linalg.lu_factor(z_k * I_m - A[t]), K[t]))

    return H


class _GCEst(_GCEstBase):
    """[seeds -> targets] state-space GC estimator."""

    name = "GC"


class _TRGCEst(_GCEstBase):
    """time-reversed[seeds -> targets] state-space GC estimator."""

    name = "GC time-reversed"

###############################################################################


_multivariate_methods = ['mic', 'mim', 'gc', 'gc_tr']
_gc_methods = ['gc', 'gc_tr']


def _epoch_spectral_connectivity(data, sig_idx, tmin_idx, tmax_idx, sfreq,
                                 method, mode, window_fun, eigvals, wavelets,
                                 freq_mask, mt_adaptive, idx_map, block_size,
                                 psd, accumulate_psd, con_method_types,
                                 con_methods, n_signals, n_signals_use,
                                 n_times, accumulate_inplace=True):
    """Estimate connectivity for one epoch (see spectral_connectivity)."""
    if any(this_method in _multivariate_methods for this_method in method):
        n_cons = 1  # UNTIL NEW INDICES FORMAT
        n_con_signals = n_signals_use ** 2
    else:
        n_cons = len(idx_map[0])
        n_con_signals = n_cons

    if wavelets is not None:
        n_times_spectrum = n_times
        n_freqs = len(wavelets)
    else:
        n_times_spectrum = 0
        n_freqs = np.sum(freq_mask)

    if not accumulate_inplace:
        # instantiate methods only for this epoch (used in parallel mode)
        con_methods = [mtype(n_cons, n_freqs, n_times_spectrum)
                       for mtype in con_method_types]

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


# map names to estimator types
_CON_METHOD_MAP = {'coh': _CohEst, 'cohy': _CohyEst, 'imcoh': _ImCohEst,
                   'plv': _PLVEst, 'ciplv': _ciPLVEst, 'ppc': _PPCEst,
                   'pli': _PLIEst, 'pli2_unbiased': _PLIUnbiasedEst,
                   'dpli': _DPLIEst, 'wpli': _WPLIEst,
                   'wpli2_debiased': _WPLIDebiasedEst, 'mic': _MICEst,
                   'mim': _MIMEst, 'gc': _GCEst, 'gc_tr': _TRGCEst}


def _check_estimators(method):
    """Check construction of connectivity estimators."""
    n_methods = len(method)
    con_method_types = list()
    for this_method in method:
        if this_method in _CON_METHOD_MAP:
            con_method_types.append(_CON_METHOD_MAP[this_method])
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

    return con_method_types, n_methods, accumulate_psd


@ verbose
@ fill_doc
def spectral_connectivity_epochs(data, names=None, method='coh', indices=None,
                                 sfreq=None,
                                 mode='multitaper', fmin=None, fmax=np.inf,
                                 fskip=0, faverage=False, tmin=None, tmax=None,
                                 mt_bandwidth=None, mt_adaptive=False,
                                 mt_low_bias=True, cwt_freqs=None,
                                 cwt_n_cycles=7, gc_n_lags=40, rank=None,
                                 block_size=1000, n_jobs=1, verbose=None):
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
        'imcoh', 'mic', 'mim', 'plv', 'ciplv', 'ppc', 'pli', 'dpli', 'wpli',
        'wpli2_debiased', 'gc', 'gc_tr']``.
    indices : tuple of array | None
        Two arrays with indices of connections for which to compute
        connectivity. If None, all connections are computed.
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
        the data is computed using :func:`mne.compute_rank` and projected to.
        Only used if ``method`` contains any of ``['mic', 'mim', 'gc',
        'gc_tr']``.
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
        The shape of each connectivity dataset is either
        (n_signals ** 2, n_freqs) mode: 'multitaper' or 'fourier'
        (n_signals ** 2, n_freqs, n_times) mode: 'cwt_morlet'
        when "indices" is None, or
        (n_con, n_freqs) mode: 'multitaper' or 'fourier'
        (n_con, n_freqs, n_times) mode: 'cwt_morlet'
        when "indices" is specified and "n_con = len(indices[0])".

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
    connections corresponding to the lower-triangular part of the
    connectivity matrix). If one is only interested in the connectivity
    between some signals, the "indices" parameter can be used. For example,
    to compute the connectivity between the signal with index 0 and signals
    "2, 3, 4" (a total of 3 connections) one can use the following::

        indices = (np.array([0, 0, 0]),    # row indices
                   np.array([2, 3, 4]))    # col indices

        con = spectral_connectivity_epochs(data, method='coh',
                                           indices=indices, ...)

    In this case con.get_data().shape = (3, n_freqs). The connectivity scores
    are in the same order as defined indices.

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

        'mic' : Maximised imaginary part of coherency (MIC)
        :footcite:`EwaldEtAl2012` given by:

            :math:`MIC=\Large{\frac{\boldsymbol{\alpha}^T \boldsymbol{E \beta}}
            {\parallel\boldsymbol{\alpha}\parallel \parallel\boldsymbol{\beta}
            \parallel}}`

            where: :math:`\boldsymbol{E}` is the imaginary part of the
            transformed cross-spectral density between seeds and targets; and
            :math:`\boldsymbol{\alpha}` and :math:`\boldsymbol{\beta}` are
            eigenvectors for the seeds and targets, such that
            :math:`\boldsymbol{\alpha}^T \boldsymbol{E \beta}` maximises
            connectivity between the seeds and targets.

        'mim' : Multivariate interaction measure (MIM)
        :footcite:`EwaldEtAl2012` given by:

            :math:`MIM=tr(\boldsymbol{EE}^T)`

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

        'gc' : State-space Granger causality (GC) :footcite:`BarnettSeth2015`
        given by:

            :math:`GC = ln\Large{(\frac{\lvert\boldsymbol{S}_{tt}\rvert}{\lvert
            \boldsymbol{S}_{tt}-\boldsymbol{H}_{ts}\boldsymbol{\Sigma}_{ss
            \lvert t}\boldsymbol{H}_{ts}^*\rvert}})`,

            where: :math:`s` and :math:`t` represent the seeds and targets,
            respectively; :math:`\boldsymbol{H}` is the spectral transfer
            function; :math:`\boldsymbol{\Sigma}` is the residuals matrix of
            the autoregressive model; and :math:`\boldsymbol{S}` is
            :math:`\boldsymbol{\Sigma}` transformed by :math:`\boldsymbol{H}`.

        'gc_tr' : State-space GC on time-reversed signals
        :footcite:`BarnettSeth2015,WinklerEtAl2016` given by the same equation
        as for 'gc', but where the autocovariance sequence from which the
        autoregressive model is produced is transposed to mimic the reversal of
        the original signal in time.

    References
    ----------
    .. footbibliography::
    """
    if n_jobs != 1:
        parallel, my_epoch_spectral_connectivity, _ = parallel_func(
            _epoch_spectral_connectivity, n_jobs, verbose=verbose)

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

    if n_bands != 1 and any(
        this_method in _gc_methods for this_method in method
    ):
        raise ValueError('computing Granger causality on multiple frequency '
                         'bands is not yet supported')

    if any(this_method in _multivariate_methods for this_method in method):
        if not all(this_method in _multivariate_methods for
                   this_method in method):
            raise ValueError(
                'bivariate and multivariate connectivity methods cannot be '
                'used in the same function call')
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
        if sfreq is None:
            raise ValueError('Sampling frequency (sfreq) is required with '
                             'array input.')

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
                indices=indices, method=method, mode=mode, fskip=fskip,
                n_bands=n_bands, cwt_freqs=cwt_freqs, faverage=faverage)

            # check rank input and compute data ranks if necessary
            if multivariate_con:
                rank = _check_rank_input(rank, data, sfreq, indices_use)
            else:
                rank = None
                gc_n_lags = None

            # get the window function, wavelets, etc for different modes
            (spectral_params, mt_adaptive, n_times_spectrum,
             n_tapers) = _assemble_spectral_params(
                mode=mode, n_times=n_times, mt_adaptive=mt_adaptive,
                mt_bandwidth=mt_bandwidth, sfreq=sfreq,
                mt_low_bias=mt_low_bias, cwt_n_cycles=cwt_n_cycles,
                cwt_freqs=cwt_freqs, freqs=freqs, freq_mask=freq_mask)

            # unique signals for which we actually need to compute PSD etc.
            sig_idx = np.unique(np.r_[indices_use[0], indices_use[1]])
            n_signals_use = len(sig_idx)

            # map indices to unique indices
            idx_map = [np.searchsorted(sig_idx, ind) for ind in indices_use]
            if multivariate_con:
                indices_use = idx_map
                idx_map = [*idx_map[0], *idx_map[1]]
                idx_map = [np.repeat(idx_map, len(sig_idx)),
                           np.tile(idx_map, len(sig_idx))]

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
                method_params = dict(n_cons=n_cons, n_freqs=n_freqs,
                                     n_times=n_times_spectrum)
                if method[mtype_i] in _multivariate_methods:
                    method_params.update(dict(n_signals=n_signals_use))
                    if method[mtype_i] in _gc_methods:
                        method_params.update(dict(n_lags=gc_n_lags))
                con_methods.append(mtype(**method_params))

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
            sig_idx=sig_idx, tmin_idx=tmin_idx, tmax_idx=tmax_idx, sfreq=sfreq,
            method=method, mode=mode, freq_mask=freq_mask, idx_map=idx_map,
            block_size=block_size,
            psd=psd, accumulate_psd=accumulate_psd,
            mt_adaptive=mt_adaptive,
            con_method_types=con_method_types,
            con_methods=con_methods if n_jobs == 1 else None,
            n_signals=n_signals, n_signals_use=n_signals_use, n_times=n_times,
            accumulate_inplace=True if n_jobs == 1 else False)
        call_params.update(**spectral_params)

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
            raise ValueError(
                'first dimension of connectivity scores does not match the '
                'number of connections; please contact the mne-connectivity '
                'developers')
        if faverage:
            if this_con.shape[1] != n_freqs:
                raise ValueError(
                    'second dimension of connectivity scores does not match '
                    'the number of frequencies; please contact the '
                    'mne-connectivity developers')
            con_shape = (n_cons, n_bands) + this_con.shape[2:]
            this_con_bands = np.empty(con_shape, dtype=this_con.dtype)
            for band_idx in range(n_bands):
                this_con_bands[:, band_idx] = np.mean(
                    this_con[:, freq_idx_bands[band_idx]], axis=1)
            this_con = this_con_bands

            if this_patterns is not None:
                this_patterns_bands = np.empty((2, n_cons), dtype=object)
                for group_i, group_patterns in enumerate(this_patterns):
                    for con_i, con_patterns in enumerate(group_patterns):
                        this_patterns_bands[group_i][con_i] = np.empty(
                            (con_patterns.shape[0], n_bands))
                        for band_i in range(n_bands):
                            this_patterns_bands[group_i][con_i][:, band_i] = (
                                np.mean(
                                    con_patterns[:, freq_idx_bands[band_i]],
                                    axis=1))

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

    if indices is None and not any(this_method in _multivariate_methods for
                                   this_method in method):
        # return all-to-all connectivity matrices
        # raveled into a 1D array
        logger.info('    assembling connectivity matrix')
        con_flat = con
        con = list()
        for this_con_flat in con_flat:
            this_con = np.zeros((n_signals, n_signals) +
                                this_con_flat.shape[1:],
                                dtype=this_con_flat.dtype)
            this_con[indices_use] = this_con_flat

            # ravel 2D connectivity into a 1D array
            # while keeping other dimensions
            this_con = this_con.reshape((n_signals ** 2,) +
                                        this_con_flat.shape[1:])
            con.append(this_con)
    # number of nodes in the original data,
    n_nodes = n_signals

    if multivariate_con:
        # UNTIL THIS INDICES FORMAT SUPPORTED BY DEFAULT
        indices = tuple(
            [[np.array(indices_use[0])], [np.array(indices_use[1])]])

    # create a list of connectivity containers
    conn_list = []
    for _con, _patterns, _method in zip(con, patterns, method):
        kwargs = dict(data=_con,
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
                      n_lags=gc_n_lags)
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


def _check_rank_input(rank, data, sfreq, indices):
    """Check the rank argument is appropriate and compute rank if missing."""
    # UNTIL NEW INDICES FORMAT
    indices = np.array([[indices[0]], [indices[1]]])

    if rank is None:

        rank = np.zeros((2, len(indices[0])), dtype=int)

        if isinstance(data, BaseEpochs):
            data_arr = data.get_data()
            info = data.info
            ch_types = data.get_channel_types()
        else:
            data_arr = data
            info = create_info([str(i) for i in range(data_arr.shape[1])],
                               sfreq)
            ch_types = ['eeg' for _ in range(data.shape[1])]
            # 'eeg' channel type used as 'misc' is not recognised as valid

        for group_i in range(2):
            for con_i, con_idcs in enumerate(indices[group_i]):
                con_info = create_info(
                    ch_names=[str(idx) for idx in con_idcs],
                    sfreq=info['sfreq'],
                    ch_types=[ch_types[idx] for idx in con_idcs])
                con_data = EpochsArray(
                    data_arr[:, con_idcs], con_info, verbose=False)

                rank[group_i][con_i] = sum(
                    compute_rank(con_data, tol=1e-10, tol_kind='relative',
                                 verbose=False).values())

        logger.info('Estimated data ranks:')
        con_i = 1
        for seed_rank, target_rank in zip(rank[0], rank[1]):
            logger.info('    connection %i - seeds (%i); targets (%i)'
                        % (con_i, seed_rank, target_rank, ))
            con_i += 1
            if seed_rank != target_rank:
                raise ValueError(
                    'currently, only seeds and targets of the same rank are '
                    'supported.')

        rank = tuple((np.array(rank[0]), np.array(rank[1])))

    else:
        for seed_idcs, target_idcs, seed_rank, target_rank in zip(
                indices[0], indices[1], rank[0], rank[1]):
            if not (0 < seed_rank <= len(seed_idcs) and
                    0 < target_rank <= len(target_idcs)):
                raise ValueError(
                    'ranks for seeds and targets must be > 0 and <= the '
                    'number of channels in the seeds and targets, '
                    'respectively, for each connection')
            if seed_rank != target_rank:
                raise ValueError(
                    'currently, only seeds and targets of the same rank are '
                    'supported.')

    return rank
