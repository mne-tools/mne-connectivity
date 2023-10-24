# Authors: Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#          Denis A. Engemann <denis.engemann@gmail.com>
#          Adam Li <adam2392@gmail.com>
#          Thomas S. Binns <t.s.binns@outlook.com>
#          Tien D. Nguyen <tien-dung.nguyen@charite.de>
#          Richard M. Köhler <koehler.richard@charite.de>
#
# License: BSD (3-clause)

import numpy as np
import scipy as sp
from mne.epochs import BaseEpochs
from mne.parallel import parallel_func
from mne.utils import ProgressBar, logger, verbose

from .epochs import (
    _AbstractConEstBase, _check_spectral_connectivity_epochs_settings,
    _check_spectral_connectivity_epochs_data, _get_n_epochs,
    _prepare_connectivity, _assemble_spectral_params,
    _compute_spectral_methods_epochs, _store_results)
from ..utils import fill_doc, check_multivariate_indices


def _check_indices(indices, method, n_signals):
    if indices is None:
        if any(this_method in _gc_methods for this_method in method):
            raise ValueError(
                'indices must be specified when computing Granger causality, '
                'as all-to-all connectivity is not supported')
        else:
            logger.info('using all indices for multivariate connectivity')
            indices_use = (np.arange(n_signals, dtype=int)[np.newaxis, :],
                           np.arange(n_signals, dtype=int)[np.newaxis, :])
    else:
        indices_use = check_multivariate_indices(indices)  # pad with -1
        if any(this_method in _gc_methods for this_method in method):
            for seed, target in zip(indices[0], indices[1]):
                intersection = np.intersect1d(seed, target)
                if np.any(intersection != -1):  # ignore padded entries
                    raise ValueError(
                        'seed and target indices must not intersect when '
                        'computing Granger causality')

    # number of connectivities to compute
    n_cons = len(indices_use[0])
    logger.info('    computing connectivity for %d connections' % n_cons)

    return n_cons, indices_use


def _check_rank_input(rank, data, indices):
    """Check the rank argument is appropriate and compute rank if missing."""
    sv_tol = 1e-10  # tolerance for non-zero singular val (rel. to largest)
    if rank is None:
        rank = np.zeros((2, len(indices[0])), dtype=int)

        if isinstance(data, BaseEpochs):
            data_arr = data.get_data()
        else:
            data_arr = data

        # XXX: Unpadding of arrays after already padding them is perhaps not so
        #      efficient. However, we need to remove the padded values to
        #      ensure only the correct channels are indexed, and having two
        #      versions of indices is a bit messy currently. A candidate for
        #      refactoring to simplify code.

        for group_i in range(2):  # seeds and targets
            for con_i, con_idcs in enumerate(indices[group_i]):
                con_idcs = con_idcs[con_idcs != -1]  # -1 is padded value
                s = np.linalg.svd(data_arr[:, con_idcs], compute_uv=False)
                rank[group_i][con_i] = np.min(
                    [np.count_nonzero(epoch >= epoch[0] * sv_tol)
                     for epoch in s])

        logger.info('Estimated data ranks:')
        con_i = 1
        for seed_rank, target_rank in zip(rank[0], rank[1]):
            logger.info('    connection %i - seeds (%i); targets (%i)'
                        % (con_i, seed_rank, target_rank, ))
            con_i += 1
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

    return rank


########################################################################
# Multivariate connectivity estimators

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

    def compute_con(self, indices, ranks, n_epochs=1):
        """Compute multivariate imag. part of coherency between signals."""
        assert self.name in ['MIC', 'MIM'], (
            'the class name is not recognised, please contact the '
            'mne-connectivity developers')

        csd = self.reshape_csd() / n_epochs
        n_times = csd.shape[0]
        times = np.arange(n_times)
        freqs = np.arange(self.n_freqs)

        if self.name == 'MIC':
            self.patterns = np.full(
                (2, self.n_cons, indices[0].shape[1], self.n_freqs, n_times),
                np.nan)

        con_i = 0
        for seed_idcs, target_idcs, seed_rank, target_rank in zip(
                indices[0], indices[1], ranks[0], ranks[1]):
            self._log_connection_number(con_i)

            seed_idcs = seed_idcs[seed_idcs != -1]
            target_idcs = target_idcs[target_idcs != -1]
            con_idcs = [*seed_idcs, *target_idcs]

            C = csd[np.ix_(times, freqs, con_idcs, con_idcs)]

            # Eqs. 32 & 33
            C_bar, U_bar_aa, U_bar_bb = self._csd_svd(
                C, seed_idcs, seed_rank, target_rank)

            # Eqs. 3 & 4
            E = self._compute_e(C_bar, n_seeds=U_bar_aa.shape[3])

            if self.name == 'MIC':
                self._compute_mic(E, C, seed_idcs, target_idcs, n_times,
                                  U_bar_aa, U_bar_bb, con_i)
            else:
                self._compute_mim(E, seed_idcs, target_idcs, con_i)

            con_i += 1

        self.reshape_results()

    def _csd_svd(self, csd, seed_idcs, seed_rank, target_rank):
        """Dimensionality reduction of CSD with SVD."""
        n_times = csd.shape[0]
        n_seeds = len(seed_idcs)
        n_targets = csd.shape[3] - n_seeds

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
            T[:, freqs] = np.array(parallel(parallel_compute_t(
                C_r[:, f], T[:, f], n_seeds) for f in freqs)
            ).transpose(1, 0, 2, 3)

        if not np.isreal(T).all() or not np.isfinite(T).all():
            raise RuntimeError(
                'the transformation matrix of the data must be real-valued '
                'and contain no NaN or infinity values; check that you are '
                'using full rank data or specify an appropriate rank for the '
                'seeds and targets that is less than or equal to their ranks')
        T = np.real(T)  # make T real if check passes

        # Eq. 4
        D = np.matmul(T, np.matmul(csd, T))

        # E as imag. part of D between seeds and targets
        return np.imag(D[..., :n_seeds, n_seeds:])

    def _compute_mic(self, E, C, seed_idcs, target_idcs, n_times, U_bar_aa,
                     U_bar_bb, con_i):
        """Compute MIC and the associated spatial patterns."""
        n_seeds = len(seed_idcs)
        n_targets = len(target_idcs)
        times = np.arange(n_times)
        freqs = np.arange(self.n_freqs)

        # Eigendecomp. to find spatial filters for seeds and targets
        w_seeds, V_seeds = np.linalg.eigh(
            np.matmul(E, E.transpose(0, 1, 3, 2)))
        w_targets, V_targets = np.linalg.eigh(
            np.matmul(E.transpose(0, 1, 3, 2), E))
        if (
            len(seed_idcs) == len(target_idcs) and
            np.all(np.sort(seed_idcs) == np.sort(target_idcs))
        ):
            # strange edge-case where the eigenvectors returned should be a set
            # of identity matrices with one rotated by 90 degrees, but are
            # instead identical (i.e. are not rotated versions of one another).
            # This leads to the case where the spatial filters are incorrectly
            # applied, resulting in connectivity estimates of ~0 when they
            # should be perfectly correlated ~1. Accordingly, we manually
            # create a set of rotated identity matrices to use as the filters.
            create_filter = False
            stop = False
            while not create_filter and not stop:
                for time_i in range(n_times):
                    for freq_i in range(self.n_freqs):
                        if np.all(V_seeds[time_i, freq_i] ==
                                  V_targets[time_i, freq_i]):
                            create_filter = True
                            break
                stop = True
            if create_filter:
                n_chans = E.shape[2]
                eye_4d = np.zeros_like(V_seeds)
                eye_4d[:, :, np.arange(n_chans), np.arange(n_chans)] = 1
                V_seeds = eye_4d
                V_targets = np.rot90(eye_4d, axes=(2, 3))

        # Spatial filters with largest eigval. for seeds and targets
        alpha = V_seeds[times[:, None], freqs, :, w_seeds.argmax(axis=2)]
        beta = V_targets[times[:, None], freqs, :, w_targets.argmax(axis=2)]

        # Eq. 46 (seed spatial patterns)
        self.patterns[0, con_i, :n_seeds] = (np.matmul(
            np.real(C[..., :n_seeds, :n_seeds]),
            np.matmul(U_bar_aa, np.expand_dims(alpha, axis=3))))[..., 0].T

        # Eq. 47 (target spatial patterns)
        self.patterns[1, con_i, :n_targets] = (np.matmul(
            np.real(C[..., n_seeds:, n_seeds:]),
            np.matmul(U_bar_bb, np.expand_dims(beta, axis=3))))[..., 0].T

        # Eq. 7
        self.con_scores[con_i] = (np.einsum(
            'ijk,ijk->ij', alpha, np.matmul(E, np.expand_dims(
                beta, axis=3))[..., 0]
        ) / np.linalg.norm(alpha, axis=2) * np.linalg.norm(beta, axis=2)).T

    def _compute_mim(self, E, seed_idcs, target_idcs, con_i):
        """Compute MIM (a.k.a. GIM if seeds == targets)."""
        # Eq. 14
        self.con_scores[con_i] = np.matmul(
            E, E.transpose(0, 1, 3, 2)).trace(axis1=2, axis2=3).T

        # Eq. 15
        if (
            len(seed_idcs) == len(target_idcs) and
            np.all(np.sort(seed_idcs) == np.sort(target_idcs))
        ):
            self.con_scores[con_i] *= 0.5

    def reshape_results(self):
        """Remove time dimension from results, if necessary."""
        if self.n_times == 0:
            self.con_scores = self.con_scores[..., 0]
            if self.patterns is not None:
                self.patterns = self.patterns[..., 0]


def _mic_mim_compute_t(C, T, n_seeds):
    """Compute T for a single frequency (used for MIC and MIM)."""
    for time_i in range(C.shape[0]):
        T[time_i, :n_seeds, :n_seeds] = sp.linalg.fractional_matrix_power(
            C[time_i, :n_seeds, :n_seeds], -0.5
        )
        T[time_i, n_seeds:, n_seeds:] = sp.linalg.fractional_matrix_power(
            C[time_i, n_seeds:, n_seeds:], -0.5
        )

    return T


class _MICEst(_MultivariateCohEstBase):
    """Multivariate imaginary part of coherency (MIC) estimator."""

    name = "MIC"


class _MIMEst(_MultivariateCohEstBase):
    """Multivariate interaction measure (MIM) estimator."""

    name = "MIM"


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

    def compute_con(self, indices, ranks, n_epochs=1):
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
                indices[0], indices[1], ranks[0], ranks[1]):
            self._log_connection_number(con_i)

            seed_idcs = seed_idcs[seed_idcs != -1]
            target_idcs = target_idcs[target_idcs != -1]
            con_idcs = [*seed_idcs, *target_idcs]

            C = csd[np.ix_(times, freqs, con_idcs, con_idcs)]

            C_bar = self._csd_svd(C, seed_idcs, seed_rank, target_rank)
            n_signals = seed_rank + target_rank
            con_seeds = np.arange(seed_rank)
            con_targets = np.arange(target_rank) + seed_rank

            autocov = self._compute_autocov(C_bar)
            if self.name == "GC time-reversed":
                autocov = autocov.transpose(0, 1, 3, 2)

            A_f, V = self._autocov_to_full_var(autocov)
            A_f_3d = np.reshape(
                A_f, (n_times, n_signals, n_signals * self.n_lags), order="F")
            A, K = self._full_var_to_iss(A_f_3d)

            self.con_scores[con_i] = self._iss_to_ugc(
                A, A_f_3d, K, V, con_seeds, con_targets)

            con_i += 1

        self.reshape_results()

    def _csd_svd(self, csd, seed_idcs, seed_rank, target_rank):
        """Dimensionality reduction of CSD with SVD on the covariance."""
        # sum over times and epochs to get cov. from CSD
        cov = csd.sum(axis=(0, 1))

        n_seeds = len(seed_idcs)
        n_targets = csd.shape[3] - n_seeds

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
            raise RuntimeError(
                'the autocovariance matrix is singular; check if your data is '
                'rank deficient and specify an appropriate rank argument <= '
                'the rank of the seeds and targets')

        A_f, V = self._whittle_lwr_recursion(autocov)

        if not np.isfinite(A_f).all():
            raise RuntimeError('at least one VAR model coefficient is '
                               'infinite or NaN; check the data you are using')

        try:
            np.linalg.cholesky(V)
        except np.linalg.LinAlgError as np_error:
            raise RuntimeError(
                'the covariance matrix of the residuals is not '
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

        try:
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
        except np.linalg.LinAlgError as np_error:
            raise RuntimeError(
                'the autocovariance matrix is singular; check if your data is '
                'rank deficient and specify an appropriate rank argument <= '
                'the rank of the seeds and targets') from np_error

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
    from scipy import linalg  # XXX: is this necessary???
    H = np.zeros((A.shape[0], C.shape[1], C.shape[1]), dtype=np.complex128)
    for t in range(A.shape[0]):
        H[t] = I_n + np.matmul(
            C[t], linalg.lu_solve(linalg.lu_factor(z_k * I_m - A[t]), K[t]))

    return H


class _GCEst(_GCEstBase):
    """[seeds -> targets] state-space GC estimator."""

    name = "GC"


class _GCTREst(_GCEstBase):
    """time-reversed[seeds -> targets] state-space GC estimator."""

    name = "GC time-reversed"

###############################################################################


# map names to estimator types
_CON_METHOD_MAP = {'mic': _MICEst, 'mim': _MIMEst, 'gc': _GCEst,
                   'gc_tr': _GCTREst}

_gc_methods = ['gc', 'gc_tr']


@ verbose
@ fill_doc
def spectral_connectivity_epochs_multivariate(
    data, names=None, method='mic', indices=None, sfreq=None,
    mode='multitaper', fmin=None, fmax=np.inf, fskip=0, faverage=False,
    tmin=None, tmax=None, mt_bandwidth=None, mt_adaptive=False,
    mt_low_bias=True, cwt_freqs=None, cwt_n_cycles=7, gc_n_lags=40, rank=None,
    block_size=1000, n_jobs=1, verbose=None
):
    r"""Compute multivariate (time-)frequency-domain connectivity measures.

    The connectivity method(s) are specified using the "method" parameter.
    All methods are based on estimates of the cross-spectral density (CSD).

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
        'gc_tr']``.
    indices : tuple of array | None
        Two arrays with indices of connections for which to compute
        connectivity. Each array for the seeds and targets should consist of
        nested arrays containing the channel indices for each multivariate
        connection. If ``None``, connections between all channels are computed,
        unless a Granger causality method is called, in which case an error is
        raised.
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
        Number of lags to use when computing Granger causality (the vector
        autoregressive model order). Higher values increase computational cost,
        but reduce the degree of spectral smoothing in the results. Must be <
        (n_freqs - 1) * 2. Only used if ``method`` contains any of ``['gc',
        'gc_tr']``.
    rank : tuple of array | None
        Two arrays with the rank to project the seed and target data to,
        respectively, using singular value decomposition. If None, the rank of
        the data is computed and projected to. Only used if ``method`` contains
        any of ``['mic', 'mim', 'gc', 'gc_tr']``.
    block_size : int
        How many CSD entries to compute at once (higher numbers are faster but
        require more memory).
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
        - ``n_cons = 1`` when ``indices=None``
        - ``n_cons = len(indices[0])`` when indices is supplied

    See Also
    --------
    mne_connectivity.spectral_connectivity_epochs
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

    By default, "indices" is None, and the connectivity between all signals is
    computed and a single connectivity spectrum will be returned (this is not
    possible if a Granger causality method is called). If one is only
    interested in the connectivity between some signals, the "indices"
    parameter can be used. Seed and target indices for each connection should
    be specified as nested array-likes. For example, to compute the
    connectivity between signals (0, 1) -> (2, 3) and (0, 1) -> (4, 5), indices
    should be specified as::

        indices = ([[0, 1], [0, 1]],  # seeds
                   [[2, 3], [4, 5]])  # targets

    More information on working with multivariate indices and handling
    connections where the number of seeds and targets are not equal can be
    found in the :doc:`../auto_examples/handling_ragged_arrays` example.

    **Supported Connectivity Measures**

    The connectivity method(s) is specified using the "method" parameter.
    Multiple measures can be computed at once by using a list/tuple, e.g.,
    ``['mic', 'gc']``. The following methods are supported:

        'mic' : Maximised Imaginary part of Coherency (MIC)
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

        'mim' : Multivariate Interaction Measure (MIM)
        :footcite:`EwaldEtAl2012` given by:

            :math:`MIM=tr(\boldsymbol{EE}^T)`

        'gc' : State-space Granger Causality (GC) :footcite:`BarnettSeth2015`
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
    (
        fmin, fmax, n_bands, method, con_method_types, accumulate_psd,
        parallel, my_epoch_spectral_connectivity
    ) = _check_spectral_connectivity_epochs_settings(
        method, fmin, fmax, n_jobs, verbose, _CON_METHOD_MAP)

    if n_bands != 1 and any(
        this_method in _gc_methods for this_method in method
    ):
        raise ValueError('computing Granger causality on multiple frequency '
                         'bands is not yet supported')

    (names, times_in, sfreq, events, event_id,
     metadata) = _check_spectral_connectivity_epochs_data(data, sfreq, names)

    # loop over data; it could be a generator that returns
    # (n_signals x n_times) arrays or SourceEstimates
    epoch_idx = 0
    logger.info('Connectivity computation...')
    warn_times = True
    for epoch_block in _get_n_epochs(data, n_jobs):
        if epoch_idx == 0:
            # initialize everything times and frequencies
            (times, n_times, times_in, n_times_in, tmin_idx, tmax_idx, n_freqs,
             freq_mask, freqs, freqs_bands, freq_idx_bands, n_signals,
             warn_times) = _prepare_connectivity(
                epoch_block=epoch_block, times_in=times_in, tmin=tmin,
                tmax=tmax, fmin=fmin, fmax=fmax, sfreq=sfreq, mode=mode,
                fskip=fskip, n_bands=n_bands, cwt_freqs=cwt_freqs,
                faverage=faverage)

            # check indices input
            n_cons, indices_use = _check_indices(indices, method, n_signals)

            # check rank input and compute data ranks
            rank = _check_rank_input(rank, data, indices_use)

            # make sure padded indices are stored in the connectivity object
            if indices is not None:
                indices = tuple(np.array(indices_use))  # create a copy

            # get the window function, wavelets, etc for different modes
            (spectral_params, mt_adaptive, n_times_spectrum,
             n_tapers) = _assemble_spectral_params(
                mode=mode, n_times=n_times, mt_adaptive=mt_adaptive,
                mt_bandwidth=mt_bandwidth, sfreq=sfreq,
                mt_low_bias=mt_low_bias, cwt_n_cycles=cwt_n_cycles,
                cwt_freqs=cwt_freqs, freqs=freqs, freq_mask=freq_mask)

            # unique signals for which we actually need to compute CSD
            sig_idx = np.unique(np.concatenate(np.concatenate(
                indices_use)))
            sig_idx = sig_idx[sig_idx != -1]
            remapping = {ch_i: sig_i for sig_i, ch_i in enumerate(sig_idx)}
            remapping[-1] = -1
            remapped_inds = (indices_use[0].copy(), indices_use[1].copy())
            con_i = 0
            for seed, target in zip(indices_use[0], indices_use[1]):
                remapped_inds[0][con_i] = np.array([
                    remapping[idx] for idx in seed])
                remapped_inds[1][con_i] = np.array([
                    remapping[idx] for idx in target])
                con_i += 1
            remapped_sig = [remapping[idx] for idx in sig_idx]
            n_signals_use = len(sig_idx)

            # map indices to unique indices
            indices_use = remapped_inds  # use remapped seeds & targets
            idx_map = [np.sort(np.repeat(remapped_sig, len(sig_idx))),
                       np.tile(remapped_sig, len(sig_idx))]

            # create instances of the connectivity estimators
            con_methods = []
            for mtype_i, mtype in enumerate(con_method_types):
                method_params = dict(n_cons=n_cons, n_freqs=n_freqs,
                                     n_times=n_times_spectrum,
                                     n_signals=n_signals_use)
                if method[mtype_i] in _gc_methods:
                    method_params.update(dict(n_lags=gc_n_lags))
                con_methods.append(mtype(**method_params))

            sep = ', '
            metrics_str = sep.join([meth.name for meth in con_methods])
            logger.info('    the following metrics will be computed: %s'
                        % metrics_str)

        call_params = dict(
            sig_idx=sig_idx, tmin_idx=tmin_idx, tmax_idx=tmax_idx, sfreq=sfreq,
            method=method, mode=mode, freq_mask=freq_mask, idx_map=idx_map,
            n_cons=n_cons, block_size=block_size,
            psd=None, accumulate_psd=accumulate_psd,
            mt_adaptive=mt_adaptive,
            con_method_types=con_method_types,
            con_methods=con_methods if n_jobs == 1 else None,
            n_signals=n_signals, n_signals_use=n_signals_use, n_times=n_times,
            gc_n_lags=gc_n_lags, multivariate_con=True,
            accumulate_inplace=True if n_jobs == 1 else False)
        call_params.update(**spectral_params)

        epoch_idx = _compute_spectral_methods_epochs(
            con_methods, epoch_block, epoch_idx, call_params, parallel,
            my_epoch_spectral_connectivity, n_jobs, n_times_in, times_in,
            warn_times)
    n_epochs = epoch_idx

    # compute final connectivity scores
    con = list()
    patterns = list()
    for conn_method in con_methods:

        # compute connectivity scores
        conn_method.compute_con(indices_use, rank, n_epochs)

        # get the connectivity scores
        this_con = conn_method.con_scores
        this_patterns = conn_method.patterns

        if this_con.shape[0] != n_cons:
            raise RuntimeError(
                'first dimension of connectivity scores does not match the '
                'number of connections; please contact the mne-connectivity '
                'developers')
        if faverage:
            if this_con.shape[1] != n_freqs:
                raise RuntimeError(
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
                patterns_shape = list(this_patterns.shape)
                patterns_shape[3] = n_bands
                this_patterns_bands = np.empty(patterns_shape,
                                               dtype=this_patterns.dtype)
                for band_idx in range(n_bands):
                    this_patterns_bands[:, :, :, band_idx] = np.mean(
                        this_patterns[:, :, :, freq_idx_bands[band_idx]],
                        axis=3)
                this_patterns = this_patterns_bands

        con.append(this_con)
        patterns.append(this_patterns)

    conn_list = _store_results(
        con=con, patterns=patterns, method=method, freqs=freqs,
        faverage=faverage, freqs_bands=freqs_bands, names=names, mode=mode,
        indices=indices, n_epochs=n_epochs, times=times, n_tapers=n_tapers,
        metadata=metadata, events=events, event_id=event_id, rank=rank,
        gc_n_lags=gc_n_lags, n_signals=n_signals)

    return conn_list
