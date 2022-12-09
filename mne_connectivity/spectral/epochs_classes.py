# Authors: Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#          Denis A. Engemann <denis.engemann@gmail.com>
#          Adam Li <adam2392@gmail.com>
#          Thomas Samuel Binns <thomas-samuel.binns@charite.de>
#
# License: BSD (3-clause)

import numpy as np
from scipy import linalg as spla
from mne.parallel import parallel_func


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


class _EpochMeanMultivarConEstBase(_AbstractConEstBase):
    """Base class for methods that estimate connectivity as mean epoch-wise."""

    def __init__(self, n_signals, n_cons, n_freqs, n_times, n_jobs):
        self.n_signals = n_signals
        self.n_cons = n_cons
        self.n_freqs = n_freqs
        self.n_times = n_times
        self.n_jobs = n_jobs

        if n_times == 0:
            self.csd_shape = (n_signals**2, n_freqs)
            self.con_scores = np.zeros((n_cons, n_freqs, 1))
        else:
            self.csd_shape = (n_signals**2, n_freqs, n_times)
            self.con_scores = np.zeros((n_cons, n_freqs, n_times))
        
        # allocate space for accumulation of CSD
        self._acc = np.zeros(self.csd_shape, dtype=np.complex128)

    def start_epoch(self):  # noqa: D401
        """Called at the start of each epoch."""
        pass  # for this type of con. method we don't do anything

    def combine(self, other):
        """Include con. accumated for some epochs in this estimate."""
        self._acc += other._acc

    def accumulate(self, con_idx, csd_xy):
        """Accumulate CSD for some connections."""
        self._acc[con_idx] += csd_xy

    def reshape_csd(self):
        """Reshapes CSD into a matrix of times x freqs x signals x signals."""
        if self.n_times == 0:
            return (
                np.reshape(self._acc, (self.n_signals, self.n_signals,
                self.n_freqs, 1)).transpose(3, 2, 0, 1)
            )
        return (
            np.reshape(self._acc, (self.n_signals, self.n_signals, self.n_freqs,
            self.n_times)).transpose(3, 2, 0, 1)
        )

    def reshape_con_scores(self):
        """Removes the time dimension from con. scores, if necessary."""
        if self.n_times == 0:
            self.con_scores = self.con_scores[:,:,0]


class _CohEstBase(_EpochMeanConEstBase):
    """Base Estimator for Coherence, Coherency, Imag. Coherence."""

    def __init__(self, n_cons, n_freqs, n_times):
        super(_CohEstBase, self).__init__(n_cons, n_freqs, n_times)

        # allocate space for accumulation of CSD
        self._acc = np.zeros(self.csd_shape, dtype=np.complex128)

    def accumulate(self, con_idx, csd_xy):
        """Accumulate CSD for some connections."""
        self._acc[con_idx] += csd_xy

class _MultivarCohEstBase(_EpochMeanMultivarConEstBase):
    """Base Estimator for coherence-based multivariate methods."""

    def cross_spectra_svd(
        self, csd, n_seeds, n_seed_components, n_target_components
    ):
        """Performs dimensionality reduction on a cross-spectral density using
        singular value decomposition (SVD)."""
        n_times = csd.shape[0]
        n_targets = csd.shape[2]-n_seeds
        C_aa = csd[:, :, :n_seeds, :n_seeds]
        C_ab = csd[:, :, :n_seeds, n_seeds:]
        C_bb = csd[:, :, n_seeds:, n_seeds:]
        C_ba = csd[:, :, n_seeds:, :n_seeds]

        # Eq. 32
        if n_seed_components is not None:
            U_aa = np.linalg.svd(np.real(C_aa), full_matrices=False)[0]
            U_bar_aa = U_aa[:, :, :, :n_seed_components]
        else:
            U_bar_aa = np.broadcast_to(np.identity(n_seeds), (n_times, self.n_freqs)+(n_seeds, n_seeds))
        if n_target_components is not None:
            U_bb = np.linalg.svd(np.real(C_bb), full_matrices=False)[0]
            U_bar_bb = U_bb[:, :, :, :n_target_components]
        else:
            U_bar_bb = np.broadcast_to(np.identity(n_targets), (n_times, self.n_freqs)+(n_targets, n_targets))

        # Eq. 33
        C_bar_aa = U_bar_aa.transpose(0, 1, 3, 2) @ (C_aa @ U_bar_aa)
        C_bar_ab = U_bar_aa.transpose(0, 1, 3, 2) @ (C_ab @ U_bar_bb)
        C_bar_bb = U_bar_bb.transpose(0, 1, 3, 2) @ (C_bb @ U_bar_bb)
        C_bar_ba = U_bar_bb.transpose(0, 1, 3, 2) @ (C_ba @ U_bar_aa)
        C_bar = np.append(
            np.append(C_bar_aa, C_bar_ab, axis=3), np.append(C_bar_ba, C_bar_bb, axis=3), axis=2
        )

        return C_bar, U_bar_aa, U_bar_bb

    def compute_e(self, csd, n_seeds):
        """Computes E as the imaginary part of the transformed cross-spectra D
        derived from the original cross-spectra "csd" between the seed and target
        signals."""
        # Equation 3
        n_freqs = csd.shape[1]
        real_csd = np.real(csd)
        if self.n_jobs > 1:
            parallel, parallel_compute_t, _ = parallel_func(
                _compute_t, self.n_jobs, verbose=False
            )
            T = np.array(parallel(
                parallel_compute_t(real_csd[:, freq_i, :, :], n_seeds)
                for freq_i in range(n_freqs)
            ))
        else:
            T = np.array([
                _compute_t(real_csd[:, freq_i, :, :], n_seeds)
                 for freq_i in range(n_freqs)
            ])
        T = T.transpose(1, 0, 2, 3)

        # Equation 4
        D = T @ (csd @ T)

        # E as the imaginary part of D between seeds and targets
        E = np.imag(D[:, :, :n_seeds, n_seeds:])

        return E

def _compute_t(csd, n_seeds):
    """Compute T as the real-valued cross-spectra of seeds and targets to the
    power -0.5."""
    T = np.zeros_like(csd)
    for time_i in range(csd.shape[0]):
        T[time_i, :n_seeds, :n_seeds] = spla.fractional_matrix_power(
            csd[time_i, :n_seeds, :n_seeds], -0.5
        ) # real(C_aa)^-1/2
        T[time_i, n_seeds:, n_seeds:] = spla.fractional_matrix_power(
            csd[time_i, n_seeds:, n_seeds:], -0.5
        ) # real(C_bb)^-1/2
    
    return T


class _MultivarGCEstBase(_EpochMeanMultivarConEstBase):
    """Base Estimator for Granger causality multivariate methods."""

    def __init__(self, n_signals, n_cons, n_freqs, n_times, n_lags, n_jobs=1):
        super(_MultivarGCEstBase, self).__init__(
            n_signals, n_cons, n_freqs, n_times, n_jobs
        )

        if n_lags:
            if n_lags >= (self.n_freqs - 1) * 2:
                raise ValueError(
                    f"The number of lags ({n_lags}) must be less than double "
                    "the frequency resolution of the cross-spectral density "
                    f"({(self.n_freqs - 1) * 2})."
                )
            self.n_lags = n_lags
        else:
            self.n_lags = self.n_freqs - 2 # freq. resolution - 1

    def compute_con(self, seeds, targets, n_epochs):
        """Computes Granger causality between sets of signals."""
        csd = self.reshape_csd()/n_epochs

        # GC from seeds -> targets (subtracting GC from targets -> seeds if
        # self.net == True)
        autocov = self.csd_to_autocov(csd)
        con_scores = self.autocov_to_gc(autocov, seeds, targets, self.net)

        # GC from seeds -> targets (subtracting GC from targets -> seeds if
        # self.net == True), subtracting GC from time-reversed seeds -> targets
        # (subtracting GC from time-reversed targets -> seeds if self.net ==
        # True)
        if self.time_reversed:
            con_scores -= self.autocov_to_gc(
                autocov.transpose(0, 1, 3, 2), seeds, targets, self.net
            )

        self.con_scores = con_scores
        self.reshape_con_scores()

    def csd_to_autocov(self, csd):
        """Computes the autocovariance sequence from the cross-spectral
        density."""
        n_times = csd.shape[0]
        circular_shifted_csd = np.concatenate(
            [np.flip(np.conj(csd[:, 1:, :, :]), axis=1),
            csd[:, :-1, :, :]],
            axis=1,
        )
        ifft_shifted_csd = self.block_ifft(
            circular_shifted_csd, (self.n_freqs - 1) * 2
        )
        lags_ifft_shifted_csd = np.reshape(
            ifft_shifted_csd[:, :self.n_lags + 1, :, :],
            (n_times, self.n_lags + 1, self.n_signals ** 2),
            order="F"
        )

        signs = [1] * (self.n_lags + 1)
        signs[1::2] = [x * -1 for x in signs[1::2]]
        sign_matrix = np.repeat(
            np.tile(
                np.asarray(signs), (self.n_signals ** 2, 1)
            )[np.newaxis, :, :],
            n_times,
            axis=0
        ).transpose(0, 2, 1)

        return (
            np.real(
                np.reshape(
                    sign_matrix * lags_ifft_shifted_csd,
                    (n_times, self.n_lags + 1, self.n_signals, self.n_signals),
                    order="F"
                )
            )
        )

    def block_ifft(self, csd, n_points):
        """Performs a 'block' inverse fast Fourier transform on the data,
        involving an n-point inverse Fourier transform."""
        csd_3d = np.reshape(
            csd,
            (csd.shape[0], csd.shape[1], csd.shape[2] * csd.shape[3]),
            order="F"
        )
        csd_ifft = np.fft.ifft(csd_3d, n=n_points, axis=1)

        return np.reshape(csd_ifft, csd.shape, order="F")

    def autocov_to_gc(self, autocov, seeds, targets, net):
        """Computes frequency-domain multivariate Granger causality from an
        autocovariance sequence."""
        n_times = autocov.shape[0]
        lags = np.arange(autocov.shape[1])
        times = np.arange(n_times)
        con_scores = np.zeros((len(seeds), self.n_freqs, n_times))

        con_i = 0
        for con_seeds, con_targets in zip(seeds, targets):
            all_idcs = [*con_seeds, *con_targets]
            con_autocov = autocov[np.ix_(times, lags, all_idcs, all_idcs)]
            new_seeds = np.arange(len(con_seeds))
            new_targets = np.arange(len(con_targets))+len(con_seeds)

            A_f, V = self.autocov_to_full_var(con_autocov)
            A_f_3d = np.reshape(
                A_f,
                (n_times, self.n_signals, self.n_signals * self.n_lags),
                order="F"
            )
            A, K = self.full_var_to_iss(A_f_3d)

            con_scores[con_i] = self.iss_to_usgc(
                A=A,
                C=A_f_3d,
                K=K,
                V=V,
                seeds=new_seeds,
                targets=new_targets,
            ) # GC from seeds -> targets

            if net:
                con_scores[con_i] -= self.iss_to_usgc(
                    A=A,
                    C=A_f_3d,
                    K=K,
                    V=V,
                    seeds=new_targets,
                    targets=new_seeds,
                ) # GC from targets -> seeds

            con_i += 1
        
        return con_scores

    def autocov_to_full_var(self, autocov):
        """Computes the full vector autoregressive (VAR) model from an
        autocovariance sequence using Whittle's recursion.

        Ref.: Whittle P., 1963. Biometrika, DOI: 10.1093/biomet/50.1-2.129.
        """
        A_f, V = self.whittle_lwr_recursion(autocov)

        if not np.isfinite(A_f).all():
            raise ValueError(
                "Some or all VAR model coefficients are infinite or NaNs. "
                "Please check the data you are computing Granger causality on."
            )

        try:
            np.linalg.cholesky(V)
        except np.linalg.linalg.LinAlgError as np_error:
            raise ValueError(
                "The residuals' covariance matrix is not positive-definite. "
                "Make sure you are computing Granger causality only on data "
                "that is full rank."
            ) from np_error

        return A_f, V

    def whittle_lwr_recursion(self, G):
        """Calculates regression coefficients and the residuals' covariance
        matrix from an autocovariance sequence by solving the Yule-Walker
        equations using Whittle's recursive Levinson, Wiggins, Robinson (LWR)
        algorithm.

        Ref.: Whittle P., 1963. Biometrika, DOI: 10.1093/biomet/50.1-2.129.
        """
        ### Initialise recursion
        n = G.shape[2]  # number of signals
        q = G.shape[1] - 1  # number of lags
        t = G.shape[0]  # number of times
        qn = n * q

        cov = G[:, 0, :, :]  # covariance
        G_f = np.reshape(
            G[:, 1:, :, :].transpose(0, 3, 1, 2), (t, qn, n), order="F"
        )  # forward autocovariance sequence
        G_b = np.reshape(
            np.flip(G[:, 1:, :, :], 1).transpose(0, 3, 2, 1), (t, n, qn),
            order="F"
        ).transpose(0, 2, 1)  # backward autocovariance sequence
            
        A_f = np.zeros((t, n, qn))  # forward coefficients
        A_b = np.zeros((t, n, qn))  # backward coefficients

        k = 1  # model order
        r = q - k
        k_f = np.arange(k * n)  # forward indices
        k_b = np.arange(r * n, qn)  # backward indices

        A_f[:, :, k_f] = np.linalg.solve(
            cov, G_b[:, k_b, :].transpose(0, 2, 1)
        ).transpose(0, 2, 1) 
        A_b[:, :, k_b] = np.linalg.solve(
            cov, G_f[:, k_f, :].transpose(0, 2, 1)
        ).transpose(0, 2, 1)

        ## Perform recursion
        for k in np.arange(2, q + 1):
            var_A = (
                G_b[:, (r - 1) * n : r * n, :] -
                (A_f[:, :, k_f] @ G_b[:, k_b, :])
            )
            var_B = cov - (A_b[:, :, k_b] @ G_b[:, k_b, :])
            AA_f = np.linalg.solve(
                var_B, var_A.transpose(0, 2, 1)
            ).transpose(0, 2, 1)
            var_A = (
                G_f[:, (k - 1) * n : k * n, :] -
                (A_b[:, :, k_b] @ G_f[:, k_f, :])
            )
            var_B = cov - (A_f[:, :, k_f] @ G_f[:, k_f, :])
            AA_b = np.linalg.solve(
                var_B, var_A.transpose(0, 2, 1)
            ).transpose(0, 2, 1)

            A_f_previous = A_f[:, :, k_f]
            A_b_previous = A_b[:, :, k_b]

            r = q - k
            k_f = np.arange(k * n)
            k_b = np.arange(r * n, qn)

            A_f[:, :, k_f] = np.dstack(
                (A_f_previous - (AA_f @ A_b_previous), AA_f)
            )
            A_b[:, :, k_b] = np.dstack(
                (AA_b, A_b_previous - (AA_b @ A_f_previous))
            )

        V = cov - (A_f @ G_f)
        A_f = np.reshape(A_f, (t, n, n, q), order="F")

        return A_f, V


    def full_var_to_iss(self, A_f):
        """Computes innovations-form parameters for a state-space model from a
        full vector autoregressive (VAR) model using Aoki's method.

        For a non-moving-average full VAR model, the state-space parameter C
        (observation matrix) is identical to AF of the VAR model.

        Ref.: Barnett, L. & Seth, A.K., 2015, Physical Review, DOI:
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
            np.zeros((t, (m * (p - 1)), m))
        ))  # Kalman gain matrix

        return A, K

    def iss_to_usgc(self, A, C, K, V, seeds, targets):
        """Computes unconditional spectral Granger causality from
        innovations-form state-space model parameters.

        Ref.: Barnett, L. & Seth, A.K., 2015, Physical Review, DOI:
        10.1103/PhysRevE.91.040101.
        """
        times = np.arange(A.shape[0])
        freqs = np.arange(self.n_freqs)
        z = np.exp(-1j * np.pi * np.linspace(0, 0.99, self.n_freqs))  # points
        # on a unit circle in the complex plane, one for each frequency

        H = self.iss_to_tf(A, C, K, z)  # spectral transfer function
        V_22_1 = np.linalg.cholesky(self.partial_covar(V, seeds, targets))
        HV = H @ np.linalg.cholesky(V)
        S = HV @ HV.conj().transpose(0, 1, 3, 2)  # Eq. 6
        S_11 = S[np.ix_(freqs, times, targets, targets)]
        HV_12 = H[np.ix_(freqs, times, targets, seeds)] @ V_22_1
        HVH = HV_12 @ HV_12.conj().transpose(0, 1, 3, 2)
        
        return np.real(
            np.log(np.linalg.det(S_11)) -
            np.log(np.linalg.det(S_11 - HVH))
        )  # Eq. 11

    def iss_to_tf(self, A, C, K, z):
        """Computes a transfer function (moving-average representation) for
        innovations-form state-space model parameters.

        In the frequency domain, the back-shift operator, z, is a vector of
        points on a unit circle in the complex plane. z = e^-iw, where -pi < w
        <= pi.

        Ref.: Barnett, L. & Seth, A.K., 2015, Physical Review, DOI:
        10.1103/PhysRevE.91.040101.
        """
        n_times = A.shape[0]
        h = self.n_freqs
        n = C.shape[1]
        m = A.shape[1]
        I_n = np.eye(n)
        I_m = np.eye(m)
        H = np.zeros((h, n_times, n, n), dtype=np.complex128)

        for time_i in range(n_times):
            for k in range(h): # compute transfer function; Eq. 4
                H[k, time_i, :, :] = I_n + (
                    C[time_i]
                    @ spla.lu_solve(
                        spla.lu_factor(z[k] * I_m - A[time_i]), K[time_i]
                    )
                )

        return H

    def partial_covar(self, V, seeds, targets):
        """Computes the partial covariance of a matrix.

        Given a covariance matrix V, the partial covariance matrix of V between
        indices i and j, given k (V_ij|k), is equivalent to
        V_ij - V_ik * V_kk^-1 * V_kj. In this case, i and j are seeds, and k is
        the targets.

        Ref.: Barnett, L. & Seth, A.K., 2015, Physical Review, DOI:
        10.1103/PhysRevE.91.040101.
        """
        times = np.arange(V.shape[0])
        W = np.linalg.solve(
            np.linalg.cholesky(V[np.ix_(times, targets, targets)]),
            V[np.ix_(times, targets, seeds)],
        )
        W = W.transpose(0, 2, 1) @ W

        return V[np.ix_(times, seeds, seeds)] - W


class _CohEst(_CohEstBase):
    """Coherence Estimator."""

    name = 'Coherence'
    accumulate_psd = True

    def compute_con(self, con_idx, n_epochs, psd_xx, psd_yy):  # lgtm
        """Compute final con. score for some connections."""
        if self.con_scores is None:
            self.con_scores = np.zeros(self.csd_shape)
        csd_mean = self._acc[con_idx] / n_epochs
        self.con_scores[con_idx] = np.abs(csd_mean) / np.sqrt(psd_xx * psd_yy)


class _CohyEst(_CohEstBase):
    """Coherency Estimator."""

    name = 'Coherency'
    accumulate_psd = True

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
    accumulate_psd = True

    def compute_con(self, con_idx, n_epochs, psd_xx, psd_yy):  # lgtm
        """Compute final con. score for some connections."""
        if self.con_scores is None:
            self.con_scores = np.zeros(self.csd_shape)
        csd_mean = self._acc[con_idx] / n_epochs
        self.con_scores[con_idx] = np.imag(csd_mean) / np.sqrt(psd_xx * psd_yy)

class _MIMEst(_MultivarCohEstBase):
    """Estimator for MIM (multivariate interaction measure)"""

    name = "MIM"
    accumulate_psd = False

    def compute_con(
        self, seeds, targets, n_seed_components, n_target_components, n_epochs
    ):
        """Computes the multivariate interaction measure between two sets of
        signals"""
        csd = self.reshape_csd()/n_epochs
        n_times = csd.shape[0]
        node_i = 0
        for seed_idcs, target_idcs in zip(seeds, targets):
            n_seeds = len(seed_idcs)
            node_idcs = [*seed_idcs, *target_idcs]
            node_csd = csd[np.ix_(np.arange(n_times), np.arange(self.n_freqs), node_idcs, node_idcs)]

            # Eqs. 32 & 33
            C_bar, U_bar_aa, _ = self.cross_spectra_svd(
                csd=node_csd,
                n_seeds=n_seeds,
                n_seed_components=n_seed_components[node_i],
                n_target_components=n_target_components[node_i],
            )

            # Eqs. 3 & 4
            E = self.compute_e(csd=C_bar, n_seeds=U_bar_aa.shape[2])

            # Equation 14
            self.con_scores[node_i, :, :] = (E @ E.transpose(0, 1, 3, 2)).trace(axis1=2, axis2=3).transpose(1, 0)
            node_i += 1
        self.reshape_con_scores()


class _MICEst(_MultivarCohEstBase):
    """Estimator for MIC (maximized imaginary coherence)"""

    name = "MIC"
    accumulate_psd = False

    def compute_con(
        self, seeds, targets, n_seed_components, n_target_components, n_epochs
    ):
        """Computes maximized imaginary coherence between sets of signals."""
        csd = self.reshape_csd()/n_epochs
        n_times = csd.shape[0]
        node_i = 0
        for seed_idcs, target_idcs in zip(seeds, targets):
            n_seeds = len(seed_idcs)
            node_idcs = [*seed_idcs, *target_idcs]
            node_csd = csd[np.ix_(np.arange(n_times), np.arange(self.n_freqs), node_idcs, node_idcs)]

            # Eqs. 32 & 33
            C_bar, U_bar_aa, _ = self.cross_spectra_svd(
                csd=node_csd,
                n_seeds=n_seeds,
                n_seed_components=n_seed_components[node_i],
                n_target_components=n_target_components[node_i],
            )

            # Eqs. 3 & 4
            E = self.compute_e(csd=C_bar, n_seeds=U_bar_aa.shape[2])

            # Weights for signals in the groups
            w_a, V_a = np.linalg.eigh(E @ E.transpose(0, 1, 3, 2))
            w_b, V_b = np.linalg.eigh(E.transpose(0, 1, 3, 2) @ E)
            alpha = V_a[np.arange(n_times)[:, None], np.arange(self.n_freqs), :, w_a.argmax(axis=2)]
            beta = V_b[np.arange(n_times)[:, None], np.arange(self.n_freqs), :, w_b.argmax(axis=2)]

            # Eq. 7
            self.con_scores[node_i, :, :] = (np.einsum(
                "ijk,ijk->ij",
                alpha,
                (E @ np.expand_dims(beta, 3))[:, :, :, 0]) / np.linalg.norm(alpha, axis=2) * np.linalg.norm(beta, axis=2)
            ).transpose(1, 0)
            node_i += 1
        self.reshape_con_scores()


class _GCEst(_MultivarGCEstBase):
    """GC Estimator; causality from: [seeds -> targets]."""

    name = "GC"
    accumulate_psd = False
    net = False
    time_reversed = False


class _NetGCEst(_MultivarGCEstBase):
    """Net GC Estimator; causality from: [seeds -> targets] - [targets ->
    seeds]."""

    name = "Net GC"
    accumulate_psd = False
    net = True
    time_reversed = False

class _TRGCEst(_MultivarGCEstBase):
    """TRGC Estimator; causality from: [seeds -> targets] - time-reversed[seeds
    -> targets]."""

    name = "TRGC"
    accumulate_psd = False
    net = False
    time_reversed = True

class _NetTRGCEst(_MultivarGCEstBase):
    """Net TRGC Estimator; causality from: ([seeds -> targets] - [targets ->
    seeds]) - (time-reversed[seeds -> targets] - time-reversed[targets ->
    seeds])."""

    name = "Net TRGC"
    accumulate_psd = False
    net = True
    time_reversed = True

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