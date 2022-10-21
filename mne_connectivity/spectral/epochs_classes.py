# Authors: Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#          Denis A. Engemann <denis.engemann@gmail.com>
#          Adam Li <adam2392@gmail.com>
#          Thomas Samuel Binns <thomas-samuel.binns@charite.de>
#
# License: BSD (3-clause)

import numpy as np
from scipy import linalg as spla


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

    def __init__(self, n_signals, n_cons, n_freqs, n_times):
        self.n_signals = n_signals
        self.n_cons = n_cons
        self.n_freqs = n_freqs
        self.n_times = n_times

        if n_times == 0:
            self.csd_shape = (n_signals**2, n_freqs)
            self.con_scores = np.zeros((n_cons, n_freqs, 1))
        else:
            self.csd_shape = (n_signals**2, n_freqs, n_times)
            self.con_scores = np.zeros((n_cons, n_freqs, n_times))

    def start_epoch(self):  # noqa: D401
        """Called at the start of each epoch."""
        pass  # for this type of con. method we don't do anything

    def combine(self, other):
        """Include con. accumated for some epochs in this estimate."""
        self._acc += other._acc


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
    """Base Estimator for coherenc-based multivariate methods."""

    def __init__(self, n_signals, n_cons, n_freqs, n_times):
        super(_MultivarCohEstBase, self).__init__(n_signals, n_cons, n_freqs, n_times)

        # allocate space for accumulation of CSD
        self._acc = np.zeros(self.csd_shape, dtype=np.complex128)

    def accumulate(self, con_idx, csd_xy):
        """Accumulate CSD for some connections."""
        self._acc[con_idx] += csd_xy

    def reshape_csd(self):
        """Reshapes CSD into a matrix of times x freqs x signals x signals."""
        if self.n_times == 0:
            return np.reshape(self._acc, (self.n_signals, self.n_signals, self.n_freqs, 1)).transpose(3, 2, 0, 1)
        return np.reshape(self._acc, (self.n_signals, self.n_signals, self.n_freqs, self.n_times)).transpose(3, 2, 0, 1)

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
        C_bar_aa = np.matmul(U_bar_aa.transpose(0, 1, 3, 2), np.matmul(C_aa, U_bar_aa))
        C_bar_ab = np.matmul(U_bar_aa.transpose(0, 1, 3, 2), np.matmul(C_ab, U_bar_bb))
        C_bar_bb = np.matmul(U_bar_bb.transpose(0, 1, 3, 2), np.matmul(C_bb, U_bar_bb))
        C_bar_ba = np.matmul(U_bar_bb.transpose(0, 1, 3, 2), np.matmul(C_ba, U_bar_aa))
        C_bar = np.append(
            np.append(C_bar_aa, C_bar_ab, axis=3), np.append(C_bar_ba, C_bar_bb, axis=3), axis=2
        )

        return C_bar, U_bar_aa, U_bar_bb

    def compute_e(self, csd, n_seeds):
        """Computes E as the imaginary part of the transformed cross-spectra D
        derived from the original cross-spectra "csd" between the seed and target
        signals."""
        # Equation 3
        n_times = csd.shape[0]
        n_freqs = csd.shape[1]
        T = np.zeros(csd.shape)
        # No clear way to do this without list comprehension (function only accepts square matrices)
        # Could be a good place for parallelisation
        T[:, :, :n_seeds, :n_seeds] = np.array([[spla.fractional_matrix_power(
            np.real(csd[time_i, freq_i, :n_seeds, :n_seeds]), -0.5
        ) for freq_i in range(n_freqs)] for time_i in range(n_times)])  # real(C_aa)^-1/2
        T[:, :, n_seeds:, n_seeds:] = np.array([[spla.fractional_matrix_power(
            np.real(csd[time_i, freq_i, n_seeds:, n_seeds:]), -0.5
        ) for freq_i in range(n_freqs)] for time_i in range(n_times)]) # real(C_bb)^-1/2

        # Equation 4
        D = np.matmul(T, np.matmul(csd, T))

        # E as the imaginary part of D between seeds and targets
        E = np.imag(D[:, :, :n_seeds, n_seeds:])

        return E
    
    def reshape_con_scores(self):
        """Removes the time dimension from con. scores, if necessary"""
        if self.n_times == 0:
            self.con_scores = self.con_scores[:,:,0]

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

class _MIMEst(_MultivarCohEstBase):
    """Estimator for MIM (multivariate interaction measure)"""

    name = "MIM"

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
            self.con_scores[node_i, :, :] = np.matmul(E, E.transpose(0, 1, 3, 2)).trace(axis1=2, axis2=3).transpose(1, 0)
            node_i += 1
        self.reshape_con_scores()


class _MICEst(_MultivarCohEstBase):
    """Estimator for MIC (maximized imaginary coherence)"""

    name = "MIC"

    def compute_con(
        self, seeds, targets, n_seed_components, n_target_components, n_epochs
    ):
        """Computes the maximized imaginary coherence between two sets of
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

            # Weights for signals in the groups
            w_a, V_a = np.linalg.eigh(np.matmul(E, E.transpose(0, 1, 3, 2)))
            w_b, V_b = np.linalg.eigh(np.matmul(E.transpose(0, 1, 3, 2), E))
            alpha = V_a[np.arange(n_times)[:, None], np.arange(self.n_freqs), :, w_a.argmax(axis=2)]
            beta = V_b[np.arange(n_times)[:, None], np.arange(self.n_freqs), :, w_b.argmax(axis=2)]

            # Eq. 7
            self.con_scores[node_i, :, :] = (np.einsum(
                "ijk,ijk->ij",
                alpha,
                np.matmul(E, np.expand_dims(beta, 3))[:, :, :, 0]) / np.linalg.norm(alpha, axis=2) * np.linalg.norm(beta, axis=2)
            ).transpose(1, 0)
            node_i += 1
        self.reshape_con_scores()


class _PLVEst(_EpochMeanConEstBase):
    """PLV Estimator."""

    name = 'PLV'

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