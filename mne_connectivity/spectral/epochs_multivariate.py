# Authors: Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#          Denis A. Engemann <denis.engemann@gmail.com>
#          Adam Li <adam2392@gmail.com>
#          Thomas S. Binns <t.s.binns@outlook.com>
#          Tien D. Nguyen <tien-dung.nguyen@charite.de>
#          Richard M. Köhler <koehler.richard@charite.de>
#          Mohammad Orabe <orabe.mhd@gmail.com>
#          Mina Jamshidi Idaji <minajamshidi91@gmail.com>
#
# License: BSD (3-clause)

import inspect
from typing import Optional

import numpy as np
from mne.epochs import BaseEpochs
from mne.parallel import parallel_func
from mne.utils import ProgressBar, logger


def _check_rank_input(rank, data, indices):
    """Check the rank argument is appropriate and compute rank if missing."""
    sv_tol = 1e-6  # tolerance for non-zero singular val (rel. to largest)
    if rank is None:
        rank = np.zeros((2, len(indices[0])), dtype=int)

        if isinstance(data, BaseEpochs):
            # XXX: remove logic once support for mne<1.6 is dropped
            kwargs = dict()
            if "copy" in inspect.getfullargspec(data.get_data).kwonlyargs:
                kwargs["copy"] = False
            data_arr = data.get_data(**kwargs)
        else:
            data_arr = data

        for group_i in range(2):  # seeds and targets
            for con_i, con_idcs in enumerate(indices[group_i]):
                s = np.linalg.svd(data_arr[:, con_idcs.compressed()], compute_uv=False)
                rank[group_i][con_i] = np.min(
                    [np.count_nonzero(epoch >= epoch[0] * sv_tol) for epoch in s]
                )

        logger.info("Estimated data ranks:")
        con_i = 1
        for seed_rank, target_rank in zip(rank[0], rank[1]):
            logger.info(
                "    connection %i - seeds (%i); targets (%i)"
                % (
                    con_i,
                    seed_rank,
                    target_rank,
                )
            )
            con_i += 1
        rank = tuple((np.array(rank[0]), np.array(rank[1])))

    else:
        if (
            len(rank) != 2
            or len(rank[0]) != len(indices[0])
            or len(rank[1]) != len(indices[1])
        ):
            raise ValueError(
                "rank argument must have shape (2, n_cons), "
                "according to n_cons in the indices"
            )
        for seed_idcs, target_idcs, seed_rank, target_rank in zip(
            indices[0], indices[1], rank[0], rank[1]
        ):
            if not (
                0 < seed_rank <= len(seed_idcs) and 0 < target_rank <= len(target_idcs)
            ):
                raise ValueError(
                    "ranks for seeds and targets must be > 0 and <= the "
                    "number of channels in the seeds and targets, "
                    "respectively, for each connection"
                )

    return rank


########################################################################
# Multivariate connectivity estimators


class _AbstractConEstBase:
    """ABC for connectivity estimators."""

    def start_epoch(self):
        raise NotImplementedError("start_epoch method not implemented")

    def accumulate(self, con_idx, csd_xy):
        raise NotImplementedError("accumulate method not implemented")

    def combine(self, other):
        raise NotImplementedError("combine method not implemented")

    def compute_con(self, con_idx, n_epochs):
        raise NotImplementedError("compute_con method not implemented")


class _EpochMeanMultivariateConEstBase(_AbstractConEstBase):
    """Base class for mean epoch-wise multivar. con. estimation methods."""

    n_steps = None
    patterns = None
    con_scores_dtype = np.float64

    def __init__(self, n_signals, n_cons, n_freqs, n_times, n_jobs=1):
        self.n_signals = n_signals
        self.n_cons = n_cons
        self.n_freqs = n_freqs
        self.n_times = n_times
        self.n_jobs = n_jobs

        # include time dimension, even when unused for indexing flexibility
        if n_times == 0:
            self.csd_shape = (n_signals**2, n_freqs)
            self.con_scores = np.zeros(
                (n_cons, n_freqs, 1), dtype=self.con_scores_dtype
            )
        else:
            self.csd_shape = (n_signals**2, n_freqs, n_times)
            self.con_scores = np.zeros(
                (n_cons, n_freqs, n_times), dtype=self.con_scores_dtype
            )

        # allocate space for accumulation of CSD
        self._acc = np.zeros(self.csd_shape, dtype=np.complex128)

        self._compute_n_progress_bar_steps()

    def start_epoch(self):  # noqa: D401
        """Call at the start of each epoch."""
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
        logger.info(
            "Computing %s for connection %i of %i"
            % (
                self.name,
                con_i + 1,
                self.n_cons,
            )
        )

    def _get_block_indices(self, block_i, limit):
        """Get indices for a computation block capped by a limit."""
        indices = np.arange(block_i * self.n_jobs, (block_i + 1) * self.n_jobs)

        return indices[np.nonzero(indices < limit)]

    def reshape_csd(self):
        """Reshape CSD into a matrix of times x freqs x signals x signals."""
        if self.n_times == 0:
            return np.reshape(
                self._acc, (self.n_signals, self.n_signals, self.n_freqs, 1)
            ).transpose(3, 2, 0, 1)

        return np.reshape(
            self._acc, (self.n_signals, self.n_signals, self.n_freqs, self.n_times)
        ).transpose(3, 2, 0, 1)


class _MultivariateCohEstBase(_EpochMeanMultivariateConEstBase):
    """Base estimator for multivariate coherency methods.

    See:
    - Imaginary part of coherency, i.e. multivariate imaginary part of
    coherency (MIC) and multivariate interaction measure (MIM): Ewald et al.
    (2012). NeuroImage. DOI: 10.1016/j.neuroimage.2011.11.084
    - Coherency/coherence, i.e. canonical coherency (CaCoh): Vidaurre et al.
    (2019). NeuroImage. DOI: 10.1016/j.neuroimage.2019.116009
    """

    name: Optional[str] = None
    accumulate_psd = False

    def __init__(self, n_signals, n_cons, n_freqs, n_times, n_jobs=1):
        super(_MultivariateCohEstBase, self).__init__(
            n_signals, n_cons, n_freqs, n_times, n_jobs
        )

    def compute_con(self, indices, ranks, n_epochs=1):
        """Compute multivariate coherency methods."""
        assert self.name in ["CaCoh", "MIC", "MIM"], (
            "the class name is not recognised, please contact the "
            "mne-connectivity developers"
        )

        csd = self.reshape_csd() / n_epochs
        n_times = csd.shape[0]
        times = np.arange(n_times)
        freqs = np.arange(self.n_freqs)

        if self.name in ["CaCoh", "MIC"]:
            self.patterns = np.full(
                (2, self.n_cons, indices[0].shape[1], self.n_freqs, n_times), np.nan
            )

        con_i = 0
        for seed_idcs, target_idcs, seed_rank, target_rank in zip(
            indices[0], indices[1], ranks[0], ranks[1]
        ):
            self._log_connection_number(con_i)

            seed_idcs = seed_idcs.compressed()
            target_idcs = target_idcs.compressed()
            con_idcs = [*seed_idcs, *target_idcs]

            C = csd[np.ix_(times, freqs, con_idcs, con_idcs)]

            # Eqs. 32 & 33 of Ewald et al.; Eq. 15 of Vidaurre et al.
            C_bar, U_bar_aa, U_bar_bb = self._csd_svd(
                C, seed_idcs, seed_rank, target_rank
            )

            self._compute_con_daughter(
                seed_idcs, target_idcs, C, C_bar, U_bar_aa, U_bar_bb, con_i
            )

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

        # Eqs. 32 (Ewald et al.) & 15 (Vidaurre et al.)
        if seed_rank != n_seeds:
            U_aa = np.linalg.svd(np.real(C_aa), full_matrices=False)[0]
            U_bar_aa = U_aa[..., :seed_rank]
        else:
            U_bar_aa = np.broadcast_to(
                np.identity(n_seeds), (n_times, self.n_freqs) + (n_seeds, n_seeds)
            )

        if target_rank != n_targets:
            U_bb = np.linalg.svd(np.real(C_bb), full_matrices=False)[0]
            U_bar_bb = U_bb[..., :target_rank]
        else:
            U_bar_bb = np.broadcast_to(
                np.identity(n_targets), (n_times, self.n_freqs) + (n_targets, n_targets)
            )

        # Eq. 33 (Ewald et al.)
        C_bar_aa = np.matmul(U_bar_aa.transpose(0, 1, 3, 2), np.matmul(C_aa, U_bar_aa))
        C_bar_ab = np.matmul(U_bar_aa.transpose(0, 1, 3, 2), np.matmul(C_ab, U_bar_bb))
        C_bar_bb = np.matmul(U_bar_bb.transpose(0, 1, 3, 2), np.matmul(C_bb, U_bar_bb))
        C_bar_ba = np.matmul(U_bar_bb.transpose(0, 1, 3, 2), np.matmul(C_ba, U_bar_aa))
        C_bar = np.append(
            np.append(C_bar_aa, C_bar_ab, axis=3),
            np.append(C_bar_ba, C_bar_bb, axis=3),
            axis=2,
        )

        return C_bar, U_bar_aa, U_bar_bb

    def _compute_con_daughter(
        self, seed_idcs, target_idcs, C, C_bar, U_bar_aa, U_bar_bb, con_i
    ):
        """Compute multivariate coherency for one connection.

        An empty method to be implemented by subclasses.
        """

    def _compute_t(self, C_r, n_seeds):
        """Compute transformation matrix, T, for frequencies (& times).

        Eq. 3 of Ewald et al.; part of Eq. 9 of Vidaurre et al.
        """
        try:
            return self._invsqrtm(C_r, n_seeds)
        except np.linalg.LinAlgError as error:
            raise RuntimeError(
                "the transformation matrix of the data could not be computed "
                "from the cross-spectral density; check that you are using "
                "full rank data or specify an appropriate rank for the seeds "
                "and targets that is less than or equal to their ranks"
            ) from error

    def _invsqrtm(self, C_r, n_seeds):
        """Compute inverse sqrt of CSD over frequencies and times.

        Parameters
        ----------
        C_r : np.ndarray, shape=(n_freqs, n_times, n_channels, n_channels)
            Real part of the CSD. Expected to be symmetric and non-singular.
        n_seeds : int
            Number of seed channels for the connection.

        Returns
        -------
        T : np.ndarray, shape=(n_freqs, n_times, n_channels, n_channels)
            Inverse square root of the real-valued CSD. Name comes from Ewald
            et al. (2012).

        Notes
        -----
        This approach is a workaround for computing the inverse square root of
        an ND array. SciPy has dedicated functions for this purpose, e.g.
        `sp.linalg.fractional_matrix_power(A, -0.5)` or `sp.linalg.inv(
        sp.linalg.sqrtm(A))`, however these only work with 2D arrays, meaning
        frequencies and times must be looped over which is very slow. There are
        no equivalent functions in NumPy for working with ND arrays (as of
        v1.26).

        The data array is expected to be symmetric and non-singular, otherwise
        a LinAlgError is raised.

        See Eq. 3 of Ewald et al. (2012). NeuroImage. DOI:
        10.1016/j.neuroimage.2011.11.084.
        """
        T = np.zeros_like(C_r, dtype=np.float64)

        # seeds
        eigvals, eigvects = np.linalg.eigh(C_r[:, :, :n_seeds, :n_seeds])
        n_zero = (eigvals == 0).sum()
        if n_zero:  # sign of non-full rank data
            raise np.linalg.LinAlgError(
                "Cannot compute inverse square root of rank-deficient matrix "
                f"with {n_zero}/{len(eigvals)} zero eigenvalue(s)"
            )
        T[:, :, :n_seeds, :n_seeds] = (
            eigvects * np.expand_dims(1.0 / np.sqrt(eigvals), (2))
        ) @ eigvects.transpose(0, 1, 3, 2)

        # targets
        eigvals, eigvects = np.linalg.eigh(C_r[:, :, n_seeds:, n_seeds:])
        n_zero = (eigvals == 0).sum()
        if n_zero:  # sign of non-full rank data
            raise np.linalg.LinAlgError(
                "Cannot compute inverse square root of rank-deficient matrix "
                f"with {n_zero}/{len(eigvals)} zero eigenvalue(s)"
            )
        T[:, :, n_seeds:, n_seeds:] = (
            eigvects * np.expand_dims(1.0 / np.sqrt(eigvals), (2))
        ) @ eigvects.transpose(0, 1, 3, 2)

        return T

    def reshape_results(self):
        """Remove time dimension from results, if necessary."""
        if self.n_times == 0:
            self.con_scores = self.con_scores[..., 0]
            if self.patterns is not None:
                self.patterns = self.patterns[..., 0]


class _MultivariateImCohEstBase(_MultivariateCohEstBase):
    """Base estimator for multivariate imag. part of coherency methods.

    See Ewald et al. (2012). NeuroImage. DOI: 10.1016/j.neuroimage.2011.11.084
    for equation references.
    """

    def _compute_con_daughter(
        self, seed_idcs, target_idcs, C, C_bar, U_bar_aa, U_bar_bb, con_i
    ):
        """Compute multivariate imag. part of coherency for one connection."""
        assert self.name in ["MIC", "MIM"], (
            "the class name is not recognised, please contact the "
            "mne-connectivity developers"
        )

        # Eqs. 3 & 4
        E = self._compute_e(C_bar, n_seeds=U_bar_aa.shape[3])

        if self.name == "MIC":
            self._compute_mic(E, C, seed_idcs, target_idcs, U_bar_aa, U_bar_bb, con_i)
        else:
            self._compute_mim(E, seed_idcs, target_idcs, con_i)

    def _compute_e(self, C, n_seeds):
        """Compute E from the CSD."""
        C_r = np.real(C)

        # Eq. 3
        T = self._compute_t(C_r, n_seeds)

        # Eq. 4
        D = np.matmul(T, np.matmul(C, T))

        # E as imag. part of D between seeds and targets
        return np.imag(D[..., :n_seeds, n_seeds:])

    def _compute_mic(self, E, C, seed_idcs, target_idcs, U_bar_aa, U_bar_bb, con_i):
        """Compute MIC & spatial patterns for one connection."""
        n_seeds = len(seed_idcs)
        n_targets = len(target_idcs)
        n_times = C.shape[0]
        times = np.arange(n_times)
        freqs = np.arange(self.n_freqs)

        # Eigendecomp. to find spatial filters for seeds and targets
        w_seeds, V_seeds = np.linalg.eigh(np.matmul(E, E.transpose(0, 1, 3, 2)))
        w_targets, V_targets = np.linalg.eigh(np.matmul(E.transpose(0, 1, 3, 2), E))
        if len(seed_idcs) == len(target_idcs) and np.all(
            np.sort(seed_idcs) == np.sort(target_idcs)
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
                        if np.all(V_seeds[time_i, freq_i] == V_targets[time_i, freq_i]):
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
        self.patterns[0, con_i, :n_seeds] = (
            np.matmul(
                np.real(C[..., :n_seeds, :n_seeds]),
                np.matmul(U_bar_aa, np.expand_dims(alpha, axis=3)),
            )
        )[..., 0].T

        # Eq. 47 (target spatial patterns)
        self.patterns[1, con_i, :n_targets] = (
            np.matmul(
                np.real(C[..., n_seeds:, n_seeds:]),
                np.matmul(U_bar_bb, np.expand_dims(beta, axis=3)),
            )
        )[..., 0].T

        # Eq. 7
        self.con_scores[con_i] = (
            np.einsum(
                "ijk,ijk->ij", alpha, np.matmul(E, np.expand_dims(beta, axis=3))[..., 0]
            )
            / np.linalg.norm(alpha, axis=2)
            * np.linalg.norm(beta, axis=2)
        ).T

    def _compute_mim(self, E, seed_idcs, target_idcs, con_i):
        """Compute MIM (a.k.a. GIM if seeds == targets) for one connection."""
        # Eq. 14
        self.con_scores[con_i] = (
            np.matmul(E, E.transpose(0, 1, 3, 2)).trace(axis1=2, axis2=3).T
        )

        # Eq. 15
        if len(seed_idcs) == len(target_idcs) and np.all(
            np.sort(seed_idcs) == np.sort(target_idcs)
        ):
            self.con_scores[con_i] *= 0.5


class _MICEst(_MultivariateImCohEstBase):
    """Multivariate imaginary part of coherency (MIC) estimator."""

    name = "MIC"


class _MIMEst(_MultivariateImCohEstBase):
    """Multivariate interaction measure (MIM) estimator."""

    name = "MIM"


class _CaCohEst(_MultivariateCohEstBase):
    """Canonical coherence (CaCoh) estimator.

    See Vidaurre et al. (2019). NeuroImage. DOI:
    10.1016/j.neuroimage.2019.116009 for equation references.
    """

    name = "CaCoh"
    con_scores_dtype = np.complex128  # CaCoh is complex-valued

    def _compute_con_daughter(
        self, seed_idcs, target_idcs, C, C_bar, U_bar_aa, U_bar_bb, con_i
    ):
        """Compute CaCoh & spatial patterns for one connection."""
        assert self.name == "CaCoh", (
            "the class name is not recognised, please contact the "
            "mne-connectivity developers"
        )
        n_seeds = len(seed_idcs)
        n_targets = len(target_idcs)

        rank_seeds = U_bar_aa.shape[3]  # n_seeds after SVD

        C_bar_ab = C_bar[..., :rank_seeds, rank_seeds:]

        # Same as Eq. 3 of Ewald et al. (2012)
        T = self._compute_t(np.real(C_bar), n_seeds=rank_seeds)
        T_aa = T[..., :rank_seeds, :rank_seeds]  # left term in Eq. 9
        T_bb = T[..., rank_seeds:, rank_seeds:]  # right term in Eq. 9

        max_coh, max_phis = self._first_optimise_phi(C_bar_ab, T_aa, T_bb)

        max_coh, max_phis = self._final_optimise_phi(
            C_bar_ab, T_aa, T_bb, max_coh, max_phis
        )

        # Store connectivity score as complex value
        self.con_scores[con_i] = (max_coh * np.exp(-1j * max_phis)).T

        self._compute_patterns(
            max_phis,
            C,
            C_bar_ab,
            T_aa,
            T_bb,
            U_bar_aa,
            U_bar_bb,
            n_seeds,
            n_targets,
            con_i,
        )

    def _first_optimise_phi(self, C_ab, T_aa, T_bb):
        """Find the rough angle, phi, at which coherence is maximised."""
        n_iters = 5

        # starting phi values to optimise over (in radians)
        phis = np.linspace(np.pi / n_iters, np.pi, n_iters)
        phis_coh = np.zeros((n_iters, *C_ab.shape[:2]))
        for iter_i, iter_phi in enumerate(phis):
            phi = np.full(C_ab.shape[:2], fill_value=iter_phi)
            phis_coh[iter_i] = self._compute_cacoh(phi, C_ab, T_aa, T_bb)

        return np.max(phis_coh, axis=0), phis[np.argmax(phis_coh, axis=0)]

    def _final_optimise_phi(self, C_ab, T_aa, T_bb, max_coh, max_phis):
        """Fine-tune the angle at which coherence is maximised.

        Uses a 2nd order Taylor expansion to approximate change in coherence
        w.r.t. phi, and determining the next phi to evaluate coherence on (over
        a total of 10 iterations).

        Depending on how the new phi affects coherence, the step size for the
        subsequent iteration is adjusted, like that in the Levenberg-Marquardt
        algorithm.

        Each time-freq. entry of coherence has its own corresponding phi.
        """
        n_iters = 10  # sufficient for (close to) exact solution
        delta_phi = 1e-6
        mus = np.ones_like(max_phis)  # optimisation step size

        for _ in range(n_iters):
            # 2nd order Taylor expansion around phi
            coh_plus = self._compute_cacoh(max_phis + delta_phi, C_ab, T_aa, T_bb)
            coh_minus = self._compute_cacoh(max_phis - delta_phi, C_ab, T_aa, T_bb)
            f_prime = (coh_plus - coh_minus) / (2 * delta_phi)
            f_prime_prime = (coh_plus + coh_minus - 2 * max_coh) / (delta_phi**2)

            # determine new phi to test
            phis = max_phis + (-f_prime / (f_prime_prime - mus))
            # bound phi in range [-pi, pi]
            phis = np.mod(phis + np.pi / 2, np.pi) - np.pi / 2

            coh = self._compute_cacoh(phis, C_ab, T_aa, T_bb)

            # find where new phi increases coh & update these values
            greater_coh = coh > max_coh
            max_coh[greater_coh] = coh[greater_coh]
            max_phis[greater_coh] = phis[greater_coh]

            # update step size
            mus[greater_coh] /= 2
            mus[~greater_coh] *= 2

        return max_coh, phis

    def _compute_cacoh(self, phis, C_ab, T_aa, T_bb):
        """Compute the maximum coherence for a given set of phis."""
        # from numerator of Eq. 5
        # for a given CSD entry, projects it onto a span with angle phi, such
        # that the magnitude of the projected line is captured in the real part
        C_ab = np.real(np.exp(-1j * np.expand_dims(phis, axis=(2, 3))) * C_ab)

        # Eq. 9; T_aa/bb is sqrt(inv(real(C_aa/bb)))
        D = np.matmul(T_aa, np.matmul(C_ab, T_bb))

        # Eq. 12
        a = np.linalg.eigh(np.matmul(D, D.transpose(0, 1, 3, 2)))[1][..., -1]
        b = np.linalg.eigh(np.matmul(D.transpose(0, 1, 3, 2), D))[1][..., -1]

        # Eq. 8
        numerator = np.einsum(
            "ijk,ijk->ij", a, np.matmul(D, np.expand_dims(b, axis=3))[..., 0]
        )
        denominator = np.sqrt(
            np.einsum("ijk,ijk->ij", a, a) * np.einsum("ijk,ijk->ij", b, b)
        )

        return np.abs(numerator / denominator)

    def _compute_patterns(
        self,
        phis,
        C,
        C_bar_ab,
        T_aa,
        T_bb,
        U_bar_aa,
        U_bar_bb,
        n_seeds,
        n_targets,
        con_i,
    ):
        """Compute CaCoh spatial patterns for the optimised phi."""
        C_bar_ab = np.real(np.exp(-1j * np.expand_dims(phis, axis=(2, 3))) * C_bar_ab)
        D = np.matmul(T_aa, np.matmul(C_bar_ab, T_bb))
        a = np.linalg.eigh(np.matmul(D, D.transpose(0, 1, 3, 2)))[1][..., -1]
        b = np.linalg.eigh(np.matmul(D.transpose(0, 1, 3, 2), D))[1][..., -1]

        # Eq. 7 rearranged - multiply both sides by sqrt(inv(real(C_aa/bb)))
        alpha = np.matmul(T_aa, np.expand_dims(a, axis=3))  # filter for seeds
        beta = np.matmul(T_bb, np.expand_dims(b, axis=3))  # filter for targets

        # Eq. 14; U_bar inclusion follows Eqs. 46 & 47 of Ewald et al. (2012)
        # seed spatial patterns
        self.patterns[0, con_i, :n_seeds] = (
            np.matmul(np.real(C[..., :n_seeds, :n_seeds]), np.matmul(U_bar_aa, alpha))
        )[..., 0].T
        # target spatial patterns
        self.patterns[1, con_i, :n_targets] = (
            np.matmul(np.real(C[..., n_seeds:, n_seeds:]), np.matmul(U_bar_bb, beta))
        )[..., 0].T


class _GCEstBase(_EpochMeanMultivariateConEstBase):
    """Base multivariate state-space Granger causality estimator."""

    accumulate_psd = False

    def __init__(self, n_signals, n_cons, n_freqs, n_times, n_lags, n_jobs=1):
        super(_GCEstBase, self).__init__(n_signals, n_cons, n_freqs, n_times, n_jobs)

        self.freq_res = (self.n_freqs - 1) * 2
        if n_lags >= self.freq_res:
            raise ValueError(
                "the number of lags (%i) must be less than double the "
                "frequency resolution (%i)"
                % (
                    n_lags,
                    self.freq_res,
                )
            )
        self.n_lags = n_lags

    def compute_con(self, indices, ranks, n_epochs=1):
        """Compute multivariate state-space Granger causality."""
        assert self.name in ["GC", "GC time-reversed"], (
            "the class name is not recognised, please contact the "
            "mne-connectivity developers"
        )

        csd = self.reshape_csd() / n_epochs

        n_times = csd.shape[0]
        times = np.arange(n_times)
        freqs = np.arange(self.n_freqs)

        con_i = 0
        for seed_idcs, target_idcs, seed_rank, target_rank in zip(
            indices[0], indices[1], ranks[0], ranks[1]
        ):
            self._log_connection_number(con_i)

            seed_idcs = seed_idcs.compressed()
            target_idcs = target_idcs.compressed()
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
                A_f, (n_times, n_signals, n_signals * self.n_lags), order="F"
            )
            A, K = self._full_var_to_iss(A_f_3d)

            self.con_scores[con_i] = self._iss_to_ugc(
                A, A_f_3d, K, V, con_seeds, con_targets
            )

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

        C_bar_aa = np.matmul(U_bar_aa.transpose(1, 0), np.matmul(C_aa, U_bar_aa))
        C_bar_ab = np.matmul(U_bar_aa.transpose(1, 0), np.matmul(C_ab, U_bar_bb))
        C_bar_bb = np.matmul(U_bar_bb.transpose(1, 0), np.matmul(C_bb, U_bar_bb))
        C_bar_ba = np.matmul(U_bar_bb.transpose(1, 0), np.matmul(C_ba, U_bar_aa))
        C_bar = np.append(
            np.append(C_bar_aa, C_bar_ab, axis=3),
            np.append(C_bar_ba, C_bar_bb, axis=3),
            axis=2,
        )

        return C_bar

    def _compute_autocov(self, csd):
        """Compute autocovariance from the CSD."""
        n_times = csd.shape[0]
        n_signals = csd.shape[2]

        circular_shifted_csd = np.concatenate(
            [np.flip(np.conj(csd[:, 1:]), axis=1), csd[:, :-1]], axis=1
        )
        ifft_shifted_csd = self._block_ifft(circular_shifted_csd, self.freq_res)
        lags_ifft_shifted_csd = np.reshape(
            ifft_shifted_csd[:, : self.n_lags + 1],
            (n_times, self.n_lags + 1, n_signals**2),
            order="F",
        )

        signs = np.repeat([1], self.n_lags + 1).tolist()
        signs[1::2] = [x * -1 for x in signs[1::2]]
        sign_matrix = np.repeat(
            np.tile(np.array(signs), (n_signals**2, 1))[np.newaxis], n_times, axis=0
        ).transpose(0, 2, 1)

        return np.real(
            np.reshape(
                sign_matrix * lags_ifft_shifted_csd,
                (n_times, self.n_lags + 1, n_signals, n_signals),
                order="F",
            )
        )

    def _block_ifft(self, csd, n_points):
        """Compute block iFFT with n points."""
        shape = csd.shape
        csd_3d = np.reshape(csd, (shape[0], shape[1], shape[2] * shape[3]), order="F")

        csd_ifft = np.fft.ifft(csd_3d, n=n_points, axis=1)

        return np.reshape(csd_ifft, shape, order="F")

    def _autocov_to_full_var(self, autocov):
        """Compute full VAR model using Whittle's LWR recursion."""
        if np.any(np.linalg.det(autocov) == 0):
            raise RuntimeError(
                "the autocovariance matrix is singular; check if your data is "
                "rank deficient and specify an appropriate rank argument <= "
                "the rank of the seeds and targets"
            )

        A_f, V = self._whittle_lwr_recursion(autocov)

        if not np.isfinite(A_f).all():
            raise RuntimeError(
                "at least one VAR model coefficient is "
                "infinite or NaN; check the data you are using"
            )

        try:
            np.linalg.cholesky(V)
        except np.linalg.LinAlgError as np_error:
            raise RuntimeError(
                "the covariance matrix of the residuals is not "
                "positive-definite; check the singular values of your data "
                "and specify an appropriate rank argument <= the rank of the "
                "seeds and targets"
            ) from np_error

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
            G[:, 1:, :, :].transpose(0, 3, 1, 2), (t, qn, n), order="F"
        )  # forward autocov
        G_b = np.reshape(
            np.flip(G[:, 1:, :, :], 1).transpose(0, 3, 2, 1), (t, n, qn), order="F"
        ).transpose(0, 2, 1)  # backward autocov

        A_f = np.zeros((t, n, qn))  # forward coefficients
        A_b = np.zeros((t, n, qn))  # backward coefficients

        k = 1  # model order
        r = q - k
        k_f = np.arange(k * n)  # forward indices
        k_b = np.arange(r * n, qn)  # backward indices

        try:
            A_f[:, :, k_f] = np.linalg.solve(
                cov, G_b[:, k_b, :].transpose(0, 2, 1)
            ).transpose(0, 2, 1)
            A_b[:, :, k_b] = np.linalg.solve(
                cov, G_f[:, k_f, :].transpose(0, 2, 1)
            ).transpose(0, 2, 1)

            # Perform recursion
            for k in np.arange(2, q + 1):
                var_A = G_b[:, (r - 1) * n : r * n, :] - np.matmul(
                    A_f[:, :, k_f], G_b[:, k_b, :]
                )
                var_B = cov - np.matmul(A_b[:, :, k_b], G_b[:, k_b, :])
                AA_f = np.linalg.solve(var_B, var_A.transpose(0, 2, 1)).transpose(
                    0, 2, 1
                )

                var_A = G_f[:, (k - 1) * n : k * n, :] - np.matmul(
                    A_b[:, :, k_b], G_f[:, k_f, :]
                )
                var_B = cov - np.matmul(A_f[:, :, k_f], G_f[:, k_f, :])
                AA_b = np.linalg.solve(var_B, var_A.transpose(0, 2, 1)).transpose(
                    0, 2, 1
                )

                A_f_previous = A_f[:, :, k_f]
                A_b_previous = A_b[:, :, k_b]

                r = q - k
                k_f = np.arange(k * n)
                k_b = np.arange(r * n, qn)

                A_f[:, :, k_f] = np.dstack(
                    (A_f_previous - np.matmul(AA_f, A_b_previous), AA_f)
                )
                A_b[:, :, k_b] = np.dstack(
                    (AA_b, A_b_previous - np.matmul(AA_b, A_f_previous))
                )
        except np.linalg.LinAlgError as np_error:
            raise RuntimeError(
                "the autocovariance matrix is singular; check if your data is "
                "rank deficient and specify an appropriate rank argument <= "
                "the rank of the seeds and targets"
            ) from np_error

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
        K = np.hstack(
            (
                np.dstack(t * [np.eye(m)]).transpose(2, 0, 1),
                np.zeros((t, (m * (p - 1)), m)),
            )
        )  # Kalman gain matrix

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
        return np.real(np.log(np.linalg.det(S_11)) - np.log(np.linalg.det(S_11 - HVH)))

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
        for block_i in ProgressBar(range(self.n_steps), mesg="frequency blocks"):
            freqs = self._get_block_indices(block_i, self.n_freqs)
            H[freqs] = parallel(
                parallel_compute_H(A, C, K, z[k], I_n, I_m) for k in freqs
            )

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
            C[t], linalg.lu_solve(linalg.lu_factor(z_k * I_m - A[t]), K[t])
        )

    return H


class _GCEst(_GCEstBase):
    """[seeds -> targets] state-space GC estimator."""

    name = "GC"


class _GCTREst(_GCEstBase):
    """time-reversed[seeds -> targets] state-space GC estimator."""

    name = "GC time-reversed"


# map names to estimator types
_CON_METHOD_MAP_MULTIVARIATE = {
    "cacoh": _CaCohEst,
    "mic": _MICEst,
    "mim": _MIMEst,
    "gc": _GCEst,
    "gc_tr": _GCTREst,
}

_multivariate_methods = ["cacoh", "mic", "mim", "gc", "gc_tr"]
_gc_methods = ["gc", "gc_tr"]
_patterns_methods = ["cacoh", "mic"]  # methods with spatial patterns
