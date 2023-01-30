import sys
import mne

import autograd.numpy as np
import scipy.optimize as spopt

from autograd import grad
from autograd import value_and_grad as vgrad
from scipy.linalg import LinAlgError

from .util import _ensure_ndim, rand_stable, rand_psd
from .util import linesearch, soft_thresh_At, block_thresh_At
from .util import relnormdiff
from .message_passing import rts_smooth, rts_smooth_fast
from .numpy_numthreads import numpy_num_threads

from .mne_util import run_pca_on_subject, apply_projs 

try:
    from autograd_linalg import solve_triangular
except ImportError:
    raise RuntimeError("must install `autograd_linalg` package")
    
from autograd.numpy import einsum

from datetime import datetime


class _Model(object):
    """ Base class for any model applied to MEG data that handles storing and
        unpacking data from tuples. """

    def __init__(self):
        self._subjectdata = None
        self._n_timepts = 0
        self._ntrials_all = 0
        self._nsubjects = 0

    def set_data(self, subjectdata):
        n_timepts_lst = [self.unpack_subject_data(e)[0].shape[1] for e in
                         subjectdata]
        assert len(list(set(n_timepts_lst))) == 1
        self._n_timepts = n_timepts_lst[0]
        ntrials_lst = [self.unpack_subject_data(e)[0].shape[0] for e in \
                       subjectdata]
        self._ntrials_all = np.sum(ntrials_lst)
        self._nsubjects = len(subjectdata)
        self._subjectdata = subjectdata

    def unpack_all_subject_data(self):
        if self._subjectdata is None:
            raise ValueError("use set_data to add subject data")
        return map(self.unpack_subject_data, self._subjectdata)

    @classmethod
    def unpack_subject_data(cls, sdata):
        obs, fwd_roi_snsr, fwd_src_snsr, snsr_cov, which_roi = sdata
        Y = obs
        w_s = 1.
        if isinstance(obs, tuple):
            if len(obs) == 2:
                Y, w_s = obs
            else:
                raise ValueError("invalid format for subject data")
        else:
            Y = obs
            w_s = 1.

        return Y, w_s, fwd_roi_snsr, fwd_src_snsr, snsr_cov, which_roi


class LDS(_Model):
    """ State-space model for MEG data, as described in "A state-space model of
        cross-region dynamic connectivity in MEG/EEG", Yang et al., NIPS 2016.
    """

    def __init__(self, lam0=0., lam1=0., penalty='ridge', store_St=True):

        super().__init__()
        self._model_initalized = False
        self.lam0 = lam0
        self.lam1 = lam1

        if penalty not in ('ridge', 'lasso', 'group-lasso'):
            raise ValueError('penalty must be one of: ridge, lasso,' \
                             + ' group-lasso')
        self._penalty = penalty

        # initialize lists of smoothed estimates
        self._mus_smooth_lst = None
        self._sigmas_smooth_lst = None
        self._sigmas_tnt_smooth_lst = None
        self._loglik = None
        self._store_St = bool(store_St)

        self._all_subject_data = list()

    #SNR boost epochs, bootstraps of 3
    def bootstrap_subject(self, epochs, subject_name, seed=8675309, sfreq=100,
                          lower=None, upper=None, nbootstrap=3, g_nsamples=-5,
                          overwrite=False, validation_set=True):

        datasets = ['train', 'validation']
        # use_erm = eq = False
        independent = False
        if g_nsamples == 0:
            print('nsamples == 0, ensuring independence of samples')
            independent = True
        elif g_nsamples == -1:
            print("using half of trials per sample")
        elif g_nsamples == -2:
            print("using empty room noise at half of trials per sample")
            # use_erm = True
        elif g_nsamples == -3:
            print("using independent and trial-count equalized samples")
            eq = True
            independent = True
        elif g_nsamples == -4:
            print("using independent, trial-count equailized, non-boosted"
                  "samples")
            assert nbootstrap == 0  # sanity check
            eq = True
            independent = True
            datasets = ['train']
        elif g_nsamples == -5:
            print("using independent, trial-count equailized, integer boosted"
                  "samples")
            eq = True
            independent = True
            datasets = ['train']

        if lower is not None or upper is not None:
            if upper is None:
                print('high-pass filtering at %.2f Hz' % lower)
            elif lower is None:
                print('low-pass filtering at %.2f Hz' % upper)
            else:
                print('band-pass filtering from %.2f-%.2f Hz' % (lower, upper))

        if sfreq is not None:
            print('resampling to %.2f Hz' % sfreq)

        print(":: processing subject %s" % subject_name)
        np.random.seed(seed)

        for dataset in datasets:

            print('  generating ', dataset, ' set')
            # datadir = './data'

            condition_map = {'auditory_left':['auditory_left'],
                             'auditory_right': ['auditory_right'],
                             'visual_left': ['visual_left'],
                             'visual_right': ['visual_right']}
            condition_eq_map = dict(auditory_left=['auditory_left'],
                                    auditory_right=['auditory_right'],
                                    visual_left=['visual_left'],
                                    visual_right='visual_right')

            if eq:
                epochs.equalize_event_counts(list(condition_map))
                cond_map = condition_eq_map

            # apply band-pass filter to limit signal to desired frequency band
            if lower is not None or upper is not None:
                epochs = epochs.filter(lower, upper)

            # perform resampling with specified sampling frequency
            if sfreq is not None:
                epochs = epochs.resample(sfreq)

            data_bs_all = list()
            events_bs_all = list()
            for cond in sorted(cond_map.keys()):
                print("    -> condition %s: bootstrapping" % cond, end='')
                ep = epochs[cond_map[cond]]
                dat = ep.get_data().copy()
                ntrials, T, p = dat.shape

                use_bootstrap = nbootstrap
                if g_nsamples == -4:
                    nsamples = 1
                    use_bootstrap = ntrials
                elif g_nsamples == -5:
                    nsamples = nbootstrap
                    use_bootstrap = ntrials // nsamples
                elif independent:
                    nsamples = (ntrials - 1) // use_bootstrap
                elif g_nsamples in (-1, -2):
                    nsamples = ntrials // 2
                else:
                    assert g_nsamples > 0
                    nsamples = g_nsamples
                print("    using %d samples (%d trials)"
                      % (nsamples, use_bootstrap))

                # bootstrap here
                if independent:  # independent
                    if nsamples == 1 and use_bootstrap == ntrials:
                        inds = np.arange(ntrials)
                    else:
                        inds = np.random.choice(ntrials,
                                                nsamples * use_bootstrap)
                    inds.shape = (use_bootstrap, nsamples)
                    dat_bs = np.mean(dat[inds], axis=1)
                    events_bs = ep.events[inds[:, 0]]
                    assert dat_bs.shape[0] == events_bs.shape[0]
                else:
                    dat_bs = np.empty((ntrials, T, p))
                    events_bs = np.empty((ntrials, 3), dtype=int)
                    for i in range(ntrials):

                        inds = list(set(range(ntrials)).difference([i]))
                        inds = np.random.choice(inds, size=nsamples,
                                                replace=False)
                        inds = np.append(inds, i)

                        dat_bs[i] = np.mean(dat[inds], axis=0)
                        events_bs[i] = ep.events[i]

                    inds = np.random.choice(ntrials, size=use_bootstrap,
                                            replace=False)
                    dat_bs = dat_bs[inds]
                    events_bs = events_bs[inds]

                assert dat_bs.shape == (use_bootstrap, T, p)
                assert events_bs.shape == (use_bootstrap, 3)
                assert (events_bs[:, 2] == events_bs[0, 2]).all()

                data_bs_all.append(dat_bs)
                events_bs_all.append(events_bs)

            # write bootstrap epochs
            info_dict = epochs.info.copy()

            dat_all = np.vstack(data_bs_all)
            events_all = np.vstack(events_bs_all)
            # replace first column with sequential list as we don't really care
            # about the raw timings
            events_all[:, 0] = np.arange(events_all.shape[0])

            epochs_bs = mne.EpochsArray(
                dat_all, info_dict, events=events_all, tmin=-0.2,
                event_id=epochs.event_id.copy(), on_missing='ignore')

            return epochs_bs

    def add_subject(self, subject,condition,epochs,labels,fwd,
                    cov):

        epochs_bs = self.bootstrap_subject(epochs, subject)
        epochs_bs = epochs_bs[condition]
        epochs = epochs_bs

        # ensure cov and fwd use correct channels
        cov = cov.pick_channels(epochs.ch_names, ordered=True)
        fwd = mne.convert_forward_solution(fwd, force_fixed=True)
        fwd = fwd.pick_channels(epochs.ch_names, ordered=True)

        if not self._model_initalized:
            n_timepts = len(epochs.times)
            num_roi = len(labels)
            self._init_model(n_timepts, num_roi)
            self._model_initalized = True
            self.n_timepts = n_timepts
            self.num_roi = num_roi
            self.times = epochs.times
        if len(epochs.times) != self._n_times:
            raise ValueError(f'Number of time points ({len(epochs.times)})' /
                             'does not match original count ({self._n_times})')
        
        # scale cov matrix according to number of bootstraps
        cov_scale = 3 # equal to number of bootstrap trials
        cov['data'] /= cov_scale
        fwd, cov = apply_projs(epochs_bs, fwd, cov)

        sdata = run_pca_on_subject(subject, epochs_bs, fwd, cov, labels,
                                   dim_mode='pctvar', mean_center=True)
        data, fwd_roi_snsr, fwd_src_snsr, snsr_cov, which_roi = sdata
        subjectdata = (data, fwd_roi_snsr, fwd_src_snsr, snsr_cov, which_roi)

        self._all_subject_data.append(subjectdata)

        self._subject_data[subject] = dict()
        self._subject_data[subject]['epochs'] = data
        self._subject_data[subject]['fwd_src_snsr'] = fwd_src_snsr
        self._subject_data[subject]['fwd_roi_snsr'] = fwd_roi_snsr
        self._subject_data[subject]['snsr_cov'] = snsr_cov
        self._subject_data[subject]['labels'] = labels
        self._subject_data[subject]['which_roi'] = which_roi


    def _init_model(self, n_timepts, num_roi, A_t_=None, roi_cov=None,
                    mu0=None, roi_cov_0=None, log_sigsq_lst=None):

        self._n_times = n_timepts
        self._subject_data = dict()

        set_default = \
                lambda prm, val, deflt: \
                    self.__setattr__(prm, val.copy() if val is not None else
                                     deflt)

        # initialize parameters
        set_default("A_t_", A_t_,
                    np.stack([rand_stable(num_roi, maxew=0.7) for _ in
                              range(n_timepts)], axis=0))
        set_default("roi_cov", roi_cov, rand_psd(num_roi))
        set_default("mu0", mu0, np.zeros(num_roi))
        set_default("roi_cov_0", roi_cov_0, rand_psd(num_roi))
        set_default("log_sigsq_lst", log_sigsq_lst,
                    [np.log(np.random.gamma(2, 1, size=num_roi+1))])

        # initialize sufficient statistics
        n_timepts, num_roi, _ = self.A_t_.shape
        self._B0 = np.zeros((num_roi, num_roi))
        self._B1 = np.zeros((n_timepts-1, num_roi, num_roi))
        self._B3 = np.zeros((n_timepts-1, num_roi, num_roi))
        self._B2 = np.zeros((n_timepts-1, num_roi, num_roi))
        self._B4 = list()

    def set_data(self, subjectdata):
        # add subject data, re-generate log_sigsq_lst if necessary
        super().set_data(subjectdata)
        if len(self.log_sigsq_lst) != self._nsubjects:
            num_roi = self.log_sigsq_lst[0].shape[0]
            self.log_sigsq_lst = [np.log(np.random.gamma(2, 1, size=num_roi))
                                  for _ in range(self._nsubjects)]

        # reset smoothed estimates and log-likelihood (no longer valid if
        # new data was added)
        self._mus_smooth_lst = None
        self._sigmas_smooth_lst = None
        self._sigmas_tnt_smooth_lst = None
        self._loglik = None
        self._B4 = [None] * self._nsubjects

    def _em_objective(self):

        _, num_roi, _ = self.A_t_.shape

        L_roi_cov_0 = np.linalg.cholesky(self.roi_cov_0)
        L_roi_cov = np.linalg.cholesky(self.roi_cov)

        L1 = 0.
        L2 = 0.
        L3 = 0.

        obj = 0.
        for s, sdata in enumerate(self.unpack_all_subject_data()):

            Y, w_s, fwd_roi_snsr, fwd_src_snsr, snsr_cov, which_roi = sdata

            ntrials, n_timepts, _ = Y.shape

            sigsq_vals = np.exp(self.log_sigsq_lst[s])
            R = LDS.R_(snsr_cov, fwd_src_snsr, sigsq_vals, which_roi)
            L_R = np.linalg.cholesky(R)

            if (self._mus_smooth_lst is None or self._sigmas_smooth_lst is None
                or self._sigmas_tnt_smooth_lst is None):
                roi_cov_t = _ensure_ndim(self.roi_cov, n_timepts, 3)
                with numpy_num_threads(1):
                    _, mus_smooth, sigmas_smooth, sigmas_tnt_smooth = \
                            rts_smooth_fast(Y, self.A_t_, fwd_roi_snsr,
                                            roi_cov_t, R, self.mu0,
                                            self.roi_cov_0,
                                            compute_lag1_cov=True)

            else:
                mus_smooth = self._mus_smooth_lst[s]
                sigmas_smooth = self._sigmas_smooth_lst[s]
                sigmas_tnt_smooth = self._sigmas_tnt_smooth_lst[s]

            x_smooth_0_outer = einsum('ri,rj->rij', mus_smooth[:,0,:num_roi],
                                                     mus_smooth[:,0,:num_roi])
            B0 = w_s*np.sum(sigmas_smooth[:,0,:num_roi,:num_roi] +
                            x_smooth_0_outer, axis=0)

            x_smooth_outer = einsum('rti,rtj->rtij', mus_smooth[:,1:,:num_roi],
                                    mus_smooth[:,1:,:num_roi])
            B1 = w_s*np.sum(sigmas_smooth[:,1:,:num_roi,:num_roi] +
                            x_smooth_outer, axis=0)
            z_smooth_outer = einsum('rti,rtj->rtij', mus_smooth[:,:-1,:],
                                                      mus_smooth[:,:-1,:])
            B3 = w_s*np.sum(sigmas_smooth[:,:-1,:,:] + z_smooth_outer, axis=0)

            mus_smooth_outer_l1 = einsum('rti,rtj->rtij',
                                          mus_smooth[:,1:,:num_roi],
                                          mus_smooth[:,:-1,:])
            B2 = w_s*np.sum(sigmas_tnt_smooth[:,:,:num_roi,:] +
                            mus_smooth_outer_l1, axis=0)

            # obj += L1(roi_cov_0)
            L_roi_cov_0_inv_B0 = solve_triangular(L_roi_cov_0, B0, lower=True)
            L1 += (ntrials*2.*np.sum(np.log(np.diag(L_roi_cov_0)))
                   + np.trace(solve_triangular(L_roi_cov_0, L_roi_cov_0_inv_B0,
                                               lower=True, trans='T')))

            At = self.A_t_[:-1]
            AtB2T = einsum('tik,tjk->tij', At, B2)
            B2AtT = einsum('tik,tjk->tij', B2, At)
            tmp = einsum('tik,tkl->til', At, B3)
            AtB3AtT = einsum('tik,tjk->tij', tmp, At)

            tmp = np.sum(B1 - AtB2T - B2AtT + AtB3AtT, axis=0)

            # obj += L2(roi_cov, At)
            L_roi_cov_inv_tmp = solve_triangular(L_roi_cov, tmp, lower=True)
            L2 += (ntrials*(n_timepts-1)*2.*np.sum(np.log(np.diag(L_roi_cov)))
                   + np.trace(solve_triangular(L_roi_cov, L_roi_cov_inv_tmp,
                                               lower=True, trans='T')))

            res = Y - einsum('ik,ntk->nti', fwd_roi_snsr,
                             mus_smooth[:,:,:num_roi])
            CP_smooth = einsum('ik,ntkj->ntij', fwd_roi_snsr,
                               sigmas_smooth[:,:,:num_roi,:num_roi])

            B4 = w_s*(np.sum(einsum('nti,ntj->ntij', res, res), axis=(0,1))
                             + np.sum(einsum('ntik,jk->ntij', CP_smooth,
                                             fwd_roi_snsr), axis=(0,1)))
            self._B4[s] = B4

            # obj += L3(sigsq_vals)
            L_R_inv_B4 = solve_triangular(L_R, B4, lower=True)
            L3 += (ntrials*n_timepts*2*np.sum(np.log(np.diag(L_R)))
                   + np.trace(solve_triangular(L_R, L_R_inv_B4, lower=True,
                                               trans='T')))

        obj = (L1 + L2 + L3) / self._ntrials_all

        # obj += penalty
        if self.lam0 > 0.:
            if self._penalty == 'ridge':
                obj += self.lam0*np.sum(At**2)
            elif self._penalty == 'lasso':
                At_diag = np.diagonal(At, axis1=-2, axis2=-1)
                sum_At_diag = np.sum(np.abs(At_diag))
                obj += self.lam0*(np.sum(np.abs(At)) - sum_At_diag)
            elif self._penalty == 'group-lasso':
                At_diag = np.diagonal(At, axis1=-2, axis2=-1)
                norm_At_diag = np.sum(np.linalg.norm(At_diag, axis=0))
                norm_At = np.sum(np.linalg.norm(At, axis=0))
                obj += self.lam1*(norm_At - norm_At_diag)
        if self.lam1 > 0.:
            AtmAtm1_2 = (At[1:] - At[:-1])**2
            obj += self.lam1*np.sum(AtmAtm1_2)

        return obj

    def fit(self, niter=100, tol=1e-6, A_t_roi_cov_niter=100,
            A_t_roi_cov_tol=1e-6, verbose=0, update_A_t_=True,
            update_roi_cov=True, update_roi_cov_0=True, stationary_A_t_=False,
            diag_roi_cov=False, update_sigsq=True, do_final_smoothing=True,
            average_mus_smooth=True, Atrue=None, tau=0.1, c1=1e-4):

        self.set_data(self._all_subject_data)

        fxn_start = datetime.now()

        n_timepts, num_roi, _ = self.A_t_.shape

        # make initial A_t_ stationary if stationary_A_t_ option specified
        if stationary_A_t_:
            self.A_t_[:] = np.mean(self.A_t_, axis=0)

        # set parameters for (A_t_, roi_cov) optimization
        self._A_t_roi_cov_niter = A_t_roi_cov_niter
        self._A_t_roi_cov_tol = A_t_roi_cov_tol

        # make initial roi_cov, roi_cov_0 diagonal if diag_roi_cov specified
        if diag_roi_cov:
            self.roi_cov_0 = np.diag(np.diag(self.roi_cov_0))
            self.roi_cov = np.diag(np.diag(self.roi_cov))

        # keeping track of objective value and best parameters
        objvals = np.zeros(niter+1)
        converged = False
        best_objval = np.finfo('float').max
        best_params = (self.A_t_.copy(), self.roi_cov.copy(), self.mu0.copy(),
                       self.roi_cov_0.copy(), [l.copy() for l in
                       self.log_sigsq_lst])

        # previous parameter values (for checking convergence)
        At_prev = None
        roi_cov_prev = None
        roi_cov_0_prev = None
        log_sigsq_lst_prev = None

        if Atrue is not None:
            import matplotlib.pyplot as plt
            fig_A_t_, ax_A_t_ = plt.subplots(num_roi, num_roi, sharex=True,
                                             sharey=True)
            plt.ion()

        # calculate initial objective value, check for updated best iterate
        # have to do e-step here to initialize suff stats for _m_step
        if (self._mus_smooth_lst is None or self._sigmas_smooth_lst is None
                or self._sigmas_tnt_smooth_lst is None):
            self._e_step(verbose=verbose-1)

        objval = self._em_objective()
        objvals[0] = objval

        for it in range(1, niter+1):

            iter_start = datetime.now()

            if verbose > 0:
                print("em: it %d / %d" % (it, niter))
                sys.stdout.flush()
                sys.stderr.flush()

            # record values from previous M-step
            At_prev = self.A_t_[:-1].copy()
            roi_cov_prev = self.roi_cov.copy()
            roi_cov_0_prev = self.roi_cov_0.copy()
            log_sigsq_lst_prev = np.array(self.log_sigsq_lst).copy()

            self._m_step(update_A_t_=update_A_t_,
                        update_roi_cov=update_roi_cov,
                        update_roi_cov_0=update_roi_cov_0,
                        stationary_A_t_=stationary_A_t_,
                        diag_roi_cov=diag_roi_cov,
                        update_sigsq=update_sigsq,
                        tau=tau, c1=c1, verbose=verbose)

            if Atrue is not None:
                for i in range(num_roi):
                    for j in range(num_roi):
                        ax_A_t_[i, j].cla()
                        ax_A_t_[i, j].plot(Atrue[:-1, i, j], color='green')
                        ax_A_t_[i, j].plot(self.A_t_[:-1, i, j], color='red',
                                        alpha=0.7)
                fig_A_t_.tight_layout()
                fig_A_t_.canvas.draw()
                plt.pause(1. / 60.)

            self._e_step(verbose=verbose-1)

            # calculate objective value, check for updated best iterate
            objval = self._em_objective()
            objvals[it] = objval

            if verbose > 0:
                print("  objective: %.4e" % objval)
                At = self.A_t_[:-1]
                maxAt = np.max(np.abs(np.triu(At, k=1) + np.tril(At, k=-1)))
                print("  max |A_t|: %.4e" % (maxAt,))
                sys.stdout.flush()
                sys.stderr.flush()

            if objval < best_objval:
                best_objval = objval
                best_params = (self.A_t_.copy(), self.roi_cov.copy(),
                               self.mu0.copy(), self.roi_cov_0.copy(),
                               [l.copy() for l in self.log_sigsq_lst])

            # check for convergence
            if it >= 1:
                relnormdiff_At = relnormdiff(self.A_t_[:-1], At_prev)
                relnormdiff_roi_cov = relnormdiff(self.roi_cov, roi_cov_prev)
                relnormdiff_roi_cov_0 = relnormdiff(self.roi_cov_0,
                                                    roi_cov_0_prev)
                relnormdiff_log_sigsq_lst = \
                    np.array(
                        [relnormdiff(self.log_sigsq_lst[s],
                                     log_sigsq_lst_prev[s])
                        for s in range(len(self.log_sigsq_lst))])
                params_converged = (relnormdiff_At <= tol) and \
                                   (relnormdiff_roi_cov <= tol) and \
                                   (relnormdiff_roi_cov_0 <= tol) and \
                                   np.all(relnormdiff_log_sigsq_lst <= tol)

                relobjdiff = np.abs((objval - objvals[it-1]) / objval)

                if verbose > 0:
                    print("  relnormdiff_At: %.3e" % relnormdiff_At)
                    print("  relnormdiff_roi_cov: %.3e" % relnormdiff_roi_cov)
                    print("  relnormdiff_roi_cov_0: %.3e" %
                          relnormdiff_roi_cov_0)
                    print("  relnormdiff_log_sigsq_lst:",
                          relnormdiff_log_sigsq_lst)
                    print("  relobjdiff:  %.3e" % relobjdiff)

                    objdiff = objval - objvals[it-1]
                    if objdiff > 0:
                        print("  \033[0;31mEM objective increased\033[0m")

                    sys.stdout.flush()
                    sys.stderr.flush()

                if params_converged or relobjdiff <= tol:
                    if verbose > 0:
                        print("EM objective converged")
                        sys.stdout.flush()
                        sys.stderr.flush()
                    converged = True
                    objvals = objvals[:it+1]
                    break

            # retrieve best parameters and load into instance variables.
            A_t_, roi_cov, mu0, roi_cov_0, log_sigsq_lst = best_params
            self.A_t_ = A_t_.copy()
            self.roi_cov = roi_cov.copy()
            self.mu0 = mu0.copy()
            self.roi_cov_0 = roi_cov_0.copy()
            self.log_sigsq_lst = [l.copy() for l in log_sigsq_lst]

            if verbose > 0:
                print()
                print("elapsed, iteration:", datetime.now() - iter_start)
                print("=" * 34)
                print()

        # perform final smoothing
        mus_smooth_lst = None
        St_lst = None
        if do_final_smoothing:
            if verbose >= 1:
                print("performing final smoothing")

            mus_smooth_lst = list()
            self._loglik = 0.
            if self._store_St:
                St_lst = list()
            for s, sdata in enumerate(self.unpack_all_subject_data()):
                Y, w_s, fwd_roi_snsr, fwd_src_snsr, snsr_cov, which_roi = sdata
                sigsq_vals = np.exp(self.log_sigsq_lst[s])
                R = LDS.R_(snsr_cov, fwd_src_snsr, sigsq_vals, which_roi)
                roi_cov_t = _ensure_ndim(self.roi_cov, self._n_timepts, 3)
                with numpy_num_threads(1):
                    loglik_subject, mus_smooth, _, _, St = \
                        rts_smooth(Y, self.A_t_, fwd_roi_snsr, roi_cov_t, R,
                                   self.mu0, self.roi_cov_0,
                                   compute_lag1_cov=False,
                                   store_St=self._store_St)
                # just save the mean of the smoothed trials
                if average_mus_smooth:
                    mus_smooth_lst.append(np.mean(mus_smooth, axis=0))
                else:
                    mus_smooth_lst.append(mus_smooth)
                self._loglik += loglik_subject
                # just save the diagonals of St b/c that's what we need for
                # connectivity
                if self._store_St:
                    St_lst.append(np.diagonal(St, axis1=-2, axis2=-1))

        if verbose > 0:
            print()
            print("elapsed, function:", datetime.now() - fxn_start)
            print("=" * 34)
            print()

        return objvals, converged, mus_smooth_lst, self._loglik, St_lst

    def _e_step(self, verbose=0):
        n_timepts, num_roi, _ = self.A_t_.shape

        # reset accumulation arrays
        self._B0[:] = 0.
        self._B1[:] = 0.
        self._B3[:] = 0.
        self._B2[:] = 0.

        self._mus_smooth_lst = list()
        self._sigmas_smooth_lst = list()
        self._sigmas_tnt_smooth_lst = list()

        if verbose > 0:
            print("  e-step")
            print("    subject", end="")

        for s, sdata in enumerate(self.unpack_all_subject_data()):

            if verbose > 0:
                print(" %d" % (s+1,), end="")
                sys.stdout.flush()
                sys.stderr.flush()

            Y, w_s, fwd_roi_snsr, fwd_src_snsr, snsr_cov, which_roi = sdata

            sigsq_vals = np.exp(self.log_sigsq_lst[s])
            R = LDS.R_(snsr_cov, fwd_src_snsr, sigsq_vals, which_roi)
            roi_cov_t = _ensure_ndim(self.roi_cov, self._n_timepts, 3)

            with numpy_num_threads(1):
                _, mus_smooth, sigmas_smooth, sigmas_tnt_smooth = \
                        rts_smooth_fast(Y, self.A_t_, fwd_roi_snsr, roi_cov_t,
                                        R, self.mu0, self.roi_cov_0,
                                        compute_lag1_cov=True)

            self._mus_smooth_lst.append(mus_smooth)
            self._sigmas_smooth_lst.append(sigmas_smooth)
            self._sigmas_tnt_smooth_lst.append(sigmas_tnt_smooth)

            x_smooth_0_outer = einsum('ri,rj->rij', mus_smooth[:,0,:num_roi],
                                                     mus_smooth[:,0,:num_roi])
            self._B0 += w_s*np.sum(sigmas_smooth[:,0,:num_roi,:num_roi] +
                                   x_smooth_0_outer, axis=0)

            x_smooth_outer = einsum('rti,rtj->rtij', mus_smooth[:,1:,:num_roi],
                                    mus_smooth[:,1:,:num_roi])
            self._B1 += w_s*np.sum(sigmas_smooth[:,1:,:num_roi,:num_roi] +
                                   x_smooth_outer, axis=0)

            z_smooth_outer = einsum('rti,rtj->rtij', mus_smooth[:,:-1,:],
                                                      mus_smooth[:,:-1,:])
            self._B3 += w_s*np.sum(sigmas_smooth[:,:-1,:,:] + z_smooth_outer,
                                   axis=0)

            mus_smooth_outer_l1 = einsum('rti,rtj->rtij',
                                          mus_smooth[:,1:,:num_roi],
                                          mus_smooth[:,:-1,:])
            self._B2 += w_s*np.sum(sigmas_tnt_smooth[:,:,:num_roi,:] +
                                   mus_smooth_outer_l1, axis=0)

        if verbose > 0:
            print("\n  done")

    def _m_step(self, update_A_t_=True, update_roi_cov=True,
                update_roi_cov_0=True, stationary_A_t_=False,
                diag_roi_cov=False, update_sigsq=True, tau=0.1, c1=1e-4,
        verbose=0):
        self._loglik = None
        if verbose > 0:
            print("  m-step")
        if update_roi_cov_0:
            self.roi_cov_0 = (1. / self._ntrials_all) * self._B0
            if diag_roi_cov:
                self.roi_cov_0 = np.diag(np.diag(self.roi_cov_0))
        self.update_A_t_and_roi_cov(update_A_t_=update_A_t_,
                                    update_roi_cov=update_roi_cov,
                                    stationary_A_t_=stationary_A_t_,
                                    diag_roi_cov=diag_roi_cov, tau=tau,
                                    c1=c1, verbose=verbose)
        if update_sigsq:
            self.update_log_sigsq_lst(verbose=verbose)

    def update_A_t_and_roi_cov(self, update_A_t_=True, update_roi_cov=True,
                               stationary_A_t_=False,
        diag_roi_cov=False, tau=0.1, c1=1e-4, verbose=0):

        if verbose > 1:
            print("    update A_t_ and roi_cov")

        # gradient descent
        At = self.A_t_[:-1]
        At_init = At.copy()
        L_roi_cov = np.linalg.cholesky(self.roi_cov)
        At_L_roi_cov_obj = lambda x, y: self.L2_obj(x, y)
        At_obj = lambda x: self.L2_obj(x, L_roi_cov)
        grad_At_obj = grad(At_obj)
        obj_diff = np.finfo('float').max
        obj = At_L_roi_cov_obj(At, L_roi_cov)
        inner_it = 0

        # specify proximal operator to use
        if self._penalty == 'ridge':
            prox_op = lambda x, y: x
        elif self._penalty == 'lasso':
            prox_op = soft_thresh_At
        elif self._penalty == 'group-lasso':
            prox_op = block_thresh_At

        while np.abs(obj_diff / obj) > self._A_t_roi_cov_tol:

            if inner_it > self._A_t_roi_cov_niter:
                break

            obj_start = At_L_roi_cov_obj(At, L_roi_cov)

            # update At using gradient descent with backtracking line search
            if update_A_t_:
                if stationary_A_t_:
                    B2_sum = np.sum(self._B2, axis=0)
                    B3_sum = np.sum(self._B3, axis=0)
                    At[:] = np.linalg.solve(B3_sum.T, B2_sum.T).T
                else:
                    grad_At = grad_At_obj(At)
                    step_size = linesearch(At_obj, grad_At_obj, At, grad_At,
                                           prox_op=prox_op, lam=self.lam0,
                                           tau=tau, c1=c1)
                    At[:] = prox_op(At - step_size * grad_At,
                                    self.lam0 * step_size)

            # update roi_cov using closed form
            if update_roi_cov:
                AtB2T = einsum('tik,tjk->tij', At, self._B2)
                B2AtT = einsum('tik,tjk->tij', self._B2, At)
                tmp = einsum('tik,tkl->til', At, self._B3)
                AtB3AtT = einsum('til,tjl->tij', tmp, At)
                elbo_2 = np.sum(self._B1 - AtB2T - B2AtT + AtB3AtT, axis=0)
                self.roi_cov = (1. / (self._ntrials_all * self._n_timepts
                                )) * elbo_2
                if diag_roi_cov:
                    self.roi_cov = np.diag(np.diag(self.roi_cov))
                L_roi_cov = np.linalg.cholesky(self.roi_cov)

            obj = At_L_roi_cov_obj(At, L_roi_cov)
            obj_diff = obj_start - obj
            inner_it += 1

        if verbose > 1:
            if not stationary_A_t_ and update_A_t_:
                grad_norm = np.linalg.norm(grad_At)
                norm_change = np.linalg.norm(At - At_init)
                print("      last step size: %.3e" % step_size)
                print("      last gradient norm: %.3e" % grad_norm)
                print("      norm of total change: %.3e" % norm_change)
                print("      number of iterations: %d" % inner_it)
            print("    done")

    def update_log_sigsq_lst(self, verbose=0):

        if verbose > 1:
            print("    update subject log-sigmasq")

        n_timepts, num_roi, _ = self.A_t_.shape

        # update log_sigsq_vals for each subject and ROI
        for s, sdata in enumerate(self.unpack_all_subject_data()):

            Y, w_s, fwd_roi_snsr, fwd_src_snsr, snsr_cov, which_roi = sdata
            ntrials, n_timepts, _ = Y.shape
            B4 = self._B4[s]

            log_sigsq = self.log_sigsq_lst[s].copy()
            log_sigsq_obj = lambda x: \
                LDS.L3_obj(x, snsr_cov, fwd_src_snsr, which_roi, B4,
                              ntrials, n_timepts)
            log_sigsq_val_and_grad = vgrad(log_sigsq_obj)

            options = {'maxiter': 500}
            opt_res = spopt.minimize(log_sigsq_val_and_grad, log_sigsq,
                                     method='L-BFGS-B', jac=True,
                                     options=options)
            if verbose > 1:
                print("      subject %d - %d iterations" % (s+1, opt_res.nit))

            if not opt_res.success:
                print("        log_sigsq opt")
                print("        %s" % opt_res.message)

            self.log_sigsq_lst[s] = opt_res.x

        if verbose > 1:
            print("\n    done")


    @staticmethod
    def R_(snsr_cov, fwd_src_snsr, sigsq_vals, which_roi):
        return snsr_cov + np.dot(fwd_src_snsr,
                                 sigsq_vals[which_roi][:,None]*fwd_src_snsr.T)

    def L2_obj(self, At, L_roi_cov):
        AtB2T = einsum('tik,tjk->tij', At, self._B2)
        B2AtT = einsum('tik,tjk->tij', self._B2, At)
        tmp = einsum('tik,tkl->til', At, self._B3)
        AtB3AtT = einsum('til,tjl->tij', tmp, At)
        elbo_2 = np.sum(self._B1 - AtB2T - B2AtT + AtB3AtT, axis=0)

        L_roi_cov_inv_elbo_2 = solve_triangular(L_roi_cov, elbo_2, lower=True)
        obj = np.trace(solve_triangular(L_roi_cov, L_roi_cov_inv_elbo_2,
                                        lower=True,
                                        trans='T'))
        obj = obj / self._ntrials_all

        if self._penalty == 'ridge':
            obj += self.lam0*np.sum(At**2)
        AtmAtm1_2 = (At[1:] - At[:-1])**2
        obj += self.lam1*np.sum(AtmAtm1_2)

        return obj

    @staticmethod
    def L3_obj(log_sigsq_vals, snsr_cov, fwd_src_snsr, which_roi, B4, ntrials,
               n_timepts):
        R = LDS.R_(snsr_cov, fwd_src_snsr, np.exp(log_sigsq_vals),
                      which_roi)
        try:
            L_R = np.linalg.cholesky(R)
        except LinAlgError:
            return np.finfo('float').max
        L_R_inv_B4 = solve_triangular(L_R, B4, lower=True)
        return (ntrials*n_timepts*2.*np.sum(np.log(np.diag(L_R)))
                + np.trace(solve_triangular(L_R, L_R_inv_B4, lower=True,
                                            trans='T')))
