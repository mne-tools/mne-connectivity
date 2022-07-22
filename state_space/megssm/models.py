import sys

import autograd.numpy as np
import scipy.optimize as spopt

from autograd import grad #autograd --> jax
from autograd import value_and_grad as vgrad
from scipy.linalg import LinAlgError

from .util import _ensure_ndim, rand_stable, rand_psd
from .util import linesearch, soft_thresh_At, block_thresh_At
from .util import relnormdiff
from .message_passing import kalman_filter, rts_smooth, rts_smooth_fast
from .message_passing import predict_step, condition
from .numpy_numthreads import numpy_num_threads

from .mne_util import ROIToSourceMap, _scale_sensor_data, run_pca_on_subject

try:
    from autograd_linalg import solve_triangular
except ImportError:
    raise RuntimeError("must install `autograd_linalg` package")

# einsum2 is a parallel version of einsum that works for two arguments
try:
    from einsum2 import einsum2
except ImportError:
    # rename standard numpy function if don't have einsum2
    print("=> WARNING: using standard numpy.einsum,",
          "consider installing einsum2 package")
    from autograd.numpy import einsum as einsum2

from datetime import datetime


# TODO: add documentation to all methods
class _MEGModel(object):
    """ Base class for any model applied to MEG data that handles storing and
        unpacking data from tuples. """

    def __init__(self):
        self._subjectdata = None
        self._timepts = 0
        self._ntrials_all = 0
        self._nsubjects = 0

    def set_data(self, subjectdata):
        timepts_lst = [self.unpack_subject_data(e)[0].shape[1] for e in subjectdata]
        assert len(list(set(timepts_lst))) == 1
        self._timepts = timepts_lst[0]
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


# TODO: add documentation to all methods
# TODO: make some methods "private" (leading underscore) if necessary
class MEGLDS(_MEGModel):
    """ State-space model for MEG data, as described in "A state-space model of
        cross-region dynamic connectivity in MEG/EEG", Yang et al., NIPS 2016.
    """

    def __init__(self, num_roi, timepts, A_t_=None, roi_cov=None, mu0=None, roi_cov_0=None, 
                 log_sigsq_lst=None, lam0=0., lam1=0., penalty='ridge', 
                 store_St=True):

        super().__init__()

        set_default = \
            lambda prm, val, deflt: \
                self.__setattr__(prm, val.copy() if val is not None else deflt)
        
        # initialize parameters
        set_default("A_t_", A_t_,
                    np.stack([rand_stable(num_roi, maxew=0.7) for _ in range(timepts)],
                              axis=0))
        set_default("roi_cov", roi_cov, rand_psd(num_roi))
        set_default("mu0", mu0, np.zeros(num_roi))
        set_default("roi_cov_0", roi_cov_0, rand_psd(num_roi))
        set_default("log_sigsq_lst", log_sigsq_lst,
                    [np.log(np.random.gamma(2, 1, size=num_roi+1))])

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

        # initialize sufficient statistics
        timepts, num_roi, _ = self.A_t_.shape
        self._B0 = np.zeros((num_roi, num_roi))
        self._B1 = np.zeros((timepts-1, num_roi, num_roi))
        self._B3 = np.zeros((timepts-1, num_roi, num_roi))
        self._B2 = np.zeros((timepts-1, num_roi, num_roi))
        self._B4 = list()
        
        self._subject_data = dict()
        
    def add_subject(self, subject, subject_dir, epochs,labels, fwd, cov):
        roi_to_src = ROIToSourceMap(fwd, labels) # compute ROI-to-source map
        fwd_src_snsr, fwd_roi_snsr, snsr_cov, epochs = \
                _scale_sensor_data(epochs, fwd, cov, roi_to_src)
        
        # cov = cov.pick_channels(epochs.ch_names)
        sdata = run_pca_on_subject(subject, epochs, fwd, cov, labels) #check for channel mismatch
        data, fwd_roi_snsr, fwd_src_snsr, snsr_cov, which_roi = sdata
        subjectdata = [(data, fwd_roi_snsr, fwd_src_snsr, snsr_cov, which_roi)]
        
        self.set_data(subjectdata)
        
        # epochs, fwd_roi_snsr, fwd_src_snsr, snsr_cov, which_roi = subjectdata
        self._subject_data[subject] = dict()
        self._subject_data[subject]['epochs'] = epochs
        self._subject_data[subject]['fwd_src_snsr'] = fwd_src_snsr
        self._subject_data[subject]['fwd_roi_snsr'] = fwd_roi_snsr
        self._subject_data[subject]['snsr_cov'] = snsr_cov
        self._subject_data[subject]['labels'] = labels
        self._subject_data[subject]['which_roi'] = which_roi

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

    # TODO: figure out how to initialize smoothed parameters so this doesn't
    #       break, e.g. if _em_objective is called before em for some reason
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

            ntrials, timepts, _ = Y.shape

            sigsq_vals = np.exp(self.log_sigsq_lst[s])
            R = MEGLDS.R_(snsr_cov, fwd_src_snsr, sigsq_vals, which_roi)
            L_R = np.linalg.cholesky(R)

            if (self._mus_smooth_lst is None or self._sigmas_smooth_lst is None
                or self._sigmas_tnt_smooth_lst is None):
                roi_cov_t = _ensure_ndim(self.roi_cov, timepts, 3)
                with numpy_num_threads(1):
                    _, mus_smooth, sigmas_smooth, sigmas_tnt_smooth = \
                            rts_smooth_fast(Y, self.A_t_, fwd_roi_snsr, roi_cov_t, R, self.mu0,
                                            self.roi_cov_0, compute_lag1_cov=True)

            else:
                mus_smooth = self._mus_smooth_lst[s]
                sigmas_smooth = self._sigmas_smooth_lst[s]
                sigmas_tnt_smooth = self._sigmas_tnt_smooth_lst[s]

            x_smooth_0_outer = einsum2('ri,rj->rij', mus_smooth[:,0,:num_roi],
                                                     mus_smooth[:,0,:num_roi])
            B0 = w_s*np.sum(sigmas_smooth[:,0,:num_roi,:num_roi] + x_smooth_0_outer,
                                  axis=0)

            x_smooth_outer = einsum2('rti,rtj->rtij', mus_smooth[:,1:,:num_roi],
                                                      mus_smooth[:,1:,:num_roi])
            B1 = w_s*np.sum(sigmas_smooth[:,1:,:num_roi,:num_roi] + x_smooth_outer, axis=0)

            z_smooth_outer = einsum2('rti,rtj->rtij', mus_smooth[:,:-1,:],
                                                      mus_smooth[:,:-1,:])
            B3 = w_s*np.sum(sigmas_smooth[:,:-1,:,:] + z_smooth_outer, axis=0)

            mus_smooth_outer_l1 = einsum2('rti,rtj->rtij',
                                          mus_smooth[:,1:,:num_roi],
                                          mus_smooth[:,:-1,:])
            B2 = w_s*np.sum(sigmas_tnt_smooth[:,:,:num_roi,:] + mus_smooth_outer_l1,
                            axis=0)

            # obj += L1(roi_cov_0)
            L_roi_cov_0_inv_B0 = solve_triangular(L_roi_cov_0, B0, lower=True)
            L1 += (ntrials*2.*np.sum(np.log(np.diag(L_roi_cov_0)))
                   + np.trace(solve_triangular(L_roi_cov_0, L_roi_cov_0_inv_B0, lower=True,
                                               trans='T')))

            At = self.A_t_[:-1]
            AtB2T = einsum2('tik,tjk->tij', At, B2)
            B2AtT = einsum2('tik,tjk->tij', B2, At)
            tmp = einsum2('tik,tkl->til', At, B3)
            AtB3AtT = einsum2('tik,tjk->tij', tmp, At)

            tmp = np.sum(B1 - AtB2T - B2AtT + AtB3AtT, axis=0)

            # obj += L2(roi_cov, At)
            L_roi_cov_inv_tmp = solve_triangular(L_roi_cov, tmp, lower=True)
            L2 += (ntrials*(timepts-1)*2.*np.sum(np.log(np.diag(L_roi_cov)))
                   + np.trace(solve_triangular(L_roi_cov, L_roi_cov_inv_tmp, lower=True,
                                               trans='T')))

            res = Y - einsum2('ik,ntk->nti', fwd_roi_snsr, mus_smooth[:,:,:num_roi])
            CP_smooth = einsum2('ik,ntkj->ntij', fwd_roi_snsr, sigmas_smooth[:,:,:num_roi,:num_roi])

            # TODO: np.sum does not parallelize over the accumulators, possible
            # bottleneck.
            B4 = w_s*(np.sum(einsum2('nti,ntj->ntij', res, res), axis=(0,1))
                             + np.sum(einsum2('ntik,jk->ntij', CP_smooth, fwd_roi_snsr),
                                    axis=(0,1)))
            self._B4[s] = B4

            # obj += L3(sigsq_vals)
            L_R_inv_B4 = solve_triangular(L_R, B4, lower=True)
            L3 += (ntrials*timepts*2*np.sum(np.log(np.diag(L_R)))
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

    def fit(self, niter=100, tol=1e-6, A_t_roi_cov_niter=100, A_t_roi_cov_tol=1e-6, verbose=0,
           update_A_t_=True, update_roi_cov=True, update_roi_cov_0=True, stationary_A_t_=False,
           diag_roi_cov=False, update_sigsq=True, do_final_smoothing=True,
           average_mus_smooth=True, Atrue=None, tau=0.1, c1=1e-4):

        fxn_start = datetime.now()
                
        timepts, num_roi, _ = self.A_t_.shape

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
                       self.roi_cov_0.copy(), [l.copy() for l in self.log_sigsq_lst])

        # previous parameter values (for checking convergence)
        At_prev = None
        roi_cov_prev = None
        roi_cov_0_prev = None
        log_sigsq_lst_prev = None

        if Atrue is not None:
            import matplotlib.pyplot as plt
            fig_A_t_, ax_A_t_ = plt.subplots(num_roi, num_roi, sharex=True, sharey=True)
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

            self._m_step(update_A_t_=update_A_t_, update_roi_cov=update_roi_cov,
                        update_roi_cov_0=update_roi_cov_0, stationary_A_t_=stationary_A_t_,
                        diag_roi_cov=diag_roi_cov, update_sigsq=update_sigsq,
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
                best_params = (self.A_t_.copy(), self.roi_cov.copy(), self.mu0.copy(),
                               self.roi_cov_0.copy(),
                               [l.copy() for l in self.log_sigsq_lst])

            # check for convergence
            if it >= 1:
                relnormdiff_At = relnormdiff(self.A_t_[:-1], At_prev)
                relnormdiff_roi_cov = relnormdiff(self.roi_cov, roi_cov_prev)
                relnormdiff_roi_cov_0 = relnormdiff(self.roi_cov_0, roi_cov_0_prev)
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
                    print("  relnormdiff_roi_cov_0: %.3e" % relnormdiff_roi_cov_0)
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
                R = MEGLDS.R_(snsr_cov, fwd_src_snsr, sigsq_vals, which_roi)
                roi_cov_t = _ensure_ndim(self.roi_cov, self._timepts, 3)
                with numpy_num_threads(1):
                    loglik_subject, mus_smooth, _, _, St = \
                        rts_smooth(Y, self.A_t_, fwd_roi_snsr, roi_cov_t, R, self.mu0, self.roi_cov_0,
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

        timepts, num_roi, _ = self.A_t_.shape

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
            R = MEGLDS.R_(snsr_cov, fwd_src_snsr, sigsq_vals, which_roi)
            L_R = np.linalg.cholesky(R)
            roi_cov_t = _ensure_ndim(self.roi_cov, self._timepts, 3)

            with numpy_num_threads(1):
                _, mus_smooth, sigmas_smooth, sigmas_tnt_smooth = \
                        rts_smooth_fast(Y, self.A_t_, fwd_roi_snsr, roi_cov_t, R, self.mu0,
                                        self.roi_cov_0, compute_lag1_cov=True)

            self._mus_smooth_lst.append(mus_smooth)
            self._sigmas_smooth_lst.append(sigmas_smooth)
            self._sigmas_tnt_smooth_lst.append(sigmas_tnt_smooth)

            x_smooth_0_outer = einsum2('ri,rj->rij', mus_smooth[:,0,:num_roi],
                                                     mus_smooth[:,0,:num_roi])
            self._B0 += w_s*np.sum(sigmas_smooth[:,0,:num_roi,:num_roi] + x_smooth_0_outer,
                                   axis=0)

            x_smooth_outer = einsum2('rti,rtj->rtij', mus_smooth[:,1:,:num_roi],
                                                      mus_smooth[:,1:,:num_roi])
            self._B1 += w_s*np.sum(sigmas_smooth[:,1:,:num_roi,:num_roi] + x_smooth_outer,
                                   axis=0)

            z_smooth_outer = einsum2('rti,rtj->rtij', mus_smooth[:,:-1,:],
                                                      mus_smooth[:,:-1,:])
            self._B3 += w_s*np.sum(sigmas_smooth[:,:-1,:,:] + z_smooth_outer,
                                   axis=0)

            mus_smooth_outer_l1 = einsum2('rti,rtj->rtij',
                                          mus_smooth[:,1:,:num_roi],
                                          mus_smooth[:,:-1,:])
            self._B2 += w_s*np.sum(sigmas_tnt_smooth[:,:,:num_roi,:] +
                                   mus_smooth_outer_l1, axis=0)

        if verbose > 0:
            print("\n  done")

    def _m_step(self, update_A_t_=True, update_roi_cov=True, update_roi_cov_0=True,
        stationary_A_t_=False, diag_roi_cov=False, update_sigsq=True, tau=0.1, c1=1e-4,
        verbose=0):
        self._loglik = None
        if verbose > 0:
            print("  m-step")
        if update_roi_cov_0:
            self.roi_cov_0 = (1. / self._ntrials_all) * self._B0
            if diag_roi_cov:
                self.roi_cov_0 = np.diag(np.diag(self.roi_cov_0))
        self.update_A_t_and_roi_cov(update_A_t_=update_A_t_, update_roi_cov=update_roi_cov,
                            stationary_A_t_=stationary_A_t_, diag_roi_cov=diag_roi_cov,
                            tau=tau, c1=c1, verbose=verbose)
        if update_sigsq:
            self.update_log_sigsq_lst(verbose=verbose)

    def update_A_t_and_roi_cov(self, update_A_t_=True, update_roi_cov=True, stationary_A_t_=False,
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
                AtB2T = einsum2('tik,tjk->tij', At, self._B2)
                B2AtT = einsum2('tik,tjk->tij', self._B2, At)
                tmp = einsum2('tik,tkl->til', At, self._B3)
                AtB3AtT = einsum2('til,tjl->tij', tmp, At)
                elbo_2 = np.sum(self._B1 - AtB2T - B2AtT + AtB3AtT, axis=0)
                self.roi_cov = (1. / (self._ntrials_all * self._timepts
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

        timepts, num_roi, _ = self.A_t_.shape

        # update log_sigsq_vals for each subject and ROI
        for s, sdata in enumerate(self.unpack_all_subject_data()):

            Y, w_s, fwd_roi_snsr, fwd_src_snsr, snsr_cov, which_roi = sdata
            ntrials, timepts, _ = Y.shape
            mus_smooth = self._mus_smooth_lst[s]
            sigmas_smooth = self._sigmas_smooth_lst[s]
            B4 = self._B4[s]

            log_sigsq = self.log_sigsq_lst[s].copy()
            log_sigsq_obj = lambda x: \
                MEGLDS.L3_obj(x, snsr_cov, fwd_src_snsr, which_roi, B4, ntrials, timepts)
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

    def calculate_smoothed_estimates(self):
        """ recalculate smoothed estimates with current model parameters """

        self._mus_smooth_lst = list()
        self._sigmas_smooth_lst = list()
        self._sigmas_tnt_smooth_lst = list()
        self._St_lst = list()
        self._loglik = 0.

        for s, sdata in enumerate(self.unpack_all_subject_data()):
            Y, w_s, fwd_roi_snsr, fwd_src_snsr, snsr_cov, which_roi = sdata
            sigsq_vals = np.exp(self.log_sigsq_lst[s])
            R = MEGLDS.R_(snsr_cov, fwd_src_snsr, sigsq_vals, which_roi)
            roi_cov_t = _ensure_ndim(self.roi_cov, self._timepts, 3)
            with numpy_num_threads(1):
                ll, mus_smooth, sigmas_smooth, sigmas_tnt_smooth, _ = \
                    rts_smooth(Y, self.A_t_, fwd_roi_snsr, roi_cov_t, R, self.mu0, self.roi_cov_0,
                               compute_lag1_cov=True, store_St=False)
            self._mus_smooth_lst.append(mus_smooth)
            self._sigmas_smooth_lst.append(sigmas_smooth)
            self._sigmas_tnt_smooth_lst.append(sigmas_tnt_smooth)
            #self._St_lst.append(np.diagonal(St, axis1=-2, axis2=-1))
            self._loglik += ll

    def log_likelihood(self):
        """ calculate log marginal likelihood using the Kalman filter """

        #if (self._mus_smooth_lst is None or self._sigmas_smooth_lst is None \
        #    or self._sigmas_tnt_smooth_lst is None):
        #    self.calculate_smoothed_estimates()
        #    return self._loglik
        if self._loglik is not None:
            return self._loglik

        self._loglik = 0.
        for s, sdata in enumerate(self.unpack_all_subject_data()):
            Y, w_s, fwd_roi_snsr, fwd_src_snsr, snsr_cov, which_roi = sdata
            sigsq_vals = np.exp(self.log_sigsq_lst[s])
            R = MEGLDS.R_(snsr_cov, fwd_src_snsr, sigsq_vals, which_roi)
            roi_cov_t = _ensure_ndim(self.roi_cov, self._timepts, 3)
            ll, _, _, _ = kalman_filter(Y, self.A_t_, fwd_roi_snsr, roi_cov_t, R, self.mu0,
                                        self.roi_cov_0, store_St=False)
            self._loglik += ll

        return self._loglik

    def nparams(self):
        timepts, p, _ = self.A_t_.shape

        # this should equal (timepts-1)*p*p unless some shrinkage is used on At
        nparams_At = np.sum(np.abs(self.A_t_[:-1]) > 0)

        # nparams = nparams(At) + nparams(roi_cov) + nparams(roi_cov_0)
        #           + nparams(log_sigsq_lst)
        return nparams_At + p*(p+1)/2 + p*(p+1)/2 \
               + np.sum([p+1 for _ in range(len(self.log_sigsq_lst))])

    def AIC(self):
        return -2*self.log_likelihood() + 2*self.nparams()

    def BIC(self):
        if self._ntrials_all == 0:
            raise RuntimeError("use set_data to add subject data before" \
                               + " computing BIC")
        return -2*self.log_likelihood() \
               + np.log(self._ntrials_all)*self.nparams()

    def save(self, filename, **kwargs):
        savedict = { 'A_t_' : self.A_t_, 'roi_cov' : self.roi_cov, 'mu0' : self.mu0,
                     'roi_cov_0' : self.roi_cov_0, 'log_sigsq_lst' : self.log_sigsq_lst,
                     'lam0' : self.lam0, 'lam1' : self.lam1}
        savedict.update(kwargs)
        np.savez_compressed(filename, **savedict)

    def load(self, filename):
        loaddict = np.load(filename)
        param_names = ['A_t_', 'roi_cov', 'mu0', 'roi_cov_0', 'log_sigsq_lst', 'lam0', 'lam1']
        for name in param_names:
            if name not in loaddict.keys():
                raise RuntimeError('specified file is not a saved model:\n%s'
                                   % (filename,))
        for name in param_names:
            if name == 'log_sigsq_lst':
                self.log_sigsq_lst = [l.copy() for l in loaddict[name]]
            elif name in ('lam0', 'lam1'):
                self.__setattr__(name, float(loaddict[name]))
            else:
                self.__setattr__(name, loaddict[name].copy())

        # return remaining saved items, if there are any
        others = {key : loaddict[key] for key in loaddict.keys() \
                                      if key not in param_names}
        if len(others.keys()) > 0:
            return others

    @staticmethod
    def R_(snsr_cov, fwd_src_snsr, sigsq_vals, which_roi):
        return snsr_cov + np.dot(fwd_src_snsr, sigsq_vals[which_roi][:,None]*fwd_src_snsr.T)

    def L2_obj(self, At, L_roi_cov):
        
        # import autograd.numpy
        # if isinstance(At,autograd.numpy.numpy_boxes.ArrayBox):
        #     At = At._value
            
        AtB2T = einsum2('tik,tjk->tij', At, self._B2)
        B2AtT = einsum2('tik,tjk->tij', self._B2, At)
        tmp = einsum2('tik,tkl->til', At, self._B3)
        AtB3AtT = einsum2('til,tjl->tij', tmp, At)
        elbo_2 = np.sum(self._B1 - AtB2T - B2AtT + AtB3AtT, axis=0)

        L_roi_cov_inv_elbo_2 = solve_triangular(L_roi_cov, elbo_2, lower=True)
        obj = np.trace(solve_triangular(L_roi_cov, L_roi_cov_inv_elbo_2, lower=True,
                                        trans='T'))
        obj = obj / self._ntrials_all

        if self._penalty == 'ridge':
            obj += self.lam0*np.sum(At**2)
        AtmAtm1_2 = (At[1:] - At[:-1])**2
        obj += self.lam1*np.sum(AtmAtm1_2)

        return obj

    # TODO: convert to instance method
    @staticmethod
    def L3_obj(log_sigsq_vals, snsr_cov, fwd_src_snsr, which_roi, B4, ntrials, timepts):
        R = MEGLDS.R_(snsr_cov, fwd_src_snsr, np.exp(log_sigsq_vals), which_roi)
        try:
            L_R = np.linalg.cholesky(R)
        except LinAlgError:
            return np.finfo('float').max
        L_R_inv_B4 = solve_triangular(L_R, B4, lower=True)
        return (ntrials*timepts*2.*np.sum(np.log(np.diag(L_R)))
                + np.trace(solve_triangular(L_R, L_R_inv_B4, lower=True,
                                            trans='T')))


    # @property
    # def A(self):
    #     return self._A

    # @A.setter
    # def A(self, A):
    #     self._A = A

    # @property
    # def roi_cov(self):
    #     return self._roi_cov

    # @roi_cov.setter
    # def roi_cov(self, roi_cov):
    #     self._roi_cov = roi_cov

    # @property
    # def mu0(self):
    #     return self._mu0

    # @mu0.setter
    # def mu0(self, mu0):
    #     self._mu0 = mu0

    # @property
    # def roi_cov_0(self):
    #     return self._roi_cov_0

    # @roi_cov_0.setter
    # def roi_cov_0(self, roi_cov_0):
    #     self._roi_cov_0 = roi_cov_0

    # @property
    # def log_sigsq_lst(self):
    #     return self._log_sigsq_lst

    # @log_sigsq_lst.setter
    # def log_sigsq_lst(self, log_sigsq_lst):
    #     self._log_sigsq_lst = log_sigsq_lst

    # @property
    # def num_roi(self):
    #     return self.A.shape[1]

    # @property
    # def timepts(self):
    #     return self._timepts

    # @property
    # def lam0(self):
    #     return self._lam0

    # @lam0.setter
    # def lam0(self, lam0):
    #     self._lam0 = lam0

    # @property
    # def lam1(self):
    #     return self._lam1

    # @lam1.setter
    # def lam1(self, lam1):
    #     self._lam1 = lam1
