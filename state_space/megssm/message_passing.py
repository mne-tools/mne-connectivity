from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import autograd.numpy as np
from autograd.scipy.linalg import block_diag

from .util import T_, sym, dot3, _ensure_ndim, component_matrix, hs

from scipy.linalg import solve_triangular as _solve_triangular


def solve_triangular(L, b, *, lower=True, trans=0):
    if hasattr(L, '_value'):  # autograd ArrayBox
        L = L._value
    if hasattr(b, '_value'):
        b = b._value
    if L.ndim == 3 and b.ndim in (2, 3):
        return np.array([
          _solve_triangular(LL, bb, lower=lower, trans=trans)
          for LL, bb in zip(L, b)], L.dtype)
    elif L.ndim == 2 and b.ndim in (2, 1):
        return _solve_triangular(L, b, lower=lower, trans=trans)
    raise RuntimeError(f'Unknown shapes {L.shape} and {b.shape}')



# einsum2 is a parallel version of einsum that works for two arguments
try:
    from einsum2 import einsum2
except ImportError:
    # rename standard numpy function if don't have einsum2
    print("=> WARNING: using standard numpy.einsum,",
          "consider installing einsum2 package")
    from numpy import einsum as einsum2


def kalman_filter(Y, A, C, Q, R, mu0, Q0, store_St=True, sum_logliks=True):
    """ Kalman filter that broadcasts over the first dimension.
        Handles multiple lag dependence using component form.

        Note: This function doesn't handle control inputs (yet).

        Y : ndarray, shape=(N, T, D)
          Observations

        A : ndarray, shape=(T, D*nlag, D*nlag)
          Time-varying dynamics matrices

        C : ndarray, shape=(p, D)
          Observation matrix

        mu0: ndarray, shape=(D,)
          mean of initial state variable

        Q0 : ndarray, shape=(D, D)
          Covariance of initial state variable

        Q : ndarray, shape=(D, D)
          Covariance of latent states

        R : ndarray, shape=(D, D)
          Covariance of observations
    """

    N = Y.shape[0]
    T, D, Dnlags = A.shape
    nlags = Dnlags // D
    AA = np.stack([component_matrix(At, nlags) for At in A], axis=0)

    p = C.shape[0]
    CC = hs([C, np.zeros((p, D*(nlags-1)))])

    QQ = np.zeros((T, Dnlags, Dnlags))
    QQ[:,:D,:D] = Q

    QQ0 = block_diag(*[Q0 for _ in range(nlags)])

    mu_predict = np.stack([np.tile(mu0, nlags) for _ in range(N)], axis=0)
    sigma_predict = np.stack([QQ0 for _ in range(N)], axis=0)

    St = np.empty((N, T, p, p)) if store_St else None

    mus_filt = np.zeros((N, T, Dnlags))
    sigmas_filt = np.zeros((N, T, Dnlags, Dnlags))

    ll = np.zeros(T)

    for t in range(T):

        # condition
        # dot3(CC, sigma_predict, CC.T) + R
        tmp1 = einsum2('ik,nkj->nij', CC, sigma_predict)
        sigma_pred = np.dot(tmp1, CC.T) + R
        sigma_pred = sym(sigma_pred)

        if St is not None:
            St[...,t,:,:] = sigma_pred

        res = Y[...,t,:] - np.dot(mu_predict, CC.T)

        L = np.linalg.cholesky(sigma_pred)
        v = solve_triangular(L, res, lower=True)

        # log-likelihood over all trials
        ll[t] = -0.5*(2.*np.sum(np.log(np.diagonal(L, axis1=1, axis2=2)))
                      + np.sum(v*v)
                      + N*p*np.log(2.*np.pi))

        mus_filt[...,t,:] = mu_predict + einsum2('nki,nk->ni', tmp1,
                                                               solve_triangular(L, v, 'T', lower=True))

        tmp2 = solve_triangular(L, tmp1, lower=True)
        sigmas_filt[...,t,:,:] = sym(sigma_predict - einsum2('nki,nkj->nij', tmp2, tmp2))

        # prediction
        mu_predict = einsum2('ik,nk->ni', AA[t], mus_filt[...,t,:])

        sigma_predict = einsum2('ik,nkl->nil', AA[t], sigmas_filt[...,t,:,:])
        sigma_predict = sym(einsum2('nil,jl->nij', sigma_predict, AA[t]) + QQ[t])

    if sum_logliks:
        ll = np.sum(ll)
    return ll, mus_filt, sigmas_filt, St


def rts_smooth(Y, A, C, Q, R, mu0, Q0, compute_lag1_cov=False,
               store_St=True):
    """ RTS smoother that broadcasts over the first dimension.
        Handles multiple lag dependence using component form.

        Note: This function doesn't handle control inputs (yet).

        Y : ndarray, shape=(N, T, D)
          Observations

        A : ndarray, shape=(T, D*nlag, D*nlag)
          Time-varying dynamics matrices

        C : ndarray, shape=(p, D)
          Observation matrix

        mu0: ndarray, shape=(D,)
          mean of initial state variable

        Q0 : ndarray, shape=(D, D)
          Covariance of initial state variable

        Q : ndarray, shape=(D, D)
          Covariance of latent states

        R : ndarray, shape=(D, D)
          Covariance of observations
    """

    N, T, _ = Y.shape
    _, D, Dnlags = A.shape
    nlags = Dnlags // D
    AA = np.stack([component_matrix(At, nlags) for At in A], axis=0)

    p = C.shape[0]
    CC = hs([C, np.zeros((p, D*(nlags-1)))])

    QQ = np.zeros((T, Dnlags, Dnlags))
    QQ[:,:D,:D] = Q

    QQ0 = block_diag(*[Q0 for _ in range(nlags)])

    mu_predict = np.empty((N, T+1, Dnlags))
    sigma_predict = np.empty((N, T+1, Dnlags, Dnlags))

    mus_smooth = np.empty((N, T, Dnlags))
    sigmas_smooth = np.empty((N, T, Dnlags, Dnlags))

    St = np.empty((N, T, p, p)) if store_St else None

    if compute_lag1_cov:
        sigmas_smooth_tnt = np.empty((N, T-1, Dnlags, Dnlags))
    else:
        sigmas_smooth_tnt = None

    ll = 0.
    mu_predict[:,0,:] = np.tile(mu0, nlags)
    sigma_predict[:,0,:,:] = QQ0.copy()

    for t in range(T):

        # condition
        # sigma_x = dot3(C, sigma_predict, C.T) + R
        tmp1 = einsum2('ik,nkj->nij', CC, sigma_predict[:,t,:,:])
        sigma_x = einsum2('nik,jk->nij', tmp1, CC) + R
        sigma_x = sym(sigma_x)

        if St is not None:
            St[...,t,:,:] = sigma_x

        L = np.linalg.cholesky(sigma_x)
        # res[n] = Y[n,t,:] = np.dot(C, mu_predict[n,t,:])
        res = Y[...,t,:] - einsum2('ik,nk->ni', CC, mu_predict[...,t,:])
        v = solve_triangular(L, res, lower=True)

        # log-likelihood over all trials
        ll += -0.5*(2.*np.sum(np.log(np.diagonal(L, axis1=1, axis2=2)))
                    + np.sum(v*v)
                    + N*p*np.log(2.*np.pi))

        mus_smooth[:,t,:] = mu_predict[:,t,:] + einsum2('nki,nk->ni',
                                                        tmp1,
                                                        solve_triangular(L, v, trans='T', lower=True))

        # tmp2 = L^{-1}*C*sigma_predict
        tmp2 = solve_triangular(L, tmp1, lower=True)
        sigmas_smooth[:,t,:,:] = sym(sigma_predict[:,t,:,:] - einsum2('nki,nkj->nij', tmp2, tmp2))

        # prediction
        #mu_predict = np.dot(A[t], mus_smooth[t])
        mu_predict[:,t+1,:] = einsum2('ik,nk->ni', AA[t], mus_smooth[:,t,:])

        #sigma_predict = dot3(A[t], sigmas_smooth[t], A[t].T) + Q[t]
        tmp = einsum2('ik,nkl->nil', AA[t], sigmas_smooth[:,t,:,:])
        sigma_predict[:,t+1,:,:] = sym(einsum2('nil,jl->nij', tmp, AA[t]) + QQ[t])

    for t in range(T-2, -1, -1):

        # these names are stolen from mattjj and slinderman
        #temp_nn = np.dot(A[t], sigmas_smooth[n,t,:,:])
        temp_nn = einsum2('ik,nkj->nij', AA[t], sigmas_smooth[:,t,:,:])

        L = np.linalg.cholesky(sigma_predict[:,t+1,:,:])
        v = solve_triangular(L, temp_nn, lower=True)
        # Look in Saarka for dfn of Gt_T
        Gt_T = solve_triangular(L, v, trans='T', lower=True)

        # {mus,sigmas}_smooth[n,t] contain the filtered estimates so we're
        # overwriting them on purpose
        #mus_smooth[n,t,:] = mus_smooth[n,t,:] + np.dot(T_(Gt_T), mus_smooth[n,t+1,:] - mu_predict[t+1,:])
        mus_smooth[:,t,:] = mus_smooth[:,t,:] + einsum2('nki,nk->ni', Gt_T, mus_smooth[:,t+1,:] - mu_predict[:,t+1,:])

        #sigmas_smooth[n,t,:,:] = sigmas_smooth[n,t,:,:] + dot3(T_(Gt_T), sigmas_smooth[n,t+1,:,:] - temp_nn, Gt_T)
        tmp = einsum2('nki,nkj->nij', Gt_T, sigmas_smooth[:,t+1,:,:] - sigma_predict[:,t+1,:,:])
        tmp = einsum2('nik,nkj->nij', tmp, Gt_T)
        sigmas_smooth[:,t,:,:] = sym(sigmas_smooth[:,t,:,:] + tmp)

        if compute_lag1_cov:
            # This matrix is NOT symmetric, so don't symmetrize!
            #sigmas_smooth_tnt[n,t,:,:] = np.dot(sigmas_smooth[n,t+1,:,:], Gt_T)
            sigmas_smooth_tnt[:,t,:,:] = einsum2('nik,nkj->nij', sigmas_smooth[:,t+1,:,:], Gt_T)

    return ll, mus_smooth, sigmas_smooth, sigmas_smooth_tnt, St


def rts_smooth_fast(Y, A, C, Q, R, mu0, Q0, compute_lag1_cov=False):
    """ RTS smoother that broadcasts over the first dimension.
        Handles multiple lag dependence using component form.

        Note: This function doesn't handle control inputs (yet).

        Y : ndarray, shape=(N, T, D)
          Observations

        A : ndarray, shape=(T, D*nlag, D*nlag)
          Time-varying dynamics matrices

        C : ndarray, shape=(p, D)
          Observation matrix

        mu0: ndarray, shape=(D,)
          mean of initial state variable

        Q0 : ndarray, shape=(D, D)
          Covariance of initial state variable

        Q : ndarray, shape=(D, D)
          Covariance of latent states

        R : ndarray, shape=(D, D)
          Covariance of observations
    """

    N, T, _ = Y.shape
    _, D, Dnlags = A.shape
    nlags = Dnlags // D
    AA = np.stack([component_matrix(At, nlags) for At in A], axis=0)

    L_R = np.linalg.cholesky(R)

    p = C.shape[0]
    CC = hs([C, np.zeros((p, D*(nlags-1)))])
    tmp = solve_triangular(L_R, CC, lower=True)
    Rinv_CC = solve_triangular(L_R, tmp, trans='T', lower=True)
    CCT_Rinv_CC = einsum2('ki,kj->ij', CC, Rinv_CC)

    # tile L_R across number of trials so solve_triangular
    # can broadcast over trials properly
    L_R = np.tile(L_R, (N, 1, 1))

    QQ = np.zeros((T, Dnlags, Dnlags))
    QQ[:,:D,:D] = Q

    QQ0 = block_diag(*[Q0 for _ in range(nlags)])

    mu_predict = np.empty((N, T+1, Dnlags))
    sigma_predict = np.empty((N, T+1, Dnlags, Dnlags))

    mus_smooth = np.empty((N, T, Dnlags))
    sigmas_smooth = np.empty((N, T, Dnlags, Dnlags))

    if compute_lag1_cov:
        sigmas_smooth_tnt = np.empty((N, T-1, Dnlags, Dnlags))
    else:
        sigmas_smooth_tnt = None

    ll = 0.
    mu_predict[:,0,:] = np.tile(mu0, nlags)
    sigma_predict[:,0,:,:] = QQ0.copy()

    I_tiled = np.tile(np.eye(Dnlags), (N, 1, 1))

    for t in range(T):

        # condition
        tmp1 = einsum2('ik,nkj->nij', CC, sigma_predict[:,t,:,:])

        res = Y[...,t,:] - einsum2('ik,nk->ni', CC, mu_predict[...,t,:])

        # Rinv * res
        tmp2 = solve_triangular(L_R, res, lower=True)
        tmp2 = solve_triangular(L_R, tmp2, trans='T', lower=True)

        # C^T Rinv * res
        tmp3 = einsum2('ki,nk->ni', Rinv_CC, res)

        # (Pinv + C^T Rinv C)_inv * tmp3
        L_P = np.linalg.cholesky(sigma_predict[:,t,:,:])
        tmp = solve_triangular(L_P, I_tiled, lower=True)
        Pinv = solve_triangular(L_P, tmp, trans='T', lower=True)
        tmp4 = sym(Pinv + CCT_Rinv_CC)
        L_tmp4 = np.linalg.cholesky(tmp4)
        tmp3 = solve_triangular(L_tmp4, tmp3, lower=True)
        tmp3 = solve_triangular(L_tmp4, tmp3, trans='T', lower=True)

        # Rinv C * tmp3
        tmp3 = einsum2('ik,nk->ni', Rinv_CC, tmp3)

        # add the two Woodbury * res terms together
        tmp = tmp2 - tmp3

        mus_smooth[:,t,:] = mu_predict[:,t,:] + einsum2('nki,nk->ni', tmp1, tmp)

        # Rinv * tmp1
        tmp2 = solve_triangular(L_R, tmp1, lower=True)
        tmp2 = solve_triangular(L_R, tmp2, trans='T', lower=True)

        # C^T Rinv * tmp1
        tmp3 = einsum2('ki,nkj->nij', Rinv_CC, tmp1)

        # (Pinv + C^T Rinv C)_inv * tmp3
        tmp3 = solve_triangular(L_tmp4, tmp3, lower=True)
        tmp3 = solve_triangular(L_tmp4, tmp3, trans='T', lower=True)

        # Rinv C * tmp3
        tmp3 = einsum2('ik,nkj->nij', Rinv_CC, tmp3)

        # add the two Woodbury * tmp1 terms together, left-multiply by tmp1
        tmp = einsum2('nki,nkj->nij', tmp1, tmp2 - tmp3)

        sigmas_smooth[:,t,:,:] = sym(sigma_predict[:,t,:,:] - tmp)

        # prediction
        mu_predict[:,t+1,:] = einsum2('ik,nk->ni', AA[t], mus_smooth[:,t,:])

        #sigma_predict = dot3(A[t], sigmas_smooth[t], A[t].T) + Q[t]
        tmp = einsum2('ik,nkl->nil', AA[t], sigmas_smooth[:,t,:,:])
        sigma_predict[:,t+1,:,:] = sym(einsum2('nil,jl->nij', tmp, AA[t]) + QQ[t])

    for t in range(T-2, -1, -1):

        # these names are stolen from mattjj and slinderman
        #temp_nn = np.dot(A[t], sigmas_smooth[n,t,:,:])
        temp_nn = einsum2('ik,nkj->nij', AA[t], sigmas_smooth[:,t,:,:])

        L = np.linalg.cholesky(sigma_predict[:,t+1,:,:])
        v = solve_triangular(L, temp_nn, lower=True)
        # Look in Saarka for dfn of Gt_T
        Gt_T = solve_triangular(L, v, trans='T', lower=True)

        # {mus,sigmas}_smooth[n,t] contain the filtered estimates so we're
        # overwriting them on purpose
        #mus_smooth[n,t,:] = mus_smooth[n,t,:] + np.dot(T_(Gt_T), mus_smooth[n,t+1,:] - mu_predict[t+1,:])
        mus_smooth[:,t,:] = mus_smooth[:,t,:] + einsum2('nki,nk->ni', Gt_T, mus_smooth[:,t+1,:] - mu_predict[:,t+1,:])

        #sigmas_smooth[n,t,:,:] = sigmas_smooth[n,t,:,:] + dot3(T_(Gt_T), sigmas_smooth[n,t+1,:,:] - temp_nn, Gt_T)
        tmp = einsum2('nki,nkj->nij', Gt_T, sigmas_smooth[:,t+1,:,:] - sigma_predict[:,t+1,:,:])
        tmp = einsum2('nik,nkj->nij', tmp, Gt_T)
        sigmas_smooth[:,t,:,:] = sym(sigmas_smooth[:,t,:,:] + tmp)

        if compute_lag1_cov:
            # This matrix is NOT symmetric, so don't symmetrize!
            #sigmas_smooth_tnt[n,t,:,:] = np.dot(sigmas_smooth[n,t+1,:,:], Gt_T)
            sigmas_smooth_tnt[:,t,:,:] = einsum2('nik,nkj->nij', sigmas_smooth[:,t+1,:,:], Gt_T)

    return ll, mus_smooth, sigmas_smooth, sigmas_smooth_tnt



def predict(Y, A, C, Q, R, mu0, Q0, pred_var=False):
    if pred_var:
        return _predict_mean_var(Y, A, C, Q, R, mu0, Q0)
    else:
        return _predict_mean(Y, A, C, Q, R, mu0, Q0)


def _predict_mean_var(Y, A, C, Q, R, mu0, Q0):
    """ Model predictions for Y given model parameters.

        Handles multiple lag dependence using component form.

        Note: This function doesn't handle control inputs (yet).

        Y : ndarray, shape=(N, T, D)
          Observations

        A : ndarray, shape=(T, D*nlag, D*nlag)
          Time-varying dynamics matrices

        C : ndarray, shape=(p, D)
          Observation matrix

        mu0: ndarray, shape=(D,)
          mean of initial state variable

        Q0 : ndarray, shape=(D, D)
          Covariance of initial state variable

        Q : ndarray, shape=(D, D)
          Covariance of latent states

        R : ndarray, shape=(D, D)
          Covariance of observations
    """

    N, T, _ = Y.shape
    _, D, Dnlags = A.shape
    nlags = Dnlags // D
    AA = np.stack([component_matrix(At, nlags) for At in A], axis=0)

    L_R = np.linalg.cholesky(R)

    p = C.shape[0]
    CC = hs([C, np.zeros((p, D*(nlags-1)))])
    tmp = solve_triangular(L_R, CC, lower=True)
    Rinv_CC = solve_triangular(L_R, tmp, trans='T', lower=True)
    CCT_Rinv_CC = einsum2('ki,kj->ij', CC, Rinv_CC)

    # tile L_R across number of trials so solve_triangular
    # can broadcast over trials properly
    L_R = np.tile(L_R, (N, 1, 1))

    QQ = np.zeros((T, Dnlags, Dnlags))
    QQ[:,:D,:D] = Q

    QQ0 = block_diag(*[Q0 for _ in range(nlags)])

    mu_predict = np.empty((N, T+1, Dnlags))
    sigma_predict = np.empty((N, T+1, Dnlags, Dnlags))

    mus_filt = np.empty((N, T, Dnlags))
    sigmas_filt = np.empty((N, T, Dnlags, Dnlags))

    mu_predict[:,0,:] = np.tile(mu0, nlags)
    sigma_predict[:,0,:,:] = QQ0.copy()

    I_tiled = np.tile(np.eye(Dnlags), (N, 1, 1))

    Yhat = np.empty_like(Y)
    St = np.empty((N, T, p, p))

    for t in range(T):

        # condition
        # sigma_x = dot3(C, sigma_predict, C.T) + R
        tmp1 = einsum2('ik,nkj->nij', CC, sigma_predict[:,t,:,:])
        sigma_x = einsum2('nik,jk->nij', tmp1, CC) + R
        sigma_x = sym(sigma_x)

        St[...,t,:,:] = sigma_x

        L = np.linalg.cholesky(sigma_x)
        Yhat[...,t,:] = einsum2('ik,nk->ni', CC, mu_predict[...,t,:])
        res = Y[...,t,:] - Yhat[...,t,:]

        v = solve_triangular(L, res, lower=True)

        mus_filt[:,t,:] = mu_predict[:,t,:] + einsum2('nki,nk->ni',
                                                        tmp1,
                                                        solve_triangular(L, v, trans='T', lower=True))

        # tmp2 = L^{-1}*C*sigma_predict
        tmp2 = solve_triangular(L, tmp1, lower=True)
        sigmas_filt[:,t,:,:] = sym(sigma_predict[:,t,:,:] - einsum2('nki,nkj->nij', tmp2, tmp2))

        # prediction
        #mu_predict = np.dot(A[t], mus_filt[t])
        mu_predict[:,t+1,:] = einsum2('ik,nk->ni', AA[t], mus_filt[:,t,:])

        #sigma_predict = dot3(A[t], sigmas_filt[t], A[t].T) + Q[t]
        tmp = einsum2('ik,nkl->nil', AA[t], sigmas_filt[:,t,:,:])
        sigma_predict[:,t+1,:,:] = sym(einsum2('nil,jl->nij', tmp, AA[t]) + QQ[t])

    # just return the diagonal of the St matrices for marginal predictive
    # variances
    return Yhat, np.diagonal(St, axis1=-2, axis2=-1)


def _predict_mean(Y, A, C, Q, R, mu0, Q0):
    """ Model predictions for Y given model parameters.
        Handles multiple lag dependence using component form.

        Note: This function doesn't handle control inputs (yet).

        Y : ndarray, shape=(N, T, D)
          Observations

        A : ndarray, shape=(T, D*nlag, D*nlag)
          Time-varying dynamics matrices

        C : ndarray, shape=(p, D)
          Observation matrix

        mu0: ndarray, shape=(D,)
          mean of initial state variable

        Q0 : ndarray, shape=(D, D)
          Covariance of initial state variable

        Q : ndarray, shape=(D, D)
          Covariance of latent states

        R : ndarray, shape=(D, D)
          Covariance of observations
    """

    N, T, _ = Y.shape
    _, D, Dnlags = A.shape
    nlags = Dnlags // D
    AA = np.stack([component_matrix(At, nlags) for At in A], axis=0)

    L_R = np.linalg.cholesky(R)

    p = C.shape[0]
    CC = hs([C, np.zeros((p, D*(nlags-1)))])
    tmp = solve_triangular(L_R, CC, lower=True)
    Rinv_CC = solve_triangular(L_R, tmp, trans='T', lower=True)
    CCT_Rinv_CC = einsum2('ki,kj->ij', CC, Rinv_CC)

    # tile L_R across number of trials so solve_triangular
    # can broadcast over trials properly
    L_R = np.tile(L_R, (N, 1, 1))

    QQ = np.zeros((T, Dnlags, Dnlags))
    QQ[:,:D,:D] = Q

    QQ0 = block_diag(*[Q0 for _ in range(nlags)])

    mu_predict = np.empty((N, T+1, Dnlags))
    sigma_predict = np.empty((N, T+1, Dnlags, Dnlags))

    mus_filt = np.empty((N, T, Dnlags))
    sigmas_filt = np.empty((N, T, Dnlags, Dnlags))

    mu_predict[:,0,:] = np.tile(mu0, nlags)
    sigma_predict[:,0,:,:] = QQ0.copy()

    I_tiled = np.tile(np.eye(Dnlags), (N, 1, 1))

    Yhat = np.empty_like(Y)

    for t in range(T):

        # condition
        tmp1 = einsum2('ik,nkj->nij', CC, sigma_predict[:,t,:,:])

        Yhat[...,t,:] = einsum2('ik,nk->ni', CC, mu_predict[...,t,:])
        res = Y[...,t,:] - Yhat[...,t,:]

        # Rinv * res
        tmp2 = solve_triangular(L_R, res, lower=True)
        tmp2 = solve_triangular(L_R, tmp2, trans='T', lower=True)

        # C^T Rinv * res
        tmp3 = einsum2('ki,nk->ni', Rinv_CC, res)

        # (Pinv + C^T Rinv C)_inv * tmp3
        L_P = np.linalg.cholesky(sigma_predict[:,t,:,:])
        tmp = solve_triangular(L_P, I_tiled, lower=True)
        Pinv = solve_triangular(L_P, tmp, trans='T', lower=True)
        tmp4 = sym(Pinv + CCT_Rinv_CC)
        L_tmp4 = np.linalg.cholesky(tmp4)
        tmp3 = solve_triangular(L_tmp4, tmp3, lower=True)
        tmp3 = solve_triangular(L_tmp4, tmp3, trans='T', lower=True)

        # Rinv C * tmp3
        tmp3 = einsum2('ik,nk->ni', Rinv_CC, tmp3)

        # add the two Woodbury * res terms together
        tmp = tmp2 - tmp3

        mus_filt[:,t,:] = mu_predict[:,t,:] + einsum2('nki,nk->ni', tmp1, tmp)

        # Rinv * tmp1
        tmp2 = solve_triangular(L_R, tmp1, lower=True)
        tmp2 = solve_triangular(L_R, tmp2, trans='T', lower=True)

        # C^T Rinv * tmp1
        tmp3 = einsum2('ki,nkj->nij', Rinv_CC, tmp1)

        # (Pinv + C^T Rinv C)_inv * tmp3
        tmp3 = solve_triangular(L_tmp4, tmp3, lower=True)
        tmp3 = solve_triangular(L_tmp4, tmp3, trans='T', lower=True)

        # Rinv C * tmp3
        tmp3 = einsum2('ik,nkj->nij', Rinv_CC, tmp3)

        # add the two Woodbury * tmp1 terms together, left-multiply by tmp1
        tmp = einsum2('nki,nkj->nij', tmp1, tmp2 - tmp3)

        sigmas_filt[:,t,:,:] = sym(sigma_predict[:,t,:,:] - tmp)

        # prediction
        mu_predict[:,t+1,:] = einsum2('ik,nk->ni', AA[t], mus_filt[:,t,:])

        #sigma_predict = dot3(A[t], sigmas_filt[t], A[t].T) + Q[t]
        tmp = einsum2('ik,nkl->nil', AA[t], sigmas_filt[:,t,:,:])
        sigma_predict[:,t+1,:,:] = sym(einsum2('nil,jl->nij', tmp, AA[t]) + QQ[t])

    return Yhat


def predict_step(mu_filt, sigma_filt, A, Q):
    mu_predict = einsum2('ik,nk->ni', A, mu_filt)
    tmp = einsum2('ik,nkl->nil', A, sigma_filt)
    sigma_predict = sym(einsum2('nil,jl->nij', tmp, A) + Q)

    return mu_predict, sigma_predict


def condition(y, C, R, mu_predict, sigma_predict):
    # dot3(C, sigma_predict, C.T) + R
    tmp1 = einsum2('ik,nkj->nij', C, sigma_predict)
    sigma_pred = einsum2('nik,jk->nij', tmp1, C) + R
    sigma_pred = sym(sigma_pred)

    L = np.linalg.cholesky(sigma_pred)
    # the transpose works b/c of how dot broadcasts
    #y_hat = np.dot(mu_predict, C.T)
    y_hat = einsum2('ik,nk->ni', C, mu_predict)
    res = y - y_hat
    v = solve_triangular(L, res, lower=True)

    mu_filt = mu_predict + einsum2('nki,nk->ni', tmp1, solve_triangular(L, v, trans='T', lower=True))

    tmp2 = solve_triangular(L, tmp1, lower=True)
    sigma_filt = sym(sigma_predict - einsum2('nki,nkj->nij', tmp2, tmp2))

    return y_hat, mu_filt, sigma_filt


def logZ(Y, A, C, Q, R, mu0, Q0):
    """ Log marginal likelihood using the Kalman filter.

        The algorithm broadcasts over the first dimension which are considered
        to be independent realizations.

        Note: This function doesn't handle control inputs (yet).

        Y : ndarray, shape=(N, T, D)
          Observations

        A : ndarray, shape=(T, D, D)
          Time-varying dynamics matrices

        C : ndarray, shape=(p, D)
          Observation matrix

        mu0: ndarray, shape=(D,)
          mean of initial state variable

        Q0 : ndarray, shape=(D, D)
          Covariance of initial state variable

        Q : ndarray, shape=(D, D)
          Covariance of latent states

        R : ndarray, shape=(D, D)
          Covariance of observations
    """

    N = Y.shape[0]
    T, D, _ = A.shape
    p = C.shape[0]

    mu_predict = np.stack([np.copy(mu0) for _ in range(N)], axis=0)
    sigma_predict = np.stack([np.copy(Q0) for _ in range(N)], axis=0)

    ll = 0.

    for t in range(T):

        # condition
        # sigma_x = dot3(C, sigma_predict, C.T) + R
        tmp1 = einsum2('ik,nkj->nij', C, sigma_predict)
        sigma_x = einsum2('nik,jk->nij', tmp1, C) + R
        sigma_x = sym(sigma_x)

        # res[n] = Y[n,t,:] = np.dot(C, mu_predict[n])
        res = Y[...,t,:] - einsum2('ik,nk->ni', C, mu_predict)

        L = np.linalg.cholesky(sigma_x)
        v = solve_triangular(L, res, lower=True)

        # log-likelihood over all trials
        ll += -0.5*(2.*np.sum(np.log(np.diagonal(L, axis1=1, axis2=2)))
                    + np.sum(v*v)
                    + N*p*np.log(2.*np.pi))

        mus_filt = mu_predict + einsum2('nki,nk->ni',
                                        tmp1,
                                        solve_triangular(L, v, trans='T', lower=True))

        # tmp2 = L^{-1}*C*sigma_predict
        tmp2 = solve_triangular(L, tmp1, lower=True)
        sigmas_filt = sigma_predict - einsum2('nki,nkj->nij', tmp2, tmp2)
        sigmas_filt = sym(sigmas_filt)

        # prediction
        #mu_predict = np.dot(A[t], mus_filt[t])
        mu_predict = einsum2('ik,nk->ni', A[t], mus_filt)

        # originally this worked with time-varying Q, but now it's fixed
        #sigma_predict = dot3(A[t], sigmas_filt[t], A[t].T) + Q[t]
        sigma_predict = einsum2('ik,nkl->nil', A[t], sigmas_filt)
        sigma_predict = einsum2('nil,jl->nij', sigma_predict, A[t]) + Q
        sigma_predict = sym(sigma_predict)

    return np.sum(ll)
