from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import autograd.numpy as np
from autograd.scipy.linalg import block_diag

from .util import sym, component_matrix, hs

try:
    from autograd_linalg import solve_triangular
except ImportError:
    raise RuntimeError("must install `autograd_linalg` package")

from numpy import einsum 

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
        tmp1 = einsum('ik,nkj->nij', CC, sigma_predict[:,t,:,:])
        sigma_x = einsum('nik,jk->nij', tmp1, CC) + R
        sigma_x = sym(sigma_x)

        if St is not None:
            St[...,t,:,:] = sigma_x

        L = np.linalg.cholesky(sigma_x)
        # res[n] = Y[n,t,:] = np.dot(C, mu_predict[n,t,:])
        res = Y[...,t,:] - einsum('ik,nk->ni', CC, mu_predict[...,t,:])
        v = solve_triangular(L, res, lower=True)

        # log-likelihood over all trials
        ll += -0.5*(2.*np.sum(np.log(np.diagonal(L, axis1=1, axis2=2)))
                    + np.sum(v*v)
                    + N*p*np.log(2.*np.pi))

        mus_smooth[:,t,:] = mu_predict[:,t,:] + \
                            einsum('nki,nk->ni', tmp1, \
                                 solve_triangular(L, v, trans='T', lower=True))

        # tmp2 = L^{-1}*C*sigma_predict
        tmp2 = solve_triangular(L, tmp1, lower=True)
        sigmas_smooth[:,t,:,:] = sym(sigma_predict[:,t,:,:] - \
                                     einsum('nki,nkj->nij', tmp2, tmp2))

        # prediction
        #mu_predict = np.dot(A[t], mus_smooth[t])
        mu_predict[:,t+1,:] = einsum('ik,nk->ni', AA[t], mus_smooth[:,t,:])

        #sigma_predict = dot3(A[t], sigmas_smooth[t], A[t].T) + Q[t]
        tmp = einsum('ik,nkl->nil', AA[t], sigmas_smooth[:,t,:,:])
        sigma_predict[:,t+1,:,:] = sym(einsum('nil,jl->nij', tmp, AA[t]) + \
                                       QQ[t])

    for t in range(T-2, -1, -1):

        # these names are stolen from mattjj and slinderman
        #temp_nn = np.dot(A[t], sigmas_smooth[n,t,:,:])
        temp_nn = einsum('ik,nkj->nij', AA[t], sigmas_smooth[:,t,:,:])

        L = np.linalg.cholesky(sigma_predict[:,t+1,:,:])
        v = solve_triangular(L, temp_nn, lower=True)
        # Look in Saarka for dfn of Gt_T
        Gt_T = solve_triangular(L, v, trans='T', lower=True)

        # {mus,sigmas}_smooth[n,t] contain the filtered estimates so we're
        # overwriting them on purpose
        mus_smooth[:,t,:] = mus_smooth[:,t,:] + \
          einsum('nki,nk->ni', Gt_T, mus_smooth[:,t+1,:] - mu_predict[:,t+1,:])

        tmp = einsum('nki,nkj->nij', Gt_T, sigmas_smooth[:,t+1,:,:] - \
                     sigma_predict[:,t+1,:,:])
        tmp = einsum('nik,nkj->nij', tmp, Gt_T)
        sigmas_smooth[:,t,:,:] = sym(sigmas_smooth[:,t,:,:] + tmp)

        if compute_lag1_cov:
            # This matrix is NOT symmetric, so don't symmetrize!
            sigmas_smooth_tnt[:,t,:,:] = einsum('nik,nkj->nij', \
                                                sigmas_smooth[:,t+1,:,:], Gt_T)

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
    CCT_Rinv_CC = einsum('ki,kj->ij', CC, Rinv_CC)

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
        tmp1 = einsum('ik,nkj->nij', CC, sigma_predict[:,t,:,:])

        res = Y[...,t,:] - einsum('ik,nk->ni', CC, mu_predict[...,t,:])

        # Rinv * res
        tmp2 = solve_triangular(L_R, res, lower=True)
        tmp2 = solve_triangular(L_R, tmp2, trans='T', lower=True)

        # C^T Rinv * res
        tmp3 = einsum('ki,nk->ni', Rinv_CC, res)

        # (Pinv + C^T Rinv C)_inv * tmp3
        L_P = np.linalg.cholesky(sigma_predict[:,t,:,:])
        tmp = solve_triangular(L_P, I_tiled, lower=True)
        Pinv = solve_triangular(L_P, tmp, trans='T', lower=True)
        tmp4 = sym(Pinv + CCT_Rinv_CC)
        L_tmp4 = np.linalg.cholesky(tmp4)
        tmp3 = solve_triangular(L_tmp4, tmp3, lower=True)
        tmp3 = solve_triangular(L_tmp4, tmp3, trans='T', lower=True)

        # Rinv C * tmp3
        tmp3 = einsum('ik,nk->ni', Rinv_CC, tmp3)

        # add the two Woodbury * res terms together
        tmp = tmp2 - tmp3

        mus_smooth[:,t,:] = mu_predict[:,t,:] + einsum('nki,nk->ni', tmp1, tmp)

        # Rinv * tmp1
        tmp2 = solve_triangular(L_R, tmp1, lower=True)
        tmp2 = solve_triangular(L_R, tmp2, trans='T', lower=True)

        # C^T Rinv * tmp1
        tmp3 = einsum('ki,nkj->nij', Rinv_CC, tmp1)

        # (Pinv + C^T Rinv C)_inv * tmp3
        tmp3 = solve_triangular(L_tmp4, tmp3, lower=True)
        tmp3 = solve_triangular(L_tmp4, tmp3, trans='T', lower=True)

        # Rinv C * tmp3
        tmp3 = einsum('ik,nkj->nij', Rinv_CC, tmp3)

        # add the two Woodbury * tmp1 terms together, left-multiply by tmp1
        tmp = einsum('nki,nkj->nij', tmp1, tmp2 - tmp3)

        sigmas_smooth[:,t,:,:] = sym(sigma_predict[:,t,:,:] - tmp)

        # prediction
        mu_predict[:,t+1,:] = einsum('ik,nk->ni', AA[t], mus_smooth[:,t,:])

        #sigma_predict = dot3(A[t], sigmas_smooth[t], A[t].T) + Q[t]
        tmp = einsum('ik,nkl->nil', AA[t], sigmas_smooth[:,t,:,:])
        sigma_predict[:,t+1,:,:] = sym(einsum('nil,jl->nij', tmp, AA[t]) + \
                                       QQ[t])

    for t in range(T-2, -1, -1):

        # these names are stolen from mattjj and slinderman
        #temp_nn = np.dot(A[t], sigmas_smooth[n,t,:,:])
        temp_nn = einsum('ik,nkj->nij', AA[t], sigmas_smooth[:,t,:,:])

        L = np.linalg.cholesky(sigma_predict[:,t+1,:,:])
        v = solve_triangular(L, temp_nn, lower=True)
        # Look in Saarka for dfn of Gt_T
        Gt_T = solve_triangular(L, v, trans='T', lower=True)

        # {mus,sigmas}_smooth[n,t] contain the filtered estimates so we're
        # overwriting them on purpose
        mus_smooth[:,t,:] = mus_smooth[:,t,:] + \
          einsum('nki,nk->ni', Gt_T, mus_smooth[:,t+1,:] - mu_predict[:,t+1,:])

        tmp = einsum('nki,nkj->nij', Gt_T, sigmas_smooth[:,t+1,:,:] - \
                     sigma_predict[:,t+1,:,:])
        tmp = einsum('nik,nkj->nij', tmp, Gt_T)
        sigmas_smooth[:,t,:,:] = sym(sigmas_smooth[:,t,:,:] + tmp)

        if compute_lag1_cov:
            # This matrix is NOT symmetric, so don't symmetrize!
            sigmas_smooth_tnt[:,t,:,:] = einsum('nik,nkj->nij', \
                                                sigmas_smooth[:,t+1,:,:], Gt_T)

    return ll, mus_smooth, sigmas_smooth, sigmas_smooth_tnt
