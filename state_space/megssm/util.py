from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import autograd.numpy as np
from numpy.lib.stride_tricks import as_strided as ast


hs = lambda *args: np.concatenate(*args, axis=-1)

def T_(X):
    return np.swapaxes(X, -1, -2)

def sym(X):
    return 0.5*(X + T_(X))

def dot3(A, B, C):
    return np.dot(A, np.dot(B, C))

def relnormdiff(A, B, min_denom=1e-9):
    return np.linalg.norm(A - B) / np.maximum(np.linalg.norm(A), min_denom)

def _ensure_ndim(X, T, ndim):
    X = np.require(X, dtype=np.float64, requirements='C')
    assert ndim-1 <= X.ndim <= ndim
    if X.ndim == ndim:
        assert X.shape[0] == T
        return X
    else:
        return ast(X, shape=(T,) + X.shape, strides=(0,) + X.strides)

def rand_psd(n, minew=0.1, maxew=1.):
    # maxew is badly named
    if n == 1:
        return maxew * np.eye(n)
    X = np.random.randn(n,n)
    S = np.dot(T_(X), X)
    S = sym(S)
    ew, ev = np.linalg.eigh(S)
    ew -= np.min(ew)
    ew /= np.max(ew)
    ew *= (maxew - minew)
    ew += minew
    return dot3(ev, np.diag(ew), T_(ev))

def rand_stable(n, maxew=0.9):
    A = np.random.randn(n, n)
    A *= maxew / np.max(np.abs(np.linalg.eigvals(A)))
    return A

def component_matrix(As, nlags):
    """ compute component form of latent VAR process
        
        [A_1 A_2 ... A_p]
        [ I   0  ...  0 ]
        [ 0   I   0   0 ]
        [ 0 ...   I   0 ]

    """

    d = As.shape[0]
    res = np.zeros((d*nlags, d*nlags))
    res[:d] = As
    
    if nlags > 1:
        res[np.arange(d,d*nlags), np.arange(d*nlags-d)] = 1

    return res

def linesearch(f, grad_f, xk, pk, step_size=1., tau=0.1, c1=1e-4,
    prox_op=None, lam=1.):
    """ find a step size via backtracking line search with armijo condition """
    obj_start = f(xk)
    grad_xk = grad_f(xk)
    obj_new = np.finfo('float').max
    armijo_condition = 0

    if prox_op is None:
        prox_op = lambda x, y: x

    while obj_new > armijo_condition:
        x_new = prox_op(xk - step_size * pk, lam*step_size)
        armijo_condition = obj_start - c1*step_size*(np.sum(pk*grad_xk))
        obj_new = f(x_new)
        step_size *= tau

    return step_size/tau

def soft_thresh_At(At, lam):
    At = At.copy()
    diag_inds = np.diag_indices(At.shape[1])
    At_diag = np.diagonal(At, axis1=-2, axis2=-1)

    At = np.sign(At) * np.maximum(np.abs(At) - lam, 0.)

    # fill in diagonal with originally updated entries as we're not
    # going to penalize them
    for tt in range(At.shape[0]):
        At[tt][diag_inds] = At_diag[tt]
    return At

def block_thresh_At(At, lam, min_norm=1e-16):
    At = At.copy()
    diag_inds = np.diag_indices(At.shape[1])
    At_diag = np.diagonal(At, axis1=-2, axis2=-1)

    norms = np.linalg.norm(At, axis=0, keepdims=True)
    norms = np.maximum(norms, min_norm)
    scales = np.maximum(norms - lam, 0.)
    At = scales * (At / norms)

    # fill in diagonal with originally updated entries as we're not
    # going to penalize them
    for tt in range(At.shape[0]):
        At[tt][diag_inds] = At_diag[tt]
    return At

