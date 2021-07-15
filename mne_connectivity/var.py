from mne_connectivity.base import Connectivity, EpochConnectivity, TemporalConnectivity
import os
from pathlib import Path
import shutil
import numpy as np
import scipy
from scipy.linalg import sqrtm
from tqdm import tqdm
from sklearn.linear_model import Ridge

from mne.utils import verbose, warn
from .utils import parallel_loop


def _construct_var_eqns(data, model_order, delta=None):
    """Construct VAR equation system (optionally with RLS constraint).

    This function was originally imported from ``scot``.

    Parameters
    ----------
    data : np.ndarray
        The multivariate data (n_epochs, n_signals, n_times).
    model_order : int
        The order of the VAR model.
    delta : float, optional
        The l2 penalty term for ridge regression, by default None, which
        will result in ordinary VAR equation. 

    Returns
    -------
    X : np.ndarray
        The predictor multivariate time-series. This will have shape
        ``() See Notes.
    Y : np.ndarray
        The predicted multivariate time-series. See Notes.

    Notes
    -----
    This function will format data such as:

        Y = A X

    where Y is time-shifted data copy of X and ``A`` defines
    how X linearly maps to Y.
    """
    # n_epochs, n_signals, n_times
    n_epochs, n_signals, n_times = np.shape(data)

    # number of linear relations
    n = (n_times - model_order) * n_epochs
    rows = n if delta is None else n + n_signals * model_order

    # Construct matrix X (predictor variables)
    X = np.zeros((rows, n_signals * model_order))
    for i in range(n_signals):
        for k in range(1, model_order + 1):
            X[:n, i * model_order + k -
                1] = np.reshape(data[:, i, model_order - k:-k].T, n)

    if delta is not None:
        np.fill_diagonal(X[n:, :], delta)

    # Construct vectors yi (response variables for each channel i)
    Y = np.zeros((rows, n_signals))
    for i in range(n_signals):
        Y[:n, i] = np.reshape(data[:, i, model_order:].T, n)

    return X, Y


def estimate_model_order(data, min_p=1, max_p=None):
    """Determine optimal model order.

    This will estimate model order by minimizing the mean squared
    generalization error.

    Parameters
    ----------
    data : array, shape (n_trials, n_channels, n_samples)
        Epoched data set on which to optimize the model order. At least two
        trials are required.
    min_p : int
        Minimal model order to check.
    max_p : int
        Maximum model order to check
    """
    data = np.asarray(data)
    if data.shape[0] < 2:
        raise ValueError("At least two trials are required.")

    msge, prange = [], []

    par, func = parallel_loop(_get_msge_with_gradient, n_jobs=self.n_jobs,
                              verbose=self.verbose)
    if self.n_jobs is None:
        npar = 1
    elif self.n_jobs < 0:
        npar = 4  # is this a sane default?
    else:
        npar = self.n_jobs

    p = min_p
    while True:
        result = par(func(data, self.delta, self.xvschema, 1, p_)
                     for p_ in range(p, p + npar))
        j, k = zip(*result)
        prange.extend(range(p, p + npar))
        msge.extend(j)
        p += npar
        if max_p is None:
            if len(msge) >= 2 and msge[-1] > msge[-2]:
                break
        else:
            if prange[-1] >= max_p:
                i = prange.index(max_p) + 1
                prange = prange[:i]
                msge = msge[:i]
                break
    self.p = prange[np.argmin(msge)]
    return zip(prange, msge)


def _construct_snapshots(snapshots, order, n_times):
    """Construct snapshots matrix.

    This will construct a matrix along the 0th axis (rows),
    stacking copies of the data based on order.

    Parameters
    ----------
    snapshots : np.ndarray
        A multivariate time-series ``(n_signals, n_times)``.
    order : int
        The order of the linear model to be estimated.
    n_times : int
        The number of times in the original dataset.

    Returns
    -------
    snaps : np.ndarray
        A snapshot matrix with copies of the original ``snapshots``
        along the rows based on the ``order`` of the model.

    Notes
    -----
    Say ``snapshots`` is an array with shape (N, T) with 
    order ``M``. We will abbreviate this matrix and call it ``X``.
    ``X_ij`` is the ith signal at time point j.

    The resulting ``snaps`` matrix would be a (N*M, T - M) array:

    +------+------+------+------------+
    | X_00 | X_01 | ...  | X_0(T-M+1) |
    +------+------+------+------------+
    | X_10 | X_11 | ...  | X_1(T-M+1) |
    +------+------+------+------------+
    | ...  | ...  | ...  | ...        |
    +------+------+------+------------+
    | X_N0 | X_N1 | ...  | X_N(T-M+1) |
    +------+------+------+------------+
    | X_01 | X_02 | ...  | X_0(T-M+2) |
    +------+------+------+------------+
    | ...  | ...  | ...  | ...        |
    +------+------+------+------------+
    | X_N1 | X_N2 | ...  | X_N(T-M+2) |
    +------+------+------+------------+
    | ...  | ...  | ...  | ...        |
    +------+------+------+------------+
    | X_NM | X_N2 | ...  | X_N(T-M+M-1)|
    +------+------+------+------------+

    """
    snaps = np.concatenate(
        [snapshots[:, i: n_times - order + i + 1] for i in range(order)],
        axis=0,
    )
    return snaps


@verbose
def var(data, times=None, names=None, model_order=1, delta=0.0,
        memmap=True, compute_fb_operator=False,
        n_jobs=-1, avg_epochs=False, verbose=None):
    """[summary]

    Parameters
    ----------
        Parameters
    ----------
    data : array-like, shape=(n_epochs, n_signals, n_times) | generator
        The data from which to compute connectivity. The epochs dimension
        is interpreted differently, depending on ``'output'`` argument.
    names : list | array-like | None
        A list of names associated with the signals in ``data``.
        If None, will be a list of indices of the number of nodes.
    model_order : int | str, optional
        Autoregressive model order, by default 1. If 'auto', then
        will use Bayesian Information Criterion to estimate the
        model order.
    delta : float, optional
        Ridge penalty parameter, by default 0.0
    memmap : bool
        Whether or not to memory map the epoch data on disk during
        joblib parallelization.
    compute_fb_operator : bool
        Whether to compute the backwards operator and average with
        the forward operator.
    avg_epochs : bool
        Whether to average over the VAR models computed from epochs,
        or to treat each epoch as a separate VAR model. See Notes.
    %(n_jobs)s
    %(verbose)s

    Returns
    -------
    conn : Connectivity | TemporalConnectivity | EpochConnectivity
        The connectivity data estimated.

    See Also
    --------
    Connectivity
    TemporalConnectivity
    EpochConnectivity

    Notes
    -----
    *b* is of shape [m, m*p], with sub matrices arranged as follows:

    +------+------+------+------+
    | b_00 | b_01 | ...  | b_0m |
    +------+------+------+------+
    | b_10 | b_11 | ...  | b_1m |
    +------+------+------+------+
    | ...  | ...  | ...  | ...  |
    +------+------+------+------+
    | b_m0 | b_m1 | ...  | b_mm |
    +------+------+------+------+

    Each sub matrix b_ij is a column vector of length p that contains the
    filter coefficients from channel j (source) to channel i (sink).
    """
    # 1. determine shape of the window of data
    n_epochs, n_nodes, n_times = data.shape

    if avg_epochs:
        # compute VAR model where each epoch is a
        # sample of the multivariate time-series of interest
        # ordinary least squares or regularized least squares (ridge regression)
        X, Y = _construct_var_eqns(data, model_order, delta=delta)

        b, res, rank, s = scipy.linalg.lstsq(X, Y)

        # get the coefficients
        coef = b.transpose()

        # create connectivity
        coef = coef.flatten()
        conn = Connectivity(data=coef, n_nodes=n_nodes, names=names,
                            n_epochs_used=n_epochs,
                            method='VAR')
    else:
        if times is None:
            raise RuntimeError('If computing time (epoch) varying VAR model, '
                               'then "times" must be passed in. From '
                               'MNE epochs, one can extract using "epochs.times".')

        # compute time-varying VAR model where each epoch
        # is one sample of a time-varying multivariate time-series
        # linear system
        conn = _system_identification(data=data, times=times, names=names, order=model_order,
                                      delta=delta, memmap=memmap, n_jobs=n_jobs,
                                      compute_fb_operator=compute_fb_operator,
                                      verbose=verbose)
    return conn


def _system_identification(data, times, names=None, order=1, delta=0,
                           random_state=None, memmap=False, n_jobs=-1,
                           compute_fb_operator=False,
                           verbose=True):
    """Solve system identification using least-squares over all epochs.

    Treats each epoch as a different window of time to estimate the model:

        X(t+1) = \sum_{i=0}^{order} A_i X(t - i)

        X(t+1) = A X(t)

    where ``data`` comprises of ``(n_signals, n_times)`` and ``X(t)`` are
    the data snapshots.
    """
    # 1. determine shape of the window of data
    n_epochs, n_nodes, n_times = data.shape

    model_params = {
        'delta': delta,
        'order': order,
        'random_state': random_state,
        'compute_fb_operator': compute_fb_operator
    }

    # compute the A matrix for all Epochs
    A_mats = np.zeros((n_epochs, n_nodes, n_nodes))
    if n_jobs == 1:
        for idx in tqdm(range(n_epochs)):
            A = _compute_lds_func(data[idx, ...], **model_params)
            A_mats[idx, ...] = A
    else:
        try:
            from joblib import Parallel, delayed, dump, load
        except ImportError as e:
            raise ImportError(e)

        if memmap:
            folder = Path("./joblib_memmap")
            folder.mkdir(exist_ok=True, parents=True)

            # dump data into memmap for joblib
            data_filename_memmap = os.path.join(folder, "data_memmap")
            dump(data, data_filename_memmap)
            arr = load(data_filename_memmap, mmap_mode="r")
        else:
            arr = data

        # run parallelized job to compute fragility over all windows
        results = Parallel(n_jobs=n_jobs)(
            delayed(_compute_lds_func)(
                arr[idx, ...], **model_params
            )
            for idx in tqdm(range(n_epochs))
        )
        for idx in range(len(results)):
            adjmat = results[idx]
            A_mats[idx, ...] = adjmat

        if memmap:
            try:
                shutil.rmtree(folder)
            except:  # noqa
                print("Could not clean-up joblib memmap automatically.")

    # create connectivity
    A_mats = A_mats.reshape((n_epochs, -1))
    conn = EpochConnectivity(data=A_mats, n_nodes=n_nodes, names=names,
                             n_epochs_used=n_epochs,
                             method='Time-varying LDS',
                             **model_params)
    return conn


def _compute_lds_func(data, order, delta, compute_fb_operator, random_state):
    """Compute linear system using VAR model.

    Allows for parallelization over epochs.
    """
    n_times = data.shape[-1]

    # create large snapshot with time-lags of order specified by
    # ``order`` value
    snaps = _construct_snapshots(
        data, order=order, n_times=n_times
    )

    # get the time-shifted components of each
    X, Y = snaps[:, :-1], snaps[:, 1:]

    # use scikit-learn Ridge Regression to fit
    fit_intercept = False
    normalize = False
    solver = 'auto'
    clf = Ridge(
        alpha=delta,
        fit_intercept=fit_intercept,
        normalize=normalize,
        solver=solver,
        random_state=random_state,
    )

    # n_samples X n_features and n_samples X n_targets
    clf.fit(X.T, Y.T)

    # n_targets X n_features
    A = clf.coef_

    if compute_fb_operator:
        """Compute foward-backward operator.

        Addresses bias in the least-square estimation [2].

        .. [2] Characterizing and correcting for the effect of sensor noise in the
            dynamic mode decomposition. https://arxiv.org/pdf/1507.02264.pdf
        """
        # compute backward linear operator
        clf.fit(Y.T, X.T)
        back_A = clf.coef_
        A = sqrtm(A.dot(np.linalg.inv(back_A)))

    return A
