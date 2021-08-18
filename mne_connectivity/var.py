import numpy as np
import scipy
from scipy.linalg import sqrtm
from tqdm import tqdm

from .utils import fill_doc
from .base import Connectivity, EpochConnectivity


@fill_doc
def vector_auto_regression(
        data, times=None, names=None, model_order=1, l2_reg=0.0,
        compute_fb_operator=False, n_jobs=1, model='dynamic', verbose=None):
    """Compute vector auto-regresssive (VAR) model.

    Parameters
    ----------
    data : array-like, shape=(n_epochs, n_signals, n_times) | generator
        The data from which to compute connectivity. The epochs dimension
        is interpreted differently, depending on ``'output'`` argument.
    times : array-like
        (Optional) The time points used to construct the epoched ``data``. If
        ``None``, then ``times_used`` in the Connectivity will not be
        available.
    %(names)s
    model_order : int | str, optional
        Autoregressive model order, by default 1.
    l2_reg : float, optional
        Ridge penalty (l2-regularization) parameter, by default 0.0
    compute_fb_operator : bool
        Whether to compute the backwards operator and average with
        the forward operator. Addresses bias in the least-square
        estimation :footcite:`Dawson_2016`.
    model : str
        Whether to compute one VAR model using all epochs as multiple
        samples of the same VAR model ('avg-epochs'), or to compute
        a separate VAR model for each epoch ('dynamic'), which results
        in a time-varying VAR model. See Notes.
    %(n_jobs)s
    %(verbose)s

    Returns
    -------
    conn : Connectivity | TemporalConnectivity | EpochConnectivity
        The connectivity data estimated.

    See Also
    --------
    mne_connectivity.Connectivity
    mne_connectivity.EpochConnectivity

    Notes
    -----
    Names can be passed in, which are then used to instantiate the nodes
    of the connectivity class. For example, they can be the electrode names
    of EEG.

    For higher-order VAR models, there are n_order ``A`` matrices,
    representing the linear dynamics with respect to that lag. These
    are represented by vertically concatenated matrices. For example, if
    the input is data where n_signals is 3, then an order-1 VAR model will
    result in a 3x3 connectivity matrix. An order-2 VAR model will result in a
    6x3 connectivity matrix, with two 3x3 matrices representing the dynamics
    at lag 1 and lag 2, respectively.

    When computing a VAR model (i.e. linear dynamical system), we require
    the input to be a ``(n_epochs, n_signals, n_times)`` 3D array. There
    are two ways one can interpret the data in the model.

    First, epochs can be treated as multiple samples observed for a single
    VAR model. That is, we have $X_1, X_2, ..., X_n$, where each $X_i$
    is a ``(n_signals, n_times)`` data array, with n epochs. We are
    interested in estimating the parameters, $(A_1, A_2, ..., A_{order})$
    from the following model over **all** epochs:

    .. math::
        X(t+1) = \sum_{i=0}^{order} A_i X(t-i)

    This results in one VAR model over all the epochs.

    The second approach treats each epoch as a different VAR model,
    estimating a time-varying VAR model. Using the same
    data as above,  we now are interested in estimating the
    parameters, $(A_1, A_2, ..., A_{order})$ for **each** epoch. The model
    would be the following for **each** epoch:

    .. math::
        X(t+1) = \sum_{i=0}^{order} A_i X(t-i)

    This results in one VAR model for each epoch. This is done according
    to the model in :footcite:`li_linear_2017`.

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

    References
    ----------
    .. footbibliography::
    """
    if model not in ['avg-epochs', 'dynamic']:
        raise ValueError(f'"model" parameter must be one of '
                         f'(avg-epochs, dynamic), not {model}.')

    # 1. determine shape of the window of data
    n_epochs, n_nodes, n_times = data.shape

    model_params = {
        'model_order': model_order,
        'l2_reg': l2_reg
    }
    if model == 'avg-epochs':
        # compute VAR model where each epoch is a
        # sample of the multivariate time-series of interest
        # ordinary least squares or regularized least squares
        # (ridge regression)
        X, Y = _construct_var_eqns(data, **model_params)

        b, res, rank, s = scipy.linalg.lstsq(X, Y)

        # get the coefficients
        coef = b.transpose()

        # create connectivity
        coef = coef.flatten()
        conn = Connectivity(data=coef, n_nodes=n_nodes, names=names,
                            n_epochs_used=n_epochs,
                            times_used=times,
                            method='VAR', **model_params)
    else:
        assert model == 'dynamic'
        if times is None and n_epochs > 1:
            raise RuntimeError('If computing time (epoch) varying VAR model, '
                               'then "times" must be passed in. From '
                               'MNE epochs, one can extract this using '
                               '"epochs.times".')

        # compute time-varying VAR model where each epoch
        # is one sample of a time-varying multivariate time-series
        # linear system
        conn = _system_identification(
            data=data, times=times, names=names, model_order=model_order,
            l2_reg=l2_reg, n_jobs=n_jobs,
            compute_fb_operator=compute_fb_operator,
            verbose=verbose)
    return conn


def _construct_var_eqns(data, model_order, l2_reg=None):
    """Construct VAR equation system (optionally with RLS constraint).

    This function was originally imported from ``scot``.

    Parameters
    ----------
    data : np.ndarray (n_epochs, n_signals, n_times)
        The multivariate data.
    model_order : int
        The order of the VAR model.
    l2_reg : float, optional
        The l2 penalty term for ridge regression, by default None, which
        will result in ordinary VAR equation.

    Returns
    -------
    X : np.ndarray
        The predictor multivariate time-series. This will have shape
        ``(model_order * (n_times - model_order),
        n_signals * model_order)``. See Notes.
    Y : np.ndarray
        The predicted multivariate time-series. This will have shape
        ``(model_order * (n_times - model_order),
        n_signals * model_order)``. See Notes.

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
    rows = n if l2_reg is None else n + n_signals * model_order

    # Construct matrix X (predictor variables)
    X = np.zeros((rows, n_signals * model_order))
    for i in range(n_signals):
        for k in range(1, model_order + 1):
            X[:n, i * model_order + k -
                1] = np.reshape(data[:, i, model_order - k:-k].T, n)

    if l2_reg is not None:
        np.fill_diagonal(X[n:, :], l2_reg)

    # Construct vectors yi (response variables for each channel i)
    Y = np.zeros((rows, n_signals))
    for i in range(n_signals):
        Y[:n, i] = np.reshape(data[:, i, model_order:].T, n)

    return X, Y


def _construct_snapshots(snapshots, order, n_times):
    """Construct snapshots matrix.

    This will construct a matrix along the 0th axis (rows),
    stacking copies of the data based on order.

    Parameters
    ----------
    snapshots : np.ndarray (n_signals, n_times)
        A multivariate time-series.
    order : int
        The order of the linear model to be estimated.
    n_times : int
        The number of times in the original dataset.

    Returns
    -------
    snaps : np.ndarray (n_signals * order, n_times - order)
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


def _system_identification(data, times, names=None, model_order=1, l2_reg=0,
                           random_state=None, n_jobs=-1,
                           compute_fb_operator=False,
                           verbose=True):
    """Solve system identification using least-squares over all epochs.

    Treats each epoch as a different window of time to estimate the model:

    .. math::
        X(t+1) = \sum_{i=0}^{order} A_i X(t - i)

    where ``data`` comprises of ``(n_signals, n_times)`` and ``X(t)`` are
    the data snapshots.
    """
    # 1. determine shape of the window of data
    n_epochs, n_nodes, n_times = data.shape

    model_params = {
        'l2_reg': l2_reg,
        'model_order': model_order,
        'random_state': random_state,
        'compute_fb_operator': compute_fb_operator
    }

    # compute the A matrix for all Epochs
    A_mats = np.zeros((n_epochs, n_nodes * model_order, n_nodes))
    if n_jobs == 1:
        for idx in tqdm(range(n_epochs)):
            A = _compute_lds_func(data[idx, ...], **model_params)
            A_mats[idx, ...] = A
    else:
        try:
            from joblib import Parallel, delayed
        except ImportError as e:
            raise ImportError(e)

        arr = data

        # run parallelized job to compute over all windows
        results = Parallel(n_jobs=n_jobs)(
            delayed(_compute_lds_func)(
                arr[idx, ...], **model_params
            )
            for idx in tqdm(range(n_epochs))
        )
        for idx in range(len(results)):
            adjmat = results[idx]
            # add additional order models in dynamic connectivity
            # along the first node axes
            for jdx in range(model_order):
                A_mats[idx, jdx * n_nodes: n_nodes * (jdx + 1), :] = adjmat[
                    -n_nodes:, jdx * n_nodes: n_nodes * (jdx + 1)
                ]

    # create connectivity
    A_mats = A_mats.reshape((n_epochs, -1))
    conn = EpochConnectivity(data=A_mats, n_nodes=n_nodes, names=names,
                             n_epochs_used=n_epochs,
                             times_used=times,
                             method='Time-varying LDS',
                             **model_params)
    return conn


def _compute_lds_func(data, model_order, l2_reg, compute_fb_operator,
                      random_state):
    """Compute linear system using VAR model.

    Allows for parallelization over epochs.
    """
    from sklearn.linear_model import Ridge

    n_times = data.shape[-1]

    # create large snapshot with time-lags of order specified by
    # ``order`` value
    snaps = _construct_snapshots(
        data, order=model_order, n_times=n_times
    )

    # get the time-shifted components of each
    X, Y = snaps[:, :-1], snaps[:, 1:]

    # use scikit-learn Ridge Regression to fit
    fit_intercept = False
    normalize = False
    solver = 'auto'
    clf = Ridge(
        alpha=l2_reg,
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
        # compute backward linear operator
        clf.fit(Y.T, X.T)
        back_A = clf.coef_
        A = sqrtm(A.dot(np.linalg.inv(back_A)))

    return A
