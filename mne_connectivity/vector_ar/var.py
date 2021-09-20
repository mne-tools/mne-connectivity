import numpy as np
import scipy
from scipy.linalg import sqrtm
from tqdm import tqdm

from mne.utils import _check_option, logger

from ..utils import fill_doc
from ..base import Connectivity, EpochConnectivity


@fill_doc
def vector_auto_regression(
        data, times=None, names=None, lags=1, trend='n', l2_reg=0.0,
        compute_fb_operator=False, model='dynamic', n_jobs=1, verbose=None):
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
    lags : int, optional
        Autoregressive model order, by default 1.
    trend : str {'n', 'c', 't', 'ct', 'ctt'}
        The trend to add.

        * 'n' add no trend.
        * 'c' add constant only.
        * 't' add trend only.
        * 'ct' add constant and linear trend.
        * 'ctt' add constant and linear and quadratic trend.
    l2_reg : float, optional
        Ridge penalty (l2-regularization) parameter, by default 0.0.
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
        X(t+1) = \\sum_{i=0}^{order} A_i X(t-i)

    This results in one VAR model over all the epochs.

    The second approach treats each epoch as a different VAR model,
    estimating a time-varying VAR model. Using the same
    data as above,  we now are interested in estimating the
    parameters, $(A_1, A_2, ..., A_{order})$ for **each** epoch. The model
    would be the following for **each** epoch:

    .. math::
        X(t+1) = \\sum_{i=0}^{order} A_i X(t-i)

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

    In order to optimize RAM usage, the estimating equations are set up
    by iterating over sample points. This assumes that there are in general
    more sample points then channels. You should not estimate a VAR model
    using less sample points then channels, unless you have good reason.

    References
    ----------
    .. footbibliography::
    """
    if model not in ['avg-epochs', 'dynamic']:
        raise ValueError(f'"model" parameter must be one of '
                         f'(avg-epochs, dynamic), not {model}.')

    # 1. determine shape of the window of data
    n_epochs, n_nodes, _ = data.shape

    model_params = {
        'lags': lags,
        'l2_reg': l2_reg,
    }

    if verbose:
        logger.info(f'Running {model} vector autoregression with parameters: '
                    f'\n{model_params}')

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
            data=data, times=times, names=names, lags=lags, trend=trend,
            l2_reg=l2_reg, n_jobs=n_jobs,
            compute_fb_operator=compute_fb_operator)
    return conn


def _construct_var_eqns(data, lags, l2_reg=None):
    """Construct VAR equation system (optionally with RLS constraint).

    This function was originally imported from ``scot``.

    Parameters
    ----------
    data : np.ndarray (n_epochs, n_signals, n_times)
        The multivariate data.
    lags : int
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
    n = (n_times - lags) * n_epochs
    rows = n if l2_reg is None else n + n_signals * lags

    # Construct matrix X (predictor variables)
    X = np.zeros((rows, n_signals * lags))
    for i in range(n_signals):
        for k in range(1, lags + 1):
            X[:n, i * lags + k -
                1] = np.reshape(data[:, i, lags - k:-k].T, n)

    if l2_reg is not None:
        np.fill_diagonal(X[n:, :], l2_reg)

    # Construct vectors yi (response variables for each channel i)
    Y = np.zeros((rows, n_signals))
    for i in range(n_signals):
        Y[:n, i] = np.reshape(data[:, i, lags:].T, n)

    return X, Y


def _system_identification(data, times, names, lags, trend, l2_reg=0,
                           n_jobs=-1, compute_fb_operator=False):
    """Solve system identification using least-squares over all epochs.

    Treats each epoch as a different window of time to estimate the model:

    .. math::
        X(t+1) = \\sum_{i=0}^{order} A_i X(t - i)

    where ``data`` comprises of ``(n_signals, n_times)`` and ``X(t)`` are
    the data snapshots.
    """
    # 1. determine shape of the window of data
    n_epochs, n_nodes, n_times = data.shape

    model_params = {
        'l2_reg': l2_reg,
        'lags': lags,
        'trend': trend,
        # 'random_state': random_state,
        'compute_fb_operator': compute_fb_operator
    }

    # storage for the A matrices, residuals and sum of squared estimated errors
    A_mats = np.zeros((n_epochs, n_nodes * lags, n_nodes))
    residuals = np.zeros((n_epochs, n_nodes, n_times - lags))
    sse_matrix = np.zeros((n_epochs, n_nodes, n_nodes))

    # compute the A matrix for all Epochs
    if n_jobs == 1:
        for idx in tqdm(range(n_epochs)):
            adjmat, resid, omega = _compute_lds_func(
                data[idx, ...], **model_params)
            # add additional order models in dynamic connectivity
            # along the first node axes
            for jdx in range(lags):
                A_mats[idx, jdx * n_nodes: n_nodes * (jdx + 1), :] = adjmat[
                    jdx * n_nodes: n_nodes * (jdx + 1), :
                ].T
            residuals[idx, ...] = resid.T
            sse_matrix[idx, ...] = omega
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
            adjmat, resid, omega = results[idx]
            residuals[idx, ...] = resid.T
            sse_matrix[idx, ...] = omega

            # add additional order models in dynamic connectivity
            # along the first node axes
            for jdx in range(lags):
                A_mats[idx, jdx * n_nodes: n_nodes * (jdx + 1), :] = adjmat[
                    jdx * n_nodes: n_nodes * (jdx + 1), :
                ].T

    # create connectivity
    A_mats = A_mats.reshape((n_epochs, -1))
    conn = EpochConnectivity(data=A_mats, n_nodes=n_nodes, names=names,
                             n_epochs_used=n_epochs,
                             times_used=times,
                             method='Time-varying LDS',
                             **model_params)
    return conn


def _compute_lds_func(data, lags, l2_reg, trend, compute_fb_operator):
    """Compute linear system using VAR model.

    Allows for parallelization over epochs.

    Note
    ----
    The ``_estimate_var`` function returns a set of A matrices that represent
    the system:

        X(t+1) = X(t) A

    Whereas we would like the system:

        X(t+1) = A X(t)

    Therefore, a transpose is needed. If there are additional lags, then each
    of these matrices need to be transposed.
    """
    # make sure data is T x K (samples, coefficients) to make use of underlying
    # functions
    data = data.T

    # get time-shifted versions
    X = data[:, :]
    A, resid, omega = _estimate_var(X, lags=lags, offset=0, trend=trend,
                                    l2_reg=l2_reg)

    if compute_fb_operator:
        # compute backward linear operator
        # original method
        back_A, back_resid, back_omega = _estimate_var(
            X[::-1, :], lags=lags, offset=0, trend=trend, l2_reg=l2_reg)
        A = sqrtm(A.dot(np.linalg.inv(back_A)))
        A = A.real  # remove numerical noise

    return A, resid, omega


def _get_trendorder(trend='c'):
    # Handle constant, etc.
    if trend == 'c':
        trendorder = 1
    elif trend == 'n':
        trendorder = 0
    elif trend == 'ct':
        trendorder = 2
    elif trend == 'ctt':
        trendorder = 3
    return trendorder


def _estimate_var(X, lags, Y=None, offset=0, trend='n', l2_reg=0):
    """Estimate a VAR model.

    Parameters
    ----------
    X : np.ndarray (n_times, n_channels)
        Endogenous variable, that predicts the exogenous.
    lags : int
        Lags of the endogenous variable.
    Y : np.ndarray (n_ytimes, n_ychannels), optional
        Exogenous variable that is additionally passed in to influence the
        endogenous variable.
    offset : int, optional
        Periods to drop from the beginning of the time-series, by default 0.
        Used for order selection, so it's an apples-to-apples comparison
    trend : str {"c", "ct", "ctt", "n"}
        "c" - add constant
        "ct" - constant and trend
        "ctt" - constant, linear and quadratic trend
        "n" - co constant, no trend
        Note that these are prepended to the columns of the dataset.
        By default 'n'
    l2_reg : int
        The amount of l2-regularization to use. Default of 0.

    Returns
    -------
    params : np.ndarray (lags, n_channels, n_channels)
        The coefficient state matrix that governs the linear system (VAR).
    resid : np.ndarray
        The residuals.
    omega : np.ndarray (n_channels, n_channels)
        Estimate of white noise process variance

    Notes
    -----
    This function was originally copied from statsmodels VAR model computation
    and modified for MNE-connectivity usage.
    """
    # determine the type of trend
    k_trend = _get_trendorder(trend)

    # get the number of observations
    n_total_obs, n_channels = X.shape
    n_obs = n_total_obs - lags - offset

    # get the number of equations we want
    n_equations = X.shape[1]

    # possibly offset the endogenous variable over the samples
    endog = X[offset:, :]

    # build the predictor matrix using the endogenous data
    # with lags and trends.
    # Note that the pure endogenous VAR model with OLS
    # makes this matrix a (n_samples - lags, n_channels * lags) matrix
    temp_z = _get_var_predictor_matrix(
        endog, lags, trend=trend, has_constant="raise"
    )

    # if exogenous variable is passed in, we will modify the predictor matrix
    # to account for additional exogenous variables that are accounted for
    # in the VAR model.
    if Y is not None:
        exog = Y[offset:, :]
        # build the predictor matrix using the exogenous data to add
        # possible trends based on the exogenous data
        # Note: if trend is 'n', then this will simply be an empty matrix
        x = _get_var_predictor_matrix(
            exog[-n_obs:, :], 0, trend=trend, has_constant="raise"
        )

        # store the observations in reverse order
        x_reverse = exog[-n_obs:, :]

        # stack the observations
        x = np.column_stack((x, x_reverse))
        del x_reverse  # free memory

        # create a 2T x K + Kp array
        z = np.empty((x.shape[0], x.shape[1] + temp_z.shape[1]))
        z[:, : k_trend] = temp_z[:, : k_trend]
        z[:, k_trend: k_trend + x.shape[1]] = x
        z[:, k_trend + x.shape[1]:] = temp_z[:, k_trend:]
        del temp_z, x  # free memory
    else:
        z = temp_z

    # the following modification of z is necessary to get the same results
    # as JMulTi for the constant-term-parameter...
    for i in range(k_trend):
        if (np.diff(z[:, i]) == 1).all():  # modify the trend-column
            z[:, i] += lags
        # make the same adjustment for the quadratic term
        if (np.diff(np.sqrt(z[:, i])) == 1).all():
            z[:, i] = (np.sqrt(z[:, i]) + lags) ** 2

    y_sample = endog[lags:]

    # Lütkepohl p75, about 5x faster than stated formula
    if l2_reg != 0:
        params = np.linalg.lstsq(z.T @ z + l2_reg * np.eye(n_channels * lags),
                                 z.T @ y_sample, rcond=1e-15)[0]
    else:
        params = np.linalg.lstsq(z, y_sample, rcond=1e-15)[0]

    # (n_samples - lags, n_channels)
    resid = y_sample - np.dot(z, params)

    # compute the degrees of freedom in residual calculation
    avobs = len(y_sample)
    if Y is not None:
        k_trend += exog.shape[1]
    df_resid = avobs - (n_equations * lags + k_trend)

    # K x K sse
    sse = np.dot(resid.T, resid)
    if df_resid:
        omega = sse / df_resid
    else:
        omega = np.full_like(sse, np.nan)

    return params, resid, omega


def _get_var_predictor_matrix(y, lags, trend='c', has_constant='skip'):
    """Make predictor matrix for VAR(p) process, Z.

    Parameters
    ----------
    y : np.ndarray (n_samples, n_channels)
        The passed in data array.
    lags : int
        The number of lags.
    trend : str, optional
        [description], by default 'c'
    has_constant : str, optional
        Can be 'raise', 'add', or 'skip'. See add_constant. By default 'skip'

    Returns
    -------
    Z : np.ndarray (n_samples, n_channels * lag_order)
        Z is a (T x Kp) matrix, with K the number of channels,
        p the lag order, and T the number of samples.
        Z := (Z_0, ..., Z_T).T (T x Kp)
        Z_t = [1 y_t y_{t-1} ... y_{t - p + 1}] (Kp x 1)

    References
    ----------
    Ref: Lütkepohl p.70 (transposed)
    """
    nobs = len(y)
    # Ravel C order, need to put in descending order
    Z = np.array([y[t - lags: t][::-1].ravel() for t in range(lags, nobs)])

    # Add constant, trend, etc.
    if trend != 'n':
        Z = _add_trend(Z, trend=trend, prepend=True,
                       has_constant=has_constant)

    return Z


def _add_trend(x, trend="c", prepend=False, has_constant="skip"):
    """
    Add a trend and/or constant to an array.

    Parameters
    ----------
    x : array_like (n_observations, n_coefficients)
        Original array of data.
    trend : str {'n', 'c', 't', 'ct', 'ctt'}
        The trend to add.

        * 'n' add no trend.
        * 'c' add constant only.
        * 't' add trend only.
        * 'ct' add constant and linear trend.
        * 'ctt' add constant and linear and quadratic trend.
    prepend : bool
        If True, prepends the new data to the columns of X.
    has_constant : str {'raise', 'add', 'skip'}
        Controls what happens when trend is 'c' and a constant column already
        exists in x. 'raise' will raise an error. 'add' will add a column of
        1s. 'skip' will return the data without change. 'skip' is the default.

    Returns
    -------
    array_like
        The original data with the additional trend columns.

    Notes
    -----
    Returns columns as ['ctt','ct','c'] whenever applicable. There is currently
    no checking for an existing trend.
    """
    prepend = _check_option("prepend", prepend, [True, False])
    trend = _check_option(
        "trend", trend, allowed_values=("n", "c", "t", "ct", "ctt"))
    has_constant = _check_option(
        "has_constant", has_constant, allowed_values=("raise", "add", "skip")
    )

    # TODO: could be generalized for trend of aribitrary order
    columns = ["const", "trend", "trend_squared"]
    if trend == "n":
        return x.copy()
    elif trend == "c":  # handles structured arrays
        columns = columns[:1]
        trendorder = 0
    elif trend == "ct" or trend == "t":
        columns = columns[:2]
        if trend == "t":
            columns = columns[1:2]
        trendorder = 1
    elif trend == "ctt":
        trendorder = 2

    x = np.asanyarray(x)
    nobs = len(x)
    trendarr = np.vander(
        np.arange(1, nobs + 1, dtype=np.float64), trendorder + 1
    )

    # put in order ctt
    trendarr = np.fliplr(trendarr)
    if trend == "t":
        trendarr = trendarr[:, 1]

    if "c" in trend:
        ptp0 = np.ptp(np.asanyarray(x), axis=0)
        col_is_const = ptp0 == 0
        nz_const = col_is_const & (x[0] != 0)
        col_const = nz_const

        if np.any(col_const):
            if has_constant == "raise":
                if x.ndim == 1:
                    base_err = "x is constant."
                else:
                    columns = np.arange(x.shape[1])[col_const]
                    const_cols = ", ".join([str(c) for c in columns])
                    base_err = (
                        "x contains one or more constant columns. Column(s) "
                        f"{const_cols} are constant."
                    )
                msg = f"{base_err} Adding a constant with trend='{trend}' is "\
                    "not allowed."
                raise ValueError(msg)
            elif has_constant == "skip":
                columns = columns[1:]
                trendarr = trendarr[:, 1:]

    order = 1 if prepend else -1
    x = [trendarr, x]
    x = np.column_stack(x[::order])
    return x
