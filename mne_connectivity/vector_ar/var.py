import numpy as np
import scipy
from scipy.linalg import sqrtm
from tqdm import tqdm
from mne import BaseEpochs

from mne.utils import logger, verbose

from ..utils import fill_doc
from ..base import Connectivity, EpochConnectivity, EpochTemporalConnectivity


@verbose
@fill_doc
def vector_auto_regression(
        data, times=None, names=None, lags=1, l2_reg=0.0,
        compute_fb_operator=False, model='dynamic', n_jobs=1, verbose=None):
    """Compute vector auto-regresssive (VAR) model.

    Parameters
    ----------
    data : array-like, shape=(n_epochs, n_signals, n_times) | Epochs | generator
        The data from which to compute connectivity. The epochs dimension
        is interpreted differently, depending on ``'output'`` argument.
    times : array-like
        (Optional) The time points used to construct the epoched ``data``. If
        ``None``, then ``times_used`` in the Connectivity will not be
        available.
    %(names)s
    lags : int, optional
        Autoregressive model order, by default 1.
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
    """  # noqa
    if model not in ['avg-epochs', 'dynamic']:
        raise ValueError(f'"model" parameter must be one of '
                         f'(avg-epochs, dynamic), not {model}.')

    events = None
    event_id = None
    if isinstance(data, BaseEpochs):
        names = data.ch_names
        events = data.events
        event_id = data.event_id
        times = data.times

        # Extract metadata from the Epochs data structure.
        # Make Annotations persist through by adding them to the metadata.
        metadata = data.metadata
        if metadata is None:
            annots_in_metadata = False
        else:
            annots_in_metadata = all(
                name not in metadata.columns for name in [
                    'annot_onset', 'annot_duration', 'annot_description'])
        if hasattr(data, 'annotations') and not annots_in_metadata:
            data.add_annotations_to_metadata(overwrite=True)
        metadata = data.metadata

        # get the actual data in numpy
        data = data.get_data()
    else:
        metadata = None

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
                            method='VAR', metadata=metadata,
                            events=events, event_id=event_id,
                            **model_params)
    else:
        assert model == 'dynamic'
        # compute time-varying VAR model where each epoch
        # is one sample of a time-varying multivariate time-series
        # linear system
        A_mats = _system_identification(
            data=data, lags=lags,
            l2_reg=l2_reg, n_jobs=n_jobs,
            compute_fb_operator=compute_fb_operator)
        # create connectivity
        if lags > 1:
            conn = EpochTemporalConnectivity(data=A_mats,
                                             times=list(range(lags)),
                                             n_nodes=n_nodes, names=names,
                                             n_epochs_used=n_epochs,
                                             times_used=times,
                                             method='Time-varying VAR(p)',
                                             metadata=metadata,
                                             events=events, event_id=event_id,
                                             **model_params)
        else:
            conn = EpochConnectivity(data=A_mats, n_nodes=n_nodes,
                                     names=names,
                                     n_epochs_used=n_epochs,
                                     times_used=times,
                                     method='Time-varying VAR(1)',
                                     metadata=metadata,
                                     events=events, event_id=event_id,
                                     **model_params)
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


def _system_identification(data, lags, l2_reg=0,
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
        'compute_fb_operator': compute_fb_operator
    }

    # storage for the A matrices, residuals and sum of squared estimated errors
    A_mats = np.zeros((n_epochs, n_nodes, n_nodes, lags))
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
                A_mats[idx, :, :, jdx] = adjmat[
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
                A_mats[idx, :, :, jdx] = adjmat[
                    jdx * n_nodes: n_nodes * (jdx + 1), :
                ].T

    # ravel the matrix
    if lags == 1:
        A_mats = A_mats.reshape((n_epochs, -1))
    else:
        A_mats = A_mats.reshape((n_epochs, -1, lags))
    return A_mats


def _compute_lds_func(data, lags, l2_reg, compute_fb_operator):
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
    A, resid, omega = _estimate_var(X, lags=lags, offset=0,
                                    l2_reg=l2_reg)

    if compute_fb_operator:
        # compute backward linear operator
        # original method
        back_A, back_resid, back_omega = _estimate_var(
            X[::-1, :], lags=lags, offset=0, l2_reg=l2_reg)
        A = sqrtm(A.dot(np.linalg.inv(back_A)))
        A = A.real  # remove numerical noise

    return A, resid, omega


def _estimate_var(X, lags, offset=0, l2_reg=0):
    """Estimate a VAR model.

    Parameters
    ----------
    X : np.ndarray (n_times, n_channels)
        Endogenous variable, that predicts the exogenous.
    lags : int
        Lags of the endogenous variable.
    offset : int, optional
        Periods to drop from the beginning of the time-series, by default 0.
        Used for order selection, so it's an apples-to-apples comparison
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
    # get the number of equations we want
    n_equations = X.shape[1]

    # possibly offset the endogenous variable over the samples
    endog = X[offset:, :]

    # build the predictor matrix using the endogenous data
    # with lags and trends.
    # Note that the pure endogenous VAR model with OLS
    # makes this matrix a (n_samples - lags, n_channels * lags) matrix
    temp_z = _get_var_predictor_matrix(
        endog, lags
    )
    z = temp_z

    y_sample = endog[lags:]
    del endog, X
    # Lütkepohl p75, about 5x faster than stated formula
    if l2_reg != 0:
        params = np.linalg.lstsq(z.T @ z + l2_reg * np.eye(n_equations * lags),
                                 z.T @ y_sample, rcond=1e-15)[0]
    else:
        params = np.linalg.lstsq(z, y_sample, rcond=1e-15)[0]

    # (n_samples - lags, n_channels)
    resid = y_sample - np.dot(z, params)

    # compute the degrees of freedom in residual calculation
    avobs = len(y_sample)
    df_resid = avobs - (n_equations * lags)

    # K x K sse
    sse = np.dot(resid.T, resid)
    omega = sse / df_resid

    return params, resid, omega


def _test_forloop(X, lags, offset=0, l2_reg=0):
    # possibly offset the endogenous variable over the samples
    endog = X[offset:, :]

    # get the number of equations we want
    n_times, n_equations = endog.shape

    y_sample = endog[lags:]

    # X.T @ X coefficient matrix
    n_channels = n_equations * lags
    XdotX = np.zeros((n_channels, n_channels))

    # X.T @ Y ordinate / dependent variable matrix
    XdotY = np.zeros((n_channels, n_channels))

    # loop over sample points and aggregate the
    # necessary elements of the normal equations
    first_component = np.zeros((n_channels, 1))
    second_component = np.zeros((1, n_channels))
    y_component = np.zeros((1, n_channels))
    for idx in range(n_times - lags):
        for jdx in range(lags):
            first_component[
                jdx * n_equations: (jdx + 1) * n_equations, :] = \
                endog[idx + jdx, :][:, np.newaxis]
            second_component[:, jdx * n_equations: (
                jdx + 1) * n_equations] = endog[idx + jdx, :][np.newaxis, :]
            y_component[:, jdx * n_equations: (jdx + 1) *
                        n_equations] = endog[idx + 1 + jdx, :][np.newaxis, :]
        # second_component = np.hstack([endog[idx + jdx, :]
        # for jdx in range(lags)])[np.newaxis, :]
        # print(second_component.shape)
        # increment for X.T @ X
        XdotX += first_component @ second_component

        # increment for X.T @ Y
        # second_component = np.hstack([endog[idx + 1 + jdx, :]
        # for jdx in range(lags)])[np.newaxis, :]
        XdotY += first_component @ y_component

    if l2_reg != 0:
        final_params = np.linalg.lstsq(
            XdotX + l2_reg * np.eye(n_equations * lags), XdotY, rcond=1e-15)[0]
    else:
        final_params = np.linalg.lstsq(XdotX, XdotY, rcond=1e-15)[0].T

    # format the final matrix as (lags * n_equations, n_equations)
    params = np.empty((lags * n_equations, n_equations))
    for idx in range(lags):
        start_col = n_equations * idx
        stop_col = n_equations * (idx + 1)
        start_row = n_equations * (lags - idx - 1)
        stop_row = n_equations * (lags - idx)
        params[start_row:stop_row, ...] = final_params[
            n_equations * (lags - 1):, start_col:stop_col].T

    # print(final_params.round(5))
    # print(params_)
    # print(params)
    # build the predictor matrix using the endogenous data
    # with lags and trends.
    # Note that the pure endogenous VAR model with OLS
    # makes this matrix a (n_samples - lags, n_channels * lags) matrix
    # z = _get_var_predictor_matrix(
        # endog, lags
    # )
    # (n_samples - lags, n_channels)
    # resid = y_sample - np.dot(z, params)
    resid = np.zeros((n_times - lags, n_equations))

    # compute the degrees of freedom in residual calculation
    avobs = len(y_sample)
    df_resid = avobs - (n_equations * lags)

    # K x K sse
    sse = np.dot(resid.T, resid)
    omega = sse / df_resid
    return params, resid, omega


def _get_var_predictor_matrix(y, lags):
    """Make predictor matrix for VAR(p) process, Z.

    Parameters
    ----------
    y : np.ndarray (n_samples, n_channels)
        The passed in data array.
    lags : int
        The number of lags.

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

    return Z
