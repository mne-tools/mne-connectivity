import numpy as np
from collections import defaultdict

from .var import _estimate_var, _get_trendorder


def select_order(X, Y=None, maxlags=None, trend="n"):
    """
    Compute lag order selections based on each of the available information
    criteria

    Parameters
    ----------
    X : np.ndarray (n_times, n_channels)
        Endogenous variable, that predicts the exogenous.
    Y : np.ndarray (n_ytimes, n_ychannels), optional
        Exogenous variable that is additionally passed in to influence the
        endogenous variable.
    maxlags : int
        if None, defaults to 12 * (nobs/100.)**(1./4)
    trend : str {"n", "c", "ct", "ctt"}
        * "n" - no deterministic terms
        * "c" - constant term
        * "ct" - constant and linear term
        * "ctt" - constant, linear, and quadratic term
        Only ``n`` is currently implemented.

    Returns
    -------
    selected_orders : dict
        The selected orders based on the information criterion.
        aic : Akaike
        fpe : Final prediction error
        hqic : Hannan-Quinn
        bic : Bayesian a.k.a. Schwarz
    """
    if trend != 'n':
        raise RuntimeError(f'Trend {trend} is not implemented for yet.')

    # get the number of observations
    n_total_obs, n_equations = X.shape

    ntrend = len(trend) if trend.startswith("c") else 0
    max_estimable = (n_total_obs - n_equations - ntrend) // (1 + n_equations)
    if maxlags is None:
        maxlags = int(round(12 * (n_total_obs / 100.0) ** (1 / 4.0)))
        # TODO: This expression shows up in a bunch of places, but
        #  in some it is `int` and in others `np.ceil`.  Also in some
        #  it multiplies by 4 instead of 12.  Let's put these all in
        #  one place and document when to use which variant.

        # Ensure enough obs to estimate model with maxlags
        maxlags = min(maxlags, max_estimable)
    else:
        if maxlags > max_estimable:
            raise ValueError(
                "maxlags is too large for the number of observations and "
                "the number of equations. The largest model cannot be "
                "estimated."
            )

    # define dictionary of information criterions
    ics = defaultdict(list)

    p_min = 0 if Y is not None or trend != "n" else 1
    for p in range(p_min, maxlags + 1):
        # exclude some periods to same amount of data used for each lag
        # order
        params, resid, sigma_u = _estimate_var(
            X, lags=p, Y=Y, offset=maxlags - p, trend=trend)

        info_criteria = _info_criteria(params, X, exog=Y, sigma_u=sigma_u,
                                       lags=p, trend=trend)
        for k, v in info_criteria.items():
            ics[k].append(v)

    selected_orders = dict(
        (k, np.array(v).argmin() + p_min) for k, v in ics.items()
    )
    return selected_orders


def _logdet_symm(m, check_symm=False):
    """
    Return log(det(m)) asserting positive definiteness of m.

    Parameters
    ----------
    m : array_like (N, N)
        2d array that is positive-definite (and symmetric)

    Returns
    -------
    logdet : float
        The log-determinant of m.
    """
    from scipy import linalg
    if check_symm:
        if not np.all(m == m.T):  # would be nice to short-circuit check
            raise ValueError("m is not symmetric.")
    c, _ = linalg.cho_factor(m, lower=True)
    return 2 * np.sum(np.log(c.diagonal()))


def _sigma_u_mle(df_resid, nobs, sigma_u):
    """(Biased) maximum likelihood estimate of noise process covariance.

    Parameters
    ----------
    df_resid : int
        Number of observations minus number of estimated parameters.
    nobs : int
        Number of observations/samples in the dataset.
    sigma_u : np.ndarray (n_channels, n_channels)
        Estimate of white noise process variance

    Returns
    -------
    sigma_u_mle : float
        The biased MLE of noise process covariance.
    """
    if not df_resid:
        return np.zeros_like(sigma_u)
    return sigma_u * df_resid / nobs


def _info_criteria(params, X, exog, sigma_u, lags, trend):
    """Compute information criteria for lagorder selection.

    Parameters
    ----------
    params : np.ndarray (lags, n_channels, n_channels)
        The coefficient state matrix that governs the linear system (VAR).
    X : np.ndarray (n_times, n_channels)
        Endogenous variable, that predicts the exogenous.
    exog : [type]
        [description]
    sigma_u : np.ndarray (n_channels, n_channels)
        Estimate of white noise process variance
    lags : int
        Lags of the endogenous variable.
    trend : str {"n", "c", "ct", "ctt"}
        * "n" - no deterministic terms
        * "c" - constant term
        * "ct" - constant and linear term
        * "ctt" - constant, linear, and quadratic term

    Returns
    -------
    result : dict
        The AIC, BIC, HQIC and FPE.
    """
    n_totobs, neqs = X.shape
    nobs = n_totobs - lags
    lag_order = lags
    k_trend = _get_trendorder(trend)
    k_ar = lags
    endog_start = k_trend

    # construct coefficient matrices
    # Each matrix needs to be transposed
    if exog is not None:
        k_exog_user = exog.shape[1]
        endog_start += k_exog_user

    # compute the number of free parameters for degrees of freedom
    coefs_exog = params[:endog_start].T
    k_exog = coefs_exog.shape[1]
    free_params = lag_order * neqs ** 2 + neqs * k_exog

    # compute the
    df_model = neqs * k_ar + k_exog
    df_resid = nobs - df_model
    if df_resid:
        ld = _logdet_symm(_sigma_u_mle(df_resid, nobs, sigma_u))
    else:
        ld = -np.inf

    # See Lütkepohl pp. 146-150
    aic = ld + (2.0 / nobs) * free_params
    bic = ld + (np.log(nobs) / nobs) * free_params
    hqic = ld + (2.0 * np.log(np.log(nobs)) / nobs) * free_params
    if df_resid:
        fpe = ((nobs + df_model) / df_resid) ** neqs * np.exp(ld)
    else:
        fpe = np.inf

    return {"aic": aic, "bic": bic, "hqic": hqic, "fpe": fpe}
