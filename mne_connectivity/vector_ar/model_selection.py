from collections import defaultdict
import numpy as np
from scipy import linalg

from .var import _estimate_var


def select_order(X, maxlags=None):
    """Compute lag order selections based on information criterion.

    Selects a lag order based on each of the available information
    criteria.

    Parameters
    ----------
    X : np.ndarray, shape (n_times, n_channels)
        Endogenous variable, that predicts the exogenous.
    maxlags : int
        The maximum number of lags to check. Will then check from
        ``1`` to ``maxlags``. If None, defaults to
        ``12 * (n_times / 100.)**(1./4)``.

    Returns
    -------
    selected_orders : dict
        The selected orders based on the following information criterion.
        * aic : Akaike
        * fpe : Final prediction error
        * hqic : Hannan-Quinn
        * bic : Bayesian a.k.a. Schwarz

        The selected order is then stored as the value.
    """
    # get the number of observations
    n_total_obs, n_equations = X.shape

    ntrend = 0
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

    p_min = 1
    for p in range(p_min, maxlags + 1):
        # exclude some periods to same amount of data used for each lag
        # order
        params, _, sigma_u = _estimate_var(
            X, lags=p, offset=maxlags - p)

        info_criteria = _info_criteria(params, X, sigma_u=sigma_u,
                                       lags=p)
        for k, v in info_criteria.items():
            ics[k].append(v)

    selected_orders = dict(
        (k, np.argmin(v) + p_min) for k, v in ics.items()
    )
    return selected_orders


def _logdet_symm(m):
    """Return log(det(m)) asserting positive definiteness of m.

    Parameters
    ----------
    m : np.ndarray, shape (N, N)
        2d array that is positive-definite (and symmetric)

    Returns
    -------
    logdet : float
        The log-determinant of m.
    """
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
    sigma_u : np.ndarray, shape (n_channels, n_channels)
        Estimate of white noise process variance

    Returns
    -------
    sigma_u_mle : float
        The biased MLE of noise process covariance.
    """
    return sigma_u * df_resid / nobs


def _info_criteria(params, X, sigma_u, lags):
    """Compute information criteria for lagorder selection.

    Parameters
    ----------
    params : np.ndarray, shape (lags, n_channels, n_channels)
        The coefficient state matrix that governs the linear system (VAR).
    X : np.ndarray (n_times, n_channels)
        Endogenous variable, that predicts the exogenous.
    sigma_u : np.ndarray, shape (n_channels, n_channels)
        Estimate of white noise process variance
    lags : int
        Lags of the endogenous variable.

    Returns
    -------
    result : dict
        The AIC, BIC, HQIC and FPE.
    """
    n_totobs, neqs = X.shape
    nobs = n_totobs - lags
    lag_order = lags
    k_trend = 0
    k_ar = lags
    endog_start = k_trend

    # compute the number of free parameters for degrees of freedom
    coefs_exog = params[:endog_start].T
    k_exog = coefs_exog.shape[1]
    free_params = lag_order * neqs ** 2 + neqs * k_exog

    # compute the
    df_model = neqs * k_ar + k_exog
    df_resid = nobs - df_model
    ld = _logdet_symm(_sigma_u_mle(df_resid, nobs, sigma_u))

    # See Lütkepohl pp. 146-150
    aic = ld + (2.0 / nobs) * free_params
    bic = ld + (np.log(nobs) / nobs) * free_params
    hqic = ld + (2.0 * np.log(np.log(nobs)) / nobs) * free_params
    fpe = ((nobs + df_model) / df_resid) ** neqs * np.exp(ld)

    return {"aic": aic, "bic": bic, "hqic": hqic, "fpe": fpe}
