import numpy as np
import scipy

from mne.utils import verbose, check_random_state
from .utils import fill_doc, parallel_loop


def _calc_q_h0(n, x, h, nt, n_jobs=1, verbose=0, random_state=None):
    """Calculate statistic under the null hypothesis of whiteness.
    """
    rng = check_random_state(random_state)
    par, func = parallel_loop(_calc_q_statistic, n_jobs, verbose)
    statistic = par(func(rng.permutation(x.T).T, h, nt) for _ in range(n))

    return np.array(statistic)


def _calc_q_statistic(x, h, nt):
    """Calculate Portmanteau statistics up to a lag of h.
    """
    t, m, n = x.shape

    # covariance matrix of x
    c0 = acm(x, 0)

    # LU factorization of covariance matrix
    c0f = scipy.linalg.lu_factor(c0, overwrite_a=False, check_finite=True)

    statistic = np.zeros((3, h + 1))
    for l in range(1, h + 1):
        cl = acm(x, l)

        # calculate tr(cl' * c0^-1 * cl * c0^-1)
        a = scipy.linalg.lu_solve(c0f, cl)
        b = scipy.linalg.lu_solve(c0f, cl.T)
        tmp = a.dot(b).trace()

        # Box-Pierce
        statistic[0, l] = tmp

        # Ljung-Box
        statistic[1, l] = tmp / (nt - l)

        # Li-McLeod
        statistic[2, l] = tmp

    statistic *= nt
    statistic[1, :] *= (nt + 2)

    statistic = np.cumsum(statistic, axis=1)

    for l in range(1, h + 1):
        statistic[2, l] = statistic[0, l] + m * m * l * (l + 1) / (2 * nt)

    return statistic


@verbose
def portmanteau(data, max_lag, model_order=0, n_repeats=100, get_q=False,
                random_state=None, n_jobs=-1, verbose=True):
    """Test if signals are white (serially uncorrelated up to a lag of max_lag).

    This function calculates the Li-McLeod Portmanteau test statistic Q to test
    against the null hypothesis H0 (the residuals are white) :footcite:`kilian_2006`.
    Surrogate data for H0 is created by sampling from random permutations of
    the residuals.

    Usually, the returned p-value is compared against a pre-defined type I
    error level of alpha=0.05 or alpha=0.01. If p<=alpha, the hypothesis of
    white residuals is rejected, which indicates that the VAR model does not
    adequately describe the data.

    Parameters
    ----------
    data : array, shape (trials, channels, samples) or (channels, samples)
        Epoched or continuous data set.
    max_lag : int
        Maximum lag that is included in the test statistic.
    p : int, optional
        Model order (if `data` are the residuals resulting from fitting a VAR
        model).
    repeats : int, optional
        Number of samples to create under the null hypothesis.
    get_q : bool, optional
        Return Q statistic along with *p*-value
    %(random_state)s
    %(n_jobs)s
    %(verbose)s

    Returns
    -------
    pvalue : float
        Probability of observing a more extreme value of Q under the assumption
        that H0 is true.
    q0 : list of float, optional (`get_q`)
        Individual surrogate estimates that were used for estimating the
        distribution of Q under H0.
    statistic : float, optional (`get_q`)
        Value of the Q statistic of the residuals.

    Notes
    -----
    According to :footcite:`Hosking1980`, max_lag must satisfy max_lag = O(n^0.5), where n is the length (time
    samples) of the residuals.

    References
    ----------
    .. footbibliography::
    """
    res = data[:, :, model_order:]
    t, m, n = res.shape
    nt = (n - model_order) * t

    q0 = _calc_q_h0(n_repeats, res, max_lag, nt, n_jobs, verbose,
                    random_state=random_state)[:, 2, -1]
    statistic = _calc_q_statistic(res, max_lag, nt)[2, -1]

    # probability of observing a result more extreme than statistic
    # under the null-hypothesis
    pvalue = np.sum(q0 >= statistic) / n_repeats

    if get_q:
        return pvalue, q0, statistic
    else:
        return pvalue
