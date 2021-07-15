
def autocov(x, l):
    """Compute autocovariance matrix at lag l.

    This function calculates the autocovariance matrix of `x` at lag `l`.

    Parameters
    ----------
    x : array, shape (n_trials, n_channels, n_samples)
        Signal data (2D or 3D for multiple trials)
    l : int
        Lag

    Returns
    -------
    c : ndarray, shape = [nchannels, n_channels]
        Autocovariance matrix of `x` at lag `l`.
    """
    x = atleast_3d(x)

    if l > x.shape[2]-1:
        raise AttributeError("lag exceeds data length")

    ## subtract mean from each trial
    #for t in range(x.shape[2]):
    #    x[:, :, t] -= np.mean(x[:, :, t], axis=0)

    if l == 0:
        a, b = x, x
    else:
        a = x[:, :, l:]
        b = x[:, :, 0:-l]

    c = np.zeros((x.shape[1], x.shape[1]))
    for t in range(x.shape[0]):
        c += a[t, :, :].dot(b[t, :, :].T) / a.shape[2]
    c /= x.shape[0]

    return c.T