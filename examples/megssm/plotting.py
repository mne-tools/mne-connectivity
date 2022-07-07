""" plotting functions """

import numpy as np
import matplotlib.pyplot as plt

def plot_At(A, ci='sd', times=None, ax=None, skipdiag=False, labels=None,
    showticks=True, **kwargs):
    """ plot traces of each entry of dynamics A in square grid of subplots """
    if A.ndim == 3:
        T, d, _ = A.shape
    elif A.ndim == 4:
        _, T, d, _ = A.shape

    if times is None:
        times = np.arange(T)

    if ax is None or ax.shape != (d, d):
        fig, ax = plt.subplots(d, d, sharex=True, sharey=True, squeeze=False)
    else:
        fig = ax[0, 0].figure

    for i in range(d):
        for j in range(d):

            # skip and hide subplots on diagonal
            if skipdiag and i == j:
                ax[i, j].set_visible(False)
                continue

            # plot A entry as trace with/without error band
            if A.ndim == 3:
                ax[i, j].plot(times[:-1], A[:-1, i, j], **kwargs)
            elif A.ndim == 4:
                plot_fill(A[:, :-1, i, j], ci=ci, times=times[:-1],
                          ax=ax[i, j], **kwargs)

            # add labels above first row and to the left of the first column
            if labels is not None:
                if i == 0 or (skipdiag and (i, j) == (1, 0)):
                    ax[i, j].set_title(labels[j], fontsize=12)
                if j == 0 or (skipdiag and (i, j) == (0, 1)):
                    ax[i, j].set_ylabel(labels[i], fontsize=12)

            # remove x- and y-ticks on subplot
            if not showticks:
                ax[i, j].set_xticks([])
                ax[i, j].set_yticks([])

    diag_lims = [0, 1]
    off_lims = [-0.25, 0.25]
    for ri, row in enumerate(ax):
        for ci, a in enumerate(row):
            ylim = diag_lims if ri == ci else off_lims
            a.set(ylim=ylim, xlim=times[[0, -1]])
            if ri == 0:
                a.set_title(a.get_title(), fontsize='small')
            if ci == 0:
                a.set_ylabel(a.get_ylabel(), fontsize='small')
            for line in a.lines:
                line.set_clip_on(False)
                line.set(lw=1.)
            if ci != 0:
                a.yaxis.set_major_formatter(plt.NullFormatter())
            if ri != len(labels) - 1:
                a.xaxis.set_major_formatter(plt.NullFormatter())
            if ri == ci:
                for spine in a.spines.values():
                    spine.set(lw=2)
            else:
                a.axhline(0, color='k', ls=':', lw=1.)
                
    return fig, ax

def plot_fill(X, times=None, ax=None, ci='sd', **kwargs):
    """ plot mean and error band across first axis of X """
    N, T = X.shape

    if times is None:
        times = np.arange(T)
    if ax is None:
        fig, ax = plt.subplots(1, 1)

    mu = np.mean(X, axis=0)

    # define lower and upper band limits based on ci
    if ci == 'sd':    # standard deviation
        sigma = np.std(X, axis=0)
        lower, upper = mu - sigma, mu + sigma
    elif ci == 'se':    # standard error
        stderr = np.std(X, axis=0) / np.sqrt(X.shape[0])
        lower, upper = mu - stderr, mu + stderr
    elif ci == '2sd':    # 2 standard deviations
        sigma = np.std(X, axis=0)
        lower, upper = mu - 2 * sigma, mu + 2 * sigma
    elif ci == 'max':    # range (min to max)
        lower, upper = np.min(X, axis=0), np.max(X, axis=0)
    elif type(ci) is float and 0 < ci < 1:
        # quantile-based confidence interval
        a = 1 - ci
        lower, upper = np.quantile(X, [a / 2, 1 - a / 2], axis=0)
    else:
        raise ValueError("ci must be in ('sd', 'se', '2sd', 'max') "
                         "or float in (0, 1)")

    lines = ax.plot(times, mu, **kwargs)
    c = lines[0].get_color()
    ax.fill_between(times, lower, upper, color=c, alpha=0.3, lw=0)
