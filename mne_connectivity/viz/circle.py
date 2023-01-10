"""Functions to plot on circle as for connectivity."""

# Authors: Adam Li <adam2392@gmail.com>
#          Alex Rockhill <aprockhill@mailbox.org>
#
# License: Simplified BSD

from mne.utils import warn
from mne.viz.circle import _plot_connectivity_circle


def plot_connectivity_circle(con, node_names, indices=None, n_lines=None,
                             node_angles=None, node_width=None,
                             node_height=1.0, node_colors=None,
                             facecolor='black', textcolor='white',
                             node_edgecolor='black', linewidth=1.5,
                             colormap='hot', vmin=None, vmax=None,
                             colorbar=True, title=None,
                             colorbar_size=0.2, colorbar_pos=(-0.3, 0.1),
                             fontsize_title=12, fontsize_names=8,
                             fontsize_colorbar=8, padding=6., ax=None,
                             fig=None, subplot=None, interactive=True,
                             node_linewidth=2., show=True):
    """Visualize connectivity as a circular graph.

    Parameters
    ----------
    con : array | Connectivity
        Connectivity scores. Can be a square matrix, or a 1D array. If a 1D
        array is provided, "indices" has to be used to define the connection
        indices.
    node_names : list of str
        Node names. The order corresponds to the order in con.
    indices : tuple of array | None
        Two arrays with indices of connections for which the connections
        strengths are defined in con. Only needed if con is a 1D array.
    n_lines : int | None
        If not None, only the n_lines strongest connections (strength=abs(con))
        are drawn.
    node_angles : array, shape (n_node_names,) | None
        Array with node positions in degrees. If None, the nodes are equally
        spaced on the circle. See mne.viz.circular_layout.
    node_width : float | None
        Width of each node in degrees. If None, the minimum angle between any
        two nodes is used as the width.
    node_height : float
        The relative height of the colored bar labeling each node. Default 1.0
        is the standard height.
    node_colors : list of tuple | list of str
        List with the color to use for each node. If fewer colors than nodes
        are provided, the colors will be repeated. Any color supported by
        matplotlib can be used, e.g., RGBA tuples, named colors.
    facecolor : str
        Color to use for background. See matplotlib.colors.
    textcolor : str
        Color to use for text. See matplotlib.colors.
    node_edgecolor : str
        Color to use for lines around nodes. See matplotlib.colors.
    linewidth : float
        Line width to use for connections.
    colormap : str | instance of matplotlib.colors.LinearSegmentedColormap
        Colormap to use for coloring the connections.
    vmin : float | None
        Minimum value for colormap. If None, it is determined automatically.
    vmax : float | None
        Maximum value for colormap. If None, it is determined automatically.
    colorbar : bool
        Display a colorbar or not.
    title : str
        The figure title.
    colorbar_size : float
        Size of the colorbar.
    colorbar_pos : tuple, shape (2,)
        Position of the colorbar.
    fontsize_title : int
        Font size to use for title.
    fontsize_names : int
        Font size to use for node names.
    fontsize_colorbar : int
        Font size to use for colorbar.
    padding : float
        Space to add around figure to accommodate long labels.
    ax : instance of matplotlib PolarAxes | None
        The axes to use to plot the connectivity circle.
    fig : None | instance of matplotlib.figure.Figure
        The figure to use. If None, a new figure with the specified background
        color will be created.

        Deprecated: will be removed in version 0.5.

    subplot : int | tuple, shape (3,)
        Location of the subplot when creating figures with multiple plots. E.g.
        121 or (1, 2, 1) for 1 row, 2 columns, plot 1. See
        matplotlib.pyplot.subplot.

        Deprecated: will be removed in version 0.5.

    interactive : bool
        When enabled, left-click on a node to show only connections to that
        node. Right-click shows all connections.
    node_linewidth : float
        Line with for nodes.
    show : bool
        Show figure if True.

    Returns
    -------
    fig : instance of matplotlib.figure.Figure
        The figure handle.
    ax : instance of matplotlib.projections.polar.PolarAxes
        The subplot handle.

    Notes
    -----
    This code is based on a circle graph example by Nicolas P. Rougier

    By default, :func:`matplotlib.pyplot.savefig` does not take ``facecolor``
    into account when saving, even if set when a figure is generated. This
    can be addressed via, e.g.::

    >>> fig.savefig(fname_fig, facecolor='black') # doctest:+SKIP

    If ``facecolor`` is not set via :func:`matplotlib.pyplot.savefig`, the
    figure labels, title, and legend may be cut off in the output figure.
    """
    import matplotlib.pyplot as plt
    from mne_connectivity.base import BaseConnectivity

    if isinstance(con, BaseConnectivity):
        con = con.get_data()

    if fig is not None or subplot is not None:
        warn('Passing a `fig` and `subplot` is deprecated and not be '
             'supported after mne-connectivity version 0.4. Please '
             'use the `ax` argument and pass a matplotlib axes object '
             'with polar coordinates instead', DeprecationWarning)
        if ax is None:  # don't overwrite ax if passed
            if fig is None:
                fig = plt.figure(figsize=(8, 8), facecolor=facecolor)
            if not isinstance(subplot, tuple):
                subplot = (subplot,)
            ax = plt.subplot(*subplot, polar=True)

    return _plot_connectivity_circle(
        con=con, node_names=node_names, indices=indices, n_lines=n_lines,
        node_angles=node_angles, node_width=node_width,
        node_height=node_height, node_colors=node_colors,
        facecolor=facecolor, textcolor=textcolor,
        node_edgecolor=node_edgecolor, linewidth=linewidth,
        colormap=colormap, vmin=vmin, vmax=vmax, colorbar=colorbar,
        title=title, colorbar_size=colorbar_size, colorbar_pos=colorbar_pos,
        fontsize_title=fontsize_title, fontsize_names=fontsize_names,
        fontsize_colorbar=fontsize_colorbar, padding=padding, ax=ax,
        interactive=interactive, node_linewidth=node_linewidth, show=show)
