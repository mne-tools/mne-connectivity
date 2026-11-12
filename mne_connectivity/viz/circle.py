"""Functions to plot on circle as for connectivity."""

# Authors: Adam Li <adam2392@gmail.com>
#          Alex Rockhill <aprockhill@mailbox.org>
#
# License: Simplified BSD

import numpy as np
from mne.viz.circle import _plot_connectivity_circle


def plot_connectivity_circle(
    con,
    node_names,
    indices=None,
    n_lines=None,
    node_angles=None,
    node_width=None,
    node_height=1.0,
    node_colors=None,
    facecolor="black",
    textcolor="white",
    node_edgecolor="black",
    linewidth=1.5,
    colormap=None,
    vmin=None,
    vmax=None,
    colorbar=True,
    title=None,
    colorbar_size=0.2,
    colorbar_pos=(-0.3, 0.1),
    fontsize_title=12,
    fontsize_names=8,
    fontsize_colorbar=8,
    padding=6.0,
    ax=None,
    interactive=True,
    node_linewidth=2.0,
    show=True,
):
    """Visualize connectivity as a circular graph.

    Parameters
    ----------
    con : array | Connectivity
        Connectivity scores. Can be a square matrix, or a 1D array. If a 1D array is
        provided, ``indices`` has to be used to define the connection indices.
    node_names : list of str
        Node names. The order corresponds to the order in ``con``.
    indices : tuple of array_like | None
        Two arrays with indices of connections for which the connections strengths are
        defined in ``con``. Only needed if ``con`` is a 1D array.
    n_lines : int | None
        If not ``None``, only the ``n_lines`` strongest connections
        (``strength=abs(con)``) are drawn.
    node_angles : array, shape (n_node_names,) | None
        Array with node positions in degrees. If ``None``, the nodes are equally spaced
        on the circle. See :func:`mne.viz.circular_layout`.
    node_width : float | None
        Width of each node in degrees. If ``None``, the minimum angle between any two
        nodes is used as the width.
    node_height : float
        The relative height of the colored bar labeling each node. Default 1.0 is the
        standard height.
    node_colors : list of tuple | list of str
        List with the color to use for each node. If fewer colors than nodes are
        provided, the colors will be repeated. Any color supported by matplotlib can be
        used, e.g., RGBA tuples, named colors.
    facecolor : str
        Color to use for background. See :mod:`matplotlib.colors`.
    textcolor : str
        Color to use for text. See :mod:`matplotlib.colors`.
    node_edgecolor : str
        Color to use for lines around nodes. See :mod:`matplotlib.colors`.
    linewidth : float
        Line width to use for connections.
    colormap : str | instance of matplotlib.colors.Colormap | None
        Colormap to use for coloring the connections. If ``None`` (default),
        ``"RdBu_r"`` is used when the plotted data has both positive and negative
        values, otherwise ``"hot"``.

        .. versionchanged:: 1.0
           The default changed from ``"hot"`` to ``None``.
    vmin : float | None
        Minimum value for colormap. If ``None``, it is determined automatically.
    vmax : float | None
        Maximum value for colormap. If ``None``, it is determined automatically.
        If both ``vmin`` and ``vmax`` are ``None`` and the plotted data has both
        positive and negative values, the limits are symmetric around zero.

        .. versionchanged:: 1.0
           Automatic limits for data with both signs are symmetric around zero.
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
    ax : instance of matplotlib.projections.polar.PolarAxes | None
        The axes to use to plot the connectivity circle.
    interactive : bool
        When enabled, left-click on a node to show only connections to that node.
        Right-click shows all connections.
    node_linewidth : float
        Line width for nodes.
    show : bool
        Show figure if ``True``.

    Returns
    -------
    fig : instance of matplotlib.figure.Figure
        The figure handle.
    ax : instance of matplotlib.projections.polar.PolarAxes
        The subplot handle.

    Notes
    -----
    This code is based on a circle graph example by Nicolas P. Rougier.

    By default, :func:`matplotlib.pyplot.savefig` does not take ``facecolor`` into
    account when saving, even if set when a figure is generated. This can be addressed
    via, e.g.::

    >>> fig.savefig(fname_fig, facecolor='black') # doctest:+SKIP

    If ``facecolor`` is not set via :func:`matplotlib.pyplot.savefig`, the figure
    labels, title, and legend may be cut off in the output figure.
    """
    from mne_connectivity.base import BaseConnectivity

    if isinstance(con, BaseConnectivity):
        con = con.get_data()

    # choose the colormap and limits
    con = np.asarray(con)
    if con.ndim == 2 and con.shape[0] == con.shape[1]:
        values = con[np.tril_indices(con.shape[0], -1)]
    else:
        values = con.ravel()
    values = values[np.isfinite(values)]
    signed = (
        not np.iscomplexobj(values)
        and values.size > 0
        and values.min() < 0 < values.max()
    )
    if colormap is None:
        colormap = "RdBu_r" if signed else "hot"
    if signed and vmin is None and vmax is None:
        vmax = np.abs(values).max()
        vmin = -vmax

    return _plot_connectivity_circle(
        con=con,
        node_names=node_names,
        indices=indices,
        n_lines=n_lines,
        node_angles=node_angles,
        node_width=node_width,
        node_height=node_height,
        node_colors=node_colors,
        facecolor=facecolor,
        textcolor=textcolor,
        node_edgecolor=node_edgecolor,
        linewidth=linewidth,
        colormap=colormap,
        vmin=vmin,
        vmax=vmax,
        colorbar=colorbar,
        title=title,
        colorbar_size=colorbar_size,
        colorbar_pos=colorbar_pos,
        fontsize_title=fontsize_title,
        fontsize_names=fontsize_names,
        fontsize_colorbar=fontsize_colorbar,
        padding=padding,
        ax=ax,
        interactive=interactive,
        node_linewidth=node_linewidth,
        show=show,
    )
