from collections.abc import Callable
from functools import partial

import mne
import numpy as np
from matplotlib import pyplot as plt
from mne._fiff.pick import pick_info
from mne.utils.check import _check_option, _validate_type
from mne.utils.numerics import _time_mask
from mne.viz.circle import _plot_connectivity_circle
from mne.viz.utils import plt_show

from ..utils import fill_doc
from .helpers import (
    _add_comps_as_connections,
    _butterfly_on_button_press,
    _butterfly_onpick,
    _check_data_is_real,
    _check_info,
    _combine_connections,
    _get_con_info,
    _get_node_names_and_indices,
    _handle_data_and_indices,
    _handle_picks,
)


@fill_doc
def plot_spectral_connectivity(
    con,
    *,
    info=None,
    picks=None,
    selection="both",
    exclude="bads",
    combine=None,
    ci="sd",
    fmin=None,
    fmax=None,
    node_aliases=None,
    colors="auto",
    cmap="turbo",
    highlight=None,
    interactive=True,
    show=True,
):
    """Plot spectral connectivity as line plots, with circle plot overviews.

    Parameters
    ----------
    con : ~mne_connectivity.SpectralConnectivity
        The spectral connectivity object to plot.
    %(viz_info)s
    %(viz_picks)s
    %(viz_selection_line)s
    %(viz_exclude)s
    %(viz_combine_line_spectral)s
    %(viz_ci)s
    %(viz_fmin_fmax)s
    %(viz_node_aliases)s
    %(viz_colors_line)s
    %(viz_cmap_line)s
    %(viz_highlight)s
    %(viz_interactive)s
    %(viz_show)s

    Returns
    -------
    %(viz_figures)s

    Notes
    -----
    %(viz_circle_line_note)s
    %(viz_components_note)s
    """
    from mne_connectivity import SpectralConnectivity

    _validate_type(con, SpectralConnectivity, "con", "SpectralConnectivity")

    return _plot_line_connectivity(
        con=con,
        info=info,
        picks=picks,
        selection=selection,
        exclude=exclude,
        combine=combine,
        ci=ci,
        xlim=(fmin, fmax),
        node_aliases=node_aliases,
        colors=colors,
        cmap=cmap,
        highlight=highlight,
        interactive=interactive,
        show=show,
        xvar=con.freqs,
        xlabel="Frequency (Hz)",
    )


@fill_doc
def plot_temporal_connectivity(
    con,
    *,
    info=None,
    picks=None,
    selection="both",
    exclude="bads",
    combine=None,
    ci="sd",
    tmin=None,
    tmax=None,
    node_aliases=None,
    colors="auto",
    cmap="turbo",
    highlight=None,
    interactive=True,
    show=True,
):
    """Plot temporal connectivity as line plots, with circle plot overviews.

    Parameters
    ----------
    con : ~mne_connectivity.TemporalConnectivity
        The temporal connectivity object to plot.
    %(viz_info)s
    %(viz_picks)s
    %(viz_selection_line)s
    %(viz_exclude)s
    %(viz_combine_line_temporal)s
    %(viz_ci)s
    %(viz_tmin_tmax)s
    %(viz_node_aliases)s
    %(viz_colors_line)s
    %(viz_cmap_line)s
    %(viz_highlight)s
    %(viz_interactive)s
    %(viz_show)s

    Returns
    -------
    %(viz_figures)s

    Notes
    -----
    %(viz_circle_line_note)s
    %(viz_components_note)s
    """
    from mne_connectivity import TemporalConnectivity

    _validate_type(con, TemporalConnectivity, "con", "TemporalConnectivity")

    return _plot_line_connectivity(
        con=con,
        info=info,
        picks=picks,
        selection=selection,
        exclude=exclude,
        combine=combine,
        ci=ci,
        xlim=(tmin, tmax),
        node_aliases=node_aliases,
        colors=colors,
        cmap=cmap,
        highlight=highlight,
        interactive=interactive,
        show=show,
        xvar=con.times,
        xlabel="Time (s)",
    )


def _plot_line_connectivity(
    con,
    info,
    picks,
    selection,
    exclude,
    combine,
    ci,
    xlim,
    node_aliases,
    colors,
    cmap,
    highlight,
    interactive,
    show,
    xvar,
    xlabel,
):
    """Plot connectivity as line plots with circle plot overviews.

    Connectivity has dims [connections, frequencies | times].
    """
    _check_data_is_real(con.get_data("raveled"))

    _check_option("con.shape", len(con.shape), [2, 3], " length")

    _check_option("selection", selection, ["both", "seeds", "targets"])

    _validate_type(info, (mne.Info, None), "`info`", "mne.Info or None")

    _validate_type(combine, (str, Callable, None), "`combine`")
    if isinstance(combine, str):
        _check_option("combine", combine, ["mean"], " as a string")

    _validate_type(ci, (str, int, float, None), "`ci`")
    if isinstance(ci, str):
        _check_option("ci", ci, ["sd", "range"], " as a string")
    elif isinstance(ci, int | float):
        if not 0 < ci <= 100:
            raise ValueError("If `ci` is a float, it must be > 0 and <= 100.")

    _validate_type(node_aliases, (dict, None), "`node_aliases`", "dict or None")

    _check_option("colors", colors, ["auto", "global", "relative"])

    _validate_type(highlight, ("array-like", None), "`highlight`", "array-like or None")
    if highlight is not None:
        _check_option("highlight", np.ndim(highlight), [1, 2], " number of dimensions")
        if np.shape(highlight)[-1] != 2:
            raise ValueError("`highlight` must have shape (2,) or (n, 2).")
        highlight = np.atleast_2d(highlight)  # so a single period can be iterated over

    _validate_type(interactive, bool, "`interactive`", "bool")
    _validate_type(show, bool, "`show`", "bool")

    ch_names = con.names
    con_method = con.method if con.method is not None else "connectivity"
    ch_info = _check_info(info, ch_names)
    data, indices, is_multivar, is_symmetric, _ = _handle_data_and_indices(con, ch_info)

    # Get info about nodes and connections
    node_names, node_indices = _get_node_names_and_indices(
        ch_names, node_aliases, indices, is_multivar
    )
    con_info = _get_con_info(ch_info, node_names, indices, node_indices, is_multivar)

    # Get requested connections
    picks = _handle_picks(picks, exclude, ch_info, indices, is_multivar, selection)
    data = data[picks]
    indices = (indices[0][picks], indices[1][picks])
    node_indices = (node_indices[0][picks], node_indices[1][picks])
    con_info = pick_info(con_info, picks)
    con_info["temp"]["con_types"] = con_info["temp"]["con_types"][picks]

    # Add multivariate components as additional connections
    n_comps = 1
    if data.ndim == 3:
        data, con_info, node_indices, n_comps = _add_comps_as_connections(
            data, con_info, node_indices, comps_axis=1
        )

    # Mask data to relevant x values
    xvar = np.asarray(xvar)
    xvar_mask = np.nonzero(
        _time_mask(
            times=xvar, tmin=xlim[0], tmax=xlim[1], sfreq=None, include_tmax=True
        )
    )[0]
    data = data[..., xvar_mask]
    xvar = xvar[xvar_mask]

    con_types = con_info["temp"]["con_types"]
    figs = []
    for con_type in np.unique(con_types):
        # Prepare connectivity info for plotting
        type_mask = con_types == con_type
        type_data = data[type_mask]
        type_con_names = np.array(con_info["ch_names"])[type_mask]
        type_node_names = node_names.copy()
        type_node_indices = tuple(idcs[type_mask] for idcs in node_indices)

        # Combine connectivity across connections
        type_ci = None
        if combine is not None:
            (
                type_data,
                type_ci,
                type_con_names,
                type_node_names,
                type_node_indices,
            ) = _combine_connections(type_data, combine, ci, n_comps)

        # Create figure and axes
        fig = plt.figure(figsize=(15, 5), facecolor="w", layout="constrained")
        plot_circle = True
        line_subplot_idx = (1, 2)
        if len(type_node_indices[0]) == 1:
            plot_circle = False  # don't plot circle for a single connection
            line_subplot_idx = (1, 3)
        line_ax = fig.add_subplot(1, 3, line_subplot_idx)
        circle_ax = None
        duplicate_cons = False  # whether to duplicate connections for circle plot
        diag_mask = None  # used alongside connection duplication
        if plot_circle:
            # Prepare circle plot values
            circle_names, circle_indices = _get_circle_names_and_indices(
                type_node_names, type_node_indices
            )
            n_circle_nodes = len(circle_names)
            node_is_selectable = _get_node_selectability(circle_indices, selection)
            # If:
            # - plot is interactive
            # - connectivity data can be represented as a full, symmetric matrix
            # then visualisation works best if connections are duplicated such that all
            # nodes are seeds and targets
            if is_symmetric and interactive:
                duplicate_cons = True
                # Don't duplicate diagonal entries
                diag_mask = circle_indices[0] == circle_indices[1]
                circle_indices = (
                    np.concatenate([circle_indices[0], circle_indices[1][~diag_mask]]),
                    np.concatenate([circle_indices[1], circle_indices[0][~diag_mask]]),
                )
            if colors == "auto":
                # If connections span full matrix (diagonal optional) and plot is
                # interactive, use 'relative' colouring, such that the connection
                # colours span the colourmap space for each node. This makes interactive
                # visualisation for large numbers of nodes much better.
                # Otherwise, have the connection colours span the colourmap space for
                # all connections, which is better for non-interactive plots and
                # non-full connectivity data.
                type_connection_colors = (
                    "relative" if is_symmetric and interactive else "global"
                )
            else:
                type_connection_colors = colors
            circle_con, circle_con_order = _get_circle_con(
                circle_indices, n_circle_nodes, type_connection_colors, selection
            )
            # Avoid a zero colour range (e.g. for a single pair of nodes), which would
            # make MNE's circle plot divide by zero
            circle_vmin, circle_vmax = circle_con.min(), circle_con.max()
            if circle_vmin == circle_vmax:
                circle_vmax = circle_vmin + 1

            circle_ax = fig.add_subplot(1, 3, 3, polar=True)

            # Plot connectivity as circle
            fig, circle_ax = _plot_connectivity_circle(
                con=circle_con,
                node_names=circle_names,
                indices=circle_indices,
                node_width=None,
                node_height=1.0,
                node_colors=["black"],  # expects list
                node_edgecolor="white",
                node_linewidth=2.0,
                facecolor="white",
                textcolor="black",
                colormap=cmap,
                vmin=circle_vmin,
                vmax=circle_vmax,
                colorbar=False,
                linewidth=1.5,
                fontsize_names=8,
                padding=6.0,
                ax=circle_ax,
                interactive=False,  # use our modified callback
                title=(
                    "Node selection\n"
                    f"({selection.replace('both', 'seeds and targets')})"
                    if interactive
                    else "Nodes"
                ),
                show=show,
            )
            _set_node_alpha(circle_ax, node_is_selectable)
            con_colors = _get_con_colors(circle_ax, circle_con_order)
        else:
            con_colors = "k"

        # Plot connectivity as lines
        fig, line_ax = _plot_connectivity_lines(
            data=type_data,
            ci=type_ci,
            con_colors=con_colors,
            con_names=type_con_names,
            duplicate_cons=duplicate_cons,
            diag_mask=diag_mask,
            fig=fig,
            ax=line_ax,
            xvar=xvar,
            xlabel=xlabel,
            title=f"{con_type} | {con_method}",
            interactive=interactive,
            line_alpha=0.75,
            ci_alpha=0.3,
            linewidth=2.0,
            highlight=highlight,
        )

        # Add connectivity selection callback
        if plot_circle and interactive:
            callback = partial(
                _plot_connectivity_circle_onpick,
                fig=fig,
                circle_ax=circle_ax,
                line_ax=line_ax,
                indices=circle_indices,
                node_angles=np.linspace(0, 2 * np.pi, n_circle_nodes, endpoint=False),
                n_cons=len(type_data),
                duplicate_cons=duplicate_cons,
                circle_con_order=circle_con_order,
                selection=selection,
                node_selectability=node_is_selectable,
                has_ci=type_ci is not None,
            )
            fig.canvas.mpl_connect("button_press_event", callback)

        # Hide duplicate symmetric connections initially
        if duplicate_cons:
            _hide_duplicate_cons(
                fig=fig,
                circle_ax=circle_ax,
                line_ax=line_ax,
                n_cons=len(type_data),
                circle_con_order=circle_con_order,
                has_ci=type_ci is not None,
            )

        figs.append(fig)

    plt_show(show)

    if len(figs) == 1:
        return figs[0]
    return figs


def _get_circle_names_and_indices(node_names, node_indices):
    """Get names of nodes and indices of connections between them for circle plot."""
    unique_nodes = np.unique(np.r_[node_indices[0], node_indices[1]])
    circle_names = [node_names[idx] for idx in unique_nodes]

    circle_indices = [np.searchsorted(unique_nodes, ind) for ind in node_indices]

    return circle_names, circle_indices


def _get_node_selectability(circle_indices, selection):
    """Get selectability of nodes in circle plot based on node selection type."""
    n_unique_nodes = len(np.unique(np.r_[circle_indices[0], circle_indices[1]]))
    if selection == "both":
        node_selectability = [True] * n_unique_nodes
    else:
        if selection == "seeds":
            relevant_indices = circle_indices[0]
        else:  # selection == "targets"
            relevant_indices = circle_indices[1]
        node_selectability = [idx in relevant_indices for idx in range(n_unique_nodes)]

    return node_selectability


def _get_circle_con(circle_indices, n_nodes, connection_colors, selection):
    """Get connectivity values for circle plot (determines colour)."""
    if connection_colors == "relative":  # values span colourbar per node
        node_angles = np.linspace(0, 2 * np.pi, n_nodes, endpoint=False)
        circle_con = np.zeros(len(circle_indices[0]))
        for con_idx, (seed, target) in enumerate(zip(*circle_indices)):
            node_diff = node_angles[seed] - node_angles[target]
            if node_diff > 0:
                node_diff -= 2 * np.pi
            circle_con[con_idx] = np.abs(node_diff)
        # Normalise values for different number of connections per node
        if selection != "both":
            consider_indices = (
                circle_indices[0] if selection == "seeds" else circle_indices[1]
            )
            for node_idx in range(n_nodes):
                node_mask = consider_indices == node_idx
                if np.any(node_mask):
                    circle_con[node_mask] -= circle_con[node_mask].min()
                    if circle_con[node_mask].size > 1:  # avoid division by zero
                        circle_con[node_mask] /= circle_con[node_mask].max()
    else:  # values span colourbar over all connections
        circle_con = np.arange(len(circle_indices[0]))

    # mne.viz.circle._plot_connectivity_circle default behaviour is to sort connections
    # by strength (valid as of MNE v1.11)
    circle_con_order = np.argsort(circle_con)  # to map cons in circle plot to indices

    return circle_con, circle_con_order


def _set_node_alpha(circle_ax, node_is_selectable):
    """Set alpha of nodes in circle plot based on selectability."""
    for node_idx, node_selectable in enumerate(node_is_selectable):
        node_patch = circle_ax.containers[0][node_idx]
        if not node_selectable:
            node_patch.set_alpha(0.3)


def _get_con_colors(circle_ax, circle_con_order):
    """Get colors of connections from circle plot."""
    con_colors = [None] * len(circle_con_order)
    for patch_idx, con_idx in enumerate(circle_con_order):
        patch = circle_ax.patches[patch_idx]
        con_colors[con_idx] = patch.get_edgecolor()

    return con_colors


def _plot_connectivity_circle_onpick(
    event,
    fig,
    circle_ax,
    line_ax,
    indices,
    node_angles,
    n_cons,
    duplicate_cons,
    circle_con_order,
    selection,
    node_selectability,
    has_ci,
    ylim=(9, 10),
):
    """Isolate connections for a single node and reflect this in the line plot.

    On left click, shows only connections related to the clicked node.
    On right click, resets all connections.

    `y_lim` radius is default in circle plot (valid in MNE v1.11).
    """
    if event.inaxes != circle_ax:
        return

    patches = circle_ax.patches
    lines = line_ax.lines
    collections = line_ax.collections
    if event.button == 1:  # left click
        if not ylim[0] <= event.ydata <= ylim[1]:
            return  # ignore click if not near nodes

        # all angles in range [0, 2*pi]
        node_angles = node_angles % (np.pi * 2)
        node = np.argmin(np.abs(event.xdata - node_angles))
        if not node_selectability[node]:
            return  # ignore click if node not selectable

        for text in line_ax.texts:
            text.set_alpha(0)  # hide any connection labels

        for circle_idx, line_idx in enumerate(circle_con_order):
            seed, target = indices[0][line_idx], indices[1][line_idx]
            if selection == "both":
                # For symmetric data with duplicate connections, selecting a node for
                # both seed and target cons would show duplicate values. Instead, only
                # show the seed connections for symmetric data (which implicitly
                # includes the target connections).
                viable_nodes = [seed, target] if not duplicate_cons else [seed]
            elif selection == "seeds":
                viable_nodes = [seed]
            else:  # selection == "targets"
                viable_nodes = [target]
            visible = node in viable_nodes
            patches[circle_idx].set_visible(visible)
            lines[line_idx].set_visible(visible)
            lines[line_idx].set_picker(0 if not visible else True)
            if has_ci:
                collections[line_idx].set_visible(visible)
        fig.canvas.draw()

    elif event.button == 3:  # right click
        # Make original connections visible and hide duplicate connections
        _hide_duplicate_cons(fig, circle_ax, line_ax, n_cons, circle_con_order, has_ci)
        for text in line_ax.texts:
            text.set_alpha(0)  # hide any connection labels


def _hide_duplicate_cons(fig, circle_ax, line_ax, n_cons, circle_con_order, has_ci):
    """Hide duplicated connections in circle and line plots."""
    for circle_idx, line_idx in enumerate(circle_con_order):
        visible = line_idx < n_cons
        circle_ax.patches[circle_idx].set_visible(visible)
        line_ax.lines[line_idx].set_visible(visible)
        line_ax.lines[line_idx].set_picker(0 if not visible else True)
        if has_ci:
            line_ax.collections[line_idx].set_visible(visible)
    fig.canvas.draw()


def _plot_connectivity_lines(
    data,
    ci,
    con_colors,
    con_names,
    duplicate_cons,
    diag_mask,
    fig,
    ax,
    xvar,
    xlabel,
    title,
    interactive,
    line_alpha,
    ci_alpha,
    linewidth,
    highlight,
):
    """Plot data as butterfly plot."""
    texts = list()
    n_cons = data.shape[0]
    idxs = np.arange(n_cons)
    if duplicate_cons:
        n_duplicated_cons = (
            n_cons - np.sum(diag_mask) if diag_mask is not None else n_cons
        )
        idxs = np.concatenate([idxs, np.arange(n_duplicated_cons) + n_cons])
    lines = list()

    if interactive:
        # Parameters for butterfly interactive plots
        if duplicate_cons:
            reverse_con_names = [
                " ~ ".join(name.split(" ~ ")[::-1]) for name in con_names[~diag_mask]
            ]
            con_names = np.concatenate([con_names, reverse_con_names])
        params = dict(
            axes=[ax],
            texts=texts,
            lines=[lines],
            ch_names=con_names,
            idxs=[idxs],
            need_draw=False,
            path_effects=None,
        )
        fig.canvas.mpl_connect("pick_event", partial(_butterfly_onpick, params=params))
        fig.canvas.mpl_connect(
            "button_press_event", partial(_butterfly_on_button_press, params=params)
        )

    # Map cons with least activity behind the more active ones
    z_ord = data.std(axis=1).argsort()[::-1]

    # Plot connections
    def _plot_connections(mask=None, base_color_idx=0):
        if mask is None:
            mask = np.ones(z_ord.shape[0], dtype=bool)
        unmasked_idx = 0
        for con_idx, z in enumerate(z_ord):
            if not mask[con_idx]:
                continue
            if ci is not None:
                ax.fill_between(
                    xvar,
                    ci[con_idx, :, 0],
                    ci[con_idx, :, 1],
                    zorder=z + 1,
                    color=con_colors[unmasked_idx + base_color_idx],
                    edgecolor=None,
                    alpha=ci_alpha,
                )
            lines.append(
                ax.plot(
                    xvar,
                    data[con_idx],
                    picker=interactive,
                    zorder=z + 1,
                    color=con_colors[unmasked_idx + base_color_idx],
                    alpha=line_alpha,
                    linewidth=linewidth,
                )[0]
            )
            lines[-1].set_pickradius(3.0)
            unmasked_idx += 1

    _plot_connections()
    if duplicate_cons:
        _plot_connections(mask=~diag_mask, base_color_idx=n_cons)

    ax.set_xlim(xvar[0], xvar[-1])

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Connectivity (A.U.)")
    texts.append(
        ax.text(
            0,
            0,
            "",
            zorder=3,
            verticalalignment="baseline",
            horizontalalignment="left",
            fontweight="bold",
            alpha=0,
            clip_on=True,
        )
    )

    ax.set_title(title)

    # Plot highlights
    if highlight is not None:
        this_ylim = ax.get_ylim()
        for this_highlight in highlight:
            ax.fill_betweenx(
                this_ylim,
                this_highlight[0],
                this_highlight[1],
                facecolor="orange",
                alpha=0.15,
                zorder=99,
            )
        # Put back the y limits as fill_betweenx messes them up
        ax.set_ylim(this_ylim)

    return fig, ax
