from weakref import WeakKeyDictionary

import mne
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator
from mne._fiff.pick import pick_info
from mne.utils.check import _check_option, _validate_type
from mne.viz.utils import _plot_masked_image, plt_show

from ..utils import fill_doc
from .helpers import (
    _add_comps_as_connections,
    _check_data_is_real,
    _check_info,
    _get_con_info,
    _get_node_names_and_indices,
    _handle_data_and_indices,
    _handle_picks,
    _setup_cmap,
    _setup_vmin_vmax,
)


@fill_doc
def plot_connectivity(
    con,
    *,
    info=None,
    picks=None,
    selection="both",
    exclude="bads",
    node_aliases=None,
    vmin=None,
    vmax=None,
    cnorm=None,
    cmap=None,
    colorbar=True,
    node_labels="ticks",
    mask=None,
    mask_style=None,
    mask_cmap="Greys",
    mask_alpha=0.1,
    show=True,
):
    """Plot connectivity as a matrix.

    Parameters
    ----------
    con : ~mne_connectivity.Connectivity
        The connectivity object to plot.
    %(viz_info)s
    %(viz_picks)s
    %(viz_selection)s
    %(viz_exclude)s
    %(viz_node_aliases)s
    %(viz_vmin_vmax)s
    %(viz_cnorm)s
    %(viz_cmap)s
    %(viz_cbar)s
    %(viz_node_labels_matrix)s
    %(viz_mask)s
    %(viz_mask_style)s
    %(viz_mask_cmap)s
    %(viz_mask_alpha)s
    %(viz_show)s

    Returns
    -------
    %(viz_figures)s

    Notes
    -----
    %(viz_components_note)s
    """
    from mne_connectivity import Connectivity

    _validate_type(con, Connectivity, "con", "Connectivity")

    _check_data_is_real(con.get_data("raveled"))

    _check_option("con.shape", len(con.shape), [1, 2], " length")

    _check_option("selection", selection, ["both", "seeds", "targets"])

    _validate_type(info, (mne.Info, None), "`info`", "mne.Info or None")

    _check_option("node_labels", node_labels, ["names", "ticks", None])

    ch_names = con.names
    con_method = con.method if con.method is not None else "connectivity"
    ch_info = _check_info(info, ch_names)
    data, indices, is_multivar, is_symmetric, has_diagonal = _handle_data_and_indices(
        con, ch_info
    )

    # Handle instances of multiple components in multivariate data
    if data.ndim == 1:
        n_comps = components = None
        data = data[:, np.newaxis]  # give a dummy components dimension
    else:
        n_comps = data.shape[1]
        components = con.coords["components"].data

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

    con_types = con_info["temp"]["con_types"]
    figs = []
    for con_type in np.unique(con_types):
        # Prepare connectivity info for plotting
        type_mask = con_types == con_type
        type_node_indices = tuple(idcs[type_mask] for idcs in node_indices)
        type_node_indices_unique = np.unique(type_node_indices)
        type_node_names = [node_names[idx] for idx in type_node_indices_unique]
        type_n_nodes = type_node_indices_unique.size
        type_node_pos = {
            node_idx: pos for pos, node_idx in enumerate(type_node_indices_unique)
        }
        if data.ndim == 1:
            data = data[:, np.newaxis]  # give non-multivar data a dummy comps dim

        # Plot data for each component separately
        for comp_idx in range(data.shape[1]):
            # Make data square for plotting
            square_matrix = np.full((type_n_nodes, type_n_nodes), np.nan)
            for con_idx, (seed_idx, target_idx) in enumerate(zip(*type_node_indices)):
                square_matrix[type_node_pos[seed_idx], type_node_pos[target_idx]] = (
                    data[con_idx, comp_idx]
                )
            if is_symmetric:
                stacked_matrix = np.dstack((square_matrix, square_matrix.T))
                square_matrix = np.nansum(stacked_matrix, axis=-1)
                # namsum sets nan + nan = 0, so we set those cases back to nan
                square_matrix[np.isnan(stacked_matrix).all(axis=-1)] = np.nan
                if has_diagonal:
                    square_matrix[np.diag_indices_from(square_matrix)] *= 0.5

            # Colormap handling
            vmin, vmax = _setup_vmin_vmax(data=square_matrix, vmin=vmin, vmax=vmax)
            cmap = _setup_cmap(cmap=cmap, vmin=vmin, vmax=vmax)
            if cnorm is None:
                cnorm = Normalize(vmin=vmin, vmax=vmax)

            # Create figure and axis
            fig, ax = plt.subplots(
                1, 1, figsize=(6, 6), facecolor="w", layout="constrained"
            )

            min_node_idx = min(type_node_indices[0].min(), type_node_indices[1].min())
            max_node_idx = max(type_node_indices[0].max(), type_node_indices[1].max())
            img, _ = _plot_masked_image(
                ax=ax,
                data=square_matrix,
                times=np.arange(min_node_idx, max_node_idx + 1),
                yvals=np.arange(min_node_idx, max_node_idx + 1),
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                mask=mask,
                mask_style=mask_style,
                mask_alpha=mask_alpha,
                mask_cmap=mask_cmap,
                yscale="linear",
                cnorm=cnorm,
            )
            ax.set_box_aspect(1)
            if colorbar:
                cbar = fig.colorbar(img, ax=ax, shrink=0.6, label="Connectivity (A.U.)")
                cbar.ax.set_zorder(ax.get_zorder() - 1)

            title = f"{con_type} | {con_method}"
            if n_comps is not None:
                title += f" | Component {components[comp_idx]}"
            ax.set_title(title)
            ax.set_xlabel("Targets")
            ax.set_ylabel("Seeds")
            if node_labels == "names":
                ax.set_xticks(np.arange(type_n_nodes))
                ax.set_yticks(np.arange(type_n_nodes))
                ax.set_xticklabels(type_node_names, rotation=45)
                ax.set_yticklabels(type_node_names)
            if node_labels is None:
                ax.set_xticks([])
                ax.set_yticks([])
            else:  # node_labels == "ticks"
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            ax.set_xlim(min_node_idx - 0.5, max_node_idx + 0.5)
            ax.set_ylim(max_node_idx + 0.5, min_node_idx - 0.5)

            def callback(event, ax=ax, fig=fig, node_names=type_node_names):
                _plot_connectivity_matrix_onclick(event, ax, fig, node_names)

            fig.canvas.mpl_connect("button_press_event", callback)

            figs.append(fig)

    plt_show(show)

    if len(figs) == 1:
        return figs[0]
    return figs


_MATRIX_ANNOTATIONS = WeakKeyDictionary()


def _plot_connectivity_matrix_onclick(event, ax, fig, node_names):
    """Annotate the clicked matrix cell with the corresponding channel names."""
    if event.inaxes is not ax or event.xdata is None or event.ydata is None:
        return

    if event.button == 3:  # right-click to remove annotation
        prev_annot = _MATRIX_ANNOTATIONS.get(ax)
        if prev_annot is not None:
            prev_annot[0].remove()
            prev_annot[1].remove()
            _MATRIX_ANNOTATIONS[ax] = None
            fig.canvas.draw_idle()
        return

    col = int(np.floor(event.xdata + 0.5))
    row = int(np.floor(event.ydata + 0.5))
    if row < 0 or row >= len(node_names) or col < 0 or col >= len(node_names):
        return

    prev_annot = _MATRIX_ANNOTATIONS.get(ax)
    if prev_annot is not None:
        prev_annot[0].remove()
        prev_annot[1].remove()
        _MATRIX_ANNOTATIONS[ax] = None
        fig.canvas.draw_idle()

    annotation = ax.text(
        col + 0.25,
        row - 0.25,
        f"{node_names[row]}\n~\n{node_names[col]}",
        ha="left",
        va="bottom",
        color="white",
        fontsize=8,
        fontweight="bold",
        bbox=dict(facecolor="black", alpha=0.6, edgecolor="none", boxstyle="round"),
    )
    annotation.set_in_layout(False)
    annotation.set_zorder(10)

    # Highlight the selected cell with a border
    rect = plt.Rectangle(
        (col - 0.5, row - 0.5),
        1,
        1,
        linewidth=2,
        edgecolor="k",
        facecolor="none",
    )
    ax.add_patch(rect)

    _MATRIX_ANNOTATIONS[ax] = (annotation, rect)
    fig.canvas.draw_idle()
