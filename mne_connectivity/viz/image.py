from collections.abc import Callable

import mne
import numpy as np
from matplotlib import pyplot as plt
from mne._fiff.pick import pick_info
from mne.utils.check import _check_option, _validate_type
from mne.utils.numerics import _time_mask
from mne.viz.utils import _plot_masked_image, plt_show

from ..utils import fill_doc
from .helpers import (
    _add_comps_as_connections,
    _check_data_is_real,
    _check_info,
    _combine_connections,
    _get_con_info,
    _get_node_names_and_indices,
    _handle_data_and_indices,
    _handle_picks,
    _setup_cmap,
    _setup_vmin_vmax,
)


@fill_doc
def plot_spectrotemporal_connectivity(
    con,
    *,
    info=None,
    picks=None,
    selection="both",
    exclude="bads",
    combine="mean",
    node_aliases=None,
    tmin=None,
    tmax=None,
    fmin=None,
    fmax=None,
    yscale="auto",
    vmin=None,
    vmax=None,
    cnorm=None,
    cmap=None,
    colorbar=True,
    mask=None,
    mask_style=None,
    mask_cmap="Greys",
    mask_alpha=0.1,
    show=True,
):
    """Plot spectro-temporal connectivity.

    Parameters
    ----------
    con : ~mne_connectivity.SpectroTemporalConnectivity
        The spectro-temporal connectivity object to plot.
    %(viz_info)s
    %(viz_picks)s
    %(viz_selection)s
    %(viz_exclude)s
    %(viz_combine_image_spectrotemporal)s
    %(viz_node_aliases)s
    %(viz_tmin_tmax)s
    %(viz_fmin_fmax)s
    %(viz_yscale_image)s
    %(viz_vmin_vmax)s
    %(viz_cnorm)s
    %(viz_cmap)s
    %(viz_cbar)s
    %(viz_mask)s
    %(viz_mask_style)s
    %(viz_mask_cmap)s
    %(viz_mask_alpha)s
    %(viz_show)s

    Returns
    -------
    %(viz_figure_image)s

    Notes
    -----
    %(viz_components_extra_con_note)s
    """
    from mne_connectivity import SpectroTemporalConnectivity

    _validate_type(
        con, SpectroTemporalConnectivity, "con", "SpectroTemporalConnectivity"
    )

    _check_data_is_real(con.get_data("raveled"))

    _check_option("con.shape", len(con.shape), [3, 4], " length")

    _check_option("selection", selection, ["both", "seeds", "targets"])

    _validate_type(info, (mne.Info, None), "`info`", "mne.Info or None")

    _validate_type(combine, (str, Callable, None), "`combine`")
    if isinstance(combine, str):
        _check_option("combine", combine, ["mean"], " as a string")

    _validate_type(node_aliases, (dict, None), "`node_aliases`", "dict or None")

    _check_option("yscale", yscale, ["linear", "log", "auto"])

    _validate_type(mask, (np.ndarray, None), "`mask`", "numpy.ndarray or None")

    _validate_type(colorbar, bool, "`colorbar`", "bool")

    _validate_type(show, bool, "`show`", "bool")

    ch_names = con.names
    con_method = con.method if con.method is not None else "connectivity"
    ch_info = _check_info(info, ch_names)
    data, indices, is_multivar, is_symmetric, duplicate_cons_mask = (
        _handle_data_and_indices(con, ch_info)
    )

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
    duplicate_cons_mask = duplicate_cons_mask[picks]
    con_info = pick_info(con_info, picks)
    con_info["temp"]["con_types"] = con_info["temp"]["con_types"][picks]

    # Add multivariate components as additional connections
    n_comps = 1
    if data.ndim == 4:
        data, con_info, node_indices, n_comps = _add_comps_as_connections(
            data, con_info, node_indices, comps_axis=1
        )
        duplicate_cons_mask = np.tile(duplicate_cons_mask, n_comps)

    if mask is not None and mask.shape != data.shape[1:]:
        raise ValueError(
            f"Mask shape {mask.shape} does not match data shape {data.shape[1:]}."
        )

    # Mask data to relevant x and y values
    xvar, yvar = np.asarray(con.times), np.asarray(con.freqs)
    xvar_mask = np.nonzero(
        _time_mask(times=xvar, tmin=tmin, tmax=tmax, sfreq=None, include_tmax=True)
    )[0]
    yvar_mask = np.nonzero(
        _time_mask(times=yvar, tmin=fmin, tmax=fmax, sfreq=None, include_tmax=True)
    )[0]
    data = data[..., yvar_mask, :][..., xvar_mask]
    if mask is not None:
        mask = mask[yvar_mask, :][:, xvar_mask]

    con_types = con_info["temp"]["con_types"]
    figs = []
    for con_type in np.unique(con_types):
        # Prepare connectivity info for plotting
        type_mask = con_types == con_type
        type_data = data[type_mask]
        type_con_names = np.array(con_info["ch_names"])[type_mask]
        type_node_indices = tuple(idcs[type_mask] for idcs in node_indices)
        type_duplicate_cons_mask = duplicate_cons_mask[type_mask]

        if all(type_duplicate_cons_mask):
            continue  # skip if all connections for this type are duplicates

        # Combine connectivity across connections
        if combine is not None:
            (
                type_data,
                _,
                type_con_names,
                _,
                _,
            ) = _combine_connections(
                data=type_data[~type_duplicate_cons_mask],
                combine=combine,
                ci=None,
                n_comps=n_comps,
            )

        # Colormap handling
        vmin, vmax = _setup_vmin_vmax(data=type_data, vmin=vmin, vmax=vmax)
        cmap = _setup_cmap(cmap=cmap, vmin=vmin, vmax=vmax)

        # Remove duplicate connections in symmetric data
        if is_symmetric and con.indices in ("lower", "upper") and combine is None:
            type_data = type_data[~type_duplicate_cons_mask]
            type_con_names = type_con_names[~type_duplicate_cons_mask]
            type_node_indices = tuple(
                idcs[~type_duplicate_cons_mask] for idcs in type_node_indices
            )

        # Plot connectivity as image
        type_figs = [
            plt.figure(layout="constrained") for _ in range(type_data.shape[0])
        ]
        type_axes = [fig.add_subplot() for fig in type_figs]
        for con_idx in range(type_data.shape[0]):
            con_ax = type_axes[con_idx]
            img, _ = _plot_masked_image(
                ax=con_ax,
                data=type_data[con_idx],
                times=xvar[xvar_mask],
                mask=mask,
                yvals=yvar[yvar_mask],
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                mask_style=mask_style,
                mask_alpha=mask_alpha,
                mask_cmap=mask_cmap,
                yscale=yscale,
                cnorm=cnorm,
            )
            con_ax.set_xlabel("Time (s)")
            con_ax.set_ylabel("Frequency (Hz)")
            if colorbar:
                con_ax.get_figure().colorbar(
                    mappable=img, ax=type_axes[con_idx], label="Connectivity (A.U.)"
                )
            con_ax.set_title(f"{con_type} | {type_con_names[con_idx]} | {con_method}")

        figs.extend(type_figs)

    plt_show(show)

    if len(figs) == 1:
        return figs[0]
    return figs
