import mne
import numpy as np
from mne._fiff.pick import _picks_to_idx
from mne.defaults import DEFAULTS
from mne.stats.permutations import bootstrap_confidence_interval
from mne.utils.check import _check_if_nan

from ..utils import (
    _check_if_multivariate_indices,
    _get_unique_multivariate_nodes_and_indices,
)


def _check_data_is_real(data):
    """Check that data is real-valued."""
    if np.iscomplexobj(data):
        raise ValueError(
            "Plotting for complex-valued connectivity data is not supported. Consider "
            "plotting the absolute values, or the real and imaginary parts separately."
        )


def _handle_data_and_indices(con, ch_info):
    """Extract data and indices from connectivity object."""
    indices = con.indices
    is_multivar = False  # note: multivar connectivity is not supported for str indices
    is_symmetric = False  # whether returned data is tril/triu symmetric
    duplicate_cons_mask = None  # mask for duplicate connections in symmetric data

    # Explicit indices provided
    if isinstance(indices, tuple):
        data = con.get_data("raveled")
        _check_if_nan(data)
        is_multivar = _check_if_multivariate_indices(indices)
        if not is_multivar:
            indices = (np.array(indices[0]), np.array(indices[1]))
        else:
            # Ragged multivariate indices can be stored as lists of arrays, but they
            # need to be arrays themselves so that connections can be picked
            indices = tuple(_ragged_to_array(idcs) for idcs in indices)
        duplicate_cons_mask = np.full(len(indices[0]), False, dtype=bool)

        return data, indices, is_multivar, is_symmetric, duplicate_cons_mask

    # Lower-tri, upper-tri, or all-to-all
    try:  # Try to get dense data with missing values filled in
        data = con.get_data("dense", missing="raise")
        missing_filled = True
    except ValueError:  # Fall back if missing can't be filled (for lower/upper)
        data = con.get_data("dense", missing=np.nan)
        missing_filled = False

    if not missing_filled:
        # Only take existing data if missing values can't be filled in
        if indices == "lower":
            indices = np.tril_indices(con.n_nodes, k=-1)
        else:  # "upper"; can't be "all", since no missing values to (fail to) fill in
            indices = np.triu_indices(con.n_nodes, k=1)
    else:
        # Check whether to ignore diagonal (if all values are the same)
        if con.n_nodes == 1:
            ignore_diag = False  # excluding diagonal would remove the only connection
        elif indices == "all":
            # Check if diagonal is all close (could be all zeros, ones, NaNs)
            diag = np.diagonal(data).ravel()
            ignore_diag = bool(np.allclose(diag[1:], diag[0], equal_nan=True))
        else:
            ignore_diag = True  # diagonal trivial for filled-in lower/upper matrices
        # Check whether the matrix is symmetric
        is_symmetric = np.allclose(
            data, data.transpose(1, 0, *range(2, data.ndim)), equal_nan=True
        )
        # Construct explicit indices
        indices = np.unravel_index(
            np.arange(con.n_nodes**2), (con.n_nodes, con.n_nodes)
        )
        if ignore_diag:
            diag_mask = indices[0] == indices[1]
            indices = (indices[0][~diag_mask], indices[1][~diag_mask])
        if is_symmetric:
            if con.indices in ["lower", "all"]:
                keep_indices = np.tril_indices(con.n_nodes, k=0)
            else:
                keep_indices = np.triu_indices(con.n_nodes, k=0)
            duplicate_cons_mask = np.array(
                [ind not in list(zip(*keep_indices)) for ind in list(zip(*indices))]
            )
    if duplicate_cons_mask is None:
        duplicate_cons_mask = np.full(len(indices[0]), False, dtype=bool)

    # Drop entries for bad channels from data and indices
    data = data[indices]
    bad_idcs = []
    if ch_info is not None:
        bad_idcs = [con.names.index(bad) for bad in ch_info["bads"]]
    if len(bad_idcs) > 0:
        good_con_mask = np.ones(data.shape[0], dtype=bool)
        for con_idx, (seed, target) in enumerate(zip(*indices)):
            if seed in bad_idcs or target in bad_idcs:
                good_con_mask[con_idx] = False
        data = data[good_con_mask]
        indices = (indices[0][good_con_mask], indices[1][good_con_mask])
        duplicate_cons_mask = duplicate_cons_mask[good_con_mask]

    _check_if_nan(data)

    return data, indices, is_multivar, is_symmetric, duplicate_cons_mask


def _ragged_to_array(indices):
    """Convert (ragged) multivariate indices to an object array of channel sets."""
    if isinstance(indices, np.ndarray):
        return indices
    array = np.empty(len(indices), dtype=object)
    for idx, node in enumerate(indices):
        array[idx] = node
    return array


def _check_info(info, ch_names):
    """Check (or create) info object and ensure all channels are present."""
    if info is None:
        info = mne.create_info(ch_names=ch_names, sfreq=1.0, ch_types="misc")

    # Make sure all channel names from con object found in info
    missing_channels = [name for name in ch_names if name not in info["ch_names"]]
    if len(missing_channels) != 0:
        raise ValueError(
            "Not all channel names from `con.names` found in `info`. Missing channels: "
            f"{missing_channels}"
        )

    return info


def _get_node_names_and_indices(ch_names, node_aliases, indices, is_multivar):
    """Get/create names of seeds/targets in connections and their indices."""
    if node_aliases is None:
        node_aliases = dict()
    # nodes are channel indices for bivariate and (ragged) channel sets for
    # multivariate connectivity, so compare them as scalars and tuples, respectively
    all_nodes = set()
    for node in (*indices[0], *indices[1]):
        if not is_multivar:
            all_nodes.add(int(node))
        elif isinstance(node, np.ma.MaskedArray):
            all_nodes.add(tuple(node.compressed()))
        else:
            all_nodes.add(tuple(node))
    if any(idx not in all_nodes for idx in node_aliases.keys()):
        raise ValueError("All keys in `node_aliases` must be present in `con.indices`.")

    # Get names of nodes (via aliases, directly, or create for multivar connections)
    if not is_multivar:
        unique_nodes = np.unique([*indices[0], *indices[1]]).tolist()
        node_indices = (indices[0].copy(), indices[1].copy())  # use original indices
        node_names = list(ch_names)  # copy, so that aliases don't rename `con.names`
        for node_ind in unique_nodes:
            if node_ind in node_aliases.keys():
                node_names[node_ind] = node_aliases[node_ind]
    else:
        unique_nodes, node_indices = _get_unique_multivariate_nodes_and_indices(indices)
        node_names = [f"node {node_idx}" for node_idx in range(len(unique_nodes))]
        for node_idx, node_ind in enumerate(unique_nodes):
            node_ind = tuple(node_ind)
            if node_ind in node_aliases.keys():
                node_names[node_idx] = node_aliases[node_ind]

    return node_names, node_indices


def _get_con_info(ch_info, node_names, indices, node_indices, is_multivar):
    """Create info object for connectivity data."""
    con_names = []
    for node_idx, name in enumerate(node_names):
        node_names[node_idx] = name.replace("~", "-")  # avoid confusion with con names
    for seed, target in zip(*node_indices):
        con_names.append(f"{node_names[seed]} ~ {node_names[target]}")

    ch_types = ch_info.get_channel_types()
    con_types = []
    for seed, target in zip(*indices):
        if not is_multivar:
            seed_type = DEFAULTS["titles"][ch_types[seed]]
            target_type = DEFAULTS["titles"][ch_types[target]]
            con_types.append(f"{seed_type} ~ {target_type}")
        else:
            if isinstance(seed, np.ma.MaskedArray):
                seed = seed.compressed()
            if isinstance(target, np.ma.MaskedArray):
                target = target.compressed()
            seed_types = np.unique(
                [DEFAULTS["titles"][ch_types[ch_idx]] for ch_idx in seed]
            )
            target_types = np.unique(
                [DEFAULTS["titles"][ch_types[ch_idx]] for ch_idx in target]
            )
            con_types.append(f"{', '.join(seed_types)} ~ {', '.join(target_types)}")

    con_info = mne.create_info(ch_names=con_names, sfreq=1.0, ch_types="misc")
    # Can't store connectivity types in ch_types as they are not recognised
    con_info["temp"] = dict()
    con_info["temp"]["con_types"] = np.array(con_types)

    return con_info


def _handle_picks(picks, exclude, ch_info, indices, is_multivar, selection):
    """Handle picks for connectivity data."""
    ch_picks = _picks_to_idx(info=ch_info, picks=picks, none="all", exclude=exclude)
    con_picks = []
    for con_idx, (seed, target) in enumerate(zip(*indices)):
        if not is_multivar:
            seed, target = [seed], [target]
        if selection == "both":
            con_nodes = np.concatenate([seed, target])
        elif selection == "seeds":
            con_nodes = seed
        else:  # selection == "targets"
            con_nodes = target
        if picks is not None:
            con_nodes = [node for node in con_nodes if node in ch_picks]
        if np.any([ch in ch_picks for ch in con_nodes]):
            con_picks.append(con_idx)

    return con_picks


def _add_comps_as_connections(data, con_info, node_indices, comps_axis):
    """Add multivariate components as additional connections."""
    n_comps = data.shape[comps_axis]
    new_shape = (data.shape[0] * n_comps,)
    if comps_axis + 1 < data.ndim:
        new_shape += data.shape[comps_axis + 1 :]
    data = np.reshape(data, new_shape)
    node_indices = (
        np.repeat(node_indices[0], n_comps),
        np.repeat(node_indices[1], n_comps),
    )

    new_con_names = []
    for con_name in con_info["ch_names"]:
        new_con_names.extend([f"{con_name} ({comp})" for comp in range(n_comps)])
    new_con_types = np.repeat(con_info["temp"]["con_types"], n_comps)

    with con_info._unlock():
        con_info["ch_names"] = new_con_names
    con_info["temp"]["con_types"] = new_con_types

    return data, con_info, node_indices, n_comps


def _combine_connections(data, combine, ci, n_comps=1):
    """Combine data over connections."""
    assert data.shape[0] % n_comps == 0, (
        "Data to combine does not have a matching number of connections for each "
        "component. Please contact the MNE-Connectivity developers."
    )
    n_cons = data.shape[0] // n_comps

    if combine == "mean":
        combine_func = lambda x: np.mean(x, axis=0)  # noqa: E731
    else:
        assert callable(combine), (
            "The `combine` parameter is not callable as expected. "
            "Please contact the MNE-Connectivity developers."
        )
        combine_func = combine

    if ci is None:
        ci_func = None
    elif ci == "sd":
        ci_func = lambda x: np.std(x, axis=0)  # noqa: E731
    elif ci == "range":
        ci_func = lambda x: (np.min(x, axis=0), np.max(x, axis=0))  # noqa: E731
    else:
        assert isinstance(ci, int | float), (
            "The `ci` parameter is not a float as expected. "
            "Please contact the MNE-Connectivity developers."
        )
        ci_func = lambda x: tuple(  # noqa: E731
            bootstrap_confidence_interval(arr=x, ci=ci / 100, stat_fun=combine_func)
        )

    data_combined = np.empty((n_comps, *data.shape[1:]), dtype=data.dtype)
    data_ci = None
    if ci_func is not None:
        data_ci = np.empty(data_combined.shape + (2,), dtype=data.dtype)
    for comp_idx in range(n_comps):
        data_combined[comp_idx] = combine_func(
            data[n_cons * comp_idx : n_cons * (comp_idx + 1)]
        )
        if ci_func is not None:
            ci_out = ci_func(data[n_cons * comp_idx : n_cons * (comp_idx + 1)])
            if isinstance(ci_out, tuple):
                assert len(ci_out) == 2, (
                    f"Expected `len(ci_out)` of 2, got {len(ci_out)}. "
                    "Please contact the MNE-Connectivity developers."
                )
                data_ci[comp_idx, ..., 0] = ci_out[0]
                data_ci[comp_idx, ..., 1] = ci_out[1]
            else:
                data_ci[comp_idx, ..., 0] = data_combined[comp_idx] - ci_out
                data_ci[comp_idx, ..., 1] = data_combined[comp_idx] + ci_out

    if n_comps == 1:
        con_names = np.array([f"combined nodes (n={n_cons})"])
        node_names = con_names.copy()
        node_indices = (np.array([0]), np.array([0]))
    else:
        con_names = np.array(
            [f"combined nodes ({comp_idx}; n={n_cons})" for comp_idx in range(n_comps)]
        )
        node_names = con_names.copy()
        node_indices = (np.arange(n_comps), np.arange(n_comps))

    return data_combined, data_ci, con_names, node_names, node_indices


def _setup_vmin_vmax(data, vmin, vmax):
    """Handle vmin and vmax parameters for visualizing connectivity.

    For the normal use-case (when `vmin` and `vmax` are None):
    - vlim is set to (-abs(max), abs(max)) when data has both pos and neg values.
    - vlim is set to (min, max) when data is exclusively pos or neg values.

    Otherwise, vmin and vmax are callables/pre-specified values that drive the
    operation.
    """
    data = data[~np.isnan(data)]

    if vmax is None and vmin is None:
        if ~np.all(data >= 0) and ~np.all(data <= 0):
            vmax = np.abs(data).max()
            vmin = -vmax
        else:
            vmin, vmax = data.min(), data.max()

    else:
        if callable(vmin):
            vmin = vmin(data)
        elif vmin is None:
            vmin = np.min(data)

        if callable(vmax):
            vmax = vmax(data)
        elif vmax is None:
            vmax = np.max(data)

    return vmin, vmax


def _setup_cmap(cmap, vmin, vmax):
    """Handle colormap for visualizing connectivity."""
    if isinstance(cmap, tuple):
        return cmap

    if cmap is None:
        if vmin >= 0 and vmax >= 0:
            return "Reds"
        if vmin < 0 and vmax < 0:
            return "Blues_r"
        return "RdBu_r"

    return cmap


def _butterfly_onpick(event, params):
    """Add a channel name on click."""
    params["need_draw"] = True
    ax = event.artist.axes
    ax_idx = np.where([ax is a for a in params["axes"]])[0]
    if len(ax_idx) == 0:  # this can happen if ax param is used
        return  # let the other axes handle it
    else:
        ax_idx = ax_idx[0]
    lidx = np.where([line is event.artist for line in params["lines"][ax_idx]])[0][0]
    ch_name = params["ch_names"][params["idxs"][ax_idx][lidx]]
    text = params["texts"][ax_idx]
    x = event.artist.get_xdata()[event.ind[0]]
    y = event.artist.get_ydata()[event.ind[0]]
    text.set_x(x)
    text.set_y(y)
    text.set_text(ch_name)
    text.set_color(event.artist.get_color())
    text.set_alpha(1.0)
    text.set_zorder(len(ax.lines))  # to make sure it goes on top of the lines
    text.set_path_effects(params["path_effects"])
    # do NOT redraw here, since for butterfly plots hundreds of lines could
    # potentially be picked -- use on_button_press (happens once per click)
    # to do the drawing


def _butterfly_on_button_press(event, params):
    """Only draw once for picking."""
    if params["need_draw"]:
        event.canvas.draw()
    else:
        idx = np.where([event.inaxes is ax for ax in params["axes"]])[0]
        if len(idx) == 1:
            text = params["texts"][idx[0]]
            text.set_alpha(0.0)
            text.set_path_effects([])
            event.canvas.draw()
    params["need_draw"] = False
