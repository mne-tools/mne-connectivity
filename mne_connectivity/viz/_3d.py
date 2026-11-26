"""Functions to make 3D plots with M/EEG data."""

# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Denis Engemann <denis.engemann@gmail.com>
#          Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#          Eric Larson <larson.eric.d@gmail.com>
#          Mainak Jas <mainak@neuro.hut.fi>
#          Mark Wronkiewicz <wronk.mark@gmail.com>
#
# License: Simplified BSD

import inspect

import mne
import numpy as np
from mne.io.constants import FIFF
from mne.utils import _validate_type, fill_doc, verbose

from .helpers import (
    _check_data_is_real,
    _check_info,
    _handle_data_and_indices,
    _handle_picks,
)

verbose_dec = verbose
FIDUCIAL_ORDER = (FIFF.FIFFV_POINT_LPA, FIFF.FIFFV_POINT_NASION, FIFF.FIFFV_POINT_RPA)


@fill_doc
def plot_sensors_connectivity(
    info,
    con,
    picks=None,
    *,
    cbar_label="Connectivity",
    n_con=20,
    cmap="RdBu",
    min_distance=0.05,
):
    """Visualize the sensor connectivity in 3D.

    Parameters
    ----------
    info : mne.Info
        The measurement info, containing the locations of all channels in
        ``con.names``. Connections involving channels marked as bad in ``info`` are
        not shown.
    con : ~mne_connectivity.Connectivity
        The connectivity data to plot.

        .. versionchanged:: 1.0
           Arrays are no longer supported.
    picks : str | array_like | slice | None
        Channels to include in the plot. Connections involving any of these channels
        are shown. Slices and lists of integers are interpreted as indices of the
        channels in ``con.names``. In lists, channel type strings (e.g.,
        ``['meg', 'eeg']``) pick channels of those types, and channel name strings
        (e.g., ``['MEG0111', 'MEG2623']``) pick the given channels. Can also be the
        string values ``'all'`` to pick all channels, or ``'data'`` to pick data
        channels. None (default) picks all good channels.

        .. versionchanged:: 1.0
           Selects the channels whose connections are shown, as in
           :func:`~mne_connectivity.viz.plot_connectivity`, instead of the channels
           in ``info`` that correspond to the rows of an array.
    cbar_label : str
        Label for the colorbar.
    n_con : int
        Number of strongest connections shown (default 20). Strength is the absolute
        value of the connectivity. Self-connections are ignored if they are the same
        for all channels (e.g., coherence), and symmetric data is counted once per
        pair of channels. Connections between sensors closer than ``min_distance`` are
        then removed, so fewer than ``n_con`` connections may be shown.
    cmap : str | instance of matplotlib.colors.Colormap
        Colormap for coloring connections by strength. If a str, must be a valid
        Matplotlib colormap (i.e. a valid key of `matplotlib.colormaps`). Default is
        ``"RdBu"``.
    min_distance : float
        The minimum distance required between two sensors to plot a connection between
        them, in meters. Default is 0.05 (i.e. 5 cm).

        .. versionadded:: 0.8

    Returns
    -------
    fig : instance of mne.viz.Figure3D
        The 3D figure.
    """
    from mne.viz.backends.renderer import _get_renderer

    from mne_connectivity import Connectivity

    _validate_type(info, "info")
    _validate_type(con, Connectivity, "`con`", "Connectivity")
    _check_data_is_real(con.get_data("raveled"))
    _check_info(info, con.names)
    # keep only the channels in con, in the same order, for the sensor locations
    info = mne.pick_info(
        info, mne.pick_channels(info["ch_names"], con.names, ordered=True)
    )

    if min_distance <= 0:
        raise ValueError(
            "The minimum distance between sensors must be greater than 0 m, got "
            f"{min_distance} m"
        )

    # Get the strongest n_con connections
    ii, jj, con_val = _get_strongest_connections(con, info, n_con, picks=picks)

    # Remove close connections
    sens_loc = np.array([ch["loc"][:3] for ch in info["chs"]])
    far = np.linalg.norm(sens_loc[ii] - sens_loc[jj], axis=1) > min_distance
    con_nodes = list(zip(ii[far], jj[far]))
    con_val = con_val[far]

    if con_val.size == 0:
        raise ValueError(
            f"None of the {n_con} strongest connections were at least {min_distance} m "
            "apart. Try decreasing `min_distance` or increasing `n_con`, and check "
            "that the coordinates of your channels in `info` are not NaNs"
        )

    # Plot the sensor locations, only once all errors have been checked
    renderer = _get_renderer(size=(600, 600), bgcolor=(0.5, 0.5, 0.5))
    renderer.sphere(
        np.c_[sens_loc[:, 0], sens_loc[:, 1], sens_loc[:, 2]],
        color=(1, 1, 1),
        opacity=1,
        scale=0.005,
    )

    # Show the connections as tubes between sensors
    vmax = np.max(con_val)
    vmin = np.min(con_val)
    for val, nodes in zip(con_val, con_nodes):
        x1, y1, z1 = sens_loc[nodes[0]]
        x2, y2, z2 = sens_loc[nodes[1]]
        tube = renderer.tube(
            origin=np.c_[x1, y1, z1],
            destination=np.c_[x2, y2, z2],
            scalars=np.c_[val, val],
            vmin=vmin,
            vmax=vmax,
            reverse_lut=True,
            colormap=cmap,
        )

    renderer.scalarbar(source=tube, title=cbar_label)

    # Add the sensor names for the connections shown
    nodes_shown = list(set([n[0] for n in con_nodes] + [n[1] for n in con_nodes]))

    kwargs = dict()
    if "font_size" in inspect.getfullargspec(renderer.text3d).args:  # mne >= 1.13.0
        kwargs["font_size"] = 12
    else:
        kwargs["scale"] = 0.005
    for node in nodes_shown:
        x, y, z = sens_loc[node]
        renderer.text3d(x, y, z, text=con.names[node], color=(0, 0, 0), **kwargs)

    renderer.set_camera(
        azimuth=-88.7,
        elevation=40.8,
        distance=0.76,
        focalpoint=np.array([-3.9e-4, -8.5e-3, -1e-2]),
    )
    renderer.show()
    return renderer.scene()


def _get_strongest_connections(con, info, n_con, picks=None):
    """Get the seeds, targets, and values of the strongest connections to plot.

    Strength is the absolute value. Trivial self-connections, the duplicate half of
    symmetric data, connections of bad channels, and connections not involving any
    picked channel are dropped beforehand. ``info`` must contain exactly the channels
    in ``con.names``, in the same order.
    """
    data, indices, is_multivar, _, duplicate_cons_mask = _handle_data_and_indices(
        con, info
    )
    if is_multivar:
        raise ValueError("Multivariate connectivity is not supported.")
    keep = ~duplicate_cons_mask
    picked = np.zeros(len(data), dtype=bool)
    picked[_handle_picks(picks, "bads", info, indices, is_multivar, "both")] = True
    keep &= picked
    data, seeds, targets = data[keep], indices[0][keep], indices[1][keep]
    order = np.argsort(-np.abs(data), kind="stable")[:n_con]
    return seeds[order], targets[order], data[order]
