"""Functions to make 3D plots with M/EEG data."""

# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Denis Engemann <denis.engemann@gmail.com>
#          Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#          Eric Larson <larson.eric.d@gmail.com>
#          Mainak Jas <mainak@neuro.hut.fi>
#          Mark Wronkiewicz <wronk.mark@gmail.com>
#
# License: Simplified BSD

import numpy as np
from mne.io.constants import FIFF
from mne.io.pick import _picks_to_idx
from mne.utils import _validate_type, fill_doc, verbose

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
        The measurement info.
    con : array, shape (n_channels, n_channels) | Connectivity
        The connectivity data to plot.
    %(picks_good_data)s
        Indices of selected channels.
    cbar_label : str
        Label for the colorbar.
    n_con : int
        Number of strongest connections shown (default 20).
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
    fig : instance of Renderer
        The 3D figure.
    """
    _validate_type(info, "info")

    from mne.viz.backends.renderer import _get_renderer

    from mne_connectivity.base import BaseConnectivity

    if isinstance(con, BaseConnectivity):
        con = con.get_data()

    renderer = _get_renderer(size=(600, 600), bgcolor=(0.5, 0.5, 0.5))

    if con.ndim != 2 or con.shape[0] != con.shape[1]:
        raise ValueError(
            "Connectivity data must be a 2D array of shape (n_channels, n_channels), "
            f"got shape {con.shape}"
        )

    picks = _picks_to_idx(info, picks)
    if len(picks) != len(con):
        raise ValueError(
            f"The number of channels picked ({len(picks)}) does not correspond to the "
            f"size of the connectivity data ({len(con)})"
        )

    if min_distance <= 0:
        raise ValueError(
            "The minimum distance between sensors must be greater than 0 m, got "
            f"{min_distance} m"
        )

    # Plot the sensor locations
    sens_loc = [info["chs"][k]["loc"][:3] for k in picks]
    sens_loc = np.array(sens_loc)

    renderer.sphere(
        np.c_[sens_loc[:, 0], sens_loc[:, 1], sens_loc[:, 2]],
        color=(1, 1, 1),
        opacity=1,
        scale=0.005,
    )

    # Get the strongest n_con connections
    threshold = np.sort(con, axis=None)[-n_con]
    ii, jj = np.where(con >= threshold)

    # Remove close connections
    con_nodes = list()
    con_val = list()
    for i, j in zip(ii, jj):
        if np.linalg.norm(sens_loc[i] - sens_loc[j]) > min_distance:
            con_nodes.append((i, j))
            con_val.append(con[i, j])
    con_val = np.array(con_val)

    if con_val.size == 0:
        raise ValueError(
            f"None of the {n_con} strongest connections were at least {min_distance} m "
            "apart. Try decreasing `min_distance` or increasing `n_con`, and check "
            "that the coordinates of your channels in `info` are not NaNs"
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

    for node in nodes_shown:
        x, y, z = sens_loc[node]
        renderer.text3d(
            x, y, z, text=info["ch_names"][picks[node]], scale=0.005, color=(0, 0, 0)
        )

    renderer.set_camera(
        azimuth=-88.7,
        elevation=40.8,
        distance=0.76,
        focalpoint=np.array([-3.9e-4, -8.5e-3, -1e-2]),
    )
    renderer.show()
    return renderer.scene()
