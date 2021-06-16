# -*- coding: utf-8 -*-
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

from mne.io.pick import _picks_to_idx
from mne.io.constants import FIFF
from mne.utils import (verbose, fill_doc, _validate_type)


verbose_dec = verbose
FIDUCIAL_ORDER = (FIFF.FIFFV_POINT_LPA, FIFF.FIFFV_POINT_NASION,
                  FIFF.FIFFV_POINT_RPA)


@fill_doc
def plot_sensors_connectivity(info, con, picks=None,
                              cbar_label='Connectivity'):
    """Visualize the sensor connectivity in 3D.

    Parameters
    ----------
    info : dict | None
        The measurement info.
    con : array, shape (n_channels, n_channels)
        The computed connectivity measure(s).
    %(picks_good_data)s
        Indices of selected channels.
    cbar_label : str
        Label for the colorbar.

    Returns
    -------
    fig : instance of mayavi.mlab.Figure
        The mayavi figure.
    """
    _validate_type(info, "info")

    from mne.viz.backends.renderer import _get_renderer

    renderer = _get_renderer(size=(600, 600), bgcolor=(0.5, 0.5, 0.5))

    picks = _picks_to_idx(info, picks)
    if len(picks) != len(con):
        raise ValueError('The number of channels picked (%s) does not '
                         'correspond to the size of the connectivity data '
                         '(%s)' % (len(picks), len(con)))

    # Plot the sensor locations
    sens_loc = [info['chs'][k]['loc'][:3] for k in picks]
    sens_loc = np.array(sens_loc)

    renderer.sphere(np.c_[sens_loc[:, 0], sens_loc[:, 1], sens_loc[:, 2]],
                    color=(1, 1, 1), opacity=1, scale=0.005)

    # Get the strongest connections
    n_con = 20  # show up to 20 connections
    min_dist = 0.05  # exclude sensors that are less than 5cm apart
    threshold = np.sort(con, axis=None)[-n_con]
    ii, jj = np.where(con >= threshold)

    # Remove close connections
    con_nodes = list()
    con_val = list()
    for i, j in zip(ii, jj):
        if np.linalg.norm(sens_loc[i] - sens_loc[j]) > min_dist:
            con_nodes.append((i, j))
            con_val.append(con[i, j])

    con_val = np.array(con_val)

    # Show the connections as tubes between sensors
    vmax = np.max(con_val)
    vmin = np.min(con_val)
    for val, nodes in zip(con_val, con_nodes):
        x1, y1, z1 = sens_loc[nodes[0]]
        x2, y2, z2 = sens_loc[nodes[1]]
        tube = renderer.tube(origin=np.c_[x1, y1, z1],
                             destination=np.c_[x2, y2, z2],
                             scalars=np.c_[val, val],
                             vmin=vmin, vmax=vmax,
                             reverse_lut=True)

    renderer.scalarbar(source=tube, title=cbar_label)

    # Add the sensor names for the connections shown
    nodes_shown = list(set([n[0] for n in con_nodes] +
                           [n[1] for n in con_nodes]))

    for node in nodes_shown:
        x, y, z = sens_loc[node]
        renderer.text3d(x, y, z, text=info['ch_names'][picks[node]],
                        scale=0.005,
                        color=(0, 0, 0))

    renderer.set_camera(azimuth=-88.7, elevation=40.8,
                        distance=0.76,
                        focalpoint=np.array([-3.9e-4, -8.5e-3, -1e-2]))
    renderer.show()
    return renderer.scene()
