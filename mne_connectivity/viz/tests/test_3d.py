import os.path as op

import mne
import numpy as np
import pytest
from matplotlib import colormaps
from mne.datasets import testing
from numpy.testing import assert_almost_equal, assert_array_equal

from mne_connectivity.viz import plot_sensors_connectivity
from mne_connectivity.viz._3d import _get_strongest_connections

data_dir = testing.data_path(download=False)


@testing.requires_testing_data
def test_plot_sensors_connectivity(renderer):
    """Test plotting of sensors connectivity."""
    data_path = data_dir
    raw_fname = op.join(data_path, "MEG", "sample", "sample_audvis_trunc_raw.fif")
    raw = mne.io.read_raw_fif(raw_fname)
    info = raw.info
    picks = mne.pick_types(
        raw.info, meg="grad", eeg=False, stim=False, eog=True, exclude="bads"
    )
    n_channels = len(picks)
    rng = np.random.default_rng(42)
    con = rng.standard_normal((n_channels, n_channels))

    cmap = "viridis"
    fig = plot_sensors_connectivity(info=info, con=con, picks=picks, cmap=cmap)
    # check colormap
    cmap_from_mpl = np.array(colormaps[cmap].colors)
    cmap_from_vtk = np.array(fig.plotter.scalar_bar.GetLookupTable().GetTable())
    # discard alpha channel and convert uint8 -> norm
    cmap_from_vtk = cmap_from_vtk[:, :3] / 255
    cmap_from_vtk = cmap_from_vtk[::-1]  # for some reason order is flipped
    assert_almost_equal(cmap_from_mpl, cmap_from_vtk, decimal=2)
    # check title
    title = list(fig.plotter.scalar_bars.values())[0].GetTitle()
    assert title == "Connectivity"


@testing.requires_testing_data
def test_plot_sensors_connectivity_error_catch(renderer):
    """Test `plot_sensors_connectivity` catches errors."""
    # Get data to plot
    data_path = data_dir
    raw_fname = op.join(data_path, "MEG", "sample", "sample_audvis_trunc_raw.fif")
    raw = mne.io.read_raw_fif(raw_fname)
    info = raw.info
    picks = mne.pick_types(
        raw.info, meg="grad", eeg=False, stim=False, eog=True, exclude="bads"
    )
    n_channels = len(picks)
    rng = np.random.default_rng(42)
    con = rng.standard_normal((n_channels, n_channels))

    # Check errors caught
    # bad Info type
    with pytest.raises(TypeError, match="must be an instance of Info"):
        plot_sensors_connectivity(info="foo", con=con, picks=picks)
    # bad connectivity array shape
    with pytest.raises(ValueError, match="Connectivity data must be a 2D array"):
        plot_sensors_connectivity(info=info, con=np.expand_dims(con, 2), picks=picks)
    with pytest.raises(ValueError, match=r"array of shape \(n_channels, n_channels\)"):
        plot_sensors_connectivity(info=info, con=con[:, :-1], picks=picks)
    # mismatched channels and picks
    with pytest.raises(ValueError, match="does not correspond to the size"):
        plot_sensors_connectivity(info=info, con=con[::2, ::2], picks=picks)
    # bad minimum distance
    with pytest.raises(ValueError, match="distance between sensors must be greater"):
        plot_sensors_connectivity(info=info, con=con, picks=picks, min_distance=0)
    with pytest.raises(ValueError, match="distance between sensors must be greater"):
        plot_sensors_connectivity(info=info, con=con, picks=picks, min_distance=-1)
    # no surviving connections for minimum distance
    with pytest.raises(ValueError, match=r"No.*connections were at least.*apart"):
        plot_sensors_connectivity(info=info, con=con, picks=picks, min_distance=1e6)


def test_get_strongest_connections():
    """Test selecting the strongest connections to plot between sensors."""
    # symmetric, with self-connections on the diagonal (like coherence)
    con = np.array(
        [
            [1.0, 0.2, 0.7, 0.1],
            [0.2, 1.0, 0.3, 0.9],
            [0.7, 0.3, 1.0, 0.4],
            [0.1, 0.9, 0.4, 1.0],
        ]
    )
    rows, cols, values = _get_strongest_connections(con, n_con=3)
    # each pair once, strongest first, no self-connections
    assert_array_equal(np.sort([rows, cols], axis=0).T, [[1, 3], [0, 2], [2, 3]])
    assert_array_equal(values, [0.9, 0.7, 0.4])

    # antisymmetric (like imcoh): the positive direction of each pair is used
    con = np.array([[0.0, 0.5, -0.8], [-0.5, 0.0, 0.1], [0.8, -0.1, 0.0]])
    rows, cols, values = _get_strongest_connections(con, n_con=2)
    assert_array_equal(values, [0.8, 0.5])
    assert_array_equal(values, con[rows, cols])

    # signed: rank by absolute value
    con = np.array([[0.0, 0.0, 0.0], [-0.9, 0.0, 0.0], [0.5, 0.7, 0.0]])
    rows, cols, values = _get_strongest_connections(con, n_con=2)
    assert_array_equal(values, [-0.9, 0.7])

    # only one triangle filled
    con = np.tril([[0.0, 0.0, 0.0], [0.3, 0.0, 0.0], [-0.6, 0.2, 0.0]])
    rows, cols, values = _get_strongest_connections(con, n_con=2)
    assert_array_equal(values, [-0.6, 0.3])
    rows, cols, values = _get_strongest_connections(con.T, n_con=2)
    assert_array_equal(values, [-0.6, 0.3])

    # directed (like Granger causality): the stronger direction is used
    con = np.array([[0.0, 0.2, 0.0], [0.6, 0.0, 0.1], [0.0, 0.4, 0.0]])
    rows, cols, values = _get_strongest_connections(con, n_con=3)
    assert_array_equal(values, [0.6, 0.4, 0.0])
    assert_array_equal(values, con[rows, cols])

    # NaNs are ignored, and asking for more connections than exist returns all
    con = np.array([[np.nan, np.nan, 0.5], [np.nan, np.nan, np.nan], [0.5, 0.2, 0]])
    rows, cols, values = _get_strongest_connections(con, n_con=10)
    assert_array_equal(values, [0.5, 0.2])


@testing.requires_testing_data
def test_plot_sensors_connectivity_self_connections(renderer):
    """Test that self-connections do not take the place of real connections."""
    raw_fname = op.join(data_dir, "MEG", "sample", "sample_audvis_trunc_raw.fif")
    info = mne.io.read_raw_fif(raw_fname).info
    picks = mne.pick_types(info, meg="grad", exclude="bads")
    rng = np.random.default_rng(0)
    con = rng.uniform(0, 0.9, (len(picks), len(picks)))
    con = (con + con.T) / 2
    np.fill_diagonal(con, 1.0)  # coherence of a channel with itself
    plot_sensors_connectivity(info=info, con=con, picks=picks, n_con=5)
