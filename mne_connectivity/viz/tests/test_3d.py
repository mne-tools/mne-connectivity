import os.path as op

import mne
import numpy as np
import pytest
from matplotlib import colormaps
from mne.datasets import testing
from numpy.testing import assert_almost_equal, assert_array_equal

from mne_connectivity import Connectivity, SpectralConnectivity
from mne_connectivity.viz import plot_sensors_connectivity
from mne_connectivity.viz._3d import _get_strongest_connections

data_dir = testing.data_path(download=False)


def _grad_con(dense, **kwargs):
    """Make an all-to-all Connectivity object for the good gradiometers."""
    raw_fname = op.join(data_dir, "MEG", "sample", "sample_audvis_trunc_raw.fif")
    info = mne.io.read_raw_fif(raw_fname).info
    picks = mne.pick_types(info, meg="grad", exclude="bads")
    names = [info["ch_names"][pick] for pick in picks]
    con = Connectivity(
        dense(len(names)).ravel(), n_nodes=len(names), names=names, indices="all"
    )
    return info, con


@testing.requires_testing_data
def test_plot_sensors_connectivity(renderer):
    """Test plotting of sensors connectivity."""
    rng = np.random.default_rng(42)
    info, con = _grad_con(lambda n: rng.standard_normal((n, n)))

    cmap = "viridis"
    fig = plot_sensors_connectivity(info=info, con=con, cmap=cmap)
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

    # self-connections (like coherence of 1 with itself) do not take the place of
    # real connections, so that this plots instead of finding no connections to show
    def coherence_like(n):
        dense = rng.uniform(0, 0.9, (n, n))
        dense = (dense + dense.T) / 2
        np.fill_diagonal(dense, 1.0)
        return dense

    info, con = _grad_con(coherence_like)
    plot_sensors_connectivity(info=info, con=con, n_con=5)


def test_plot_sensors_connectivity_errors():
    """Test `plot_sensors_connectivity` catches errors before plotting."""
    names = ["a", "b", "c", "d"]
    info = mne.create_info(names, 1000.0, "eeg")
    ch_pos = {name: [0.1 * idx, 0.0, 0.0] for idx, name in enumerate(names)}  # 10 cm
    info.set_montage(mne.channels.make_dig_montage(ch_pos, coord_frame="head"))
    con = Connectivity(
        np.random.default_rng(0).standard_normal(16),
        n_nodes=4,
        names=names,
        indices="all",
    )

    # bad Info type
    with pytest.raises(TypeError, match="must be an instance of Info"):
        plot_sensors_connectivity(info="foo", con=con)
    # only Connectivity objects are accepted as data
    with pytest.raises(TypeError, match="must be an instance of Connectivity"):
        plot_sensors_connectivity(info, np.zeros((4, 4)))
    spectral = SpectralConnectivity(
        np.zeros((16, 1)), freqs=[10.0], n_nodes=4, names=names, indices="all"
    )
    with pytest.raises(TypeError, match="must be an instance of Connectivity"):
        plot_sensors_connectivity(info, spectral)
    # channels missing from info
    with pytest.raises(ValueError, match="Not all channel names"):
        plot_sensors_connectivity(info=mne.pick_info(info, [0, 1]), con=con)
    # bad minimum distance
    with pytest.raises(ValueError, match="distance between sensors must be greater"):
        plot_sensors_connectivity(info=info, con=con, min_distance=0)
    with pytest.raises(ValueError, match="distance between sensors must be greater"):
        plot_sensors_connectivity(info=info, con=con, min_distance=-1)
    # no surviving connections for minimum distance
    with pytest.raises(ValueError, match=r"No.*connections were at least.*apart"):
        plot_sensors_connectivity(info=info, con=con, min_distance=1e6)


def test_get_strongest_connections():
    """Test selecting the strongest connections to plot between sensors."""
    names = ["a", "b", "c", "d"]
    info = mne.create_info(names, 1000.0, "eeg")

    def all_to_all(dense, method=None):
        return Connectivity(
            np.array(dense).ravel(),
            n_nodes=4,
            names=names,
            indices="all",
            method=method,
        )

    # symmetric, with self-connections on the diagonal (like coherence): the diagonal
    # and the duplicate half are ignored
    dense = [
        [1.0, 0.2, 0.7, 0.1],
        [0.2, 1.0, 0.3, 0.9],
        [0.7, 0.3, 1.0, 0.4],
        [0.1, 0.9, 0.4, 1.0],
    ]
    seeds, targets, values = _get_strongest_connections(all_to_all(dense), info, 3)
    assert_array_equal(np.sort([seeds, targets], axis=0).T, [[1, 3], [0, 2], [2, 3]])
    assert_array_equal(values, [0.9, 0.7, 0.4])

    # signed: rank by absolute value
    dense = np.zeros((4, 4))
    dense[1, 0] = dense[0, 1] = -0.9
    dense[2, 1] = dense[1, 2] = 0.7
    dense[3, 2] = dense[2, 3] = 0.5
    _, _, values = _get_strongest_connections(all_to_all(dense), info, 2)
    assert_array_equal(values, [-0.9, 0.7])

    # connections of bad channels are not shown
    info_bad = info.copy()
    info_bad["bads"] = ["b"]
    seeds, targets, values = _get_strongest_connections(all_to_all(dense), info_bad, 1)
    assert_array_equal([seeds, targets], [[3], [2]])
    assert_array_equal(values, [0.5])

    # picks keeps the connections involving any of the picked channels
    dense = np.arange(16.0).reshape(4, 4)
    dense = dense + dense.T  # symmetric
    np.fill_diagonal(dense, 0.0)
    seeds, targets, _ = _get_strongest_connections(
        all_to_all(dense), info, 6, picks=["a"]
    )
    assert_array_equal(np.sort([seeds, targets], axis=0).T, [[0, 3], [0, 2], [0, 1]])

    # explicit indices are used as given
    con = Connectivity(
        np.array([0.3, -0.6]),
        n_nodes=4,
        names=names,
        indices=(np.array([0, 1]), np.array([2, 3])),
    )
    seeds, targets, values = _get_strongest_connections(con, info, 5)
    assert_array_equal(seeds, [1, 0])
    assert_array_equal(targets, [3, 2])
    assert_array_equal(values, [-0.6, 0.3])

    # multivariate connectivity is not supported
    con = Connectivity(
        np.array([0.5]),
        n_nodes=4,
        names=names,
        indices=(np.array([[0, 1]]), np.array([[2, 3]])),
        method="cacoh",
    )
    with pytest.raises(ValueError, match="Multivariate connectivity is not supported"):
        _get_strongest_connections(con, info, 5)
