import os.path as op

import mne
import numpy as np
import pytest
from matplotlib import colormaps
from mne.datasets import testing
from numpy.testing import assert_almost_equal

from mne_connectivity.viz import plot_sensors_connectivity

data_dir = testing.data_path(download=False)


@testing.requires_testing_data
def test_plot_sensors_connectivity(renderer):
    """Test plotting of sensors connectivity."""
    data_path = data_dir
    raw_fname = op.join(data_path, "MEG", "sample", "sample_audvis_trunc_raw.fif")

    raw = mne.io.read_raw_fif(raw_fname)
    picks = mne.pick_types(
        raw.info, meg="grad", eeg=False, stim=False, eog=True, exclude="bads"
    )
    n_channels = len(picks)
    con = np.random.RandomState(42).randn(n_channels, n_channels)
    info = raw.info
    with pytest.raises(TypeError, match="must be an instance of Info"):
        plot_sensors_connectivity(info="foo", con=con, picks=picks)
    with pytest.raises(ValueError, match="does not correspond to the size"):
        plot_sensors_connectivity(info=info, con=con[::2, ::2], picks=picks)
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
