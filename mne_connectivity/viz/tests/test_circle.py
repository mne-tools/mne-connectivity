# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Denis Engemann <denis.engemann@gmail.com>
#          Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#
# License: Simplified BSD


import matplotlib as mpl
import numpy as np
import pytest
from matplotlib.figure import Figure
from matplotlib.patches import PathPatch
from matplotlib.projections.polar import PolarAxes
from mne.viz import circular_layout

from mne_connectivity import Connectivity
from mne_connectivity.viz import plot_connectivity_circle


def test_plot_connectivity_circle():
    """Test plotting connectivity circle."""
    node_order = [
        "frontalpole-lh",
        "parsorbitalis-lh",
        "lateralorbitofrontal-lh",
        "rostralmiddlefrontal-lh",
        "medialorbitofrontal-lh",
        "parstriangularis-lh",
        "rostralanteriorcingulate-lh",
        "temporalpole-lh",
        "parsopercularis-lh",
        "caudalanteriorcingulate-lh",
        "entorhinal-lh",
        "superiorfrontal-lh",
        "insula-lh",
        "caudalmiddlefrontal-lh",
        "superiortemporal-lh",
        "parahippocampal-lh",
        "middletemporal-lh",
        "inferiortemporal-lh",
        "precentral-lh",
        "transversetemporal-lh",
        "posteriorcingulate-lh",
        "fusiform-lh",
        "postcentral-lh",
        "bankssts-lh",
        "supramarginal-lh",
        "isthmuscingulate-lh",
        "paracentral-lh",
        "lingual-lh",
        "precuneus-lh",
        "inferiorparietal-lh",
        "superiorparietal-lh",
        "pericalcarine-lh",
        "lateraloccipital-lh",
        "cuneus-lh",
        "cuneus-rh",
        "lateraloccipital-rh",
        "pericalcarine-rh",
        "superiorparietal-rh",
        "inferiorparietal-rh",
        "precuneus-rh",
        "lingual-rh",
        "paracentral-rh",
        "isthmuscingulate-rh",
        "supramarginal-rh",
        "bankssts-rh",
        "postcentral-rh",
        "fusiform-rh",
        "posteriorcingulate-rh",
        "transversetemporal-rh",
        "precentral-rh",
        "inferiortemporal-rh",
        "middletemporal-rh",
        "parahippocampal-rh",
        "superiortemporal-rh",
        "caudalmiddlefrontal-rh",
        "insula-rh",
        "superiorfrontal-rh",
        "entorhinal-rh",
        "caudalanteriorcingulate-rh",
        "parsopercularis-rh",
        "temporalpole-rh",
        "rostralanteriorcingulate-rh",
        "parstriangularis-rh",
        "medialorbitofrontal-rh",
        "rostralmiddlefrontal-rh",
        "lateralorbitofrontal-rh",
        "parsorbitalis-rh",
        "frontalpole-rh",
    ]
    label_names = [
        "bankssts-lh",
        "bankssts-rh",
        "caudalanteriorcingulate-lh",
        "caudalanteriorcingulate-rh",
        "caudalmiddlefrontal-lh",
        "caudalmiddlefrontal-rh",
        "cuneus-lh",
        "cuneus-rh",
        "entorhinal-lh",
        "entorhinal-rh",
        "frontalpole-lh",
        "frontalpole-rh",
        "fusiform-lh",
        "fusiform-rh",
        "inferiorparietal-lh",
        "inferiorparietal-rh",
        "inferiortemporal-lh",
        "inferiortemporal-rh",
        "insula-lh",
        "insula-rh",
        "isthmuscingulate-lh",
        "isthmuscingulate-rh",
        "lateraloccipital-lh",
        "lateraloccipital-rh",
        "lateralorbitofrontal-lh",
        "lateralorbitofrontal-rh",
        "lingual-lh",
        "lingual-rh",
        "medialorbitofrontal-lh",
        "medialorbitofrontal-rh",
        "middletemporal-lh",
        "middletemporal-rh",
        "paracentral-lh",
        "paracentral-rh",
        "parahippocampal-lh",
        "parahippocampal-rh",
        "parsopercularis-lh",
        "parsopercularis-rh",
        "parsorbitalis-lh",
        "parsorbitalis-rh",
        "parstriangularis-lh",
        "parstriangularis-rh",
        "pericalcarine-lh",
        "pericalcarine-rh",
        "postcentral-lh",
        "postcentral-rh",
        "posteriorcingulate-lh",
        "posteriorcingulate-rh",
        "precentral-lh",
        "precentral-rh",
        "precuneus-lh",
        "precuneus-rh",
        "rostralanteriorcingulate-lh",
        "rostralanteriorcingulate-rh",
        "rostralmiddlefrontal-lh",
        "rostralmiddlefrontal-rh",
        "superiorfrontal-lh",
        "superiorfrontal-rh",
        "superiorparietal-lh",
        "superiorparietal-rh",
        "superiortemporal-lh",
        "superiortemporal-rh",
        "supramarginal-lh",
        "supramarginal-rh",
        "temporalpole-lh",
        "temporalpole-rh",
        "transversetemporal-lh",
        "transversetemporal-rh",
    ]

    group_boundaries = [0, len(label_names) / 2]
    node_angles = circular_layout(
        label_names, node_order, start_pos=90, group_boundaries=group_boundaries
    )
    con = np.random.default_rng(0).standard_normal((68, 68))
    plot_connectivity_circle(
        con,
        label_names,
        n_lines=300,
        node_angles=node_angles,
        title="test",
    )

    # the plot_circle method of connectivity containers returns the figure and axes
    con = Connectivity(
        data=np.array([0.25, 0.5, 0.75]),
        n_nodes=3,
        names=["a", "b", "c"],
        indices="lower",
        method="coh",
    )
    fig, ax = con.plot_circle(show=False)
    assert isinstance(fig, Figure)
    assert isinstance(ax, PolarAxes)

    pytest.raises(
        ValueError, circular_layout, label_names, node_order, group_boundaries=[-1]
    )
    pytest.raises(
        ValueError, circular_layout, label_names, node_order, group_boundaries=[20, 0]
    )


def _line_colors(ax):
    """Get the colors of the connection lines, in drawing order."""
    return np.array([p.get_edgecolor() for p in ax.patches if isinstance(p, PathPatch)])


def test_plot_connectivity_circle_colormap():
    """Test automatic colormap and color limits of the circle plot."""
    names = ["a", "b", "c"]
    # lines are drawn in order of increasing absolute value: 0.25, 0.5, -1
    con = np.array([[0, 0, 0], [-1, 0, 0], [0.5, 0.25, 0]])

    # mixed signs: diverging colormap with limits symmetric around zero
    _, ax = plot_connectivity_circle(con, names, show=False)
    cmap = mpl.colormaps["RdBu_r"]
    want = cmap(np.array([0.625, 0.75, 0.0]))  # (value + 1) / 2
    np.testing.assert_allclose(_line_colors(ax), want)

    # symmetric limits also apply to a user-specified colormap...
    _, ax = plot_connectivity_circle(con, names, colormap="viridis", show=False)
    want = mpl.colormaps["viridis"](np.array([0.625, 0.75, 0.0]))
    np.testing.assert_allclose(_line_colors(ax), want)

    # ...but not when the user gives the limits
    _, ax = plot_connectivity_circle(con, names, vmin=-1, vmax=0.5, show=False)
    want = cmap(np.array([1.25 / 1.5, 1.0, 0.0]))  # (value + 1) / 1.5
    np.testing.assert_allclose(_line_colors(ax), want)

    # only the lower triangle is plotted, so the upper triangle's sign is ignored
    one_signed = np.abs(con) - np.abs(con).T  # negative only in the upper triangle
    _, ax_auto = plot_connectivity_circle(one_signed, names, show=False)
    _, ax_hot = plot_connectivity_circle(one_signed, names, colormap="hot", show=False)
    np.testing.assert_array_equal(_line_colors(ax_auto), _line_colors(ax_hot))

    # one-signed data keeps the previous default colormap and limits
    pos = np.abs(con)
    _, ax_auto = plot_connectivity_circle(pos, names, show=False)
    _, ax_hot = plot_connectivity_circle(pos, names, colormap="hot", show=False)
    np.testing.assert_array_equal(_line_colors(ax_auto), _line_colors(ax_hot))

    # the same applies to 1D data with indices
    indices = np.tril_indices(3, -1)
    _, ax = plot_connectivity_circle(con[indices], names, indices=indices, show=False)
    want = cmap(np.array([0.625, 0.75, 0.0]))
    np.testing.assert_allclose(_line_colors(ax), want)
