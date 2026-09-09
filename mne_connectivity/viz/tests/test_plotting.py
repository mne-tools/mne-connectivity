# Authors: The MNE-Connectivity developers.
#
# License: BSD-3-Clause

import mne
import numpy as np
import pytest
from matplotlib.colors import LogNorm
from mne.viz.utils import _fake_click
from numpy.testing import assert_allclose

from mne_connectivity import (
    Connectivity,
    SpectralConnectivity,
    SpectroTemporalConnectivity,
    TemporalConnectivity,
    plot_connectivity,
    plot_spectral_connectivity,
    plot_spectrotemporal_connectivity,
    plot_temporal_connectivity,
    seed_target_multivariate_indices,
)

N_NODES, N_FREQS, N_TIMES = 4, 5, 3
FREQS = np.arange(5.0, 5.0 + N_FREQS)
TIMES = np.arange(N_TIMES) / 10.0
# kind -> (plot function, class, extra positional args, per-connection data shape)
PLOTTERS = dict(
    matrix=(plot_connectivity, Connectivity, (), ()),
    spectral=(plot_spectral_connectivity, SpectralConnectivity, (FREQS,), (N_FREQS,)),
    temporal=(plot_temporal_connectivity, TemporalConnectivity, (TIMES,), (N_TIMES,)),
    spectrotemporal=(
        plot_spectrotemporal_connectivity,
        SpectroTemporalConnectivity,
        (FREQS, TIMES),
        (N_FREQS, N_TIMES),
    ),
)
# kwargs cropping the data, and the resulting x-axis limits, per kind
CROP = dict(
    spectral=(dict(fmin=6.0, fmax=8.0), (6.0, 8.0)),
    temporal=(dict(tmin=0.0, tmax=0.1), (0.0, 0.1)),
)
LINE_KINDS = ("spectral", "temporal")


def make_con(
    kind,
    *,
    indices="lower",
    symmetric=True,
    consistent_diag=True,
    n_comps=None,
    n_nodes=N_NODES,
    names=None,
):
    """Create a connectivity object of the requested kind, with random data."""
    _, klass, args, dims = PLOTTERS[kind]
    comps = () if n_comps is None else (n_comps,)
    kwargs = dict() if n_comps is None else dict(components=np.arange(n_comps))

    # Create random data for full connectivity
    data = np.random.default_rng(44).random((n_nodes, n_nodes, *comps, *dims))
    assert symmetric in (True, False, "unknown")
    if symmetric == "unknown":
        method = "unknown"
    elif symmetric:
        method = "coh"
        data = (data + data.transpose(1, 0, *range(2, data.ndim))) / 2.0
    else:
        method = "imcoh"
    if consistent_diag:
        data[np.arange(n_nodes), np.arange(n_nodes)] = 0.0

    # Trim the data to the requested connections
    if indices == "explicit":  # all-to-all tuple
        indices = np.unravel_index(np.arange(n_nodes**2), (n_nodes, n_nodes))
    assert indices in ("all", "lower", "upper") or isinstance(indices, tuple)
    if isinstance(indices, tuple):
        data = data[indices]
    elif indices == "all":
        data = data.reshape((-1, *data.shape[2:]))
    elif indices == "lower":
        data = data[np.tril_indices(n_nodes, -1)]
    else:  # upper
        data = data[np.triu_indices(n_nodes, 1)]

    if names is None:
        names = [f"ch{ii}" for ii in range(n_nodes)]

    return klass(
        data, *args, n_nodes=n_nodes, names=names, indices=indices, method=method,
        **kwargs,
    )  # fmt: skip


def click_node(fig, circle_ax, node, n_nodes, button=1):
    """Left/right click on a node of the circle plot."""
    # nodes sit at radius 9-10; offset the angle slightly so that the click never
    # lands exactly on the seam of the polar axes patch (where it is not contained)
    angle = 2 * np.pi * node / n_nodes + 0.05
    _fake_click(fig, circle_ax, (angle, 9.5), xform="data", button=button)


def unpack(out, kind):
    """Return the figures and the (line/image) axes of a plotting call as lists."""
    figs, axes = (out, None) if kind == "matrix" else out
    if not isinstance(figs, list):
        figs, axes = [figs], [axes]
    if kind == "matrix":
        axes = [fig.axes[0] for fig in figs]
    elif kind in LINE_KINDS:
        axes = [line_ax for line_ax, _ in axes]
    return figs, axes


def visible(ax):
    """Return the visibility of every line in an axes."""
    return [line.get_visible() for line in ax.lines]


@pytest.fixture
def info():
    """Return an info with two channel types (2 EEG, 2 grad) and no bads."""
    return mne.create_info(
        ["e0", "e1", "g0", "g1"], 100.0, ["eeg", "eeg", "grad", "grad"]
    )


def test_plot_matrix_connectivity():
    """Test plotting connectivity as a matrix."""
    con = make_con("matrix")

    fig = plot_connectivity(con, show=True)
    fig.canvas.draw()
    ax = fig.axes[0]
    assert (ax.get_xlabel(), ax.get_ylabel()) == ("Targets", "Seeds")
    assert ax.get_title() == f"misc ~ misc | {con.method}"
    assert ax.images[-1].colorbar.ax.get_ylabel() == "Connectivity (A.U.)"
    # Should be a square matrix spanning min-max nodes
    assert ax.get_xlim() == (-0.5, N_NODES - 0.5)
    assert ax.get_ylim() == (N_NODES - 0.5, -0.5)

    # Check square matrix also plotted when picking subset of nodes
    picks = (1,)
    indices = np.tril_indices(N_NODES, k=-1)  # explicit "lower" indices
    for selection in ("seeds", "targets", "both"):
        fig = plot_connectivity(con, picks=picks, selection=selection, show=True)
        fig.canvas.draw()
        ax = fig.axes[0]
        # Find which connections should be plotted
        eligible_seeds = [idx for idx, seed in enumerate(indices[0]) if seed in picks]
        eligible_targets = [
            idx for idx, target in enumerate(indices[1]) if target in picks
        ]
        if selection == "both":
            eligible_cons = sorted(set(eligible_seeds) | set(eligible_targets))
        elif selection == "seeds":
            eligible_cons = eligible_seeds
        else:  # selection == "targets"
            eligible_cons = eligible_targets
        min_node = np.min([indices[0][eligible_cons], indices[1][eligible_cons]])
        max_node = np.max([indices[0][eligible_cons], indices[1][eligible_cons]])
        assert ax.get_xlim() == (min_node - 0.5, max_node + 0.5)
        assert ax.get_ylim() == (max_node + 0.5, min_node - 0.5)


@pytest.mark.parametrize("symmetric", (True, False, "unknown"))
@pytest.mark.parametrize(
    ["indices", "consistent_diag"],
    [
        ("all", True),
        ("all", False),
        ("lower", True),
        ("upper", True),
        ("explicit", True),
        ("explicit", False),
    ],
)  # lower/upper with indonsistent diag isn't a valid combination
def test_plot_matrix_connectivity_visible_cons(indices, symmetric, consistent_diag):
    """Test connection visibility in the matrix plots."""
    con = make_con(
        "matrix", indices=indices, symmetric=symmetric, consistent_diag=consistent_diag
    )
    n_cons = con.get_data("raveled").shape[0]

    fig = plot_connectivity(con, show=False)
    fig.canvas.draw()
    ax = fig.axes[0]
    # Get image data, removing masked (i.e., NaN) values for missing entries
    data = ax.images[-1].get_array().compressed()

    # Check cell contents
    if indices == "explicit":
        # Show everything for explicit indices, regardless of possible symmetry or
        # diagonal having no actual info
        assert data.size == n_cons
    else:  # lower, upper, or all
        # Full matrix of connections will be plotted for data when all-to-all
        # connectivity is present, or full matrix can be inferred from tril/triu portion
        n_plotted_cons = N_NODES**2  # all cons as baseline
        n_tri_cons = N_NODES * (N_NODES - 1) // 2
        # Remove a tril/triu portion from plotted & visible when this cannot be inferred
        if indices != "all" and symmetric == "unknown":
            n_plotted_cons -= n_tri_cons  # remove a tril/triu portion
        # Remove diagonal from plotted & visible when this is not informative
        if consistent_diag:
            n_plotted_cons -= N_NODES
        assert data.size == n_plotted_cons


def test_plot_matrix_connectivity_options():
    """Test the colormap, colorbar, and masking options of the matrix plot."""
    con = make_con("matrix")
    # only the lower triangle of the all-to-all data is plotted
    data = con.get_data("dense")[np.tril_indices(N_NODES, -1)]
    lo, hi = data.min(), data.max()
    mixed = np.abs(data - 0.5).max()
    raveled = con.get_data("raveled")

    plot_connectivity(con)

    def remake(con, values):  # a copy of `con` with different data
        return Connectivity(
            values,
            n_nodes=con.n_nodes,
            indices=con.indices,
            names=con.names,
            method=con.method,
        )

    for this_con, kwargs, clim, cmap in (
        (con, dict(), (lo, hi), "Reds"),  # all-positive data spans its own limits
        (con, dict(vmin=0.2, vmax=0.8, cmap="viridis"), (0.2, 0.8), "viridis"),
        (con, dict(vmin=np.min, vmax=np.max), (lo, hi), "Reds"),  # callable bounds
        (con, dict(vmin=0.2), (0.2, hi), "Reds"),  # a missing bound falls back
        (con, dict(vmax=0.8), (lo, 0.8), "Reds"),
        (remake(con, -raveled), dict(), (-hi, -lo), "Blues_r"),  # all-negative data
        (remake(con, raveled - 0.5), dict(), (-mixed, mixed), "RdBu_r"),  # symmetric
    ):
        img = plot_connectivity(this_con, show=False, **kwargs).axes[0].images[0]
        assert (img.get_clim(), img.cmap.name) == (clim, cmap), kwargs

    # nodes are labelled by name, by tick index, or not at all
    for node_labels, expected in (("names", con.names), ("ticks", ["0"]), (None, [])):
        fig = plot_connectivity(con, node_labels=node_labels, show=False)
        fig.canvas.draw()
        labels = [text.get_text() for text in fig.axes[0].get_yticklabels()]
        assert set(expected) <= set(labels)
        assert (labels == []) == (node_labels is None)

    # the colorbar can be turned off, and an explicit normalization wins
    assert len(plot_connectivity(con, colorbar=False, show=False).axes) == 1
    cnorm = LogNorm(vmin=0.1, vmax=1.0)
    fig = plot_connectivity(con, cnorm=cnorm, show=False)
    assert fig.axes[0].images[0].norm is cnorm

    # masking
    mask = np.zeros((N_NODES, N_NODES), dtype=bool)
    mask[np.tril_indices(N_NODES, -2)] = True
    ax = plot_connectivity(con, mask=mask, mask_style="both", show=False).axes[0]
    assert len(ax.images) == 2  # masked and unmasked images
    assert len(ax.collections) > 0  # contour around the mask


def test_plot_matrix_connectivity_click():
    """Test clicking cells of the matrix plot to annotate them."""
    con = make_con("matrix")
    fig = plot_connectivity(con, show=False)
    fig.canvas.draw()
    ax = fig.axes[0]
    assert len(ax.texts) == 0

    _fake_click(fig, ax, (1.0, 2.0), xform="data")
    assert [text.get_text() for text in ax.texts] == ["ch2\n~\nch1"]
    assert len(ax.patches) == 1  # cell highlighted with a rectangle

    # clicking another cell replaces the previous annotation
    _fake_click(fig, ax, (0.0, 3.0), xform="data")
    assert [text.get_text() for text in ax.texts] == ["ch3\n~\nch0"]
    assert len(ax.patches) == 1

    # right-clicking clears it (twice is a no-op)
    for _ in range(2):
        _fake_click(fig, ax, (0.0, 3.0), xform="data", button=3)
        assert len(ax.texts) == 0 and len(ax.patches) == 0

    # clicks outside the axes or outside the matrix are ignored
    _fake_click(fig, fig.axes[1], (0.5, 0.5))
    _fake_click(fig, ax, (N_NODES + 1.0, 0.0), xform="data")
    assert len(ax.texts) == 0


@pytest.mark.parametrize("kind", LINE_KINDS)
def test_plot_line_connectivity(kind):
    """Test plotting connectivity as lines with a circle plot overview."""
    plot_func = PLOTTERS[kind][0]
    con = make_con(kind)
    n_cons = con.get_data("raveled").shape[0]
    xvar = con.freqs if kind == "spectral" else con.times
    xlabel = "Frequency (Hz)" if kind == "spectral" else "Time (s)"

    fig, (line_ax, circle_ax) = plot_func(con, show=False)
    assert (line_ax.get_xlabel(), line_ax.get_ylabel()) == (
        xlabel,
        "Connectivity (A.U.)",
    )
    assert line_ax.get_title() == f"misc ~ misc | {con.method}"
    assert circle_ax.get_title() == "Node selection\n(seeds and targets)"
    assert line_ax.get_xlim() == (xvar[0], xvar[-1])

    # cropping the x axis
    crop_kwargs, xlim = CROP[kind]
    _, (line_ax, _) = plot_func(con, show=False, **crop_kwargs)
    assert line_ax.get_xlim() == xlim

    # highlighting, both as a single (start, stop) pair and as several
    for highlight, n_extra in (((xvar[0], xvar[1]), 1), ([xvar[:2], xvar[-2:]], 2)):
        _, (line_ax, _) = plot_func(con, highlight=highlight, show=False)
        assert len(line_ax.collections) == n_extra

    # without interactivity the connections aren't pickable, and all are visible
    _, (line_ax, circle_ax) = plot_func(con, interactive=False, show=False)
    assert circle_ax.get_title() == "Nodes"
    assert visible(line_ax) == [True] * n_cons
    assert not any(line.get_picker() for line in line_ax.lines)


@pytest.mark.parametrize("kind", LINE_KINDS)
@pytest.mark.parametrize("symmetric", (True, False, "unknown"))
@pytest.mark.parametrize(
    ["indices", "consistent_diag"],
    [
        ("all", True),
        ("all", False),
        ("lower", True),
        ("upper", True),
        ("explicit", True),
        ("explicit", False),
    ],
)  # lower/upper with indonsistent diag isn't a valid combination
@pytest.mark.parametrize("selection", ("seeds", "targets", "both"))
def test_plot_line_connectivity_visible_cons(
    kind, indices, symmetric, consistent_diag, selection
):
    """Test connection visibility in the line plots.

    For connectivity data that is symmetric, it is better to start with only a subset of
    the connections visible, and alter this as different nodes are selected in the
    circle plot. For symmetric lower-/upper-triangular data, we also fill in the missing
    values to improve interactive visualisation. We exlude the diagonal connections from
    plotting if they are not informative (i.e., if they are all the same value).
    """
    plot_func = PLOTTERS[kind][0]
    con = make_con(
        kind, indices=indices, symmetric=symmetric, consistent_diag=consistent_diag
    )
    n_cons = con.get_data("raveled").shape[0]

    fig, (line_ax, circle_ax) = plot_func(con, selection=selection, show=False)

    # Check line visibility
    if indices == "explicit":
        # Show everything for explicit indices, regardless of possible symmetry or
        # diagonal having no actual info
        assert visible(line_ax) == [True] * n_cons
    else:  # lower, upper, or all
        # Full matrix of connections will be plotted for data when all-to-all
        # connectivity is present, or full matrix can be inferred from tril/triu portion
        n_plotted_cons = n_visible_cons = N_NODES**2  # all cons as baseline
        n_tri_cons = N_NODES * (N_NODES - 1) // 2
        # Remove a tril/triu portion from plotted & visible when this cannot be inferred
        if indices != "all" and symmetric == "unknown":
            n_plotted_cons -= n_tri_cons  # remove a tril/triu portion
            n_visible_cons -= n_tri_cons
        # Remove a tril/triu portion from visible when data is symmetric
        if symmetric is True:
            n_visible_cons -= n_tri_cons
        # Remove diagonal from plotted & visible when this is not informative
        if consistent_diag:
            n_plotted_cons -= N_NODES
            n_visible_cons -= N_NODES
        assert visible(line_ax) == [True] * n_visible_cons + [False] * (
            n_plotted_cons - n_visible_cons
        )


@pytest.mark.parametrize("kind", LINE_KINDS)
@pytest.mark.parametrize("ci", ("sd", "range", 95.0, None))
def test_plot_line_connectivity_combine(kind, ci):
    """Test aggregating connections in the line plots."""
    plot_func = PLOTTERS[kind][0]
    con = make_con(kind)
    data = con.get_data("dense")[np.tril_indices(N_NODES, -1)]

    fig, (line_ax, circle_ax) = plot_func(con, combine="mean", ci=ci, show=False)
    assert circle_ax is None  # a single (combined) connection needs no circle plot
    assert len(line_ax.lines) == 1
    assert_allclose(line_ax.lines[0].get_ydata(), data.mean(axis=0))
    assert len(line_ax.collections) == (0 if ci is None else 1)

    # a callable combine is used as-is
    _, (line_ax, _) = plot_func(
        con, combine=lambda x: np.max(x, axis=0), ci=None, show=False
    )
    assert_allclose(line_ax.lines[0].get_ydata(), data.max(axis=0))


def test_plot_line_connectivity_interactive():
    """Test selecting nodes in the circle plot and connections in the line plot."""
    con = make_con("spectral")
    fig, (line_ax, circle_ax) = plot_spectral_connectivity(con, show=False)
    fig.canvas.draw()
    seeds, targets = np.tril_indices(N_NODES, -1)
    # connections are duplicated with the seeds and targets swapped, so that every
    # node can be selected as a seed
    dup_seeds = np.concatenate([seeds, targets])
    start = visible(line_ax)

    for node in range(N_NODES):
        click_node(fig, circle_ax, node, N_NODES)
        assert visible(line_ax) == list(dup_seeds == node), f"node {node}"
        # the connection labels are hidden again on selection
        assert all(text.get_alpha() == 0 for text in line_ax.texts)
    click_node(fig, circle_ax, 0, N_NODES, button=3)  # right click resets
    assert visible(line_ax) == start

    # clicks away from the nodes, with another button, or outside the circle plot
    # do nothing
    _fake_click(fig, circle_ax, (0.05, 5.0), xform="data")
    click_node(fig, circle_ax, 1, N_NODES, button=2)
    _fake_click(fig, line_ax, (0.5, 0.5))
    assert visible(line_ax) == start

    # clicking a connection labels it, clicking elsewhere hides the label
    line = line_ax.lines[0]
    _fake_click(fig, line_ax, (line.get_xdata()[1], line.get_ydata()[1]), xform="data")
    (text,) = line_ax.texts
    assert (text.get_text(), text.get_alpha()) == ("ch1 ~ ch0", 1.0)
    _fake_click(fig, line_ax, (line_ax.get_xlim()[0], line_ax.get_ylim()[1]), "data")
    assert text.get_alpha() == 0.0


@pytest.mark.parametrize(
    "selection, unselectable, selected, n_selected",
    [("seeds", "ch0", "ch2", 2), ("targets", "ch3", "ch2", 1)],
)
@pytest.mark.parametrize("colors", ("auto", "global", "relative"))
def test_plot_line_connectivity_selection(
    selection, unselectable, selected, n_selected, colors
):
    """Test restricting the plotted (and selectable) connections to picked channels."""
    con = make_con("spectral")
    fig, (line_ax, circle_ax) = plot_spectral_connectivity(
        con, picks=["ch1", "ch2"], selection=selection, colors=colors,
        cmap="viridis", show=False,
    )  # fmt: skip
    fig.canvas.draw()
    assert len(line_ax.lines) == 3  # connections are duplicated only for "both"
    node_names = [text.get_text() for text in circle_ax.texts]
    assert len(node_names) == 3

    # nodes that cannot act as the selected role are drawn faded out
    alphas = [node.get_alpha() for node in circle_ax.containers[0]]
    assert [name for name, alpha in zip(node_names, alphas) if alpha is not None] == [
        unselectable
    ]

    # clicking a faded node does nothing; a selectable one isolates its connections
    click_node(fig, circle_ax, node_names.index(unselectable), len(node_names))
    assert visible(line_ax) == [True] * 3
    click_node(fig, circle_ax, node_names.index(selected), len(node_names))
    assert sum(visible(line_ax)) == n_selected


def test_plot_spectrotemporal_connectivity():
    """Test plotting spectro-temporal connectivity as images."""
    con = make_con("spectrotemporal")
    data = con.get_data("dense")[np.tril_indices(N_NODES, -1)]

    # connections are averaged by default, giving one figure instead of one each
    fig, ax = plot_spectrotemporal_connectivity(con, show=False)
    assert (ax.get_xlabel(), ax.get_ylabel()) == ("Time (s)", "Frequency (Hz)")
    assert ax.get_title() == f"misc ~ misc | combined nodes (n={N_CONS}) | coh"
    assert_allclose(ax.images[0].get_array(), data.mean(axis=0))
    figs, axes = plot_spectrotemporal_connectivity(con, combine=None, show=False)
    assert len(figs) == len(axes) == N_CONS
    assert axes[0].get_title() == "misc ~ misc | ch1 ~ ch0 | coh"

    # cropping in time and frequency, masking, and a log-spaced frequency axis
    mask = np.zeros((N_FREQS, N_TIMES), dtype=bool)
    mask[1:, 1:] = True
    _, ax = plot_spectrotemporal_connectivity(
        con, fmin=6.0, fmax=8.0, tmin=0.0, tmax=0.1, mask=mask, mask_style="mask",
        mask_cmap=None, colorbar=False, show=False,
    )  # fmt: skip
    assert ax.images[0].get_array().shape == (3, 2)  # cropped
    assert len(ax.images) == 2 and len(ax.get_figure().axes) == 1  # masked, no cbar
    _, ax = plot_spectrotemporal_connectivity(con, yscale="log", show=False)
    assert ax.get_yscale() == "log"


@pytest.mark.parametrize("kind", list(PLOTTERS))
def test_plot_connectivity_channel_types(kind, info):
    """Test splitting figures by channel type and dropping bad channels."""
    plot_func = PLOTTERS[kind][0]
    con = make_con(kind, names=info["ch_names"])

    types = ["EEG ~ EEG", "Gradiometers ~ EEG", "Gradiometers ~ Gradiometers"]
    figs, axes = unpack(plot_func(con, info=info, show=False), kind)
    assert len(figs) == len(axes) == 3
    assert [ax.get_title().split(" | ")[0] for ax in axes] == types

    # dropping a bad EEG channel leaves only one EEG node, so no EEG ~ EEG figure
    info["bads"] = ["e1"]
    figs, axes = unpack(plot_func(con, info=info, show=False), kind)
    assert len(figs) == len(axes) == 2
    assert [ax.get_title().split(" | ")[0] for ax in axes] == types[1:]


@pytest.mark.parametrize("kind", list(PLOTTERS))
@pytest.mark.parametrize("form", ("dense", "ragged", "masked"))
def test_plot_connectivity_multivariate(kind, form):
    """Test plotting multivariate connectivity with components and node aliases."""
    plot_func = PLOTTERS[kind][0]
    aliases = {(0, 1): "left", (2, 3, 4): "right"}
    if form == "dense":  # a single pair of equally sized nodes, in both directions
        indices = (np.array([[0, 1], [2, 3]]), np.array([[2, 3], [0, 1]]))
        aliases = {(0, 1): "left", (2, 3): "right"}
    elif form == "ragged":  # nodes of unequal size, giving 2 x 2 = 4 connections
        indices = seed_target_multivariate_indices(
            [[0, 1], [2, 3, 4]], [[2, 3, 4], [0, 1]]
        )
    else:  # the same nodes, padded out into a rectangular masked array
        indices = tuple(
            np.ma.masked_values(idcs, -1)
            for idcs in ([[0, 1, -1], [2, 3, 4]], [[2, 3, 4], [0, 1, -1]])
        )
    n_cons = len(indices[0])
    con = make_con(kind, indices=indices, n_comps=2, n_nodes=5)

    figs, axes = unpack(plot_func(con, node_aliases=aliases, show=False), kind)
    if kind == "spectrotemporal":  # connections are combined per component
        assert len(figs) == 2
        title = f"misc ~ misc | combined nodes (0; n={n_cons}) | coh"
        assert axes[0].get_title() == title
        return
    assert len(figs) == len(axes) == 1
    if kind in LINE_KINDS:
        # each component of each connection is drawn as its own line
        assert len(axes[0].lines) == n_cons * 2
        figs[0].canvas.draw()
        _fake_click(figs[0], axes[0], axes[0].lines[0].get_xydata()[1], xform="data")
        assert axes[0].texts[0].get_text() == "left ~ right (0)"


@pytest.mark.parametrize(
    "kind, kwargs, error, match",
    [
        # checks specific to these functions
        ("spectral", dict(con=0), TypeError, "instance of SpectralConnectivity"),
        ("matrix", dict(complex=True), ValueError, "complex-valued connectivity"),
        ("spectral", dict(bad_info=True), ValueError, "Missing channels"),
        ("spectral", dict(node_aliases={9: "x"}), ValueError, "must be present in"),
        ("spectral", dict(ci=101.0), ValueError, "must be > 0 and <= 100"),
        ("spectral", dict(highlight=[1.0, 2.0, 3.0]), ValueError, "shape \\(2,\\)"),
        (
            "spectrotemporal",
            dict(mask=np.zeros((2, 2), dtype=bool)),
            ValueError,
            "Mask shape .* does not match data shape",
        ),
        # one representative check per set of allowed option values
        ("spectral", dict(selection="bad"), ValueError, "the 'selection' parameter"),
        ("spectral", dict(combine="bad"), ValueError, "the 'combine' parameter"),
        ("matrix", dict(node_labels="bad"), ValueError, "the 'node_labels' parameter"),
    ],
)
def test_plot_connectivity_errors(kind, kwargs, error, match):
    """Test the input validation of the connectivity plotting functions."""
    plot_func, klass, args, _ = PLOTTERS[kind]
    con = kwargs.pop("con", None)
    if con is None:
        con = make_con(kind)
        if kwargs.pop("complex", False):
            con = klass(
                con.get_data("raveled") * 1j, *args, n_nodes=N_NODES, names=con.names
            )
    if kwargs.pop("bad_info", False):
        kwargs["info"] = mne.create_info(["other"], 1.0, "misc")
    with pytest.raises(error, match=match):
        plot_func(con, show=False, **kwargs)


def test_plot_connectivity_explicit_indices():
    """Test plotting bivariate connectivity for explicitly indexed connections."""
    # upper-triangular, i.e. all-to-all but not in the layout the plot expects
    indices = (np.array([0, 0, 1]), np.array([1, 2, 2]))
    con = make_con("spectral", indices=indices)
    fig, (line_ax, _) = plot_spectral_connectivity(
        con, node_aliases={0: "first"}, show=True
    )
    fig.canvas.draw()
    assert len(line_ax.lines) == 3  # not all-to-all, so no duplicated connections
    _fake_click(fig, line_ax, line_ax.lines[0].get_xydata()[1], xform="data")
    assert line_ax.texts[0].get_text() == "first ~ ch1"
    assert con.names == [f"ch{ii}" for ii in range(N_NODES)]  # not renamed in place
