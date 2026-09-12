# Authors: Adam Li <adam2392@gmail.com>
#
# License: BSD (3-clause)

import os
from functools import partial

import numpy as np
import pandas as pd
import pytest
from mne import create_info, make_fixed_length_epochs
from mne.annotations import Annotations
from mne.epochs import BaseEpochs
from mne.io import RawArray
from numpy.testing import assert_allclose, assert_array_equal

from mne_connectivity import (
    Connectivity,
    EpochConnectivity,
    EpochSpectralConnectivity,
    EpochSpectroTemporalConnectivity,
    EpochTemporalConnectivity,
    SpectralConnectivity,
    SpectroTemporalConnectivity,
    TemporalConnectivity,
    envelope_correlation,
    phase_slope_index,
    read_connectivity,
    spectral_connectivity_epochs,
    spectral_connectivity_time,
    vector_auto_regression,
    wsmi,
)


def _make_test_epochs():
    sfreq = 50.0
    n_signals = 3
    n_epochs = 10
    n_times = 500
    rng = np.random.RandomState(42)
    data = rng.randn(n_signals, n_epochs * n_times)

    # create Epochs
    info = create_info(
        np.arange(n_signals).astype(str).tolist(), sfreq=sfreq, ch_types="eeg"
    )
    onset = [0, 0.5, 3]
    duration = [0, 0, 0]
    description = ["test1", "test2", "test3"]
    annots = Annotations(onset=onset, duration=duration, description=description)
    raw = RawArray(data, info)
    raw = raw.set_annotations(annots)
    epochs = make_fixed_length_epochs(raw, duration=1, preload=True)

    # make sure Epochs has metadata
    epochs.add_annotations_to_metadata()
    return epochs


def _prep_correct_connectivity_input(
    conn_cls, n_nodes=3, tril=False, n_epochs=4, indices=None, n_components=0
):
    correct_numpy_shape = []

    extra_kwargs = dict()
    if conn_cls.is_epoched:
        correct_numpy_shape.append(n_epochs)

    if indices is None:
        if tril:
            correct_numpy_shape.append((n_nodes - 1) * n_nodes // 2)
        else:
            correct_numpy_shape.append(n_nodes**2)
    else:
        correct_numpy_shape.append(len(indices[0]))

    if n_components:
        correct_numpy_shape.append(n_components)
        extra_kwargs["components"] = np.arange(n_components) + 1

    if conn_cls in (
        SpectralConnectivity,
        SpectroTemporalConnectivity,
        EpochSpectralConnectivity,
        EpochSpectroTemporalConnectivity,
    ):
        extra_kwargs["freqs"] = np.arange(4)
        correct_numpy_shape.append(4)
    if conn_cls in (
        TemporalConnectivity,
        SpectroTemporalConnectivity,
        EpochTemporalConnectivity,
        EpochSpectroTemporalConnectivity,
    ):
        extra_kwargs["times"] = np.arange(3)
        correct_numpy_shape.append(3)

    return correct_numpy_shape, extra_kwargs


@pytest.mark.parametrize(
    "conn_cls",
    [
        Connectivity,
        EpochConnectivity,
        SpectralConnectivity,
        TemporalConnectivity,
        SpectroTemporalConnectivity,
        EpochTemporalConnectivity,
        EpochSpectralConnectivity,
        EpochSpectroTemporalConnectivity,
    ],
)
def test_connectivity_containers(conn_cls):
    """Test connectivity classes."""
    n_epochs = 4
    n_nodes = 3
    data = [
        [1, 0, 0],
        [3, 4, 5],
        [0, 1, 2],
    ]
    bad_numpy_input = np.zeros((3, 3, 4, 5, 6))
    bad_indices = ([1, 0], [2])

    if conn_cls.is_epoched:
        bad_numpy_input = np.zeros((3, 3, 3, 4, 5, 6))

    correct_numpy_shape, extra_kwargs = _prep_correct_connectivity_input(
        conn_cls, n_nodes=n_nodes, tril=False, n_epochs=n_epochs
    )

    correct_numpy_input = np.ones(correct_numpy_shape)

    # test initialization error checks
    with pytest.raises(
        TypeError, match="Connectivity data must be passed in as a numpy array"
    ):
        conn_cls(data=data, n_nodes=2, **extra_kwargs)
    with pytest.raises(RuntimeError, match="Data"):
        conn_cls(data=bad_numpy_input, n_nodes=2, **extra_kwargs)
    with pytest.raises(ValueError, match="If indices are passed"):
        conn_cls(
            data=correct_numpy_input, indices=bad_indices, n_nodes=2, **extra_kwargs
        )
    with pytest.raises(ValueError, match="Invalid value for the 'indices' parameter"):
        conn_cls(data=correct_numpy_input, indices="square", n_nodes=2, **extra_kwargs)

    # test connectivity instantiation with 'all's
    conn = conn_cls(data=correct_numpy_input, n_nodes=3, indices="all", **extra_kwargs)
    with pytest.raises(ValueError, match="If `indices` is 'all'"):
        conn_cls(data=correct_numpy_input, n_nodes=2, indices="all", **extra_kwargs)

    # test that get_data works as intended
    assert conn.shape == tuple(correct_numpy_shape)
    assert conn.get_data(output="raveled").shape == tuple(correct_numpy_shape)
    assert conn.get_data(output="dense").ndim == len(correct_numpy_shape) + 1

    # test renaming nodes error checks
    with pytest.raises(ValueError, match="Name"):
        conn.rename_nodes({"100": "new_name"})
    with pytest.raises(ValueError, match="mapping must be"):
        conn.rename_nodes(["0", "new_name"])
    with pytest.raises(ValueError, match="New channel names"):
        conn.rename_nodes({"0": "1"})

    # test renaming nodes
    orig_names = conn.names
    conn.rename_nodes({"0": "new_name"})
    new_names = conn.names
    assert all(
        [
            name_1 == name_2
            for name_1, name_2 in zip(orig_names, new_names)
            if name_2 != "new_name"
        ]
    )
    conn.rename_nodes(lambda x: "0" if x == "new_name" else x)
    assert_array_equal(orig_names, conn.names)

    # test connectivity instantiation with indices
    indices = ([0, 1], [1, 0])
    indexed_numpy_shape, index_kwargs = _prep_correct_connectivity_input(
        conn_cls, n_nodes=n_nodes, tril=False, n_epochs=n_epochs, indices=indices
    )
    indexed_numpy_input = np.ones(indexed_numpy_shape)
    conn2 = conn_cls(
        data=indexed_numpy_input, n_nodes=2, indices=indices, **index_kwargs
    )
    conn3 = conn_cls(
        data=indexed_numpy_input, n_nodes=3, indices=indices, **index_kwargs
    )

    # the number of nodes helps define the full dense output, but
    # if unraveled, with indices then they should match exactly
    assert_array_equal(conn2.get_data(), conn3.get_data())

    # test getting data with indices specified
    with pytest.raises(ValueError, match="The number of indices"):
        conn_cls(data=correct_numpy_input, n_nodes=3, indices=indices, **extra_kwargs)

    # test lower-/upper-triangular input
    correct_numpy_shape, extra_kwargs = _prep_correct_connectivity_input(
        conn_cls, n_nodes=3, tril=True
    )
    correct_numpy_input = np.ones(correct_numpy_shape)

    for indices in ["lower", "upper"]:
        with pytest.raises(ValueError, match="If `indices` is 'lower' or 'upper'"):
            conn_cls(
                data=correct_numpy_input, n_nodes=2, indices=indices, **extra_kwargs
            )
        tri_conn = conn_cls(
            data=correct_numpy_input,
            n_nodes=n_nodes,
            indices=indices,
            method="coh",  # use a method where we can auto-fill missing values
            **extra_kwargs,
        )
        assert tri_conn.n_nodes == n_nodes

        # test that conversion to dense maps properly
        dense_out = tri_conn.get_data(missing=np.nan)
        if conn_cls.is_epoched:
            dense_out = np.moveaxis(dense_out, 0, -1)  # move epochs for indexing
            correct_numpy_input = np.moveaxis(correct_numpy_input, 0, -1)
        tril_inds = np.tril_indices(n_nodes, k=-1)
        triu_inds = np.triu_indices(n_nodes, k=1)
        if indices == "lower":
            assert_array_equal(dense_out[tril_inds], correct_numpy_input)
            assert_array_equal(dense_out[triu_inds], np.nan)
        else:
            assert_array_equal(dense_out[triu_inds], correct_numpy_input)
            assert_array_equal(dense_out[tril_inds], np.nan)
        if conn_cls.is_epoched:
            correct_numpy_input = np.moveaxis(correct_numpy_input, -1, 0)

    # raveled shape should be the same
    assert_array_equal(tri_conn.get_data(output="raveled").shape, correct_numpy_shape)

    # should be ([n_epochs], n_nodes, n_nodes, ...) dense shape
    dense_shape = []
    if conn_cls.is_epoched:
        dense_shape.append(n_epochs)
    dense_shape.extend([n_nodes, n_nodes])
    assert all(
        [
            tri_conn.get_data(output="dense").shape[idx] == dense_shape[idx]
            for idx in range(len(dense_shape))
        ]
    )


@pytest.mark.parametrize(
    "conn_cls",
    [
        Connectivity,
        EpochConnectivity,
        SpectralConnectivity,
        TemporalConnectivity,
        SpectroTemporalConnectivity,
        EpochTemporalConnectivity,
        EpochSpectralConnectivity,
        EpochSpectroTemporalConnectivity,
    ],
)
def test_connectivity_containers_multivariate(conn_cls):
    """Test that connectivity containers work properly with multivariate data."""
    indices = (
        np.array([[0, 1], [0, 1], [2, 3]]),
        np.array([[2, 3], [4, 5], [4, 5]]),
    )  # should map to upper-triangular elements

    # Find individual channels and nodes (sets of channels) in the data
    chans, nodes = set(), set()
    for seed, target in zip(*indices):
        for ch in seed:
            chans.add(ch)
        for ch in target:
            chans.add(ch)
        nodes.add(tuple(seed))
        nodes.add(tuple(target))

    # Create connectivity container
    correct_numpy_shape, index_kwargs = _prep_correct_connectivity_input(
        conn_cls, n_nodes=len(chans), indices=indices, n_components=2
    )
    data = np.ones(correct_numpy_shape, dtype=np.float64)
    con = conn_cls(data=data, indices=indices, n_nodes=len(chans), **index_kwargs)

    # Check no manipulation is performed for raveled output
    matrix = con.get_data(output="raveled")
    assert isinstance(matrix, np.ndarray)  # just data expected
    assert_array_equal(data, matrix)

    # Check that output gets mapped to new space for dense output
    out = con.get_data(output="dense", missing=np.nan)
    assert isinstance(out, tuple)  # data and multivariate_nodes expected
    assert len(out) == 2
    matrix, multivariate_nodes = out
    assert isinstance(matrix, np.ndarray)
    assert isinstance(multivariate_nodes, tuple)
    assert np.all(isinstance(ind, np.ndarray) for ind in multivariate_nodes)
    assert set(tuple(ind) for ind in multivariate_nodes) == nodes
    triu_indices = np.triu_indices(len(nodes), k=1)
    # TODO VERSION: use [*triu_indices] when Py3.10 dropped
    if conn_cls.is_epoched:
        assert_array_equal(matrix[:, triu_indices[0], triu_indices[1]], data)
    else:
        assert_array_equal(matrix[triu_indices[0], triu_indices[1]], data)


def test_get_data_error_catch():
    """Test that bad calls are caught for get_data()."""
    n_nodes = 3
    con = Connectivity(
        data=np.arange(n_nodes**2),
        n_nodes=n_nodes,
        indices="all",
        method="coh",  # use known method that support filling missing values
    )

    # Check bad output is caught
    with pytest.raises(ValueError, match="Invalid value for the 'output' parameter"):
        con.get_data(output="square")

    # Check bad missing is caught
    with pytest.raises(
        TypeError, match="`missing` must be an instance of str or numeric"
    ):
        con.get_data(missing=True)
    with pytest.raises(ValueError, match="Invalid value for the 'missing' parameter"):
        con.get_data(missing="warn")

    # Check that non-all-to-all indices errors when trying to fill missing values
    non_all_indices = ([0, 1], [1, 0])
    con_non_all = Connectivity(
        data=np.arange(len(non_all_indices[0])),
        n_nodes=n_nodes,
        indices=non_all_indices,
    )
    with pytest.raises(
        ValueError,
        match=(
            "Cannot fill missing values for connectivity data when indices are "
            "specified"
        ),
    ):
        con_non_all.get_data("dense")

    # Check that unknown method errors when trying to fill missing values
    for indices in ["lower", "upper"]:
        con_bad_meth = Connectivity(
            data=np.arange(n_nodes * (n_nodes - 1) // 2),
            n_nodes=n_nodes,
            indices=indices,
            method="who_knows",
        )
        with pytest.raises(
            ValueError,
            match="Cannot fill missing values for connectivity data for the method",
        ):
            con_bad_meth.get_data()


@pytest.mark.parametrize("indices", ["lower", "upper"])
def test_make_unknown_method_full(indices):
    """Test that filling missing values in unknown methods errors."""
    n_nodes = 3
    correct_numpy_shape, extra_kwargs = _prep_correct_connectivity_input(
        Connectivity, n_nodes=n_nodes, tril=True
    )
    correct_numpy_input = np.ones(correct_numpy_shape)
    con = Connectivity(
        data=correct_numpy_input, n_nodes=n_nodes, indices=indices, **extra_kwargs
    )

    with pytest.raises(ValueError, match="Cannot fill missing values for connectivity"):
        con.get_data()


def test_make_full_with_indices():
    """Test filling missing values in connectivity data with tuple indices."""
    n_nodes = 3
    correct_numpy_input = np.arange(n_nodes**2)

    # Check that connectivity data with explicit all-to-all indices works
    indices = np.unravel_index(np.arange(n_nodes**2), (n_nodes, n_nodes))
    con_ind = Connectivity(data=correct_numpy_input, n_nodes=n_nodes, indices=indices)
    con_all = Connectivity(data=correct_numpy_input, n_nodes=n_nodes, indices="all")
    assert_array_equal(con_ind.get_data("dense"), con_all.get_data())


# Time-resolved CIPLV can involve division by zero on diagonal
@pytest.mark.filterwarnings("ignore:divide by zero encountered in divide")
@pytest.mark.parametrize("kind", ["epochs", "time"])
def test_make_spec_conn_full(data_make_full, kind):
    """Test that filling missing values in spectral conn methods works correctly.

    For some methods, diagonal can be spurious depending on sample size, so we ignore it
    for those methods.
    """
    n_channels = data_make_full.info["nchan"]
    methods = ("coh", "cohy", "imcoh", "plv", "ciplv", "pli", "wpli")
    ignore_diag_methods = ("pli", "wpli", "pli2_unbiased", "dpli", "wpli2_debiased")

    # Get spectral coeffs
    if kind == "epochs":
        coeffs = data_make_full.compute_psd(method="welch", output="complex")
        methods += ("ppc", "pli2_unbiased", "dpli", "wpli2_debiased")
        conn_func = spectral_connectivity_epochs
    else:  # kind == "time"
        coeffs = data_make_full.compute_tfr(
            method="morlet", freqs=np.arange(15, 20), n_cycles=3, output="complex"
        )
        ignore_diag_methods += ("ciplv",)  # Can have inf diag
        conn_func = partial(spectral_connectivity_time, average=True)

    # Compute connectivity
    lower = conn_func(coeffs, method=methods, indices="lower")
    upper = conn_func(coeffs, method=methods, indices="upper")
    indices_all = np.unravel_index(np.arange(n_channels**2), (n_channels, n_channels))
    full = conn_func(coeffs, method=methods, indices=indices_all)

    # Check results are equivalent
    for this_lower, this_upper, this_full in zip(lower, upper, full, strict=True):
        lower_data = this_lower.get_data()
        upper_data = this_upper.get_data()
        full_data = this_full.get_data("dense")
        assert_allclose(lower_data, upper_data, atol=1e-6)
        if this_lower.method in ignore_diag_methods:
            lower_data[np.diag_indices(n_channels)] = 0.0
            full_data[np.diag_indices(n_channels)] = 0.0
        assert_allclose(lower_data, full_data, atol=1e-6)


@pytest.mark.parametrize("kind", ["epochs", "time"])
def test_make_psi_full(data_make_full, kind):
    """Test that filling missing values in PSI data works correctly."""
    n_channels = data_make_full.info["nchan"]

    # Get spectral coeffs
    if kind == "epochs":
        coeffs = data_make_full.compute_psd(method="welch", output="complex")
    else:  # kind == "time"
        coeffs = data_make_full.compute_tfr(
            method="morlet", freqs=np.arange(15, 20), n_cycles=3, output="complex"
        )

    # Compute connectivity
    lower = phase_slope_index(coeffs, indices="lower")
    upper = phase_slope_index(coeffs, indices="upper")
    indices_all = np.unravel_index(np.arange(n_channels**2), (n_channels, n_channels))
    full = phase_slope_index(coeffs, indices=indices_all)

    # Check results are equivalent
    assert_allclose(lower.get_data(), upper.get_data(), atol=1e-6)
    assert_allclose(lower.get_data(), full.get_data("dense"), atol=1e-6)


@pytest.mark.parametrize("weighted", [True, False])
def test_make_smi_full(data_make_full, weighted):
    """Test that filling missing values in SMI data works correctly."""
    n_channels = data_make_full.info["nchan"]

    # Compute connectivity
    lower = wsmi(data_make_full, kernel=3, tau=1, indices="lower", weighted=weighted)
    upper = wsmi(data_make_full, kernel=3, tau=1, indices="upper", weighted=weighted)
    indices_all = np.unravel_index(np.arange(n_channels**2), (n_channels, n_channels))
    full = wsmi(data_make_full, kernel=3, tau=1, indices=indices_all, weighted=weighted)

    # Check results are equivalent
    assert_allclose(lower.get_data(), upper.get_data(), atol=1e-6)
    assert_allclose(lower.get_data(), full.get_data("dense"), atol=1e-6)


@pytest.mark.parametrize(
    "conn_cls",
    [
        Connectivity,
        EpochConnectivity,
        SpectralConnectivity,
        TemporalConnectivity,
        SpectroTemporalConnectivity,
        EpochTemporalConnectivity,
        EpochSpectralConnectivity,
        EpochSpectroTemporalConnectivity,
    ],
)
def test_io(conn_cls, tmpdir):
    """Test writing and reading connectivity data."""
    correct_numpy_shape = []
    extra_kwargs = dict()
    if conn_cls.is_epoched:
        correct_numpy_shape.append(4)
    correct_numpy_shape.append(4)
    if conn_cls in (
        SpectralConnectivity,
        SpectroTemporalConnectivity,
        EpochSpectralConnectivity,
        EpochSpectroTemporalConnectivity,
    ):
        extra_kwargs["freqs"] = np.arange(4)
        correct_numpy_shape.append(4)
    if conn_cls in (
        TemporalConnectivity,
        SpectroTemporalConnectivity,
        EpochTemporalConnectivity,
        EpochSpectroTemporalConnectivity,
    ):
        extra_kwargs["times"] = np.arange(3)
        correct_numpy_shape.append(3)

    correct_numpy_input = np.ones(correct_numpy_shape)

    # create the connectivity data structure
    conn = conn_cls(data=correct_numpy_input, n_nodes=2, **extra_kwargs)

    # temporary conn save
    fname = os.path.join(tmpdir, "connectivity.nc")
    conn.save(fname)

    # re-read the file in
    new_conn = read_connectivity(fname)

    # assert these two objects are the same
    assert_array_equal(conn.names, new_conn.names)
    assert conn.dims == new_conn.dims
    for key, val in conn.coords.items():
        assert_array_equal(val, new_conn.coords[key])
    assert_array_equal(conn.get_data(), new_conn.get_data())


@pytest.mark.parametrize(
    "conn_cls",
    [
        EpochConnectivity,
        EpochTemporalConnectivity,
        EpochSpectralConnectivity,
        EpochSpectroTemporalConnectivity,
    ],
)
def test_append(conn_cls):
    """Test appending connectivity data."""
    correct_numpy_shape = []
    extra_kwargs = dict()
    if conn_cls.is_epoched:
        correct_numpy_shape.append(4)
    correct_numpy_shape.append(4)
    if conn_cls in (
        SpectralConnectivity,
        SpectroTemporalConnectivity,
        EpochSpectralConnectivity,
        EpochSpectroTemporalConnectivity,
    ):
        extra_kwargs["freqs"] = np.arange(4)
        correct_numpy_shape.append(4)
    if conn_cls in (
        TemporalConnectivity,
        SpectroTemporalConnectivity,
        EpochTemporalConnectivity,
        EpochSpectroTemporalConnectivity,
    ):
        extra_kwargs["times"] = np.arange(50)
        correct_numpy_shape.append(50)

    correct_numpy_input = np.ones(correct_numpy_shape)
    events = np.zeros((correct_numpy_input.shape[0], 3), dtype=int)
    events[:, -1] = 1  # event ID
    events[:, 0] = np.linspace(0, 50, len(events))

    # create the connectivity data structure
    conn = conn_cls(data=correct_numpy_input, n_nodes=2, events=events, **extra_kwargs)

    # create a copy of the connectivity
    conn_2 = conn.copy()

    # append epochs
    conn.append(conn_2)
    assert conn.n_epochs == conn_2.n_epochs * 2
    assert len(conn.events) == conn.n_epochs


@pytest.mark.parametrize(
    "conn_func",
    [
        vector_auto_regression,
        spectral_connectivity_epochs,
        envelope_correlation,
        phase_slope_index,
    ],
)
def test_events_handling(conn_func):
    """Test that events and event_id are passed through correctly."""
    epochs = _make_test_epochs()
    n_epochs = len(epochs)
    assert len(epochs.events) == n_epochs

    # create the connectivity data structure
    conn = conn_func(epochs, verbose=False)
    assert len(conn.events) == n_epochs


@pytest.mark.parametrize(
    "epochs", [_make_test_epochs(), np.random.RandomState(0).random((10, 3, 500))]
)
@pytest.mark.parametrize(
    "func",
    [
        vector_auto_regression,
        spectral_connectivity_epochs,
        envelope_correlation,
        phase_slope_index,
    ],
)
def test_metadata_handling(func, tmpdir, epochs):
    """Test the presence of metadata is handled properly.

    Test both with the cases of having an array input and
    an ``mne.Epochs`` object input.
    """
    kwargs = dict()
    if isinstance(epochs, np.ndarray) and func in (
        spectral_connectivity_epochs,
        phase_slope_index,
    ):
        kwargs["sfreq"] = 5

    # for each function, check that Annotations were added to the metadata
    # and are handled correctly
    conn = func(epochs, verbose=False, **kwargs)
    metadata = conn.metadata

    if isinstance(epochs, BaseEpochs):
        # each metadata frame should have an Annotations column with n_epochs
        # number of rows
        assert "annot_onset" in metadata.columns
        assert "annot_duration" in metadata.columns
        assert "annot_description" in metadata.columns
        assert len(metadata) == len(epochs)

    # temporary conn save
    fname = os.path.join(tmpdir, "connectivity.nc")
    conn.save(fname)

    new_conn = read_connectivity(fname)
    # assert these two objects are the same
    assert_array_equal(conn.names, new_conn.names)
    assert conn.dims == new_conn.dims
    for key, val in conn.coords.items():
        assert_array_equal(val, new_conn.coords[key])
    assert_array_equal(conn.get_data(), new_conn.get_data())
    if isinstance(epochs, BaseEpochs):
        assert metadata.equals(new_conn.metadata)
    else:
        assert isinstance(new_conn.metadata, pd.DataFrame)
        assert metadata.empty


@pytest.mark.parametrize("indices", ["all", "lower", "upper"])
@pytest.mark.parametrize(
    ["output", "missing"],
    [["raveled", "raise"], ["dense", "raise"], ["dense", np.nan]],
)
def test_get_data_complex(indices, output, missing):
    """Test that get_data works properly with complex data."""
    n_nodes = 3
    data = np.ones((n_nodes * n_nodes), dtype=np.complex128)
    if indices == "lower":
        tril_inds = np.tril_indices(n_nodes, k=-1)
        data = data[np.ravel_multi_index(tril_inds, (n_nodes, n_nodes))]
    if indices == "upper":
        triu_inds = np.triu_indices(n_nodes, k=1)
        data = data[np.ravel_multi_index(triu_inds, (n_nodes, n_nodes))]

    # use known method so missing values can be filled for dense when default missing
    conn = Connectivity(data=data, indices=indices, n_nodes=n_nodes, method="coh")
    out_data = conn.get_data(output=output, missing=missing)
    assert np.iscomplexobj(out_data)
