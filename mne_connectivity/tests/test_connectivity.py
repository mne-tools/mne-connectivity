# Authors: Adam Li <adam2392@gmail.com>
#
# License: BSD (3-clause)

import os
from mne.annotations import Annotations
from mne.epochs import BaseEpochs
from mne.io.meas_info import create_info

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal
from mne.io import RawArray
from mne import make_fixed_length_epochs

from mne_connectivity import (Connectivity, EpochConnectivity,
                              EpochSpectralConnectivity,
                              EpochSpectroTemporalConnectivity,
                              EpochTemporalConnectivity, SpectralConnectivity,
                              SpectroTemporalConnectivity,
                              TemporalConnectivity)
from mne_connectivity.effective import phase_slope_index
from mne_connectivity.io import read_connectivity
from mne_connectivity import envelope_correlation, vector_auto_regression
from mne_connectivity.spectral import spectral_connectivity_epochs


def _make_test_epochs():
    sfreq = 50.
    n_signals = 3
    n_epochs = 10
    n_times = 500
    rng = np.random.RandomState(42)
    data = rng.randn(n_signals, n_epochs * n_times)

    # create Epochs
    info = create_info(np.arange(n_signals).astype(str).tolist(), sfreq=sfreq,
                       ch_types='eeg')
    onset = [0, 0.5, 3]
    duration = [0, 0, 0]
    description = ['test1', 'test2', 'test3']
    annots = Annotations(onset=onset, duration=duration,
                         description=description)
    raw = RawArray(data, info)
    raw = raw.set_annotations(annots)
    epochs = make_fixed_length_epochs(raw, duration=1, preload=True)

    # make sure Epochs has metadata
    epochs.add_annotations_to_metadata()
    return epochs


def _prep_correct_connectivity_input(conn_cls, n_nodes=3, symmetric=False,
                                     n_epochs=4, indices=None):
    correct_numpy_shape = []

    extra_kwargs = dict()
    if conn_cls.is_epoched:
        correct_numpy_shape.append(n_epochs)

    if indices is None:
        if symmetric:
            correct_numpy_shape.append((n_nodes + 1) * n_nodes // 2)
        else:
            correct_numpy_shape.append(n_nodes**2)
    else:
        correct_numpy_shape.append(len(indices[0]))

    if conn_cls in (SpectralConnectivity, SpectroTemporalConnectivity,
                    EpochSpectralConnectivity,
                    EpochSpectroTemporalConnectivity):
        extra_kwargs['freqs'] = np.arange(4)
        correct_numpy_shape.append(4)
    if conn_cls in (TemporalConnectivity, SpectroTemporalConnectivity,
                    EpochTemporalConnectivity,
                    EpochSpectroTemporalConnectivity):
        extra_kwargs['times'] = np.arange(3)
        correct_numpy_shape.append(3)

    return correct_numpy_shape, extra_kwargs


@pytest.mark.parametrize(
    'conn_cls', [Connectivity, EpochConnectivity,
                 SpectralConnectivity,
                 TemporalConnectivity,
                 SpectroTemporalConnectivity,
                 EpochTemporalConnectivity,
                 EpochSpectralConnectivity,
                 EpochSpectroTemporalConnectivity],
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
    bad_numpy_input = np.zeros((3, 3, 4, 5))
    bad_indices = ([1, 0], [2])

    if conn_cls.is_epoched:
        bad_numpy_input = np.zeros((3, 3, 3, 4, 5))

    correct_numpy_shape, extra_kwargs = _prep_correct_connectivity_input(
        conn_cls, n_nodes=n_nodes, symmetric=False, n_epochs=n_epochs
    )

    correct_numpy_input = np.ones(correct_numpy_shape)

    # test initialization error checks
    with pytest.raises(TypeError, match='Connectivity data '
                       'must be passed in as a '
                       'numpy array'):
        conn_cls(data=data, n_nodes=2, **extra_kwargs)
    with pytest.raises(RuntimeError, match='Data*.'):
        conn_cls(data=bad_numpy_input, n_nodes=2, **extra_kwargs)
    with pytest.raises(ValueError, match='If indices are passed*.'):
        conn_cls(data=correct_numpy_input, indices=bad_indices,
                 n_nodes=2, **extra_kwargs)
    with pytest.raises(ValueError, match='Indices can only be*.'):
        conn_cls(data=correct_numpy_input, indices='square',
                 n_nodes=2, **extra_kwargs)

    indices = ([0, 1], [1, 0])
    conn = conn_cls(data=correct_numpy_input, n_nodes=3, **extra_kwargs)

    # test that get_data works as intended
    with pytest.raises(ValueError, match="Invalid value for the "
                                         "'output' parameter*."):
        conn.get_data(output='blah')

    assert conn.shape == tuple(correct_numpy_shape)
    assert conn.get_data(output='raveled').shape == tuple(correct_numpy_shape)
    assert conn.get_data(output='dense').ndim == len(correct_numpy_shape) + 1

    # test renaming nodes error checks
    with pytest.raises(ValueError, match="Name*."):
        conn.rename_nodes({'100': 'new_name'})
    with pytest.raises(ValueError, match="mapping must be*"):
        conn.rename_nodes(['0', 'new_name'])
    with pytest.raises(ValueError, match="New channel names*"):
        conn.rename_nodes({'0': '1'})

    # test renaming nodes
    orig_names = conn.names
    conn.rename_nodes({'0': 'new_name'})
    new_names = conn.names
    assert all([name_1 == name_2 for name_1, name_2 in
                zip(orig_names, new_names)
                if name_2 != 'new_name'])
    conn.rename_nodes(lambda x: '0' if x == 'new_name' else x)
    assert_array_equal(orig_names, conn.names)

    # test connectivity instantiation with indices
    indexed_numpy_shape, index_kwargs = _prep_correct_connectivity_input(
        conn_cls, n_nodes=n_nodes, symmetric=False, n_epochs=n_epochs,
        indices=indices
    )
    indexed_numpy_input = np.ones(indexed_numpy_shape)
    conn2 = conn_cls(data=indexed_numpy_input, n_nodes=2, indices=indices,
                     **index_kwargs)
    conn3 = conn_cls(data=indexed_numpy_input, n_nodes=3, indices=indices,
                     **index_kwargs)

    # the number of nodes helps define the full dense output, but
    # if unraveled, with indices then they should match exactly
    assert_array_equal(
        conn2.get_data(), conn3.get_data())

    # test getting data with indices specified
    with pytest.raises(ValueError, match='The number of indices'):
        conn_cls(data=correct_numpy_input, n_nodes=3, indices=indices,
                 **extra_kwargs)

    # test symmetric input
    correct_numpy_shape, extra_kwargs = _prep_correct_connectivity_input(
        conn_cls, n_nodes=3, symmetric=True
    )
    correct_numpy_input = np.ones(correct_numpy_shape)

    with pytest.raises(ValueError, match='If "indices" is "symmetric"'):
        conn_cls(data=correct_numpy_input, n_nodes=2,
                 indices='symmetric',
                 **extra_kwargs)
    symm_conn = conn_cls(data=correct_numpy_input, n_nodes=n_nodes,
                         indices='symmetric',
                         **extra_kwargs)
    assert symm_conn.n_nodes == n_nodes

    # raveled shape should be the same
    assert_array_equal(symm_conn.get_data(output='raveled').shape,
                       correct_numpy_shape)

    # should be ([n_epochs], n_nodes, n_nodes, ...) dense shape
    dense_shape = []
    if conn_cls.is_epoched:
        dense_shape.append(n_epochs)
    dense_shape.extend([n_nodes, n_nodes])
    assert all([symm_conn.get_data(
        output='dense').shape[idx] == dense_shape[idx]
        for idx in range(len(dense_shape))])


@pytest.mark.parametrize(
    'conn_cls', [Connectivity, EpochConnectivity,
                 SpectralConnectivity,
                 TemporalConnectivity,
                 SpectroTemporalConnectivity,
                 EpochTemporalConnectivity,
                 EpochSpectralConnectivity,
                 EpochSpectroTemporalConnectivity],
)
def test_io(conn_cls, tmpdir):
    """Test writing and reading connectivity data."""
    correct_numpy_shape = []
    extra_kwargs = dict()
    if conn_cls.is_epoched:
        correct_numpy_shape.append(4)
    correct_numpy_shape.append(4)
    if conn_cls in (SpectralConnectivity, SpectroTemporalConnectivity,
                    EpochSpectralConnectivity,
                    EpochSpectroTemporalConnectivity):
        extra_kwargs['freqs'] = np.arange(4)
        correct_numpy_shape.append(4)
    if conn_cls in (TemporalConnectivity, SpectroTemporalConnectivity,
                    EpochTemporalConnectivity,
                    EpochSpectroTemporalConnectivity):
        extra_kwargs['times'] = np.arange(3)
        correct_numpy_shape.append(3)

    correct_numpy_input = np.ones(correct_numpy_shape)

    # create the connectivity data structure
    conn = conn_cls(data=correct_numpy_input, n_nodes=2, **extra_kwargs)

    # temporary conn save
    fname = os.path.join(tmpdir, 'connectivity.nc')
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
    'conn_cls', [EpochConnectivity,
                 EpochTemporalConnectivity,
                 EpochSpectralConnectivity,
                 EpochSpectroTemporalConnectivity],
)
def test_append(conn_cls):
    """Test appending connectivity data."""
    correct_numpy_shape = []
    extra_kwargs = dict()
    if conn_cls.is_epoched:
        correct_numpy_shape.append(4)
    correct_numpy_shape.append(4)
    if conn_cls in (SpectralConnectivity, SpectroTemporalConnectivity,
                    EpochSpectralConnectivity,
                    EpochSpectroTemporalConnectivity):
        extra_kwargs['freqs'] = np.arange(4)
        correct_numpy_shape.append(4)
    if conn_cls in (TemporalConnectivity, SpectroTemporalConnectivity,
                    EpochTemporalConnectivity,
                    EpochSpectroTemporalConnectivity):
        extra_kwargs['times'] = np.arange(50)
        correct_numpy_shape.append(50)

    correct_numpy_input = np.ones(correct_numpy_shape)
    events = np.zeros((correct_numpy_input.shape[0], 3), dtype=int)
    events[:, -1] = 1  # event ID
    events[:, 0] = np.linspace(0, 50, len(events))

    # create the connectivity data structure
    conn = conn_cls(data=correct_numpy_input, n_nodes=2, events=events,
                    **extra_kwargs)

    # create a copy of the connectivity
    conn_2 = conn.copy()

    # append epochs
    conn.append(conn_2)
    assert conn.n_epochs == conn_2.n_epochs * 2
    assert len(conn.events) == conn.n_epochs


@pytest.mark.parametrize(
    'conn_func',
    [vector_auto_regression, spectral_connectivity_epochs,
     envelope_correlation, phase_slope_index]
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
    'epochs', [
        _make_test_epochs(),
        np.random.RandomState(0).random((10, 3, 500))
    ])
@pytest.mark.parametrize(
    'func', [
        vector_auto_regression,
        spectral_connectivity_epochs,
        envelope_correlation,
        phase_slope_index
    ])
def test_metadata_handling(func, tmpdir, epochs):
    """Test the presence of metadata is handled properly.

    Test both with the cases of having an array input and
    an ``mne.Epochs`` object input.
    """
    kwargs = dict()
    if isinstance(epochs, np.ndarray) and func == spectral_connectivity_epochs:
        kwargs['sfreq'] = 5

    # for each function, check that Annotations were added to the metadata
    # and are handled correctly
    conn = func(epochs, verbose=False, **kwargs)
    metadata = conn.metadata

    if isinstance(epochs, BaseEpochs):
        # each metadata frame should have an Annotations column with n_epochs
        # number of rows
        assert 'annot_onset' in metadata.columns
        assert 'annot_duration' in metadata.columns
        assert 'annot_description' in metadata.columns
        assert len(metadata) == len(epochs)

    # temporary conn save
    fname = os.path.join(tmpdir, 'connectivity.nc')
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
