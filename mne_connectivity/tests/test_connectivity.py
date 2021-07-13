# Authors: Adam Li <adam2392@gmail.com>
#
# License: BSD (3-clause)

import os

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from mne_connectivity import (Connectivity, EpochConnectivity,
                              EpochSpectralConnectivity,
                              EpochSpectroTemporalConnectivity,
                              EpochTemporalConnectivity, SpectralConnectivity,
                              SpectroTemporalConnectivity,
                              TemporalConnectivity)
from mne_connectivity.io import read_connectivity


def _prep_correct_connectivity_input(conn_cls, n_nodes=3, symmetric=False,
                                     n_epochs=4):
    correct_numpy_shape = []

    extra_kwargs = dict()
    if conn_cls.is_epoched:
        correct_numpy_shape.append(n_epochs)

    if symmetric:
        correct_numpy_shape.append((n_nodes + 1) * n_nodes // 2)
    else:
        correct_numpy_shape.append(n_nodes**2)

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
    assert conn.get_data().shape == tuple(correct_numpy_shape)
    assert conn.get_data(output='dense').ndim == len(correct_numpy_shape) + 1
    orig_names = conn.names
    conn.rename_nodes({'0': 'new_name'})
    new_names = conn.names
    assert all([name_1 == name_2 for name_1, name_2 in
                zip(orig_names, new_names)
                if name_2 != 'new_name'])

    conn2 = conn_cls(data=correct_numpy_input, n_nodes=3, indices=indices,
                     **extra_kwargs)
    conn3 = conn_cls(data=correct_numpy_input, n_nodes=3, indices=indices,
                     **extra_kwargs)
    np.testing.assert_array_equal(conn2.get_data(), conn3.get_data())

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
    assert_array_equal(symm_conn.get_data().shape, correct_numpy_shape)

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
    assert conn.names == new_conn.names
    assert conn.dims == new_conn.dims
    for key, val in conn.coords.items():
        assert_array_equal(val, new_conn.coords[key])
    assert_array_equal(conn.get_data(), new_conn.get_data())
