# Authors: Adam Li <adam2392@gmail.com>
#
# License: BSD (3-clause)

import pytest

import numpy as np

from mne_connectivity import (
    SpectralConnectivity, TemporalConnectivity,
    SpectroTemporalConnectivity, EpochTemporalConnectivity,
    EpochSpectralConnectivity, EpochSpectroTemporalConnectivity)


@pytest.mark.parametrize(
    'conn_cls', [SpectralConnectivity,
                 TemporalConnectivity,
                 SpectroTemporalConnectivity,
                 EpochTemporalConnectivity,
                 EpochSpectralConnectivity,
                 EpochSpectroTemporalConnectivity],
)
def test_connectivity_containers(conn_cls):
    data = [
        [1, 0, 0],
        [3, 4, 5],
        [0, 1, 2],
    ]
    bad_numpy_input = np.zeros((3, 3, 4, 5))
    correct_numpy_shape = []
    bad_indices = ([1, 0], [2])

    extra_kwargs = dict()
    if conn_cls.is_epoched:
        correct_numpy_shape.append(4)
        bad_numpy_input = np.zeros((3, 3, 3, 4, 5))
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

    indices = ([0, 1], [1, 0])
    conn = conn_cls(data=correct_numpy_input, n_nodes=2, **extra_kwargs)
    assert conn.shape == tuple(correct_numpy_shape)
    assert conn.get_data().shape == tuple(correct_numpy_shape)
    assert conn.get_data(output='full').ndim == len(correct_numpy_shape) + 1
    orig_names = conn.names
    conn.rename_nodes({'0': 'new_name'})
    new_names = conn.names
    assert all([name_1 == name_2 for name_1, name_2 in
                zip(orig_names, new_names)
                if name_2 != 'new_name'])

    conn2 = conn_cls(data=correct_numpy_input, n_nodes=2, indices=indices,
                     **extra_kwargs)
    conn3 = conn_cls(data=correct_numpy_input, n_nodes=3, indices=indices,
                     **extra_kwargs)
    np.testing.assert_array_equal(conn2.get_data(), conn3.get_data())
