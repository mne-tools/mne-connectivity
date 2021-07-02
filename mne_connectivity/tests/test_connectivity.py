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
    correct_numpy_input = np.ones((4, 4, 3))
    bad_indices = ([1, 0], [2])

    extra_kwargs = dict()
    if conn_cls in (TemporalConnectivity, SpectroTemporalConnectivity,
                    EpochTemporalConnectivity,
                    EpochSpectroTemporalConnectivity):
        extra_kwargs['times'] = np.arange(3)
    if conn_cls in (SpectralConnectivity, SpectroTemporalConnectivity,
                    EpochSpectralConnectivity,
                    EpochSpectroTemporalConnectivity):
        extra_kwargs['freqs'] = np.arange(4)
    if conn_cls.is_epoched:
        correct_numpy_input = np.ones((4, 4, 4, 3))
        bad_numpy_input = np.zeros((3, 3, 3, 4, 5))

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
