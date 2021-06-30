# Authors: Adam Li <adam2392@gmail.com>
#
# License: BSD (3-clause)

import pytest

from mne_connectivity import (
    SpectralConnectivity, TemporalConnectivity,
    SpectroTemporalConnectivity, EpochTemporalConnectivity,
    EpochSpectralConnectivity, EpochSpectroTemporalConnectivity)


@pytest.mark.skip()
@pytest.mark.parametrize(
    'conn_cls', [SpectralConnectivity, TemporalConnectivity,
                 SpectroTemporalConnectivity, EpochTemporalConnectivity,
                 EpochSpectralConnectivity,
                 EpochSpectroTemporalConnectivity])
def test_connectivity_containers(conn_cls):
    data = [
        [1, 0, 0],
        [3, 4, 5],
        [0, 1, 2],
    ]

    with pytest.raises(TypeError, match='Connectivity data*.numpy array'):
        conn_cls(data=data)
