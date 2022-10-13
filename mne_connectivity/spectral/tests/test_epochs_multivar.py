import numpy as np
from numpy.testing import (
    assert_allclose, 
    assert_array_almost_equal,
    assert_array_less
    )
import pytest

from mne_connectivity import (
    SpectralConnectivity, 
    spectral_connectivity_epochs
    )
from mne_connectivity.spectral.epochs import _CohEst, _get_n_epochs

from .test_spectral import create_test_dataset


@pytest.mark.parametrize('method', [
    'mic', 'mim', 'gc', 'net_gc', 'trgc', 'net_trgc',
    ['mic', 'mim', 'gc', 'net_gc', 'trgc', 'net_trgc']])
@pytest.mark.parametrize('mode', ['multitaper', 'fourier', 'cwt_morlet'])
def test_multivar_spectral_connectivity(method, mode):
    """Test frequency-domain multivariate connectivity methods."""
    sfreq = 50.
    n_signals = 3
    n_epochs = 8
    n_times = 256
    trans_bandwidth = 2.
    tmin = 0.
    tmax = (n_times - 1) / sfreq

    # 5Hz..15Hz
    fstart, fend = 5.0, 15.0
    data, times_data = create_test_dataset(
        sfreq, n_signals=n_signals, n_epochs=n_epochs, n_times=n_times,
        tmin=tmin, tmax=tmax,
        fstart=fstart, fend=fend, trans_bandwidth=trans_bandwidth)
    # First we test some invalid parameters:
    pytest.raises(ValueError, test_spectral_connectivity_epochs,
                  data, method='notamethod')
    pytest.raises(ValueError, test_spectral_connectivity_epochs, data,
                  mode='notamode')

    # test invalid fmin fmax settings
    pytest.raises(ValueError, test_spectral_connectivity_epochs, data, fmin=10,
                  fmax=10 + 0.5 * (sfreq / float(n_times)))
    pytest.raises(ValueError, test_spectral_connectivity_epochs,
                  data, fmin=10, fmax=5)
    pytest.raises(ValueError, test_spectral_connectivity_epochs, data, fmin=(0, 11),
                  fmax=(5, 10))
    pytest.raises(ValueError, test_spectral_connectivity_epochs, data, fmin=(11,),
                  fmax=(12, 15))
