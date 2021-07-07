import numpy as np
from numpy.testing import assert_array_almost_equal

from mne_connectivity.effective import phase_slope_index


def test_psi():
    """Test Phase Slope Index (PSI) estimation."""
    sfreq = 50.
    n_signals = 3
    n_epochs = 10
    n_times = 500
    rng = np.random.RandomState(42)
    data = rng.randn(n_epochs, n_signals, n_times)

    # simulate time shifts
    for i in range(n_epochs):
        data[i, 1, 10:] = data[i, 0, :-10]  # signal 0 is ahead
        data[i, 2, :-10] = data[i, 0, 10:]  # signal 2 is ahead

    conn = phase_slope_index(
        data, mode='fourier', sfreq=sfreq)

    assert conn.get_data(output='dense')[1, 0, 0] < 0
    assert conn.get_data(output='dense')[2, 0, 0] > 0

    # only compute for a subset of the indices
    indices = (np.array([0]), np.array([1]))
    conn_2 = phase_slope_index(
        data, mode='fourier', sfreq=sfreq, indices=indices)

    # the measure is symmetric (sign flip)
    assert_array_almost_equal(
        conn_2.get_data()[0, 0],
        -conn.get_data(output='dense')[1, 0, 0])

    cwt_freqs = np.arange(5., 20, 0.5)
    conn_cwt = phase_slope_index(
        data, mode='cwt_morlet', sfreq=sfreq, cwt_freqs=cwt_freqs,
        indices=indices)

    assert np.all(conn_cwt.get_data() > 0)
    assert conn_cwt.shape[-1] == n_times
