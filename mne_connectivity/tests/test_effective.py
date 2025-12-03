import numpy as np
from numpy.testing import assert_array_almost_equal

from mne_connectivity import EpochSpectralConnectivity, SpectralConnectivity
from mne_connectivity.effective import phase_slope_index, phase_slope_index_time


def test_psi():
    """Test Phase Slope Index (PSI) estimation."""
    sfreq = 50.0
    n_signals = 3
    n_epochs = 10
    n_times = 500
    rng = np.random.RandomState(42)
    data = rng.randn(n_epochs, n_signals, n_times)

    # simulate time shifts
    for i in range(n_epochs):
        data[i, 1, 10:] = data[i, 0, :-10]  # signal 0 is ahead
        data[i, 2, :-10] = data[i, 0, 10:]  # signal 2 is ahead

    conn = phase_slope_index(data, mode="fourier", sfreq=sfreq)

    assert conn.get_data(output="dense")[1, 0, 0] < 0
    assert conn.get_data(output="dense")[2, 0, 0] > 0

    # only compute for a subset of the indices
    indices = (np.array([0]), np.array([1]))
    conn_2 = phase_slope_index(data, mode="fourier", sfreq=sfreq, indices=indices)

    # the measure is symmetric (sign flip)
    assert_array_almost_equal(
        conn_2.get_data()[0, 0], -conn.get_data(output="dense")[1, 0, 0]
    )

    cwt_freqs = np.arange(5.0, 20, 0.5)
    conn_cwt = phase_slope_index(
        data, mode="cwt_morlet", sfreq=sfreq, cwt_freqs=cwt_freqs, indices=indices
    )

    assert np.all(conn_cwt.get_data() > 0)
    assert conn_cwt.shape[-1] == n_times


def test_psi_time():
    """Test Phase Slope Index (PSI) estimation across time."""
    sfreq = 50.0
    n_signals = 3
    n_epochs = 10
    n_times = 500
    rng = np.random.RandomState(42)
    data = rng.randn(n_epochs, n_signals, n_times)

    # simulate time shifts
    for i in range(n_epochs):
        data[i, 1, 10:] = data[i, 0, :-10]  # signal 0 is ahead
        data[i, 2, :-10] = data[i, 0, 10:]  # signal 2 is ahead

    # only compute for a subset of the indices
    indices = (np.array([0]), np.array([1]))

    freqs = np.arange(5.0, 20, 0.5)
    conn_cwt = phase_slope_index_time(
        data, mode="cwt_morlet", sfreq=sfreq, freqs=freqs, indices=indices
    )

    assert np.all(conn_cwt.get_data() > 0)
    assert conn_cwt.shape[0] == n_epochs
    assert isinstance(conn_cwt, EpochSpectralConnectivity)

    # Test with average=False (explicit)
    conn_no_avg = phase_slope_index_time(
        data, mode="cwt_morlet", sfreq=sfreq, freqs=freqs, indices=indices, average=False
    )
    assert isinstance(conn_no_avg, EpochSpectralConnectivity)
    assert conn_no_avg.shape[0] == n_epochs

    # Test with average=True
    conn_avg = phase_slope_index_time(
        data, mode="cwt_morlet", sfreq=sfreq, freqs=freqs, indices=indices, average=True
    )
    assert isinstance(conn_avg, SpectralConnectivity)
    # When averaged, epoch dimension should be removed
    assert len(conn_avg.shape) == 2  # (n_con, n_bands)
    assert conn_avg.shape[0] == len(indices[0])
    # Verify that averaged result matches manual average
    assert_array_almost_equal(conn_avg.get_data(), np.mean(conn_no_avg.get_data(), axis=0))

    # Test with single epoch (no epoch dimension in input)
    single_epoch_data = data[0:1]  # shape (1, n_signals, n_times)
    conn_single = phase_slope_index_time(
        single_epoch_data, mode="cwt_morlet", sfreq=sfreq, freqs=freqs, indices=indices
    )
    assert isinstance(conn_single, EpochSpectralConnectivity)
    assert conn_single.shape[0] == 1  # single epoch

    # Test with single epoch and average=True
    conn_single_avg = phase_slope_index_time(
        single_epoch_data, mode="cwt_morlet", sfreq=sfreq, freqs=freqs, indices=indices, average=True
    )
    assert isinstance(conn_single_avg, SpectralConnectivity)
    assert len(conn_single_avg.shape) == 2  # (n_con, n_bands)
