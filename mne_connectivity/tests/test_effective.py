import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal

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


def test_psi_time_properties():
    """Test Phase Slope Index (PSI) estimation over time result properties."""
    sfreq = 50.0
    n_epochs, n_signals, n_times = 10, 3, 500
    rng = np.random.default_rng(42)
    data = rng.standard_normal((n_epochs, n_signals, n_times))

    # simulate time shifts
    for i in range(n_epochs):
        data[i, 1, 10:] = data[i, 0, :-10]  # signal 0 is ahead
        data[i, 2, :-10] = data[i, 0, 10:]  # signal 2 is ahead

    freqs = np.arange(5.0, 20, 0.5)
    indices = (np.array([0, 0, 1, 2]), np.array([1, 2, 0, 0]))
    conn = phase_slope_index_time(
        data, sfreq=sfreq, freqs=freqs, indices=indices, average=True
    )

    # Check directionality captured
    assert conn.get_data()[0] > 0  # signal 0 leads signal 1
    assert conn.get_data()[1] < 0  # signal 0 follows signal 2

    # Check measure is symmetric (sign flip)
    assert_array_almost_equal(conn.get_data()[:2], -conn.get_data()[2:])


@pytest.mark.parametrize("average", [True, False])
@pytest.mark.parametrize("n_epochs", [1, 5])
def test_psi_time_epoch_averaging(average, n_epochs):
    """Test PSI time with and without epoch averaging."""
    sfreq = 50.0
    n_signals, n_times = 2, 500
    rng = np.random.default_rng(42)
    data = rng.standard_normal((n_epochs, n_signals, n_times))

    freqs = np.arange(5.0, 20, 0.5)
    indices = (np.array([0]), np.array([1]))
    conn = phase_slope_index_time(
        data, sfreq=sfreq, freqs=freqs, indices=indices, average=average
    )
    expected_shape = (len(indices[0]), 1)  # (n_con, n_bands)

    # Check container type
    if average:
        expected_class = SpectralConnectivity
    else:
        expected_class = EpochSpectralConnectivity
        expected_shape = (n_epochs,) + expected_shape  # (n_epochs, n_con, n_bands)

    assert isinstance(conn, expected_class)
    assert conn.shape == expected_shape

    # Check averaging applied correctly (i.e., to PSI, not preceding coherency)
    if average and n_epochs > 1:
        conn_no_avg = phase_slope_index_time(
            data, sfreq=sfreq, freqs=freqs, indices=indices, average=False
        )
        assert_array_equal(conn.get_data(), conn_no_avg.get_data().mean(axis=0))
