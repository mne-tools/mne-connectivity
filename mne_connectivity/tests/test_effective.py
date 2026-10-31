import numpy as np
import pytest
from numpy.testing import (
    assert_array_almost_equal,
    assert_array_equal,
    assert_array_less,
)

from mne_connectivity import (
    EpochSpectralConnectivity,
    SpectralConnectivity,
    make_signals_in_freq_bands,
)
from mne_connectivity.effective import phase_slope_index, phase_slope_index_time


@pytest.mark.parametrize("psi_func", [phase_slope_index, phase_slope_index_time])
def test_psi_properties(psi_func):
    """Test result properties of Phase Slope Index (PSI) estimation over epochs/time."""
    # Simulate signals interacting in two distinct frequency bands
    fmin = (10, 25)
    fmax = (15, 30)
    data_10_15 = make_signals_in_freq_bands(
        n_seeds=1,
        n_targets=1,
        freq_band=(fmin[0], fmax[0]),
        duration=2.0,
        connection_delay=0.05,
        rng_seed=42,
    )
    data_25_30 = make_signals_in_freq_bands(
        n_seeds=1,
        n_targets=1,
        freq_band=(fmin[1], fmax[1]),
        duration=2.0,
        connection_delay=0.1,
        rng_seed=44,
    )
    data = data_10_15.add_channels([data_25_30])

    # Compute PSI between signals in each frequency band
    freqs = np.arange(3, 33, 1)
    indices = (np.array([0, 1, 2, 3]), np.array([1, 0, 3, 2]))
    func_kwargs = {}
    if psi_func is phase_slope_index_time:
        freqs = np.arange(3, 33, 1)
        func_kwargs["freqs"] = freqs
        func_kwargs["n_cycles"] = freqs / 1.25
        func_kwargs["average"] = True
    conn = psi_func(data, indices=indices, fmin=fmin, fmax=fmax, **func_kwargs)
    conn_data = conn.get_data()

    # Check directionality captured
    CON_THRESH = 0.4
    NOISE_THRESH = 0.1
    # signals interacting at 10-15 Hz
    assert_array_less(CON_THRESH, conn_data[0, 0])  # 10-15 Hz; seed leads target
    assert_array_less(conn_data[1, 0], -CON_THRESH)  # 10-15 Hz; target follows seed
    assert_array_less(np.abs(conn_data[:2, 1]), NOISE_THRESH)  # no 25-30 Hz interaction
    # signals interacting at 25-30 Hz
    assert_array_less(CON_THRESH, conn_data[2, 1])  # 25-30 Hz; seed leads target
    assert_array_less(conn_data[3, 1], -CON_THRESH)  # 25-30 Hz; target follows seed
    assert_array_less(np.abs(conn_data[2:, 0]), NOISE_THRESH)  # no 10-15 Hz interaction
    # Check measure is symmetric (sign flip)
    assert_array_almost_equal(conn_data[(0, 2), :], -conn_data[(1, 3), :])


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
