# Authors: Giovanni Marraffini <giovanni.marraffini@gmail.com>
#
# License: BSD (3-clause)

import mne
import numpy as np
import pytest
from mne import EpochsArray, create_info
from numpy.testing import assert_allclose, assert_array_equal

from mne_connectivity import wsmi


def test_wsmi_input_validation_and_errors():
    """Test input validation and error handling."""
    sfreq = 100.0
    n_epochs, n_channels, n_times = 2, 3, 200
    rng = np.random.RandomState(42)
    data = rng.randn(n_epochs, n_channels, n_times)

    ch_names = ["Fz", "Cz", "Pz"]
    info = create_info(ch_names, sfreq=sfreq, ch_types="eeg")
    epochs = EpochsArray(data, info, tmin=0.0)

    # Set a standard montage
    montage = mne.channels.make_standard_montage("standard_1020")
    epochs.set_montage(montage, match_case=False)

    # Test invalid parameters
    with pytest.raises(ValueError, match="kernel.*must be > 1"):
        wsmi(epochs, kernel=0, tau=1)

    with pytest.raises(ValueError, match="tau.*must be > 0"):
        wsmi(epochs, kernel=3, tau=0)

    with pytest.raises(TypeError, match="anti_aliasing must be an instance of bool"):
        wsmi(epochs, kernel=3, tau=1, anti_aliasing="yes")

    with pytest.raises(TypeError, match="weighted must be an instance of bool"):
        wsmi(epochs, kernel=3, tau=1, weighted="yes")

    # Test single channel error
    single_ch_data = data[:, :1, :]
    single_ch_info = create_info(["A"], sfreq=sfreq, ch_types="eeg")
    single_ch_epochs = EpochsArray(single_ch_data, single_ch_info, tmin=0.0)

    with pytest.raises(ValueError, match="At least 2 channels are required"):
        wsmi(single_ch_epochs, kernel=3, tau=1)

    # Test insufficient samples error
    with pytest.raises(ValueError, match=r"but at least[\s\S]*are needed"):
        wsmi(epochs, kernel=5, tau=3, tmin=0.05, tmax=0.1)


def test_wsmi_known_coupling_patterns():
    """Test wSMI with known coupling patterns to validate core properties."""
    sfreq = 100.0
    n_epochs, n_times = 3, 250
    t = np.linspace(0, n_times / sfreq, n_times)

    # Create test data focusing on fundamental wSMI properties
    data = np.zeros((n_epochs, 3, n_times))
    for epoch in range(n_epochs):
        # Channel 0: deterministic base signal
        base = np.sin(2 * np.pi * 10 * t)
        data[epoch, 0, :] = base

        # Channel 1: identical copy (must give wSMI = 0)
        data[epoch, 1, :] = base

        # Channel 2: strongly nonlinear transformation
        # Use a clear nonlinear relationship that wSMI should detect
        data[epoch, 2, :] = np.tanh(2 * base) + 0.5 * np.sin(2 * np.pi * 15 * t)

    ch_names = ["base", "identical", "coupled"]
    info = create_info(ch_names, sfreq=sfreq, ch_types="eeg")
    epochs = EpochsArray(data, info, tmin=0.0)

    # Compute wSMI
    conn = wsmi(epochs, kernel=3, tau=1)
    conn_data = conn.get_data()

    # Basic validation
    assert conn_data.shape == (n_epochs, 3)  # 3 choose 2 = 3 connections
    assert np.all(np.isfinite(conn_data))

    # Average connectivity values
    avg_conn = np.mean(conn_data, axis=0)

    # Connection indices: (0,1)=0, (0,2)=1, (1,2)=2
    identical_wsmi = avg_conn[0]  # base-identical
    coupled_wsmi = avg_conn[1]  # base-coupled
    identical_coupled_wsmi = avg_conn[2]  # identical-coupled

    # Test fundamental wSMI properties:

    # 1. Identical signals MUST have zero wSMI (core requirement)
    assert identical_wsmi == 0.0, "Identical channels must have wSMI = 0"

    # 2. All wSMI values must be non-negative
    assert np.all(avg_conn >= 0), "All wSMI values must be non-negative"

    # 3. Coupled signals should have positive wSMI
    assert coupled_wsmi > 0, f"Coupled signals should have wSMI > 0: {coupled_wsmi:.4f}"

    # 4. Since channel 1 is identical to channel 0, its coupling to channel 2
    #    should be the same as channel 0's coupling to channel 2
    assert abs(coupled_wsmi - identical_coupled_wsmi) < 1e-10, (
        "wSMI should be identical for identical source channels"
    )

    # 5. Reasonable bounds (wSMI should not exceed theoretical maximum)
    assert np.all(avg_conn < 2.0), f"wSMI values too high: max={np.max(avg_conn):.3f}"

    # 6. Test reproducibility across epochs
    assert np.all(np.std(conn_data, axis=0) < 0.5), (
        "wSMI should be stable across epochs"
    )


def test_wsmi_tau_nonlinear_detection():
    """Test that higher tau values better detect nonlinear coupling."""
    sfreq = 100.0
    n_epochs, n_times = 5, 400
    t = np.linspace(0, n_times / sfreq, n_times)

    # Create test data with clear nonlinear coupling
    data = np.zeros((n_epochs, 3, n_times))

    for epoch in range(n_epochs):
        # Base signal
        base = np.sin(2 * np.pi * 8 * t) + 0.1 * np.random.RandomState(
            42 + epoch
        ).randn(n_times)
        data[epoch, 0, :] = base

        # Strong nonlinear coupling (quadratic relationship)
        data[epoch, 1, :] = base**2 + 0.3 * np.sin(2 * np.pi * 12 * t)

        # Independent signal
        data[epoch, 2, :] = np.sin(2 * np.pi * 20 * t) + 0.2 * np.random.RandomState(
            100 + epoch
        ).randn(n_times)

    ch_names = ["base", "nonlinear", "independent"]
    info = create_info(ch_names, sfreq=sfreq, ch_types="eeg")
    epochs = EpochsArray(data, info, tmin=0.0)

    # Test tau=1 vs tau=2
    conn_tau1 = wsmi(epochs, kernel=3, tau=1)
    conn_tau2 = wsmi(epochs, kernel=3, tau=2)

    data_tau1 = np.mean(conn_tau1.get_data(), axis=0)
    data_tau2 = np.mean(conn_tau2.get_data(), axis=0)

    # Connection indices: (0,1)=0, (0,2)=1, (1,2)=2
    nonlinear_tau1 = data_tau1[0]  # base-nonlinear with tau=1
    independent_tau1 = data_tau1[1]  # base-independent with tau=1
    nonlinear_tau2 = data_tau2[0]  # base-nonlinear with tau=2
    independent_tau2 = data_tau2[1]  # base-independent with tau=2

    # Test that tau=2 shows better discrimination than tau=1
    if independent_tau1 > 0 and independent_tau2 > 0:
        ratio_tau1 = nonlinear_tau1 / independent_tau1
        ratio_tau2 = nonlinear_tau2 / independent_tau2

        # tau=2 should show better discrimination (higher ratio)
        assert ratio_tau2 > ratio_tau1, (
            f"tau=2 should show better nonlinear discrimination: "
            f"tau=1 ratio={ratio_tau1:.2f}, tau=2 ratio={ratio_tau2:.2f}"
        )

        # tau=2 should show substantial improvement (at least 2x better)
        assert ratio_tau2 > 2 * ratio_tau1, (
            f"tau=2 should show substantial improvement over tau=1: "
            f"tau=2 ratio should be > 2x tau=1 ratio"
            f"tau=2 ratio / tau=1 ratio: {ratio_tau2 / ratio_tau1:.2f}"
        )

    # Both should detect some coupling
    assert nonlinear_tau1 > 0, "tau=1 should detect some nonlinear coupling"
    assert nonlinear_tau2 > 0, "tau=2 should detect some nonlinear coupling"


def test_wsmi_parameter_effects():
    """Test that different kernel and tau values produce different results."""
    sfreq = 100.0
    n_epochs, n_times = 2, 200
    t = np.linspace(0, n_times / sfreq, n_times)

    # Create structured data
    data = np.zeros((n_epochs, 3, n_times))
    for epoch in range(n_epochs):
        data[epoch, 0, :] = np.sin(2 * np.pi * 10 * t)
        data[epoch, 1, :] = np.sin(2 * np.pi * 10 * t + np.pi / 4)  # Phase shifted
        data[epoch, 2, :] = np.sin(2 * np.pi * 15 * t)  # Different frequency

    ch_names = ["A", "B", "C"]
    info = create_info(ch_names, sfreq=sfreq, ch_types="eeg")
    epochs = EpochsArray(data, info, tmin=0.0)

    # Test different parameter combinations
    conn_k3_t1 = wsmi(epochs, kernel=3, tau=1)
    conn_k4_t1 = wsmi(epochs, kernel=4, tau=1)
    conn_k3_t2 = wsmi(epochs, kernel=3, tau=2)

    # Results should be different for different parameters
    assert not np.array_equal(conn_k3_t1.get_data(), conn_k4_t1.get_data()), (
        "Different kernel should produce different results"
    )
    assert not np.array_equal(conn_k3_t1.get_data(), conn_k3_t2.get_data()), (
        "Different tau should produce different results"
    )

    # All should have same shape and finite values
    for conn in [conn_k3_t1, conn_k4_t1, conn_k3_t2]:
        assert conn.get_data().shape == (n_epochs, 3)  # 3 choose 2 = 3 connections
        assert np.all(np.isfinite(conn.get_data()))


def test_wsmi_weighted_vs_unweighted():
    """Test weighted parameter produces different results for wSMI vs SMI."""
    sfreq = 100.0
    n_epochs, n_times = 2, 200

    # Create simple structured data
    data = np.zeros((n_epochs, 3, n_times))
    t = np.linspace(0, n_times / sfreq, n_times)

    for epoch in range(n_epochs):
        data[epoch, 0, :] = np.sin(2 * np.pi * 10 * t)
        data[epoch, 1, :] = np.sin(2 * np.pi * 10 * t)  # Identical
        data[epoch, 2, :] = np.sin(2 * np.pi * 12 * t)  # Different

    ch_names = ["A", "B", "C"]
    info = create_info(ch_names, sfreq=sfreq, ch_types="eeg")
    epochs = EpochsArray(data, info, tmin=0.0)

    # Test wSMI vs SMI
    conn_wsmi = wsmi(epochs, kernel=3, tau=1, weighted=True)
    conn_smi = wsmi(epochs, kernel=3, tau=1, weighted=False)

    # Should have different method names
    assert conn_wsmi.method == "wSMI"
    assert conn_smi.method == "SMI"

    # Should produce different results due to weighting
    assert not np.array_equal(conn_wsmi.get_data(), conn_smi.get_data()), (
        "wSMI and SMI should produce different results"
    )

    # Both should have finite values
    assert np.all(np.isfinite(conn_wsmi.get_data()))
    assert np.all(np.isfinite(conn_smi.get_data()))


def test_wsmi_basic_functionality():
    """Test basic wSMI functionality and object properties."""
    sfreq = 100.0
    n_epochs, n_channels, n_times = 2, 4, 200

    # Create deterministic data
    t = np.linspace(0, n_times / sfreq, n_times)
    data = np.zeros((n_epochs, n_channels, n_times))

    for epoch in range(n_epochs):
        data[epoch, 0, :] = np.sin(2 * np.pi * 10 * t)  # 10 Hz
        data[epoch, 1, :] = np.sin(2 * np.pi * 12 * t)  # 12 Hz
        data[epoch, 2, :] = np.sin(2 * np.pi * 15 * t)  # 15 Hz
        data[epoch, 3, :] = np.sin(2 * np.pi * 8 * t)  # 8 Hz

    ch_names = ["Fz", "Cz", "Pz", "Oz"]
    info = create_info(ch_names, sfreq=sfreq, ch_types="eeg")
    epochs = EpochsArray(data, info, tmin=0.0)

    montage = mne.channels.make_standard_montage("standard_1020")
    epochs.set_montage(montage, on_missing="ignore")

    # Test basic computation
    conn = wsmi(epochs, kernel=3, tau=1)

    # Check basic properties
    assert conn.method == "wSMI"
    assert conn.n_nodes == 4
    assert conn.n_epochs_used == 2

    # Check data shape and validity
    data_matrix = conn.get_data()
    expected_connections = 4 * 3 // 2  # 4 choose 2 = 6 connections
    assert data_matrix.shape == (2, expected_connections)
    assert np.all(np.isfinite(data_matrix))

    # Test with different channel types (suppress unit change warning)
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        epochs.set_channel_types({"Fz": "eeg", "Cz": "eeg", "Pz": "mag", "Oz": "mag"})
        conn_mixed = wsmi(epochs, kernel=3, tau=1)
        assert conn_mixed.n_nodes == 4
        assert np.all(np.isfinite(conn_mixed.get_data()))


def test_wsmi_averaging_and_indices():
    """Test averaging and indices parameters."""
    sfreq = 100.0
    n_epochs, n_times = 4, 150
    data = np.zeros((n_epochs, 4, n_times))

    # Create simple test data
    for epoch in range(n_epochs):
        for ch in range(4):
            data[epoch, ch, :] = np.sin(
                2 * np.pi * (10 + ch) * np.linspace(0, n_times / sfreq, n_times)
            )

    ch_names = ["A", "B", "C", "D"]
    info = create_info(ch_names, sfreq=sfreq, ch_types="eeg")
    epochs = EpochsArray(data, info, tmin=0.0)

    # Test averaging
    conn_no_avg = wsmi(epochs, kernel=3, tau=1, average=False)
    conn_avg = wsmi(epochs, kernel=3, tau=1, average=True)

    # Check types and shapes
    from mne_connectivity.base import Connectivity, EpochConnectivity

    assert isinstance(conn_no_avg, EpochConnectivity)
    assert isinstance(conn_avg, Connectivity)

    expected_connections = 6  # 4 choose 2
    assert conn_no_avg.get_data().shape == (n_epochs, expected_connections)
    assert conn_avg.get_data().shape == (expected_connections,)

    # Averaged should equal manual average
    manual_avg = np.mean(conn_no_avg.get_data(), axis=0)
    assert_allclose(conn_avg.get_data(), manual_avg)

    # Test indices parameter
    indices = (np.array([0, 1]), np.array([2, 3]))  # A-C, B-D connections
    conn_indices = wsmi(epochs, kernel=3, tau=1, indices=indices)
    assert conn_indices.get_data().shape == (n_epochs, 2)

    # Test invalid indices
    with pytest.raises(ValueError, match="Index.*is out of range"):
        wsmi(epochs, kernel=3, tau=1, indices=(np.array([0]), np.array([5])))

    with pytest.raises(ValueError, match="Self-connectivity not supported"):
        wsmi(epochs, kernel=3, tau=1, indices=(np.array([0]), np.array([0])))


def test_wsmi_array_input():
    """Test wSMI with numpy array input."""
    sfreq = 100.0
    n_epochs, n_channels, n_times = 2, 3, 150
    data = np.zeros((n_epochs, n_channels, n_times))

    # Create simple test data
    t = np.linspace(0, n_times / sfreq, n_times)
    for epoch in range(n_epochs):
        for ch in range(n_channels):
            data[epoch, ch, :] = np.sin(2 * np.pi * (10 + ch * 2) * t)

    # Test array input
    conn_array = wsmi(data, kernel=3, tau=1, sfreq=sfreq)
    assert conn_array.get_data().shape == (n_epochs, 3)  # 3 choose 2
    assert np.all(np.isfinite(conn_array.get_data()))

    # Test with custom names
    names = ["X", "Y", "Z"]
    conn_named = wsmi(data, kernel=3, tau=1, sfreq=sfreq, names=names)
    assert conn_named.names == names
    assert_array_equal(conn_array.get_data(), conn_named.get_data())

    # Test errors
    with pytest.raises(ValueError, match="Sampling frequency \\(sfreq\\) is required"):
        wsmi(data, kernel=3, tau=1)

    with pytest.raises(ValueError, match="Array input must be 3D"):
        wsmi(data[0], kernel=3, tau=1, sfreq=sfreq)

    with pytest.raises(
        ValueError, match="Number of names .* must match number of channels"
    ):
        wsmi(data, kernel=3, tau=1, sfreq=sfreq, names=["X", "Y"])

    # Test equivalence with Epochs input
    ch_names = ["ch1", "ch2", "ch3"]
    info = create_info(ch_names, sfreq=sfreq, ch_types="eeg")
    epochs = EpochsArray(data, info, tmin=0.0)

    conn_epochs = wsmi(epochs, kernel=3, tau=1)
    conn_array_equiv = wsmi(data, kernel=3, tau=1, sfreq=sfreq, names=ch_names)

    assert_allclose(conn_epochs.get_data(), conn_array_equiv.get_data(), rtol=1e-10)


def test_wsmi_deterministic():
    """Test that wSMI produces deterministic results."""
    sfreq = 100.0
    n_epochs, n_channels, n_times = 2, 3, 150

    # Create identical datasets
    data = np.zeros((n_epochs, n_channels, n_times))
    t = np.linspace(0, n_times / sfreq, n_times)
    for epoch in range(n_epochs):
        for ch in range(n_channels):
            data[epoch, ch, :] = np.sin(2 * np.pi * (8 + ch * 3) * t)

    ch_names = ["A", "B", "C"]
    info = create_info(ch_names, sfreq=sfreq, ch_types="eeg")

    epochs1 = EpochsArray(data.copy(), info, tmin=0.0)
    epochs2 = EpochsArray(data.copy(), info, tmin=0.0)

    # Compute connectivity with identical parameters
    conn1 = wsmi(epochs1, kernel=3, tau=1)
    conn2 = wsmi(epochs2, kernel=3, tau=1)

    # Results should be identical
    assert_array_equal(conn1.get_data(), conn2.get_data())


def test_wsmi_anti_aliasing():
    """Test anti-aliasing parameter."""
    sfreq = 100.0
    n_epochs, n_times = 2, 200
    data = np.zeros((n_epochs, 3, n_times))

    t = np.linspace(0, n_times / sfreq, n_times)
    for epoch in range(n_epochs):
        data[epoch, 0, :] = np.sin(2 * np.pi * 10 * t)
        data[epoch, 1, :] = np.sin(2 * np.pi * 12 * t)
        data[epoch, 2, :] = np.sin(2 * np.pi * 15 * t)

    ch_names = ["A", "B", "C"]
    info = create_info(ch_names, sfreq=sfreq, ch_types="eeg")
    epochs = EpochsArray(data, info, tmin=0.0)

    # Test with anti-aliasing enabled
    conn_with_filter = wsmi(epochs, kernel=3, tau=1, anti_aliasing=True)

    # Test with anti-aliasing disabled (should produce warning)
    import warnings

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        conn_without_filter = wsmi(epochs, kernel=3, tau=1, anti_aliasing=False)
        assert len(w) > 0
        assert "Anti-aliasing disabled" in str(w[0].message)

    # Both should produce valid results
    assert np.all(np.isfinite(conn_with_filter.get_data()))
    assert np.all(np.isfinite(conn_without_filter.get_data()))


# Ground truth validation tests (simplified)
def _load_test_data():
    """Load the wsmi test data."""
    import os
    import pickle

    test_data_path = os.path.join(
        os.path.dirname(__file__), "data", "wsmi_test_data.pkl"
    )

    if not os.path.exists(test_data_path):
        raise FileNotFoundError(
            f"Required wSMI test data not found at {test_data_path}. "
            "Ground truth validation tests cannot run without this data."
        )

    with open(test_data_path, "rb") as f:
        test_data = pickle.load(f)

    return test_data


def test_wsmi_ground_truth_validation():
    """Test wSMI against ground truth data for regression testing."""
    test_data = _load_test_data()

    for test_case in test_data["tests"]:
        # Extract input parameters
        input_params = test_case["input_params"]

        # Run our implementation
        conn = wsmi(
            input_params["epochs"],
            kernel=input_params["kernel"],
            tau=input_params["tau"],
            tmin=input_params["tmin"],
            tmax=input_params["tmax"],
        )

        # Compare with expected output
        expected_data = test_case["expected_output"].get_data()
        new_data = conn.get_data()

        assert expected_data.shape == new_data.shape
        assert_allclose(expected_data, new_data, rtol=1e-10, atol=1e-10)
