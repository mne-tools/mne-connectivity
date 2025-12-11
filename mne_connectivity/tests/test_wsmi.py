# Authors: Giovanni Marraffini <giovanni.marraffini@gmail.com>
#
# License: BSD (3-clause)

import os
import pickle

import numpy as np
import pytest
from mne import EpochsArray, create_info
from numpy.testing import assert_allclose, assert_array_equal, assert_array_less

from mne_connectivity import wsmi


def test_wsmi_input_output_validation():
    """Test input/output validation and error handling."""
    sfreq = 100.0
    n_epochs, n_channels, n_times = 2, 3, 200
    rng = np.random.default_rng(42)
    data = rng.standard_normal((n_epochs, n_channels, n_times))
    indices = np.tril_indices(n_channels, -1)

    info = create_info(n_channels, sfreq=sfreq, ch_types="eeg")
    epochs = EpochsArray(data, info, tmin=0.0)

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

    # Test invalid indices
    with pytest.raises(ValueError, match="Index.*is out of range"):
        wsmi(epochs, kernel=3, tau=1, indices=(np.array([0]), np.array([5])))

    with pytest.raises(ValueError, match="Self-connectivity not supported"):
        wsmi(epochs, kernel=3, tau=1, indices=(np.array([0]), np.array([0])))

    # Test with different channel types
    with pytest.warns(RuntimeWarning, match="The unit for channel"):
        epochs.set_channel_types({"0": "eeg", "1": "mag", "2": "grad"})
        conn_mixed = wsmi(epochs, kernel=3, tau=1)
        assert conn_mixed.n_nodes == 3
        assert np.all(np.isfinite(conn_mixed.get_data()))

    # Test with array data input
    with pytest.raises(ValueError, match="Sampling frequency \\(sfreq\\) is required"):
        wsmi(data, kernel=3, tau=1)
    with pytest.raises(ValueError, match="Array input must be 3D"):
        wsmi(data[0], kernel=3, tau=1, sfreq=sfreq)
    with pytest.raises(
        ValueError, match="Number of names .* must match number of channels"
    ):
        wsmi(data, kernel=3, tau=1, sfreq=sfreq, names=["X", "Y"])

    # Check output properties
    # Check averaging works
    conn = wsmi(epochs, kernel=3, tau=1, indices=indices, average=False)
    assert conn.get_data().shape == (n_epochs, len(indices[0]))
    conn_avg = wsmi(epochs, kernel=3, tau=1, indices=indices, average=True)
    assert conn_avg.get_data().shape == (len(indices[0]),)

    # Check equivalence of array and Epochs data inputs
    conn_epochs = wsmi(epochs, kernel=3, tau=1)
    conn_array = wsmi(data, kernel=3, tau=1, sfreq=sfreq)
    assert_array_equal(conn_epochs.get_data(), conn_array.get_data())


def test_wsmi_known_coupling_patterns():
    """Test wSMI with known coupling patterns to validate core properties."""
    sfreq = 100.0
    n_epochs, n_channels, n_times = 3, 3, 250
    t = np.linspace(0, n_times / sfreq, n_times)

    # Create test data focusing on fundamental wSMI properties
    data = np.zeros((1, n_channels, n_times))
    base_signal = np.sin(2 * np.pi * 10 * t)
    nonlinear_signal = np.tanh(2 * base_signal) + 0.5 * np.sin(2 * np.pi * 15 * t)
    data[:, 0] = base_signal  # Channel 0: base signal
    data[:, 1] = base_signal  # Channel 1: identical copy
    data[:, 2] = nonlinear_signal  # Channel 2: nonlinear transformation
    data = np.repeat(data, n_epochs, axis=0)

    ch_names = ["base", "identical", "coupled"]
    info = create_info(ch_names, sfreq=sfreq, ch_types="eeg")
    epochs = EpochsArray(data, info, tmin=0.0)
    indices = (np.array([0, 0, 1]), np.array([1, 2, 2]))

    # Compute wSMI
    conn = wsmi(epochs, kernel=3, tau=1, indices=indices)
    conn_data = conn.get_data()

    # Average connectivity values
    avg_conn = np.mean(conn_data, axis=0)

    # Connection indices: (0,1)=0, (0,2)=1, (1,2)=2
    identical_wsmi = avg_conn[0]  # base-identical
    coupled_wsmi = avg_conn[1]  # base-coupled
    identical_coupled_wsmi = avg_conn[2]  # identical-coupled

    # Test fundamental wSMI properties:

    # 1. Identical signals MUST have zero wSMI (core requirement)
    assert identical_wsmi == 0.0, "Identical channels must have wSMI = 0"

    # 2. All wSMI values in this example must be non-negative
    assert np.all(avg_conn >= 0), (
        "All wSMI values with correlated data should be non-negative"
    )

    # 3. Coupled signals should have positive wSMI
    assert coupled_wsmi > 0, f"Coupled signals should have wSMI > 0: {coupled_wsmi:.4f}"

    # 4. Since channel 1 is identical to channel 0, its coupling to channel 2
    #    should be exactly the same as channel 0's coupling to channel 2
    assert_array_equal(
        coupled_wsmi,
        identical_coupled_wsmi,
        "wSMI should be identical for identical source channels",
    )

    # 5. Reasonable bounds (wSMI should not exceed theoretical maximum)
    assert np.all(avg_conn < 2.0), f"wSMI values too high: max={np.max(avg_conn):.3f}"

    # 6. Test reproducibility across epochs
    std_conn = np.std(conn_data, axis=0)
    # Connection 0 (base-identical) should be exactly 0 (no variance)
    assert std_conn[0] == 0, "Identical channels must have zero variance"
    # Connections 1 and 2 should have low variance (Different in epochs just coupled)
    assert std_conn[1] < 0.01, "Coupled channels should have low variance"
    assert std_conn[2] < 0.01, "Coupled channels should have low variance"
    # Connections 1 and 2 should have identical variance (same signal coupling)
    assert std_conn[1] == std_conn[2], (
        "Identical source channels should have identical variance patterns"
    )


def test_wsmi_tau_nonlinear_detection():
    """Test that higher tau values better detect nonlinear coupling."""
    sfreq = 100.0
    n_epochs, n_channels, n_times = 5, 3, 400
    t = np.linspace(0, n_times / sfreq, n_times)

    # Create test data with clear nonlinear coupling
    rng_base = np.random.default_rng(42)
    rng_indep = np.random.default_rng(100)

    data = np.zeros((n_epochs, n_channels, n_times))
    # Base signal with different noise per epoch
    data[:, 0, :] = np.sin(2 * np.pi * 8 * t) + 0.1 * rng_base.standard_normal(
        (n_epochs, n_times)
    )
    # Strong nonlinear coupling (quadratic relationship)
    data[:, 1, :] = data[:, 0, :] ** 2 + 0.3 * np.sin(2 * np.pi * 12 * t)
    # Independent signal
    data[:, 2, :] = np.sin(2 * np.pi * 20 * t) + 0.2 * rng_indep.standard_normal(
        (n_epochs, n_times)
    )

    ch_names = ["base", "nonlinear", "independent"]
    info = create_info(ch_names, sfreq=sfreq, ch_types="eeg")
    epochs = EpochsArray(data, info, tmin=0.0)
    indices = (np.array([0, 0]), np.array([1, 2]))

    # Test tau=1 vs tau=2
    data_tau1 = wsmi(epochs, kernel=3, tau=1, indices=indices, average=True).get_data()
    data_tau2 = wsmi(epochs, kernel=3, tau=2, indices=indices, average=True).get_data()

    # Both tau values should detect coupling
    assert_array_less(0, [data_tau1, data_tau2])

    # Connection indices: (0,1)=0, (0,2)=1
    nonlinear_tau1 = data_tau1[0]  # base-nonlinear
    independent_tau1 = data_tau1[1]  # base-independent
    nonlinear_tau2 = data_tau2[0]  # base-nonlinear
    independent_tau2 = data_tau2[1]  # base-independent

    # tau=2 should show better discrimination (at least 2x better ratio)
    ratio_tau1 = nonlinear_tau1 / independent_tau1
    ratio_tau2 = nonlinear_tau2 / independent_tau2
    assert ratio_tau2 > 2 * ratio_tau1, (
        f"tau=2 discrimination ratio ({ratio_tau2:.2f}) should be > 2x "
        f"tau=1 ratio ({ratio_tau1:.2f})"
    )


def test_wsmi_kernel_phaseshift_detection():
    """Test larger kernel detects stronger coupling for phase-shifted signals."""
    sfreq = 100.0
    n_epochs, n_channels, n_times = 2, 2, 200
    t = np.linspace(0, n_times / sfreq, n_times)

    # Create structured data with phase-shifted coupling
    data = np.zeros((1, n_channels, n_times))
    data[:, 0] = np.sin(2 * np.pi * 10 * t)  # Base signal
    data[:, 1] = np.sin(2 * np.pi * 10 * t + np.pi / 4)  # Phase shifted
    data = np.repeat(data, n_epochs, axis=0)

    info = create_info(n_channels, sfreq=sfreq, ch_types="eeg")
    epochs = EpochsArray(data, info, tmin=0.0)
    indices = (np.array([0]), np.array([1]))

    # Test different kernel sizes
    data_k3 = wsmi(epochs, kernel=3, tau=1, indices=indices, average=True).get_data()
    data_k5 = wsmi(epochs, kernel=5, tau=1, indices=indices, average=True).get_data()

    # Larger kernel should detect stronger coupling for phase-shifted signals
    # Connection 0: A-B (phase-shifted, same frequency)
    # Larger kernel captures more temporal structure
    assert data_k5[0] > data_k3[0], (
        f"Larger kernel should capture more phase coupling: "
        f"k=5 ({data_k5[0]:.3f}) > k=3 ({data_k3[0]:.3f})"
    )


def test_wsmi_weighted_vs_unweighted():
    """Test wSMI filters identical patterns while SMI does not."""
    sfreq = 100.0
    n_epochs, n_channels, n_times = 2, 3, 200

    # Create test data with identical and different channels
    data = np.zeros((1, n_channels, n_times))
    t = np.linspace(0, n_times / sfreq, n_times)
    data[:, 0] = np.sin(2 * np.pi * 10 * t)  # 10 Hz signal
    data[:, 1] = data[:, 0]  # Identical 10 Hz signal
    data[:, 2] = np.sin(2 * np.pi * 12 * t)  # 12 Hz signal
    data = np.repeat(data, n_epochs, axis=0)

    ch_names = ["10Hz_1", "10Hz_2", "12Hz"]
    info = create_info(ch_names, sfreq=sfreq, ch_types="eeg")
    epochs = EpochsArray(data, info, tmin=0.0)
    indices = (np.array([0, 0]), np.array([1, 2]))

    # Test wSMI vs SMI
    conn_wsmi = wsmi(
        epochs, kernel=3, tau=1, weighted=True, average=True, indices=indices
    )
    conn_smi = wsmi(
        epochs, kernel=3, tau=1, weighted=False, average=True, indices=indices
    )
    assert conn_wsmi.method == "wSMI"
    assert conn_smi.method == "SMI"

    wsmi_data = conn_wsmi.get_data()
    smi_data = conn_smi.get_data()
    assert np.all(np.isfinite(wsmi_data))
    assert np.all(np.isfinite(smi_data))

    # For identical channels (10Hz-10Hz), wSMI should be exactly 0
    assert wsmi_data[0] == 0, "wSMI must be 0 for identical channels"
    # SMI should detect coupling for identical channels
    assert smi_data[0] > 0, "SMI should detect coupling for identical channels"

    # For different channels (10Hz-12Hz), wSMI should be smaller than SMI
    # (wSMI filters out some patterns that SMI includes)
    assert wsmi_data[1] < smi_data[1], (
        f"wSMI ({wsmi_data[1]:.3f}) should be < SMI ({smi_data[1]:.3f}) "
        "due to pattern filtering"
    )


def test_wsmi_anti_aliasing():
    """Test anti-aliasing parameter."""
    sfreq = 100.0
    signal_freqs = [10, 12, 15]  # Hz
    n_epochs, n_channels, n_times = 2, len(signal_freqs), 200

    t = np.linspace(0, n_times / sfreq, n_times)
    data = np.zeros((1, n_channels, n_times))
    for ch_idx, freq in enumerate(signal_freqs):
        data[0, ch_idx, :] = np.sin(2 * np.pi * freq * t)  # Sine wave at `freq` Hz
    data = np.repeat(data, n_epochs, axis=0)

    info = create_info(n_channels, sfreq=sfreq, ch_types="eeg")
    epochs = EpochsArray(data, info, tmin=0.0)

    # Test with anti-aliasing enabled
    conn_with_filter = wsmi(epochs, kernel=3, tau=1, anti_aliasing=True)

    # Test with anti-aliasing disabled (should produce warning)
    with pytest.warns(UserWarning, match="Anti-aliasing disabled"):
        conn_without_filter = wsmi(epochs, kernel=3, tau=1, anti_aliasing=False)

    # Both should produce valid results
    assert np.all(np.isfinite(conn_with_filter.get_data()))
    assert np.all(np.isfinite(conn_without_filter.get_data()))


def test_wsmi_ground_truth_validation():
    """Test wSMI against ground truth data for regression testing."""
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
            indices=(np.array([1, 2, 2]), np.array([0, 0, 1])),
        )

        # Compare with expected output
        expected_data = test_case["expected_output"].get_data()
        new_data = conn.get_data()

        assert expected_data.shape == new_data.shape
        assert_allclose(expected_data, new_data, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("picks", [None, "all", "goods"])
def test_wsmi_bad_channels(picks):
    """Test wsmi bad channels handling."""
    # Simulate data
    rng = np.random.default_rng(0)
    n_epochs = 2
    n_channels = 3  # do not change!
    sfreq = 50
    data = rng.standard_normal((n_epochs, n_channels, sfreq))
    info = create_info(n_channels, sfreq, "eeg")
    data = EpochsArray(data, info)

    # Mark a channel as bad
    data.info["bads"] = [data.ch_names[1]]

    # Create indices
    if picks is not None:
        if picks == "all":  # explicit bad inclusion
            indices = (np.array([1, 2, 2]), np.array([0, 0, 1]))
        else:  # ("goods") explicit bad exclusion
            indices = (np.array([2]), np.array([0]))
        n_cons = len(indices[0])
    else:
        indices = None  # implicit bad exclusion
        n_cons = n_channels**2

    # Compute connectivity
    con = wsmi(data, kernel=3, tau=1, indices=indices, average=True)

    # Check connectivity object properties
    assert con.n_nodes == n_channels
    assert con.names == data.ch_names

    # Check dense shape same regardless of indices
    assert con.get_data("dense").shape == (n_channels, n_channels)

    # Check raveled shape and contents depends on indices
    raveled_data = con.get_data("raveled")
    assert raveled_data.shape == (n_cons,)  # n_cons depends on picks
    if picks is not None:
        # with "all" channels used, bads entries are present and are non-zero
        # with "goods" channels used, bads entries are non-existent
        # in both cases, all entries are non-zero
        assert_array_less(0, raveled_data)
    else:  # indices=None → all-to-all connectivity
        # bads entries present, but filled with zeros
        assert_array_equal(raveled_data[[3, 7]], 0)  # bads indices
        # (use np.ravel_multi_index to find dense array indices in raveled array)
