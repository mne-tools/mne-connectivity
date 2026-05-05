from collections.abc import Generator

import numpy as np
import pytest
from mne import EpochsArray, create_info
from mne.time_frequency import EpochsSpectrumArray

from mne_connectivity import (
    make_signals_in_freq_bands,
    make_surrogate_data,
    make_surrogate_evoked_data,
    make_surrogate_resting_data,
    seed_target_indices,
    spectral_connectivity_epochs,
)


@pytest.mark.parametrize("n_seeds", [1, 3])
@pytest.mark.parametrize("n_targets", [1, 3])
@pytest.mark.parametrize("snr", [0.7, 0.4])
@pytest.mark.parametrize("connection_delay", [0, 3, -3])
@pytest.mark.parametrize("mode", ["multitaper", "fourier", "cwt_morlet"])
def test_make_signals_in_freq_bands(n_seeds, n_targets, snr, connection_delay, mode):
    """Test `make_signals_in_freq_bands` simulates connectivity properly."""
    # Case with no spurious correlations (avoids tests randomly failing)
    rng_seed = 0

    # Simulate data
    freq_band = (5, 10)  # fmin, fmax (Hz)
    sfreq = 100  # Hz
    trans_bandwidth = 1  # Hz
    data = make_signals_in_freq_bands(
        n_seeds=n_seeds,
        n_targets=n_targets,
        freq_band=freq_band,
        n_epochs=30,
        n_times=200,
        sfreq=sfreq,
        trans_bandwidth=trans_bandwidth,
        snr=snr,
        connection_delay=connection_delay,
        rng_seed=rng_seed,
    )

    # Compute connectivity
    methods = ["coh", "imcoh", "dpli"]
    indices = seed_target_indices(
        seeds=np.arange(n_seeds), targets=np.arange(n_targets) + n_seeds
    )
    fmin = 3
    fmax = sfreq // 2
    if mode == "cwt_morlet":
        cwt_params = {"cwt_freqs": np.arange(fmin, fmax), "cwt_n_cycles": 3.5}
    else:
        cwt_params = dict()
    con = spectral_connectivity_epochs(
        data,
        method=methods,
        indices=indices,
        mode=mode,
        fmin=fmin,
        fmax=fmax,
        **cwt_params,
    )
    freqs = np.array(con[0].freqs)

    # Define expected connectivity values
    thresh_good = dict()
    thresh_bad = dict()
    # Coh
    thresh_good["coh"] = (0.2, 0.9)
    thresh_bad["coh"] = (0.0, 0.2)
    # ImCoh
    if connection_delay == 0:
        thresh_good["imcoh"] = (0.0, 0.17)
        thresh_bad["imcoh"] = (0.0, 0.17)
    else:
        thresh_good["imcoh"] = (0.17, 0.8)
        thresh_bad["imcoh"] = (0.0, 0.17)
    # DPLI
    if connection_delay == 0:
        thresh_good["dpli"] = (0.3, 0.6)
        thresh_bad["dpli"] = (0.3, 0.6)
    elif connection_delay > 0:
        thresh_good["dpli"] = (0.5, 1)
        thresh_bad["dpli"] = (0.3, 0.6)
    else:
        thresh_good["dpli"] = (0, 0.5)
        thresh_bad["dpli"] = (0.3, 0.6)

    # Check connectivity values are acceptable
    freqs_good = np.argwhere(
        (freqs >= freq_band[0]) & (freqs <= freq_band[1])
    ).flatten()
    freqs_bad = np.argwhere(
        (freqs < freq_band[0] - trans_bandwidth * 2)
        | (freqs > freq_band[1] + trans_bandwidth * 2)
    ).flatten()
    for method_name, method_con in zip(methods, con):
        con_values = method_con.get_data()
        if method_name == "imcoh":
            con_values = np.abs(con_values)
        # freq. band of interest
        con_values_good = np.mean(con_values[:, freqs_good])
        assert (
            con_values_good >= thresh_good[method_name][0]
            and con_values_good <= thresh_good[method_name][1]
        )

        # other freqs.
        con_values_bad = np.mean(con_values[:, freqs_bad])
        assert (
            con_values_bad >= thresh_bad[method_name][0]
            and con_values_bad <= thresh_bad[method_name][1]
        )


def test_make_signals_in_freq_bands_error_catch():
    """Test error catching for `make_signals_in_freq_bands`."""
    freq_band = (5, 10)

    # check bad n_seeds/targets caught
    with pytest.raises(
        ValueError, match="Number of seeds and targets must each be at least 1."
    ):
        make_signals_in_freq_bands(n_seeds=0, n_targets=1, freq_band=freq_band)
    with pytest.raises(
        ValueError, match="Number of seeds and targets must each be at least 1."
    ):
        make_signals_in_freq_bands(n_seeds=1, n_targets=0, freq_band=freq_band)

    # check bad freq_band caught
    with pytest.raises(TypeError, match="freq_band must be an instance of tuple."):
        make_signals_in_freq_bands(n_seeds=1, n_targets=1, freq_band=1)
    with pytest.raises(ValueError, match="Frequency band must contain two numbers."):
        make_signals_in_freq_bands(n_seeds=1, n_targets=1, freq_band=(1, 2, 3))

    # check bad n_times
    with pytest.raises(ValueError, match="Number of timepoints must be at least 1."):
        make_signals_in_freq_bands(
            n_seeds=1, n_targets=1, freq_band=freq_band, n_times=0
        )

    # check bad n_epochs
    with pytest.raises(ValueError, match="Number of epochs must be at least 1."):
        make_signals_in_freq_bands(
            n_seeds=1, n_targets=1, freq_band=freq_band, n_epochs=0
        )

    # check bad sfreq
    with pytest.raises(ValueError, match="Sampling frequency must be > 0."):
        make_signals_in_freq_bands(n_seeds=1, n_targets=1, freq_band=freq_band, sfreq=0)

    # check bad snr
    with pytest.raises(
        ValueError, match="Signal-to-noise ratio must be between 0 and 1."
    ):
        make_signals_in_freq_bands(n_seeds=1, n_targets=1, freq_band=freq_band, snr=-1)
    with pytest.raises(
        ValueError, match="Signal-to-noise ratio must be between 0 and 1."
    ):
        make_signals_in_freq_bands(n_seeds=1, n_targets=1, freq_band=freq_band, snr=2)

    # check bad connection_delay
    with pytest.raises(
        ValueError,
        match="Connection delay must be less than the total number of timepoints.",
    ):
        make_signals_in_freq_bands(
            n_seeds=1,
            n_targets=1,
            freq_band=freq_band,
            n_epochs=1,
            n_times=1,
            connection_delay=1,
        )


# TODO Version: remove in 0.10
def test_make_surrogate_data_deprecation():
    """Test `make_surrogate_data` warning about deprecation."""
    n_epochs = 5
    n_chans = 6
    n_freqs = 50
    sfreq = n_freqs * 2
    rng = np.random.default_rng(44)
    data = rng.random((n_epochs, n_chans, n_freqs)).astype(np.complex128)
    data += data * 1j  # complex dtypes not supported for simulation, so make complex
    info = create_info(ch_names=n_chans, sfreq=sfreq, ch_types="eeg")
    data = EpochsSpectrumArray(data=data, info=info, freqs=np.arange(n_freqs))

    with pytest.warns(
        FutureWarning,
        match="`make_surrogate_data` is deprecated and will be removed in 0.10.",
    ):
        make_surrogate_data(data, n_shuffles=5)


@pytest.mark.parametrize(("snr", "should_be_significant"), ([0.4, True], [0.1, False]))
@pytest.mark.parametrize(
    ("state", "method"),
    (
        ["resting", "multitaper"],
        ["resting", "welch"],
        ["resting", "morlet"],
        ["evoked", "morlet"],
    ),
)
def test_make_surrogate_data(snr, should_be_significant, state, method):
    """Test `make_surrogate_xxx_data` creates data for null hypothesis testing."""
    # Generate data
    n_seeds = 2
    n_targets = 2
    freq_band = (15, 20)
    if state == "evoked":
        connection_time, connection_width = 0.5, 0.2  # seconds
    else:
        connection_time, connection_width = None, None
    n_epochs = 30
    sfreq = 100
    n_times = sfreq
    trans_bw = 1
    n_shuffles = 1000
    rng_seed = 0
    data = make_signals_in_freq_bands(
        n_seeds=n_seeds,
        n_targets=n_targets,
        freq_band=freq_band,
        n_epochs=n_epochs,
        n_times=n_times,
        sfreq=sfreq,
        trans_bandwidth=trans_bw,
        snr=snr,  # using very high SNR seems to alter properties of data beyond fband
        connection_time=connection_time,
        connection_width=connection_width,
        rng_seed=rng_seed,
    )
    indices = seed_target_indices(
        seeds=np.arange(n_seeds), targets=np.arange(n_targets) + n_seeds
    )

    # Compute Fourier coefficients and generate surrogates
    fmin, fmax = 5, 30
    if method == "morlet":
        freqs = np.arange(fmin, fmax + 1, 1)
        coeffs = data.compute_tfr(
            method=method, freqs=freqs, n_cycles=freqs / 2, output="complex"
        )
    else:
        coeffs = data.compute_psd(method=method, fmin=fmin, fmax=fmax, output="complex")

    surrogate_func = (
        make_surrogate_evoked_data if state == "evoked" else make_surrogate_resting_data
    )
    surrogate_coeffs = surrogate_func(data=coeffs, n_shuffles=1000, rng_seed=rng_seed)

    # Compute connectivity
    con = spectral_connectivity_epochs(data=coeffs, method="coh", indices=indices)
    freqs = np.array(con.freqs)
    connectivity = np.zeros((n_shuffles + 1, *con.shape))
    connectivity[0] = con.get_data()  # first entry is original data
    for shuffle_i, shuffle_data in enumerate(surrogate_coeffs):
        connectivity[shuffle_i + 1] = spectral_connectivity_epochs(
            data=shuffle_data, method="coh", indices=indices, verbose=False
        ).get_data()
    if method == "morlet" and state == "resting":
        connectivity = np.mean(connectivity, axis=-1)  # average over time

    # Determine if connectivity significant
    alpha = 0.05
    con_freqs = (freqs >= freq_band[0] - trans_bw * 2) & (
        freqs <= freq_band[1] + trans_bw * 2
    )
    if state == "evoked":
        times = np.array(con.times)
        con_times = (times >= connection_time - connection_width / 2) & (
            times <= connection_time + connection_width / 2
        )
        con_points = con_freqs[:, np.newaxis] & con_times[np.newaxis, :]
    else:
        con_points = con_freqs
    noise_points = np.invert(con_points)

    pval_con = (
        np.sum(
            np.mean(connectivity[0, :, con_points])  # aggregate cons & freqs (& times)
            <= np.mean(connectivity[1:, :, con_points], axis=(1, 2))  # same aggr. here
        )
        / n_shuffles
    )

    pval_noise = (
        np.sum(
            np.mean(connectivity[0, :, noise_points])
            > np.mean(connectivity[1:, :, noise_points], axis=(1, 2))
        )
        / n_shuffles
    )

    if should_be_significant:
        assert pval_con < alpha, f"pval_con: {pval_con}"
    else:
        assert pval_con >= alpha, f"pval_con: {pval_con}"

    # Freqs where nothing simulated should never be significant
    assert pval_noise >= alpha, f"pval_noise: {pval_noise}"


def test_make_surrogate_resting_data_kind_consistency():
    """Test `make_surrogate_resting_data` is consistent for epochs, PSD, & TFR data.

    N.B. This test does not work for `make_surrogate_evoked_data`, because the cutting
    of timepoints changes the temporal structure, and there is a smearing of information
    across times when computiong TFRs. So, the results of epochs → surrogate epochs →
    TFRs vs. epochs → TFRs → surrogate TFRs will have differences in the time domain.

    This is not the case for `make_surrogate_resting_data`, because the shuffling is
    performed for whole epochs (i.e., temporal structure within epochs is preserved).
    """
    # Generate random data for packaging into Epochs
    n_epochs = 5
    n_chans = 6
    n_times = 200
    sfreq = 50
    rng = np.random.default_rng(44)
    data = rng.random((n_epochs, n_chans, n_times))
    info = create_info(ch_names=n_chans, sfreq=sfreq, ch_types="eeg")
    data = EpochsArray(data=data, info=info)

    # Get info for calls to spectral compute & surrogate generation functions
    coeff_funcs_kwargs = [
        ("compute_psd", dict(output="complex")),
        (
            "compute_tfr",
            dict(method="morlet", freqs=np.arange(5, sfreq // 2), output="complex"),
        ),
    ]
    surrogate_kwargs = dict(n_shuffles=5, rng_seed=44)

    # Check surrogates from different data kinds matches, i.e.:
    # epochs → surrogate epochs → coeffs == epochs → coeffs → surrogate coeffs
    epochs_surrogates = make_surrogate_resting_data(data=data, **surrogate_kwargs)
    for func_name, func_kwargs in coeff_funcs_kwargs:
        # Compute coefficients from Epochs, then get surrogates from those coefficients
        coeff_func = getattr(data, func_name)
        coeffs = coeff_func(**func_kwargs)
        coeffs_surrogates = make_surrogate_resting_data(data=coeffs, **surrogate_kwargs)
        # Compare surrogates from both flows
        for epochs_surrogate, coeffs_surrogate in zip(
            epochs_surrogates, coeffs_surrogates
        ):
            coeff_func = getattr(epochs_surrogate, func_name)
            assert np.allclose(
                coeff_func(**func_kwargs).get_data(), coeffs_surrogate.get_data()
            ), "Surrogate data not consistent across epochs and PSD coeffs."


@pytest.mark.parametrize(
    "surrogate_func", [make_surrogate_resting_data, make_surrogate_evoked_data]
)
def test_make_surrogate_data_generator(surrogate_func):
    """Test `return_generator` parameter works in `make_surrogate_xxx_data`."""
    # Generate random data for packaging into Epochs
    n_epochs = 5
    n_chans = 6
    n_times = 200
    sfreq = 50
    rng = np.random.default_rng(44)
    data = rng.random((n_epochs, n_chans, n_times))
    info = create_info(ch_names=n_chans, sfreq=sfreq, ch_types="eeg")
    epochs = EpochsArray(data=data, info=info)
    coeffs = epochs.compute_tfr(
        method="morlet", freqs=np.arange(5, sfreq // 2), output="complex"
    )

    # Test generator (not) returned when requested
    for input_data in (epochs, coeffs):
        for return_generator, expects in zip([True, False], [Generator, list]):
            surrogate_data = surrogate_func(
                data=input_data, n_shuffles=5, return_generator=return_generator
            )
            assert isinstance(surrogate_data, expects), type(surrogate_data)


@pytest.mark.parametrize(
    "surrogate_func", [make_surrogate_resting_data, make_surrogate_evoked_data]
)
def test_make_surrogate_data_error_catch(surrogate_func):
    """Test error catching for `make_surrogate_xxx_data`."""
    # Generate random data for packaging into EpochsTFR
    n_epochs = 5
    n_chans = 6
    n_times = 200
    sfreq = 50
    rng = np.random.default_rng(44)
    data = rng.random((n_epochs, n_chans, n_times))
    info = create_info(ch_names=n_chans, sfreq=sfreq, ch_types="eeg")
    epochs = EpochsArray(data=data, info=info)

    # check bad data container type
    bad_type_message = r"data must be an instance of.*Epochs"
    if surrogate_func is make_surrogate_resting_data:
        bad_type_message += r".*EpochsSpectrum"
    bad_type_message += r".*EpochsTFR"
    with pytest.raises(TypeError, match=bad_type_message):
        surrogate_func(data=data)

    # check bad coeffs dtype
    with pytest.raises(TypeError, match="Values in `data` must be complex-valued"):
        bad_dtype_data = epochs.compute_tfr(
            method="morlet", freqs=np.arange(5, sfreq // 2), output="power"
        )
        surrogate_func(data=bad_dtype_data)

    # check bad data shape
    if surrogate_func is make_surrogate_resting_data:
        with pytest.raises(ValueError, match="Data must contain more than one epoch"):
            bad_nepochs_data = epochs[0]
            surrogate_func(data=bad_nepochs_data)
    else:
        with pytest.raises(
            ValueError, match="Data must contain more than one timepoint"
        ):
            bad_ntimes_data = EpochsArray(data=data[..., [0]], info=info)
            surrogate_func(data=bad_ntimes_data)
    with pytest.raises(ValueError, match="Data must contain more than one channel"):
        bad_nchans_data = EpochsArray(
            data=data[:, [0]], info=create_info(ch_names=1, sfreq=sfreq, ch_types="eeg")
        )
        surrogate_func(data=bad_nchans_data)

    # check bad n_shuffles
    with pytest.raises(TypeError, match="n_shuffles must be an instance of int"):
        surrogate_func(data=epochs, n_shuffles="all")
    with pytest.raises(ValueError, match="Number of shuffles must be >= 1"):
        surrogate_func(data=epochs, n_shuffles=0)
    with pytest.raises(ValueError, match="Number of shuffles must be >= 1"):
        surrogate_func(data=epochs, n_shuffles=-1)

    # check bad return_generator
    with pytest.raises(TypeError, match="return_generator must be an instance of bool"):
        surrogate_func(data=epochs, return_generator="yes")
