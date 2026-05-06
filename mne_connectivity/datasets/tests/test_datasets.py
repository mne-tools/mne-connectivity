from collections.abc import Generator

import numpy as np
import pytest
from mne import create_info
from mne.time_frequency import EpochsSpectrumArray

from mne_connectivity import (
    make_signals_in_freq_bands,
    make_surrogate_data,
    seed_target_indices,
    spectral_connectivity_epochs,
)


@pytest.mark.parametrize("n_seeds", [1, 3])
@pytest.mark.parametrize("n_targets", [1, 3])
@pytest.mark.parametrize("snr", ["high", "low"])
@pytest.mark.parametrize("connection_delay", [0, 3, -3])
@pytest.mark.parametrize(
    ("mode", "state"),
    (
        ["multitaper", "resting"],
        ["welch", "resting"],
        ["morlet", "resting"],
        ["morlet", "evoked"],
    ),
)
def test_make_signals_in_freq_bands(
    n_seeds, n_targets, snr, connection_delay, mode, state
):
    """Test `make_signals_in_freq_bands` simulates connectivity properly."""
    # Case with no spurious correlations (avoids tests randomly failing)
    rng_seed = 0

    # Simulate data
    freq_band = (10, 15)  # fmin, fmax (Hz)
    sfreq = 100  # Hz
    n_times = 200  # samples
    trans_bandwidth = 1  # Hz
    if state == "evoked":
        epoch_dur = n_times / sfreq
        connection_time = epoch_dur / 2  # connectivity occurs in middle of epoch
        connection_width = epoch_dur / 4
    else:
        connection_time, connection_width = None, None
    data = make_signals_in_freq_bands(
        n_seeds=n_seeds,
        n_targets=n_targets,
        freq_band=freq_band,
        n_epochs=30,
        n_times=n_times,
        sfreq=sfreq,
        trans_bandwidth=trans_bandwidth,
        snr=0.75 if snr == "high" else 0.25,
        connection_delay=connection_delay,
        connection_time=connection_time,
        connection_width=connection_width,
        rng_seed=rng_seed,
    )

    # Compute connectivity
    methods = ["coh", "imcoh", "dpli"]
    indices = seed_target_indices(
        seeds=np.arange(n_seeds), targets=np.arange(n_targets) + n_seeds
    )
    n_cons = len(indices[0])
    fmin, fmax = 5, 30
    if mode == "morlet":
        freqs = np.arange(fmin, fmax + 1, 1)
        coeffs = data.compute_tfr(
            method=mode, freqs=freqs, n_cycles=freqs / 2, output="complex"
        )
    else:
        coeffs = data.compute_psd(method=mode, fmin=fmin, fmax=fmax, output="complex")
    con = spectral_connectivity_epochs(coeffs, method=methods, indices=indices)
    freqs = np.array(con[0].freqs)
    if mode == "morlet":
        times = np.array(con[0].times)

    # Define expected connectivity values
    con_thresh = dict()
    noise_thresh = dict()
    # Coh
    con_thresh["coh"] = (0.6, 1.0)
    noise_thresh["coh"] = (0.0, 0.3)
    # ImCoh
    if connection_delay == 0:
        con_thresh["imcoh"] = (0.0, 0.2)
        noise_thresh["imcoh"] = (0.0, 0.2)
    else:
        con_thresh["imcoh"] = (0.4, 1.0)  # min imcoh can be < min coh, due to phase
        noise_thresh["imcoh"] = (0.0, 0.2)
    # DPLI
    if connection_delay == 0:
        con_thresh["dpli"] = (0.3, 0.7)
        noise_thresh["dpli"] = (0.3, 0.7)
    elif connection_delay > 0:
        con_thresh["dpli"] = (0.7, 1.0)
        noise_thresh["dpli"] = (0.3, 0.7)
    else:
        con_thresh["dpli"] = (0.0, 0.3)
        noise_thresh["dpli"] = (0.3, 0.7)

    # Define points where connectivity should vs. should not be present
    con_freqs = np.argwhere((freqs >= freq_band[0]) & (freqs <= freq_band[1])).flatten()
    noise_freqs = np.argwhere(
        (freqs < freq_band[0] - trans_bandwidth * 2)
        | (freqs > freq_band[1] + trans_bandwidth * 2)
    ).flatten()
    if state == "evoked":  # connectivity only in certain time range
        con_times = np.argwhere(
            (times >= connection_time - connection_width / 2)
            & (times <= connection_time + connection_width / 2)
        ).flatten()
        noise_times = np.setdiff1d(np.arange(times.size), con_times)
        con_points = np.ix_(np.arange(n_cons), con_freqs, con_times)
        noise_points = np.ix_(np.arange(n_cons), noise_freqs, noise_times)
    elif mode == "morlet":  # resting-state, so connectivity in all time points
        con_points = np.ix_(np.arange(n_cons), con_freqs, np.arange(times.size))
        noise_points = np.ix_(np.arange(n_cons), noise_freqs, np.arange(times.size))
    else:  # no time dimension
        con_points = np.ix_(np.arange(n_cons), con_freqs)
        noise_points = np.ix_(np.arange(n_cons), noise_freqs)

    # Check connectivity values are acceptable
    for method_name, method_con in zip(methods, con):
        con_data = method_con.get_data()
        if method_name == "imcoh":
            con_data = np.abs(con_data)

        # freq. band (and times) of interest
        con_values = np.mean(con_data[con_points])
        if snr == "high":
            assert (
                con_thresh[method_name][0] <= con_values <= con_thresh[method_name][1]
            ), (
                f"{method_name} - expected range {con_thresh[method_name]}, got "
                f"{con_values:.3f}"
            )
        else:
            assert (
                noise_thresh[method_name][0]
                <= con_values
                <= noise_thresh[method_name][1]
            ), (
                f"{method_name} - expected range {noise_thresh[method_name]}, got "
                f"{con_values:.3f}"
            )

        # other freqs. (and times) where no connectivity should be present
        noise_values = np.mean(con_data[noise_points])
        assert (
            noise_thresh[method_name][0] <= noise_values <= noise_thresh[method_name][1]
        ), (
            f"{method_name} - expected range {noise_thresh[method_name]}, got "
            f"{noise_values:.3f}"
        )


def test_make_signals_in_freq_bands_con_time():
    """Test `connection_time` and related params in `make_signals_in_freq_bands`."""
    # Simulate data
    sfreq = 100  # Hz
    n_times = 200  # samples
    epoch_dur = n_times / sfreq
    sim_kwargs = dict(
        n_seeds=1,
        n_targets=1,
        freq_band=(10, 15),
        n_epochs=30,
        n_times=n_times,
        sfreq=sfreq,
        connection_width=epoch_dur / 4,
    )

    # Test that interaction centre outside of epoch gets caught
    connection_time = epoch_dur + 1 / sfreq
    with pytest.raises(
        ValueError, match=r"`connection_time`.*must be within the epoch time range"
    ):
        make_signals_in_freq_bands(connection_time=connection_time, **sim_kwargs)

    # Test that interaction centre


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

    # check bad connection_time/width
    with pytest.raises(
        TypeError, match="connection_time must be an instance of numeric or None."
    ):
        make_signals_in_freq_bands(
            n_seeds=1, n_targets=1, freq_band=freq_band, connection_time="middle"
        )
    n_times = 100
    sfreq = 50
    epoch_dur = n_times / sfreq
    with pytest.raises(
        ValueError,
        match=r"`connection_time`.*must be within the epoch time range",
    ):
        make_signals_in_freq_bands(
            n_seeds=1,
            n_targets=1,
            freq_band=freq_band,
            n_times=n_times,
            sfreq=sfreq,
            connection_time=epoch_dur + 1 / sfreq,
            connection_width=epoch_dur / 4,
        )
    with pytest.raises(
        ValueError,
        match=(
            "`connection_width` must be specified when `connection_time` is not None."
        ),
    ):
        make_signals_in_freq_bands(
            n_seeds=1,
            n_targets=1,
            freq_band=freq_band,
            n_times=n_times,
            sfreq=sfreq,
            connection_time=epoch_dur / 2,
        )
    with pytest.warns(
        UserWarning, match="`connection_width` is not None, but `connection_time` is"
    ):
        make_signals_in_freq_bands(
            n_seeds=1,
            n_targets=1,
            freq_band=freq_band,
            n_times=n_times,
            sfreq=sfreq,
            connection_width=epoch_dur / 4,
        )
    with pytest.raises(ValueError, match="`connection_width` must be > 0."):
        make_signals_in_freq_bands(
            n_seeds=1,
            n_targets=1,
            freq_band=freq_band,
            n_times=n_times,
            sfreq=sfreq,
            connection_time=epoch_dur / 2,
            connection_width=-1,
        )
    # not an error catch, but check that connection_width overlapping with end of epochs
    # is allowed
    make_signals_in_freq_bands(
        n_seeds=1,
        n_targets=1,
        freq_band=freq_band,
        n_times=n_times,
        sfreq=sfreq,
        connection_time=0,
        connection_width=epoch_dur / 4,
    )

    # check bad window_alpha
    with pytest.raises(TypeError, match="window_alpha must be an instance of float."):
        make_signals_in_freq_bands(
            n_seeds=1, n_targets=1, freq_band=freq_band, window_alpha="tukey"
        )
    with pytest.raises(ValueError, match="`window_alpha` must be between 0 and 1."):
        make_signals_in_freq_bands(
            n_seeds=1, n_targets=1, freq_band=freq_band, window_alpha=-0.5
        )


@pytest.mark.parametrize(("snr", "should_be_significant"), ([0.3, True], [0.1, False]))
@pytest.mark.parametrize("method", ["multitaper", "welch", "morlet"])
def test_make_surrogate_data(snr, should_be_significant, method):
    """Test `make_surrogate_data` creates data for null hypothesis testing."""
    # Generate data
    n_seeds = 2
    n_targets = 2
    freq_band = (10, 15)
    n_epochs = 30
    sfreq = 100
    n_times = sfreq * 2
    n_shuffles = 1000
    rng_seed = 1
    data = make_signals_in_freq_bands(
        n_seeds=n_seeds,
        n_targets=n_targets,
        freq_band=freq_band,
        n_epochs=n_epochs,
        n_times=n_times,
        sfreq=sfreq,
        snr=snr,  # using very high SNR seems to alter properties of data beyond fband
        rng_seed=rng_seed,
    )
    indices = seed_target_indices(
        seeds=np.arange(n_seeds), targets=np.arange(n_targets) + n_seeds
    )

    # Compute Fourier coefficients and generate surrogates
    fmin, fmax = 6, 50
    if method == "morlet":
        coeffs = data.compute_tfr(
            method=method, freqs=np.arange(fmin, fmax + 1, 1), output="complex"
        )
    else:
        coeffs = data.compute_psd(method=method, fmin=fmin, fmax=fmax, output="complex")
    surrogate_coeffs = make_surrogate_data(
        data=coeffs, n_shuffles=1000, rng_seed=rng_seed
    )

    # Compute connectivity
    con = spectral_connectivity_epochs(data=coeffs, method="coh", indices=indices)
    freqs = np.array(con.freqs)
    connectivity = np.zeros((n_shuffles + 1, *con.shape))
    connectivity[0] = con.get_data()  # first entry is original data
    for shuffle_i, shuffle_data in enumerate(surrogate_coeffs):
        connectivity[shuffle_i + 1] = spectral_connectivity_epochs(
            data=shuffle_data, method="coh", indices=indices, verbose=False
        ).get_data()
    if method == "morlet":
        connectivity = np.mean(connectivity, axis=-1)  # average over time

    # Determine if connectivity significant
    alpha = 0.05
    con_freqs = (freqs >= freq_band[0]) & (freqs <= freq_band[1])
    noise_freqs = np.invert(con_freqs)

    pval_con_freqs = (
        np.sum(
            np.mean(connectivity[0, :, con_freqs])  # aggregate cons and freqs
            <= np.mean(connectivity[1:, :, con_freqs], axis=(1, 2))  # same aggr. here
        )
        / n_shuffles
    )

    pval_noise_freqs = (
        np.sum(
            np.mean(connectivity[0, :, noise_freqs])
            <= np.mean(connectivity[1:, :, noise_freqs], axis=(1, 2))
        )
        / n_shuffles
    )

    if should_be_significant:
        assert pval_con_freqs < alpha, f"pval_con_freqs: {pval_con_freqs}"
    else:
        assert pval_con_freqs >= alpha, f"pval_con_freqs: {pval_con_freqs}"

    # Freqs where nothing simulated should never be significant
    assert pval_noise_freqs > alpha, f"pval_noise_freqs: {pval_noise_freqs}"


def test_make_surrogate_data_generator():
    """Test `return_generator` parameter works in `make_surrogate_data`."""
    # Generate random data for packaging into EpochsSpectrum
    n_epochs = 5
    n_chans = 6
    n_freqs = 50
    sfreq = n_freqs * 2
    rng = np.random.default_rng(44)
    data = rng.random((n_epochs, n_chans, n_freqs)).astype(np.complex128)
    data += data * 1j  # complex dtypes not supported for simulation, so make complex
    info = create_info(ch_names=n_chans, sfreq=sfreq, ch_types="eeg")
    spectrum = EpochsSpectrumArray(data=data, info=info, freqs=np.arange(n_freqs))

    # Test generator (not) returned when requested
    surrogate_data = make_surrogate_data(data=spectrum, return_generator=True)
    assert isinstance(surrogate_data, Generator), type(surrogate_data)
    surrogate_data = make_surrogate_data(data=spectrum, return_generator=False)
    assert isinstance(surrogate_data, list), type(surrogate_data)


def test_make_surrogate_data_error_catch():
    """Test error catching for `make_surrogate_data`."""
    # Generate random data for packaging into EpochsSpectrum
    n_epochs = 5
    n_chans = 6
    n_freqs = 50
    sfreq = n_freqs * 2
    rng = np.random.default_rng(44)
    data = rng.random((n_epochs, n_chans, n_freqs)).astype(np.complex128)
    data += data * 1j  # complex dtypes not supported for simulation, so make complex
    info = create_info(ch_names=n_chans, sfreq=sfreq, ch_types="eeg")
    spectrum = EpochsSpectrumArray(data=data, info=info, freqs=np.arange(n_freqs))

    # check bad data
    with pytest.raises(
        TypeError, match=r"data must be an instance of.*EpochsSpectrum.*EpochsTFR"
    ):
        make_surrogate_data(data=data)
    with pytest.raises(TypeError, match="values in `data` must be complex-valued"):
        bad_dtype_data = EpochsSpectrumArray(
            data=np.abs(data), info=info, freqs=np.arange(n_freqs)
        )
        make_surrogate_data(data=bad_dtype_data)
    with pytest.raises(ValueError, match="data must contain more than one epoch"):
        bad_nepochs_data = EpochsSpectrumArray(
            data=data[[0]], info=info, freqs=np.arange(n_freqs)
        )
        make_surrogate_data(data=bad_nepochs_data)
    with pytest.raises(ValueError, match="data must contain more than one channel"):
        bad_nchans_data = EpochsSpectrumArray(
            data=data[:, [0]],
            info=create_info(ch_names=1, sfreq=sfreq, ch_types="eeg"),
            freqs=np.arange(n_freqs),
        )
        make_surrogate_data(data=bad_nchans_data)

    # check bad n_shuffles
    with pytest.raises(TypeError, match="n_shuffles must be an instance of int"):
        make_surrogate_data(data=spectrum, n_shuffles="all")
    with pytest.raises(ValueError, match="number of shuffles must be >= 1"):
        make_surrogate_data(data=spectrum, n_shuffles=0)
    with pytest.raises(ValueError, match="number of shuffles must be >= 1"):
        make_surrogate_data(data=spectrum, n_shuffles=-1)

    # check bad return_generator
    with pytest.raises(TypeError, match="return_generator must be an instance of bool"):
        make_surrogate_data(data=spectrum, return_generator="yes")
