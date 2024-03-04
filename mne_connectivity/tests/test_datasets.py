import numpy as np
import pytest

from mne_connectivity import (
    make_signals_in_freq_bands,
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


def test_make_signals_error_catch():
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
    with pytest.raises(TypeError, match="Frequency band must be a tuple."):
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
