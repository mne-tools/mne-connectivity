import numpy as np
import pytest
from numpy.testing import assert_allclose

from mne_connectivity import (
    make_signals_in_freq_bands,
    seed_target_indices,
    spectral_connectivity_epochs,
)
from mne_connectivity.decoding import CoherencyDecomposition
from mne_connectivity.utils import _check_multivariate_indices


@pytest.mark.parametrize("method", ["cacoh", "mic"])
@pytest.mark.parametrize("mode", ["multitaper", "fourier", "cwt_morlet"])
def test_spectral_decomposition(method, mode):
    """Test spectral decomposition classes run and give expected results."""
    # SIMULATE DATA
    # Settings
    n_seeds = 3
    n_targets = 3
    n_signals = n_seeds + n_targets
    n_epochs = 60
    trans_bandwidth = 1
    fstart = 5  # start computing connectivity
    fend = 30  # stop computing connectivity

    # Get data with connectivity to optimise (~90° angle good for MIC & CaCoh)
    fmin_optimise = 11
    fmax_optimise = 14
    epochs_optimise = make_signals_in_freq_bands(
        n_seeds=n_seeds,
        n_targets=n_targets,
        freq_band=(fmin_optimise, fmax_optimise),
        n_epochs=n_epochs,
        trans_bandwidth=trans_bandwidth,
        snr=0.5,
        connection_delay=10,  # ~90° interaction angle for this freq. band
        rng_seed=44,
    )

    # Get data with connectivity to ignore
    fmin_ignore = 21
    fmax_ignore = 24
    epochs_ignore = make_signals_in_freq_bands(
        n_seeds=n_seeds,
        n_targets=n_targets,
        freq_band=(fmin_ignore, fmax_ignore),
        n_epochs=n_epochs,
        trans_bandwidth=trans_bandwidth,
        snr=0.5,
        connection_delay=6,  # ~90° interaction angle for this freq. band
        rng_seed=42,
    )

    # Combine data and get indices
    epochs = epochs_optimise.add_channels([epochs_ignore])
    seeds = np.concatenate((np.arange(n_seeds), np.arange(n_seeds) + n_signals))
    targets = np.concatenate(
        (np.arange(n_targets) + n_seeds, np.arange(n_targets) + n_signals + n_seeds)
    )
    indices = (seeds, targets)

    if method == "cacoh":
        bivariate_method = "coh"
        multivariate_method = "cacoh"
    else:
        bivariate_method = "imcoh"
        multivariate_method = "mic"

    cwt_freq_res = 0.5
    cwt_freqs = np.arange(fmin_optimise, fmax_optimise + cwt_freq_res, cwt_freq_res)
    cwt_n_cycles = 6

    # TEST FITTING AND TRANSFORMING SAME DATA EXTRACTS CONNECTIVITY
    decomp_class = CoherencyDecomposition(
        info=epochs.info,
        method=method,
        indices=indices,
        mode=mode,
        fmin=fmin_optimise,
        fmax=fmax_optimise,
        cwt_freqs=cwt_freqs,
        cwt_n_cycles=cwt_n_cycles,
    )
    epochs_transformed = decomp_class.fit_transform(
        X=epochs[: n_epochs // 2].get_data()
    )
    con_mv_class = spectral_connectivity_epochs(
        epochs_transformed,
        method=bivariate_method,
        indices=decomp_class.get_transformed_indices(),
        sfreq=epochs.info["sfreq"],
        mode=mode,
        fmin=fstart,
        fmax=fend,
        cwt_freqs=np.arange(fstart, fend + cwt_freq_res, cwt_freq_res),
        cwt_n_cycles=cwt_n_cycles,
    )
    con_mv_func = spectral_connectivity_epochs(
        epochs[: n_epochs // 2],
        method=multivariate_method,
        indices=([seeds], [targets]),
        mode=mode,
        fmin=fstart,
        fmax=fend,
        cwt_freqs=np.arange(fstart, fend + cwt_freq_res, cwt_freq_res),
        cwt_n_cycles=cwt_n_cycles,
    )
    con_bv_func = spectral_connectivity_epochs(
        epochs[: n_epochs // 2],
        method=bivariate_method,
        indices=seed_target_indices(seeds, targets),
        mode=mode,
        fmin=fstart,
        fmax=fend,
        cwt_freqs=np.arange(fstart, fend + cwt_freq_res, cwt_freq_res),
        cwt_n_cycles=cwt_n_cycles,
    )

    # Frequencies of interest
    freqs = np.array(con_mv_class.freqs)
    freqs_optimise = (freqs >= fmin_optimise) & (freqs <= fmax_optimise)
    freqs_ignore = (freqs >= fmin_ignore) & (freqs <= fmax_ignore)

    # Thresholds for checking validity of connectivity (work across all modes)
    optimisation_diff = 0.35  # optimisation causes big increase in connectivity
    similarity_thresh = 0.15  # freqs. being optimised or ignored should be very similar

    # Test selective optimisation of desired freq. band vs. no optimisation
    assert (
        np.abs(con_mv_class.get_data()[0, freqs_optimise]).mean()
        > np.abs(con_bv_func.get_data()[:, freqs_optimise]).mean() + optimisation_diff
    )  # check connectivity for optimised freq. band higher than without optimisation
    assert_allclose(
        np.abs(con_mv_class.get_data()[0, freqs_ignore]).mean(),
        np.abs(con_bv_func.get_data()[:, freqs_ignore]).mean(),
        atol=similarity_thresh,
    )  # check connectivity for ignored freq. band similar to no optimisation

    # Test band-wise optimisation similar to bin-wise optimisation
    assert_allclose(
        np.abs(con_mv_class.get_data()[0, freqs_optimise]).mean(),
        np.abs(con_mv_func.get_data()[0, freqs_optimise]).mean(),
        atol=similarity_thresh,
    )  # check connectivity for optimised freq. band similar for both versions
    assert (
        np.abs(con_mv_class.get_data()[0, freqs_ignore]).mean()
        < np.abs(con_mv_func.get_data()[0, freqs_ignore]).mean() - optimisation_diff
    )  # check connectivity for ignored freq. band lower than with optimisation

    # Test `fit_transform` equivalent to `fit` and `transform` separately
    decomp_class_2 = CoherencyDecomposition(
        info=epochs.info,
        method=method,
        indices=indices,
        mode=mode,
        fmin=fmin_optimise,
        fmax=fmax_optimise,
        cwt_freqs=cwt_freqs,
        cwt_n_cycles=cwt_n_cycles,
    )
    decomp_class_2.fit(X=epochs[: n_epochs // 2].get_data())
    epochs_transformed_2 = decomp_class_2.transform(
        X=epochs[: n_epochs // 2].get_data()
    )
    assert_allclose(epochs_transformed, epochs_transformed_2, rtol=1e-5)
    assert_allclose(decomp_class.filters_, decomp_class_2.filters_, rtol=1e-5)
    assert_allclose(decomp_class.patterns_, decomp_class_2.patterns_, rtol=1e-5)

    # TEST FITTING ON ONE PIECE OF DATA AND TRANSFORMING ANOTHER
    con_mv_class_unseen_data = spectral_connectivity_epochs(
        decomp_class.transform(X=epochs[n_epochs // 2 :].get_data()),
        method=bivariate_method,
        indices=decomp_class.get_transformed_indices(),
        sfreq=epochs.info["sfreq"],
        mode=mode,
        fmin=fstart,
        fmax=fend,
        cwt_freqs=np.arange(fstart, fend + cwt_freq_res, cwt_freq_res),
        cwt_n_cycles=cwt_n_cycles,
    )
    assert_allclose(
        np.abs(con_mv_class.get_data()[0, freqs_optimise]).mean(),
        np.abs(con_mv_class_unseen_data.get_data()[0, freqs_optimise]).mean(),
        atol=similarity_thresh,
    )  # check connectivity for optimised freq. band similarly high for seen & unseen
    assert_allclose(
        np.abs(con_mv_class.get_data()[0, freqs_ignore]).mean(),
        np.abs(con_mv_class_unseen_data.get_data()[0, freqs_ignore]).mean(),
        atol=similarity_thresh,
    )  # check connectivity for optimised freq. band similarly low for seen & unseen

    # XXX: TEST FILTERS/PATTERNS HAS CORRECT SHAPE WHEN N_COMPONENTS > 1 SUPPORTED

    # TEST GETTERS & SETTERS
    # Test indices internal storage and returned format
    assert np.all(np.array(decomp_class.indices) == np.array((seeds, targets)))
    assert np.all(
        decomp_class._indices
        == _check_multivariate_indices(([seeds], [targets]), n_signals)
    )
    decomp_class.set_params(indices=(targets, seeds))
    assert np.all(np.array(decomp_class.indices) == np.array((targets, seeds)))
    assert np.all(
        decomp_class._indices
        == _check_multivariate_indices(([targets], [seeds]), n_signals)
    )

    # Test rank internal storage and returned format
    assert np.all(decomp_class.rank == (n_signals, n_signals))
    assert np.all(decomp_class._rank == ([n_signals], [n_signals]))
    decomp_class.set_params(rank=(1, 2))
    assert np.all(decomp_class.rank == (1, 2))
    assert np.all(decomp_class._rank == ([1], [2]))

    # Test rank can be reset to default
    decomp_class.set_params(rank=None)


@pytest.mark.parametrize("method", ["cacoh", "mic"])
@pytest.mark.parametrize("mode", ["multitaper", "fourier", "cwt_morlet"])
def test_spectral_decomposition_parallel(method, mode):
    """Test spectral decomposition classes run with parallelisation."""
    # SIMULATE DATA
    n_seeds = 3
    n_targets = 3
    fmin = 10
    fmax = 15
    epochs = make_signals_in_freq_bands(
        n_seeds=n_seeds,
        n_targets=n_targets,
        freq_band=(fmin, fmax),
        snr=0.5,
        rng_seed=44,
    )

    # RUN DECOMPOSITION
    decomp_class = CoherencyDecomposition(
        info=epochs.info,
        method=method,
        indices=(np.arange(n_seeds), np.arange(n_targets) + n_seeds),
        mode=mode,
        fmin=fmin,
        fmax=fmax,
        cwt_freqs=np.arange(fmin, fmax + 0.5, 0.5),
        cwt_n_cycles=6,
        n_jobs=2,  # use parallelisation
    )
    decomp_class.fit_transform(X=epochs.get_data())


@pytest.mark.parametrize("method", ["cacoh", "mic"])
@pytest.mark.parametrize("mode", ["multitaper", "fourier", "cwt_morlet"])
def test_spectral_decomposition_error_catch(method, mode):
    """Test error catching for spectral decomposition classes."""
    # SIMULATE DATA
    n_seeds = 3
    n_targets = 3
    fmin = 15
    fmax = 20
    epochs = make_signals_in_freq_bands(
        n_seeds=n_seeds, n_targets=n_targets, freq_band=(fmin, fmax), rng_seed=44
    )
    indices = (np.arange(n_seeds), np.arange(n_targets) + n_seeds)
    cwt_freqs = np.arange(fmin, fmax + 0.5, 0.5)
    cwt_n_cycles = 6

    # TEST BAD INITIALISATION
    # Test info
    with pytest.raises(TypeError, match="`info` must be an instance of mne.Info"):
        CoherencyDecomposition(info="info", method=method, indices=indices)

    # Test indices
    with pytest.raises(
        TypeError, match="`indices` must be an instance of tuple of array-likes"
    ):
        CoherencyDecomposition(info=epochs.info, method=method, indices=list(indices))
    with pytest.raises(
        TypeError, match="`indices` must be an instance of tuple of array-likes"
    ):
        CoherencyDecomposition(info=epochs.info, method=method, indices=(0, 1))
    with pytest.raises(ValueError, match="`indices` must have length 2"):
        CoherencyDecomposition(info=epochs.info, method=method, indices=(indices[0],))
    with pytest.raises(
        ValueError,
        match=(
            "multivariate indices cannot contain repeated channels within a seed or "
            "target"
        ),
    ):
        CoherencyDecomposition(
            info=epochs.info, method=method, indices=([0, 0], [1, 2])
        )
    with pytest.raises(
        ValueError,
        match=(
            "multivariate indices cannot contain repeated channels within a seed or "
            "target"
        ),
    ):
        CoherencyDecomposition(
            info=epochs.info, method=method, indices=([0, 1], [2, 2])
        )
    with pytest.raises(
        ValueError, match="a negative channel index is not present in the data"
    ):
        CoherencyDecomposition(
            info=epochs.info, method=method, indices=([0], [(n_seeds + n_targets) * -1])
        )
    with pytest.raises(
        ValueError,
        match=(
            "at least one entry in `indices` is greater than the number of channels in "
            "`info`"
        ),
    ):
        CoherencyDecomposition(
            info=epochs.info, method=method, indices=([0], [n_seeds + n_targets])
        )

    # Test mode
    with pytest.raises(ValueError, match="Invalid value for the 'mode' parameter"):
        CoherencyDecomposition(
            info=epochs.info, method=method, indices=indices, mode="notamode"
        )

    base_kwargs = dict(
        info=epochs.info,
        method=method,
        indices=indices,
        mode=mode,
    )

    # Test fmin & fmax
    if mode in ["multitaper", "fourier"]:
        with pytest.raises(
            TypeError,
            match=(
                "`fmin` and `fmax` must not be None if `mode` is 'multitaper' or "
                "'fourier'"
            ),
        ):
            CoherencyDecomposition(**base_kwargs, fmin=None, fmax=fmax)
        with pytest.raises(
            TypeError,
            match=(
                "`fmin` and `fmax` must not be None if `mode` is 'multitaper' or "
                "'fourier'"
            ),
        ):
            CoherencyDecomposition(**base_kwargs, fmin=fmin, fmax=None)
        with pytest.raises(
            TypeError, match="`fmin` must be an instance of int or float"
        ):
            CoherencyDecomposition(**base_kwargs, fmin="15", fmax=fmax)
        with pytest.raises(
            TypeError, match="`fmax` must be an instance of int or float"
        ):
            CoherencyDecomposition(**base_kwargs, fmin=fmin, fmax="20")
        with pytest.raises(ValueError, match="`fmax` must be larger than `fmin`"):
            CoherencyDecomposition(**base_kwargs, fmin=fmax, fmax=fmin)
        with pytest.raises(
            ValueError, match="`fmax` cannot be larger than the Nyquist frequency"
        ):
            CoherencyDecomposition(
                **base_kwargs, fmin=fmin, fmax=epochs.info["sfreq"] / 2 + 1
            )

    # Test multitaper settings
    if mode == "multitaper":
        with pytest.raises(
            TypeError, match="`mt_bandwidth` must be an instance of int, float, or None"
        ):
            CoherencyDecomposition(
                **base_kwargs, fmin=fmin, fmax=fmax, mt_bandwidth="5"
            )
        with pytest.raises(
            TypeError, match="`mt_adaptive` must be an instance of bool"
        ):
            CoherencyDecomposition(**base_kwargs, fmin=fmin, fmax=fmax, mt_adaptive=1)
        with pytest.raises(
            TypeError, match="`mt_low_bias` must be an instance of bool"
        ):
            CoherencyDecomposition(**base_kwargs, fmin=fmin, fmax=fmax, mt_low_bias=1)

    # Test wavelet settings
    if mode == "cwt_morlet":
        with pytest.raises(
            TypeError, match="`cwt_freqs` must not be None if `mode` is 'cwt_morlet'"
        ):
            CoherencyDecomposition(**base_kwargs, cwt_freqs=None)
        with pytest.raises(
            TypeError, match="`cwt_freqs` must be an instance of array-like"
        ):
            CoherencyDecomposition(**base_kwargs, cwt_freqs="1")
        with pytest.raises(
            ValueError,
            match=(
                "last entry of `cwt_freqs` cannot be larger than the Nyquist frequency"
            ),
        ):
            CoherencyDecomposition(
                **base_kwargs,
                cwt_freqs=np.array([epochs.info["sfreq"] / 2 + 1]),
                cwt_n_cycles=cwt_n_cycles,
            )
        with pytest.raises(
            TypeError,
            match="`cwt_n_cycles` must be an instance of int, float, or array-like",
        ):
            CoherencyDecomposition(**base_kwargs, cwt_freqs=cwt_freqs, cwt_n_cycles="5")
        with pytest.raises(
            ValueError,
            match="`cwt_n_cycles` array-like must have the same length as `cwt_freqs`",
        ):
            CoherencyDecomposition(
                **base_kwargs,
                cwt_freqs=cwt_freqs,
                cwt_n_cycles=np.full(cwt_freqs.shape[0] - 1, 5),
            )

    base_kwargs.update(
        fmin=fmin, fmax=fmax, cwt_freqs=cwt_freqs, cwt_n_cycles=cwt_n_cycles
    )

    # Test n_components
    with pytest.raises(
        TypeError, match="`n_components` must be an instance of int or None"
    ):
        CoherencyDecomposition(**base_kwargs, n_components="2")

    # Test rank
    with pytest.raises(
        TypeError, match="`rank` must be an instance of tuple of ints or None"
    ):
        CoherencyDecomposition(**base_kwargs, rank="2")
    with pytest.raises(
        TypeError, match="`rank` must be an instance of tuple of ints or None"
    ):
        CoherencyDecomposition(**base_kwargs, rank=("2", "2"))
    with pytest.raises(ValueError, match="`rank` must have length 2"):
        CoherencyDecomposition(**base_kwargs, rank=(2,))
    with pytest.raises(ValueError, match="entries of `rank` must be > 0"):
        CoherencyDecomposition(**base_kwargs, rank=(0, 1))
    with pytest.raises(
        ValueError,
        match=(
            "at least one entry in `rank` is greater than the number of seed/target "
            "channels in `indices`"
        ),
    ):
        CoherencyDecomposition(**base_kwargs, rank=(n_seeds + 1, n_targets))
    with pytest.raises(
        ValueError,
        match=(
            "at least one entry in `rank` is greater than the number of seed/target "
            "channels in `indices`"
        ),
    ):
        CoherencyDecomposition(**base_kwargs, rank=(n_seeds, n_targets + 1))

    decomp_class = CoherencyDecomposition(**base_kwargs)

    # TEST BAD FITTING
    # Test input data format
    with pytest.raises(TypeError, match="`X` must be an instance of NumPy array"):
        decomp_class.fit(X=epochs.get_data().tolist())
    with pytest.raises(ValueError, match="Invalid value for the '`X.ndim`' parameter"):
        decomp_class.fit(X=epochs.get_data()[0])
    with pytest.raises(ValueError, match="`X` does not match Info"):
        decomp_class.fit(X=epochs.get_data()[:, :-1])
    # Test rank of input data is compatible with n_components
    decomp_class.set_params(n_components=3)
    with pytest.raises(
        ValueError, match="`n_components` is greater than the minimum rank of the data"
    ):
        rank_def_data = epochs.get_data(copy=True)
        rank_def_data[:, n_seeds - 1] = rank_def_data[:, n_seeds - 2]
        decomp_class.fit(X=rank_def_data)
    with pytest.raises(
        ValueError, match="`n_components` is greater than the minimum rank of the data"
    ):
        rank_def_data = epochs.get_data(copy=True)
        rank_def_data[:, n_seeds + n_targets - 1] = rank_def_data[
            :, n_seeds + n_targets - 2
        ]
        decomp_class.fit(X=rank_def_data)

    # TEST TRANSFORM BEFORE FITTING
    with pytest.raises(
        RuntimeError,
        match="no filters are available, please call the `fit` method first",
    ):
        decomp_class.transform(X=epochs.get_data())

    decomp_class.set_params(n_components=None)  # reset to default
    decomp_class.fit(X=epochs.get_data())

    # TEST BAD TRANSFORMING
    with pytest.raises(TypeError, match="`X` must be an instance of NumPy array"):
        decomp_class.transform(X=epochs.get_data().tolist())
    with pytest.raises(ValueError, match="Invalid value for the '`X.ndim`' parameter"):
        decomp_class.transform(X=epochs.get_data()[0, 0])
    with pytest.raises(ValueError, match="`X` does not match Info"):
        decomp_class.transform(X=epochs.get_data()[:, :-1])
