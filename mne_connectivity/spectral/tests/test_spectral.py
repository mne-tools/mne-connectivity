import inspect
import os
import platform

import numpy as np
import pandas as pd
import pytest
from mne import EpochsArray, SourceEstimate, create_info
from mne.filter import filter_data
from numpy.testing import assert_allclose, assert_array_almost_equal, assert_array_less

from mne_connectivity import (
    SpectralConnectivity,
    make_signals_in_freq_bands,
    read_connectivity,
    spectral_connectivity_epochs,
    spectral_connectivity_time,
)
from mne_connectivity.spectral.epochs import (
    _compute_freq_mask,
    _compute_freqs,
    _get_n_epochs,
)
from mne_connectivity.spectral.epochs_bivariate import _CohEst


# TODO: Replace with `make_signals_in_freq_bands` after tweaking tolerances in tests
def create_test_dataset(
    sfreq, n_signals, n_epochs, n_times, tmin, tmax, fstart, fend, trans_bandwidth=2.0
):
    """Create test dataset with no spurious correlations.

    Parameters
    ----------
    sfreq : float
        The simulated data sampling rate.
    n_signals : int
        The number of channels/signals to simulate.
    n_epochs : int
        The number of Epochs to simulate.
    n_times : int
        The number of time points at which the Epoch data is "sampled".
    tmin : int
        The start time of the Epoch data.
    tmax : int
        The end time of the Epoch data.
    fstart : int
        The frequency at which connectivity starts. The lower end of the
        spectral connectivity.
    fend : int
        The frequency at which connectivity ends. The upper end of the
        spectral connectivity.
    trans_bandwidth : int, optional
        The bandwidth of the filtering operation, by default 2.

    Returns
    -------
    data : np.ndarray of shape (n_epochs, n_signals, n_times)
        The epoched dataset.
    times_data : np.ndarray of shape (n_times, )
        The times at which each sample of the ``data`` occurs at.
    """
    # Use a case known to have no spurious correlations (it would bad if
    # tests could randomly fail):
    rng = np.random.RandomState(0)

    data = rng.randn(n_signals, n_epochs * n_times)
    times_data = np.linspace(tmin, tmax, n_times)

    # simulate connectivity from fstart to fend
    data[1, :] = filter_data(
        data[0, :],
        sfreq,
        fstart,
        fend,
        filter_length="auto",
        fir_design="firwin2",
        l_trans_bandwidth=trans_bandwidth,
        h_trans_bandwidth=trans_bandwidth,
    )
    # add some noise, so the spectrum is not exactly zero
    data[1, :] += 1e-2 * rng.randn(n_times * n_epochs)
    data = data.reshape(n_signals, n_epochs, n_times)
    data = np.transpose(data, [1, 0, 2])
    return data, times_data


def _stc_gen(data, sfreq, tmin, combo=False):
    """Simulate a SourceEstimate generator."""
    vertices = [np.arange(data.shape[1]), np.empty(0)]
    for d in data:
        if not combo:
            stc = SourceEstimate(
                data=d, vertices=vertices, tmin=tmin, tstep=1 / float(sfreq)
            )
            yield stc
        else:
            # simulate a combination of array and source estimate
            arr = d[0]
            stc = SourceEstimate(
                data=d[1:], vertices=vertices, tmin=tmin, tstep=1 / float(sfreq)
            )
            yield (arr, stc)


@pytest.mark.parametrize("method", ["coh", "cohy", "imcoh", "plv"])
@pytest.mark.parametrize("mode", ["multitaper", "fourier", "cwt_morlet"])
def test_spectral_connectivity_parallel(method, mode, tmp_path):
    """Test saving spectral connectivity with parallel functions."""
    n_jobs = 2  # test with parallelization

    data = make_signals_in_freq_bands(
        n_seeds=2,
        n_targets=1,
        freq_band=(5, 15),
        n_epochs=8,
        n_times=256,
        sfreq=50,
        trans_bandwidth=2.0,
        rng_seed=0,  # case with no spurious correlations (avoid tests randomly failing)
    )

    # define some frequencies for cwt
    cwt_freqs = np.arange(3, 24.5, 1)

    if method == "coh" and mode == "multitaper":
        # only check adaptive estimation for coh to reduce test time
        check_adaptive = [False, True]
    else:
        check_adaptive = [False]

    if method == "coh" and mode == "cwt_morlet":
        # so we also test using an array for num cycles
        cwt_n_cycles = 7.0 * np.ones(len(cwt_freqs))
    else:
        cwt_n_cycles = 7.0

    for adaptive in check_adaptive:
        if adaptive:
            mt_bandwidth = 1.0
        else:
            mt_bandwidth = None

        con = spectral_connectivity_epochs(
            data,
            method=method,
            mode=mode,
            indices=None,
            mt_adaptive=adaptive,
            mt_low_bias=True,
            mt_bandwidth=mt_bandwidth,
            cwt_freqs=cwt_freqs,
            cwt_n_cycles=cwt_n_cycles,
            n_jobs=n_jobs,
        )

        tmp_file = tmp_path / "temp_file.nc"
        con.save(tmp_file)

        read_con = read_connectivity(tmp_file)
        assert_array_almost_equal(con.get_data(), read_con.get_data())
        # split `repr` before the file size (`~23 kB` for example)
        a = repr(con).split("~")[0]
        b = repr(read_con).split("~")[0]
        assert a == b


@pytest.mark.parametrize(
    "method",
    [
        "coh",
        "cohy",
        "imcoh",
        "plv",
        [
            "ciplv",
            "ppc",
            "pli",
            "pli2_unbiased",
            "dpli",
            "wpli",
            "wpli2_debiased",
            "coh",
        ],
    ],
)
@pytest.mark.parametrize("mode", ["multitaper", "fourier", "cwt_morlet"])
def test_spectral_connectivity(method, mode):
    """Test frequency-domain connectivity methods."""
    sfreq = 50.0
    n_signals = 3
    n_epochs = 8
    n_times = 256
    trans_bandwidth = 2.0
    tmin = 0.0
    tmax = (n_times - 1) / sfreq

    # 5Hz..15Hz
    fstart, fend = 5.0, 15.0
    # TODO: Replace with `make_signals_in_freq_bands` after tweaking tolerances in tests
    data, times_data = create_test_dataset(
        sfreq,
        n_signals=n_signals,
        n_epochs=n_epochs,
        n_times=n_times,
        tmin=tmin,
        tmax=tmax,
        fstart=fstart,
        fend=fend,
        trans_bandwidth=trans_bandwidth,
    )

    # First we test some invalid parameters:
    pytest.raises(ValueError, spectral_connectivity_epochs, data, method="notamethod")
    pytest.raises(ValueError, spectral_connectivity_epochs, data, mode="notamode")

    # test invalid fmin fmax settings
    pytest.raises(
        ValueError,
        spectral_connectivity_epochs,
        data,
        fmin=10,
        fmax=10 + 0.5 * (sfreq / float(n_times)),
    )
    pytest.raises(ValueError, spectral_connectivity_epochs, data, fmin=10, fmax=5)
    pytest.raises(
        ValueError, spectral_connectivity_epochs, data, fmin=(0, 11), fmax=(5, 10)
    )
    pytest.raises(
        ValueError, spectral_connectivity_epochs, data, fmin=(11,), fmax=(12, 15)
    )

    # define some frequencies for cwt
    cwt_freqs = np.arange(3, 24.5, 1)

    if method == "coh" and mode == "multitaper":
        # only check adaptive estimation for coh to reduce test time
        check_adaptive = [False, True]
    else:
        check_adaptive = [False]

    if method == "coh" and mode == "cwt_morlet":
        # so we also test using an array for num cycles
        cwt_n_cycles = 7.0 * np.ones(len(cwt_freqs))
    else:
        cwt_n_cycles = 7.0

    for adaptive in check_adaptive:
        if adaptive:
            mt_bandwidth = 1.0
        else:
            mt_bandwidth = None

        con = spectral_connectivity_epochs(
            data,
            method=method,
            mode=mode,
            indices=None,
            sfreq=sfreq,
            mt_adaptive=adaptive,
            mt_low_bias=True,
            mt_bandwidth=mt_bandwidth,
            cwt_freqs=cwt_freqs,
            cwt_n_cycles=cwt_n_cycles,
        )

        if isinstance(method, list):
            this_con = con[0]
        else:
            this_con = con
        freqs = this_con.attrs.get("freqs_used")
        n = this_con.n_epochs_used
        if isinstance(this_con, SpectralConnectivity):
            times = this_con.attrs.get("times_used")
        else:
            times = this_con.times

        assert n == n_epochs
        assert_array_almost_equal(times_data, times)

        if mode == "multitaper":
            upper_t = 0.95
            lower_t = 0.5
        else:  # mode == 'fourier' or mode == 'cwt_morlet'
            # other estimates have higher variance
            upper_t = 0.8
            lower_t = 0.75

        # test the simulated signal
        gidx = np.searchsorted(freqs, (fstart, fend))
        bidx = np.searchsorted(
            freqs, (fstart - trans_bandwidth * 2, fend + trans_bandwidth * 2)
        )
        if method == "coh":
            assert np.all(
                con.get_data(output="dense")[1, 0, gidx[0] : gidx[1]] > upper_t
            ), con.get_data()[1, 0, gidx[0] : gidx[1]].min()
            # we see something for zero-lag
            assert_array_less(con.get_data(output="dense")[1, 0, : bidx[0]], lower_t)
            assert np.all(
                con.get_data(output="dense")[1, 0, bidx[1] :] < lower_t
            ), con.get_data()[1, 0, bidx[1:]].max()
        elif method == "cohy":
            # imaginary coh will be zero
            check = np.imag(con.get_data(output="dense")[1, 0, gidx[0] : gidx[1]])
            assert np.all(check < lower_t), check.max()
            # we see something for zero-lag
            assert_array_less(
                upper_t, np.abs(con.get_data(output="dense")[1, 0, gidx[0] : gidx[1]])
            )
            assert_array_less(
                np.abs(con.get_data(output="dense")[1, 0, : bidx[0]]), lower_t
            )
            assert_array_less(
                np.abs(con.get_data(output="dense")[1, 0, bidx[1] :]), lower_t
            )
        elif method == "imcoh":
            # imaginary coh will be zero
            assert_array_less(
                con.get_data(output="dense")[1, 0, gidx[0] : gidx[1]], lower_t
            )
            assert_array_less(con.get_data(output="dense")[1, 0, : bidx[0]], lower_t)
            assert_array_less(con.get_data(output="dense")[1, 0, bidx[1] :], lower_t)
            assert np.all(
                con.get_data(output="dense")[1, 0, bidx[1] :] < lower_t
            ), con.get_data()[1, 0, bidx[1] :].max()

        # compute a subset of connections using indices and 2 jobs
        indices = (np.array([2, 1]), np.array([0, 0]))

        if not isinstance(method, list):
            test_methods = (method, _CohEst)
        else:
            test_methods = method

        stc_data = _stc_gen(data, sfreq, tmin)
        con2 = spectral_connectivity_epochs(
            stc_data,
            method=test_methods,
            mode=mode,
            indices=indices,
            sfreq=sfreq,
            mt_adaptive=adaptive,
            mt_low_bias=True,
            mt_bandwidth=mt_bandwidth,
            tmin=tmin,
            tmax=tmax,
            cwt_freqs=cwt_freqs,
            cwt_n_cycles=cwt_n_cycles,
        )

        assert isinstance(con2, list)
        assert len(con2) == len(test_methods)
        freqs2 = con2[0].attrs.get("freqs_used")
        if "times" in con2[0].dims:
            times2 = con2[0].times
        else:
            times2 = con2[0].attrs.get("times_used")
        n2 = con2[0].n_epochs_used

        if method == "coh":
            assert_array_almost_equal(con2[0].get_data(), con2[1].get_data())

        if not isinstance(method, list):
            con2 = con2[0]  # only keep the first method

            # we get the same result for the probed connections
            assert_array_almost_equal(freqs, freqs2)

            # "con2" is a raveled array already, so
            # simulate setting indices on the full output in "con"
            assert_array_almost_equal(
                con.get_data(output="dense")[indices], con2.get_data()
            )
            assert n == n2
            assert_array_almost_equal(times_data, times2)
        else:
            # we get the same result for the probed connections
            assert len(con) == len(con2)
            for c, c2 in zip(con, con2):
                assert_array_almost_equal(freqs, freqs2)
                assert_array_almost_equal(
                    c.get_data(output="dense")[indices], c2.get_data()
                )
                assert n == n2
                assert_array_almost_equal(times_data, times2)

        # Test with faverage
        # compute same connections for two bands, fskip=1, and f. avg.
        fmin = (5.0, 15.0)
        fmax = (15.0, 30.0)
        con3 = spectral_connectivity_epochs(
            data,
            method=method,
            mode=mode,
            indices=indices,
            sfreq=sfreq,
            fmin=fmin,
            fmax=fmax,
            fskip=1,
            faverage=True,
            mt_adaptive=adaptive,
            mt_low_bias=True,
            mt_bandwidth=mt_bandwidth,
            cwt_freqs=cwt_freqs,
            cwt_n_cycles=cwt_n_cycles,
        )

        if isinstance(method, list):
            freqs3 = con3[0].attrs.get("freqs_used")
        else:
            freqs3 = con3.attrs.get("freqs_used")

        assert isinstance(freqs3, list)
        assert len(freqs3) == len(fmin)
        for i in range(len(freqs3)):
            _fmin = max(fmin[i], min(cwt_freqs))
            _fmax = min(fmax[i], max(cwt_freqs))
            assert_allclose(freqs3[i][0], _fmin, atol=1)
            assert_allclose(freqs3[i][1], _fmax, atol=1)

        # average con2 "manually" and we get the same result
        fskip = 1
        if not isinstance(method, list):
            for i in range(len(freqs3)):
                # now we want to get the frequency indices
                # create a frequency mask for all bands
                n_times = len(con2.attrs.get("times_used"))

                # compute frequencies to analyze based on number of samples,
                # sampling rate, specified wavelet frequencies and mode
                freqs = _compute_freqs(n_times, sfreq, cwt_freqs, mode)

                # compute the mask based on specified min/max and decim factor
                freq_mask = _compute_freq_mask(freqs, [fmin[i]], [fmax[i]], fskip)
                freqs = freqs[freq_mask]
                freqs_idx = np.searchsorted(freqs2, freqs)
                con2_avg = np.mean(con2.get_data()[:, freqs_idx], axis=1)
                assert_array_almost_equal(con2_avg, con3.get_data()[:, i])
        else:
            for j in range(len(con2)):
                for i in range(len(freqs3)):
                    # now we want to get the frequency indices
                    # create a frequency mask for all bands
                    n_times = len(con2[0].attrs.get("times_used"))

                    # compute frequencies to analyze based on number of
                    # samples, sampling rate, specified wavelet frequencies
                    # and mode
                    freqs = _compute_freqs(n_times, sfreq, cwt_freqs, mode)

                    # compute the mask based on specified min/max and
                    # decim factor
                    freq_mask = _compute_freq_mask(freqs, [fmin[i]], [fmax[i]], fskip)
                    freqs = freqs[freq_mask]
                    freqs_idx = np.searchsorted(freqs2, freqs)

                    con2_avg = np.mean(con2[j].get_data()[:, freqs_idx], axis=1)
                    assert_array_almost_equal(con2_avg, con3[j].get_data()[:, i])

    # test _get_n_epochs
    full_list = list(range(10))
    out_lens = np.array([len(x) for x in _get_n_epochs(full_list, 4)])
    assert (out_lens == np.array([4, 4, 2])).all()
    out_lens = np.array([len(x) for x in _get_n_epochs(full_list, 11)])
    assert len(out_lens) > 0
    assert out_lens[0] == 10


_gc_marks = []
if platform.system() == "Darwin" and platform.processor() == "arm":
    _gc_marks.extend(
        [
            pytest.mark.filterwarnings("ignore:divide by zero encountered in det:"),
            pytest.mark.filterwarnings("ignore:invalid value encountered in det:"),
        ]
    )
_gc = pytest.param("gc", marks=_gc_marks, id="gc")
_gc_tr = pytest.param("gc_tr", marks=_gc_marks, id="gc_tr")


@pytest.mark.parametrize("method", ["cacoh", "mic", "mim", _gc])
def test_spectral_connectivity_epochs_multivariate(method):
    """Test over-epoch multivariate connectivity methods."""
    mode = "multitaper"  # stick with single mode in interest of time

    sfreq = 100.0  # Hz
    n_signals = 4  # should be even!
    n_seeds = n_signals // 2
    n_epochs = 10
    n_times = 200  # samples
    trans_bandwidth = 2.0  # Hz
    delay = 10  # samples (non-zero delay needed for ImCoh and GC to be >> 0)

    indices = (
        np.arange(n_seeds)[np.newaxis, :],
        np.arange(n_seeds)[np.newaxis, :] + n_seeds,
    )
    n_targets = n_seeds

    # 15-25 Hz connectivity
    fstart, fend = 15.0, 25.0
    rng = np.random.RandomState(0)
    # TODO: Replace with `make_signals_in_freq_bands` after tweaking tolerances in tests
    data = rng.randn(n_signals, n_epochs * n_times + delay)
    # simulate connectivity from fstart to fend
    data[n_seeds:, :] = filter_data(
        data[:n_seeds, :],
        sfreq,
        fstart,
        fend,
        filter_length="auto",
        fir_design="firwin2",
        l_trans_bandwidth=trans_bandwidth,
        h_trans_bandwidth=trans_bandwidth,
    )
    # add some noise, so the spectrum is not exactly zero
    data[n_seeds:, :] += 1e-2 * rng.randn(n_seeds, n_times * n_epochs + delay)
    # shift the seeds to that the targets are a delayed version of them
    data[:n_seeds, : n_epochs * n_times] = data[:n_seeds, delay:]
    data = data[:, : n_times * n_epochs]
    data = data.reshape(n_signals, n_epochs, n_times)
    data = np.transpose(data, [1, 0, 2])

    con = spectral_connectivity_epochs(
        data, method=method, mode=mode, indices=indices, sfreq=sfreq, gc_n_lags=20
    )
    freqs = con.freqs
    gidx = (freqs.index(fstart), freqs.index(fend) + 1)
    bidx = (
        freqs.index(fstart - trans_bandwidth * 2),
        freqs.index(fend + trans_bandwidth * 2) + 1,
    )

    if method in ["cacoh", "mic", "mim"]:
        lower_t = 0.2
        upper_t = 0.5

        assert np.abs(con.get_data())[0, gidx[0] : gidx[1]].mean() > upper_t
        assert np.abs(con.get_data())[0, : bidx[0]].mean() < lower_t
        assert np.abs(con.get_data())[0, bidx[1] :].mean() < lower_t

    elif method == "gc":
        lower_t = 0.2
        upper_t = 0.8

        assert con.get_data()[0, gidx[0] : gidx[1]].mean() > upper_t
        assert con.get_data()[0, : bidx[0]].mean() < lower_t
        assert con.get_data()[0, bidx[1] :].mean() < lower_t

        # check that target -> seed connectivity is low
        indices_ts = (indices[1], indices[0])
        con_ts = spectral_connectivity_epochs(
            data,
            method=method,
            mode=mode,
            indices=indices_ts,
            sfreq=sfreq,
            gc_n_lags=20,
        )
        assert con_ts.get_data()[0, gidx[0] : gidx[1]].mean() < lower_t

        # check that TRGC is positive (i.e. net seed -> target connectivity not
        # due to noise)
        con_tr = spectral_connectivity_epochs(
            data, method="gc_tr", mode=mode, indices=indices, sfreq=sfreq, gc_n_lags=20
        )
        con_ts_tr = spectral_connectivity_epochs(
            data,
            method="gc_tr",
            mode=mode,
            indices=indices_ts,
            sfreq=sfreq,
            gc_n_lags=20,
        )
        trgc = (con.get_data() - con_ts.get_data()) - (
            con_tr.get_data() - con_ts_tr.get_data()
        )
        # checks that TRGC is positive and >> 0 (for 15-25 Hz)
        assert np.all(trgc[0, gidx[0] : gidx[1]] > 0)
        assert np.all(trgc[0, gidx[0] : gidx[1]] > upper_t)
        # checks that TRGC is ~ 0 for other frequencies
        assert np.allclose(trgc[0, : bidx[0]].mean(), 0, atol=lower_t)
        assert np.allclose(trgc[0, bidx[1] :].mean(), 0, atol=lower_t)

    # check all-to-all conn. computed for CaCoh/MIC/MIM when no indices given
    if method in ["cacoh", "mic", "mim"]:
        con = spectral_connectivity_epochs(
            data, method=method, mode=mode, indices=None, sfreq=sfreq
        )
        assert con.indices is None
        assert con.n_nodes == n_signals
        if method in ["cacoh", "mic"]:
            assert np.array(con.attrs["patterns"]).shape[2] == n_signals

    # check ragged indices padded correctly
    ragged_indices = ([[0]], [[1, 2]])
    con = spectral_connectivity_epochs(
        data, method=method, mode=mode, indices=ragged_indices, sfreq=sfreq
    )
    assert np.all(np.array(con.indices) == np.array([[[0, -1]], [[1, 2]]]))

    # check shape of CaCoh/MIC patterns
    if method in ["cacoh", "mic"]:
        for mode in ["multitaper", "cwt_morlet"]:
            con = spectral_connectivity_epochs(
                data,
                method=method,
                mode=mode,
                indices=indices,
                sfreq=sfreq,
                fmin=10,
                fmax=25,
                cwt_freqs=np.arange(10, 25),
                faverage=True,
            )

            if mode == "cwt_morlet":
                patterns_shape = (
                    (n_seeds, len(con.freqs), len(con.times)),
                    (n_targets, len(con.freqs), len(con.times)),
                )
            else:
                patterns_shape = (
                    (n_seeds, len(con.freqs)),
                    (n_targets, len(con.freqs)),
                )
            assert np.shape(con.attrs["patterns"][0][0]) == patterns_shape[0]
            assert np.shape(con.attrs["patterns"][1][0]) == patterns_shape[1]

            # only check these once for speed
            if mode == "multitaper":
                # check patterns averaged over freqs
                fmin = (5.0, 15.0)
                fmax = (15.0, 30.0)
                con = spectral_connectivity_epochs(
                    data,
                    method=method,
                    mode=mode,
                    indices=indices,
                    sfreq=sfreq,
                    fmin=fmin,
                    fmax=fmax,
                    faverage=True,
                )
                assert np.shape(con.attrs["patterns"][0][0])[1] == len(fmin)
                assert np.shape(con.attrs["patterns"][1][0])[1] == len(fmin)

                # check patterns shape matches input data, not rank
                rank = ([1], [1])
                con = spectral_connectivity_epochs(
                    data,
                    method=method,
                    mode=mode,
                    indices=indices,
                    sfreq=sfreq,
                    rank=rank,
                )
                assert np.shape(con.attrs["patterns"][0][0])[0] == n_seeds
                assert np.shape(con.attrs["patterns"][1][0])[0] == n_targets

                # check patterns padded correctly
                ragged_indices = ([[0]], [[1, 2]])
                con = spectral_connectivity_epochs(
                    data, method=method, mode=mode, indices=ragged_indices, sfreq=sfreq
                )
                patterns = np.array(con.attrs["patterns"])
                patterns_shape = (
                    (n_seeds, len(con.freqs)),
                    (n_targets, len(con.freqs)),
                )
                assert patterns[0, 0].shape == patterns_shape[0]
                assert patterns[1, 0].shape == patterns_shape[1]
                assert not np.any(np.isnan(patterns[0, 0, 0]))
                assert np.all(np.isnan(patterns[0, 0, 1]))
                assert not np.any(np.isnan(patterns[1, 0]))


# marked with _gc_marks below
def test_multivariate_spectral_connectivity_epochs_regression():
    """Test multivar. spectral connectivity over epochs for regression.

    The multivariate methods were originally implemented in MATLAB by their
    respective authors. To show that this Python implementation is identical
    and to avoid any future regressions, we compare the results of the Python
    and MATLAB implementations on some example data (randomly generated).

    As the MNE code for computing the cross-spectral density matrix is not
    available in MATLAB, the CSD matrix was computed using MNE and then loaded
    into MATLAB to compute the connectivity from the original implementations
    using the same processing settings in MATLAB and Python.

    It is therefore important that no changes are made to the settings for
    computing the CSD or the final connectivity scores!
    """
    fpath = os.path.dirname(os.path.realpath(__file__))
    data = pd.read_pickle(os.path.join(fpath, "data", "example_multivariate_data.pkl"))
    sfreq = 100
    indices = ([[0, 1]], [[2, 3]])
    methods = ["cacoh", "mic", "mim", "gc", "gc_tr"]
    con = spectral_connectivity_epochs(
        data,
        method=methods,
        indices=indices,
        mode="multitaper",
        sfreq=sfreq,
        fskip=0,
        faverage=False,
        tmin=0,
        tmax=None,
        mt_bandwidth=4,
        mt_low_bias=True,
        mt_adaptive=False,
        gc_n_lags=20,
        rank=tuple([[2], [2]]),
        n_jobs=1,
    )

    mne_results = {}
    for this_con in con:
        # must take the absolute of the MIC scores, as the MATLAB
        # implementation returns the absolute values.
        if this_con.method == "mic":
            mne_results[this_con.method] = np.abs(this_con.get_data())
        else:
            mne_results[this_con.method] = this_con.get_data()

    matlab_results = pd.read_pickle(
        os.path.join(fpath, "data", "example_multivariate_matlab_results.pkl")
    )
    for method in methods:
        assert_allclose(matlab_results[method], mne_results[method], 1e-5)


@pytest.mark.parametrize(
    "method",
    ["cacoh", "mic", "mim", _gc, _gc_tr, ["cacoh", "mic", "mim", "gc", "gc_tr"]],
)
@pytest.mark.parametrize("mode", ["multitaper", "fourier", "cwt_morlet"])
def test_multivar_spectral_connectivity_epochs_error_catch(method, mode):
    """Test error catching for multivar. freq.-domain connectivity methods."""
    sfreq = 50  # Hz
    data = make_signals_in_freq_bands(
        n_seeds=2,  # do not change!
        n_targets=2,  # do not change!
        freq_band=(10, 20),  # arbitrary for this test
        n_epochs=8,
        n_times=256,
        sfreq=sfreq,
        rng_seed=0,
    )

    indices = ([[0, 1]], [[2, 3]])
    cwt_freqs = np.arange(10, 25 + 1)

    # check bad indices without nested array caught
    with pytest.raises(
        TypeError, match="multivariate indices must contain array-likes"
    ):
        non_nested_indices = ([0, 1], [2, 3])
        spectral_connectivity_epochs(
            data, method=method, mode=mode, indices=non_nested_indices, gc_n_lags=10
        )

    # check bad indices with repeated channels caught
    with pytest.raises(
        ValueError, match="multivariate indices cannot contain repeated"
    ):
        repeated_indices = ([[0, 1, 1]], [[2, 2, 3]])
        spectral_connectivity_epochs(
            data, method=method, mode=mode, indices=repeated_indices, gc_n_lags=10
        )

    # check mixed methods caught
    with pytest.raises(ValueError, match="bivariate and multivariate connectivity"):
        if isinstance(method, str):
            mixed_methods = [method, "coh"]
        elif isinstance(method, list):
            mixed_methods = [*method, "coh"]
        spectral_connectivity_epochs(
            data, method=mixed_methods, mode=mode, indices=indices, cwt_freqs=cwt_freqs
        )

    # check bad rank args caught
    too_low_rank = ([0], [0])
    with pytest.raises(ValueError, match="ranks for seeds and targets must be"):
        spectral_connectivity_epochs(
            data,
            method=method,
            mode=mode,
            indices=indices,
            rank=too_low_rank,
            cwt_freqs=cwt_freqs,
        )
    too_high_rank = ([3], [3])
    with pytest.raises(ValueError, match="ranks for seeds and targets must be"):
        spectral_connectivity_epochs(
            data,
            method=method,
            mode=mode,
            indices=indices,
            rank=too_high_rank,
            cwt_freqs=cwt_freqs,
        )
    too_few_rank = ([], [])
    with pytest.raises(ValueError, match="rank argument must have shape"):
        spectral_connectivity_epochs(
            data,
            method=method,
            mode=mode,
            indices=indices,
            rank=too_few_rank,
            cwt_freqs=cwt_freqs,
        )
    too_much_rank = ([2, 2], [2, 2])
    with pytest.raises(ValueError, match="rank argument must have shape"):
        spectral_connectivity_epochs(
            data,
            method=method,
            mode=mode,
            indices=indices,
            rank=too_much_rank,
            cwt_freqs=cwt_freqs,
        )

    # check rank-deficient data caught
    # XXX: remove logic once support for mne<1.6 is dropped
    kwargs = dict()
    if "copy" in inspect.getfullargspec(data.get_data).kwonlyargs:
        kwargs["copy"] = False
    bad_data = data.get_data(**kwargs)
    bad_data[:, 1] = bad_data[:, 0]
    bad_data[:, 3] = bad_data[:, 2]
    assert np.all(np.linalg.matrix_rank(bad_data[:, (0, 1), :]) == 1)
    assert np.all(np.linalg.matrix_rank(bad_data[:, (2, 3), :]) == 1)
    if isinstance(method, str):
        rank_con = spectral_connectivity_epochs(
            bad_data,
            method=method,
            mode=mode,
            indices=indices,
            sfreq=sfreq,
            gc_n_lags=10,
            cwt_freqs=cwt_freqs,
        )
        assert rank_con.attrs["rank"] == ([1], [1])

    if method in ["cacoh", "mic", "mim"]:
        # check rank-deficient transformation matrix caught
        with pytest.raises(RuntimeError, match="the transformation matrix"):
            spectral_connectivity_epochs(
                bad_data,
                method=method,
                mode=mode,
                indices=indices,
                sfreq=sfreq,
                rank=([2], [2]),
                cwt_freqs=cwt_freqs,
            )

    # only check these once (e.g. only with multitaper) for speed
    if method == "gc" and mode == "multitaper":
        # check bad n_lags caught
        frange = (5, 10)
        n_lags = 200  # will be far too high
        with pytest.raises(ValueError, match="the number of lags"):
            spectral_connectivity_epochs(
                data,
                method=method,
                mode=mode,
                indices=indices,
                fmin=frange[0],
                fmax=frange[1],
                gc_n_lags=n_lags,
                cwt_freqs=cwt_freqs,
            )

        # check no indices caught
        with pytest.raises(ValueError, match="indices must be specified"):
            spectral_connectivity_epochs(
                data, method=method, mode=mode, indices=None, cwt_freqs=cwt_freqs
            )

        # check intersecting indices caught
        bad_indices = ([[0, 1]], [[0, 2]])
        with pytest.raises(
            ValueError, match="seed and target indices must not intersect"
        ):
            spectral_connectivity_epochs(
                data, method=method, mode=mode, indices=bad_indices, cwt_freqs=cwt_freqs
            )

        # check bad fmin/fmax caught
        with pytest.raises(ValueError, match="computing Granger causality on multiple"):
            spectral_connectivity_epochs(
                data,
                method=method,
                mode=mode,
                indices=indices,
                fmin=(10.0, 15.0),
                fmax=(15.0, 20.0),
                cwt_freqs=cwt_freqs,
            )

        # check rank-deficient autocovariance caught
        with pytest.raises(RuntimeError, match="the autocovariance matrix is singular"):
            spectral_connectivity_epochs(
                bad_data,
                method=method,
                mode=mode,
                indices=indices,
                sfreq=sfreq,
                rank=([2], [2]),
                cwt_freqs=cwt_freqs,
            )


@pytest.mark.parametrize("method", ["cacoh", "mic", "mim", _gc, _gc_tr])
def test_multivar_spectral_connectivity_parallel(method):
    """Test multivar. freq.-domain connectivity methods run in parallel."""
    data = make_signals_in_freq_bands(
        n_seeds=2,  # do not change!
        n_targets=2,  # do not change!
        freq_band=(10, 20),  # arbitrary for this test
        n_epochs=8,
        n_times=256,
        sfreq=50,
        rng_seed=0,
    )

    indices = ([[0, 1]], [[2, 3]])

    spectral_connectivity_epochs(
        data, method=method, mode="multitaper", indices=indices, gc_n_lags=10, n_jobs=2
    )
    spectral_connectivity_time(
        data,
        freqs=np.arange(10, 25),
        method=method,
        mode="multitaper",
        indices=indices,
        gc_n_lags=10,
        n_jobs=2,
    )


def test_multivar_spectral_connectivity_flipped_indices():
    """Test multivar. indices structure maintained by connectivity methods."""
    data = make_signals_in_freq_bands(
        n_seeds=2,  # do not change!
        n_targets=2,  # do not change!
        freq_band=(10, 20),  # arbitrary for this test
        n_epochs=8,
        n_times=256,
        sfreq=50,
        rng_seed=0,
    )

    freqs = np.arange(10, 20)

    # if we're not careful, when finding the channels we need to compute the
    # CSD for, we might accidentally reorder the connectivity indices
    indices = ([[0, 1]], [[2, 3]])
    flipped_indices = ([[2, 3]], [[0, 1]])
    concat_indices = ([[0, 1], [2, 3]], [[2, 3], [0, 1]])

    # we test on GC since this is a directed connectivity measure
    method = "gc"

    con_st = spectral_connectivity_epochs(  # seed -> target
        data, method=method, indices=indices, gc_n_lags=10
    )
    con_ts = spectral_connectivity_epochs(  # target -> seed
        data, method=method, indices=flipped_indices, gc_n_lags=10
    )
    con_st_ts = spectral_connectivity_epochs(  # seed -> target; target -> seed
        data, method=method, indices=concat_indices, gc_n_lags=10
    )
    assert not np.all(con_st.get_data() == con_ts.get_data())
    assert_allclose(con_st.get_data()[0], con_st_ts.get_data()[0])
    assert_allclose(con_ts.get_data()[0], con_st_ts.get_data()[1])

    con_st = spectral_connectivity_time(  # seed -> target
        data, freqs, method=method, indices=indices, gc_n_lags=10
    )
    con_ts = spectral_connectivity_time(  # target -> seed
        data, freqs, method=method, indices=flipped_indices, gc_n_lags=10
    )
    con_st_ts = spectral_connectivity_time(  # seed -> target; target -> seed
        data, freqs, method=method, indices=concat_indices, gc_n_lags=10
    )
    assert not np.all(con_st.get_data() == con_ts.get_data())
    assert_allclose(con_st.get_data()[:, 0], con_st_ts.get_data()[:, 0])
    assert_allclose(con_ts.get_data()[:, 0], con_st_ts.get_data()[:, 1])


@pytest.mark.parametrize("kind", ("epochs", "ndarray", "stc", "combo"))
def test_epochs_tmin_tmax(kind):
    """Test spectral.spectral_connectivity_epochs with epochs and arrays."""
    rng = np.random.RandomState(0)
    n_epochs, n_chs, n_times, sfreq, f = 10, 2, 2000, 1000.0, 20.0
    data = rng.randn(n_epochs, n_chs, n_times)
    sig = np.sin(2 * np.pi * f * np.arange(1000) / sfreq) * np.hanning(1000)
    data[:, :, 500:1500] += sig
    info = create_info(n_chs, sfreq, "eeg")
    if kind == "epochs":
        tmin = -1
        X = EpochsArray(data, info, tmin=tmin)
    elif kind == "stc":
        tmin = -1
        X = [SourceEstimate(d, [[0], [0]], tmin, 1.0 / sfreq) for d in data]
    elif kind == "combo":
        tmin = -1
        X = [
            (d[[0]], SourceEstimate(d[[1]], [[0], []], tmin, 1.0 / sfreq)) for d in data
        ]
    else:
        assert kind == "ndarray"
        tmin = 0
        X = data
    want_times = np.arange(n_times) / sfreq + tmin

    # Parameters for computing connectivity
    fmin, fmax = f - 2, f + 2
    kwargs = {
        "method": "coh",
        "mode": "multitaper",
        "sfreq": sfreq,
        "fmin": fmin,
        "fmax": fmax,
        "faverage": True,
        "mt_adaptive": False,
        "n_jobs": 1,
    }

    # Check the entire interval
    conn = spectral_connectivity_epochs(X, **kwargs)
    assert 0.89 < conn.get_data(output="dense")[1, 0] < 0.91
    assert_allclose(conn.attrs.get("times_used"), want_times)
    # Check a time interval before the sinusoid
    conn = spectral_connectivity_epochs(X, tmax=tmin + 0.5, **kwargs)
    assert 0 < conn.get_data(output="dense")[1, 0] < 0.15
    # Check a time during the sinusoid
    conn = spectral_connectivity_epochs(X, tmin=tmin + 0.5, tmax=tmin + 1.5, **kwargs)
    assert 0.93 < conn.get_data(output="dense")[1, 0] <= 0.94
    # Check a time interval after the sinusoid
    conn = spectral_connectivity_epochs(X, tmin=tmin + 1.5, tmax=tmin + 1.9, **kwargs)
    assert 0 < conn.get_data(output="dense")[1, 0] < 0.15

    # Check for warning if tmin, tmax is outside of the time limits of data
    with pytest.warns(RuntimeWarning, match="start time tmin"):
        spectral_connectivity_epochs(X, **kwargs, tmin=tmin - 0.1)

    with pytest.warns(RuntimeWarning, match="stop time tmax"):
        spectral_connectivity_epochs(X, **kwargs, tmax=tmin + 2.5)

    # make one with mismatched times
    if kind != "combo":
        return
    X = [
        (
            SourceEstimate(d[[0]], [[0], []], tmin - 1, 1.0 / sfreq),
            SourceEstimate(d[[1]], [[0], []], tmin, 1.0 / sfreq),
        )
        for d in data
    ]
    with pytest.warns(RuntimeWarning, match="time scales of input") as w:
        spectral_connectivity_epochs(X, **kwargs)
    assert len(w) == 1  # just one even though there were multiple epochs


@pytest.mark.parametrize(
    "method", ["coh", "cacoh", "mic", "mim", "plv", "pli", "wpli", "ciplv"]
)
@pytest.mark.parametrize("mode", ["cwt_morlet", "multitaper"])
@pytest.mark.parametrize("data_option", ["sync", "random"])
def test_spectral_connectivity_time_phaselocked(method, mode, data_option):
    """Test time-resolved spectral connectivity with simulated phase-locked data."""
    rng = np.random.default_rng(0)
    n_epochs = 5
    n_channels = 4
    n_times = 1000
    sfreq = 250
    data = np.zeros((n_epochs, n_channels, n_times))
    if data_option == "random":
        # Data is random, there should be no consistent phase differences.
        data = rng.random((n_epochs, n_channels, n_times))
    if data_option == "sync":
        # Data consists of phase-locked 10Hz sine waves with constant phase
        # difference within each epoch.
        wave_freq = 10
        epoch_length = n_times / sfreq
        for i in range(n_epochs):
            for c in range(n_channels):
                phase = rng.random() * 10
                x = np.linspace(
                    -wave_freq * epoch_length * np.pi + phase,
                    wave_freq * epoch_length * np.pi + phase,
                    n_times,
                )
                data[i, c] = np.squeeze(np.sin(x))

    multivar_methods = ["cacoh", "mic", "mim"]

    if method == "cacoh":
        # CaCoh within set of signals will always be 1, so need to specify
        # distinct seeds and targets
        indices = ([[0, 1]], [[2, 3]])
    else:
        indices = None

    # the frequency band should contain the frequency at which there is a
    # hypothesized "connection"
    freq_band_low_limit = 8.0
    freq_band_high_limit = 13.0
    freqs = np.arange(freq_band_low_limit, freq_band_high_limit + 1)
    con = spectral_connectivity_time(
        data,
        freqs,
        indices=indices,
        method=method,
        mode=mode,
        sfreq=sfreq,
        fmin=freq_band_low_limit,
        fmax=freq_band_high_limit,
        n_jobs=1,
        faverage=method not in ["cacoh", "mic"],
        average=method not in ["cacoh", "mic"],
        sm_times=0,
    )
    con_matrix = con.get_data()

    # CaCoh/MIC values can be pos. and neg., so must be averaged after taking
    # the absolute values for the test to work
    if method in multivar_methods:
        if method in ["cacoh", "mic"]:
            con_matrix = np.mean(np.abs(con_matrix), axis=(0, 2))
            assert con.shape == (n_epochs, 1, len(con.freqs))
        else:
            assert con.shape == (1, len(con.freqs))
    else:
        assert con.shape == (n_channels**2, len(con.freqs))
        con_matrix = np.reshape(con_matrix, (n_channels, n_channels))[
            np.tril_indices(n_channels, -1)
        ]

    if data_option == "sync":
        # signals are perfectly phase-locked, connectivity matrix should be
        # a matrix of ones
        assert np.allclose(con_matrix, np.ones(con_matrix.shape), atol=0.01)
    if data_option == "random":
        # signals are random, all connectivity values should be small
        # 0.5 is picked rather arbitrarily such that the obsolete wrong
        # implementation fails
        assert np.all(con_matrix <= 0.5)


def test_spectral_connectivity_time_delayed():
    """Test per-epoch Granger causality with time-delayed data.

    N.B.: the spectral_connectivity_time method seems to be more unstable than
    spectral_connectivity_epochs for GC estimation. Accordingly, we assess
    Granger scores only in the context of the noise-corrected TRGC metric,
    where the true directionality of the connections seems to identified.
    """
    mode = "multitaper"  # stick with single mode in interest of time

    sfreq = 100.0  # Hz
    n_signals = 4  # should be even!
    n_seeds = n_signals // 2
    n_epochs = 10
    n_times = 200  # samples
    trans_bandwidth = 2.0  # Hz
    delay = 5  # samples (non-zero delay needed for GC to be >> 0)

    indices = ([[0, 1]], [[2, 3]])

    # 20-30 Hz connectivity
    fstart, fend = 20.0, 30.0
    rng = np.random.RandomState(0)
    # TODO: Replace with `make_signals_in_freq_bands` after tweaking tolerances in tests
    data = rng.randn(n_signals, n_epochs * n_times + delay)
    # simulate connectivity from fstart to fend
    data[n_seeds:, :] = filter_data(
        data[:n_seeds, :],
        sfreq,
        fstart,
        fend,
        filter_length="auto",
        fir_design="firwin2",
        l_trans_bandwidth=trans_bandwidth,
        h_trans_bandwidth=trans_bandwidth,
    )
    # add some noise, so the spectrum is not exactly zero
    data[n_seeds:, :] += 1e-2 * rng.randn(n_seeds, n_times * n_epochs + delay)
    # shift the seeds to that the targets are a delayed version of them
    data[:n_seeds, : n_epochs * n_times] = data[:n_seeds, delay:]
    data = data[:, : n_times * n_epochs]
    data = data.reshape(n_signals, n_epochs, n_times)
    data = np.transpose(data, [1, 0, 2])

    freqs = np.arange(2.5, 50, 0.5)
    con_st = spectral_connectivity_time(
        data,
        freqs,
        method=["gc", "gc_tr"],
        indices=indices,
        mode=mode,
        sfreq=sfreq,
        n_jobs=1,
        gc_n_lags=20,
        n_cycles=5,
        average=True,
    )
    con_ts = spectral_connectivity_time(
        data,
        freqs,
        method=["gc", "gc_tr"],
        indices=(indices[1], indices[0]),
        mode=mode,
        sfreq=sfreq,
        n_jobs=1,
        gc_n_lags=20,
        n_cycles=5,
        average=True,
    )
    st = con_st[0].get_data()
    st_tr = con_st[1].get_data()
    ts = con_ts[0].get_data()
    ts_tr = con_ts[1].get_data()
    trgc = (st - ts) - (st_tr - ts_tr)

    freqs = con_st[0].freqs
    gidx = (freqs.index(fstart), freqs.index(fend) + 1)
    bidx = (
        freqs.index(fstart - trans_bandwidth * 2),
        freqs.index(fend + trans_bandwidth * 2) + 1,
    )

    # assert that TRGC (i.e. net, noise-corrected connectivity) is positive and
    # >> 0 (i.e. that there is indeed a flow of info. from  seeds to targets,
    # as simulated)
    assert np.all(trgc[:, gidx[0] : gidx[1]] > 0)
    assert trgc[:, gidx[0] : gidx[1]].mean() > 0.4
    # check that non-interacting freqs. have close to zero connectivity
    assert np.allclose(trgc[0, : bidx[0]].mean(), 0, atol=0.1)
    assert np.allclose(trgc[0, bidx[1] :].mean(), 0, atol=0.1)


@pytest.mark.parametrize("method", ["coh", "plv", "pli", "wpli", "ciplv"])
@pytest.mark.parametrize("freqs", [[8.0, 10.0], [8, 10], 10.0, 10])
@pytest.mark.parametrize("mode", ["cwt_morlet", "multitaper"])
def test_spectral_connectivity_time_freqs(method, freqs, mode):
    """Test time-resolved spectral connectivity with int and float values for freqs."""
    rng = np.random.default_rng(0)
    n_epochs = 5
    n_channels = 3
    n_times = 1000
    sfreq = 250
    data = np.zeros((n_epochs, n_channels, n_times))

    # Data consists of phase-locked 10Hz sine waves with constant phase
    # difference within each epoch.
    wave_freq = 10
    epoch_length = n_times / sfreq
    for i in range(n_epochs):
        for c in range(n_channels):
            phase = rng.random() * 10
            x = np.linspace(
                -wave_freq * epoch_length * np.pi + phase,
                wave_freq * epoch_length * np.pi + phase,
                n_times,
            )
            data[i, c] = np.squeeze(np.sin(x))
    # the frequency band should contain the frequency at which there is a
    # hypothesized "connection"
    con = spectral_connectivity_time(
        data,
        freqs,
        method=method,
        mode=mode,
        sfreq=sfreq,
        fmin=np.min(freqs),
        fmax=np.max(freqs),
        n_jobs=1,
        faverage=True,
        average=True,
        sm_times=0,
    )
    assert con.shape == (n_channels**2, len(con.freqs))
    con_matrix = con.get_data("dense")[..., 0]

    # signals are perfectly phase-locked, connectivity matrix should be
    # a lower triangular matrix of ones
    assert np.allclose(con_matrix, np.tril(np.ones(con_matrix.shape), k=-1), atol=0.01)


@pytest.mark.parametrize("method", ["coh", "plv", "pli", "wpli"])
@pytest.mark.parametrize("mode", ["cwt_morlet", "multitaper"])
def test_spectral_connectivity_time_resolved(method, mode):
    """Test time-resolved spectral connectivity."""
    sfreq = 50.0
    n_signals = 3
    n_epochs = 2
    n_times = 1000
    trans_bandwidth = 2.0
    tmin = 0.0
    tmax = (n_times - 1) / sfreq
    # 5Hz..15Hz
    fstart, fend = 5.0, 15.0
    # TODO: Replace with `make_signals_in_freq_bands` after tweaking tolerances in tests
    data, _ = create_test_dataset(
        sfreq,
        n_signals=n_signals,
        n_epochs=n_epochs,
        n_times=n_times,
        tmin=tmin,
        tmax=tmax,
        fstart=fstart,
        fend=fend,
        trans_bandwidth=trans_bandwidth,
    )
    ch_names = np.arange(n_signals).astype(str).tolist()
    info = create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    data = EpochsArray(data, info)

    # define some frequencies for tfr
    freqs = np.arange(3, 20.5, 1)

    # run connectivity estimation
    con = spectral_connectivity_time(
        data, freqs, sfreq=sfreq, method=method, mode=mode, n_cycles=5
    )
    assert con.shape == (n_epochs, n_signals**2, len(con.freqs))
    assert con.get_data(output="dense").shape == (
        n_epochs,
        n_signals,
        n_signals,
        len(con.freqs),
    )

    # test the simulated signal
    triu_inds = np.vstack(np.triu_indices(n_signals, k=1)).T

    # average over frequencies
    conn_data = con.get_data(output="dense").mean(axis=-1)

    # the indices at which there is a correlation should be greater
    # then the rest of the components
    for epoch_idx in range(n_epochs):
        high_conn_val = conn_data[epoch_idx, 0, 1]
        assert all(
            high_conn_val >= conn_data[epoch_idx, idx, jdx] for idx, jdx in triu_inds
        )


@pytest.mark.parametrize("method", ["coh", "plv", "pli", "wpli"])
@pytest.mark.parametrize("mode", ["cwt_morlet", "multitaper"])
@pytest.mark.parametrize("padding", [0, 1, 5])
def test_spectral_connectivity_time_padding(method, mode, padding):
    """Test time-resolved spectral connectivity with padding."""
    sfreq = 50.0
    n_signals = 3
    n_epochs = 2
    n_times = 300
    trans_bandwidth = 2.0
    tmin = 0.0
    tmax = (n_times - 1) / sfreq
    # 5Hz..15Hz
    fstart, fend = 5.0, 15.0
    # TODO: Replace with `make_signals_in_freq_bands` after tweaking tolerances in tests
    data, _ = create_test_dataset(
        sfreq,
        n_signals=n_signals,
        n_epochs=n_epochs,
        n_times=n_times,
        tmin=tmin,
        tmax=tmax,
        fstart=fstart,
        fend=fend,
        trans_bandwidth=trans_bandwidth,
    )
    ch_names = np.arange(n_signals).astype(str).tolist()
    info = create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    data = EpochsArray(data, info)

    # define some frequencies for tfr
    freqs = np.arange(3, 20.5, 1)

    # run connectivity estimation
    if padding == 5:
        with pytest.raises(
            ValueError, match="Padding cannot be larger than " "half of data length"
        ):
            con = spectral_connectivity_time(
                data,
                freqs,
                sfreq=sfreq,
                method=method,
                mode=mode,
                n_cycles=5,
                padding=padding,
            )
        return
    else:
        con = spectral_connectivity_time(
            data,
            freqs,
            sfreq=sfreq,
            method=method,
            mode=mode,
            n_cycles=5,
            padding=padding,
        )

    assert con.shape == (n_epochs, n_signals**2, len(con.freqs))
    assert con.get_data(output="dense").shape == (
        n_epochs,
        n_signals,
        n_signals,
        len(con.freqs),
    )

    # test the simulated signal
    triu_inds = np.vstack(np.triu_indices(n_signals, k=1)).T

    # average over frequencies
    conn_data = con.get_data(output="dense").mean(axis=-1)

    # the indices at which there is a correlation should be greater
    # then the rest of the components
    for epoch_idx in range(n_epochs):
        high_conn_val = conn_data[epoch_idx, 0, 1]
        assert all(
            high_conn_val >= conn_data[epoch_idx, idx, jdx] for idx, jdx in triu_inds
        )


@pytest.mark.parametrize("method", ["cacoh", "mic", "mim", _gc, _gc_tr])
@pytest.mark.parametrize("average", [True, False])
@pytest.mark.parametrize("faverage", [True, False])
def test_multivar_spectral_connectivity_time_shapes(method, average, faverage):
    """Test result shapes of time-resolved multivar. connectivity methods."""
    n_epochs = 8
    data = make_signals_in_freq_bands(
        n_seeds=2,  # do not change!
        n_targets=2,  # do not change!
        freq_band=(10, 20),  # arbitrary for this test
        n_epochs=n_epochs,
        n_times=256,
        sfreq=50,
        rng_seed=0,
    )

    indices = ([[0, 1]], [[2, 3]])
    n_cons = len(indices[0])
    freqs = np.arange(10, 25 + 1)

    con_shape = [1]
    if faverage:
        con_shape.append(1)
    else:
        con_shape.append(len(freqs))
    if not average:
        con_shape = [n_epochs, *con_shape]

    # check shape of results when averaging across epochs
    con = spectral_connectivity_time(
        data,
        freqs,
        indices=indices,
        method=method,
        faverage=faverage,
        average=average,
        gc_n_lags=10,
    )
    assert con.shape == tuple(con_shape)

    # check shape of CaCoh/MIC patterns are correct
    if method in ["cacoh", "mic"]:
        for indices_type in ["full", "ragged"]:
            if indices_type == "full":
                indices = ([[0, 1]], [[2, 3]])
            else:
                indices = ([[0, 1]], [[2]])
            max_n_chans = 2
            patterns_shape = [n_cons, max_n_chans]
            if faverage:
                patterns_shape.append(1)
            else:
                patterns_shape.append(len(freqs))
            if not average:
                patterns_shape = [n_epochs, *patterns_shape]
            patterns_shape = [2, *patterns_shape]

            con = spectral_connectivity_time(
                data,
                freqs,
                indices=indices,
                method=method,
                faverage=faverage,
                average=average,
                gc_n_lags=10,
            )

            patterns = np.array(con.attrs["patterns"])
            # 2 (x epochs) x cons x channels x freqs|fbands
            assert patterns.shape == tuple(patterns_shape)
            if indices_type == "ragged":
                assert not np.any(np.isnan(patterns[0, ..., :, :]))
                assert not np.any(np.isnan(patterns[0, ..., 0, :]))
                assert np.all(np.isnan(patterns[1, ..., 1, :]))  # padded entry
                assert np.all(np.array(con.indices) == np.array(([[0, 1]], [[2, -1]])))


@pytest.mark.parametrize("method", ["cacoh", "mic", "mim", _gc, _gc_tr])
@pytest.mark.parametrize("mode", ["multitaper", "cwt_morlet"])
def test_multivar_spectral_connectivity_time_error_catch(method, mode):
    """Test error catching for time-resolved multivar. connectivity methods."""
    n_seeds = 2  # do not change!
    n_targets = 2  # do not change!
    n_signals = n_seeds + n_targets
    data = make_signals_in_freq_bands(
        n_seeds=n_seeds,
        n_targets=n_targets,
        freq_band=(10, 20),  # arbitrary for this test
        n_epochs=8,
        n_times=256,
        sfreq=50,
        rng_seed=0,
    )

    indices = ([[0, 1]], [[2, 3]])
    freqs = np.arange(10, 25 + 1)

    # test type-checking of data
    with pytest.raises(TypeError, match="must be an instance of Epochs or a NumPy arr"):
        spectral_connectivity_time(data="foo", freqs=freqs)

    # check bad indices without nested array caught
    with pytest.raises(
        TypeError, match="multivariate indices must contain array-likes"
    ):
        non_nested_indices = ([0, 1], [2, 3])
        spectral_connectivity_time(
            data, freqs, method=method, mode=mode, indices=non_nested_indices
        )

    # check bad indices with repeated channels caught
    with pytest.raises(
        ValueError, match="multivariate indices cannot contain repeated"
    ):
        repeated_indices = ([[0, 1, 1]], [[2, 2, 3]])
        spectral_connectivity_time(
            data, freqs, method=method, mode=mode, indices=repeated_indices
        )

    # check mixed methods caught
    with pytest.raises(ValueError, match="bivariate and multivariate connectivity"):
        mixed_methods = [method, "coh"]
        spectral_connectivity_time(
            data, freqs, method=mixed_methods, mode=mode, indices=indices
        )

    # check bad rank args caught
    too_low_rank = ([0], [0])
    with pytest.raises(ValueError, match="ranks for seeds and targets must be"):
        spectral_connectivity_time(
            data, freqs, method=method, indices=indices, mode=mode, rank=too_low_rank
        )
    too_high_rank = ([3], [3])
    with pytest.raises(ValueError, match="ranks for seeds and targets must be"):
        spectral_connectivity_time(
            data, freqs, method=method, indices=indices, mode=mode, rank=too_high_rank
        )
    too_few_rank = ([], [])
    with pytest.raises(ValueError, match="rank argument must have shape"):
        spectral_connectivity_time(
            data, freqs, method=method, indices=indices, mode=mode, rank=too_few_rank
        )
    too_much_rank = ([2, 2], [2, 2])
    with pytest.raises(ValueError, match="rank argument must have shape"):
        spectral_connectivity_time(
            data, freqs, method=method, indices=indices, mode=mode, rank=too_much_rank
        )

    # check all-to-all conn. computed for CaCoh/MIC/MIM when no indices given
    if method in ["cacoh", "mic", "mim"]:
        con = spectral_connectivity_time(
            data, freqs, method=method, indices=None, mode=mode
        )
        assert con.indices is None
        assert con.n_nodes == n_signals
        if method in ["cacoh", "mic"]:
            assert np.array(con.attrs["patterns"]).shape[3] == n_signals

    if method in ["gc", "gc_tr"]:
        # check no indices caught
        with pytest.raises(ValueError, match="indices must be specified"):
            spectral_connectivity_time(
                data, freqs, method=method, mode=mode, indices=None
            )

        # check intersecting indices caught
        bad_indices = ([[0, 1]], [[0, 2]])
        with pytest.raises(
            ValueError, match="seed and target indices must not intersect"
        ):
            spectral_connectivity_time(
                data, freqs, method=method, mode=mode, indices=bad_indices
            )

        # check bad fmin/fmax caught
        with pytest.raises(ValueError, match="computing Granger causality on multiple"):
            spectral_connectivity_time(
                data,
                freqs,
                method=method,
                mode=mode,
                indices=indices,
                fmin=(5.0, 15.0),
                fmax=(15.0, 30.0),
            )


def test_save(tmp_path):
    """Test saving results of spectral connectivity."""
    epochs = make_signals_in_freq_bands(
        n_seeds=2,
        n_targets=1,
        freq_band=(18, 22),  # arbitrary for this test
        n_epochs=10,
        n_times=2000,
        sfreq=1000,
        rng_seed=0,
    )

    conn = spectral_connectivity_epochs(
        epochs, fmin=(4, 8, 13, 30), fmax=(8, 13, 30, 45), faverage=True
    )
    conn.save(tmp_path / "foo.nc")


# marked with _gc_marks below
def test_multivar_save_load(tmp_path):
    """Test saving and loading results of multivariate connectivity."""
    epochs = make_signals_in_freq_bands(
        n_seeds=2,
        n_targets=2,
        freq_band=(18, 22),  # arbitrary for this test
        n_epochs=5,
        n_times=2000,
        sfreq=1000,
        rng_seed=0,
    )

    tmp_file = os.path.join(tmp_path, "foo_mvc.nc")

    non_ragged_indices = ([[0, 1]], [[2, 3]])
    ragged_indices = ([[0, 1]], [[2]])
    for indices in [non_ragged_indices, ragged_indices]:
        con = spectral_connectivity_epochs(
            epochs,
            method=["cacoh", "mic", "mim", "gc", "gc_tr"],
            indices=indices,
            fmin=10,
            fmax=30,
        )
        for this_con in con:
            this_con.save(tmp_file)
            read_con = read_connectivity(tmp_file)
            assert_array_almost_equal(this_con.get_data(), read_con.get_data("raveled"))
            if this_con.attrs["patterns"] is not None:
                assert_array_almost_equal(
                    np.array(this_con.attrs["patterns"]),
                    np.array(read_con.attrs["patterns"]),
                )
            # split `repr` before the file size (`~23 kB` for example)
            a = repr(this_con).split("~")[0]
            b = repr(read_con).split("~")[0]
            assert a == b


@pytest.mark.parametrize("method", ["coh", "plv", "pli", "wpli", "ciplv"])
@pytest.mark.parametrize("indices", [None, ([0, 1], [2, 3])])
def test_spectral_connectivity_indices_roundtrip_io(tmp_path, method, indices):
    """Test that indices values and type is maintained after saving.

    If `indices` is None, `indices` in the returned connectivity object should
    be None, otherwise, `indices` should be a tuple. The type of `indices` and
    its values should be retained after saving and reloading.
    """
    epochs = make_signals_in_freq_bands(
        n_seeds=2,
        n_targets=2,
        freq_band=(18, 22),  # arbitrary for this test
        n_epochs=10,
        n_times=200,
        sfreq=100,
        rng_seed=0,
    )

    freqs = np.arange(10, 31)
    tmp_file = os.path.join(tmp_path, "foo_mvc.nc")

    # test the pair of method and indices defined to check the output indices
    con_epochs = spectral_connectivity_epochs(
        epochs, method=method, indices=indices, fmin=10, fmax=30
    )
    con_time = spectral_connectivity_time(epochs, freqs, method=method, indices=indices)

    for con in [con_epochs, con_time]:
        con.save(tmp_file)
        read_con = read_connectivity(tmp_file)

        if indices is not None:
            # check indices of same type (tuples)
            assert isinstance(con.indices, tuple) and isinstance(
                read_con.indices, tuple
            )
            # check indices have same values
            assert np.all(np.array(con.indices) == np.array(read_con.indices))
        else:
            assert con.indices is None and read_con.indices is None


@pytest.mark.parametrize("method", ["cacoh", "mic", "mim", _gc, _gc_tr])
@pytest.mark.parametrize("indices", [None, ([[0, 1]], [[2, 3]])])
def test_multivar_spectral_connectivity_indices_roundtrip_io(tmp_path, method, indices):
    """Test that indices values and type is maintained after saving.

    If `indices` is None, `indices` in the returned connectivity object should
    be None, otherwise, `indices` should be a tuple. The type of `indices` and
    its values should be retained after saving and reloading.
    """
    epochs = make_signals_in_freq_bands(
        n_seeds=2,
        n_targets=2,
        freq_band=(18, 22),  # arbitrary for this test
        n_epochs=10,
        n_times=200,
        sfreq=100,
        rng_seed=0,
    )

    freqs = np.arange(10, 31)
    tmp_file = os.path.join(tmp_path, "foo_mvc.nc")

    # test the pair of method and indices defined to check the output indices
    if indices is None and method in ["gc", "gc_tr"]:
        # indicesmust be specified for GC
        pytest.skip()

    con_epochs = spectral_connectivity_epochs(
        epochs, method=method, indices=indices, fmin=10, fmax=30, gc_n_lags=10
    )
    con_time = spectral_connectivity_time(
        epochs, freqs, method=method, indices=indices, gc_n_lags=10
    )

    for con in [con_epochs, con_time]:
        con.save(tmp_file)
        read_con = read_connectivity(tmp_file)

        if indices is not None:
            # check indices of same type (tuples)
            assert isinstance(con.indices, tuple) and isinstance(
                read_con.indices, tuple
            )
            # check indices are masked
            assert all(
                [np.ma.isMA(inds) for inds in con.indices]
                and [np.ma.isMA(inds) for inds in read_con.indices]
            )
            # check indices have same values
            assert np.all(
                [
                    con_inds == read_inds
                    for con_inds, read_inds in zip(con.indices, read_con.indices)
                ]
            )
        else:
            assert con.indices is None and read_con.indices is None


for _mark in _gc_marks:
    test_multivar_save_load = _mark(test_multivar_save_load)
    test_multivariate_spectral_connectivity_epochs_regression = _mark(
        test_multivariate_spectral_connectivity_epochs_regression
    )
    test_spectral_connectivity_time_delayed = _mark(
        test_spectral_connectivity_time_delayed
    )
    test_multivar_spectral_connectivity_flipped_indices = _mark(
        test_multivar_spectral_connectivity_flipped_indices
    )
