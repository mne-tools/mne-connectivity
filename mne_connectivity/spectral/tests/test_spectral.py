import inspect
import os
import platform

import numpy as np
import pandas as pd
import pytest
from mne import EpochsArray, SourceEstimate, create_info
from mne.filter import filter_data
from mne.utils import check_version
from numpy.testing import assert_allclose, assert_array_almost_equal, assert_array_less

from mne_connectivity import (
    SpectralConnectivity,
    SpectroTemporalConnectivity,
    make_signals_in_freq_bands,
    read_connectivity,
    seed_target_indices,
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
    data : array of shape (n_epochs, n_signals, n_times)
        The epoched dataset.
    times_data : array of shape (n_times, )
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
            assert np.all(con.get_data(output="dense")[1, 0, bidx[1] :] < lower_t), (
                con.get_data()[1, 0, bidx[1:]].max()
            )
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
            assert np.all(con.get_data(output="dense")[1, 0, bidx[1] :] < lower_t), (
                con.get_data()[1, 0, bidx[1] :].max()
            )

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


@pytest.mark.skipif(
    not check_version("mne", "1.10"), reason="Requires MNE v1.10.0 or higher"
)  # Taper weights in TFR objects added in MNE v1.10.0
@pytest.mark.parametrize("method", ["coh", "cacoh"])
@pytest.mark.parametrize(
    "mode, spectra_as_tfr",
    [
        ("multitaper", False),  # test multitaper in normal...
        ("multitaper", True),  # ... and TFR mode
        ("fourier", False),
        ("cwt_morlet", True),
    ],
)
def test_spectral_connectivity_epochs_spectrum_tfr_input(method, mode, spectra_as_tfr):
    """Test spec_conn_epochs works with EpochsSpectrum/TFR data as input.

    Important to test both bivariate and multivariate methods, as the latter involves
    additional steps (e.g., rank computation).

    Since spec_conn_epochs doesn't have a way to compute multitaper TFR from timeseries
    data, we can't compare the results, but we can check that the connectivity values
    are in an expected range.
    """
    # Simulation parameters & data generation
    sfreq = 100.0  # Hz
    n_seeds = 2
    n_targets = 2
    fband = (15, 20)  # Hz
    n_epochs = 30
    n_times = 200  # samples
    trans_bandwidth = 1.0  # Hz
    delay = 5  # samples

    data = make_signals_in_freq_bands(
        n_seeds=n_seeds,
        n_targets=n_targets,
        freq_band=fband,
        n_epochs=n_epochs,
        n_times=n_times,
        sfreq=sfreq,
        trans_bandwidth=trans_bandwidth,
        snr=0.7,
        connection_delay=delay,
        rng_seed=44,
    )

    if method == "coh":
        indices = seed_target_indices(
            seeds=np.arange(n_seeds), targets=np.arange(n_targets) + n_seeds
        )
    else:
        indices = ([np.arange(n_seeds)], [np.arange(n_targets) + n_seeds])

    # Compute spectral coefficients
    tfr_freqs = np.arange(10, 50)  # similar to Fourier & multitaper modes
    kwargs = dict()
    if mode == "fourier":
        kwargs.update(window="hann")  # default is Hamming, but we need Hanning
        spec_mode = "welch"
    elif mode == "cwt_morlet":
        kwargs.update(freqs=tfr_freqs)
        spec_mode = "morlet"
    else:  # multitaper
        if spectra_as_tfr:
            kwargs.update(freqs=tfr_freqs)
        spec_mode = mode
    compute_coeffs_method = data.compute_tfr if spectra_as_tfr else data.compute_psd
    coeffs = compute_coeffs_method(method=spec_mode, output="complex", **kwargs)

    # Compute connectivity
    con = spectral_connectivity_epochs(data=coeffs, method=method, indices=indices)

    # Check connectivity classes are correct and that freqs/times match input data
    if spectra_as_tfr:
        assert isinstance(con, SpectroTemporalConnectivity), "wrong class type"
        assert np.all(con.times == coeffs.times), "times do not match input data"
    else:
        assert isinstance(con, SpectralConnectivity), "wrong class type"
    assert np.all(con.freqs == coeffs.freqs), "freqs do not match input data"

    # Check connectivity from Epochs and EpochsSpectrum/TFR are equivalent
    if mode == "multitaper" and spectra_as_tfr:
        pass  # no multitaper TFR computation from timeseries in spec_conn_epochs
    else:
        con_from_epochs = spectral_connectivity_epochs(
            data=data, method=method, indices=indices, mode=mode, cwt_freqs=tfr_freqs
        )
        # Works for multitaper & Morlet, but Welch of Spectrum and Fourier of spec_conn
        # are slightly off (max. abs. diff. ~0.006). This is due to the Spectrum object
        # using scipy.signal.spectrogram to compute the coefficients, while spec_conn
        # uses scipy.signal.rfft, which give slightly different outputs even with
        # identical settings.
        if mode == "fourier":
            atol = 7e-3
        else:
            atol = 0
        # spec_conn_epochs excludes freqs without at least 5 cycles, but not Spectrum
        fstart = con.freqs.index(con_from_epochs.freqs[0])
        assert_allclose(
            np.abs(con.get_data()[:, fstart:]),
            np.abs(con_from_epochs.get_data()),
            atol=atol,
        )

    # Check connectivity values are as expected
    freqs = np.array(con.freqs)
    freqs_con = (freqs >= fband[0]) & (freqs <= fband[1])
    freqs_noise = (freqs < fband[0] - trans_bandwidth * 2) | (
        freqs > fband[1] + trans_bandwidth * 2
    )
    WEAK_CONN_OR_NOISE = 0.3  # conn values outside of simulated fband should be < this
    STRONG_CONN = 0.6  # conn values inside simulated fband should be > this
    # check freqs of simulated interaction show strong connectivity
    assert_array_less(STRONG_CONN, np.abs(con.get_data()[:, freqs_con].mean()))
    # check freqs of no simulated interaction (just noise) show weak connectivity
    assert_array_less(np.abs(con.get_data()[:, freqs_noise].mean()), WEAK_CONN_OR_NOISE)


# TODO: Add general test for error catching for spec_conn_epochs
@pytest.mark.skipif(
    not check_version("mne", "1.10"), reason="Requires MNE v1.10.0 or higher"
)  # Taper weights in TFR objects added in MNE v1.10.0
def test_spectral_connectivity_epochs_spectrum_tfr_input_error_catch():
    """Test spec_conn_epochs catches errors with EpochsSpectrum/TFR data as input."""
    # Generate data
    rng = np.random.default_rng(44)
    n_epochs, n_chans, n_times = (5, 2, 50)
    sfreq = 50
    data = rng.random((n_epochs, n_chans, n_times))
    info = create_info(ch_names=n_chans, sfreq=sfreq, ch_types="eeg")
    data = EpochsArray(data=data, info=info)

    # Test not Fourier coefficients caught
    with pytest.raises(TypeError, match="must contain complex-valued Fourier coeff"):
        spectrum = data.compute_psd(output="power")
        spectral_connectivity_epochs(data=spectrum)
    with pytest.raises(TypeError, match="must contain complex-valued Fourier coeff"):
        tfr = data.compute_tfr(method="morlet", freqs=np.arange(15, 20), output="power")
        spectral_connectivity_epochs(data=tfr)

    # Test unaggregated segments caught
    with pytest.raises(ValueError, match=r"cannot contain Fourier coeff.*segments"):
        spectrum = data.compute_psd(method="welch", average=False, output="complex")
        spectral_connectivity_epochs(data=spectrum)

    # Simulate missing weights attr in EpochsSpectrum/TFR object
    spectrum = data.compute_psd(method="multitaper", output="complex")
    with pytest.raises(AttributeError, match="weights are required for multitaper"):
        spectrum_copy = spectrum.copy()
        del spectrum_copy._weights
        spectral_connectivity_epochs(data=spectrum_copy)
    with pytest.raises(AttributeError, match="weights are required for multitaper"):
        spectrum._weights = None
        spectral_connectivity_epochs(data=spectrum)


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
@pytest.mark.parametrize("n_components", [1, 2, 3, None])
def test_spectral_connectivity_epochs_multivariate(method, n_components):
    """Test over-epoch multivariate connectivity methods."""
    if method == "gc" and n_components != 1:
        return  # GC only supports n_components=1, so don't bother running otherwise

    mode = "multitaper"  # stick with single mode in interest of time
    gc_n_lags = 20  # reduce GC compute time

    sfreq = 100.0  # Hz
    n_seeds = 3
    n_targets = 4
    n_signals = n_seeds + n_targets
    fstart = 15  # Hz
    fend = 20  # Hz
    n_epochs = 60
    n_times = 200  # samples
    trans_bandwidth = 1.0  # Hz
    delay = 10  # samples (non-zero delay needed for ImCoh and GC to be >> 0)

    indices = (
        np.arange(n_seeds)[np.newaxis, :],
        np.arange(n_targets)[np.newaxis, :] + n_seeds,
    )

    # Simulate connectivity
    data = make_signals_in_freq_bands(
        n_seeds=n_seeds,
        n_targets=n_targets,
        freq_band=(fstart, fend),
        n_epochs=n_epochs,
        n_times=n_times,
        sfreq=sfreq,
        trans_bandwidth=trans_bandwidth,
        snr=0.7,
        connection_delay=delay,
        rng_seed=44,
    )

    # Compute connectivity (only 1 component)
    con = spectral_connectivity_epochs(
        data, method=method, mode=mode, indices=indices, gc_n_lags=gc_n_lags
    )
    # Frequencies of interest
    freqs = np.array(con.freqs)
    freqs_con = (freqs >= fstart) & (freqs <= fend)
    freqs_noise = (freqs < fstart - trans_bandwidth * 2) | (
        freqs > fend + trans_bandwidth * 2
    )

    # Check connectivity scores are in expected range
    if method in ["cacoh", "mic", "mim"]:
        if method in ["cacoh", "mic"]:
            lower_t = 0.2
            upper_t = 0.5
        if method == "mim":  # MIM will have lower strength
            lower_t = 0.1
            upper_t = 0.3

        assert np.abs(con.get_data())[0, freqs_con].mean() > upper_t
        assert np.abs(con.get_data())[0, freqs_noise].mean() < lower_t

    elif method == "gc":
        lower_t = 0.2
        upper_t = 0.8

        assert con.get_data()[0, freqs_con].mean() > upper_t
        assert con.get_data()[0, freqs_noise].mean() < lower_t

        # check that target -> seed connectivity is low
        indices_ts = (indices[1], indices[0])
        con_ts = spectral_connectivity_epochs(
            data, method=method, mode=mode, indices=indices_ts, gc_n_lags=gc_n_lags
        )
        assert con_ts.get_data()[0, freqs_con].mean() < lower_t

        # check that TRGC is positive (i.e. net seed -> target connectivity not
        # due to noise)
        con_tr = spectral_connectivity_epochs(
            data, method="gc_tr", mode=mode, indices=indices, gc_n_lags=gc_n_lags
        )
        con_ts_tr = spectral_connectivity_epochs(
            data, method="gc_tr", mode=mode, indices=indices_ts, gc_n_lags=gc_n_lags
        )
        trgc = (con.get_data() - con_ts.get_data()) - (
            con_tr.get_data() - con_ts_tr.get_data()
        )
        # checks that TRGC is >> 0 (for simulated range)
        assert np.all(trgc[0, freqs_con] > upper_t)
        # checks that TRGC is ~ 0 for other frequencies
        assert np.allclose(trgc[0, freqs_noise].mean(), 0, atol=lower_t)

    # check all-to-all conn. computed for CaCoh/MIC/MIM when no indices given
    if method in ["cacoh", "mic", "mim"]:
        con = spectral_connectivity_epochs(
            data, method=method, mode=mode, indices=None, gc_n_lags=gc_n_lags
        )
        assert con.indices is None
        assert con.n_nodes == n_signals
        if method in ["cacoh", "mic"]:
            assert np.array(con.attrs["patterns"]).shape[2] == n_signals

    # check ragged indices padded correctly
    ragged_indices = ([[0]], [[1, 2]])
    con = spectral_connectivity_epochs(
        data, method=method, mode=mode, indices=ragged_indices
    )
    assert np.all(np.array(con.indices) == np.array([[[0, -1]], [[1, 2]]]))

    # check shape of results
    conn_kwargs = dict(
        gc_n_lags=gc_n_lags,
        fmin=10,
        fmax=25,
        cwt_freqs=np.arange(10, 25),
        cwt_n_cycles=4,
        n_components=n_components,
    )
    for mode in ["fourier", "multitaper", "cwt_morlet"]:
        con = spectral_connectivity_epochs(
            data, method=method, mode=mode, indices=indices, **conn_kwargs
        )

        conn_shape = (len(indices[0]), len(con.freqs))
        patterns_shape = (
            2,
            len(indices[0]),
            np.max((n_seeds, n_targets)),
            len(con.freqs),
        )
        if mode == "cwt_morlet":
            conn_shape = (*conn_shape, len(con.times))
            patterns_shape = (*patterns_shape, len(con.times))

        if n_components != 1 and method in ["cacoh", "mic"]:
            if n_components is None:
                actual_n_components = np.min((n_seeds, n_targets))
            else:
                actual_n_components = n_components
            conn_shape = (conn_shape[0], actual_n_components, *conn_shape[1:])
            patterns_shape = (
                *patterns_shape[:2],
                actual_n_components,
                *patterns_shape[2:],
            )
        else:
            actual_n_components = 1

        assert con.get_data().shape == conn_shape
        if method in ["cacoh", "mic"]:
            assert np.shape(con.attrs["patterns"]) == patterns_shape

        if method in ["cacoh", "mic"]:
            # check patterns shape matches input data, not rank
            con = spectral_connectivity_epochs(
                data,
                method=method,
                mode=mode,
                indices=indices,
                rank=([actual_n_components], [actual_n_components]),
                **conn_kwargs,
            )
            assert np.shape(con.attrs["patterns"]) == patterns_shape

            # check patterns padded correctly
            if actual_n_components <= 2:  # can't test if n_comps > rank
                ragged_indices = ([[0, 1]], [[3, 4, 5, 6]])  # seeds should be padded
                con = spectral_connectivity_epochs(
                    data,
                    method=method,
                    mode=mode,
                    indices=ragged_indices,
                    fmin=10,
                    fmax=25,
                    cwt_freqs=np.arange(10, 25),
                    cwt_n_cycles=4,
                    gc_n_lags=gc_n_lags,
                    n_components=n_components,
                )
                patterns = np.array(con.attrs["patterns"])
                assert patterns.shape == patterns_shape
                if n_components == 1:
                    assert not np.any(np.isnan(patterns[0, :, :2]))  # seeds 1-2 present
                    assert np.all(np.isnan(patterns[0, :, 2:]))  # seeds 3-4 padded
                    assert not np.any(np.isnan(patterns[1, :, :]))  # targs 1-4 present
                else:
                    assert not np.any(np.isnan(patterns[0, :, :, :2]))  # s 1-2 present
                    assert np.all(np.isnan(patterns[0, :, :, 2:]))  # s 3-4 padded
                    assert not np.any(np.isnan(patterns[1, :, :, :]))  # t 1-4 present

        # check results averaged over freqs
        if method == "gc":  # multiple freq bands not supported for GC
            fmin = (5.0,)
            fmax = (30.0,)
        else:
            fmin = (5.0, 15.0)
            fmax = (15.0, 30.0)
        con = spectral_connectivity_epochs(
            data,
            method=method,
            mode=mode,
            indices=indices,
            fmin=fmin,
            fmax=fmax,
            faverage=True,
            cwt_freqs=np.arange(3, 35),
            cwt_n_cycles=4,
            gc_n_lags=gc_n_lags,
            n_components=n_components,
        )

        conn_shape = list(conn_shape)
        patterns_shape = list(patterns_shape)
        if mode == "cwt_morlet":
            freq_dim = -2
        else:
            freq_dim = -1
        conn_shape[freq_dim] = len(fmin)
        patterns_shape[freq_dim] = len(fmin)
        conn_shape = tuple(conn_shape)
        patterns_shape = tuple(patterns_shape)

        assert con.get_data().shape == conn_shape
        if method in ["cacoh", "mic"]:
            assert np.shape(con.attrs["patterns"]) == patterns_shape


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
    methods = ["cacoh", "mic", "mim", "gc", "gc_tr"]
    con = spectral_connectivity_epochs(
        data,
        method=methods,
        indices=([[0, 1]], [[2, 3]]),
        mode="multitaper",
        sfreq=100,
        fskip=0,
        faverage=False,
        tmin=0,
        tmax=None,
        mt_bandwidth=4,
        mt_low_bias=True,
        mt_adaptive=False,
        gc_n_lags=20,
        rank=tuple([[2], [2]]),
        n_components=1,
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
    conn_kwargs = dict(mode=mode, cwt_freqs=cwt_freqs)

    # check bad indices without nested array caught
    with pytest.raises(
        TypeError, match="multivariate indices must contain array-likes"
    ):
        non_nested_indices = ([0, 1], [2, 3])
        spectral_connectivity_epochs(
            data, method=method, indices=non_nested_indices, gc_n_lags=10, **conn_kwargs
        )

    # check bad indices with repeated channels caught
    with pytest.raises(
        ValueError, match="multivariate indices cannot contain repeated"
    ):
        repeated_indices = ([[0, 1, 1]], [[2, 2, 3]])
        spectral_connectivity_epochs(
            data, method=method, indices=repeated_indices, gc_n_lags=10, **conn_kwargs
        )

    # check mixed methods caught
    with pytest.raises(ValueError, match="bivariate and multivariate connectivity"):
        if isinstance(method, str):
            mixed_methods = [method, "coh"]
        elif isinstance(method, list):
            mixed_methods = [*method, "coh"]
        spectral_connectivity_epochs(
            data, method=mixed_methods, indices=indices, **conn_kwargs
        )

    # check bad rank args caught
    too_low_rank = ([0], [0])
    with pytest.raises(ValueError, match="ranks for seeds and targets must be"):
        spectral_connectivity_epochs(
            data, method=method, indices=indices, rank=too_low_rank, **conn_kwargs
        )
    too_high_rank = ([3], [3])
    with pytest.raises(ValueError, match="ranks for seeds and targets must be"):
        spectral_connectivity_epochs(
            data, method=method, indices=indices, rank=too_high_rank, **conn_kwargs
        )
    too_few_rank = ([], [])
    with pytest.raises(ValueError, match="rank argument must have shape"):
        spectral_connectivity_epochs(
            data, method=method, indices=indices, rank=too_few_rank, **conn_kwargs
        )
    too_much_rank = ([2, 2], [2, 2])
    with pytest.raises(ValueError, match="rank argument must have shape"):
        spectral_connectivity_epochs(
            data, method=method, indices=indices, rank=too_much_rank, **conn_kwargs
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
            indices=indices,
            sfreq=sfreq,
            gc_n_lags=10,
            **conn_kwargs,
        )
        assert rank_con.attrs["rank"] == ([1], [1])

    if method in ["cacoh", "mic", "mim"]:
        # check rank-deficient transformation matrix caught
        with pytest.raises(RuntimeError, match="the transformation matrix"):
            spectral_connectivity_epochs(
                bad_data,
                method=method,
                indices=indices,
                sfreq=sfreq,
                rank=([2], [2]),
                **conn_kwargs,
            )

    # check bad n_components caught
    with pytest.raises(TypeError, match="`n_components` must be an instance of int"):
        spectral_connectivity_epochs(
            data, method=method, indices=indices, n_components=[1], **conn_kwargs
        )
    with pytest.raises(ValueError, match="`n_components` must be >= 1"):
        spectral_connectivity_epochs(
            data, method=method, indices=indices, n_components=0, **conn_kwargs
        )
    with pytest.raises(
        ValueError, match="`n_components` is greater than the minimum rank of the data"
    ):
        spectral_connectivity_epochs(
            data, method=method, indices=indices, n_components=3, **conn_kwargs
        )
    with pytest.raises(
        ValueError, match="`n_components` is greater than the minimum rank of the data"
    ):
        spectral_connectivity_epochs(
            data,
            method=method,
            indices=indices,
            rank=([1], [1]),
            n_components=2,
            **conn_kwargs,
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
                indices=indices,
                fmin=frange[0],
                fmax=frange[1],
                gc_n_lags=n_lags,
                **conn_kwargs,
            )

        # check no indices caught
        with pytest.raises(ValueError, match="indices must be specified"):
            spectral_connectivity_epochs(
                data, method=method, indices=None, **conn_kwargs
            )

        # check intersecting indices caught
        bad_indices = ([[0, 1]], [[0, 2]])
        with pytest.raises(
            ValueError, match="seed and target indices must not intersect"
        ):
            spectral_connectivity_epochs(
                data, method=method, indices=bad_indices, **conn_kwargs
            )

        # check bad fmin/fmax caught
        with pytest.raises(ValueError, match="computing Granger causality on multiple"):
            spectral_connectivity_epochs(
                data,
                method=method,
                indices=indices,
                fmin=(10.0, 15.0),
                fmax=(15.0, 20.0),
                **conn_kwargs,
            )

        # check rank-deficient autocovariance caught
        with pytest.raises(RuntimeError, match="the autocovariance matrix is singular"):
            spectral_connectivity_epochs(
                bad_data,
                method=method,
                indices=indices,
                sfreq=sfreq,
                rank=([2], [2]),
                **conn_kwargs,
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
    where the true directionality of the connections seems to be identified.
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
            ValueError, match="Padding cannot be larger than half of data length"
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
@pytest.mark.parametrize("n_components", [1, 2, 3, None])
def test_multivar_spectral_connectivity_time_shapes(
    method, average, faverage, n_components
):
    """Test result shapes of time-resolved multivar. connectivity methods."""
    if method in ["mim", "gc", "gc_tr"] and n_components != 1:
        return  # don't bother testing on methods that don't support multiple comps

    n_epochs = 8
    data = make_signals_in_freq_bands(
        n_seeds=3,  # do not change!
        n_targets=3,  # do not change!
        freq_band=(10, 20),  # arbitrary for this test
        n_epochs=n_epochs,
        n_times=256,
        sfreq=50,
        rng_seed=0,
    )

    # test with full indices
    indices = ([[0, 1, 2]], [[3, 4, 5]])
    n_cons = len(indices[0])
    max_n_chans = np.shape(indices)[2]
    n_actual_components = max_n_chans if n_components is None else n_components
    freqs = np.arange(10, 25 + 1)

    con_shape = []
    if not average:
        con_shape.append(n_epochs)  # epochs
    con_shape.append(n_cons)  # n_cons
    if n_actual_components != 1:
        con_shape.append(n_actual_components)  # n_comps
    con_shape.append(len(freqs) if not faverage else 1)  # n_freqs

    patterns_shape = [2]  # seeds/targets
    if not average:
        patterns_shape.append(n_epochs)  # epochs
    patterns_shape.append(n_cons)  # n_cons
    if n_actual_components != 1:
        patterns_shape.append(n_actual_components)  # n_comps
    patterns_shape.append(max_n_chans)  # n_chans
    patterns_shape.append(len(freqs) if not faverage else 1)  # n_freqs

    # check shape of con scores
    con = spectral_connectivity_time(
        data,
        freqs,
        indices=indices,
        method=method,
        faverage=faverage,
        average=average,
        gc_n_lags=10,
        n_components=n_components,
    )
    assert con.shape == tuple(con_shape)

    # check shape of patterns
    if method in ["cacoh", "mic"]:
        patterns = np.array(con.attrs["patterns"])
        assert patterns.shape == tuple(patterns_shape)
        assert not np.any(np.isnan(patterns))  # no padded entries

        # test with ragged indices
        if n_components is not None and n_components > 2:
            return  # cannot test when n_comps > rank of data
        indices = ([[0, 1, 2]], [[3, 4]])
        n_actual_components = 2 if n_components is None else n_components

        patterns_shape = [2]  # seeds/targets
        if not average:
            patterns_shape.append(n_epochs)  # epochs
        patterns_shape.append(n_cons)  # n_cons
        if n_actual_components != 1:
            patterns_shape.append(n_actual_components)  # n_comps
        patterns_shape.append(max_n_chans)  # n_chans
        patterns_shape.append(len(freqs) if not faverage else 1)  # n_freqs

        con = spectral_connectivity_time(
            data,
            freqs,
            indices=indices,
            method=method,
            faverage=faverage,
            average=average,
            n_components=n_components,
        )
        patterns = np.array(con.attrs["patterns"])

        # 2 x [epochs] x cons x [comps] x channels x freqs|fbands
        assert patterns.shape == tuple(patterns_shape)
        assert not np.any(np.isnan(patterns[0, ..., :, :]))  # seeds 0-3 present
        assert not np.any(np.isnan(patterns[1, ..., :2, :]))  # targets 1-2 present
        assert np.all(np.isnan(patterns[1, ..., 2:, :]))  # targets 3 padded
        assert np.all(np.array(con.indices) == np.array(([[0, 1, 2]], [[3, 4, -1]])))


@pytest.mark.skipif(
    not check_version("mne", "1.10"), reason="Requires MNE v1.10.0 or higher"
)  # Taper weights in TFR objects added in MNE v1.10.0
@pytest.mark.parametrize("method", ["coh", "cacoh"])
@pytest.mark.parametrize("mode", ["multitaper", "cwt_morlet"])
def test_spectral_connectivity_time_tfr_input(method, mode):
    """Test spec_conn_time works with EpochsTFR data as input.

    Important to test both bivariate and multivariate methods, as the latter involves
    additional steps (e.g., rank computation).
    """
    # Simulation parameters & data generation
    n_seeds = 2
    n_targets = 2
    fband = (15, 20)  # Hz
    trans_bandwidth = 1.0  # Hz

    data = make_signals_in_freq_bands(
        n_seeds=n_seeds,
        n_targets=n_targets,
        freq_band=fband,
        n_epochs=30,
        n_times=200,
        sfreq=100,
        trans_bandwidth=trans_bandwidth,
        snr=0.7,
        connection_delay=5,
        rng_seed=44,
    )

    if method == "coh":
        indices = seed_target_indices(
            seeds=np.arange(n_seeds), targets=np.arange(n_targets) + n_seeds
        )
    else:
        indices = ([np.arange(n_seeds)], [np.arange(n_targets) + n_seeds])

    # Compute TFR
    freqs = np.arange(10, 50)
    n_cycles = 5.0  # non-default value to avoid warning in spec_conn_time
    mt_bandwidth = 4.0
    kwargs = dict()
    if mode == "cwt_morlet":
        kwargs.update(zero_mean=False)  # default in spec_conn_time
        spec_mode = "morlet"
    else:
        kwargs.update(time_bandwidth=mt_bandwidth)
        spec_mode = mode
    coeffs = data.compute_tfr(
        method=spec_mode, freqs=freqs, n_cycles=n_cycles, output="complex", **kwargs
    )

    # Compute connectivity
    con_kwargs = dict(
        method=method,
        indices=indices,
        mode=mode,
        freqs=freqs,
        n_cycles=n_cycles,
        mt_bandwidth=mt_bandwidth,
        average=True,
    )
    con = spectral_connectivity_time(data=coeffs, **con_kwargs)

    # Check connectivity from Epochs and EpochsTFR are equivalent (small but non-zero
    # tolerance given due to some platform-dependent variation)
    con_from_epochs = spectral_connectivity_time(data=data, **con_kwargs)
    assert_allclose(
        np.abs(con.get_data()), np.abs(con_from_epochs.get_data()), atol=1e-7
    )

    # Check connectivity values are as expected
    freqs_con = (freqs >= fband[0]) & (freqs <= fband[1])
    freqs_noise = (freqs < fband[0] - trans_bandwidth * 2) | (
        freqs > fband[1] + trans_bandwidth * 2
    )
    # check freqs of simulated interaction show strong connectivity
    assert_array_less(0.6, np.abs(con.get_data()[:, freqs_con].mean()))
    # check freqs of no simulated interaction (just noise) show weak connectivity
    assert_array_less(np.abs(con.get_data()[:, freqs_noise].mean()), 0.3)


# TODO: Add general test for error catching for spec_conn_time
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
    with pytest.raises(TypeError, match="Epochs, EpochsTFR, or a NumPy arr"):
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

    # check bad n_components caught
    with pytest.raises(TypeError, match="`n_components` must be an instance of int"):
        spectral_connectivity_time(
            data, freqs, method=method, indices=indices, mode=mode, n_components=[1]
        )
    with pytest.raises(ValueError, match="`n_components` must be >= 1"):
        spectral_connectivity_time(
            data, freqs, method=method, indices=indices, mode=mode, n_components=0
        )
    with pytest.raises(
        ValueError, match="`n_components` is greater than the minimum rank of the data"
    ):
        spectral_connectivity_time(
            data, freqs, method=method, indices=indices, mode=mode, n_components=3
        )
    with pytest.raises(
        ValueError, match="`n_components` is greater than the minimum rank of the data"
    ):
        spectral_connectivity_time(
            data,
            freqs,
            method=method,
            indices=indices,
            mode=mode,
            rank=([1], [1]),
            n_components=2,
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


@pytest.mark.skipif(
    not check_version("mne", "1.10"), reason="Requires MNE v1.10.0 or higher"
)  # Taper weights in TFR objects added in MNE v1.10.0
def test_spectral_connectivity_time_tfr_input_error_catch():
    """Test spec_conn_time catches errors with EpochsTFR data as input."""
    # Generate data
    rng = np.random.default_rng(44)
    n_epochs, n_chans, n_times = (5, 2, 100)
    sfreq = 50
    data = rng.random((n_epochs, n_chans, n_times))
    info = create_info(ch_names=n_chans, sfreq=sfreq, ch_types="eeg")
    data = EpochsArray(data=data, info=info)
    freqs = np.arange(10, 20)

    # Test not Fourier coefficients caught
    with pytest.raises(TypeError, match="must contain complex-valued Fourier coeff"):
        tfr = data.compute_tfr(method="morlet", freqs=freqs, output="power")
        spectral_connectivity_time(data=tfr, freqs=freqs)

    # Simulate missing weights attr in EpochsTFR object
    tfr = data.compute_tfr(method="multitaper", output="complex", freqs=freqs)
    with pytest.raises(AttributeError, match="weights are required for multitaper"):
        tfr_copy = tfr.copy()
        del tfr_copy._weights
        spectral_connectivity_time(data=tfr_copy)
    with pytest.raises(AttributeError, match="weights are required for multitaper"):
        tfr._weights = None
        spectral_connectivity_time(data=tfr)

    # Test no freqs caught for non-TFR input
    with pytest.raises(TypeError, match="`freqs` must be specified"):
        spectral_connectivity_time(data=data)


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
