import numpy as np
from numpy.testing import (assert_allclose, assert_array_almost_equal,
                           assert_array_less)
import pytest
from mne import (EpochsArray, SourceEstimate, create_info)
from mne.filter import filter_data

from mne_connectivity import (
    SpectralConnectivity, spectral_connectivity_epochs,
    read_connectivity, spectral_connectivity_time)
from mne_connectivity.spectral.epochs import _CohEst, _get_n_epochs
from mne_connectivity.spectral.epochs import (
    _compute_freq_mask, _compute_freqs)


def create_test_dataset(sfreq, n_signals, n_epochs, n_times, tmin, tmax,
                        fstart, fend, trans_bandwidth=2.):
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
    data[1, :] = filter_data(data[0, :], sfreq, fstart, fend,
                             filter_length='auto', fir_design='firwin2',
                             l_trans_bandwidth=trans_bandwidth,
                             h_trans_bandwidth=trans_bandwidth)
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
            stc = SourceEstimate(data=d, vertices=vertices,
                                 tmin=tmin, tstep=1 / float(sfreq))
            yield stc
        else:
            # simulate a combination of array and source estimate
            arr = d[0]
            stc = SourceEstimate(data=d[1:], vertices=vertices,
                                 tmin=tmin, tstep=1 / float(sfreq))
            yield (arr, stc)


@pytest.mark.parametrize('method', ['coh', 'cohy', 'imcoh', 'plv'])
@pytest.mark.parametrize('mode', ['multitaper', 'fourier', 'cwt_morlet'])
def test_spectral_connectivity_parallel(method, mode, tmp_path):
    """Test saving spectral connectivity with parallel functions."""
    # Use a case known to have no spurious correlations (it would bad if
    # tests could randomly fail):
    rng = np.random.RandomState(0)
    trans_bandwidth = 2.

    sfreq = 50.
    n_signals = 3
    n_epochs = 8
    n_times = 256
    n_jobs = 2  # test with parallelization

    data = rng.randn(n_signals, n_epochs * n_times)
    # simulate connectivity from 5Hz..15Hz
    fstart, fend = 5.0, 15.0
    data[1, :] = filter_data(data[0, :], sfreq, fstart, fend,
                             filter_length='auto', fir_design='firwin2',
                             l_trans_bandwidth=trans_bandwidth,
                             h_trans_bandwidth=trans_bandwidth)
    # add some noise, so the spectrum is not exactly zero
    data[1, :] += 1e-2 * rng.randn(n_times * n_epochs)
    data = data.reshape(n_signals, n_epochs, n_times)
    data = np.transpose(data, [1, 0, 2])

    # define some frequencies for cwt
    cwt_freqs = np.arange(3, 24.5, 1)

    if method == 'coh' and mode == 'multitaper':
        # only check adaptive estimation for coh to reduce test time
        check_adaptive = [False, True]
    else:
        check_adaptive = [False]

    if method == 'coh' and mode == 'cwt_morlet':
        # so we also test using an array for num cycles
        cwt_n_cycles = 7. * np.ones(len(cwt_freqs))
    else:
        cwt_n_cycles = 7.

    for adaptive in check_adaptive:

        if adaptive:
            mt_bandwidth = 1.
        else:
            mt_bandwidth = None

        con = spectral_connectivity_epochs(
            data, method=method, mode=mode, indices=None, sfreq=sfreq,
            mt_adaptive=adaptive, mt_low_bias=True,
            mt_bandwidth=mt_bandwidth, cwt_freqs=cwt_freqs,
            cwt_n_cycles=cwt_n_cycles, n_jobs=n_jobs)

        tmp_file = tmp_path / 'temp_file.nc'
        con.save(tmp_file)

        read_con = read_connectivity(tmp_file)
        assert_array_almost_equal(con.get_data(), read_con.get_data())
        assert repr(con) == repr(read_con)


@pytest.mark.parametrize('method', ['coh', 'cohy', 'imcoh', 'plv',
                                    ['ciplv', 'ppc', 'pli', 'pli2_unbiased',
                                     'dpli', 'wpli', 'wpli2_debiased', 'coh']])
@pytest.mark.parametrize('mode', ['multitaper', 'fourier', 'cwt_morlet'])
def test_spectral_connectivity(method, mode):
    """Test frequency-domain connectivity methods."""
    sfreq = 50.
    n_signals = 3
    n_epochs = 8
    n_times = 256
    trans_bandwidth = 2.
    tmin = 0.
    tmax = (n_times - 1) / sfreq

    # 5Hz..15Hz
    fstart, fend = 5.0, 15.0
    data, times_data = create_test_dataset(
        sfreq, n_signals=n_signals, n_epochs=n_epochs, n_times=n_times,
        tmin=tmin, tmax=tmax,
        fstart=fstart, fend=fend, trans_bandwidth=trans_bandwidth)

    # First we test some invalid parameters:
    pytest.raises(ValueError, spectral_connectivity_epochs,
                  data, method='notamethod')
    pytest.raises(ValueError, spectral_connectivity_epochs, data,
                  mode='notamode')

    # test invalid fmin fmax settings
    pytest.raises(ValueError, spectral_connectivity_epochs, data, fmin=10,
                  fmax=10 + 0.5 * (sfreq / float(n_times)))
    pytest.raises(ValueError, spectral_connectivity_epochs,
                  data, fmin=10, fmax=5)
    pytest.raises(ValueError, spectral_connectivity_epochs, data, fmin=(0, 11),
                  fmax=(5, 10))
    pytest.raises(ValueError, spectral_connectivity_epochs, data, fmin=(11,),
                  fmax=(12, 15))

    # define some frequencies for cwt
    cwt_freqs = np.arange(3, 24.5, 1)

    if method == 'coh' and mode == 'multitaper':
        # only check adaptive estimation for coh to reduce test time
        check_adaptive = [False, True]
    else:
        check_adaptive = [False]

    if method == 'coh' and mode == 'cwt_morlet':
        # so we also test using an array for num cycles
        cwt_n_cycles = 7. * np.ones(len(cwt_freqs))
    else:
        cwt_n_cycles = 7.

    for adaptive in check_adaptive:

        if adaptive:
            mt_bandwidth = 1.
        else:
            mt_bandwidth = None

        con = spectral_connectivity_epochs(
            data, method=method, mode=mode, indices=None, sfreq=sfreq,
            mt_adaptive=adaptive, mt_low_bias=True,
            mt_bandwidth=mt_bandwidth, cwt_freqs=cwt_freqs,
            cwt_n_cycles=cwt_n_cycles)

        if isinstance(method, list):
            this_con = con[0]
        else:
            this_con = con
        freqs = this_con.attrs.get('freqs_used')
        n = this_con.n_epochs_used
        if isinstance(this_con, SpectralConnectivity):
            times = this_con.attrs.get('times_used')
        else:
            times = this_con.times

        assert (n == n_epochs)
        assert_array_almost_equal(times_data, times)

        if mode == 'multitaper':
            upper_t = 0.95
            lower_t = 0.5
        else:  # mode == 'fourier' or mode == 'cwt_morlet'
            # other estimates have higher variance
            upper_t = 0.8
            lower_t = 0.75

        # test the simulated signal
        gidx = np.searchsorted(freqs, (fstart, fend))
        bidx = np.searchsorted(freqs,
                               (fstart - trans_bandwidth * 2,
                                fend + trans_bandwidth * 2))
        if method == 'coh':
            assert np.all(
                con.get_data(output='dense')[
                    1, 0, gidx[0]:gidx[1]
                ] > upper_t), \
                con.get_data()[
                    1, 0, gidx[0]:gidx[1]].min()
            # we see something for zero-lag
            assert_array_less(
                con.get_data(output='dense')
                [1, 0, :bidx[0]],
                lower_t)
            assert np.all(
                con.get_data(output='dense')[1, 0, bidx[1]:] < lower_t), \
                con.get_data()[1, 0, bidx[1:]].max()
        elif method == 'cohy':
            # imaginary coh will be zero
            check = np.imag(con.get_data(output='dense')
                            [1, 0, gidx[0]:gidx[1]])
            assert np.all(check < lower_t), check.max()
            # we see something for zero-lag
            assert_array_less(
                upper_t,
                np.abs(con.get_data(output='dense')[
                    1, 0, gidx[0]:gidx[1]
                ]))
            assert_array_less(
                np.abs(con.get_data(output='dense')[1, 0, :bidx[0]]),
                lower_t)
            assert_array_less(
                np.abs(con.get_data(output='dense')[1, 0, bidx[1]:]),
                lower_t)
        elif method == 'imcoh':
            # imaginary coh will be zero
            assert_array_less(
                con.get_data(output='dense')[1, 0, gidx[0]:gidx[1]],
                lower_t)
            assert_array_less(
                con.get_data(output='dense')[1, 0, :bidx[0]],
                lower_t)
            assert_array_less(
                con.get_data(output='dense')[1, 0, bidx[1]:], lower_t),
            assert np.all(
                con.get_data(output='dense')[1, 0, bidx[1]:] < lower_t), \
                con.get_data()[1, 0, bidx[1]:].max()

        # compute a subset of connections using indices and 2 jobs
        indices = (np.array([2, 1]), np.array([0, 0]))

        if not isinstance(method, list):
            test_methods = (method, _CohEst)
        else:
            test_methods = method

        stc_data = _stc_gen(data, sfreq, tmin)
        con2 = spectral_connectivity_epochs(
            stc_data, method=test_methods, mode=mode, indices=indices,
            sfreq=sfreq, mt_adaptive=adaptive, mt_low_bias=True,
            mt_bandwidth=mt_bandwidth, tmin=tmin, tmax=tmax,
            cwt_freqs=cwt_freqs, cwt_n_cycles=cwt_n_cycles)

        assert isinstance(con2, list)
        assert len(con2) == len(test_methods)
        freqs2 = con2[0].attrs.get('freqs_used')
        if 'times' in con2[0].dims:
            times2 = con2[0].times
        else:
            times2 = con2[0].attrs.get('times_used')
        n2 = con2[0].n_epochs_used

        if method == 'coh':
            assert_array_almost_equal(con2[0].get_data(), con2[1].get_data())

        if not isinstance(method, list):
            con2 = con2[0]  # only keep the first method

            # we get the same result for the probed connections
            assert_array_almost_equal(freqs, freqs2)

            # "con2" is a raveled array already, so
            # simulate setting indices on the full output in "con"
            assert_array_almost_equal(con.get_data(output='dense')[indices],
                                      con2.get_data())
            assert (n == n2)
            assert_array_almost_equal(times_data, times2)
        else:
            # we get the same result for the probed connections
            assert (len(con) == len(con2))
            for c, c2 in zip(con, con2):
                assert_array_almost_equal(freqs, freqs2)
                assert_array_almost_equal(c.get_data(output='dense')[indices],
                                          c2.get_data())
                assert (n == n2)
                assert_array_almost_equal(times_data, times2)

        # Test with faverage
        # compute same connections for two bands, fskip=1, and f. avg.
        fmin = (5., 15.)
        fmax = (15., 30.)
        con3 = spectral_connectivity_epochs(
            data, method=method, mode=mode, indices=indices,
            sfreq=sfreq, fmin=fmin, fmax=fmax, fskip=1, faverage=True,
            mt_adaptive=adaptive, mt_low_bias=True,
            mt_bandwidth=mt_bandwidth, cwt_freqs=cwt_freqs,
            cwt_n_cycles=cwt_n_cycles)

        if isinstance(method, list):
            freqs3 = con3[0].attrs.get('freqs_used')
        else:
            freqs3 = con3.attrs.get('freqs_used')

        assert (isinstance(freqs3, list))
        assert (len(freqs3) == len(fmin))
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
                n_times = len(con2.attrs.get('times_used'))

                # compute frequencies to analyze based on number of samples,
                # sampling rate, specified wavelet frequencies and mode
                freqs = _compute_freqs(n_times, sfreq, cwt_freqs, mode)

                # compute the mask based on specified min/max and decim factor
                freq_mask = _compute_freq_mask(
                    freqs, [fmin[i]], [fmax[i]], fskip)
                freqs = freqs[freq_mask]
                freqs_idx = np.searchsorted(freqs2, freqs)
                con2_avg = np.mean(con2.get_data()[:, freqs_idx], axis=1)
                assert_array_almost_equal(con2_avg, con3.get_data()[:, i])
        else:
            for j in range(len(con2)):
                for i in range(len(freqs3)):
                    # now we want to get the frequency indices
                    # create a frequency mask for all bands
                    n_times = len(con2[0].attrs.get('times_used'))

                    # compute frequencies to analyze based on number of
                    # samples, sampling rate, specified wavelet frequencies
                    # and mode
                    freqs = _compute_freqs(n_times, sfreq, cwt_freqs, mode)

                    # compute the mask based on specified min/max and
                    # decim factor
                    freq_mask = _compute_freq_mask(
                        freqs, [fmin[i]], [fmax[i]], fskip)
                    freqs = freqs[freq_mask]
                    freqs_idx = np.searchsorted(freqs2, freqs)

                    con2_avg = np.mean(con2[j].get_data()[
                                       :, freqs_idx], axis=1)
                    assert_array_almost_equal(
                        con2_avg, con3[j].get_data()[:, i])

    # test _get_n_epochs
    full_list = list(range(10))
    out_lens = np.array([len(x) for x in _get_n_epochs(full_list, 4)])
    assert ((out_lens == np.array([4, 4, 2])).all())
    out_lens = np.array([len(x) for x in _get_n_epochs(full_list, 11)])
    assert (len(out_lens) > 0)
    assert (out_lens[0] == 10)


@ pytest.mark.parametrize('kind', ('epochs', 'ndarray', 'stc', 'combo'))
def test_epochs_tmin_tmax(kind):
    """Test spectral.spectral_connectivity_epochs with epochs and arrays."""
    rng = np.random.RandomState(0)
    n_epochs, n_chs, n_times, sfreq, f = 10, 2, 2000, 1000., 20.
    data = rng.randn(n_epochs, n_chs, n_times)
    sig = np.sin(2 * np.pi * f * np.arange(1000) / sfreq) * np.hanning(1000)
    data[:, :, 500:1500] += sig
    info = create_info(n_chs, sfreq, 'eeg')
    if kind == 'epochs':
        tmin = -1
        X = EpochsArray(data, info, tmin=tmin)
    elif kind == 'stc':
        tmin = -1
        X = [SourceEstimate(d, [[0], [0]], tmin, 1. / sfreq) for d in data]
    elif kind == 'combo':
        tmin = -1
        X = [(d[[0]], SourceEstimate(d[[1]], [[0], []], tmin, 1. / sfreq))
             for d in data]
    else:
        assert kind == 'ndarray'
        tmin = 0
        X = data
    want_times = np.arange(n_times) / sfreq + tmin

    # Parameters for computing connectivity
    fmin, fmax = f - 2, f + 2
    kwargs = {'method': 'coh', 'mode': 'multitaper', 'sfreq': sfreq,
              'fmin': fmin, 'fmax': fmax, 'faverage': True,
              'mt_adaptive': False, 'n_jobs': 1}

    # Check the entire interval
    conn = spectral_connectivity_epochs(X, **kwargs)
    assert 0.89 < conn.get_data(output='dense')[1, 0] < 0.91
    assert_allclose(conn.attrs.get('times_used'), want_times)
    # Check a time interval before the sinusoid
    conn = spectral_connectivity_epochs(X, tmax=tmin + 0.5, **kwargs)
    assert 0 < conn.get_data(output='dense')[1, 0] < 0.15
    # Check a time during the sinusoid
    conn = spectral_connectivity_epochs(
        X, tmin=tmin + 0.5, tmax=tmin + 1.5, **kwargs)
    assert 0.93 < conn.get_data(output='dense')[1, 0] <= 0.94
    # Check a time interval after the sinusoid
    conn = spectral_connectivity_epochs(
        X, tmin=tmin + 1.5, tmax=tmin + 1.9, **kwargs)
    assert 0 < conn.get_data(output='dense')[1, 0] < 0.15

    # Check for warning if tmin, tmax is outside of the time limits of data
    with pytest.warns(RuntimeWarning, match='start time tmin'):
        spectral_connectivity_epochs(X, **kwargs, tmin=tmin - 0.1)

    with pytest.warns(RuntimeWarning, match='stop time tmax'):
        spectral_connectivity_epochs(X, **kwargs, tmax=tmin + 2.5)

    # make one with mismatched times
    if kind != 'combo':
        return
    X = [(SourceEstimate(d[[0]], [[0], []], tmin - 1, 1. / sfreq),
          SourceEstimate(d[[1]], [[0], []], tmin, 1. / sfreq)) for d in data]
    with pytest.warns(RuntimeWarning, match='time scales of input') as w:
        spectral_connectivity_epochs(X, **kwargs)
    assert len(w) == 1  # just one even though there were multiple epochs


@pytest.mark.parametrize('method', ['coh', 'plv', 'pli', 'wpli', 'ciplv'])
@pytest.mark.parametrize(
    'mode', ['cwt_morlet', 'multitaper'])
@pytest.mark.parametrize('data_option', ['sync', 'random'])
def test_spectral_connectivity_time_phaselocked(method, mode, data_option):
    """Test time-resolved spectral connectivity with simulated phase-locked
    data."""
    rng = np.random.default_rng(0)
    n_epochs = 5
    n_channels = 3
    n_times = 1000
    sfreq = 250
    data = np.zeros((n_epochs, n_channels, n_times))
    if data_option == 'random':
        # Data is random, there should be no consistent phase differences.
        data = rng.random((n_epochs, n_channels, n_times))
    if data_option == 'sync':
        # Data consists of phase-locked 10Hz sine waves with constant phase
        # difference within each epoch.
        wave_freq = 10
        epoch_length = n_times / sfreq
        for i in range(n_epochs):
            for c in range(n_channels):
                phase = rng.random() * 10
                x = np.linspace(-wave_freq * epoch_length * np.pi + phase,
                                wave_freq * epoch_length * np.pi + phase,
                                n_times)
                data[i, c] = np.squeeze(np.sin(x))
    # the frequency band should contain the frequency at which there is a
    # hypothesized "connection"
    freq_band_low_limit = (8.)
    freq_band_high_limit = (13.)
    freqs = np.arange(freq_band_low_limit, freq_band_high_limit + 1)
    con = spectral_connectivity_time(data, freqs, method=method, mode=mode,
                                     sfreq=sfreq, fmin=freq_band_low_limit,
                                     fmax=freq_band_high_limit,
                                     n_jobs=1,
                                     faverage=True, average=True, sm_times=0)
    assert con.shape == (n_channels ** 2, len(con.freqs))
    con_matrix = con.get_data('dense')[..., 0]
    if data_option == 'sync':
        # signals are perfectly phase-locked, connectivity matrix should be
        # a lower triangular matrix of ones
        assert np.allclose(con_matrix,
                           np.tril(np.ones(con_matrix.shape),
                                   k=-1),
                           atol=0.01)
    if data_option == 'random':
        # signals are random, all connectivity values should be small
        # 0.5 is picked rather arbitrarily such that the obsolete wrong
        # implementation fails
        assert np.all(con_matrix) <= 0.5


@pytest.mark.parametrize('method', ['coh', 'plv', 'pli', 'wpli', 'ciplv'])
@pytest.mark.parametrize(
    'freqs', [[8., 10.], [8, 10], 10., 10])
@pytest.mark.parametrize('mode', ['cwt_morlet', 'multitaper'])
def test_spectral_connectivity_time_freqs(method, freqs, mode):
    """Test time-resolved spectral connectivity with int and float values for
    freqs."""
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
            x = np.linspace(-wave_freq * epoch_length * np.pi + phase,
                            wave_freq * epoch_length * np.pi + phase,
                            n_times)
            data[i, c] = np.squeeze(np.sin(x))
    # the frequency band should contain the frequency at which there is a
    # hypothesized "connection"
    con = spectral_connectivity_time(data, freqs, method=method,
                                     mode=mode, sfreq=sfreq,
                                     fmin=np.min(freqs),
                                     fmax=np.max(freqs), n_jobs=1,
                                     faverage=True, average=True, sm_times=0)
    assert con.shape == (n_channels ** 2, len(con.freqs))
    con_matrix = con.get_data('dense')[..., 0]

    # signals are perfectly phase-locked, connectivity matrix should be
    # a lower triangular matrix of ones
    assert np.allclose(con_matrix, np.tril(np.ones(con_matrix.shape), k=-1),
                       atol=0.01)


@pytest.mark.parametrize('method', ['coh', 'plv', 'pli', 'wpli'])
@pytest.mark.parametrize(
    'mode', ['cwt_morlet', 'multitaper'])
def test_spectral_connectivity_time_resolved(method, mode):
    """Test time-resolved spectral connectivity."""
    sfreq = 50.
    n_signals = 3
    n_epochs = 2
    n_times = 1000
    trans_bandwidth = 2.
    tmin = 0.
    tmax = (n_times - 1) / sfreq
    # 5Hz..15Hz
    fstart, fend = 5.0, 15.0
    data, _ = create_test_dataset(
        sfreq, n_signals=n_signals, n_epochs=n_epochs, n_times=n_times,
        tmin=tmin, tmax=tmax,
        fstart=fstart, fend=fend, trans_bandwidth=trans_bandwidth)
    ch_names = np.arange(n_signals).astype(str).tolist()
    info = create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
    data = EpochsArray(data, info)

    # define some frequencies for tfr
    freqs = np.arange(3, 20.5, 1)

    # run connectivity estimation
    con = spectral_connectivity_time(
        data, freqs, sfreq=sfreq, method=method, mode=mode,
        n_cycles=5)
    assert con.shape == (n_epochs, n_signals ** 2, len(con.freqs))
    assert con.get_data(output='dense').shape == \
        (n_epochs, n_signals, n_signals, len(con.freqs))

    # test the simulated signal
    triu_inds = np.vstack(np.triu_indices(n_signals, k=1)).T

    # average over frequencies
    conn_data = con.get_data(output='dense').mean(axis=-1)

    # the indices at which there is a correlation should be greater
    # then the rest of the components
    for epoch_idx in range(n_epochs):
        high_conn_val = conn_data[epoch_idx, 0, 1]
        assert all(high_conn_val >= conn_data[epoch_idx, idx, jdx]
                   for idx, jdx in triu_inds)


@pytest.mark.parametrize('method', ['coh', 'plv', 'pli', 'wpli'])
@pytest.mark.parametrize(
    'mode', ['cwt_morlet', 'multitaper'])
@pytest.mark.parametrize('padding', [0, 1, 5])
def test_spectral_connectivity_time_padding(method, mode, padding):
    """Test time-resolved spectral connectivity with padding."""
    sfreq = 50.
    n_signals = 3
    n_epochs = 2
    n_times = 300
    trans_bandwidth = 2.
    tmin = 0.
    tmax = (n_times - 1) / sfreq
    # 5Hz..15Hz
    fstart, fend = 5.0, 15.0
    data, _ = create_test_dataset(
        sfreq, n_signals=n_signals, n_epochs=n_epochs, n_times=n_times,
        tmin=tmin, tmax=tmax,
        fstart=fstart, fend=fend, trans_bandwidth=trans_bandwidth)
    ch_names = np.arange(n_signals).astype(str).tolist()
    info = create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
    data = EpochsArray(data, info)

    # define some frequencies for tfr
    freqs = np.arange(3, 20.5, 1)

    # run connectivity estimation
    if padding == 5:
        with pytest.raises(ValueError, match='Padding cannot be larger than '
                                             'half of data length'):
            con = spectral_connectivity_time(
                data, freqs, sfreq=sfreq, method=method, mode=mode,
                n_cycles=5, padding=padding)
        return
    else:
        con = spectral_connectivity_time(
            data, freqs, sfreq=sfreq, method=method, mode=mode,
            n_cycles=5, padding=padding)

    assert con.shape == (n_epochs, n_signals ** 2, len(con.freqs))
    assert con.get_data(output='dense').shape == \
        (n_epochs, n_signals, n_signals, len(con.freqs))

    # test the simulated signal
    triu_inds = np.vstack(np.triu_indices(n_signals, k=1)).T

    # average over frequencies
    conn_data = con.get_data(output='dense').mean(axis=-1)

    # the indices at which there is a correlation should be greater
    # then the rest of the components
    for epoch_idx in range(n_epochs):
        high_conn_val = conn_data[epoch_idx, 0, 1]
        assert all(high_conn_val >= conn_data[epoch_idx, idx, jdx]
                   for idx, jdx in triu_inds)


def test_save(tmp_path):
    """Test saving results of spectral connectivity."""
    rng = np.random.RandomState(0)
    n_epochs, n_chs, n_times, sfreq, f = 10, 2, 2000, 1000., 20.
    data = rng.randn(n_epochs, n_chs, n_times)
    sig = np.sin(2 * np.pi * f * np.arange(1000) / sfreq) * np.hanning(1000)
    data[:, :, 500:1500] += sig
    info = create_info(n_chs, sfreq, 'eeg')
    tmin = -1
    epochs = EpochsArray(data, info, tmin=tmin)

    conn = spectral_connectivity_epochs(
        epochs, fmin=(4, 8, 13, 30), fmax=(8, 13, 30, 45),
        faverage=True)
    conn.save(tmp_path / 'foo.nc')
