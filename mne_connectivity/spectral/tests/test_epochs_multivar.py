import numpy as np
from numpy.testing import (
    assert_allclose, 
    assert_array_almost_equal,
    assert_array_less
    )
import pytest

from mne_connectivity import (
    SpectralConnectivity, 
    multivar_spectral_connectivity_epochs
    )
from mne_connectivity.spectral.epochs import _CohEst, _get_n_epochs

from .test_spectral import create_test_dataset


@pytest.mark.parametrize('method', [
    'mic', 'mim', ['mic', 'mim']])
@pytest.mark.parametrize('mode', ['multitaper', 'fourier', 'cwt_morlet'])
def test_multivar_spectral_connectivity(method, mode):
    """Test frequency-domain multivariate connectivity methods."""
    sfreq = 50.
    n_signals = 9
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

    class _InvalidClass:
        pass

    # First we test some invalid parameters:
    pytest.raises(ValueError, multivar_spectral_connectivity_epochs,
                  data, method='notamethod', 
                  match='is not a valid connectivity method')
    pytest.raises(ValueError, multivar_spectral_connectivity_epochs,
                  data, method=_InvalidClass, 
                  match='The supplied connectivity method does not have the method')
    pytest.raises(ValueError, multivar_spectral_connectivity_epochs, data,
                  mode='notamode', match='mode has an invalid value')

    # test invalid fmin fmax settings
    pytest.raises(ValueError, multivar_spectral_connectivity_epochs, data, fmin=10,
                  fmax=10 + 0.5 * (sfreq / float(n_times)), 
                  method='There are no frequency points between')
    pytest.raises(ValueError, multivar_spectral_connectivity_epochs,
                  data, fmin=10, fmax=5, method='fmax must be larger than fmin')
    pytest.raises(ValueError, multivar_spectral_connectivity_epochs, data, fmin=(0, 11),
                  fmax=(5, 10), method='fmax must be larger than fmin')
    pytest.raises(ValueError, multivar_spectral_connectivity_epochs, data, fmin=(11,),
                  fmax=(12, 15), 
                  method='fmin and fmax must have the same length')

    # define some frequencies for cwt
    cwt_freqs = np.arange(3, 24.5, 1)

    if method == 'mic' and mode == 'multitaper':
        # only check adaptive estimation for coh to reduce test time
        check_adaptive = [False, True]
    else:
        check_adaptive = [False]

    if method == 'mic' and mode == 'cwt_morlet':
        # so we also test using an array for num cycles
        cwt_n_cycles = 7. * np.ones(len(cwt_freqs))
    else:
        cwt_n_cycles = 7.


    for adaptive in check_adaptive:

        if adaptive:
            mt_bandwidth = 1.
        else:
            mt_bandwidth = None

        # indices cannot be None
        pytest.raises(ValueError, multivar_spectral_connectivity_epochs,
            data, method=method, mode=mode, indices=None, sfreq=sfreq,
            mt_adaptive=adaptive, mt_low_bias=True,
            mt_bandwidth=mt_bandwidth, cwt_freqs=cwt_freqs,
            cwt_n_cycles=cwt_n_cycles, match='Please put the matching string here')

        indices = ([[0, 1, 2], [3, 4, 5]], [[6, 7, 8], [6, 7, 8]])
        con = multivar_spectral_connectivity_epochs(
            data, method=method, mode=mode, indices=indices, sfreq=sfreq,
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
