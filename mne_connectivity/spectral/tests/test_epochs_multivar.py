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

def create_test_dataset_multivar(sfreq, n_signals, n_epochs, n_times, tmin, tmax,
                        fstart, fend, trans_bandwidth=2., shift=None):
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
    shift : int, optional
        Shift the correlated signal by a given number of samples, by default 
        None.

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
    if shift is not None:
        data[1, :] = np.roll(data[1,:], shift=shift)

    # add some noise, so the spectrum is not exactly zero
    data[1, :] += 1e-2 * rng.randn(n_times * n_epochs)
    data = data.reshape(n_signals, n_epochs, n_times)
    data = np.transpose(data, [1, 0, 2])
    return data, times_data

@pytest.mark.parametrize('method', [
    'mic', 'mim', ['mic', 'mim']])
@pytest.mark.parametrize('mode', ['multitaper', 'fourier', 'cwt_morlet'])
def test_multivar_spectral_connectivity(method, mode):
    """Test frequency-domain multivariate connectivity methods."""
    sfreq = 50.
    #n_signals = 9
    n_epochs = 8
    n_times = 256
    trans_bandwidth = 2.
    tmin = 0.
    tmax = (n_times - 1) / sfreq

    # 5Hz..15Hz
    fstart, fend = 5.0, 15.0
    
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
            con = con[0]

        freqs = con.attrs.get('freqs_used')
        n = con.n_epochs_used
        if isinstance(con, SpectralConnectivity):
            times = con.attrs.get('times_used')
        else:
            times = con.times

        assert (n == n_epochs)
        assert_array_almost_equal(times_data, times)

        upper_t = 0.4
        lower_t = 0.5

        # test the simulated signal
        gidx = np.searchsorted(freqs, (fstart, fend))
        bidx = np.searchsorted(freqs,
                               (fstart - trans_bandwidth * 2,
                                fend + trans_bandwidth * 2))
        
        # 0-lag, 2 signals
        data, times_data = create_test_dataset_multivar(
        sfreq, n_signals=2, n_epochs=n_epochs, n_times=n_times,
        tmin=tmin, tmax=tmax,
        fstart=fstart, fend=fend, trans_bandwidth=trans_bandwidth, shift=None)
        con = multivar_spectral_connectivity_epochs(
            data, method=method, mode=mode, indices=([[0]], [[1]]), sfreq=sfreq,
            mt_adaptive=adaptive, mt_low_bias=True,
            mt_bandwidth=mt_bandwidth, cwt_freqs=cwt_freqs,
            cwt_n_cycles=cwt_n_cycles, n_seed_components=None, 
            n_target_components=None)
        assert_array_less(con.get_data(output='raveled')[ 0, :bidx[0]],lower_t)

        #1-lag, 4 signals
        data, times_data = create_test_dataset_multivar(
        sfreq, n_signals=4, n_epochs=n_epochs, n_times=n_times,
        tmin=tmin, tmax=tmax,
        fstart=fstart, fend=fend, trans_bandwidth=trans_bandwidth, shift=None)
        con = multivar_spectral_connectivity_epochs(
            data, method=method, mode=mode, indices=([[0,2]], [[1,3]]), sfreq=sfreq,
            mt_adaptive=adaptive, mt_low_bias=True,
            mt_bandwidth=mt_bandwidth, cwt_freqs=cwt_freqs,
            cwt_n_cycles=cwt_n_cycles, n_seed_components=None, 
            n_target_components=None)
        assert np.all(con.get_data(output='raveled')[0, gidx[0]:gidx[1]] > upper_t), \
            con.get_data()[0, gidx[0]:gidx[1]].min()

    #1-lag, 4 signals, 1 seed, 1 target
        data, times_data = create_test_dataset_multivar(
        sfreq, n_signals=4, n_epochs=n_epochs, n_times=n_times,
        tmin=tmin, tmax=tmax,
        fstart=fstart, fend=fend, trans_bandwidth=trans_bandwidth, shift=None)
        con = multivar_spectral_connectivity_epochs(
            data, method=method, mode=mode, indices=([[0,2]], [[1,3]]), sfreq=sfreq,
            mt_adaptive=adaptive, mt_low_bias=True,
            mt_bandwidth=mt_bandwidth, cwt_freqs=cwt_freqs,
            cwt_n_cycles=cwt_n_cycles, n_seed_components=(1,), 
            n_target_components=(1,))
    #1-lag, 4 signals, 2 seeds, 2 targets
        data, times_data = create_test_dataset_multivar(
        sfreq, n_signals=4, n_epochs=n_epochs, n_times=n_times,
        tmin=tmin, tmax=tmax,
        fstart=fstart, fend=fend, trans_bandwidth=trans_bandwidth, shift=None)
        con = multivar_spectral_connectivity_epochs(
            data, method=method, mode=mode, indices=([[0,2]], [[1,3]]), sfreq=sfreq,
            mt_adaptive=adaptive, mt_low_bias=True,
            mt_bandwidth=mt_bandwidth, cwt_freqs=cwt_freqs,
            cwt_n_cycles=cwt_n_cycles, n_seed_components=(2,), 
            n_target_components=(2,))
    #1-lag, 4 signals, 3 seeds, 2 targets
        with pytest.raises(ValueError, match = "Please insert match!!!!!"):
            data, times_data = create_test_dataset_multivar(
            sfreq, n_signals=4, n_epochs=n_epochs, n_times=n_times,
            tmin=tmin, tmax=tmax,
            fstart=fstart, fend=fend, trans_bandwidth=trans_bandwidth, shift=None)
            con = multivar_spectral_connectivity_epochs(
                data, method=method, mode=mode, indices=([[0,2]], [[1,3]]), sfreq=sfreq,
                mt_adaptive=adaptive, mt_low_bias=True,
                mt_bandwidth=mt_bandwidth, cwt_freqs=cwt_freqs,
                cwt_n_cycles=cwt_n_cycles, n_seed_components=(3,), 
                n_target_components=(2,))
    #1-lag, 4 signals, 2 seeds, 3 targets
        with pytest.raises(ValueError, match = "Please insert match!!!!!"):
            data, times_data = create_test_dataset_multivar(
            sfreq, n_signals=4, n_epochs=n_epochs, n_times=n_times,
            tmin=tmin, tmax=tmax,
            fstart=fstart, fend=fend, trans_bandwidth=trans_bandwidth, shift=None)
            con = multivar_spectral_connectivity_epochs(
                data, method=method, mode=mode, indices=([[0,2]], [[1,3]]), sfreq=sfreq,
                mt_adaptive=adaptive, mt_low_bias=True,
                mt_bandwidth=mt_bandwidth, cwt_freqs=cwt_freqs,
                cwt_n_cycles=cwt_n_cycles, n_seed_components=(2,), 
                n_target_components=(3,))
        