
import numpy as np
from numpy.testing import assert_array_almost_equal
import mne
from mne import make_fixed_length_epochs
from mne_bids import BIDSPath, read_raw_bids

from frites.conn import conn_spec

from mne_connectivity import spectral_connectivity_epochs


def main():
    # paths to mne datasets - sample ECoG
    bids_root = mne.datasets.epilepsy_ecog.data_path()

    # first define the BIDS path
    bids_path = BIDSPath(root=bids_root, subject='pt1', session='presurgery',
                         task='ictal', datatype='ieeg', extension='vhdr')

    # Then we'll use it to load in the sample dataset. Here we use a format (iEEG)
    # that is only available in MNE-BIDS 0.7+, so it will emit a warning on
    # versions <= 0.6
    raw = read_raw_bids(bids_path=bids_path, verbose=False)

    line_freq = raw.info['line_freq']
    print(f'Data has a power line frequency at {line_freq}.')

    # Pick only the ECoG channels, removing the ECG channels
    raw.pick_types(ecog=True)

    # drop bad channels
    raw.drop_channels(raw.info['bads'])

    # Load the data
    raw.load_data()

    # Then we remove line frequency interference
    raw.notch_filter(line_freq)

    # crop data
    raw = raw.crop(tmin=0, tmax=10, include_tmax=False)

    epochs = make_fixed_length_epochs(raw=raw, duration=2., overlap=1.)

    # run frites analysis
    epoch_data = epochs.get_data()
    freqs = [30, 90]
    frite_conn = conn_spec(epoch_data, freqs=freqs, sfreq=raw.info['sfreq'])

    print(epoch_data.shape)
    print(frite_conn.shape)
    frite_data = frite_conn.data
    np.save('./test_frite_dataset.npy', frite_data)

    # run mne-connectivity version
    conn = spectral_connectivity_epochs(epochs, freqs=freqs, n_jobs=1)
    assert_array_almost_equal(conn, frite_conn)


def test_inner_funcs():
    # paths to mne datasets - sample ECoG
    bids_root = mne.datasets.epilepsy_ecog.data_path()

    # first define the BIDS path
    bids_path = BIDSPath(root=bids_root, subject='pt1', session='presurgery',
                         task='ictal', datatype='ieeg', extension='vhdr')

    # Then we'll use it to load in the sample dataset. Here we use a format (iEEG)
    # that is only available in MNE-BIDS 0.7+, so it will emit a warning on
    # versions <= 0.6
    raw = read_raw_bids(bids_path=bids_path, verbose=False)

    line_freq = raw.info['line_freq']
    print(f'Data has a power line frequency at {line_freq}.')

    # Pick only the ECoG channels, removing the ECG channels
    raw.pick_types(ecog=True)

    # drop bad channels
    raw.drop_channels(raw.info['bads'])

    # Load the data
    raw.load_data()

    # Then we remove line frequency interference
    raw.notch_filter(line_freq)

    # crop data
    raw = raw.crop(tmin=0, tmax=10, include_tmax=False)

    epochs = make_fixed_length_epochs(raw=raw, duration=2., overlap=1.)

    from mne_connectivity.spectral import _spectral_connectivity, _create_kernel
    data = epochs.get_data()
    method = 'coh'
    mode = 'cwt_morlet'
    decim = 1
    sm_times = 0.5
    sfreq = epochs.info['sfreq']
    sm_freqs = 1
    times = epochs.times
    sm_kernel = 'hanning'
    freqs = [30, 90]
    n_cycles = 7
    foi_idx = [0]
    source_idx = [0, 1, 2, 3]
    target_idx = [1, 2, 3, 4]

    # convert kernel width in time to samples
    if isinstance(sm_times, (int, float)):
        sm_times = int(np.round(sm_times * sfreq))

    # convert frequency smoothing from hz to samples
    if isinstance(sm_freqs, (int, float)):
        sm_freqs = int(np.round(max(sm_freqs, 1)))

    # temporal decimation
    if isinstance(decim, int):
        times = times[::decim]
        sm_times = int(np.round(sm_times / decim))
        sm_times = max(sm_times, 1)
    mt_bandwidth = 4

    # Create smoothing kernel
    kernel = _create_kernel(sm_times, sm_freqs, kernel=sm_kernel)

    conn = _spectral_connectivity(data, method, kernel, foi_idx,
                                  source_idx, target_idx,
                                  mode, sfreq, freqs, n_cycles, mt_bandwidth=mt_bandwidth,
                                  decim=1, kw_cwt={}, kw_mt={}, n_jobs=1,
                                  verbose=False)
    # merge results
    conn = np.stack(conn, axis=1)

    print('New computed connectivity...')
    print(conn.shape)

    from frites.conn.conn_spec import _coh
    from frites.conn.conn_tf import _tf_decomp
    mode = 'morlet'
    w = _tf_decomp(
        data, sfreq, freqs, n_cycles=n_cycles, decim=decim,
        mode=mode, mt_bandwidth=mt_bandwidth,
        n_jobs=1)

    # computes conn across trials
    kw_para = dict(n_jobs=1, verbose=False)
    conn_tr = _coh(w, kernel, foi_idx, source_idx, target_idx, kw_para)
    conn_tr = np.stack(conn_tr, axis=1)
    print(conn_tr.shape)
    assert_array_almost_equal(conn, conn_tr)


if __name__ == '__main__':
    test_inner_funcs()
    main()
