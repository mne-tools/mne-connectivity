"""
=====================================================================
Generate Test Dataset for Spectral Connectivity Over Time With Frites
=====================================================================

As of v0.3, mne-connectivity ported over an implemenetation for 
spectral connectivity over epochs. This file will generate the test
dataset used with Frites v0.4.1.
"""
import mne
import numpy as np
from frites.conn import conn_spec
from mne import make_fixed_length_epochs
from mne_bids import BIDSPath, read_raw_bids
from numpy.testing import assert_array_almost_equal

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
    raw = raw.pick_channels(raw.ch_names[:3])

    # Load the data
    raw.load_data()

    # Then we remove line frequency interference
    raw.notch_filter(line_freq)

    # crop data
    raw = raw.crop(tmin=0, tmax=4, include_tmax=False)

    epochs = make_fixed_length_epochs(raw=raw, duration=2., overlap=1.)

    # run frites analysis
    epoch_data = epochs.get_data()
    n_signals = len(epochs.ch_names)
    freqs = [30, 90]
    for method in ['coh', 'plv', 'sxy']:
        for mode in ['morlet', 'multitaper']:
            frite_conn = conn_spec(epoch_data, freqs=freqs, sfreq=raw.info['sfreq'], metric=method, mode=mode)

            print(epoch_data.shape)
            print(frite_conn.shape)
            frite_data = frite_conn.data
            np.save(f'./mne_connectivity/tests/data/test_frite_dataset_{mode}_{method}.npy', frite_data)

            # run mne-connectivity version
            if mode == 'morlet':
                mode = 'cwt_morlet'
            conn = spectral_connectivity_epochs(epochs, freqs=freqs, n_jobs=1, method=method, mode=mode)

            # test the simulated signal
            row_triu_inds, col_triu_inds = np.triu_indices(len(raw.ch_names), k=1)
            # triu_inds_ravel = np.ravel_multi_index(triu_inds, dims=(n_signals, n_signals)).astype(int)

            print(conn.get_data(output='dense').shape)
            print(frite_conn.shape)
            # print(triu_inds_ravel)
            conn_data = conn.get_data(output='dense')[:, row_triu_inds, col_triu_inds, ...]
            print(conn_data.shape)
            # conn_data = conn_data[: triu_inds_ravel, ...]
            assert_array_almost_equal(conn_data, frite_conn)


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

    from mne_connectivity.spectral.time import _create_kernel, _spectral_connectivity
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
    # test_inner_funcs()
    main()
