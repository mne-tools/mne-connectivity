"""Test frequency-domain multivariate connectivity methods."""
import mne
import numpy as np
import pytest
from mne.filter import filter_data
from mne_connectivity import (
    MultivariateSpectralConnectivity, multivariate_spectral_connectivity_epochs,
    read_connectivity
)
from numpy.testing import assert_array_almost_equal, assert_array_less




def create_test_dataset_multivariate(sfreq, n_signals, n_epochs, n_times, tmin, tmax,
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


class TestMultivarSpectralConnectivity:
    sfreq = 50.
    n_signals = 4
    n_epochs = 8
    n_times = 200
    trans_bandwidth = 2.
    tmin = 0.
    tmax = (n_times - 1) / sfreq
    fstart = 5.0
    fend = 10.0
    test_data, test_times = create_test_dataset_multivariate(
                sfreq, n_signals=n_signals, n_epochs=n_epochs, 
                n_times=n_times, tmin=tmin, tmax=tmax, 
                fstart=fstart, fend=fend, 
                trans_bandwidth=trans_bandwidth, shift=None
                )


    def test_invalid_method_or_mode(self):      
        class _InvalidClass:
            pass

        with pytest.raises(
            ValueError, 
            match='is not a valid connectivity method'
            ):
            multivariate_spectral_connectivity_epochs(
                self.test_data, indices=([[0,2]], [[1,3]]), method='notamethod'
                )

        with pytest.raises(
            ValueError, 
            match='The supplied connectivity method does not have the method'
            ):
            multivariate_spectral_connectivity_epochs(
                self.test_data, indices=([[0,2]], [[1,3]]), method=_InvalidClass,
                )

        with pytest.raises(
            ValueError, 
            match='mode has an invalid value'
            ):
            multivariate_spectral_connectivity_epochs(
                self.test_data, indices=([[0,2]], [[1,3]]), mode='notamode'
                )


    def test_invalid_fmin_or_fmax(self):

        with pytest.raises(
            ValueError, 
            match='There are no frequency points between'
            ):
            multivariate_spectral_connectivity_epochs(
                self.test_data, indices=([[0,2]], [[1,3]]), fmin=10,
                fmax=10 + 0.5 * (self.sfreq / float(self.n_times))
                )

        with pytest.raises(ValueError, match='fmax must be larger than fmin'):
            multivariate_spectral_connectivity_epochs(
                self.test_data, indices=([[0,2]], [[1,3]]), fmin=10, fmax=5
                )

        with pytest.raises(ValueError, match='fmax must be larger than fmin'):
            multivariate_spectral_connectivity_epochs(
                self.test_data, 
                indices=([[0,2]], [[1,3]]), fmin=(0, 11), fmax=(5, 10)
                )

        with pytest.raises(
            ValueError, 
            match='fmin and fmax must have the same length'
            ):
            multivariate_spectral_connectivity_epochs(
                self.test_data, indices=([[0,2]], [[1,3]]), fmin=(11,), 
                fmax=(12, 15)
                )


    def test_invalid_indices(self):
        # Indices cannot be None
        with pytest.raises(
            ValueError, match='indices must be specified'):
            multivariate_spectral_connectivity_epochs(
                self.test_data, indices=None
            )

        with pytest.raises(
            ValueError, match='the number of seeds'):
            multivariate_spectral_connectivity_epochs(
                self.test_data, indices=([[0,2]], [[1,3], [1,3]])
            )

        with pytest.raises(
            TypeError,
            match='each connection must be given as a list of ints'
            ):
            multivariate_spectral_connectivity_epochs(
                self.test_data, indices=([[0,2]], [(1,3)])
            )

        with pytest.raises(
            TypeError,
            match='each connection must be given as a list of ints'
            ):
            multivariate_spectral_connectivity_epochs(
                self.test_data, indices=([[0,2]], [[1,3.0]])
            )

        with pytest.raises(
            ValueError,
            match='there are common indices present in the seeds and targets'
            ):
            multivariate_spectral_connectivity_epochs(
                self.test_data, indices=([[0,2]], [[0,3]])
            )
        

    def test_compute_separate_gc_csd_and_connectivity(self):
        multivariate_spectral_connectivity_epochs(
            self.test_data, indices=([[0,2]], [[1,3]]), sfreq=self.sfreq,
            n_components="rank", method="gc"
        )


    def test_n_components(self):
        """Tests that n_components cannot be of the wrong type and that there
        cannot be too few or too many components requested."""
        # Check 1 seed, 1 target component
        n_components = ([1], [1])
        con = multivariate_spectral_connectivity_epochs(
            self.test_data, indices=([[0,2]], [[1,3]]), sfreq=self.sfreq,
            n_components=n_components
            )
        assert(con.n_components == n_components)

        # Check 2 seed, 2 target components
        n_components = ([2], [2])
        con = multivariate_spectral_connectivity_epochs(
            self.test_data, indices=([[0,2]], [[1,3]]), sfreq=self.sfreq,
            n_components=n_components
            )
        assert(con.n_components == n_components)


        # Check that string 'rank' works for n_components
        multivariate_spectral_connectivity_epochs(
            self.test_data, indices=([[0,2]], [[1,3]]), sfreq=self.sfreq,
            n_components='rank'
            )

        # Check too many seed components
        with pytest.raises(ValueError, 
            match='the number of components to take cannot '):
            multivariate_spectral_connectivity_epochs(
                self.test_data, indices=([[0,2]], [[1,3]]), sfreq=self.sfreq,
                n_components=([3], [2])
                )

        # Check too many target components
        with pytest.raises(ValueError, 
            match='the number of components to take cannot '):
            multivariate_spectral_connectivity_epochs(
                self.test_data, indices=([[0,2]], [[1,3]]), sfreq=self.sfreq,
                n_components=([2], [3])
                )

        # Check n_components < 0
        with pytest.raises(ValueError, match='must be greater than 0'):
            multivariate_spectral_connectivity_epochs(
                self.test_data, indices=([[0,2]], [[1,3]]), sfreq=self.sfreq,
                n_components=([-1], [1])
                )

        # Check wrong length of n_seed_components
        with pytest.raises(ValueError,
            match='entries of n_components must have the '):
            multivariate_spectral_connectivity_epochs(
                self.test_data, indices=([[0,2]], [[1,3]]), sfreq=self.sfreq,
                n_components=([2, 2], [2])
                )

        # Check wrong length of n_target_components
        with pytest.raises(ValueError,
            match='entries of n_components must have the '):
            multivariate_spectral_connectivity_epochs(
                self.test_data, indices=([[0,2]], [[1,3]]), sfreq=self.sfreq,
                n_components=([2], [2, 2])
                )

        # Check n_components is a tuple
        with pytest.raises(TypeError, match='n_components must be a tuple'):
            multivariate_spectral_connectivity_epochs(
                self.test_data, indices=([[0,2]], [[1,3]]), sfreq=self.sfreq,
                n_components=[2, 2]
            )

        # Check n_components entries are lists
        with pytest.raises(TypeError,
            match='entries of n_components must be lists'):
            multivariate_spectral_connectivity_epochs(
                self.test_data, indices=([[0,2]], [[1,3]]), sfreq=self.sfreq,
                n_components=(2, 2)
                )

        # Check that invalid string raises error for n_components
        with pytest.raises(ValueError, match='must be the string "rank"'):
            multivariate_spectral_connectivity_epochs(
                self.test_data, indices=([[0,2]], [[1,3]]), sfreq=self.sfreq,
                n_components=(['rank'], ['invalid'])
            )

        # Check that invalid type (e.g. float) raises error for n_components
        with pytest.raises(TypeError, match='must be tuples of lists of '):
            multivariate_spectral_connectivity_epochs(
                self.test_data, indices=([[0,2]], [[1,3]]), sfreq=self.sfreq,
                n_components=([2], [2.5])
            )


    @pytest.mark.parametrize('mt_adaptive', [True, False])
    @pytest.mark.parametrize('mt_low_bias', [True, False])
    def test_adaptive(self, mt_adaptive, mt_low_bias):
        if mt_adaptive:
            mt_bandwidth = 1.
        else: 
            mt_bandwidth = None
        multivariate_spectral_connectivity_epochs(
            self.test_data, indices=([[0]], [[1]]), method='mic', 
            mode='multitaper',sfreq=self.sfreq, mt_adaptive=mt_adaptive, 
            mt_low_bias=mt_low_bias, mt_bandwidth=mt_bandwidth)


    @pytest.mark.parametrize('method',
        ['mic', 'mim', 'gc', 'net_gc', 'trgc', 'net_trgc', 
        ['mic', 'mim', 'gc', 'net_gc', 'trgc', 'net_trgc']]
    )
    @pytest.mark.parametrize('mode', ['multitaper', 'fourier', 'cwt_morlet'])
    def test_methods_and_modes(self, method, mode):
        # define some frequencies for cwt
        cwt_freqs = np.arange(3, 14.5, 1)
        if method == 'mic' and mode == 'cwt_morlet':
            # so we also test using an array for num cycles
            cwt_n_cycles = 7 * np.ones(len(cwt_freqs))
        else:
            cwt_n_cycles = 7

        indices = ([[0]], [[1]])
        con = multivariate_spectral_connectivity_epochs(
            self.test_data, indices=indices, method=method, mode=mode,
            sfreq=self.sfreq, cwt_freqs=cwt_freqs,
            cwt_n_cycles=cwt_n_cycles, gc_n_lags=5
            )

        if not isinstance(method, list):
            freqs = con.attrs.get('freqs_used')
            n = con.n_epochs_used
            if isinstance(con, MultivariateSpectralConnectivity):
                times = con.attrs.get('times_used')
            else:
                times = con.times

            assert (n == self.n_epochs)
            assert_array_almost_equal(self.test_times, times)

            # Check topographies do/do not exist, and whether they have the
            # correct shape
            if method == 'mic':
                results = con.get_data()
                topographies = con.topographies
                assert (topographies is not None)
                assert (len(topographies) == 2)
                for group_topos, group_inds in zip(topographies, indices):
                    assert len(group_topos) == len(group_inds)
                    con_i = 0
                    for con_topos, con_inds in zip(group_topos, group_inds):
                        assert (
                            con_topos.shape ==
                            (len(con_inds), *results[con_i].shape)
                        )
                        con_i += 1
            else:
                assert (con.topographies is None)
            
            if method in ['gc', 'net_gc', 'trgc', 'net_trgc']:
                assert (con.n_lags is not None)
            else:
                assert (con.n_lags is None)

            # Everything below should be applicable for GC too, albeit with a
            # modification of the thresholds to reflect the fact that:
            #   - GC can be > 1 (although it is not often)
            #   - net GC, TRGC, and net TRGC values can be negative and are not
            #       bounded between +/- 1 (although, like for GC, they are not
            #       often outside of this bound)

            if method in ['mic', 'mim']:
                upper_t = 0.3
                lower_t = 0.5

                # test the simulated signal
                gidx = np.searchsorted(freqs, (self.fstart, self.fend))
                bidx = np.searchsorted(
                    freqs,
                    (self.fstart - self.trans_bandwidth * 2, 
                    self.fend + self.trans_bandwidth * 2)
                )
                
                # Check 0-lag, 2 signals
                data, _ = create_test_dataset_multivariate(
                    self.sfreq, n_signals=2, n_epochs=self.n_epochs,
                    n_times=self.n_times, tmin=self.tmin, tmax=self.tmax, 
                    fstart=self.fstart, fend=self.fend, 
                    trans_bandwidth=self.trans_bandwidth, shift=0
                )
                con = multivariate_spectral_connectivity_epochs(
                    data, indices=([[0]], [[1]]), method=method, mode=mode, 
                    sfreq=self.sfreq, cwt_freqs=cwt_freqs,
                    cwt_n_cycles=cwt_n_cycles, n_components=None
                )
                assert_array_less(
                    con.get_data(output='raveled')[ 0, :bidx[0]], lower_t
                )

                # Check 1-lag, 4 signals
                data, _ = create_test_dataset_multivariate(
                    self.sfreq, n_signals=4, n_epochs=self.n_epochs, 
                    n_times=self.n_times, tmin=self.tmin, tmax=self.tmax, 
                    fstart=self.fstart, fend=self.fend, 
                    trans_bandwidth=self.trans_bandwidth, shift=1
                )

                con = multivariate_spectral_connectivity_epochs(
                    data, indices=([[0,2]], [[1,3]]), method=method, mode=mode, 
                    sfreq=self.sfreq, cwt_freqs=cwt_freqs, 
                    cwt_n_cycles=cwt_n_cycles, n_components=None
                )
                assert np.all(con.get_data('raveled')[0, gidx[0]:gidx[1]] > upper_t), \
                    con.get_data()[0, gidx[0]:gidx[1]].min()


    def test_multiple_methods_with_svd(self):
        """Tests that calling SVD does not raise any error when multiple methods
        are called."""
        multivariate_spectral_connectivity_epochs(
            self.test_data, indices=([[0, 2]], [[1, 3]]),
            method=['gc', 'mic'], sfreq=self.sfreq, n_components='rank'
        )


    def test_invalid_n_lags(self):
        """Tests whether an invalid number of lags for GC is caught.
        n_lags cannot be >= (n_freqs - 1) * 2"""
        # use cwt_freqs so we can easily know how many freqs will be present
        freqs = np.arange(7, 10)
        with pytest.raises(ValueError, match='the number of lags'):
            multivariate_spectral_connectivity_epochs(
                self.test_data, indices=([[0]], [[1]]), method='gc',
                mode='cwt_morlet', cwt_freqs=freqs, sfreq=self.sfreq,
                gc_n_lags=len(freqs) * 2
            )
    

    def test_net_gc_mirrored(self):
        """Tests that net GC and net TRGC from [seeds -> targets] equals net GC
        and net TRGC, respectively, from [targets -> seeds]*-1 (i.e. they are
        sign flipped, mirrored around 0)."""
        seeds = [[0]]
        targets = [[1]]

        seeds_targets = multivariate_spectral_connectivity_epochs(
            self.test_data, indices=(seeds, targets),
            method=['net_gc', 'net_trgc'], sfreq = self.sfreq
        )

        targets_seeds = multivariate_spectral_connectivity_epochs(
            self.test_data, indices=(targets, seeds),
            method=['net_gc', 'net_trgc'], sfreq = self.sfreq
        )

        assert_array_almost_equal(
            seeds_targets[0].get_data(),
            targets_seeds[0].get_data() * -1
        )

        assert_array_almost_equal(
            seeds_targets[1].get_data(),
            targets_seeds[1].get_data() * -1
        )
    

    def test_non_full_rank_catch(self):
        """Tests that computing multivariate connectivity on non-full rank data
        raises errors, and that performing SVD to make the data full rank
        alleviates this."""
        # create non-full-rank data (e.g. repeat a seed or target channel)
        data = np.copy(self.test_data)
        data[:, 2, :] = data[:, 0, :] * 2
        rank_orig = np.linalg.matrix_rank(self.test_data, tol=1e-10)
        rank = np.linalg.matrix_rank(data, tol=1e-10)
        # Check our data truly doesn't have full rank
        assert np.all(rank_orig - rank == 1)

        with pytest.raises(ValueError, match='the autocovariance matrix is'):
            multivariate_spectral_connectivity_epochs(
                data, indices=([[0,2]], [[1,3]]), method='gc', sfreq=self.sfreq,
                fmin=3, fmax=20,
            )
        
        with pytest.raises(ValueError, match='the transformation matrix of'):
            multivariate_spectral_connectivity_epochs(
                data, indices=([[0,2]], [[1,3]]), method='mic', sfreq=self.sfreq
            )

        con = multivariate_spectral_connectivity_epochs(
            data, indices=([[0,2]], [[1,3]]), sfreq=self.sfreq,
            n_components='rank'
        )
        assert con.n_components == ([1], [2])


    # Could add checks that results of method calls separately match those given
    # together


    def test_parallel(self):
        """Test parallel computation."""
        n_jobs = 2
        multivariate_spectral_connectivity_epochs(
            self.test_data, indices=([[0]], [[1]]), sfreq=self.sfreq, 
            n_jobs=n_jobs
            )


    def test_epochs_object(self):
        info = mne.create_info(
            ch_names=[f"ch_{n}" for n in range(self.n_signals)],
            sfreq=self.sfreq
            )
        epochs = mne.EpochsArray(self.test_data, info)
        
        con_from_data = multivariate_spectral_connectivity_epochs(
            self.test_data, indices=([[0]], [[1]]), sfreq=self.sfreq, method="gc"
            )

        con_from_epochs = multivariate_spectral_connectivity_epochs(
            epochs, indices=([[0]], [[1]]), method="gc"
            )

        assert_array_almost_equal(
            con_from_epochs.get_data(), 
            con_from_data.get_data()
            )

        con_from_epochs = multivariate_spectral_connectivity_epochs(
            epochs, indices=([[0]], [[1]])
            )

        annotations = mne.Annotations(
            onset=[0, 3], duration=[1, 0.25],
            description=['Start', 'Noise'],
        )
        epochs.set_annotations((annotations))
        con_with_annot = multivariate_spectral_connectivity_epochs(
            epochs, indices=([[0]], [[1]]), method="gc"
            )
        assert_array_almost_equal(
            con_with_annot.get_data(), 
            con_from_data.get_data()
            )

        epochs = mne.EpochsArray(self.test_data, info)
        epochs.set_annotations((annotations))
        epochs.add_annotations_to_metadata()
        con_with_metadata = multivariate_spectral_connectivity_epochs(
            epochs, indices=([[0]], [[1]]), method="gc"
        )
        assert_array_almost_equal(
            con_with_metadata.get_data(), 
            con_from_data.get_data()
        )

        # Test SVD works with Epochs object and gc methods
        multivariate_spectral_connectivity_epochs(
            epochs, indices=([[0,2]], [[1,3]]), method="gc",
            n_components=([1], [1])
        )

        # cwt_freqs is a discontinuous array
        multivariate_spectral_connectivity_epochs(
            epochs, indices=([[0]], [[1]]), fmin=(3, 9), fmax=(7, 14), 
            method="gc", gc_n_lags=4
        )


    def test_faverage(self):
        multivariate_spectral_connectivity_epochs(
            self.test_data, indices=([[0,2]], [[1,3]]), sfreq=self.sfreq,
            fmin=(3, 8, 15), fmax=(7, 14, 20), faverage=True, method="mic"
            )

        multivariate_spectral_connectivity_epochs(
            self.test_data, indices=([[0,2]], [[1,3]]), sfreq=self.sfreq,
            fmin=(3, 8, 15), fmax=(7, 14, 20), faverage=True, method="gc"
            )

        # Add checks that performing faverage in function call matches manual
        # result, and that same is seen for MIC topographies


    def test_check_for_discontinuous_freqs(self):
        # cwt_freqs is a discontinuous array
        multivariate_spectral_connectivity_epochs(
            self.test_data, indices=([[0]], [[1]]), sfreq=self.sfreq,
            fmin=(3, 9), fmax=(7, 14), method="gc", gc_n_lags=4
            )

    def test_save(self, tmp_path):
        """Tests that saving the connectivity objects works and re-loading the
        objects gives the correct results.
        
        This is necessary given the need to pad ragged indices and topographies
        attributes, which would otherwise be converted to object arrays, and
        which the saving engine used by MNE does not support.
        """
        tmp_file = tmp_path / 'temp_file.nc'
        # generate 'ragged' connectivity stored in a
        # MultivariateSpectralConnectivity object
        con = multivariate_spectral_connectivity_epochs(
            self.test_data, indices=([[0]], [[1, 3]]), sfreq=self.sfreq
        )
        con.save(tmp_file)
        read_con = read_connectivity(tmp_file)
        assert_array_almost_equal(con.get_data(), read_con.get_data())
        assert repr(con) == repr(read_con)

        # generate 'ragged' connectivity stored in a
        # MultivariateSpectroTemporalConnectivity object
        con = multivariate_spectral_connectivity_epochs(
            self.test_data, indices=([[0]], [[1, 3]]), mode='cwt_morlet',
            cwt_freqs=np.arange(self.fstart, self.fend), sfreq=self.sfreq
        )
        con.save(tmp_file)
        read_con = read_connectivity(tmp_file)
        assert_array_almost_equal(con.get_data(), read_con.get_data())
        assert repr(con) == repr(read_con)

test = TestMultivarSpectralConnectivity
test.test_n_components(test)