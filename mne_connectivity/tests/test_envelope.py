# Authors: Eric Larson <larson.eric.d@gmail.com>
#          Sheraz Khan <sheraz@khansheraz.com>
#          Denis Engemann <denis.engemann@gmail.com>
#          Adam Li <adam2392@gmail.com>
#
# License: BSD (3-clause)

from mne_connectivity.io import read_connectivity
import numpy as np
import pytest
from numpy.testing import (assert_allclose, assert_array_equal,
                           assert_array_less)
from scipy.signal import hilbert

from mne.utils import catch_logging, use_log_level
from mne_connectivity.envelope import envelope_correlation, symmetric_orth


def _compute_corrs_orig(data):
    # This is the version of the code by Sheraz and Denis.
    # For this version (epochs, labels, time) must be -> (labels, time, epochs)
    n_epochs, n_labels, _ = data.shape
    corr = np.zeros((n_labels, n_labels))
    for epoch_data in data:
        for ii in range(n_labels):
            for jj in range(n_labels):
                # Get timeseries for each pair
                x, y = epoch_data[ii], epoch_data[jj]
                x_mag = np.abs(x)
                x_conj_scaled = x.conj()
                x_conj_scaled /= x_mag
                # Calculate orthogonalization
                y_orth_x = (y * x_conj_scaled).imag
                y_orth_x_mag = np.abs(y_orth_x)
                # Estimate correlation
                corr[ii, jj] += np.abs(np.corrcoef(x_mag, y_orth_x_mag)[0, 1])
    corr = (corr + corr.T) / (2. * n_epochs)
    corr.flat[::n_labels + 1] = 0.
    return corr


def test_roundtrip_envelope_correlation(tmp_path):
    """Test write/read roundtrip for envelope correlation."""
    rng = np.random.RandomState(0)
    n_epochs, n_signals, n_times = 1, 4, 64
    data = rng.randn(n_epochs, n_signals, n_times)
    data_hilbert = hilbert(data, axis=-1)
    corr = envelope_correlation(data_hilbert)
    tmp_file = tmp_path / 'temp_file.nc'
    corr.save(tmp_file)

    read_corr = read_connectivity(tmp_file)
    assert_array_equal(corr.get_data(), read_corr.get_data())


def test_empty_epochs_correlation():
    """Test empty epochs object results in error."""
    rng = np.random.RandomState(0)
    n_epochs, n_signals, n_times = 0, 4, 64
    data = rng.randn(n_epochs, n_signals, n_times)
    data_hilbert = hilbert(data, axis=-1)

    with pytest.raises(RuntimeError, match='Passing in empty epochs'):
        envelope_correlation(data_hilbert)


def test_envelope_correlation():
    """Test the envelope correlation function."""
    rng = np.random.RandomState(0)
    n_epochs, n_signals, n_times = 2, 4, 64
    data = rng.randn(n_epochs, n_signals, n_times)
    data_hilbert = hilbert(data, axis=-1)
    corr_orig = _compute_corrs_orig(data_hilbert)
    assert (0 <= corr_orig).all()
    assert (corr_orig < 1).all()

    # upper triangular indices to access the "corr_orig"
    triu_inds = np.triu_indices(n_signals, k=0)
    raveled_triu_inds = np.ravel_multi_index(
        triu_inds, dims=(n_signals, n_signals))
    condensed_n_estimates = len(raveled_triu_inds)

    # using complex data
    corr = envelope_correlation(data_hilbert)
    assert_allclose(
        np.mean(corr.get_data(output='raveled'), axis=0).squeeze(),
        corr_orig.flatten()[raveled_triu_inds])

    # do Hilbert internally, and don't combine
    corr = envelope_correlation(data)
    assert corr.shape == (data.shape[0],) + \
        (condensed_n_estimates,) + (1,)
    corr = np.mean(corr.get_data(output='dense'), axis=0)
    assert_allclose(corr.squeeze(), corr_orig)

    # degenerate
    with pytest.raises(ValueError, match='dtype must be float or complex'):
        envelope_correlation(data.astype(int))
    with pytest.raises(ValueError, match='entry in data must be 2D'):
        envelope_correlation(data[np.newaxis])
    with pytest.raises(ValueError, match='n_nodes mismatch'):
        envelope_correlation([rng.randn(2, 8), rng.randn(3, 8)])
    with pytest.raises(ValueError, match='Invalid value.*orthogonalize.*'):
        envelope_correlation(data, orthogonalize='foo')

    # test non-orthogonal computation
    corr_plain = envelope_correlation(data, orthogonalize=False)
    assert corr_plain.shape == (data.shape[0],) + \
        (condensed_n_estimates,) + (1,)
    assert corr_plain.get_data(output='dense').shape == \
        (data.shape[0],) + \
        (corr_orig.shape[0], corr_orig.shape[1],) + (1,)
    assert np.min(corr_plain.get_data()) < 0
    corr_plain_mean = np.mean(corr_plain.get_data(output='dense'), axis=0)
    assert_allclose(np.diag(corr_plain_mean.squeeze()), 1)
    np_corr = np.array([np.corrcoef(np.abs(x)) for x in data_hilbert])
    assert_allclose(corr_plain.get_data(output='dense').squeeze(), np_corr)

    # test resulting Epoch -> non-Epoch data structure
    # using callable
    corr = envelope_correlation(data_hilbert)
    corr_combine = corr.combine(combine=lambda data: np.mean(data, axis=0))
    assert_allclose(corr_combine.get_data(output='dense').squeeze(),
                    corr_orig)
    with pytest.raises(ValueError, match='Combine option'):
        corr.combine(combine=1.)
    with pytest.raises(ValueError, match='Combine option'):
        corr.combine(combine='foo')

    # check against FieldTrip, which uses the square-log-norm version
    # from scipy.io import savemat
    # savemat('data.mat', dict(data_hilbert=data_hilbert))
    # matlab
    # load data
    # ft_connectivity_powcorr_ortho(reshape(data_hilbert(1,:,:), [4, 64]))
    # ft_connectivity_powcorr_ortho(reshape(data_hilbert(2,:,:), [4, 64]))
    ft_vals = np.array([
        [[np.nan, 0.196734553900236, 0.063173148355451, -0.242638384630448],
         [0.196734553900236, np.nan, 0.041799775495150, -0.088205187548542],
         [0.063173148355451, 0.041799775495150, np.nan, 0.090331428512317],
         [-0.242638384630448, -0.088205187548542, 0.090331428512317, np.nan]],
        [[np.nan, -0.013270857462890, 0.185200598081295, 0.140284351572544],
         [-0.013270857462890, np.nan, 0.150981508043722, -0.000671809276372],
         [0.185200598081295, 0.150981508043722, np.nan, 0.137460244313337],
         [0.140284351572544, -0.000671809276372, 0.137460244313337, np.nan]],
    ], float)
    ft_vals[np.isnan(ft_vals)] = 0
    corr_log = envelope_correlation(
        data, log=True, absolute=False)
    assert_allclose(corr_log.get_data(output='dense').squeeze(), ft_vals)


@pytest.mark.parametrize('ndim, generator', [
    (2, False),
    (3, False),
    (3, True),
])
def test_symmetric_orth(ndim, generator):
    n_ch, n_time = 5, 1000
    rng = np.random.RandomState(0)
    Z = rng.randn(n_ch, n_time)
    mixing = rng.randn(n_ch, n_ch)
    mixing = mixing @ mixing.T
    mixing += np.eye(n_ch)
    Z = mixing @ Z
    assert ndim in (2, 3)
    if generator:
        assert ndim == 3
        Z = [Z]
    elif ndim == 3:
        Z = Z[np.newaxis]
    with catch_logging() as log:
        P = symmetric_orth(Z, verbose='debug')
        if generator:
            assert not isinstance(P, np.ndarray)
            with use_log_level('debug'):
                P = np.array(list(P))
    assert isinstance(P, np.ndarray)
    if ndim == 3:
        assert P.ndim == 3
        assert P.shape[0] == 1
        Z, P = Z[0], P[0]
    log = log.getvalue()
    assert 'Convergence reached' in log
    vals = P @ P.T
    diag = np.diag(vals)
    orig = np.diag(Z @ Z.T)
    assert_array_less(diag, orig)  # some power lost
    assert_array_less(orig * 0.1, diag)  # but not too much
    off = np.triu(vals, k=1)
    assert_allclose(off, 0., atol=1e-6)
    # Degenerate cases
    with pytest.raises(RuntimeError, match='at least as many time points'):
        symmetric_orth(Z[:, :1])
    with pytest.warns(RuntimeWarning, match='did not converge'):
        symmetric_orth(Z, n_iter=1)
    Z_bad = Z.copy()
    Z_bad[0] = Z[1] + Z[2]
    with pytest.warns(RuntimeWarning, match='rank deficient'):
        symmetric_orth(Z_bad)
