import numpy as np
import pytest
from numpy.testing import (
    assert_array_almost_equal, assert_array_equal
)

from mne_connectivity import var


def test_var():
    """Test the var function."""
    rng = np.random.RandomState(0)
    n_epochs, n_signals, n_times = 2, 4, 64
    data = rng.randn(n_epochs, n_signals, n_times)
    times = np.arange(n_times)

    with pytest.raises(RuntimeError, match='If computing time'):
        var(data)
    with pytest.raises(ValueError, match='"model" parameter'):
        var(data, model='static')

    # compute time-varying var
    conn = var(data, times=times)

    # parallel conn should be exactly the same
    parr_conn = var(data, times=times, n_jobs=-1)
    assert_array_equal(parr_conn.get_data(), conn.get_data())

    # compute connectivity with forward-backwards operator
    var(data, times=times, compute_fb_operator=True, n_jobs=-1)

    # compute single var
    single_conn = var(data, model='avg-epochs')
    assert_array_almost_equal(conn.get_data().mean(axis=0),
                              single_conn.get_data(), decimal=1)

    # compute residuals
    residuals = data - parr_conn.predict(data)
    assert np.max(np.abs(residuals)) < 1

    # simulate data
    sim_data = parr_conn.simulate(n_samples=100)
    sim_data = parr_conn.simulate(n_samples=100, noise_func=np.random.normal)

    assert sim_data.shape == (4, 100)
