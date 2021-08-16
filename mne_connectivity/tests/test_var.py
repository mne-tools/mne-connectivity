import numpy as np
import pytest
from numpy.testing import (
    assert_array_almost_equal, assert_array_equal,
    assert_almost_equal
)

from mne_connectivity import vector_auto_regression


def create_noisy_data(
    add_noise,
    asymmetric=False,
    sigma=1e-4,
    m=100
):
    """Create noisy test data.

    Generate a 2x2 linear system, and perturb
    observations of the state variables with Gaussian noise.

    Parameters
    ----------
    add_noise : bool
        Whether to add noise or not.
    sigma : float
        noise standard deviation.
    m : int
        The number of samples.
    return_A : bool
        Whether to return the A matrix

    Returns
    -------
    sample_data : np.ndarray
        Observed sample data. Possibly with noise.
    sample_eigs : np.ndarray
        The true eigenvalues of the system.
    sample_A : np.ndarray
        (Optional) if ``return_A`` is True, then returns the
        true linear system matrix.
    """
    mu = 0.0
    noise = np.random.normal(mu, sigma, m)  # gaussian noise
    if asymmetric:
        A = np.array([[1.0, 1.5], [-1.0, 2.0]])
    else:
        A = np.array([[1.0, 1.0], [-1.0, 2.0]])
    A /= np.sqrt(3)

    # compute true eigenvalues
    true_eigvals = np.linalg.eigvals(A)

    n = 2
    X = np.zeros((n, m))
    X[:, 0] = np.array([0.5, 1.0])
    # evolve the system and perturb the data with noise
    for k in range(1, m):
        X[:, k] = A.dot(X[:, k - 1])

        if add_noise:
            X[:, k - 1] += noise[k - 1]

    return X.T, true_eigvals, A


def test_vector_auto_regression_computation():
    """Test VAR model computation accuracy.

    Tests eigenvalue and state matrix recovery.
    """
    np.random.RandomState(12345)
    sample_data, sample_eigs, sample_A = create_noisy_data(
        add_noise=True)

    # create 3D array input
    sample_data = sample_data.T[np.newaxis, ...]

    # compute the model
    model = vector_auto_regression(sample_data)

    # test the recovered model
    assert_array_almost_equal(
        model.get_data(output='dense').squeeze(), sample_A,
        decimal=2)

    # get the eigenvalues and test accuracy of recovery
    eigs = np.linalg.eigvals(model.get_data(output='dense').squeeze())
    eigvals_diff = np.linalg.norm(eigs - sample_eigs)
    assert_almost_equal(
        np.linalg.norm(eigs[1]), np.linalg.norm(sample_eigs[1]), decimal=2
    )
    assert_almost_equal(eigvals_diff, 0, decimal=2)


def test_vector_auto_regression():
    """Test the var function."""
    rng = np.random.RandomState(0)
    n_epochs, n_signals, n_times = 2, 4, 64
    data = rng.randn(n_epochs, n_signals, n_times)
    times = np.arange(n_times)

    with pytest.raises(RuntimeError, match='If computing time'):
        vector_auto_regression(data)
    with pytest.raises(ValueError, match='"model" parameter'):
        vector_auto_regression(data, model='static')

    # compute time-varying var
    conn = vector_auto_regression(data, times=times)

    # parallel conn should be exactly the same
    parr_conn = vector_auto_regression(data, times=times, n_jobs=-1)
    assert_array_equal(parr_conn.get_data(), conn.get_data())

    # compute connectivity with forward-backwards operator
    vector_auto_regression(data, times=times,
                           compute_fb_operator=True, n_jobs=-1)

    # compute single var
    single_conn = vector_auto_regression(data, model='avg-epochs')
    assert_array_almost_equal(conn.get_data().mean(axis=0),
                              single_conn.get_data(), decimal=1)

    # compute residuals
    residuals = data - parr_conn.predict(data)
    assert residuals.shape == data.shape

    # Dynamic "Connectivity" errors
    with pytest.raises(ValueError, match='Data passed in must be'):
        parr_conn.predict(np.zeros((4,)))
    with pytest.raises(RuntimeError, match='If there is a VAR model'):
        parr_conn.predict(np.zeros((4, 4)))
    with pytest.raises(RuntimeError, match='If there is a single VAR'):
        single_conn.predict(data)

    # prediction should work with a 2D array when non epoched
    single_conn.predict(rng.randn(n_signals, n_times))

    # simulate data
    sim_data = parr_conn.simulate(n_samples=100)
    sim_data = parr_conn.simulate(n_samples=100, noise_func=np.random.normal)
    assert sim_data.shape == (4, 100)

    # simulate data over many epochs
    big_epoch_data = rng.randn(n_times * 2, n_signals, n_times)
    parr_conn = vector_auto_regression(big_epoch_data, times=times, n_jobs=-1)
    parr_conn.predict(big_epoch_data)
