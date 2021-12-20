import numpy as np
import pytest
from numpy.testing import (
    assert_array_almost_equal, assert_array_equal,
    assert_almost_equal
)
from mne.utils import assert_object_equal

from mne_connectivity import vector_auto_regression, select_order


warning_str = dict(
    sm_depr='ignore:Using or importing*.:DeprecationWarning',  # noqa
)


def bivariate_var_data():
    """A bivariate dataset for VAR estimation."""
    rng = np.random.RandomState(12345)
    e = rng.standard_normal((252, 2))
    y = np.zeros_like(e)
    y[:2] = e[:2]
    for i in range(2, 252):
        y[i] = 0.2 * y[i - 1] + 0.1 * y[i - 2] + e[i]
    return y


def create_noisy_data(
    add_noise,
    sigma=1e-4,
    m=100,
    random_state=12345,
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
    random_state : None | int | instance of ~numpy.random.RandomState
        If ``random_state`` is an :class:`int`, it will be used as a seed for
        :class:`~numpy.random.RandomState`. If ``None``, the seed will be
        obtained from the operating system (see
        :class:`~numpy.random.RandomState` for details). Default is
        ``None``.

    Returns
    -------
    sample_data : ndarray, shape (n_channels, n_samples)
        Observed sample data. Possibly with noise.
    sample_eigs : np.ndarray
        The true eigenvalues of the system.
    sample_A : np.ndarray
        (Optional) if ``return_A`` is True, then returns the
        true linear system matrix.
    """
    rng = np.random.RandomState(random_state)

    mu = 0.0
    noise = rng.normal(mu, sigma, m)  # gaussian noise
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

    return X, true_eigvals, A


# XXX: get this to work with all trends
@pytest.mark.parametrize(
    ['lags', 'trend'],
    [
        (1, 'n'),
        (2, 'n'),
        (3, 'n'),
    ]
)
@pytest.mark.filterwarnings(warning_str['sm_depr'])
def test_regression_against_statsmodels(lags, trend):
    """Test regression against any statsmodels changes in VAR model."""
    from statsmodels.tsa.vector_ar.var_model import VAR
    sample_data, _, sample_A = create_noisy_data(
        add_noise=False, random_state=12345)
    block_size = sample_data.shape[0]

    # statsmodels feeds in (n_samples, n_channels)
    sm_var = VAR(endog=sample_data.T)
    sm_params = sm_var.fit(maxlags=lags, trend=trend)

    # the returned VAR model is transposed
    sm_A = sm_params.params.T

    # compute the model
    model = vector_auto_regression(sample_data[np.newaxis, ...], lags=lags)

    # the models should match against the sample A matrix without noise
    if lags == 1 and trend == 'n':
        assert_array_almost_equal(
            model.get_data(output='dense').squeeze(),
            sample_A)

    # the models should match each other
    if lags == 1:
        assert_array_almost_equal(
            model.get_data().squeeze(),
            sm_A.squeeze())
    else:
        for idx in range(lags):
            assert_array_almost_equal(
                model.get_data().squeeze()[..., idx],
                sm_A.squeeze()[:, idx * block_size: (idx + 1) * block_size])

    if lags == 3:
        if np.max(np.abs(np.linalg.eigvals(model.companion))) < 1.:
            # the regressed model should be stable for sufficient lags
            assert model.is_stable()
        else:
            assert not model.is_stable()


@pytest.mark.filterwarnings(warning_str['sm_depr'])
@pytest.mark.parametrize(
    ['lags'],
    [
        (None,), (5,), (200,)
    ]
)
def test_regression_select_order(lags):
    from statsmodels.tsa.vector_ar.var_model import VAR
    x = bivariate_var_data()
    sm_model = VAR(endog=x)
    if lags != 200:
        results = sm_model.select_order(maxlags=lags)
    else:
        with pytest.raises(ValueError, match='maxlags is too large'):
            select_order(X=x, maxlags=lags)
        return

    sm_selected_orders = results.selected_orders

    # compare with our version
    selected_orders = select_order(X=x, maxlags=lags)
    assert_object_equal(sm_selected_orders, selected_orders)


def test_regression():
    """Regression test to prevent numerical changes for VAR.

    This was copied over from the working version that
    matches statsmodels VAR answer.
    """
    sample_data, sample_eigs, sample_A = create_noisy_data(
        add_noise=True)

    # create 3D array input
    sample_data = sample_data[np.newaxis, ...]

    # compute the model
    model = vector_auto_regression(sample_data, lags=1)

    # test the recovered model
    expected_A = np.array([[0.57733211,  0.57736152],
                           [-0.57735815, 1.15471885]])
    assert_array_almost_equal(
        model.get_data(output='dense').squeeze(),
        expected_A)

    assert_array_almost_equal(
        sample_eigs, np.linalg.eigvals(model.get_data().squeeze()))

    # without noise, the estimated A should match exactly
    sample_data, _, sample_A = create_noisy_data(
        add_noise=False)
    sample_data = sample_data[np.newaxis, ...]
    model = vector_auto_regression(sample_data)
    assert_array_almost_equal(model.get_data(output='dense').squeeze(),
                              sample_A)


def test_var_debiased():
    """Test forward backward operator."""
    sample_data, sample_eigs, _ = create_noisy_data(
        add_noise=True, sigma=1e-2)

    # test without forward backward de-biasing
    model = vector_auto_regression(sample_data[np.newaxis, ...])

    # test with forward backward de-biasing
    model_fb = vector_auto_regression(sample_data[np.newaxis, ...],
                                      compute_fb_operator=True)

    # manually solve things using pseudoinverse
    eigvals = model.eigvals()
    eigvals_fb = model_fb.eigvals()

    assert np.linalg.norm(eigvals - sample_eigs) > \
        np.linalg.norm(eigvals_fb - sample_eigs)


@pytest.mark.parametrize('l2_reg', np.linspace(0, 5, 10))
def test_var_l2_reg(l2_reg):
    """Test l2 regularization works as expected."""
    sample_data, _, _ = create_noisy_data(
        add_noise=True)

    # test l2 regularization
    model = vector_auto_regression(sample_data[np.newaxis, ...], l2_reg=l2_reg)

    # manually solve things using pseudoinverse
    sample_data = sample_data.T
    X, Y = sample_data.squeeze()[:-1, :], sample_data.squeeze()[1:, :]
    n_col = X.shape[1]
    manual_A = np.linalg.lstsq(
        X.T.dot(X) + l2_reg * np.eye(n_col), X.T.dot(Y), rcond=-1)[0].T
    assert_array_almost_equal(model.get_data().squeeze(),
                              manual_A)


@pytest.mark.timeout(15)
def test_vector_auto_regression_computation():
    """Test VAR model computation accuracy.

    Tests eigenvalue and state matrix recovery.
    """
    np.random.RandomState(12345)
    sample_data, sample_eigs, sample_A = create_noisy_data(
        add_noise=True)

    # create 3D array input
    sample_data = sample_data[np.newaxis, ...]

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
