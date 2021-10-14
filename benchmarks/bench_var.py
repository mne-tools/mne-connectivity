import numpy as np
from memory_profiler import profile

from statsmodels.tsa.vector_ar.var_model import VAR
from mne_connectivity import vector_auto_regression


@profile
def run_experiment(data, times):
    """Run RAM experiment.

    python -m memory_profiler ./benchmarks/bench_var.py
    """
    # compute time-varying var
    conn = vector_auto_regression(data, times=times, lags=5)


@profile
def run_sm_experiment(sample_data):
    """Run RAM expeirment with statsmodels."""
    # statsmodels feeds in (n_samples, n_channels)
    sm_var = VAR(endog=sample_data.squeeze().T)
    sm_params = sm_var.fit(maxlags=5, trend='n')


if __name__ == '__main__':
    rng = np.random.RandomState(0)
    n_epochs, n_signals, n_times = 1, 500, 1000
    data = rng.randn(n_epochs, n_signals, n_times)
    times = np.arange(n_times)

    # use conn and statsmodels with pytest memory
    run_experiment(data, times)
    # run_sm_experiment(data)
