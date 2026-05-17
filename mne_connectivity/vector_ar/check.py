import numpy as np

from mne_connectivity import vector_auto_regression

rng = np.random.RandomState(0)
n_epochs, n_signals, n_times = 2, 3, 64
data = rng.randn(n_epochs, n_signals, n_times)
times = np.arange(n_times)

conn_dyn = vector_auto_regression(data, model="dynamic", lags=4)
conn_avg = vector_auto_regression(data, model="avg-epochs", lags=4)
out_avg = conn_avg.predict(data)
out_dyn = conn_dyn.predict(data)
print("jeff")
