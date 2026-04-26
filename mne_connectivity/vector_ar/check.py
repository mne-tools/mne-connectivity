import numpy as np

from mne_connectivity import vector_auto_regression

rng = np.random.RandomState(0)
n_epochs, n_signals, n_times = 2, 3, 64
data = rng.randn(n_epochs, n_signals, n_times)
times = np.arange(n_times)

conn = vector_auto_regression(data, model="dynamic", lags=4)
conn.predict(data)

print("jeff")
