"""
.. _ex-var:

===================================================
Compute vector autoregressive model (linear system)
===================================================

Compute a VAR (linear system) model from time-series
activity :footcite:`LiEtAl2017`.

In this example, we will demonstrate how to compute
a VAR model with different statistical assumptions.
"""

# Authors: Adam Li <adam2392@gmail.com>
#          Alex Rockhill <aprockhill@mailbox.org>
#
# License: BSD (3-clause)

import numpy as np

import mne
from mne.datasets import sample

import matplotlib.pyplot as plt

from mne_connectivity import vector_auto_regression

# %%
# Load the data
# -------------
# Here, we first download a somatosensory dataset.
#
# Then, we will do some basic filtering and preprocessing using MNE-Python.

data_path = sample.data_path()
raw_fname = data_path / 'MEG' / 'sample' / \
    'sample_audvis_filt-0-40_raw.fif'
event_fname = data_path / 'MEG' / 'sample' / \
    'sample_audvis_filt-0-40_raw-eve.fif'

# Setup for reading the raw data
raw = mne.io.read_raw_fif(raw_fname)
events = mne.read_events(event_fname)

# Add a bad channel
raw.info['bads'] += ['MEG 2443']

# Pick MEG gradiometers
picks = mne.pick_types(raw.info, meg='grad', eeg=False, stim=False, eog=True,
                       exclude='bads')

# Create epochs for the visual condition
event_id, tmin, tmax = 3, -0.2, 1.5  # need a long enough epoch for 5 cycles
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                    baseline=(None, 0), reject=dict(grad=4000e-13, eog=150e-6))


# %%
# Compute the VAR model for all windows
# -------------------------------------
# Now, we are ready to compute our VAR model. We will compute a VAR model for
# each Epoch and return an EpochConnectivity data structure. Each Epoch here
# represents a separate VAR model. Taken together, these represent a
# time-varying linear system.

conn = vector_auto_regression(
    data=epochs.get_data(), times=epochs.times, names=epochs.ch_names)

# this returns a connectivity structure over time
print(conn)

# %%
# Evaluate the VAR model fit
# ---------------------------
# We can now evaluate the model fit by computing the residuals of the model and
# visualizing them. In addition, we can evaluate the covariance of the
# residuals. This will compute an independent VAR model for each epoch (window)
# of data.

predicted_data = conn.predict(epochs.get_data())

# compute residuals
residuals = epochs.get_data() - predicted_data

# visualize the residuals
fig, ax = plt.subplots()
ax.plot(residuals.flatten(), '*')
ax.set(
    title='Residuals of fitted VAR model',
    ylabel='Magnitude'
)

# compute the covariance of the residuals
model_order = conn.attrs.get('model_order')
t = residuals.shape[0]
sampled_residuals = np.concatenate(
    np.split(residuals[:, :, model_order:], t, 0),
    axis=2
).squeeze(0)
rescov = np.cov(sampled_residuals)

# Next, we visualize the covariance of residuals.
# Here we will see that because we use ordinary
# least-squares as an estimation method, the residuals
# should come with low covariances.
fig, ax = plt.subplots()
cax = fig.add_axes([0.27, 0.8, 0.5, 0.05])
im = ax.imshow(rescov, cmap='viridis', aspect='equal', interpolation='none')
fig.colorbar(im, cax=cax, orientation='horizontal')

# %%
# Estimate significant connections using a time-shuffled null distribution
# ------------------------------------------------------------------------
# We can maintain autocorrelations by shuffling the channels in time relative
# to one another as explained in :footcite:`RecanatesiEtAl2022`. The pairwise
# correlations will then be an estimate of connectivity under a null model
# of uncorrelated neural data.

null_dist = list()
data = epochs.get_data()
rng = np.random.default_rng(99)
for niter in range(10):  # 1000 or more would be reasonable for a real analysis
    print(f'Computing null connectivity {niter}')
    for epo_idx in range(data.shape[0]):
        for ch_idx in range(data.shape[1]):
            # pick a random starting time for each epoch and channel
            start_idx = np.round(rng.random() * data.shape[2]).astype(int)
            data[epo_idx, ch_idx] = np.concatenate(
                [data[epo_idx, ch_idx, start_idx:],
                 data[epo_idx, ch_idx, :start_idx]])
    null_dist.append(vector_auto_regression(
        data=data, times=epochs.times, names=epochs.ch_names).get_data())

null_dist = np.array(null_dist)

# %%
# Visualize significant connections over time with animation
# ----------------------------------------------------------
# Let's animate over time to visualize the significant connections at each
# epoch.

con_data = conn.get_data()

# to bonferroni correct across epochs, use the following:
threshes = np.quantile(abs(null_dist), 1 - (0.05 / con_data.shape[0]),
                       axis=(0, 1))

# now, plot the connectivity as it changes for each epoch
n_lines = np.sum(abs(con_data) > threshes, axis=(1, 2))
fig, ax = plt.subplots(subplot_kw=dict(projection='polar'), figsize=(10, 10))
anim, ax = conn.plot_circle(n_lines=n_lines, fontsize_names=4,
                            fig=fig, ax=ax)

# %%
# Compute one VAR model using all epochs
# --------------------------------------
# By setting ``model='dynamic'``, we instead treat each Epoch as a sample of
# the same VAR model and thus we only estimate one VAR model. One might do this
# when one suspects the data is stationary and one VAR model represents all
# epochs.

conn = vector_auto_regression(
    data=epochs.get_data(), times=epochs.times, names=epochs.ch_names,
    model='avg-epochs')

# this returns a connectivity structure over time
print(conn)

# %%
# Evaluate model fit again
# ------------------------
# We can now evaluate the model fit again as done earlier. This model fit will
# of course have higher residuals than before as we are only fitting 1 VAR
# model to all the epochs.

first_epoch = epochs.get_data()[0, ...]
predicted_data = conn.predict(first_epoch)

# compute residuals
residuals = epochs.get_data() - predicted_data

# visualize the residuals
fig, ax = plt.subplots()
ax.plot(residuals.flatten(), '*')
ax.set(
    title='Residuals of fitted VAR model',
    ylabel='Magnitude'
)

# compute the covariance of the residuals
model_order = conn.attrs.get('model_order')
t = residuals.shape[0]
sampled_residuals = np.concatenate(
    np.split(residuals[:, :, model_order:], t, 0),
    axis=2
).squeeze(0)
rescov = np.cov(sampled_residuals)

# Next, we visualize the covariance of residuals as before.
# Here we will see a similar trend with the covariances as
# with the covariances for time-varying VAR model.
fig, ax = plt.subplots()
cax = fig.add_axes([0.27, 0.8, 0.5, 0.05])
im = ax.imshow(rescov, cmap='viridis', aspect='equal', interpolation='none')
fig.colorbar(im, cax=cax, orientation='horizontal')
