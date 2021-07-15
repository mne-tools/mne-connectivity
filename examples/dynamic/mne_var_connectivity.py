"""
.. _ex-var:

===================================================
Compute vector autoregressive model (linear system)
===================================================

Compute a VAR (linear system) model from time-series
activity :footcite:`li_linear_2017` using a
continuous iEEG recording.

In this example, we will demonstrate how to compute
a VAR model with different statistical assumptions.
"""

# Authors: Adam Li <adam2392@gmail.com>
#
# License: BSD (3-clause)
import numpy as np

import mne
from mne import make_fixed_length_epochs
from mne_bids import BIDSPath, read_raw_bids

import matplotlib.pyplot as plt

from mne_connectivity import var

##############################################################################
# Load the data
# -------------
# Here, we first download an ECoG dataset that was
# recorded from a patient with epilepsy. To facilitate
# loading the data, we use `mne-bids
# # <https://mne.tools/mne-bids/>`_.
#
# Then, we will do some basic filtering and preprocessing
# via MNE-Python's API.

# paths to mne datasets - sample ECoG
bids_root = mne.datasets.epilepsy_ecog.data_path()

# first define the bids path
bids_path = BIDSPath(root=bids_root, subject='pt1', session='presurgery',
                     task='ictal', datatype='ieeg', extension='vhdr')

# then we'll use it to load in the sample dataset
# Here we use a format (iEEG) that is only available in MNE-BIDS 0.7+, so it
# will emit a warning on versions <= 0.6
raw = read_raw_bids(bids_path=bids_path, verbose=False)

line_freq = raw.info['line_freq']
print(f'Data has a power line frequency at {line_freq}.')

# Pick only the ECoG channels, removing the EKG channels
raw.pick_types(ecog=True)

# Load the data
raw.load_data()

# Then we remove line frequency interference
raw.notch_filter(line_freq)

# drop bad channels
raw.drop_channels(raw.info['bads'])

##############################################################################
# Crop the data for this example
# ------------------------------
#
# We find the onset time of the seizure and remove
# all data after that time. In this example, we
# are only interested in analyzing the interictal
# (non-seizure) data period.
#
# One might be interested in analyzing the seizure
# period also, which we leave as an exercise for
# our readers!

# Find the annotated events
events, event_id = mne.events_from_annotations(raw)

# get sample at which seizure starts
onset_id = event_id['onset']
onset_idx = np.argwhere(events[:, 2] == onset_id)
onset_sample = events[onset_idx, 0].squeeze()
onset_sec = onset_sample / raw.info['sfreq']

# remove all data after the seizure onset
raw = raw.crop(tmin=0, tmax=onset_sec, include_tmax=False)

##############################################################################
# Create Windows of Data (Epochs) Using MNE-Python
# ------------------------------------------------
# We have a continuous iEEG snapshot that is about 60
# seconds long (after cropping). We would like to estimate
# a VAR model over a sliding window of 500 milliseconds with
# a 250 millisecond step size.
#
# We can use `mne.make_fixed_length_epochs` to create an
# Epochs data structure representing this sliding window.

epochs = make_fixed_length_epochs(raw=raw, duration=0.5, overlap=0.25)
times = epochs.times
ch_names = epochs.ch_names

print(epochs)

##############################################################################
# Compute the VAR model for all windows
# -------------------------------------
# Now, we are ready to compute our VAR model.
# This will compute a VAR model for each Epoch and
# return an EpochConnectivity data structure.
# Each Epoch here represents the VAR model in the window
# of data. Taken together, these represent a
# a time-varying linear system.

conn = var(data=epochs.get_data(), times=times, names=ch_names)

# this returns a connectivity structure over time
print(conn)

##############################################################################
# Evaluate this VAR model fit
# ---------------------------
# We can now evaluate the model fit by computing
# the residuals of the model and visualizing it.
# In addition, we can evaluate the covariance of the
# residuals.

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

# visualize the covariance of residuals
fig, ax = plt.subplots()
cax = fig.add_axes([0.27, 0.8, 0.5, 0.05])
im = ax.imshow(rescov, cmap='viridis', aspect='equal', interpolation='none')
fig.colorbar(im, cax=cax, orientation='horizontal')

##############################################################################
# Compute one VAR model using all epochs
# --------------------------------------
# By setting ``model='dynamic'``, we instead treat each
# Epoch as a sample of the same VAR model

conn = var(data=epochs.get_data(), times=times, names=ch_names,
           model='avg-epochs')

# this returns a connectivity structure over time
print(conn)

##############################################################################
# Evaluate model fit again
# ------------------------
# We can now evaluate the model fit again as done
# earlier. This model fit will of course have
# higher residuals then before as we are only
# fitting 1 VAR model to all the epochs.

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

# visualize the covariance of residuals
fig, ax = plt.subplots()
cax = fig.add_axes([0.27, 0.8, 0.5, 0.05])
im = ax.imshow(rescov, cmap='viridis', aspect='equal', interpolation='none')
fig.colorbar(im, cax=cax, orientation='horizontal')

##############################################################################
# Other properties of VAR models
# ------------------------------
# We can now evaluate the model fit again as done
# earlier. This model fit will of course have
# higher residuals then before as we are only
# fitting 1 VAR model to all the epochs.
