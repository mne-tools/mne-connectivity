"""
.. _ex-connectivity-measures:

==============================
Using the connectivity classes
==============================

Compute different connectivity measures and then demonstrate
the utility of the class.

Here we compute the Phase Lag Index (PLI) between all gradiometers and showcase
how we can interact with the connectivity class.
"""

# Authors: Adam Li <adam2392@gmail.com>
#
# License: BSD (3-clause)

import os.path as op
import numpy as np

import mne
from mne_connectivity import spectral_connectivity_epochs
from mne.datasets import sample

# %%
# Set parameters
data_path = sample.data_path()
raw_fname = op.join(data_path, 'MEG', 'sample',
                    'sample_audvis_filt-0-40_raw.fif')
event_fname = op.join(data_path, 'MEG', 'sample',
                      'sample_audvis_filt-0-40_raw-eve.fif')

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

# Compute connectivity for the alpha band that contains the evoked response
# (4-9 Hz). We exclude the baseline period:
fmin, fmax = 4., 9.
cwt_freqs = np.linspace(fmin, fmax, 5)
sfreq = raw.info['sfreq']  # the sampling frequency
tmin = 0.0  # exclude the baseline period
epochs.load_data().pick_types(meg='grad')  # just keep MEG and no EOG now
con = spectral_connectivity_epochs(
    epochs, method='pli', mode='cwt_morlet', sfreq=sfreq, fmin=fmin, fmax=fmax,
    faverage=False, tmin=tmin, cwt_freqs=cwt_freqs, mt_adaptive=False,
    n_jobs=1)

# %%
# Now, we can look at different functionalities of the connectivity
# class returned by :func:`mne_connectivity.spectral_connectivity_epochs`. The
# following are some basic attributes of connectivity classes.

# the dimensions of the data corresponding to each axis
print(con.dims)

# the coordinates for each axis of the data
print(con.coords)

# the number of nodes matches the number of electrodes used to compute the
# spectral measure
print(con.n_nodes)

# the names of each node correspond to the electrode names
print(con.names)

# %% Connectivity Measure Data Shapes
# The underlying connectivity measure can be stored in two ways: i) raveled
# and ii) dense. Raveled storage will be a 1D column flattened array, similar
# to what one might expect when using `numpy.ravel`. However, if you ask for
# the dense data, then the shape will show the N by N connectivity.
# In general, you might prefer the raveled version if you specify a subset of
# indices (e.g. some subset of sources) for the computation
# of a bivariate connectivity measure or if you have a symmetric measure
# (e.g. coherence). The 'dense' output on the other hand provides an actual
# square matrix, which can be used for post-hoc analysis that expects a matrix
# shape.

# the underlying data is stored "raveled", and the connectivity measure is
# flattened into one dimension
print(con.shape)

# the 'dense' output will show the connectivity measure's N x N axis
print(con.get_data(output='dense').shape)

# %% Connectivity Measure XArray Attributes
# The underlying data is stored as an xarray, so we have access
# to DataArray attributes. Each connectivity measure function automatically
# stores relevant metadata. For example, the method used in this example
# is the phase-lag index ('pli').
print(con.attrs.keys())
print(con.attrs.get('method'))

# You can also store additional metadata relevant to your experiment, which can
# easily be done, because ``attrs`` is just a dictionary.
con.attrs['experimenter'] = 'mne'
print(con.attrs.keys())

# %%
# Other properties of the connectivity class, special to
# the spectro-temporal connectivity class.
#
# .. note:: Not all connectivity classes will have these properties.

# a frequency axis shows the different frequencies used in estimating
# the spectral measure
print(con.freqs)

# a time axis shows the different time points because the spectral
# measure is time-resolved
print(con.times)
