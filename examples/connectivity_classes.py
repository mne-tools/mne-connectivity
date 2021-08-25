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

import numpy as np

import mne
from mne_connectivity import spectral_connectivity
from mne.datasets import sample

# %%
# Set parameters
data_path = sample.data_path()
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
event_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw-eve.fif'

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
cwt_freqs = np.linspace(fmin, fmax, 20)
sfreq = raw.info['sfreq']  # the sampling frequency
tmin = 0.0  # exclude the baseline period
epochs.load_data().pick_types(meg='grad')  # just keep MEG and no EOG now
con = spectral_connectivity(
    epochs, method='pli', mode='cwt_morlet', sfreq=sfreq, fmin=fmin, fmax=fmax,
    faverage=False, tmin=tmin, cwt_freqs=cwt_freqs, mt_adaptive=False,
    n_jobs=1)

# %%
# Now, we can look at different functionalities of the connectivity
# class returned by :func:`mne_connectivity.spectral_connectivity`.

# the dimensions of the data corresponding to each axis
print(con.dims)

# the coordinates for each axis of the data
print(con.coords)

# the underlying data is stored "raveled"
print(con.shape)

# However, if one asks for the output dense data, then the data will represent the full
# N by N connectivity. In general, you might prefer the raveled version if you
# specify a subset of indices (e.g. some subset of sources) for the computation
# of a bivariate connectivity measure or if you have a symmetric measure
# (e.g. coherence). The 'dense' output on the other hand provides an actual square
# matrix, which can be used for post-hoc analysis that expects a matrix shape.
print(con.get_data(output='dense').shape)

# the number of nodes matches the number of electrodes used to compute the
# spectral measure
print(con.n_nodes)

# the names of each node correspond to the electrode names
print(con.names)

# The underlying data is stored as an xarray, so we have access
# to DataArray attributes. You can store metadata relevant to the
# estimated measure, or other relevant metadata. For example,
# the method name used to compute this connectivity is the 'pli' measure.
print(con.attrs)
print(con.attrs.get('method'))

# %%
# Other properties of the connectivity class, special for
# the spectro temporal connectivity.
#
# .. note:: Not all connectivity classes will have these properties.

# a frequency axis shows the different frequencies used in estimating
# the spectral measure
print(con.freqs)

# a time axis shows the different time points because the spectral
# measure is time-resolved
print(con.times)
