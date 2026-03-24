#!/usr/bin/env python

#!/usr/bin/env python
# coding: utf-8

"""
=====================================================================
Time-Frequency Seed Connectivity Using Debiased wPLI
=====================================================================

This example demonstrates how to compute time–frequency connectivity between
two seed regions of interest (ROIs) using the debiased weighted phase lag
index (wPLI) :footcite:`VinckEtAl2011` on MEG data.

Connectivity is estimated in the time–frequency domain using Morlet wavelets,
allowing characterization of dynamic interactions across both spectral and
temporal dimensions. This enables observation of event-related
synchronization and desynchronization, providing a 2D representation of
connectivity strength as a function of time and frequency.

This workflow can be applied to other connectivity metrics supported by
:meth:`~mne_connectivity.spectral_connectivity_epochs`.
"""
# Demo Author: Kinkini Monaragala <kinkini.monaragala@gmail.com>
#
# License: BSD (3-clause)

import matplotlib.pyplot as plt
import mne
import numpy as np
from mne.datasets import sample

from mne_connectivity import seed_target_indices, spectral_connectivity_epochs

print(__doc__)

# Demo will be conducted on the sample MEG data recorded during visual stimulation.

data_path = sample.data_path()
raw_fname = data_path / "MEG/sample/sample_audvis_filt-0-40_raw.fif"
event_fname = data_path / "MEG/sample/sample_audvis_filt-0-40_raw-eve.fif"
raw = mne.io.read_raw_fif(raw_fname)
events = mne.read_events(event_fname)

# gradiometers selection
picks = mne.pick_types(raw.info, meg='grad', ref_meg=False)

# epochs creation
event_id, tmin, tmax = 3, -0.2, 1.5  # need a long enough epoch for 5 cycles
epochs = mne.Epochs(
    raw,
    events,
    event_id,
    tmin,
    tmax,
    picks=picks,
    baseline=(None, 0),  # normalize each epoch to pre-stimulus interval -0.2 to 0
    reject=dict(grad=4000e-13),
)

epochs.load_data()

info = raw.info
print(info)


# Channels were chosen to target the visual cortex (V1: MEG 2141, 2131) and anterior frontal regions (MEG 0123) for comparing connectivity and assessing visual attentional modulation.

info_picked = mne.pick_info(info, picks)

mne.viz.plot_sensors(
    info_picked,
    kind='topomap',
    show_names=True
)


# Set seeds
seed_ch = "MEG 2343"
seed2_ch = "MEG 0123"

picks_ch_names = [raw.ch_names[i] for i in picks]

seed_idx = picks_ch_names.index(seed_ch)
seed2_ch = picks_ch_names.index(seed2_ch)

indices = seed_target_indices(seed_idx, seed2_ch)


# Wpli2 debiased Connectvity
# Define wavelet frequencies and number of cycles
cwt_freqs = np.arange(5, 40, 2)
cwt_n_cycles = cwt_freqs / 5.0

# Run the connectivity analysis using 2 parallel jobs
sfreq = raw.info["sfreq"]  # the sampling frequency
con = spectral_connectivity_epochs(
    epochs,
    indices=indices,
    method="wpli2_debiased",
    mode="cwt_morlet",
    sfreq=sfreq,
    cwt_freqs=cwt_freqs,
    cwt_n_cycles=cwt_n_cycles,
    n_jobs=1,
)
times = con.times
freqs = con.freqs


### Plotting
con_item = np.array(con).item()
data = np.squeeze(con_item.get_data())

tmin, tmax = times[0], times[-1]
fmin, fmax = freqs[0], freqs[-1]

n_rows, n_cols = data.T.shape

plt.imshow(
    data.T,
    cmap='RdBu_r',
    vmin = 0,
    vmax = np.max(data.T),
    aspect='auto',
    extent=[tmin, tmax, fmin, fmax],
    origin='lower'
)

plt.axvline(0, color='black', linestyle='--', linewidth=1)
plt.colorbar(label='Connectivity')
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")

plt.show()


### References
# .. footbibliography::
#
