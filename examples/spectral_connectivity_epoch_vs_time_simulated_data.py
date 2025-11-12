"""
======================================================================
Contrasting the methods to calculate connectivity over epochs and time using simulated data
======================================================================

This example shows how to use the spectral connectivity measures using simulated data.

Spectral connectivity is generally used to caculate connectivity between electrodes or regions in
different frequency bands

When there are multiple epochs for a session, like ERP data, the spectral_connectivity_epochs
method can be used to infer the connectivity structure between channels across the epochs. It
will return the connectivity over time estimated from all the epochs.

When the connectivity is to be calculated on a single trial basis across the channels,
the spectral_connectivity_time can be used.
"""
# Author: Divyesh Narayanan <divyesh.narayanan@gmail.com>
#
# License: BSD (3-clause)

import numpy as np
import mne
from mne_connectivity import spectral_connectivity_epochs, spectral_connectivity_time
from matplotlib import pyplot as plt
print(__doc__)

###############################################################################

# Generate some data
n_epochs = 2
n_channels = 3
n_samples = 1000
sfreq = 250
time = np.arange(n_samples)/sfreq
# 4 secs

# Things I tried
# 1 epoch - all connectivity values give 1
# Should serve as useful warning. All give 1 when the channels values are different but there is
# only one epoch

# multiple epochs - over epochs returns 1 when the connectivity between 2 channels are 1 in all
# the epochs. Otherwise it is some form of estimate across the epochs. Not exactly the average of
# the single epoch connectivity values.

# Simulating data
rng = np.random.RandomState(0)
x1 = rng.rand(1, 1, n_samples)
x2 = np.sin(2*np.pi*10*time)
x3 = np.sin(2*np.pi*15*time)

data = np.zeros((n_epochs,n_channels,n_samples))
data[0,0,:] = x1
data[0,1,:] = x2
data[0,2,:] = x2 # x3

# Same value for each channel in all the epochs
# for i in range(n_epochs):
#     data[i] = data[0]

# Different values in different epochs
data[1,0,:] = x1
data[1,1,:] = x3
data[1,2,:] = x3

# Create epochs object for mne compatibility
ch_names = ["T1","T2","T3"] # random names
info = mne.create_info(ch_names, sfreq, ch_types="eeg")
data_epoch = mne.EpochsArray(data,info)

# Visualize the data
data_epoch.plot(scalings=2)

# Calculate coh over epochs/trials
con_Epoch = spectral_connectivity_epochs(data_epoch, method="coh",
    mode="cwt_morlet", sfreq=sfreq, cwt_freqs=np.array([10]))

c_ep = con_Epoch.get_data(output='dense').squeeze(2) # squeezing freq ind since only one freq
# average over time
print(c_ep.mean(2))

con_epoch = con_Epoch.get_data(output="raveled")
plt.plot(con_epoch.squeeze(1).T)
plt.show()

# Calculating time-resolved spectral connectivity for each epoch
con_Time = spectral_connectivity_time(data_epoch, method="coh",
    mode="cwt_morlet", sfreq=sfreq, freqs=10)

con_time = con_Time.get_data("raveled")
con_time = con_time.squeeze(2) # removing the freq axis
c_t1 = con_time[0]
plt.plot(c_t1.T)
plt.show()

c_t2 = con_time[1]
plt.plot(c_t2.T)
plt.show()

c_ti = con_Time.get_data('dense')
c_ti = c_ti.squeeze(3)
a = c_ti[0]
b = c_ti[1]
print(a.mean(2))
print(b.mean(2))

# other testing to compare with epochs method
print((a+b).mean(2)/2)
print((a.mean(2)+b.mean(2))/2)

print()
