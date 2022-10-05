'''
=============================
Comparing PLI, wPLI, and dPLI
=============================

This example demonstrates the different connectivity information captured by
the phase lag index (PLI) :footcite:`StamEtAl2007`, weighted phase lag index
(wPLI) :footcite:`VinckEtAl2011`, and directed phase lag index (dPLI)
:footcite:`StamEtAl2012` on simulated data.
'''

# Authors: Kenji Marshall <kenji.marshall99@gmail.com>
#          Charlotte Maschke <charlotte.maschke@mail.mcgill.ca>
#          Stefanie Blain-Moraes <stefanie.blain-moraes@mcgill.ca>
#
# License: BSD (3-clause)


import mne
import numpy as np
import matplotlib.pyplot as plt

from mne_connectivity import spectral_connectivity_epochs
from mne.datasets import sample

###############################################################################
# Background
# ----------
#
# The formulae for PLI, wPLI, and dPLI are given below. In these equations,
# :math:`X_{ij}` is the cross-spectral density (CSD) between two signals
# :math:`i, j`. Importantly, the imaginary component of the CSD is maximal
# when the signals have a phase difference given by :math:`k\pi+\frac{\pi}{2}`,
# and is :math:`0` when the phase difference is given by :math:`k\pi` (where
# :math:`k \in \mathbb{Z}`). This property provides protection against
# recognizing volume conduction effects as connectivity, and is the backbone
# for these methods :footcite:`VinckEtAl2011`. In the equations below,
# :math:`\mathcal{I}` refers to the imaginary component,
# :math:`\mathcal{H}` refers to the Heaviside step function, and
# :math:`sgn` refers to the sign function.
#
# :math:`PLI = |E[sgn(\mathcal{I}(X_{ij}))]|` :footcite:`StamEtAl2007`
#
# :math:`wPLI = \frac{|E[\mathcal{I}(X_{ij})]|}{E[|\mathcal{I}(X_{ij})|]}`
# :footcite:`VinckEtAl2011`
#
# :math:`dPLI = E[\mathcal{H}(\mathcal{I}(X_{ij}))]` :footcite:`StamEtAl2012`
#
# All three of these metrics are bounded in the range :math:`[0, 1]`.
#
# * For PLI, :math:`0` means that signal :math:`i` leads and lags signal
#   :math:`j` equally often, while a value greater than :math:`0` means that
#   there is an imbalance in the likelihood for signal :math:`i` to be leading
#   or lagging. A value of :math:`1` means that signal :math:`i` only leads or
#   only lags signal :math:`j`.
# * For wPLI, :math:`0` means that the total weight (not the quantity) of all
#   leading relationships equals the total weight of lagging relationships,
#   while a value greater than :math:`0` means that there is an imbalance
#   between these weights. A value of :math:`1`, just as in PLI, means that
#   signal :math:`i` only leads or only lags signal :math:`j`.
# * With dPLI, we gain the ability to distinguish whether signal :math:`i` is
#   leading or lagging signal :math:`j`, complementing the information provided
#   by PLI or wPLI. A value of :math:`0.5` means that signal :math:`i` leads
#   and lags signal :math:`j` equally often. A value in the range
#   :math:`(0.5, 1.0]` means that signal :math:`i` leads signal :math:`j` more
#   often than it lags, with a value of :math:`1` meaning that signal :math:`i`
#   always leads signal :math:`j`. A value in the range :math:`[0.0, 0.5)`
#   means that signal :math:`i` lags signal :math:`j` more often than it leads,
#   with a value of :math:`0` meaning that signal :math:`i` always lags signal
#   :math:`j`. The PLI can actually be extracted from the dPLI by the
#   relationship :math:`PLI = 2|dPLI - 0.5|`, but this relationship is not
#   invertible (dPLI can not be estimated from the PLI).
#
#
# Overall, these different approaches are closely related but have subtle
# differences, as will be demonstrated throughout the rest of this example.

###############################################################################
# Capturing Leading/Lagging Phase Relationships with dPLI
# -------------------------------------------------------
#
# The main advantage of dPLI is that it's *directed*, meaning it can
# differentiate between phase relationships which are leading or lagging.
# To illustrate this, we generate sinusoids with Gaussian noise. In particular,
# we generate signals with phase differences of
# :math:`[-\pi, -\frac{\pi}{2}, 0, \frac{\pi}{2}, \pi]` relative to a reference
# signal. A negative difference means that the reference signal is lagging the
# other signal.


fs = 250  # sampling rate (Hz)
n_e = 300  # number of epochs
T = 10  # length of epochs (s)
f = 10  # frequency of sinusoids (Hz)
t = np.arange(0, T, 1 / fs)
A = 1  # noise amplitude
sigma = 0.5  # Gaussian noise variance

data = []

phase_differences = [0, -np.pi, -np.pi / 2, 0, np.pi / 2, np.pi]
for ps in zip(phase_differences):
    sig = []
    for _ in range(n_e):
        sig.append(np.sin(2 * np.pi * f * t - ps) +
                   A * np.random.normal(0, sigma, size=t.shape))
    data.append(sig)

data = np.swapaxes(np.array(data), 0, 1)  # make epochs the first dimension

###############################################################################
# A snippet of this simulated data is shown below. The blue signal is the
# reference signal.

# %%
fig, ax = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
ax[0].plot(t[:fs], data[0, 0, :fs], label="Reference")
ax[0].plot(t[:fs], data[0, 2, :fs])

ax[0].set_title(r"Phase Lagging ($-\pi/2$ Phase Difference)")
ax[0].set_xlabel("Time (s)")
ax[0].set_ylabel("Signal")
ax[0].legend()

ax[1].plot(t[:fs], data[0, 0, :fs], label="Reference")
ax[1].plot(t[:fs], data[0, 4, :fs])
ax[1].set_title(r"Phase Leading ($\pi/2$ Phase Difference)")
ax[1].set_xlabel("Time (s)")

plt.show()

###############################################################################
# We will now compute PLI, wPLI, and dPLI for each phase relationship.

# %%
conn = []
indices = ([0, 0, 0, 0, 0], [1, 2, 3, 4, 5])
for method in ['pli', 'wpli', 'dpli']:
    conn.append(
        spectral_connectivity_epochs(
            data, method=method, sfreq=fs, indices=indices,
            fmin=9, fmax=11, faverage=True).get_data()[:, 0])
conn = np.array(conn)

###############################################################################
# The estimated connectivites are shown in the figure below, which provides
# insight into the differences between PLI/wPLI, and dPLI.
#
#
# **Similarities Of All Measures**
#
# * Capture presence of connectivity in same situations (phase difference of
#   :math:`\pm\frac{\pi}{2}`)
# * Do not predict connectivity when phase difference is a multiple of
#   :math:`\pi`
# * Bounded between :math:`0` and :math:`1`
#
# **How dPLI is Different Than PLI/wPLI**
#
# * Null connectivity is :math:`0` for PLI and wPLI, but :math:`0.5` for dPLI
# * dPLI differentiates whether the reference signal is leading or lagging the
#   other signal (lagging if :math:`0 <= dPlI < 0.5`, leading if
#   :math:`0.5 < dPLI <= 1.0`)


x = np.arange(5)

plt.figure()
plt.bar(x - 0.2, conn[0], 0.2, align='center', label="PLI")
plt.bar(x, conn[1], 0.2, align='center', label="wPLI")
plt.bar(x + 0.2, conn[2], 0.2, align='center', label="dPLI")

plt.title("Connectivity Estimation Comparison")
plt.xticks(x, (r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"))
plt.legend()
plt.xlabel("Phase Difference")
plt.ylabel("Estimated Connectivity")

plt.show()

###############################################################################
# Robustness to Outliers and Noise with wPLI
# ------------------------------------------
#
# The previous experiment illustrated the advantages conferred by dPLI when
# differentiating leading and lagging phase relationships. This experiment
# will now focus on understanding the advantages of wPLI, and explore how it
# extends upon PLI.
#
# The main difference between PLI and wPLI is in how different phase
# relationships are *weighted*. In PLI, phase differences are weighted as
# :math:`-1` or :math:`1` according to their sign. In wPLI, phase differences
# are weighted based on their value, meaning that phase differences closer to
# :math:`\pm\frac{\pi}{2}` are weighted more heavily than those close to
# :math:`0` or any other multiple of :math:`\pi`.
#
# This avoids a discontinuity at the transition between positive and negative
# phase, treating all phase differences near this transition in a similar way.
# This provides some robustness against outliers and noise when estimating
# connectivity. For instance, volume conduction can distort EEG/MEG recordings,
# wherein signals emanating from the same neural source will be picked up by
# multiple sensors on the scalp. This can effect connectivity estimations,
# bringing the relative phase differences between two signals close to
# :math:`0`. wPLI minimizes the contribution of phase relationships that are
# small but non-zero (and may thus be attributed to volume conduciton), while
# PLI weighs these in the same way as phase relationships of
# :math:`\pm\frac{\pi}{2}`.
#
# To demonstrate this, we recreate a result from (Vinck et al, 2011)
# :footcite:`VinckEtAl2011`. Two sinusoids are simulated, where the phase
# difference for half of the epochs is :math:`\frac{\pi}{2}`, and is
# :math:`-\frac{\pi}{100}` for the others. We also explore the effect of
# applying uniform noise to this phase difference.

# %%
n_noise = 41  # amount of noise amplitude samples in [0, 4]
data = [[]]

# Generate reference
for _ in range(n_e):
    data[0].append(np.sin(2 * np.pi * f * t))

A_list = np.linspace(0, 4, n_noise)

for A in A_list:
    sig = []
    # Generate other signal
    for _ in range(int(n_e / 2)):  # phase difference -pi/100
        sig.append(np.sin(2 * np.pi * f * t + np.pi /
                   100 + A * np.random.uniform(-1, 1)))
    for _ in range(int(n_e / 2), n_e):  # phase difference pi/2
        sig.append(np.sin(2 * np.pi * f * t - np.pi /
                   2 + A * np.random.uniform(-1, 1)))
    data.append(sig)

data = np.swapaxes(np.array(data), 0, 1)

# Visualize the data
fig, ax = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
ax[0].plot(t[:10], data[0, 0, :10], label="Reference")
ax[0].plot(t[:10], data[1, 1, :10])

ax[0].set_title(r"Phase Lagging ($-\pi/100$ Phase Difference)")
ax[0].set_xlabel("Time (s)")
ax[0].set_ylabel("Signal")
ax[0].legend()

ax[1].plot(t[:fs], data[0, 0, :fs], label="Reference")
ax[1].plot(t[:fs], data[-1, 1, :fs])
ax[1].set_title(r"Phase Leading ($\pi/2$ Phase Difference)")
ax[1].set_xlabel("Time (s)")

plt.show()

###############################################################################
# We can now compute PLI and wPLI

conn = []
indices = ([0] * n_noise, np.arange(1, n_noise + 1))
for method in ['pli', 'wpli']:
    conn.append(
        spectral_connectivity_epochs(
            data, method=method, sfreq=fs, indices=indices,
            fmin=9, fmax=11, faverage=True).get_data()[:, 0])
conn = np.array(conn)

###############################################################################
# The results from the simulation are shown in the figure below. In the case
# without noise, the difference between wPLI and PLI is made obvious. In PLI,
# no connectivity is detected, as the :math:`-\frac{\pi}{100}` phase
# differences are weighted in the exact same way as the :math:`\frac{\pi}{2}`
# relationships. wPLI is able to avoid the cancellation of the
# :math:`\frac{\pi}{2}` relationships.
#
# As noise gets added, PLI increases since the :math:`-\frac{\pi}{100}`
# relationships are made positive more often than the :math:`\frac{\pi}{2}`
# relationships are made negative. However, wPLI maintains an advantage in
# its ability to distinguish the underlying structure. Beyond a certain point,
# the noise dominates any pre-defined structure, and both methods behave
# similarly, tending toward :math:`0`. For a more detailed analysis of this
# result and the properties of wPLI, please refer to (Vinck et al, 2011)
# :footcite:`VinckEtAl2011`.

plt.figure()
plt.plot(A_list, conn[0], "o-", label="PLI")
plt.plot(A_list, conn[1], "o-", label="wPLI")
plt.legend()
plt.xlabel("Noise Amplitude")
plt.ylabel("Connectivity Measure")
plt.title("wPLI and PLI Under Increasing Noise")
plt.show()

###############################################################################
# Demo On MEG Data
# ----------------
#
# To finish this example, we also quickly demonstrate these methods on some
# sample MEG data recorded during visual stimulation.

data_path = sample.data_path()
raw_fname = data_path / 'MEG/sample/sample_audvis_filt-0-40_raw.fif'
event_fname = data_path / 'MEG/sample/sample_audvis_filt-0-40_raw-eve.fif'
raw = mne.io.read_raw_fif(raw_fname)
events = mne.read_events(event_fname)


# Select gradiometers
picks = mne.pick_types(raw.info, meg='grad', eeg=False, stim=False, eog=True,
                       exclude='bads')

# Create epochs
event_id, tmin, tmax = 3, -0.2, 1.5  # need a long enough epoch for 5 cycles
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                    baseline=(None, 0), reject=dict(grad=4000e-13, eog=150e-6))
epochs.load_data().pick_types(meg='grad')  # just keep MEG and no EOG now

fmin, fmax = 4., 9.  # compute connectivity within 4-9 Hz
sfreq = raw.info['sfreq']  # the sampling frequency
tmin = 0.0  # exclude the baseline period

# Compute PLI, wPLI, and dPLI
con_pli = spectral_connectivity_epochs(
    epochs, method='pli', mode='multitaper', sfreq=sfreq, fmin=fmin,
    fmax=fmax, faverage=True, tmin=tmin, mt_adaptive=False, n_jobs=1)

con_wpli = spectral_connectivity_epochs(
    epochs, method='wpli', mode='multitaper', sfreq=sfreq, fmin=fmin,
    fmax=fmax, faverage=True, tmin=tmin, mt_adaptive=False, n_jobs=1)

con_dpli = spectral_connectivity_epochs(
    epochs, method='dpli', mode='multitaper', sfreq=sfreq, fmin=fmin,
    fmax=fmax, faverage=True, tmin=tmin, mt_adaptive=False, n_jobs=1)

###############################################################################
# In this example, there is strong connectivity between sensors 190-200 and
# sensors 110-160.
#
# Moreover, after observing the presence of connectivity, dPLI can be used to
# ascertain the direction of the phase relationship. Here, it appears that the
# dPLI connectivity in this area is less than :math:`0.5`, and thus sensors
# 190-200 are lagging sensors 110-160.
#
# In keeping with the previous simulation, we can see that wPLI identifies
# stronger connectivity relationships than PLI. This is due to its robustness
# against volume conduction effects decreasing the detected connectivity
# strength, as was mentioned earlier.

fig, axs = plt.subplots(1, 3, figsize=(14, 5), sharey=True)
axs[0].imshow(con_pli.get_data('dense'), vmin=0, vmax=1)
axs[0].set_title("PLI")
axs[0].set_ylabel("Sensor 1")
axs[0].set_xlabel("Sensor 2")

axs[1].imshow(con_wpli.get_data('dense'), vmin=0, vmax=1)
axs[1].set_title("wPLI")
axs[1].set_xlabel("Sensor 2")

im = axs[2].imshow(con_dpli.get_data('dense'), vmin=0, vmax=1)
axs[2].set_title("dPLI")
axs[2].set_xlabel("Sensor 2")

fig.colorbar(im, ax=axs.ravel())
plt.show()

###############################################################################
# Conclusions
# -----------
#
# Both wPLI and dPLI are extensions upon the original PLI method, and provide
# complementary information about underlying connectivity.
#
# * To identify the presence of an underlying phase relationship, wPLI is the
#   method of choice for most researchers as it provides an improvement in
#   robustness over the original PLI method
# * To know the directionality of the connectivity identified by wPLI, dPLI
#   should be used
#
# Ultimately, these methods work great together, providing a comprehensive
# estimate of phase-based connectivity.

###############################################################################
# References
# ----------
# .. footbibliography::
