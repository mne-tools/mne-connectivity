"""
=========================================================
Weighted Symbolic Mutual Information (wSMI) connectivity.
=========================================================

This example demonstrates the weighted Symbolic Mutual Information (wSMI)
connectivity measure :footcite:`KingEtAl2013` for detecting nonlinear
dependencies between brain signals. wSMI is particularly useful for studying
consciousness and information integration in neural networks as it can
capture complex, nonlinear relationships that linear methods might miss.

The method works by:

1. Converting time series to symbolic patterns (ordinal patterns)
2. Computing mutual information between symbolic sequences
3. Weighting patterns based on their temporal structure

This makes wSMI sensitive to information flow and temporal dependencies
that are characteristic of conscious brain states.
"""

# Authors: Giovanni Marraffini <giovanni.marraffini@gmail.com>
#          Laouen Belloli <laouen.belloli@gmail.com>
#
# License: BSD (3-clause)

import warnings

import matplotlib.pyplot as plt
import mne
import numpy as np
from mne.datasets import sample

from mne_connectivity import wsmi

# %%
# Simulating Data with Different Connectivity Patterns
# =====================================================
#
# To demonstrate wSMI's capabilities, we'll create synthetic EEG data with
# different types of connectivity: linear coupling, nonlinear coupling, and
# independent signals.


def create_synthetic_eeg_data(sfreq=500, n_epochs=50, epoch_length=2.0):
    """Create synthetic EEG data with different connectivity patterns."""
    n_times = int(sfreq * epoch_length)

    # Create time vector
    times = np.arange(n_times) / sfreq

    # Initialize data array (n_epochs, n_channels, n_times)
    data = np.zeros((n_epochs, 4, n_times))

    for epoch in range(n_epochs):
        # Set random seed for reproducibility within epoch variation
        rng = np.random.RandomState(42 + epoch)

        # Channel 1: Source signal (alpha rhythm + noise)
        alpha_freq = 10 + rng.normal(0, 0.5)  # Variable alpha frequency
        ch1 = np.sin(2 * np.pi * alpha_freq * times) + 0.3 * rng.randn(n_times)

        # Channel 2: Linear coupling to ch1 (strong connectivity expected)
        coupling_strength = 0.7 + rng.normal(0, 0.1)
        delay_samples = int(0.05 * sfreq)  # 50ms delay
        ch2_base = coupling_strength * np.roll(ch1, delay_samples)
        ch2 = ch2_base + 0.2 * rng.randn(n_times)

        # Channel 3: Nonlinear coupling to ch1 (wSMI should detect this)
        # Phase-amplitude coupling + quadratic transformation
        ch3_phase = np.angle(np.sin(2 * np.pi * alpha_freq * times))
        ch3_amp = 0.5 * (1 + np.tanh(2 * ch1))  # Nonlinear transformation
        ch3 = ch3_amp * np.sin(ch3_phase + np.pi / 4) + 0.3 * rng.randn(n_times)

        # Channel 4: Independent signal (low connectivity expected)
        beta_freq = 20 + rng.normal(0, 1)
        ch4 = np.sin(2 * np.pi * beta_freq * times) + 0.4 * rng.randn(n_times)

        data[epoch, 0, :] = ch1
        data[epoch, 1, :] = ch2
        data[epoch, 2, :] = ch3
        data[epoch, 3, :] = ch4

    return data


# Create synthetic data
sfreq = 500  # Hz
data = create_synthetic_eeg_data(sfreq=sfreq, n_epochs=50, epoch_length=2.0)

# Create MNE Epochs object
ch_names = ["Source", "Linear_Coupled", "Nonlinear_Coupled", "Independent"]
ch_types = ["eeg"] * 4
info = mne.create_info(ch_names, sfreq=sfreq, ch_types=ch_types)
epochs = mne.EpochsArray(data, info, tmin=0.0, verbose=False)

print(f"Created synthetic EEG data: {epochs}")

# %%
# Visualizing the Synthetic Data
# ===============================
#
# Let's examine our synthetic data to understand the different connectivity
# patterns we've created.

# Plot a sample of the data
fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
epoch_data = epochs.get_data()[0]  # First epoch
times = epochs.times[:500]  # First 1 second

for i, ch_name in enumerate(ch_names):
    axes[i].plot(times, epoch_data[i, :500])
    axes[i].set_ylabel(ch_name)
    axes[i].grid(True, alpha=0.3)

axes[-1].set_xlabel("Time (s)")
plt.suptitle("Synthetic EEG Data with Different Connectivity Patterns")
plt.tight_layout()
plt.show()

# %%
# Computing wSMI with Default Parameters
# ======================================
#
# First, let's compute wSMI with default parameters. By default, wSMI uses:
#
# - Anti-aliasing filtering enabled (``anti_aliasing=True``)
# - Weighted symbolic mutual information (``weighted=True``)
# - All channel pairs (``indices=None``)
# - Individual epoch connectivity (``average=False``)

# Compute wSMI with default parameters
conn_default = wsmi(epochs, kernel=3, tau=1)

print(f"wSMI connectivity shape: {conn_default.get_data().shape}")
print(f"Method: {conn_default.method}")
print(f"Number of connections: {len(conn_default.indices)}")

# Extract connectivity matrix for visualization
conn_matrix = conn_default.get_data().mean(axis=0)  # Average over epochs
n_channels = len(ch_names)

# Create full connectivity matrix from lower triangular
full_matrix = np.zeros((n_channels, n_channels))
for idx, (i, j) in enumerate(conn_default.indices):
    full_matrix[i, j] = conn_matrix[idx]
    full_matrix[j, i] = conn_matrix[idx]  # Make symmetric

# %%
# Visualizing wSMI Connectivity Results
# =====================================

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
im = ax.imshow(full_matrix, cmap="viridis", vmin=0, vmax=full_matrix.max())
ax.set_xticks(range(n_channels))
ax.set_yticks(range(n_channels))
ax.set_xticklabels(ch_names, rotation=45)
ax.set_yticklabels(ch_names)

# Add connectivity values as text
for i in range(n_channels):
    for j in range(n_channels):
        text = ax.text(
            j,
            i,
            f"{full_matrix[i, j]:.3f}",
            ha="center",
            va="center",
            color="black" if full_matrix[i, j] > full_matrix.max() / 2 else "white",
        )

plt.colorbar(im, ax=ax, label="wSMI")
plt.title("wSMI Connectivity Matrix\n(Higher values = stronger connectivity)")
plt.tight_layout()
plt.show()

# Print connectivity summary
print("\nConnectivity Summary:")
print(f"Source ↔ Linear Coupled: {full_matrix[0, 1]:.3f}")
print(f"Source ↔ Nonlinear Coupled: {full_matrix[0, 2]:.3f}")
print(f"Source ↔ Independent: {full_matrix[0, 3]:.3f}")

# %%
# Exploring Parameter Effects: kernel and tau
# ============================================
#
# The kernel and tau parameters control the temporal scale of the analysis:
#
# - ``kernel``: Number of time points in each symbolic pattern (2-7 typical)
# - ``tau``: Time delay between pattern elements (1 = consecutive samples)
#
# Let's explore how these parameters affect connectivity detection.

# Test different parameter combinations
param_combinations = [
    (3, 1),  # Fine temporal resolution
    (3, 2),  # Coarser temporal resolution
    (4, 1),  # More complex patterns
    (4, 2),  # Complex patterns + coarser resolution
]

connectivity_results = {}

for kernel, tau in param_combinations:
    conn = wsmi(epochs, kernel=kernel, tau=tau, verbose=False)
    conn_avg = conn.get_data().mean(axis=0)

    # Extract Source ↔ Linear Coupled connectivity (first connection)
    source_linear = conn_avg[0]  # Connection between channels 0 and 1
    connectivity_results[f"k{kernel}_t{tau}"] = source_linear

# Plot parameter effects
fig, ax = plt.subplots(figsize=(10, 6))
params = list(connectivity_results.keys())
values = list(connectivity_results.values())

bars = ax.bar(params, values, color=["skyblue", "lightcoral", "lightgreen", "gold"])
ax.set_ylabel("wSMI (Source ↔ Linear Coupled)")
ax.set_xlabel("Parameter Combination (kernel_tau)")
ax.set_title("Effect of kernel and tau Parameters on wSMI")
ax.grid(True, alpha=0.3)

# Add value labels on bars
for bar, value in zip(bars, values):
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2.0,
        height + 0.001,
        f"{value:.3f}",
        ha="center",
        va="bottom",
    )

plt.tight_layout()
plt.show()

# %%
# Comparing wSMI vs SMI (weighted vs unweighted)
# ===============================================
#
# wSMI uses distance-based weighting of symbolic patterns, while standard SMI
# treats all patterns equally. Let's compare their sensitivity.

# Compute both weighted and unweighted versions
conn_wsmi = wsmi(epochs, kernel=3, tau=1, weighted=True, verbose=False)
conn_smi = wsmi(epochs, kernel=3, tau=1, weighted=False, verbose=False)

# Extract connectivity values (Source ↔ Linear Coupled)
wsmi_values = conn_wsmi.get_data().mean(axis=0)[0]
smi_values = conn_smi.get_data().mean(axis=0)[0]

print("\nComparison of wSMI vs SMI:")
print(f"wSMI (weighted):     {wsmi_values:.3f}")
print(f"SMI (unweighted):    {smi_values:.3f}")
print(f"Improvement ratio:   {wsmi_values / smi_values:.2f}x")

# %%
# Anti-aliasing: Understanding the Preprocessing
# ==============================================
#
# wSMI includes automatic anti-aliasing filtering to prevent artifacts when
# ``tau > 1``. Let's demonstrate the difference this makes.


# Compute with and without anti-aliasing
conn_filtered = wsmi(epochs, kernel=3, tau=2, anti_aliasing=True, verbose=False)

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    conn_unfiltered = wsmi(epochs, kernel=3, tau=2, anti_aliasing=False, verbose=False)
    if w:
        print(f"Warning when anti-aliasing disabled: {w[0].message}")

# Compare results
filtered_connectivity = conn_filtered.get_data().mean(axis=0)[0]
unfiltered_connectivity = conn_unfiltered.get_data().mean(axis=0)[0]

print("\nAnti-aliasing Effect (tau=2):")
print(f"With anti-aliasing:    {filtered_connectivity:.3f}")
print(f"Without anti-aliasing: {unfiltered_connectivity:.3f}")
print(
    f"Difference:            {abs(filtered_connectivity - unfiltered_connectivity):.3f}"
)

# %%
# Using Real EEG Data: Sample Dataset
# ===================================
#
# Now let's apply wSMI to real EEG data from the MNE sample dataset.

# Load sample data
data_path = sample.data_path()
raw_fname = data_path / "MEG" / "sample" / "sample_audvis_filt-0-40_raw.fif"
event_fname = data_path / "MEG" / "sample" / "sample_audvis_filt-0-40_raw-eve.fif"

# Load and preprocess data
raw = mne.io.read_raw_fif(raw_fname, verbose=False)
events = mne.read_events(event_fname, verbose=False)

# Pick a few EEG channels for demonstration
eeg_picks = ["EEG 001", "EEG 002", "EEG 003", "EEG 004"]
raw.pick(eeg_picks)

# Create epochs around visual stimuli
epochs_real = mne.Epochs(
    raw,
    events,
    event_id=3,
    tmin=-0.2,
    tmax=0.8,
    baseline=(-0.2, 0),
    preload=True,
    verbose=False,
)

# Apply minimal preprocessing
epochs_real.filter(1, 30, verbose=False)  # Basic band-pass filter

print(f"Real EEG data: {epochs_real}")

# %%
# Computing wSMI on Real Data
# ===========================

# Compute wSMI on real EEG data
conn_real = wsmi(epochs_real, kernel=3, tau=1, verbose=False)

# Create connectivity matrix for visualization
real_conn_matrix = conn_real.get_data().mean(axis=0)
n_eeg = len(eeg_picks)
real_full_matrix = np.zeros((n_eeg, n_eeg))

for idx, (i, j) in enumerate(conn_real.indices):
    real_full_matrix[i, j] = real_conn_matrix[idx]
    real_full_matrix[j, i] = real_conn_matrix[idx]

# Plot connectivity matrix
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
im = ax.imshow(real_full_matrix, cmap="viridis", vmin=0)
ax.set_xticks(range(n_eeg))
ax.set_yticks(range(n_eeg))
ax.set_xticklabels(eeg_picks)
ax.set_yticklabels(eeg_picks)

# Add connectivity values
for i in range(n_eeg):
    for j in range(n_eeg):
        text = ax.text(
            j,
            i,
            f"{real_full_matrix[i, j]:.3f}",
            ha="center",
            va="center",
            color="black"
            if real_full_matrix[i, j] > real_full_matrix.max() / 2
            else "white",
        )

plt.colorbar(im, ax=ax, label="wSMI")
plt.title("wSMI Connectivity: Real EEG Data\n(Visual stimulus epochs)")
plt.tight_layout()
plt.show()

# Brief interpretation of results
print("\nReal EEG Results Interpretation:")
print("The connectivity matrix shows wSMI values between electrode pairs during")
print("visual stimulus processing. Higher values indicate stronger information")
print("sharing between brain regions, which may reflect:")
print("- Coordinated visual processing networks")
print("- Task-related functional connectivity")
print("- Individual differences in brain network organization")
print("Note: Values depend on electrode placement, stimulus type, and individual")
print("brain anatomy. Clinical interpretation requires comparison with controls.")

# %%
# Selective Connectivity Analysis
# ===============================
#
# For large datasets, you might want to compute connectivity only between
# specific channel pairs using the indices parameter.

# Define specific connections of interest
indices = ([0, 0, 1], [1, 2, 2])  # EEG001↔EEG002, EEG001↔EEG003, EEG002↔EEG003

conn_selective = wsmi(epochs_real, kernel=3, tau=1, indices=indices, verbose=False)

print("\nSelective connectivity analysis:")
print(f"Number of connections computed: {len(conn_selective.indices)}")
print(
    f"Connections: {[(eeg_picks[i], eeg_picks[j]) for i, j in conn_selective.indices]}"
)

# Show connectivity values
selective_values = conn_selective.get_data().mean(axis=0)
for idx, (i, j) in enumerate(conn_selective.indices):
    print(f"{eeg_picks[i]} ↔ {eeg_picks[j]}: {selective_values[idx]:.3f}")

# %%
# Averaging Across Epochs
# ========================
#
# For group-level analysis, you might want connectivity averaged across epochs.

conn_averaged = wsmi(epochs_real, kernel=3, tau=1, average=True, verbose=False)

print("\nEpoch-averaged connectivity:")
print(f"Data shape: {conn_averaged.get_data().shape}")
print(f"Type: {type(conn_averaged)}")

# %%
# Summary and Best Practices
# ===========================
#
# **When to use wSMI:**
#
# - Studying nonlinear brain connectivity
# - Investigating information integration and consciousness
# - Analyzing temporal dependencies in neural signals
# - When linear methods (coherence, PLV) may miss important relationships
#
# **Parameter selection guidelines:**
#
# - ``kernel``: Start with 3, increase to 4-5 for more complex patterns
#   (requires more data)
# - ``tau``: Use 1 for high-resolution analysis, 2-3 for slower dynamics
# - ``anti_aliasing``: Keep ``True`` unless you've already filtered appropriately
# - ``weighted``: Keep ``True`` for most applications (wSMI vs SMI)
#
# **Data requirements:**
#
# - Sufficient epoch length: At least ``tau*(kernel-1)+1`` samples per epoch
# - Adequate SNR: wSMI is robust but benefits from clean data
# - Stationary signals: Works best with preprocessed, artifact-free data
#
# **Interpretation:**
#
# - Values near 0: Minimal information sharing
# - Higher values: Stronger information integration
# - Compare relative values rather than absolute thresholds
# - Consider the temporal scale defined by your ``tau`` parameter

print("\nwSMI analysis complete!")
print("This example demonstrated:")
print("- Basic wSMI computation and visualization")
print("- Parameter exploration (kernel, tau)")
print("- Comparison of weighted vs unweighted SMI")
print("- Anti-aliasing effects")
print("- Application to real EEG data")
print("- Selective connectivity analysis")
print("- Epoch averaging for group analysis")

# %%
# References
# ----------
# .. footbibliography::
