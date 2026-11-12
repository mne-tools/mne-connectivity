"""
=========================================================================
Compute source space connectivity and visualize it using a circular graph
=========================================================================

This example computes the all-to-all connectivity between 68 regions in
source space based on dSPM inverse solutions and a FreeSurfer cortical
parcellation. The connectivity is visualized using a circular graph which
is ordered based on the locations of the regions in the axial plane.
"""
# Authors: Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#          Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Nicolas P. Rougier (graph code borrowed from his matplotlib gallery)
#          Payam S. Shabestari <payam.sadeghi74@gmail.com>
#
# License: BSD (3-clause)
# sphinx_gallery_thumbnail_number = 2

import matplotlib.pyplot as plt
import mne
import numpy as np
from matplotlib.patches import Patch
from mne.datasets import sample
from mne.minimum_norm import apply_inverse_epochs, read_inverse_operator
from mne.viz import circular_layout

from mne_connectivity import spectral_connectivity_epochs
from mne_connectivity.viz import plot_connectivity_circle

print(__doc__)

###############################################################################
# Load our data
# -------------
#
# First we'll load the data we'll use in connectivity estimation. We'll use
# the sample MEG data provided with MNE.

data_path = sample.data_path()
subjects_dir = data_path / "subjects"
fname_inv = data_path / "MEG/sample/sample_audvis-meg-oct-6-meg-inv.fif"
fname_raw = data_path / "MEG/sample/sample_audvis_filt-0-40_raw.fif"
fname_event = data_path / "MEG/sample/sample_audvis_filt-0-40_raw-eve.fif"

# Load data
inverse_operator = read_inverse_operator(fname_inv)
raw = mne.io.read_raw_fif(fname_raw)
events = mne.read_events(fname_event)

# Add a bad channel
raw.info["bads"] += ["MEG 2443"]

# Pick MEG channels
picks = mne.pick_types(
    raw.info, meg=True, eeg=False, stim=False, eog=True, exclude="bads"
)

# Define epochs for left-auditory condition
event_id, tmin, tmax = 1, -0.2, 0.5
epochs = mne.Epochs(
    raw,
    events,
    event_id,
    tmin,
    tmax,
    picks=picks,
    baseline=(None, 0),
    reject=dict(mag=4e-12, grad=4000e-13, eog=150e-6),
)

###############################################################################
# Compute inverse solutions and their connectivity
# ------------------------------------------------
#
# Next, we need to compute the inverse solution for this data. This will return
# the sources / source activity that we'll use in computing connectivity. We'll
# compute the connectivity in the alpha band of these sources. We can specify
# particular frequencies to include in the connectivity with the ``fmin`` and
# ``fmax`` flags. Notice from the status messages how mne-python:
#
# 1. reads an epoch from the raw file
# 2. applies SSP and baseline correction
# 3. computes the inverse to obtain a source estimate
# 4. averages the source estimate to obtain a time series for each label
# 5. includes the label time series in the connectivity computation
# 6. moves to the next epoch.
#
# This behaviour is because we are using generators. Since we only need to
# operate on the data one epoch at a time, using a generator allows us to
# compute connectivity in a computationally efficient manner where the amount
# of memory (RAM) needed is independent from the number of epochs.

# Compute inverse solution and for each epoch. By using "return_generator=True"
# stcs will be a generator object instead of a list.
snr = 1.0  # use lower SNR for single epochs
lambda2 = 1.0 / snr**2
method = "dSPM"  # use dSPM method (could also be MNE or sLORETA)
stcs = apply_inverse_epochs(
    epochs, inverse_operator, lambda2, method, pick_ori="normal", return_generator=True
)

# Get labels for FreeSurfer 'aparc' cortical parcellation with 34 labels/hemi
labels = mne.read_labels_from_annot("sample", parc="aparc", subjects_dir=subjects_dir)
label_colors = [label.color for label in labels]

# Average the source estimates within each label using sign-flips to reduce
# signal cancellations, also here we return a generator
src = inverse_operator["src"]
label_ts = mne.extract_label_time_course(
    stcs, labels, src, mode="mean_flip", return_generator=True
)

fmin = 8.0
fmax = 13.0
sfreq = raw.info["sfreq"]  # the sampling frequency
con_methods = ["pli", "wpli2_debiased", "ciplv"]
con = spectral_connectivity_epochs(
    label_ts,
    method=con_methods,
    mode="multitaper",
    sfreq=sfreq,
    fmin=fmin,
    fmax=fmax,
    faverage=True,
    mt_adaptive=True,
    n_jobs=1,
)

# con is a 3D array, get the connectivity for the first (and only) freq. band
# for each method
con_res = dict()
for method, c in zip(con_methods, con):
    con_res[method] = c.get_data(output="dense")[:, :, 0]

###############################################################################
# Make a connectivity plot
# ------------------------
#
# Now, we visualize this connectivity using a circular graph layout.

# First, we reorder the labels based on their location in the left hemi
label_names = [label.name for label in labels]

lh_labels = [name for name in label_names if name.endswith("lh")]

# Get the y-location of the label
label_ypos = list()
for name in lh_labels:
    idx = label_names.index(name)
    ypos = np.mean(labels[idx].pos[:, 1])
    label_ypos.append(ypos)

# Reorder the labels based on their location
lh_labels = [label for (yp, label) in sorted(zip(label_ypos, lh_labels))]

# For the right hemi
rh_labels = [label[:-2] + "rh" for label in lh_labels]

# Save the plot order and create a circular layout
node_order = list()
node_order.extend(lh_labels[::-1])  # reverse the order
node_order.extend(rh_labels)

node_angles = circular_layout(
    label_names, node_order, start_pos=90, group_boundaries=[0, len(label_names) / 2]
)

# Plot the graph using node colors from the FreeSurfer parcellation. We only
# show the 300 strongest connections.
fig, ax = plt.subplots(figsize=(8, 8), facecolor="black", subplot_kw=dict(polar=True))
plot_connectivity_circle(
    con_res["pli"],
    label_names,
    n_lines=300,
    node_angles=node_angles,
    node_colors=label_colors,
    title="All-to-All Connectivity left-Auditory Condition (PLI)",
    ax=ax,
)
fig.tight_layout()

###############################################################################
# Group the regions into lobes
# ----------------------------
#
# With many regions, the names on the circle become hard to read. Here, we
# group the regions into lobes and show one name and one color per lobe. This
# only needs the existing parameters: ``node_angles`` orders the nodes by lobe
# and leaves a gap between lobes, ``node_colors`` gives all regions of a lobe
# the same color, and empty strings hide all but one name per lobe.

# The lobes of the regions in the 'aparc' parcellation, from front to back
lobes = {
    "frontal": [
        "superiorfrontal",
        "rostralmiddlefrontal",
        "caudalmiddlefrontal",
        "parsopercularis",
        "parstriangularis",
        "parsorbitalis",
        "lateralorbitofrontal",
        "medialorbitofrontal",
        "precentral",
        "paracentral",
        "frontalpole",
    ],
    "insula": ["insula"],
    "cingulate": [
        "rostralanteriorcingulate",
        "caudalanteriorcingulate",
        "posteriorcingulate",
        "isthmuscingulate",
    ],
    "temporal": [
        "superiortemporal",
        "middletemporal",
        "inferiortemporal",
        "bankssts",
        "fusiform",
        "transversetemporal",
        "entorhinal",
        "temporalpole",
        "parahippocampal",
    ],
    "parietal": [
        "superiorparietal",
        "inferiorparietal",
        "supramarginal",
        "postcentral",
        "precuneus",
    ],
    "occipital": ["lateraloccipital", "lingual", "cuneus", "pericalcarine"],
}
lobe_of = {region: lobe for lobe, regions in lobes.items() for region in regions}
lobe_colors = dict(zip(lobes, plt.colormaps["tab10"].colors))

# Order the nodes by lobe with the front at the top, mirroring the hemispheres
lh_order = [f"{region}-lh" for regions in lobes.values() for region in regions]
rh_order = [f"{region}-rh" for regions in lobes.values() for region in regions]
lobe_order = lh_order + rh_order[::-1]

# Start a new group wherever the lobe or the hemisphere changes
groups = [lobe_of[name[:-3]] + name[-3:] for name in lobe_order]
group_starts = [0] + [i for i in range(1, len(groups)) if groups[i] != groups[i - 1]]
lobe_angles = circular_layout(
    label_names,
    lobe_order,
    start_pos=90,
    group_boundaries=group_starts,
    group_sep=4,
)

# Color each region by its lobe, and name only the middle region of each lobe
lobe_node_colors = [lobe_colors[lobe_of[name[:-3]]] for name in label_names]
lobe_names = [""] * len(label_names)
for start, stop in zip(group_starts, group_starts[1:] + [len(lobe_order)]):
    middle = lobe_order[(start + stop - 1) // 2]
    lobe_names[label_names.index(middle)] = lobe_of[middle[:-3]]

fig, ax = plt.subplots(figsize=(8, 8), facecolor="black", subplot_kw=dict(polar=True))
plot_connectivity_circle(
    con_res["pli"],
    lobe_names,
    n_lines=300,
    node_angles=lobe_angles,
    node_colors=lobe_node_colors,
    title="All-to-All Connectivity left-Auditory Condition (PLI)",
    ax=ax,
)
fig.tight_layout()

###############################################################################
# Make multiple connectivity plots in the same figure
# ---------------------------------------------------
#
# We can also assign these connectivity plots to axes in a figure. Below we'll
# show the connectivity plot using three different connectivity methods. The
# plots are too small for node names, so we add a legend for the lobe colors
# instead.

fig, axes = plt.subplots(
    1, 3, figsize=(8, 4), facecolor="black", subplot_kw=dict(polar=True)
)
no_names = [""] * len(label_names)
for ax, method in zip(axes, con_methods):
    plot_connectivity_circle(
        con_res[method],
        no_names,
        n_lines=300,
        node_angles=lobe_angles,
        node_colors=lobe_node_colors,
        title=method,
        padding=0,
        fontsize_colorbar=6,
        ax=ax,
    )
handles = [Patch(color=color, label=lobe) for lobe, color in lobe_colors.items()]
fig.legend(
    handles=handles,
    loc="lower center",
    ncols=len(handles),
    frameon=False,
    labelcolor="white",
    fontsize=8,
)


###############################################################################
# Save the figure (optional)
# --------------------------
#
# By default matplotlib does not save using the facecolor, even though this was
# set when the figure was generated. If not set via savefig, the labels, title,
# and legend will be cut off from the output png file.
#
# .. code-block:: python
#
#     import os.path as op
#     fname_fig = op.join(data_path, 'MEG', 'sample',
#                         'plot_inverse_connect.png')
#     fig.savefig(fname_fig, facecolor=fig.get_facecolor())
