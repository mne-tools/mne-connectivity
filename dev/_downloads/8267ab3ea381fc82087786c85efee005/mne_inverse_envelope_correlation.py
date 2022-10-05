"""
.. _ex-envelope-correlation:

=============================================
Compute envelope correlations in source space
=============================================

Compute envelope correlations of orthogonalized activity
:footcite:`HippEtAl2012,KhanEtAl2018` using pairwise and symmetric
orthogonalization :footcite:`ColcloughEtAl2015` in source space using
resting state CTF data.

Note that the original procedure for symmetric orthogonalization in
:footcite:`ColcloughEtAl2015` is:

1. Extract inverse label data from raw
2. Symmetric orthogonalization
3. Band-pass filter
4. Hilbert transform and absolute value
5. Low-pass (1 Hz)

Here we follow the procedure:

1. Epoch data, then for each
2. Extract inverse label data for each epoch
3. Symmetric orthogonalization for each epoch
4. Band-pass filter each epoch
5. Hilbert transform and absolute value (inside envelope_correlation)

The differences between these two should hopefully be fairly minimal given
the pairwise orthogonalization used in :footcite:`KhanEtAl2018` used a similar
pipeline.
"""

# Authors: Eric Larson <larson.eric.d@gmail.com>
#          Sheraz Khan <sheraz@khansheraz.com>
#          Denis Engemann <denis.engemann@gmail.com>
#
# License: BSD (3-clause)

import os.path as op

import numpy as np
import matplotlib.pyplot as plt

import mne
import mne_connectivity
from mne_connectivity import envelope_correlation
from mne.minimum_norm import make_inverse_operator, apply_inverse_epochs
from mne.preprocessing import compute_proj_ecg, compute_proj_eog

data_path = mne.datasets.brainstorm.bst_resting.data_path()
subjects_dir = op.join(data_path, 'subjects')
subject = 'bst_resting'
trans = op.join(data_path, 'MEG', 'bst_resting', 'bst_resting-trans.fif')
src = op.join(subjects_dir, subject, 'bem', subject + '-oct-6-src.fif')
bem = op.join(subjects_dir, subject, 'bem', subject + '-5120-bem-sol.fif')
raw_fname = op.join(data_path, 'MEG', 'bst_resting',
                    'subj002_spontaneous_20111102_01_AUX.ds')

##############################################################################
# Here we do some things in the name of speed, such as crop (which will
# hurt SNR) and downsample. Then we compute SSP projectors and apply them.

raw = mne.io.read_raw_ctf(raw_fname, verbose='error')
raw.crop(0, 60).pick_types(meg=True, eeg=False).load_data().resample(80)
raw.apply_gradient_compensation(3)
projs_ecg, _ = compute_proj_ecg(raw, n_grad=1, n_mag=2)
projs_eog, _ = compute_proj_eog(raw, n_grad=1, n_mag=2, ch_name='MLT31-4407')
raw.add_proj(projs_ecg + projs_eog)
raw.apply_proj()
raw.filter(0.1, None)  # this helps with symmetric orthogonalization later
cov = mne.compute_raw_covariance(raw)  # compute before band-pass of interest

##############################################################################
# Compute the forward and inverse
# -------------------------------

src = mne.read_source_spaces(src)
fwd = mne.make_forward_solution(raw.info, trans, src, bem, verbose=True)
del src
inv = make_inverse_operator(raw.info, fwd, cov)
del fwd

##############################################################################
# Now we create epochs and prepare to band-pass filter them.

duration = 10.
events = mne.make_fixed_length_events(raw, duration=duration)
tmax = duration - 1. / raw.info['sfreq']
epochs = mne.Epochs(raw, events=events, tmin=0, tmax=tmax,
                    baseline=None, reject=dict(mag=20e-13))
sfreq = epochs.info['sfreq']
del raw, projs_ecg, projs_eog

# %%
# Do pairwise-orthogonalized envelope correlation
# -----------------------------------------------

# sphinx_gallery_thumbnail_number = 2

labels = mne.read_labels_from_annot(subject, 'aparc_sub',
                                    subjects_dir=subjects_dir)
stcs = apply_inverse_epochs(epochs, inv, lambda2=1. / 9., pick_ori='normal',
                            return_generator=True)
label_ts = mne.extract_label_time_course(
    stcs, labels, inv['src'], return_generator=False)
del stcs


def bp_gen(label_ts):
    """Make a generator that band-passes on the fly."""
    for ts in label_ts:
        yield mne.filter.filter_data(ts, sfreq, 14, 30)


corr_obj = envelope_correlation(
    bp_gen(label_ts), orthogonalize='pairwise')
corr = corr_obj.combine()
corr = corr.get_data(output='dense')[:, :, 0]


def plot_corr(corr, title):
    fig, ax = plt.subplots(figsize=(4, 4), constrained_layout=True)
    ax.imshow(corr, cmap='viridis', clim=np.percentile(corr, [5, 95]))
    fig.suptitle(title)


plot_corr(corr, 'Pairwise')


def plot_degree(corr, title):
    threshold_prop = 0.15  # percentage of strongest edges to keep in the graph
    degree = mne_connectivity.degree(corr, threshold_prop=threshold_prop)
    stc = mne.labels_to_stc(labels, degree)
    stc = stc.in_label(mne.Label(inv['src'][0]['vertno'], hemi='lh') +
                       mne.Label(inv['src'][1]['vertno'], hemi='rh'))
    return stc.plot(
        clim=dict(kind='percent', lims=[75, 85, 95]), colormap='gnuplot',
        subjects_dir=subjects_dir, views='dorsal', hemi='both',
        smoothing_steps=25, time_label=title)


brain = plot_degree(corr, 'Beta (pairwise, aparc_sub)')

# %%
# Do symmetric-orthogonalized envelope correlation
# ------------------------------------------------
# Here we need the number of labels to be less than the rank of the data
# (here around 200), because all label time courses are orthogonalized
# relative to one another. ``'aparc_sub'`` has over 400 labels, so here we
# use ``'aparc.a2009s'``, which has fewer than 200.

labels = mne.read_labels_from_annot(subject, 'aparc.a2009s',
                                    subjects_dir=subjects_dir)
stcs = apply_inverse_epochs(epochs, inv, lambda2=1. / 9., pick_ori='normal',
                            return_generator=True)
label_ts = mne.extract_label_time_course(
    stcs, labels, inv['src'], return_generator=True)
del stcs, epochs

label_ts_orth = mne_connectivity.envelope.symmetric_orth(label_ts)
corr_obj = envelope_correlation(  # already orthogonalized earlier
    bp_gen(label_ts_orth), orthogonalize=False)

# average over epochs, take absolute value, and plot
corr = corr_obj.combine()
corr = corr.get_data(output='dense')[:, :, 0]
corr.flat[::corr.shape[0] + 1] = 0  # zero out the diagonal
corr = np.abs(corr)

plot_corr(corr, 'Symmetric')
plot_degree(corr, 'Beta (symmetric, aparc.a2009s)')
# %%
# References
# ----------
# .. footbibliography::
