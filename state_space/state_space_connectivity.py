#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Authors: Jordan Drew <jadrew43@uw.edu>

"""

'''
For 'mne-connectivity/examples/' to show usage of LDS
Use MNE-sample-data for auditory/left
'''

## import necessary libraries
import mne
import matplotlib.pyplot as plt
import matplotlib as mpl


from megssm.models import MEGLDS as LDS
from megssm.plotting import plot_A_t_


# define paths to sample data
path = None
data_path = mne.datasets.sample.data_path(path=path)
sample_folder = data_path / 'MEG/sample'
subjects_dir = data_path / 'subjects'

## import raw data and find events
raw_fname = sample_folder / 'sample_audvis_raw.fif'
raw = mne.io.read_raw_fif(raw_fname).crop(tmax=60)
events = mne.find_events(raw, stim_channel='STI 014')

## define epochs using event_dict
event_dict = {'auditory_left': 1, 'auditory_right': 2, 'visual_left': 3,
              'visual_right': 4}
epochs = mne.Epochs(raw, events, tmin=-0.2, tmax=0.7, event_id=event_dict,
                    preload=True).pick_types(meg=True,eeg=True)
condition = 'auditory_left'

## read forward solution, remove bad channels
fwd_fname = sample_folder / 'sample_audvis-meg-eeg-oct-6-fwd.fif'
fwd = mne.read_forward_solution(fwd_fname)

## read in covariance
cov_fname = sample_folder / 'sample_audvis-cov.fif'
cov = mne.read_cov(cov_fname)

## read labels for analysis
regexp = '^(G_temp_sup-G_T_transv.*|Pole_occipital)'
labels = mne.read_labels_from_annot(
    'sample', 'aparc.a2009s', regexp=regexp, subjects_dir=subjects_dir)
label_names = [label.name for label in labels]
assert len(label_names) == 4
# brain = mne.viz.Brain('sample', surf='inflated', subjects_dir=subjects_dir)
# for label in labels:
#     brain.add_label(label)
# raise RuntimeError


# initiate model
model = LDS(lam0=0, lam1=100)
model.add_subject('sample', condition, epochs, labels, fwd, cov)
model.fit(niter=100, verbose=2)

#plot model output
num_roi = model.num_roi
n_timepts = model.n_timepts
times = model.times
A_t_ = model.A_t_
assert A_t_.shape == (n_timepts, num_roi, num_roi)

with mpl.rc_context():
    {'xtick.labelsize': 'x-small', 'ytick.labelsize': 'x-small'}
    fig, ax = plt.subplots(num_roi, num_roi, constrained_layout=True,
                           squeeze=False, figsize=(12, 10))
    plot_A_t_(A_t_, labels=label_names, times=times, ax=ax)
    fig.suptitle('testing_')
    diag_lims = [0, 1]
    off_lims = [-0.6, 0.6]
    for ri, row in enumerate(ax):
        for ci, a in enumerate(row):
            ylim = diag_lims if ri == ci else off_lims
            a.set(ylim=ylim, xlim=times[[0, -1]])
            if ri == 0:
                a.set_title(a.get_title(), fontsize='small')
            if ci == 0:
                a.set_ylabel(a.get_ylabel(), fontsize='small')
            for line in a.lines:
                line.set_clip_on(False)
                line.set(lw=1.)
            if ci != 0:
                a.yaxis.set_major_formatter(plt.NullFormatter())
            if ri != len(label_names) - 1:
                a.xaxis.set_major_formatter(plt.NullFormatter())
            if ri == ci:
                for spine in a.spines.values():
                    spine.set(lw=2)
            else:
                a.axhline(0, color='k', ls=':', lw=1.)
