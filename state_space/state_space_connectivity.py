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

#where should these files live within mne-connectivity repo?
from megssm.models import MEGLDS as LDS
from megssm.plotting import plot_A_t_

## define paths to sample data
path = None
path = '/Users/jordandrew/Documents/MEG/mne_data'#'/MNE-sample-data'
data_path = mne.datasets.sample.data_path(path=path)
sample_folder = data_path / 'MEG/sample'
subjects_dir = data_path / 'subjects'

## import raw data and find events 
raw_fname = sample_folder / 'sample_audvis_raw.fif' 
raw = mne.io.read_raw_fif(raw_fname).crop(tmax=60)
events = mne.find_events(raw, stim_channel='STI 014')

## define epochs using event_dict
event_dict = {'auditory/left': 1, 'auditory/right': 2, 'visual/left': 3,
              'visual/right': 4, 'face': 5, 'buttonpress': 32}
epochs = mne.Epochs(raw, events, tmin=-0.3, tmax=0.7, event_id=event_dict,
                    preload=True).pick_types(meg=True,eeg=True)
condition = 'visual'
epochs = epochs[condition] # choose condition for analysis

## read forward solution, remove bad channels
fwd_fname = sample_folder / 'sample_audvis-meg-eeg-oct-6-fwd.fif' 
fwd = mne.read_forward_solution(fwd_fname)

## read in covariance OR compute noise covariance? noise_cov drops bad chs
cov_fname = sample_folder / 'sample_audvis-cov.fif'
cov = mne.read_cov(cov_fname) # drop bad channels in add_subject
# noise_cov = mne.compute_covariance(epochs, tmax=0)

## read labels for analysis
label_names = ['AUD-lh', 'AUD-rh', 'Vis-lh', 'Vis-rh']
labels = [mne.read_label(sample_folder / 'labels' / f'{label}.label')
          for label in label_names]

## initiate model
num_rois = len(labels)
timepts = len(epochs.times)
model = LDS(num_rois, timepts, lam0=0, lam1=100)  # only needs the forward, labels, and noise_cov to be initialized

model.add_subject('sample', subjects_dir, epochs, labels, fwd, cov)
#when to use compute_cov vs read_cov? ie cov vs noise_cov

model.fit(niter=100, verbose=1)
A_t_ = model.A_t_
assert A_t_.shape == (timepts, num_rois, num_rois)

with mpl.rc_context():
    {'xtick.labelsize': 'x-small', 'ytick.labelsize': 'x-small'}
    fig, ax = plt.subplots(num_rois, num_rois, constrained_layout=True, squeeze=False,
                       figsize=(12, 10))
    plot_A_t_(A_t_, labels=label_names, times=epochs.times, ax=ax)
    fig.suptitle(condition)















