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
import numpy as np 
import matplotlib.pyplot as plt

#where should these files live within mne-connectivity repo?
from megssm.mne_util import ROIToSourceMap, _scale_sensor_data #mne_util is from MEGLDS repo
from megssm.models import MEGLDS as LDS

## define paths to sample data
path = None
path = '/Users/jordandrew/Documents/MEG/mne_data'#'/MNE-sample-data'
data_path = mne.datasets.sample.data_path(path=path)
sample_folder = '/MEG/sample'
raw_fname = data_path + sample_folder + '/sample_audvis_raw.fif' #how many subjects?
subjects_dir = data_path + '/subjects'
fwd_fname = data_path + sample_folder + '/sample_audvis-meg-eeg-oct-6-fwd.fif' #EEG ONLY

## import raw data and find events 
raw = mne.io.read_raw_fif(raw_fname).crop(tmax=60)
events = mne.find_events(raw, stim_channel='STI 014')

## read forward solution, remove bad channels
fwd = mne.read_forward_solution(fwd_fname,exclude=raw.info['bads'])
fwd = mne.convert_forward_solution(fwd, force_fixed=True)

## define epochs using event_dict
event_dict = {'auditory/left': 1, 'auditory/right': 2, 'visual/left': 3,
              'visual/right': 4, 'face': 5, 'buttonpress': 32}
epochs = mne.Epochs(raw, events, tmin=-0.3, tmax=0.7, event_id=event_dict,
                    preload=True)

## compute covariance 
noise_cov = mne.compute_covariance(epochs, tmax=0)
labels = mne.read_labels_from_annot('sample', subjects_dir=subjects_dir)
#when to select specific labels/ROIs for processing?

#make internal to LDS
# roi_to_src = ROIToSourceMap(fwd, labels) # compute ROI-to-source map
# scales = {'eeg_scale' : 1, 'mag_scale' : 1, 'grad_scale' : 1}
# fwd_src_snsr, fwd_roi_snsr, snsr_src_cov, epochs = \
#         _scale_sensor_data(epochs, fwd, noise_cov, roi_to_src, **scales)

# snsr_Q_J = mne.make_ad_hoc_cov(raw.info) #why not square matrix?




num_rois = len(labels)
timepts = len(epochs.times)
model = LDS(num_rois, timepts, fwd, labels, noise_cov)  # only needs the forward, labels, and noise_cov to be initialized
# subjectdata = [(epochs, fwd_roi_snsr(C), fwd_src_snsr(G), snsr_src_cov(R,Q_snsr), roi_idx(Q_J))]
# model.set_data(subject_data)
# model.fit(epochs)  # now only needs epochs to fit

















