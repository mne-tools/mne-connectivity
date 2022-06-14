#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 13:15:43 2022

@author: jordandrew

For 'mne-connectivity/examples/' to show usage of MEGLDS 
Use MNE-sample-data for auditory/left
"""
import mne
import numpy as np 
import matplotlib.pyplot as plt
from mne_util import ROIToSourceMap, scale_sensor_data

# data_path = mne.datasets.sample.data_path()
data_path = '/Users/jordandrew/Documents/MEG/mne_data/MNE-sample-data'
sample_folder = '/MEG/sample'
raw_fname = data_path + sample_folder + '/sample_audvis_raw.fif' #how many subjects?
subjects_dir = data_path + '/subjects'

raw = mne.io.read_raw_fif(raw_fname).crop(tmax=60)
events = mne.find_events(raw, stim_channel='STI 014')
""" OR
raw_events_fname = data_path + sample_folder + '/sample_audvis_raw-eve.fif'
events = mne.read_events(raw_events_fname)
"""


## compute forward solution
sphere = mne.make_sphere_model('auto', 'auto', raw.info)
src = mne.setup_volume_source_space(sphere=sphere, exclude=30., pos=15.)
fwd = mne.make_forward_solution(raw.info, trans=None, src=src, bem=sphere)
fwd['src'].append( fwd['src'][0]) #fwd['src'] needs lh and rh; duplicated here


#event_id = 1
event_dict = {'auditory/left': 1, 'auditory/right': 2, 'visual/left': 3,
              'visual/right': 4, 'face': 5, 'buttonpress': 32}
epochs = mne.Epochs(raw, events, tmin=-0.3, tmax=0.7, event_id=event_dict,
                    preload=True)
# del raw

## compute covariance 
noise_cov = mne.compute_covariance(epochs, tmax=0) #tmax=0 assuming no activity from tmin to tmax?
labels = mne.read_labels_from_annot('sample', subjects_dir=subjects_dir)
roi_to_src = ROIToSourceMap(fwd, labels) # compute ROI-to-source map
scales = {'eeg_scale' : 1, 'mag_scale' : 1, 'grad_scale' : 1}
fwd_sr_sn, fwd_roi_sn, snsr_cov, epochs = \
        scale_sensor_data(epochs, fwd, noise_cov, roi_to_src, **scales)







# model = MEGLDS(fwd, labels, noise_cov)  # only needs the forward, labels, and noise_cov to be initialized
# model.fit(epochs)  # now only needs epochs to fit