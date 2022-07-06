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
# import numpy as np 
# import matplotlib.pyplot as plt
# from scipy.sparse import csr_matrix 

#where should these files live within mne-connectivity repo?
# from megssm.mne_util import ROIToSourceMap, _scale_sensor_data #mne_util is from MEGLDS repo
from megssm.models import MEGLDS as LDS

## define paths to sample data
path = None
path = '/Users/jordandrew/Documents/MEG/mne_data'#'/MNE-sample-data'
data_path = mne.datasets.sample.data_path(path=path)
sample_folder = data_path / 'MEG/sample'
subjects_dir = data_path / 'subjects'

## import raw data and find events 
raw_fname = sample_folder + '/sample_audvis_raw.fif' 
raw = mne.io.read_raw_fif(raw_fname).crop(tmax=60)
events = mne.find_events(raw, stim_channel='STI 014')

## define epochs using event_dict
event_dict = {'auditory/left': 1, 'auditory/right': 2, 'visual/left': 3,
              'visual/right': 4, 'face': 5, 'buttonpress': 32}
epochs = mne.Epochs(raw, events, tmin=-0.3, tmax=0.7, event_id=event_dict,
                    preload=True).pick_types(meg=True,eeg=True,exclude='bads')

## read forward solution, remove bad channels
fwd_fname = sample_folder + '/sample_audvis-meg-eeg-oct-6-fwd.fif' 
fwd = mne.read_forward_solution(fwd_fname,exclude=raw.info['bads'])
fwd = mne.convert_forward_solution(fwd, force_fixed=True)

## read in covariance OR compute noise covariance? noise_cov drops bad chs
cov_fname = sample_folder + '/sample_audvis-cov.fif'
cov = mne.read_cov(cov_fname) #has all 366 channels; drop 2?
noise_cov = mne.compute_covariance(epochs, tmax=0)

## read labels for analysis
label_names = ['AUD-lh', 'AUD-rh', 'Vis-lh', 'Vis-rh']
labels = [mne.read_label(sample_folder + 'labels/' + f'{label}.label')
          for label in label_names]

## initiate model
num_rois = len(labels)
timepts = len(epochs.times)
model = LDS(num_rois, timepts, lam0=0, lam1=100)  # only needs the forward, labels, and noise_cov to be initialized

model.add_subject('sample', subjects_dir, epochs, labels, fwd, noise_cov)
#when to use compute_cov vs read_cov?
#should remove epochs



model.em(niter=2)

















