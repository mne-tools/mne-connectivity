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
from scipy.sparse import csr_matrix 

#where should these files live within mne-connectivity repo?
from megssm.mne_util import ROIToSourceMap, _scale_sensor_data #mne_util is from MEGLDS repo
from megssm.models import MEGLDS as LDS

# from util import Carray ##skip import just pasted; util also from MEGLDS repo
Carray64 = lambda X: np.require(X, dtype=np.float64, requirements='C')
Carray32 = lambda X: np.require(X, dtype=np.float32, requirements='C')
Carray = Carray64

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

prepochs = epochs
#make internal to LDS
roi_to_src = ROIToSourceMap(fwd, labels) # compute ROI-to-source map
scales = {'eeg_scale' : 1, 'mag_scale' : 1, 'grad_scale' : 1}
fwd_src_snsr, fwd_roi_snsr, snsr_cov, epochs = \
        _scale_sensor_data(epochs, fwd, noise_cov, roi_to_src, **scales)
#return G/fwd_src_snsr, GL/fwd_roi_snsr, Q_snsr/snsr_cov, epochs
#without scale_sensor_data()...equivalent? no
epochs_cov = mne.make_ad_hoc_cov(prepochs.info)
W, _ = mne.cov.compute_whitener(epochs_cov, prepochs.info)
fwd_data = fwd['sol']['data'].copy()
data = prepochs.get_data().copy()
#------------
G = np.dot(W,fwd_data) #fwd_src_snsr
GL = Carray(csr_matrix.dot(roi_to_src.fwd_src_roi.T, fwd_data.T).T) #fwd_roi_snsr
cov = np.dot(W, np.dot(noise_cov.data.copy(), W.T)) #snsr_cov
# postpochs = np.dot(W, data)


# num_rois = len(labels)
# timepts = len(epochs.times)
# model = LDS(num_rois, timepts, fwd, labels, noise_cov)  # only needs the forward, labels, and noise_cov to be initialized
# subjectdata = [(epochs, fwd_roi_snsr(C), fwd_src_snsr(G), snsr_cov(Q_e,Q_snsr), roi_idx(Q_J))]
# model.set_data(subject_data)
# model.fit(epochs)  # now only needs epochs to fit

















