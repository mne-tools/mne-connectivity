#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Authors: Jordan Drew <jadrew43@uw.edu>

"""

'''
Test script to ensure LDS API is functioning properly
'''

import pickle
import mne
from megssm.models import LDS
import numpy as np

def test_state_space_output():
    
    # define paths to sample data
    data_path = mne.datasets.sample.data_path()
    sample_folder = data_path / 'MEG/sample'

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
    label_names = ['Aud-lh', 'Aud-rh', 'Vis-lh', 'Vis-rh']
    labels = [mne.read_label(sample_folder / 'labels' / f'{label}.label',
                              subject='sample') for label in label_names]

    # initiate model
    model = LDS(lam0=0, lam1=100)
    model.add_subject('sample', condition, epochs, labels, fwd, cov)
    model.fit(niter=50, verbose=2)
    
    with open('sample A_t', 'rb') as f:
        A_t_ = pickle.load(f)
    np.testing.assert_allclose(A_t_, model.A_t_)
    print('Model is working!')
    
test_state_space_output()