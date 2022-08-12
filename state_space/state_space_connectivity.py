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
import pickle

load = 0

if not load:
# define paths to sample data
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
    event_dict = {'auditory_left': 1, 'auditory_right': 2, 'visual_left': 3,
                  'visual_right': 4}#, 'face': 5, 'buttonpress': 32}
    epochs = mne.Epochs(raw, events, tmin=-0.3, tmax=0.7, event_id=event_dict,
                        preload=True).pick_types(meg=True,eeg=True)
    condition = 'auditory_left'
    epochs = epochs[condition] # choose condition for analysis
    
    
    
    #SNR boost epochs, bootstraps of 3    
    # def bootstrap_subject(subject_name, seed=8675309, sfreq=100, lower=None, 
    #                       upper=None, nbootstrap=3, g_nsamples=-5, 
    #                       overwrite=False, validation_set=True):
    #     import autograd.numpy as np
    #     import os
        
    #     # subjects =  ['sample']
    #     datasets = ['train', 'validation']
    #     use_erm = eq = False
    #     independent = False
    #     if g_nsamples == 0:
    #         print('nsamples == 0, ensuring independence of samples')
    #         independent = True
    #     elif g_nsamples == -1:
    #         print("using half of trials per sample")
    #     elif g_nsamples == -2:
    #         print("using empty room noise at half of trials per sample")
    #         use_erm = True
    #     elif g_nsamples == -3:
    #         print("using independent and trial-count equalized samples")
    #         eq = True
    #         independent = True
    #     elif g_nsamples == -4:
    #         print("using independent, trial-count equailized, non-boosted samples")
    #         assert nbootstrap == 0  # sanity check
    #         eq = True
    #         independent = True
    #         datasets = ['train']
    #     elif g_nsamples == -5:
    #         print("using independent, trial-count equailized, integer boosted samples")
    #         eq = True
    #         independent = True
    #         datasets = ['train']

    #     if lower is not None or upper is not None:
    #         if upper is None:
    #             print('high-pass filtering at %.2f Hz' % lower)
    #         elif lower is None:
    #             print('low-pass filtering at %.2f Hz' % upper)
    #         else:
    #             print('band-pass filtering from %.2f-%.2f Hz' % (lower, upper))

    #     if sfreq is not None:
    #         print('resampling to %.2f Hz' % sfreq)
            
    #     print(":: processing subject %s" % subject_name)
    #     np.random.seed(seed)

    #     for dataset in datasets:

    #         print('  generating ', dataset, ' set')
    #         datadir = './data'

    #         subj_dir = os.path.join(datadir, subject_name)
    #         print("subject dir:" + subj_dir)
    #         if not os.path.exists(subj_dir):
    #             print('  %s not found, skipping' % subject_name)
    #             return

    #         epochs_dir = os.path.join(datadir, subject_name, 'epochs')
    #         epochs_fname = "All_55-sss_%s-epo.fif" % subject_name
    #         epochs_bs_fname = (epochs_fname.split('-epo')[0] +
    #                            "-bootstrap_%d-nsamples_%d-seed_%d%s%s%s-"
    #                            % (nbootstrap, g_nsamples, seed,
    #                               '-lower_%.2e' % lower if lower is not None else '',
    #                               '-upper_%.2e' % upper if upper is not None else '',
    #                               '-sfreq_%.2e' % sfreq if sfreq is not None else '') +
    #                            dataset + "-epo.fif")

    #         if os.path.exists(os.path.join(epochs_dir, epochs_bs_fname)) and \
    #                 not overwrite:
    #             print("    => found existing bootstrapped epochs, skipping")
    #             return

    #         epochs = mne.read_epochs(os.path.join(epochs_dir, epochs_fname),
    #                                  preload=True)
    #         condition_map = {'auditory_left':['auditory_left'],'auditory_right': ['auditory_right'],
    #                          'visual_left': ['visual_left'], 'visual_right': ['visual_right']}
    #         condition_eq_map = dict(auditory_left=['auditory_left'], auditory_right=['auditory_right'],
    #                                 visual_left=['visual_left'], visual_right='visual_right')
    #         if eq:
    #             epochs.equalize_event_counts(list(condition_map))
    #             cond_map = condition_eq_map
            
    #         # apply band-pass filter to limit signal to desired frequency band
    #         if lower is not None or upper is not None:
    #             epochs = epochs.filter(lower, upper)

    #         # perform resampling with specified sampling frequency
    #         if sfreq is not None:
    #             epochs = epochs.resample(sfreq)

    #         data_bs_all = list()
    #         events_bs_all = list()
    #         for cond in sorted(cond_map.keys()):
    #             print("    -> condition %s: bootstrapping" % cond, end='')            
    #             ep = epochs[cond_map[cond]] 
    #             dat = ep.get_data().copy()
    #             ntrials, T, p = dat.shape 
                
    #             use_bootstrap = nbootstrap
    #             if g_nsamples == -4:
    #                 nsamples = 1
    #                 use_bootstrap = ntrials
    #             elif g_nsamples == -5:
    #                 nsamples = nbootstrap
    #                 use_bootstrap = ntrials // nsamples
    #             elif independent:
    #                 nsamples = (ntrials - 1) // use_bootstrap
    #             elif g_nsamples in (-1, -2):
    #                 nsamples = ntrials // 2
    #             else:
    #                 assert g_nsamples > 0
    #                 nsamples = g_nsamples
    #             print("    using %d samples (%d trials)"
    #                   % (nsamples, use_bootstrap))

    #             # bootstrap here
    #             if independent:  # independent
    #                 if nsamples == 1 and use_bootstrap == ntrials:
    #                     inds = np.arange(ntrials)
    #                 else:
    #                     inds = np.random.choice(ntrials, nsamples * use_bootstrap)
    #                 inds.shape = (use_bootstrap, nsamples)
    #                 dat_bs = np.mean(dat[inds], axis=1)
    #                 events_bs = ep.events[inds[:, 0]]
    #                 assert dat_bs.shape[0] == events_bs.shape[0]
    #             else:
    #                 dat_bs = np.empty((ntrials, T, p))
    #                 events_bs = np.empty((ntrials, 3), dtype=int)
    #                 for i in range(ntrials):

    #                     inds = list(set(range(ntrials)).difference([i]))
    #                     inds = np.random.choice(inds, size=nsamples,
    #                                             replace=False)
    #                     inds = np.append(inds, i)

    #                     dat_bs[i] = np.mean(dat[inds], axis=0)
    #                     events_bs[i] = ep.events[i]

    #                 inds = np.random.choice(ntrials, size=use_bootstrap,
    #                                         replace=False)
    #                 dat_bs = dat_bs[inds]
    #                 events_bs = events_bs[inds]
                
    #             assert dat_bs.shape == (use_bootstrap, T, p)
    #             assert events_bs.shape == (use_bootstrap, 3)
    #             assert (events_bs[:, 2] == events_bs[0, 2]).all() #not working for sample_info

    #             data_bs_all.append(dat_bs)
    #             events_bs_all.append(events_bs)

    #         # write bootstrap epochs
    #         info_dict = epochs.info.copy()

    #         dat_all = np.vstack(data_bs_all)
    #         events_all = np.vstack(events_bs_all)
    #         # replace first column with sequential list as we don't really care
    #         # about the raw timings
    #         events_all[:, 0] = np.arange(events_all.shape[0])

    #         epochs_bs = mne.EpochsArray(
    #             dat_all, info_dict, events=events_all, tmin=-0.2,
    #             event_id=epochs.event_id.copy(), on_missing='ignore')

    #         print("    saving bootstrapped epochs (%s)" % (epochs_bs_fname,))
    #         epochs_bs.save(os.path.join(epochs_dir, epochs_bs_fname))
    # bootstrap_subject('sample')
    
    
    
    
    
    
    
    ## read forward solution, remove bad channels
    fwd_fname = sample_folder / 'sample_audvis-meg-eeg-oct-6-fwd.fif' 
    fwd = mne.read_forward_solution(fwd_fname)
    
    ## read in covariance OR compute noise covariance? noise_cov drops bad chs
    cov_fname = sample_folder / 'sample_audvis-cov.fif'
    cov = mne.read_cov(cov_fname) # drop bad channels in add_subject
    # noise_cov = mne.compute_covariance(epochs, tmax=0)
    
    ## read labels for analysis
    label_names = ['AUD-lh', 'AUD-rh', 'Vis-lh', 'Vis-rh']
    labels = [mne.read_label(sample_folder / 'labels' / f'{label}.label',
                              subject='sample') for label in label_names]
    
    # initiate model
    num_rois = len(labels)
    n_timepts = len(epochs.times)
    times = epochs.times
    # model = LDS(num_rois, n_timepts, lam0=0, lam1=100)  # only needs the forward, labels, and noise_cov to be initialized
    model = LDS(lam0=0, lam1=100)
    
    model.add_subject('sample', subjects_dir, epochs, labels, fwd, cov)
    # #when to use compute_cov vs read_cov? ie cov vs noise_cov
    
    # model.fit(niter=100, verbose=1)
    # A_t_ = model.A_t_
    # assert A_t_.shape == (n_timepts, num_rois, num_rois)
    
    # file = open('sample_subj_stdNone.pkl','wb')
    # pickle.dump([model, num_rois, n_timepts, times, condition, label_names],file)
    # file.close

# if load:
#     with open('sample_subj_stdNone.pkl','rb') as f:
#         model, num_rois, n_timepts, times, condition, label_names = pickle.load(f)

# with mpl.rc_context():
#     {'xtick.labelsize': 'x-small', 'ytick.labelsize': 'x-small'}
#     fig, ax = plt.subplots(num_rois, num_rois, constrained_layout=True, squeeze=False,
#                         figsize=(12, 10))
#     plot_A_t_(model.A_t_, labels=label_names, times=times, ax=ax)
#     fig.suptitle(condition)
    
















