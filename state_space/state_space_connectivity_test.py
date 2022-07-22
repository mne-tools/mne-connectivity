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
from megssm import label_util

## define paths to sample data
data_path = '/Users/jordandrew/Documents/MEG/meglds-master/data/sps'
# sample_folder = data_path / 'MEG/sample'
# subjects_dir = data_path / 'subjects'

subjects = ['eric_sps_03','eric_sps_04','eric_sps_05','eric_sps_06',
            'eric_sps_07','eric_sps_09','eric_sps_10','eric_sps_15',
            'eric_sps_17','eric_sps_18','eric_sps_19','eric_sps_21',
            'eric_sps_25','eric_sps_26','eric_sps_31','eric_sps_32']

label_names = ['ACC', 'DLPFC-lh', 'DLPFC-rh', 'AUD-lh', 'AUD-rh', 'FEF-lh',
               'FEF-rh', 'Vis', 'IPS-lh', 'LIPSP', 'IPS-rh', 'RTPJ']
label_func = 'sps_meglds_base_vision_extra'
labels = getattr(label_util, label_func)()
labels = sorted(labels, key=lambda x: x.name)

def eq_trials(epochs, kind):
    """ equalize trial counts """
    import numpy as np
    import mne
    assert kind in ('sub', 'big')
    print('    equalizing trial counts', end='')
    in_names = [
        'LL3', 'LR3', 'LU3', 'LD3', 'RL3', 'RR3', 'RU3', 'RD3',
        'UL3', 'UR3', 'UU3', 'UD3', 'DL3', 'DR3', 'DU3', 'DD3',
        'LL4', 'LR4', 'LU4', 'LD4', 'RL4', 'RR4', 'RU4', 'RD4',
        'UL4', 'UR4', 'UU4', 'UD4', 'DL4', 'DR4', 'DU4', 'DD4',
        'VS_', 'VM_',
        'Junk',
        ]
    out_names = ['LL', 'LR', 'LX', 'UX', 'UU', 'UD', 'VS', 'VM']

    # strip 3/4 and combine
    clean_names = np.unique([ii[:2] for ii in in_names
                             if not ii.startswith('V')])
    for name in clean_names:
        combs = [in_name for in_name in in_names if in_name.startswith(name)]
        new_id = {name: epochs.event_id[combs[-1]] + 1}
        mne.epochs.combine_event_ids(epochs, combs, new_id, copy=False)

    # Now we equalize LU+LD, RU+RD, UL+UR, DL+DR, and combine those
    for n1, n2, oname in zip(('LU', 'RU', 'UL', 'DL'),
                             ('LD', 'RD', 'UR', 'DR'),
                             ('LX', 'RX', 'UX', 'DX')):
        if kind == 'sub':
            epochs.equalize_event_counts([n1, n2])
        new_id = {oname: epochs.event_id[n1] + 1}
        mne.epochs.combine_event_ids(epochs, [n1, n2], new_id, copy=False)

    # Now we equalize "sides"
    cs = dict(L='R', R='L', U='D', D='U')
    for n1 in ['L', 'R', 'U', 'D']:
        # first equalize it with its complement in the second pos
        if kind == 'sub':
            epochs.equalize_event_counts([n1 + n1, n1 + cs[n1]])
            epochs.equalize_event_counts([n1 + n1, cs[n1] + n1])
            epochs.equalize_event_counts([n1 + 'X', cs[n1] + 'X'])

    # now combine cross types
    for n1 in ['L', 'U']:
        # LR+RL=LR, UD+DU=UD
        old_ids = [n1 + cs[n1], cs[n1] + n1]
        if kind == 'sub':
            epochs.equalize_event_counts(old_ids)
        new_id = {n1 + cs[n1]: epochs.event_id[n1 + cs[n1]] + 1}
        mne.epochs.combine_event_ids(epochs, old_ids, new_id, copy=False)
        # LL+RR=LL, UU+DD=UU
        old_ids = [n1 + n1, cs[n1] + cs[n1]]
        if kind == 'sub':
            epochs.equalize_event_counts(old_ids)
        new_id = {n1 + n1: epochs.event_id[n1 + n1] + 1}
        mne.epochs.combine_event_ids(epochs, old_ids, new_id, copy=False)
        # LC+RC=LC
        old_ids = [n1 + 'X', cs[n1] + 'X']
        if kind == 'sub':
            epochs.equalize_event_counts(old_ids)
        new_id = {n1 + 'X': epochs.event_id[n1 + 'X'] + 1}
        mne.epochs.combine_event_ids(epochs, old_ids, new_id, copy=False)

    mne.epochs.combine_event_ids(epochs, ['VM_'], dict(VM=96), copy=False)
    assert 'Ju' in epochs.event_id
    epochs.drop(np.where(epochs.events[:, 2] ==
                         epochs.event_id['Ju'])[0])
    mne.epochs.combine_event_ids(epochs, ['VS_', 'Ju'], dict(VS=97),
                                 copy=False)

    # at this point we only care about:
    eq_names = ('LX', 'UX', 'LL', 'LR', 'UU', 'UD', 'VS')
    assert set(eq_names + ('VM',)) == set(epochs.event_id.keys())
    assert set(eq_names + ('VM',)) == set(out_names)
    orig_len = len(epochs['LL'])
    epochs.equalize_event_counts(eq_names)
    new_len = len(epochs['LL'])
    print(' (reduced LL %s -> %s)' % (orig_len, new_len))
    for ni, out_name in enumerate(out_names):
        idx = (epochs.events[:, 2] == epochs.event_id[out_name])
        epochs.event_id[out_name] = ni + 1
        epochs.events[idx, 2] = ni + 1
    return epochs

for subject in subjects:
    
    subject_dir = f'{data_path}/{subject}'
    
    epochs_fname = f'{subject_dir}/epochs/All_55-sss_{subject}-epo.fif'
    epochs = mne.read_epochs(epochs_fname)
    epochs = eq_trials(epochs, kind='sub')
    epochs = epochs['LL']
    
    fwd_fname = f'{subject_dir}/forward/{subject}-sss-fwd.fif'
    fwd = mne.read_forward_solution(fwd_fname)
    
    cov_fname = f'{subject_dir}/covariance/{subject}-55-sss-cov.fif'
    cov = mne.read_cov(cov_fname)
    
    if subject == subjects[0]:
        num_rois = len(labels)
        timepts = len(epochs.times)
        model = LDS(num_rois, timepts, lam0=0, lam1=100)

    model.add_subject(subject, subject_dir, epochs, labels, fwd, cov) #not using subject_dir


# model.fit(niter=100, verbose=1)
# A_t_ = model.A_t_
# assert A_t_.shape == (timepts, num_rois, num_rois)

# with mpl.rc_context():
#     {'xtick.labelsize': 'x-small', 'ytick.labelsize': 'x-small'}
#     fig, ax = plt.subplots(num_rois, num_rois, constrained_layout=True, squeeze=False,
#                        figsize=(12, 10))
#     plot_A_t_(A_t_, labels=label_names, times=epochs.times, ax=ax)
#     fig.suptitle(condition)















