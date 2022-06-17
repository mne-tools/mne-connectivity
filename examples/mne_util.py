""" MNE-Python utility functions for preprocessing data and constructing
    matrices necessary for MEGLDS analysis """

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import mne
import numpy as np
import os.path as op

from mne.io.pick import pick_types
from mne.utils import logger
from mne import label_sign_flip

from scipy.sparse import csc_matrix, csr_matrix, diags


class ROIToSourceMap(object):
    """ class for computing ROI-to-source space mapping matrix """

    def __init__(self, fwd, labels, label_flip=False):

        src = fwd['src']

        roiidx = list()
        vertidx = list()

        n_lhverts = len(src[0]['vertno'])
        n_rhverts = len(src[1]['vertno'])
        n_verts = n_lhverts + n_rhverts
        offsets = {'lh': 0, 'rh': n_lhverts}

        hemis = {'lh': 0, 'rh': 1}

        # index vector of which ROI a source point belongs to
        Q_J = np.zeros(n_verts, dtype=np.int64)

        data = []
        for li, lab in enumerate(labels):

            this_data = np.round(label_sign_flip(lab, src))
            if not label_flip:
                this_data.fill(1.)
            data.append(this_data)
            if isinstance(lab, mne.Label):
                comp_labs = [lab]
            elif isinstance(lab, mne.BiHemiLabel):
                comp_labs = [lab.lh, lab.rh]

            for clab in comp_labs:
                hemi = clab.hemi
                hi = 0 if hemi == 'lh' else 1

                lverts = clab.get_vertices_used(vertices=src[hi]['vertno'])

                # gets the indices in the source space vertex array, not the huge
                # array.
                # use `src[hi]['vertno'][lverts]` to get surface vertex indices to
                # plot.
                lverts = np.searchsorted(src[hi]['vertno'], lverts)
                lverts += offsets[hemi]
                vertidx.extend(lverts)
                roiidx.extend(np.full(lverts.size, li, dtype=np.int64))

                # add 1 b/c 0 corresponds to unassigned variance
                Q_J[lverts] = li + 1

        N = len(labels)
        M = n_verts

        # construct sparse L matrix
        data = np.concatenate(data)
        vertidx = np.array(vertidx, int)
        roiidx = np.array(roiidx, int)
        assert data.shape == vertidx.shape == roiidx.shape
        L = csc_matrix((data, (vertidx, roiidx)), shape=(M, N))

        self.fwd = fwd
        self.L = L
        self.Q_J = Q_J
        self.offsets = offsets
        self.n_lhverts = n_lhverts
        self.n_rhverts = n_rhverts
        self.labels = labels

        return

    @property
    def G(self):
        return self.fwd['sol']['data']

    @property
    def L(self):
        return self._L

    @L.setter
    def L(self, val):
        self._L = val

    @property
    def Q_J(self):
        return self._Q_J

    @Q_J.setter
    def Q_J(self, val):
        self._Q_J = val

    @property
    def GL(self):
        from util import Carray
        return Carray(csr_matrix.dot(self.L.T, self.G.T).T)

    def get_label_vinds(self, label):
        li = self.labels.index(label)
        if isinstance(label, mne.Label):
            label_vert_idx = self.L[:, li].nonzero()[0]
            label_vert_idx -= self.offsets[label.hemi]
            return label_vert_idx
        elif isinstance(label, mne.BiHemiLabel):
            # these labels store both hemispheres so subtract the rh offset
            # from that part of the vertex array
            lh_label_vert_idx = self.L[:self.n_lhverts, li].nonzero()[0]
            rh_label_vert_idx = self.L[self.n_lhverts:, li].nonzero()[0]
            rh_label_vert_idx[self.n_lhverts:] -= self.offsets['rh']
            return [lh_label_vert_idx, rh_label_vert_idx]

    def get_label_verts(self, label, src):
        # if you're thinking of using this to plot, why not just use
        # brain.add_label from pysurfer?
        if isinstance(label, mne.Label):
            hi = 0 if label.hemi == 'lh' else 1
            label_vert_idx = self.get_label_vinds(label)
            varray = src[hi]['vertno'][label_vert_idx]
        elif isinstance(label, mne.BiHemiLabel):
            lh_label_vert_idx, rh_label_vert_idx = self.get_label_vinds(label)
            varray = [src[0]['vertno'][lh_label_vert_idx],
                      src[1]['vertno'][rh_label_vert_idx]]
        return varray

    def get_hemi_idx(self, label):
        if isinstance(label, mne.Label):
            return 0 if label.hemi == 'lh' else 1
        elif isinstance(label, mne.BiHemiLabel):
            hemis = [None] * 2
            for i, lab in enumerate([label.lh, label.rh]):
                hemis[i] = 0 if lab.hemi == 'lh' else 1
            return hemis


def morph_labels(labels, subject_to, subjects_dir=None):
    """ morph labels from fsaverage to specified subject """

    if subjects_dir is None:
        subjects_dir = mne.utils.get_subjects_dir()

    if isinstance(labels, mne.Label):
        labels = [labels]

    labels_morphed = list()
    for lab in labels:
        if isinstance(lab, mne.Label):
            labels_morphed.append(lab.copy())
        elif isinstance(lab, mne.BiHemiLabel):
            labels_morphed.append(lab.lh.copy() + lab.rh.copy())

    for i, l in enumerate(labels_morphed):
        if l.subject == subject_to:
            continue
        elif l.subject == 'unknown':
            print("uknown subject for label %s" % l.name,
                  "assuming if is 'fsaverage' and morphing")
            l.subject = 'fsaverage'

        if isinstance(l, mne.Label):
            l.values.fill(1.0)
            labels_morphed[i] = l.morph(subject_to=subject_to,
                                        subjects_dir=subjects_dir)
        elif isinstance(l, mne.BiHemiLabel):
            l.lh.values.fill(1.0)
            l.rh.values.fill(1.0)
            labels_morphed[i].lh = l.lh.morph(subject_to=subject_to,
                                              subjects_dir=subjects_dir)
            labels_morphed[i].rh = l.rh.morph(subject_to=subject_to,
                                              subjects_dir=subjects_dir)

    # make sure there are no duplicate labels
    labels_morphed = sorted(list(set(labels_morphed)), key=lambda x: x.name)

    return labels_morphed


def apply_projs(epochs, fwd, cov):
    """ apply projection operators to fwd and cov """
    proj, _ = mne.io.proj.setup_proj(epochs.info, activate=False)
    G = fwd['sol']['data']
    fwd['sol']['data'] = np.dot(proj, G)

    Q = cov.data
    if not np.allclose(np.dot(proj, Q), Q):
        Q = np.dot(proj, np.dot(Q, proj.T))
        cov.data = Q

    return fwd, cov


def scale_sensor_data(epochs, fwd, cov, roi_to_src, eeg_scale=1., mag_scale=1.,
    grad_scale=1.):
    """ apply per-channel-type scaling to epochs, forward, and covariance """
    # from util import Carray ##skip import just pasted; util also from MEGLDS repo
    Carray64 = lambda X: np.require(X, dtype=np.float64, requirements='C')
    Carray32 = lambda X: np.require(X, dtype=np.float32, requirements='C')
    Carray = Carray64

    # get indices for each channel type
    ch_names = cov['names']  # same as self.fwd['info']['ch_names']
    sel_eeg = pick_types(fwd['info'], meg=False, eeg=True, ref_meg=False)
    sel_mag = pick_types(fwd['info'], meg='mag', eeg=False, ref_meg=False)
    sel_grad = pick_types(fwd['info'], meg='grad', eeg=False, ref_meg=False)
    #2 channels are removed so idx != ch_name   
    #can we do idx = c for c in sel??
    #idx_eeg = [ch_names.index(ch_names[c]) for c in sel_eeg]
    #idx_mag = [ch_names.index(ch_names[c]) for c in sel_mag]
    #idx_grad = [ch_names.index(ch_names[c]) for c in sel_grad]
    idx_eeg = [c for c in sel_eeg]
    idx_mag = [c for c in sel_mag]
    idx_grad = [c for c in sel_grad]

    # retrieve forward and sensor covariance
    G = fwd['sol']['data'].copy()
    Q = cov.data.copy()

    # scale forward matrix
    G[idx_eeg,:] *= eeg_scale
    G[idx_mag,:] *= mag_scale
    G[idx_grad,:] *= grad_scale

    # construct GL matrix
    GL = Carray(csr_matrix.dot(roi_to_src.L.T, G.T).T)

    # scale sensor covariance
    Q[np.ix_(idx_eeg, idx_eeg)] *= eeg_scale**2
    Q[np.ix_(idx_mag, idx_mag)] *= mag_scale**2
    Q[np.ix_(idx_grad, idx_grad)] *= grad_scale**2

    # scale epochs
    info = epochs.info.copy()
    data = epochs.get_data().copy()

    data[:,idx_eeg,:] *= eeg_scale
    data[:,idx_mag,:] *= mag_scale
    data[:,idx_grad,:] *= grad_scale

    epochs = mne.EpochsArray(data, info)

    return G, GL, Q, epochs


def combine_medial_labels(labels, subject='fsaverage', surf='white',
                          dist_limit=0.02):
    """ combine each hemi pair of labels on medial wall into single label """
    subjects_dir = mne.get_config('SUBJECTS_DIR')
    rrs = dict((hemi, mne.read_surface(op.join(subjects_dir, subject, 'surf',
                                       '%s.%s' % (hemi, surf)))[0] / 1000.)
               for hemi in ('lh', 'rh'))
    use_labels = list()
    used = np.zeros(len(labels), bool)

    logger.info('Matching medial regions for %s labels on %s %s, d=%0.1f mm'
                % (len(labels), subject, surf, 1000 * dist_limit))

    for li1, l1 in enumerate(labels):
        if used[li1]:
            continue
        used[li1] = True
        use_label = l1.copy()
        rr1 = rrs[l1.hemi][l1.vertices]
        for li2 in np.where(~used)[0]:
            l2 = labels[li2]
            same_name = (l2.name.replace(l2.hemi, '') ==
                         l1.name.replace(l1.hemi, ''))
            if l2.hemi != l1.hemi and same_name:
                rr2 = rrs[l2.hemi][l2.vertices]
                mean_min = np.mean(mne.surface._compute_nearest(
                    rr1, rr2, return_dists=True)[1])
                if mean_min <= dist_limit:
                    use_label += l2
                    used[li2] = True
                    logger.info('  Matched: ' + l1.name)
        use_labels.append(use_label)

    logger.info('Total %d labels' % (len(use_labels),))

    return use_labels
