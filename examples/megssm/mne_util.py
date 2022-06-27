""" MNE-Python utility functions for preprocessing data and constructing
    matrices necessary for MEGLDS analysis """

import mne
import numpy as np
import os.path as op

from mne.io.pick import pick_types
from mne.utils import logger
from mne import label_sign_flip

from scipy.sparse import csc_matrix, csr_matrix, diags


class ROIToSourceMap(object):
    """ class for computing ROI-to-source space mapping matrix 
    
    Notes
    -----
    The following variables defined here correspond to various matrices
    defined in :footcite:`yang_state-space_2016`:
    - fwd_src_snsr : G
    - fwd_roi_snsr : C
    - fwd_src_roi : L
    - snsr_cov : Q_e
    - roi_cov : Q
    - roi_cov_0 : Q0         """

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
        which_roi = np.zeros(n_verts, dtype=np.int64)

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
                which_roi[lverts] = li + 1

        N = len(labels)
        M = n_verts

        # construct sparse fwd_src_roi matrix
        data = np.concatenate(data)
        vertidx = np.array(vertidx, int)
        roiidx = np.array(roiidx, int)
        assert data.shape == vertidx.shape == roiidx.shape
        fwd_src_roi = csc_matrix((data, (vertidx, roiidx)), shape=(M, N))

        self.fwd = fwd
        self.fwd_src_roi = fwd_src_roi
        self.which_roi = which_roi
        self.offsets = offsets
        self.n_lhverts = n_lhverts
        self.n_rhverts = n_rhverts
        self.labels = labels

        return

    @property
    def fwd_src_sn(self):
        return self.fwd['sol']['data']

    @property
    def fwd_src_roi(self):
        return self._fwd_src_roi

    @fwd_src_roi.setter
    def fwd_src_roi(self, val):
        self._fwd_src_roi = val

    @property
    def which_roi(self):
        return self._which_roi

    @which_roi.setter
    def which_roi(self, val):
        self._which_roi = val

    @property
    def fwd_roi_snsr(self):
        from util import Carray
        return Carray(csr_matrix.dot(self.fwd_src_roi.T, self.fwd_src_sn.T).T)

    def get_label_vinds(self, label):
        li = self.labels.index(label)
        if isinstance(label, mne.Label):
            label_vert_idx = self.fwd_src_roi[:, li].nonzero()[0]
            label_vert_idx -= self.offsets[label.hemi]
            return label_vert_idx
        elif isinstance(label, mne.BiHemiLabel):
            # these labels store both hemispheres so subtract the rh offset
            # from that part of the vertex array
            lh_label_vert_idx = self.fwd_src_roi[:self.n_lhverts, li].nonzero()[0]
            rh_label_vert_idx = self.fwd_src_roi[self.n_lhverts:, li].nonzero()[0]
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

def apply_projs(epochs, fwd, cov):
    """ apply projection operators to fwd and cov """
    proj, _ = mne.io.proj.setup_proj(epochs.info, activate=False)
    fwd_src_sn = fwd['sol']['data']
    fwd['sol']['data'] = np.dot(proj, fwd_src_sn)

    roi_cov = cov.data
    if not np.allclose(np.dot(proj, roi_cov), roi_cov):
        roi_cov = np.dot(proj, np.dot(roi_cov, proj.T))
        cov.data = roi_cov

    return fwd, cov


def _scale_sensor_data(epochs, fwd, cov, roi_to_src, eeg_scale=1., mag_scale=1.,
    grad_scale=1.):
    """ apply per-channel-type scaling to epochs, forward, and covariance """
    # from util import Carray ##skip import just pasted; util also from MEGLDS repo
    Carray64 = lambda X: np.require(X, dtype=np.float64, requirements='C')
    Carray32 = lambda X: np.require(X, dtype=np.float32, requirements='C')
    Carray = Carray64

    # get indices for each channel type
    ch_names = cov['names'] # same as self.fwd['info']['ch_names']
    sel_eeg = pick_types(fwd['info'], meg=False, eeg=True, ref_meg=False)
    sel_mag = pick_types(fwd['info'], meg='mag', eeg=False, ref_meg=False)
    sel_grad = pick_types(fwd['info'], meg='grad', eeg=False, ref_meg=False)
    idx_eeg = [ch_names.index(ch_names[c]) for c in sel_eeg]
    idx_mag = [ch_names.index(ch_names[c]) for c in sel_mag]
    idx_grad = [ch_names.index(ch_names[c]) for c in sel_grad]

    # retrieve forward and sensor covariance
    fwd_src_snsr = fwd['sol']['data'].copy()
    snsr_cov = cov.data.copy()

    # scale forward matrix
    fwd_src_snsr[idx_eeg,:] *= eeg_scale
    fwd_src_snsr[idx_mag,:] *= mag_scale
    fwd_src_snsr[idx_grad,:] *= grad_scale

    # construct fwd_roi_snsr matrix
    fwd_roi_snsr = Carray(csr_matrix.dot(roi_to_src.fwd_src_roi.T, fwd_src_snsr.T).T)

    # scale sensor covariance
    snsr_cov[np.ix_(idx_eeg, idx_eeg)] *= eeg_scale**2
    snsr_cov[np.ix_(idx_mag, idx_mag)] *= mag_scale**2
    snsr_cov[np.ix_(idx_grad, idx_grad)] *= grad_scale**2

    # scale epochs
    info = epochs.info.copy()
    data = epochs.get_data().copy()

    data[:,idx_eeg,:] *= eeg_scale
    data[:,idx_mag,:] *= mag_scale
    data[:,idx_grad,:] *= grad_scale

    epochs = mne.EpochsArray(data, info)

    return fwd_src_snsr, fwd_roi_snsr, snsr_cov, epochs


