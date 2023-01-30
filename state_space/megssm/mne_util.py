""" MNE-Python utility functions for preprocessing data and constructing
    matrices necessary for MEGLDS analysis """

import mne
import numpy as np
from mne import label_sign_flip
from scipy.sparse import csc_matrix, csr_matrix
from sklearn.decomposition import PCA

Carray = lambda X: np.require(X, dtype=np.float64, requirements='C')

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

                # gets the indices in the source space vertex array, not the
                # huge array.
                # use `src[hi]['vertno'][lverts]` to get surface vertex indices
                # to plot.
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


def _scale_sensor_data(epochs, fwd, cov, roi_to_src, eeg_scale=1.,
                       mag_scale=1., grad_scale=1.):
    """ apply per-channel-type scaling to epochs, forward, and covariance """
    
    # get indices for each channel type
    ch_names = cov['names']
    
    # build scaler
    info = epochs.info.copy()
    std = dict(grad=1. / grad_scale, mag=1. / mag_scale, eeg=1. / eeg_scale)
    noproj_info = info.copy()
    with noproj_info._unlock():
        noproj_info['projs'] = []
    rescale_cov = mne.make_ad_hoc_cov(noproj_info, std=std)
    scaler, ch_names = mne.cov.compute_whitener(rescale_cov, noproj_info)
    np.testing.assert_array_equal(np.diag(np.diag(scaler)), scaler)
    assert ch_names == info['ch_names']
        
    # retrieve forward and sensor covariance
    fwd_src_snsr = fwd['sol']['data'].copy()
    roi_cov = cov.data.copy()

    # scale forward matrix
    fwd_src_snsr = scaler @ fwd_src_snsr

    # construct fwd_roi_snsr matrix
    fwd_roi_snsr = Carray(csr_matrix.dot(roi_to_src.fwd_src_roi.T, fwd_src_snsr.T).T)

    # scale sensor covariance
    roi_cov = scaler.T @ roi_cov @ scaler 

    # scale epochs
    data = epochs.get_data().copy()
    data = scaler.T @ data
    epochs = mne.EpochsArray(data, info)

    return fwd_src_snsr, fwd_roi_snsr, roi_cov, epochs


def run_pca_on_subject(subject_name, epochs, fwd, cov, labels, dim_mode='rank',
                       pctvar=0.99, mean_center=False, label_flip=False):
    """ apply sensor scaling, PCA dimensionality reduction with/without
        whitening, and mean-centering to subject data """

    if dim_mode not in ['rank', 'pctvar', 'whiten']:
        raise ValueError("dim_mode must be in {'rank', 'pctvar', 'whiten'}")

    print("running pca for subject %s" % subject_name)

    scales = {'eeg_scale' : 1e8, 'mag_scale' : 1e16, 'grad_scale' : 1e14}

    # compute ROI-to-source map
    roi_to_src = ROIToSourceMap(fwd, labels, label_flip)

    if dim_mode == 'whiten':

        fwd_src_snsr, fwd_roi_snsr, cov_snsr, epochs = \
            _scale_sensor_data(epochs, fwd, cov, roi_to_src)
        dat = epochs.get_data()
        dat = Carray(np.swapaxes(dat, -1, -2))

        if mean_center:
            dat -= np.mean(dat, axis=1, keepdims=True)

        dat_stacked = np.reshape(dat, (-1, dat.shape[-1]))

        W, _ = mne.cov.compute_whitener(subject.sensor_cov,
                                        info=subject.epochs_list[0].info,
                                        pca=True)
        print("whitener for subject %s using %d principal components" %
              (subject_name, W.shape[0]))

    else:

        fwd_src_snsr, fwd_roi_snsr, cov_snsr, epochs = _scale_sensor_data(
            epochs, fwd, cov, roi_to_src, **scales)

        dat = epochs.get_data().copy()
        dat = Carray(np.swapaxes(dat, -1, -2))

        if mean_center:
            dat -= np.mean(dat, axis=1, keepdims=True)

        dat_stacked = np.reshape(dat, (-1, dat.shape[-1]))
        pca = PCA()
        pca.fit(dat_stacked)

        if dim_mode == 'rank':
            idx = np.linalg.matrix_rank(np.cov(dat_stacked, rowvar=False))
        else:
            idx = np.where(np.cumsum(pca.explained_variance_ratio_) >
                           pctvar)[0][0]

        idx = np.maximum(idx, len(labels))
        W = pca.components_[:idx]
        print("subject %s using %d principal components" % (subject_name, idx))

    ntrials, T, _ = dat.shape
    dat_pca = np.dot(dat_stacked, W.T)
    dat_pca = np.reshape(dat_pca, (ntrials, T, -1))

    fwd_src_snsr_pca = np.dot(W, fwd_src_snsr)
    fwd_roi_snsr_pca = np.dot(W, fwd_roi_snsr)
    cov_snsr_pca = np.dot(W,np.dot(cov_snsr, W.T))

    data = dat_pca

    return (data, fwd_roi_snsr_pca, fwd_src_snsr_pca, cov_snsr_pca,
            roi_to_src.which_roi)
