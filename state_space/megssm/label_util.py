from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import mne
import numpy as np
import os

from megssm.mne_util import combine_medial_labels

subjects_dir = mne.utils.get_subjects_dir()
rtpj_modes = ('hcp', 'labsn', 'intersect')

label_shortnames = {'Early Auditory Cortex-lh': 'AUD-lh',
                    'Early Auditory Cortex-rh': 'AUD-rh',
                    'Premotor Cortex-lh': 'FEF-lh',
                    'Premotor Cortex-rh': 'FEF-rh',
                    'lh.IPS-labsn-lh': 'IPS-lh',
                    'rh.IPS-labsn-rh': 'IPS-rh',
                    'lh.LIPSP-lh': 'LIPSP',
                    'rh.RTPJ-rh': 'RTPJ',
                    'rh.RTPJIntersect-rh-rh': 'RTPJ-intersect',
		    'Primary Visual Cortex (V1)-lh + Primary Visual Cortex (V1)-rh + Early Visual Cortex-lh + Early Visual Cortex-rh': 'Vis',
                    'Anterior Cingulate and Medial Prefrontal Cortex-lh + Anterior Cingulate and Medial Prefrontal Cortex-rh': 'ACC',
                    'DorsoLateral Prefrontal Cortex-lh': 'DLPFC-lh',
                    'DorsoLateral Prefrontal Cortex-rh': 'DLPFC-rh',
                    'Temporo-Parieto-Occipital Junction-lh': 'TPOJ-lh',
                    'Temporo-Parieto-Occipital Junction-rh': 'TPOJ-rh'
		   }


def _sps_meglds_base():
    hcp_mmp1_labels = mne.read_labels_from_annot('fsaverage',
                                                 parc='HCPMMP1_combined')
    hcp_mmp1_labels = combine_medial_labels(hcp_mmp1_labels)
    label_names = [l.name for l in hcp_mmp1_labels]

    ips_str = os.path.join(subjects_dir, "fsaverage/label/*.IPS-labsn.label")
    ips_fnames = glob.glob(ips_str)
    assert len(ips_fnames) == 2, ips_fnames
    ips_labels = [mne.read_label(fn, subject='fsaverage') for fn in ips_fnames]

    pmc_labs = [l for l in hcp_mmp1_labels if 'Premotor Cortex' in l.name]
    eac_labs = [l for l in hcp_mmp1_labels if 'Early Auditory Cortex' in l.name]

    labels = list()
    labels.extend(pmc_labs)
    labels.extend(eac_labs)
    labels.extend(ips_labels)

    rtpj_str = os.path.join(subjects_dir, 'fsaverage/label/rh.RTPJ.label')
    rtpj = mne.read_label(rtpj_str, subject='fsaverage')
    labels.append(rtpj)

    lipsp_str = os.path.join(subjects_dir, 'fsaverage/label/lh.LIPSP.label')
    lipsp = mne.read_label(lipsp_str, subject='fsaverage')
    labels.append(lipsp)

    return sorted(labels, key=lambda x: x.name), hcp_mmp1_labels


def sps_meglds_base():
    return _sps_meglds_base()[0]


def _sps_meglds_base_vision():

    labels, hcp_mmp1_labels = _sps_meglds_base()

    prim_visual = [l for l in hcp_mmp1_labels if 'Primary Visual Cortex' in l.name]

    # there should be only one b/c of medial merge
    prim_visual = prim_visual[0]

    label_names = [l.name for l in hcp_mmp1_labels]
    early_visual_lh = label_names.index("Early Visual Cortex-lh")
    early_visual_rh = label_names.index("Early Visual Cortex-rh")
    early_visual_lh = hcp_mmp1_labels[early_visual_lh]
    early_visual_rh = hcp_mmp1_labels[early_visual_rh]
    visual = prim_visual + early_visual_lh + early_visual_rh

    labels.append(visual)

    return sorted(labels, key=lambda x: x.name), hcp_mmp1_labels


def sps_meglds_base_vision():
    return _sps_meglds_base_vision()[0]


def sps_meglds_base_vision_extra():

    labels, hcp_mmp1_labels = _sps_meglds_base_vision()

    # glasser 19
    ac_mpc_labs = [l for l in hcp_mmp1_labels if 'Anterior Cingulate and Medial Prefrontal Cortex' in l.name]
    labels.extend(ac_mpc_labs)

    # glasser 22
    dpc_labs = [l for l in hcp_mmp1_labels if 'DorsoLateral Prefrontal Cortex' in l.name]
    labels.extend(dpc_labs)

    return sorted(labels, key=lambda x: x.name)


def sps_meglds_base_extra():
    labels, hcp_mmp1_labels = _sps_meglds_base()

    # glasser 19
    ac_mpc_labs = [l for l in hcp_mmp1_labels if 'Anterior Cingulate and Medial Prefrontal Cortex' in l.name]
    labels.extend(ac_mpc_labs)

    # glasser 22
    dpc_labs = [l for l in hcp_mmp1_labels if 'DorsoLateral Prefrontal Cortex' in l.name]
    labels.extend(dpc_labs)


    return sorted(labels, key=lambda x: x.name)



def load_labsn_7_labels():
    label_str = os.path.join(subjects_dir, "fsaverage/label/*labsn*")
    rtpj_str = os.path.join(subjects_dir, 'fsaverage/label/rh.RTPJ.label')
    label_fnames = glob.glob(label_str)
    assert len(label_fnames) == 6
    label_fnames.insert(0, rtpj_str)
    labels = [mne.read_label(fn, subject='fsaverage') for fn in label_fnames]
    labels = sorted(labels, key=lambda x: x.name)

    return labels


def load_hcpmmp1_combined():

    labels = mne.read_labels_from_annot('fsaverage', parc='HCPMMP1_combined')
    labels = sorted(labels, key=lambda x: x.name)
    labels = combine_medial_labels(labels)

    return labels


def load_labsn_hcpmmp1_7_labels(include_visual=False, rtpj_mode='intersect'):

    if rtpj_mode not in rtpj_modes:
        raise ValueError("rtpj must be one of", rtpj_modes)

    hcp_mmp1_labels = mne.read_labels_from_annot('fsaverage',
                                                 parc='HCPMMP1_combined')
    hcp_mmp1_labels = combine_medial_labels(hcp_mmp1_labels)
    label_names = [l.name for l in hcp_mmp1_labels]

    ips_str = os.path.join(subjects_dir, "fsaverage/label/*.IPS-labsn.label")
    ips_fnames = glob.glob(ips_str)
    assert len(ips_fnames) == 2
    ips_labels = [mne.read_label(fn, subject='fsaverage') for fn in ips_fnames]

    pmc_labs = [l for l in hcp_mmp1_labels if 'Premotor Cortex' in l.name]
    eac_labs = [l for l in hcp_mmp1_labels if 'Early Auditory Cortex' in l.name]

    labels = list()
    labels.extend(pmc_labs)
    labels.extend(eac_labs)
    labels.extend(ips_labels)

    # this is in place of original rtpj
    #ipc_labs = [l for l in hcp_mmp1_labels if 'Inferior Parietal Cortex' in l.name]
    if rtpj_mode == 'hcp':
        rtpj = [l for l in hcp_mmp1_labels
                if 'Inferior Parietal Cortex' in l.name and l.hemi == 'rh']
        rtpj = rtpj[0]
    elif rtpj_mode == 'labsn':
        #rtpj_str = os.path.join(subjects_dir, 'fsaverage/label/rh.RTPJAnatomical-rh.label')
        rtpj_str = os.path.join(subjects_dir, 'fsaverage/label/rh.RTPJ.label')
        rtpj = mne.read_label(rtpj_str, subject='fsaverage')
        #tmp = [l for l in ipc_labs if l.hemi == 'lh'] + [rtpj]
        #ipc_labs = tmp
    elif rtpj_mode == 'intersect':
        rtpj_str = os.path.join(subjects_dir, 'fsaverage/label/rh.RTPJIntersect-rh.label')
        rtpj = mne.read_label(rtpj_str, subject='fsaverage')

        #tmp = [l for l in ipc_labs if l.hemi == 'lh'] + [rtpj_hcp]
        #ipc_labs = tmp

    labels.append(rtpj)

    #labels.extend(ipc_labs)

    # optionally include early visual regions as controls
    if include_visual:
        prim_visual = [l for l in hcp_mmp1_labels if 'Primary Visual Cortex' in l.name]

        # there should be only one b/c of medial merge
        prim_visual = prim_visual[0]

        early_visual_lh = label_names.index("Early Visual Cortex-lh")
        early_visual_rh = label_names.index("Early Visual Cortex-rh")
        early_visual_lh = hcp_mmp1_labels[early_visual_lh]
        early_visual_rh = hcp_mmp1_labels[early_visual_rh]
        visual = prim_visual + early_visual_lh + early_visual_rh

        labels.append(visual)

    return labels


def load_labsn_hcpmmp1_7_rtpj_hcp_plus_vision_labels():
    return load_labsn_hcpmmp1_7_labels(include_visual=True, rtpj_mode='hcp')


def load_labsn_hcpmmp1_7_rtpj_intersect_plus_vision_labels():
    return load_labsn_hcpmmp1_7_labels(include_visual=True, rtpj_mode='intersect')


def load_labsn_hcpmmp1_7_rtpj_sphere_plus_vision_labels():
    return load_labsn_hcpmmp1_7_labels(include_visual=True, rtpj_mode='labsn')


def load_labsn_hcpmmp1_av_rois_small():

    hcp_mmp1_labels = mne.read_labels_from_annot('fsaverage',
                                                 parc='HCPMMP1_combined')
    hcp_mmp1_labels = combine_medial_labels(hcp_mmp1_labels)
    label_names = [l.name for l in hcp_mmp1_labels]

    #prim_visual_lh = label_names.index("Primary Visual Cortex (V1)-lh")
    #prim_visual_rh = label_names.index("Primary Visual Cortex (V1)-rh")
    #prim_visual_lh = hcp_mmp1_labels[prim_visual_lh]
    #prim_visual_rh = hcp_mmp1_labels[prim_visual_rh]
    prim_visual = [l for l in hcp_mmp1_labels if 'Primary Visual Cortex' in l.name]

    # there should be only one b/c of medial merge
    prim_visual = prim_visual[0]

    early_visual_lh = label_names.index("Early Visual Cortex-lh")
    early_visual_rh = label_names.index("Early Visual Cortex-rh")
    early_visual_lh = hcp_mmp1_labels[early_visual_lh]
    early_visual_rh = hcp_mmp1_labels[early_visual_rh]

    #visual_lh = prim_visual_lh + early_visual_lh
    #visual_rh = prim_visual_rh + early_visual_rh

    visual = prim_visual + early_visual_lh + early_visual_rh
    labels = [visual]

    #labels = [visual_lh, visual_rh]

    eac_labs = [l for l in hcp_mmp1_labels if 'Early Auditory Cortex' in l.name]
    labels.extend(eac_labs)

    tpo_labs = [l for l in hcp_mmp1_labels if 'Temporo-Parieto-Occipital Junction' in l.name]
    labels.extend(tpo_labs)

    dpc_labs = [l for l in hcp_mmp1_labels if 'DorsoLateral Prefrontal Cortex' in l.name]
    labels.extend(dpc_labs)

    ## extra labels KC wanted
    #pmc_labs = [l for l in hcp_mmp1_labels if 'Premotor Cortex' in l.name]
    #labels.extend(pmc_labs)

    #ips_str = glob.glob(os.path.join(subjects_dir, "fsaverage/label/*IPS*labsn*"))
    #ips_labs = [mne.read_label(fn, subject='fsaverage') for fn in ips_str]
    #labels.extend(ips_labs)

    #rtpj_labs = [l for l in hcp_mmp1_labels if 'Inferior Parietal Cortex-rh' in l.name]
    #labels.extend(rtpj_labs)

    return labels


def load_labsn_hcpmmp1_av_rois_large():

    hcp_mmp1_labels = mne.read_labels_from_annot('fsaverage',
                                                 parc='HCPMMP1_combined')
    hcp_mmp1_labels = combine_medial_labels(hcp_mmp1_labels)
    label_names = [l.name for l in hcp_mmp1_labels]

    #prim_visual_lh = label_names.index("Primary Visual Cortex (V1)-lh")
    #prim_visual_rh = label_names.index("Primary Visual Cortex (V1)-rh")
    #prim_visual_lh = hcp_mmp1_labels[prim_visual_lh]
    #prim_visual_rh = hcp_mmp1_labels[prim_visual_rh]
    prim_visual = [l for l in hcp_mmp1_labels if 'Primary Visual Cortex' in l.name]

    # there should be only one b/c of medial merge
    prim_visual = prim_visual[0]

    early_visual_lh = label_names.index("Early Visual Cortex-lh")
    early_visual_rh = label_names.index("Early Visual Cortex-rh")
    early_visual_lh = hcp_mmp1_labels[early_visual_lh]
    early_visual_rh = hcp_mmp1_labels[early_visual_rh]

    #visual_lh = prim_visual_lh + early_visual_lh
    #visual_rh = prim_visual_rh + early_visual_rh

    visual = prim_visual + early_visual_lh + early_visual_rh
    labels = [visual]

    #labels = [visual_lh, visual_rh]

    eac_labs = [l for l in hcp_mmp1_labels if 'Early Auditory Cortex' in l.name]
    labels.extend(eac_labs)

    tpo_labs = [l for l in hcp_mmp1_labels if 'Temporo-Parieto-Occipital Junction' in l.name]
    labels.extend(tpo_labs)

    dpc_labs = [l for l in hcp_mmp1_labels if 'DorsoLateral Prefrontal Cortex' in l.name]
    labels.extend(dpc_labs)

    # extra labels KC wanted
    pmc_labs = [l for l in hcp_mmp1_labels if 'Premotor Cortex' in l.name]
    labels.extend(pmc_labs)

    #ips_str = glob.glob(os.path.join(subjects_dir, "fsaverage/label/*IPS*labsn*"))
    #ips_labs = [mne.read_label(fn, subject='fsaverage') for fn in ips_str]
    #labels.extend(ips_labs)

    #rtpj_labs = [l for l in hcp_mmp1_labels if 'Inferior Parietal Cortex-rh' in l.name]
    #labels.extend(rtpj_labs)

    # glasser 19
    ac_mpc_labs = [l for l in hcp_mmp1_labels if 'Anterior Cingulate and Medial Prefrontal Cortex' in l.name]
    labels.extend(ac_mpc_labs)

    return labels


def load_labsn_hcpmmp1_av_rois_large_plus_IPS():

    hcp_mmp1_labels = mne.read_labels_from_annot('fsaverage',
                                                 parc='HCPMMP1_combined')
    hcp_mmp1_labels = combine_medial_labels(hcp_mmp1_labels)
    label_names = [l.name for l in hcp_mmp1_labels]

    #prim_visual_lh = label_names.index("Primary Visual Cortex (V1)-lh")
    #prim_visual_rh = label_names.index("Primary Visual Cortex (V1)-rh")
    #prim_visual_lh = hcp_mmp1_labels[prim_visual_lh]
    #prim_visual_rh = hcp_mmp1_labels[prim_visual_rh]
    prim_visual = [l for l in hcp_mmp1_labels if 'Primary Visual Cortex' in l.name]

    # there should be only one b/c of medial merge
    prim_visual = prim_visual[0]

    early_visual_lh = label_names.index("Early Visual Cortex-lh")
    early_visual_rh = label_names.index("Early Visual Cortex-rh")
    early_visual_lh = hcp_mmp1_labels[early_visual_lh]
    early_visual_rh = hcp_mmp1_labels[early_visual_rh]

    #visual_lh = prim_visual_lh + early_visual_lh
    #visual_rh = prim_visual_rh + early_visual_rh

    visual = prim_visual + early_visual_lh + early_visual_rh
    labels = [visual]

    #labels = [visual_lh, visual_rh]

    eac_labs = [l for l in hcp_mmp1_labels if 'Early Auditory Cortex' in l.name]
    labels.extend(eac_labs)

    tpo_labs = [l for l in hcp_mmp1_labels if 'Temporo-Parieto-Occipital Junction' in l.name]
    labels.extend(tpo_labs)

    dpc_labs = [l for l in hcp_mmp1_labels if 'DorsoLateral Prefrontal Cortex' in l.name]
    labels.extend(dpc_labs)

    # extra labels KC wanted
    pmc_labs = [l for l in hcp_mmp1_labels if 'Premotor Cortex' in l.name]
    labels.extend(pmc_labs)

    ips_str = os.path.join(subjects_dir, "fsaverage/label/*.IPS-labsn.label")
    ips_fnames = glob.glob(ips_str)
    ips_labels = [mne.read_label(fn, subject='fsaverage') for fn in ips_fnames]
    labels.extend(ips_labels)

    #rtpj_labs = [l for l in hcp_mmp1_labels if 'Inferior Parietal Cortex-rh' in l.name]
    #labels.extend(rtpj_labs)

    # glasser 19
    ac_mpc_labs = [l for l in hcp_mmp1_labels if 'Anterior Cingulate and Medial Prefrontal Cortex' in l.name]
    labels.extend(ac_mpc_labs)

    return labels


def make_rtpj_intersect():
    labels = mne.read_labels_from_annot('fsaverage', 'HCPMMP1', 'rh',
                                        subjects_dir=subjects_dir)

    rtpj_str = os.path.join(subjects_dir, 'fsaverage/label/rh.RTPJAnatomical-rh.label')
    rtpj = mne.read_label(rtpj_str, subject='fsaverage')
    src = mne.read_source_spaces(subjects_dir + '/fsaverage/bem/fsaverage-5-src.fif')
    rtpj = rtpj.fill(src)

    mne.write_label(os.path.join(subjects_dir,
                                 'fsaverage/label/rh.RTPJ.label'),
                    rtpj)

    props = np.zeros((len(labels), 2))
    for li, label in enumerate(labels):
        props[li] = [np.in1d(rtpj.vertices, label.vertices).mean(),
                     np.in1d(label.vertices, rtpj.vertices).mean()]
    order = np.argsort(props[:, 0])[::-1]
    for oi in order:
        if props[oi, 0] > 0:
            name = labels[oi].name.rstrip('-rh').lstrip('R_')
            print('%4.1f%% RTPJ vertices cover %4.1f%% of %s'
                  % (100*props[oi,0], 100*props[oi,1], name))

    for ii, oi in enumerate(order[:4]):
        if ii == 0:
            rtpj = labels[oi].copy()
        else:
            rtpj += labels[oi]

    mne.write_label(os.path.join(subjects_dir,
                                 'fsaverage/label/rh.RTPJIntersect-rh.label'),
                    rtpj)


def fixup_lipsp():
    labels = mne.read_labels_from_annot('fsaverage', 'HCPMMP1', 'rh',
                                        subjects_dir=subjects_dir)

    lipsp_str = os.path.join(subjects_dir, 'fsaverage/label/lh.LIPSP_tf.label')
    lipsp = mne.read_label(lipsp_str, subject='fsaverage')
    lipsp.vertices = lipsp.vertices[lipsp.vertices < 10242]

    src = mne.read_source_spaces(subjects_dir + '/fsaverage/bem/fsaverage-5-src.fif')
    lipsp = lipsp.fill(src)


    mne.write_label(os.path.join(subjects_dir, 'fsaverage/label/lh.LIPSP.label'),
                    lipsp)

    return lipsp


#if __name__ == "__main__":
#
#    from surfer import Brain
#    labels = sps_meglds_base()
#
#    subject_id = 'fsaverage'
#    hemi = 'both'
#    surf = 'inflated'
#
#    brain = Brain(subject_id, hemi, surf)
#    for l in labels:
#        brain.add_label(l)
