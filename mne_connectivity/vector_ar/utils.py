import numpy as np


def _block_companion(mats):
    """Form a block companion matrix."""
    eye_n = np.eye(np.sum([x.shape[1] for x in mats[:-1]]))
    return np.block([[*mats],
                     [eye_n, np.zeros([eye_n.shape[0], mats[-1].shape[1]])]])
