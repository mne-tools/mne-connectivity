import numpy as np


def _block_companion(mats):
    """Form a block companion matrix."""
    eye_n = np.eye(np.sum([x.shape[1] for x in mats[:-1]]))
    return np.block([[*mats],
                     [eye_n, np.zeros([eye_n.shape[0], mats[-1].shape[1]])]])


if __name__ == '__main__':
    A = np.array([[1.1, 1.2], [2.1, 2.2]])
    B = np.array([[5, 10], [-2, -3]])

    companion = _block_companion([A, B, A])
    print('\n\nGot outside!')
    print(companion)
    print(companion.shape)
