import numpy as np
import pytest
from numpy.testing import assert_array_equal

from mne.io import RawArray
from mne.epochs import Epochs, make_fixed_length_epochs
from mne.io.meas_info import create_info

from mne_connectivity import Connectivity
from mne_connectivity.utils import (
    degree, seed_target_indices, map_epoch_annotations_to_epoch)


def test_indices():
    """Test connectivity indexing methods."""
    n_seeds_test = [1, 3, 4]
    n_targets_test = [2, 3, 200]
    rng = np.random.RandomState(42)
    for n_seeds in n_seeds_test:
        for n_targets in n_targets_test:
            idx = rng.permutation(np.arange(n_seeds + n_targets))
            seeds = idx[:n_seeds]
            targets = idx[n_seeds:]
            indices = seed_target_indices(seeds, targets)
            assert len(indices) == 2
            assert len(indices[0]) == len(indices[1])
            assert len(indices[0]) == n_seeds * n_targets
            for seed in seeds:
                assert np.sum(indices[0] == seed) == n_targets
            for target in targets:
                assert np.sum(indices[1] == target) == n_seeds


def test_degree():
    """Test degree function."""
    # degenerate conditions
    with pytest.raises(ValueError, match='threshold'):
        degree(np.eye(3), 2.)
    # a simple one
    corr = np.eye(10)
    assert_array_equal(degree(corr), np.zeros(10))
    # more interesting
    corr = np.array([[0.5, 0.7, 0.4],
                     [0.1, 0.3, 0.6],
                     [0.2, 0.8, 0.9]])
    deg = degree(corr, 1)
    assert_array_equal(deg, [2, 2, 2])

    # The values for assert_array_equal below were obtained with:
    #
    # >>> import bct
    # >>> bct.degrees_und(bct.utils.threshold_proportional(corr, 0.25) > 0)
    #
    # But they can also be figured out just from the structure.

    # Asymmetric (6 usable nodes)
    assert_array_equal(degree(corr, 0.33), [0, 2, 0])
    assert_array_equal(degree(corr, 0.5), [0, 2, 1])
    # Symmetric (3 usable nodes)
    corr = (corr + corr.T) / 2.
    assert_array_equal(degree(corr, 0.33), [0, 1, 1])
    assert_array_equal(degree(corr, 0.66), [1, 2, 1])

    # check error when connectivity array is > 2D
    with pytest.raises(ValueError, match='connectivity must be have shape'):
        degree(np.zeros((5, 5, 5)))

    # call degree using a connectivity object
    conn = Connectivity(data=np.zeros((4,)), n_nodes=2)
    deg = degree(conn)
    assert_array_equal(deg, [0, 0])


def test_mapping_epochs_to_epochs():
    """Test map_epoch_annotations_to_epoch function."""
    n_times = 1000
    sfreq = 100
    data = np.random.random((2, n_times))
    info = create_info(ch_names=['A1', 'A2'], sfreq=sfreq,
                       ch_types='mag')
    raw = RawArray(data, info)

    # create two different sets of Epochs
    # the first one is just a contiguous chunks of 1 seconds
    epoch_one = make_fixed_length_epochs(raw, duration=1, overlap=0)

    events = np.zeros((2, 3), dtype=int)
    events[:, 0] = [100, 900]
    epoch_two = Epochs(raw, events, tmin=-0.5, tmax=0.5)

    # map Epochs from two to one
    all_cases = map_epoch_annotations_to_epoch(epoch_one, epoch_two)
    assert all_cases.shape == (2, 10)

    # only 1-3 Epochs of epoch_one should overlap with the epoch_two's
    # 1st Epoch
    assert all(all_cases[0, :2])
    assert all(all_cases[1, -2:])

    # map Epochs from one to two
    all_cases = map_epoch_annotations_to_epoch(epoch_two, epoch_one)
    assert all_cases.shape == (10, 2)

    assert all(all_cases[:2, 0])
    assert all(all_cases[-2:, 1])
