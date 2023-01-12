import numpy as np
import pytest
from numpy.testing import assert_array_equal

from mne_connectivity import Connectivity
from mne_connectivity.utils import (
    degree, multivariate_seed_target_indices, seed_target_indices
)


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

def test_multivariate_indices():
    """Test multivariate connectivity indexing methods."""
    # tests that incorrect input types are caught for seeds
    with pytest.raises(TypeError, match='seeds and targets should'):
        multivariate_seed_target_indices(0, [[1]])
    with pytest.raises(TypeError, match='entries of seeds and targets should'):
        multivariate_seed_target_indices([0], [[1]])

    # tests that incorrect input types are caught for targets
    with pytest.raises(TypeError, match='seeds and targets should'):
        multivariate_seed_target_indices([[0]], 1)
    with pytest.raises(TypeError, match='entries of seeds and targets should'):
        multivariate_seed_target_indices([[0]], [1])

    # tests that outputs are correct
    n_seeds_test = [1, 3, 4]
    n_targets_test = [2, 3, 100]
    max_ch_idx = 100 # could be any value >= 1
    max_n_chs_per_con = 20 # could be any value >= 1
    rng = np.random.RandomState(44)
    for n_seeds in n_seeds_test:
        for n_targets in n_targets_test:
            idx = []
            for _ in range(n_seeds + n_targets):
                idx.append(rng.randint(
                        1, max_ch_idx, (rng.randint(1, max_n_chs_per_con))
                ))
            seeds = idx[:n_seeds]
            targets = idx[n_seeds:]
            indices = multivariate_seed_target_indices(seeds, targets)
            assert len(indices) == 2
            assert len(indices[0]) == len(indices[1])
            assert len(indices[0]) == n_seeds * n_targets
            for seed in seeds:
                n_equal = 0
                for index in indices[0]:
                    if np.array_equal(index, seed):
                        n_equal += 1
                assert n_equal == n_targets
            for target in targets:
                n_equal = 0
                for index in indices[1]:
                    if np.array_equal(index, target):
                        n_equal += 1
                assert n_equal == n_seeds


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

test_indices()
test_multivariate_indices()