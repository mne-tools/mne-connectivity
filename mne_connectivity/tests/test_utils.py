import numpy as np
import pytest
from numpy.testing import assert_array_equal

from mne_connectivity import Connectivity
from mne_connectivity.utils import (
    _check_multivariate_indices,
    check_indices,
    degree,
    seed_target_indices,
    seed_target_multivariate_indices,
)


def test_seed_target_indices():
    """Test indices generation functions."""
    # bivariate indices
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

    # multivariate indices
    # non-ragged indices
    seeds = [[0, 1]]
    targets = [[2, 3], [3, 4]]
    indices = seed_target_multivariate_indices(seeds, targets)
    match_indices = (
        np.array([[0, 1], [0, 1]], dtype=object),
        np.array([[2, 3], [3, 4]], dtype=object),
    )
    for type_i in range(2):
        for con_i in range(len(indices[0])):
            assert np.all(indices[type_i][con_i] == match_indices[type_i][con_i])
    # ragged indices
    seeds = [[0, 1]]
    targets = [[2, 3, 4], [4]]
    indices = seed_target_multivariate_indices(seeds, targets)
    match_indices = (
        np.array([[0, 1], [0, 1]], dtype=object),
        np.array([[2, 3, 4], [4]], dtype=object),
    )
    for type_i in range(2):
        for con_i in range(len(indices[0])):
            assert np.all(indices[type_i][con_i] == match_indices[type_i][con_i])
    # test error catching
    # non-array-like seeds/targets
    with pytest.raises(TypeError, match="`seeds` and `targets` must be array-like"):
        seed_target_multivariate_indices(0, 1)
    # non-nested seeds/targets
    with pytest.raises(TypeError, match="`seeds` and `targets` must contain nested"):
        seed_target_multivariate_indices([0], [1])
    # repeated seeds/targets
    with pytest.raises(
        ValueError, match="`seeds` and `targets` cannot contain repeated"
    ):
        seed_target_multivariate_indices([[0, 1, 1]], [[2, 2, 3]])


def test_check_indices():
    """Test check_indices function."""
    # bivariate indices
    # test error catching
    with pytest.raises(ValueError, match="indices must be a tuple of length 2"):
        non_tuple_indices = [[0], [1]]
        check_indices(non_tuple_indices)
    with pytest.raises(ValueError, match="indices must be a tuple of length 2"):
        non_len2_indices = ([0], [1], [2])
        check_indices(non_len2_indices)
    with pytest.raises(ValueError, match="Index arrays indices"):
        non_equal_len_indices = ([0], [1, 2])
        check_indices(non_equal_len_indices)
    with pytest.raises(TypeError, match="Channel indices must be integers, not array"):
        nested_indices = ([[0]], [[1]])
        check_indices(nested_indices)


def test_check_multivariate_indices():
    """Test _check_multivariate_indices function."""
    n_signals = 5
    mask_value = -1
    # non-ragged indices
    seeds = [[0, 1], [0, 1]]
    targets = [[2, 3], [3, 4]]
    indices = _check_multivariate_indices((seeds, targets), n_signals)
    assert np.ma.isMA(indices)
    assert indices.fill_value == mask_value
    assert np.all(indices == np.array([[[0, 1], [0, 1]], [[2, 3], [3, 4]]]))
    # non-ragged indices with negative values
    seeds = [[0, 1], [0, 1]]
    targets = [[2, 3], [3, -1]]
    indices = _check_multivariate_indices((seeds, targets), n_signals)
    assert np.ma.isMA(indices)
    assert indices.fill_value == mask_value
    assert np.all(indices == np.array([[[0, 1], [0, 1]], [[2, 3], [3, 4]]]))
    # ragged indices
    seeds = [[0, 1], [0, 1]]
    targets = [[2, 3, 4], [4]]
    indices = _check_multivariate_indices((seeds, targets), n_signals)
    assert np.ma.isMA(indices)
    assert indices.fill_value == mask_value
    assert np.all(
        indices == np.array([[[0, 1, -1], [0, 1, -1]], [[2, 3, 4], [4, -1, -1]]])
    )
    # ragged indices with negative values
    seeds = [[0, 1], [0, 1]]
    targets = [[2, 3, 4], [-1]]
    indices = _check_multivariate_indices((seeds, targets), n_signals)
    assert np.ma.isMA(indices)
    assert indices.fill_value == mask_value
    assert np.all(
        indices == np.array([[[0, 1, -1], [0, 1, -1]], [[2, 3, 4], [4, -1, -1]]])
    )

    # test error catching
    with pytest.raises(ValueError, match="indices must be a tuple of length 2"):
        non_tuple_indices = [np.array([0, 1]), np.array([2, 3])]
        _check_multivariate_indices(non_tuple_indices, n_signals)
    with pytest.raises(ValueError, match="indices must be a tuple of length 2"):
        non_len2_indices = (np.array([0]), np.array([1]), np.array([2]))
        _check_multivariate_indices(non_len2_indices, n_signals)
    with pytest.raises(ValueError, match="index arrays indices"):
        non_equal_len_indices = (np.array([[0]]), np.array([[1], [2]]))
        _check_multivariate_indices(non_equal_len_indices, n_signals)
    with pytest.raises(
        TypeError, match="multivariate indices must contain array-likes"
    ):
        non_nested_indices = (np.array([0, 1]), np.array([2, 3]))
        _check_multivariate_indices(non_nested_indices, n_signals)
    with pytest.raises(
        ValueError, match="multivariate indices cannot contain repeated"
    ):
        repeated_indices = (np.array([[0, 1, 1]]), np.array([[2, 2, 3]]))
        _check_multivariate_indices(repeated_indices, n_signals)
    with pytest.raises(
        ValueError, match="a negative channel index is not present in the"
    ):
        missing_chan_indices = (np.array([[0, 1]]), np.array([[2, -5]]))
        _check_multivariate_indices(missing_chan_indices, n_signals)


def test_degree():
    """Test degree function."""
    # degenerate conditions
    with pytest.raises(ValueError, match="threshold"):
        degree(np.eye(3), 2.0)
    # a simple one
    corr = np.eye(10)
    assert_array_equal(degree(corr), np.zeros(10))
    # more interesting
    corr = np.array([[0.5, 0.7, 0.4], [0.1, 0.3, 0.6], [0.2, 0.8, 0.9]])
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
    corr = (corr + corr.T) / 2.0
    assert_array_equal(degree(corr, 0.33), [0, 1, 1])
    assert_array_equal(degree(corr, 0.66), [1, 2, 1])

    # check error when connectivity array is > 2D
    with pytest.raises(ValueError, match="connectivity must be have shape"):
        degree(np.zeros((5, 5, 5)))

    # call degree using a connectivity object
    conn = Connectivity(data=np.zeros((4,)), n_nodes=2)
    deg = degree(conn)
    assert_array_equal(deg, [0, 0])
