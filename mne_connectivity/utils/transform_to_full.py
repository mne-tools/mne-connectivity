import numpy as np


def _get_full_connectivity(data, indices, n_nodes, method, missing):
    """Get full connectivity matrix from raveled data.

    Attempts to fill missing values in the connectivity data where needed, if the
    method is known and the logic to do so is possible from what data is available.
    """
    # Fill with specified missing value if requested
    if missing != "raise":
        return _make_square(data, indices, n_nodes, fill=missing)

    # Check if data is already full (no need to fill missing values)
    if indices == "all":
        return _make_square(data, indices, n_nodes)
    if isinstance(indices, tuple):
        indices = tuple(np.asarray(ind) for ind in indices)
        full_indices = np.unravel_index(np.arange(n_nodes**2), (n_nodes, n_nodes))
        if not np.array_equal(indices, full_indices):
            raise ValueError(
                "Cannot fill missing values for connectivity data when indices are "
                "specified as a tuple that does not represent the full connectivity "
                "matrix."
            )
        return _make_square(data, indices, n_nodes)

    # Check if we can infer missing entries in the data for the method
    if method not in _CAN_FILL_MISSING:
        raise ValueError(
            f"Cannot fill missing values for connectivity data for the method {method}."
        )
    data = _make_square(data, indices, n_nodes)
    return _CAN_FILL_MISSING[method](data, indices)


def _make_square(data, indices, n_nodes, fill=0.0):
    """Make raveled connectivity data square [n_nodes, n_nodes(, ...)]."""
    if np.iscomplexobj(data):
        fill = fill + 1j * fill
    square_matrix = np.full(
        (n_nodes, n_nodes, *data.shape[1:]), fill_value=fill, dtype=data.dtype
    )

    if indices == "all":
        indices = np.unravel_index(np.arange(n_nodes**2), (n_nodes, n_nodes))
    elif indices == "lower":
        indices = np.tril_indices(n_nodes, k=-1)
    elif indices == "upper":
        indices = np.triu_indices(n_nodes, k=1)
    # else indices is a tuple of arrays and can be used directly

    square_matrix[indices] = data

    return square_matrix


def _make_full(data, indices, diag, transpose_extra=None):
    """Fill missing values in the connectivity data."""
    assert indices in ("lower", "upper"), (
        "Expected indices to be 'lower' or 'upper' for symmetrisation, got "
        f"{indices}. Please contact the MNE-Connectivity developers."
    )
    # Always transpose
    data = data + data.transpose(1, 0, *range(2, data.ndim))
    # Perform something on top of the transpose if needed
    if transpose_extra is not None:
        if indices == "lower":
            triu = np.triu_indices(data.shape[0], k=1)
            data[triu] = transpose_extra(data[triu])
        else:  # "upper"
            tril = np.tril_indices(data.shape[0], k=-1)
            data[tril] = transpose_extra(data[tril])
    # Set the diagonal
    data[np.diag_indices(data.shape[0])] = diag
    return data


def _transpose_zero_diag(data, indices):
    """Fill missing values by transposing, with zeros on the diagonal."""
    return _make_full(data, indices, diag=0.0)


def _transpose_one_diag(data, indices):
    """Fill missing values by transposing, with ones on the diagonal."""
    return _make_full(data, indices, diag=1.0)


def _transpose_conj_one_diag(data, indices):
    """Fill missing values by transposing, with ones on the diagonal and conjugate."""
    return _make_full(data, indices, diag=1.0 + 0.0j, transpose_extra=np.conj)


def _transpose_sign_flip_zero_diag(data, indices):
    """Fill missing values by transposing with sign flip, and zeros on the diagonal."""
    return _make_full(data, indices, diag=0.0, transpose_extra=lambda x: -x)


def _fill_dpli(data, indices):
    """Fill missing directed phase lag index values."""
    return _make_full(data, indices, diag=0.5, transpose_extra=lambda x: 1.0 - x)


_CAN_FILL_MISSING = {
    "coh": _transpose_one_diag,
    "cohy": _transpose_conj_one_diag,
    "imcoh": _transpose_sign_flip_zero_diag,
    "plv": _transpose_one_diag,
    "ciplv": _transpose_zero_diag,
    "ppc": _transpose_one_diag,
    "pli": _transpose_zero_diag,
    "pli2_unbiased": _transpose_zero_diag,
    "dpli": _fill_dpli,
    "wpli": _transpose_zero_diag,
    "wpli2_debiased": _transpose_zero_diag,
    "phase-slope-index": _transpose_sign_flip_zero_diag,
    "SMI": _transpose_zero_diag,
    "wSMI": _transpose_zero_diag,
}
