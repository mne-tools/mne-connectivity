# Authors: Giovanni Marraffini <giovanni.marraffini@gmail.com>
#          Laouen Belloli <laouen.belloli@gmail.com>
#          Based on the work of Jean-Remy King, Jacobo Sitt and Federico Raimondo
#
# License: BSD (3-clause)

import math
import warnings
from itertools import permutations

import numpy as np
from mne._fiff.pick import _picks_to_idx
from mne.epochs import BaseEpochs
from mne.fixes import jit
from mne.utils import _time_mask, logger, verbose
from mne.utils.check import _check_option, _validate_type
from mne.utils.docs import fill_doc
from scipy.signal import butter, filtfilt

from .base import Connectivity, EpochConnectivity
from .utils import check_indices


def _define_symbols(kernel):
    """Define all possible symbols for a given kernel size (original implementation)."""
    result_dict = dict()
    total_symbols = math.factorial(kernel)
    cursymbol = 0
    for perm in permutations(range(kernel)):
        order = "".join(map(str, perm))
        if order not in result_dict:
            result_dict[order] = cursymbol
            cursymbol = cursymbol + 1
            result_dict[order[::-1]] = total_symbols - cursymbol
    result = []
    for v in range(total_symbols):
        for symbol, value in result_dict.items():
            if value == v:
                result += [symbol]
    return result


def _symb(data, kernel, tau):
    """Compute symbolic transform using original logic but optimized.

    This matches the original _symb_python exactly but with optimizations.
    """
    symbols = _define_symbols(kernel)
    dims = data.shape

    signal_sym_shape = list(dims)
    signal_sym_shape[1] = data.shape[1] - tau * (kernel - 1)
    signal_sym = np.zeros(signal_sym_shape, np.int32)

    count_shape = list(dims)
    count_shape[1] = len(symbols)
    count = np.zeros(count_shape, np.int32)

    # Create a dict for fast lookup (instead of symbols.index which is O(n))
    symbol_to_idx = {symbol: idx for idx, symbol in enumerate(symbols)}

    for k in range(signal_sym_shape[1]):
        subsamples = range(k, k + kernel * tau, tau)
        ind = np.argsort(data[:, subsamples], 1)

        # Process each channel and epoch
        for ch in range(data.shape[0]):
            for ep in range(data.shape[2]):
                symbol_str = "".join(map(str, ind[ch, :, ep]))
                signal_sym[ch, k, ep] = symbol_to_idx[symbol_str]

    count = np.double(
        np.apply_along_axis(
            lambda x: np.bincount(x, minlength=len(symbols)), 1, signal_sym
        )
    )

    return signal_sym, (count / signal_sym_shape[1])


def _get_weights_matrix(nsym):
    """Aux function (original implementation)."""
    wts = np.ones((nsym, nsym))
    np.fill_diagonal(wts, 0)
    wts = np.fliplr(wts)
    np.fill_diagonal(wts, 0)
    wts = np.fliplr(wts)
    return wts


@jit(parallel=True)  # Enabled parallel execution
def _wsmi_jitted(  # pragma: no cover
    data_sym, counts, wts_matrix, weighted=True
):
    """Compute raw wSMI or SMI from symbolic data (Numba-jitted, if installed).

    Parameters
    ----------
    data_sym : ndarray
        Symbolic data.
    counts : ndarray
        Symbol counts.
    wts_matrix : ndarray
        Weights matrix.
    weighted : bool
        If True, compute wSMI. If False, compute SMI.

    Returns
    -------
    result : ndarray
        Computed connectivity values (either wSMI or SMI).
    """
    nchannels, nsamples_after_symb, ntrials = data_sym.shape
    n_unique_symbols = counts.shape[1]

    result = np.zeros((nchannels, nchannels, ntrials), dtype=np.double)

    epsilon = 1e-15
    log_counts = np.log(counts + epsilon)

    for trial_idx in range(ntrials):
        for ch1_idx in range(nchannels):
            for ch2_idx in range(ch1_idx + 1, nchannels):
                pxy = np.zeros((n_unique_symbols, n_unique_symbols), dtype=np.double)
                for sample_idx in range(nsamples_after_symb):
                    sym1 = data_sym[ch1_idx, sample_idx, trial_idx]
                    sym2 = data_sym[ch2_idx, sample_idx, trial_idx]

                    pxy[sym1, sym2] += 1

                if nsamples_after_symb > 0:
                    pxy /= nsamples_after_symb

                current_result_val = 0.0

                # Compute MI terms manually to avoid broadcasting issues in Numba
                for r_idx in range(n_unique_symbols):
                    for c_idx in range(n_unique_symbols):
                        if pxy[r_idx, c_idx] > epsilon:
                            log_pxy_val = np.log(pxy[r_idx, c_idx])
                            log_px_val = log_counts[ch1_idx, r_idx, trial_idx]
                            log_py_val = log_counts[ch2_idx, c_idx, trial_idx]

                            mi_term = pxy[r_idx, c_idx] * (
                                log_pxy_val - log_px_val - log_py_val
                            )

                            if weighted:
                                current_result_val += wts_matrix[r_idx, c_idx] * mi_term
                            else:
                                current_result_val += mi_term

                result[ch1_idx, ch2_idx, trial_idx] = current_result_val

    if n_unique_symbols > 1:
        norm_factor = np.log(n_unique_symbols)
        if norm_factor > epsilon:
            result /= norm_factor
    else:
        result_fill_val = 0.0
        result[:, :, :] = result_fill_val

    return result + result.transpose(1, 0, 2)  # make symmetric


def _apply_anti_aliasing(data, sfreq, kernel, tau, anti_aliasing, is_epochs, info=None):
    """Apply anti-aliasing filtering based on parameters and data type.

    Parameters
    ----------
    data : ndarray
        Data array of shape (n_epochs, n_channels, n_times).
    sfreq : float
        Sampling frequency in Hz.
    kernel : int
        Pattern length for symbolic analysis.
    tau : int
        Time delay between pattern elements.
    anti_aliasing : bool | str
        Anti-aliasing mode: True (always), False (never), or "auto" (smart detection).
    is_epochs : bool
        Whether the original data was an MNE Epochs object.
    info : mne.Info | None
        MNE Info object (only available if is_epochs=True).

    Returns
    -------
    filtered_data : ndarray
        Data array of shape (n_channels, n_times, n_epochs) ready for symbolic
        transformation, with anti-aliasing applied if needed.
    """
    n_epochs = data.shape[0]
    anti_alias_freq = np.double(sfreq) / kernel / tau
    nyquist_freq = sfreq / 2.0

    # Determine if filtering is needed based on anti_aliasing mode
    should_filter = False
    skip_reason = None

    if anti_aliasing is False:
        # Never filter - warn about potential issues
        effective_sfreq = sfreq / tau
        warnings.warn(
            f"Anti-aliasing disabled. Effective sampling rate for symbolic "
            f"transformation is {effective_sfreq:.1f} Hz (sfreq/tau={sfreq}/{tau}). "
            f"Ensure your data is appropriately filtered to prevent aliasing.",
            UserWarning,
        )
        should_filter = False
    else:  # True or "auto"
        # Check if anti-aliasing frequency is too close to Nyquist
        if anti_alias_freq >= nyquist_freq * 0.99:
            skip_reason = (
                f"Anti-aliasing frequency ({anti_alias_freq:.2f} Hz) too close to "
                f"Nyquist frequency ({nyquist_freq:.2f} Hz)"
            )
            should_filter = False
        else:
            if anti_aliasing is True:
                should_filter = True
            else:  # Auto mode: smart detection based on data type and preprocessing
                if not is_epochs:
                    # Array input: always filter since we don't know preprocessing
                    logger.info(
                        "Auto anti-aliasing: Array input detected, applying filter "
                        "(preprocessing history unknown)."
                    )
                    should_filter = True
                else:
                    # MNE Epochs: check if already appropriately filtered
                    existing_lowpass = info.get("lowpass", None)

                    if (
                        existing_lowpass is not None
                        and existing_lowpass <= anti_alias_freq
                    ):
                        # Data already filtered at or below required frequency
                        logger.info(
                            f"Auto anti-aliasing: Data already low-pass filtered at "
                            f"{existing_lowpass:.2f} Hz (<= {anti_alias_freq:.2f} Hz). "
                            f"Skipping additional filtering."
                        )
                        should_filter = False
                    else:
                        # Need to apply filtering
                        if existing_lowpass is not None:
                            logger.info(
                                f"Auto anti-aliasing: Existing lowpass "
                                f"({existing_lowpass:.2f} Hz) > required "
                                f"({anti_alias_freq:.2f} Hz). Applying filter."
                            )
                        else:
                            logger.info(
                                f"Auto anti-aliasing: No lowpass filter info found. "
                                f"Applying filter at {anti_alias_freq:.2f} Hz."
                            )
                        should_filter = True

    # Apply filtering if needed
    if should_filter:
        logger.info(f"Applying anti-aliasing filter at {anti_alias_freq:.2f} Hz")

        # Make a copy to avoid modifying original data
        data = data.copy()

        # Design and apply low-pass filter
        normalized_freq = 2.0 * anti_alias_freq / np.double(sfreq)
        b, a = butter(6, normalized_freq, "lowpass")

        # Concatenate epochs horizontally for filtering
        data_concatenated = np.hstack(data)

        # Filter the concatenated data
        fdata_concatenated = filtfilt(b, a, data_concatenated)

        # Split back into epochs and transpose to match expected format
        # Output shape: (n_channels, n_times, n_epochs)
        filtered_data = np.transpose(
            np.array(np.split(fdata_concatenated, n_epochs, axis=1)), [1, 2, 0]
        )
    else:
        if skip_reason:
            logger.info(f"{skip_reason}. Skipping anti-aliasing filter.")
        # Transpose to match expected format: (n_channels, n_times, n_epochs)
        filtered_data = data.transpose(1, 2, 0)

    return filtered_data


def _validate_kernel(kernel, tau):
    """Validate kernel and tau parameters for wSMI computation.

    Parameters
    ----------
    kernel : int
        Pattern length (symbol dimension) for symbolic analysis.
    tau : int
        Time delay (lag) between consecutive pattern elements.

    Raises
    ------
    ValueError
        If kernel or tau parameters are invalid.
    """
    _validate_type(kernel, "int", "kernel")
    _validate_type(tau, "int", "tau")

    if kernel <= 1:
        raise ValueError(f"kernel (pattern length) must be > 1, got {kernel}")
    if tau <= 0:
        raise ValueError(f"tau (delay) must be > 0, got {tau}")

    # Warn about potentially large memory requirements for large kernels
    if kernel > 7:  # Factorial grows extremely fast beyond this
        n_symbols = math.factorial(kernel)
        memory_gb = (n_symbols**2 * 8) / (1024**3)  # 8 bytes per double
        warnings.warn(
            f"kernel={kernel} will require ~{memory_gb:.1f} GB of memory "
            f"(factorial({kernel}) = {n_symbols} symbols). "
            f"Consider using kernel <= 7 if you encounter memory errors.",
            UserWarning,
            stacklevel=3,
        )


@fill_doc
@verbose
def wsmi(
    data,
    kernel,
    tau,
    indices=None,
    sfreq=None,
    names=None,
    tmin=None,
    tmax=None,
    anti_aliasing="auto",
    weighted=True,
    average=False,
    verbose=None,
):
    """Compute weighted symbolic mutual information (wSMI).

    Parameters
    ----------
    data : array_like, shape (n_epochs, n_signals, n_times) | ~mne.Epochs
        The data from which to compute connectivity. Can be an :class:`mne.Epochs`
        object or array-like data.
    kernel : int
        Pattern length (symbol dimension) for symbolic analysis.
        Must be > 1. Values > 7 may require significant memory.
    tau : int
        Time delay (lag) between consecutive pattern elements.
        Must be > 0.
    indices : tuple of array_like | None
        Two array-likes with indices of connections for which to compute connectivity.
        If ``None``, all connections are computed (lower triangular matrix).
        For example, to compute connectivity between channels 0 and 2, and between
        channels 1 and 3, use ``indices = (np.array([0, 1]), np.array([2, 3]))``.
    sfreq : float | None
        The sampling frequency. Required if ``data`` is an array-like.
    names : list | None
        Channel names. If None, default names will be used.
    tmin : float | None
        Time to start connectivity estimation. If ``None``, uses beginning
        of epoch.
    tmax : float | None
        Time to end connectivity estimation. If ``None``, uses end of epoch.
    anti_aliasing : bool | str
        Controls anti-aliasing low-pass filtering before symbolic transformation.

        - ``"auto"`` (default): Smart detection based on data type and preprocessing.
          For array inputs, always applies filtering. For MNE Epochs, checks
          ``info['lowpass']`` to determine if data is already appropriately filtered.
          Only applies filtering if existing lowpass > required frequency.
        - ``True``: Always apply anti-aliasing filter at ``sfreq / (kernel * tau)`` Hz.
        - ``False``: Never apply filtering. Use only if you have already applied
          appropriate low-pass filtering to your data.

        .. warning::
            Setting to ``False`` may produce unreliable results if the
            effective sampling rate (``sfreq / tau``) violates the Nyquist criterion
            for the spectral content of your data.
    weighted : bool
        Whether to compute weighted SMI (wSMI) or standard SMI.
        If ``True`` (default), computes wSMI with distance-based weights.
        If ``False``, computes standard SMI without weights.
    average : bool
        Whether to average connectivity across epochs. If ``True``, returns
        connectivity averaged over epochs. If ``False`` (default), returns
        connectivity for each epoch separately.
    %(verbose)s

    Returns
    -------
    conn : instance of Connectivity or EpochConnectivity
        Computed connectivity measure. If ``average=True``, returns a
        :class:`Connectivity` instance with connectivity averaged across epochs. If
        ``average=False``, returns an :class:`EpochConnectivity` instance with
        connectivity for each epoch.

    Notes
    -----
    The weighted Symbolic Mutual Information (wSMI) is a connectivity measure
    that quantifies non-linear statistical dependencies between time series
    based on symbolic dynamics :footcite:`KingEtAl2013`.

    The method involves:

    1. Symbolic transformation of time series using ordinal patterns
    2. Computation of mutual information between symbolic sequences
    3. Weighting based on pattern distance for enhanced sensitivity

    When ``weighted=False``, the function computes standard Symbolic Mutual Information
    (SMI) without distance-based weighting.

    **Anti-aliasing filtering**:
    By default, the function applies automatic low-pass filtering to prevent aliasing
    artifacts that can corrupt ordinal patterns when ``tau > 1``. The filter frequency
    is set to ``sfreq / (kernel * tau)`` Hz, which ensures the spectral content matches
    the effective temporal sampling rate of the symbolic transformation. Users who have
    already applied appropriate preprocessing can disable this by setting
    ``anti_aliasing=False``.

    References
    ----------
    .. footbibliography::
    """
    # Input validation and data handling for both Epochs and arrays
    _validate_type(weighted, bool, "weighted")
    _validate_type(average, bool, "average")
    _check_option("anti_aliasing", anti_aliasing, (True, False, "auto"))

    # Handle both MNE Epochs and array inputs
    picks = None
    is_epochs = isinstance(data, BaseEpochs)
    info = None

    if is_epochs:
        info = data.info
        sfreq = info["sfreq"]
        events = data.events
        event_id = data.event_id
        metadata = data.metadata
        ch_names = data.ch_names
        times = data.times

        # Get data
        data_for_comp = data.get_data()
        n_epochs, n_nodes, n_times_epoch = data_for_comp.shape

        # Only exclude bad channels when indices is None
        if indices is None:
            picks = _picks_to_idx(info, picks="all", exclude="bads")
            # Apply picks to data for computation
            data_for_comp = data_for_comp[:, picks, :]
            n_epochs, n_channels, n_times_epoch = data_for_comp.shape
        else:
            # User provided explicit indices, use all channels
            n_channels = n_nodes
    else:
        # Array-like input
        if sfreq is None:
            raise ValueError("Sampling frequency (sfreq) is required with array input.")

        data_for_comp = np.asarray(data)
        if data_for_comp.ndim != 3:
            raise ValueError(
                f"Array input must be 3D (n_epochs, n_channels, n_times), "
                f"got shape {data_for_comp.shape}"
            )
        n_epochs, n_channels, n_times_epoch = data_for_comp.shape
        n_nodes = n_channels
        picks = np.arange(n_channels)
        times = np.arange(n_times_epoch) / sfreq

        # Set default values for array input
        events = None
        event_id = None
        metadata = None

        # Handle names parameter - just validate if provided
        if names is not None and len(names) != n_channels:
            raise ValueError(
                f"Number of names ({len(names)}) must match number of "
                f"channels ({n_channels})"
            )
        ch_names = names

    # Validate all parameters early
    _validate_kernel(kernel, tau)

    # Check for insufficient channels for connectivity computation
    if n_channels < 2:
        raise ValueError(
            f"At least 2 channels are required for connectivity computation, "
            f"but only {n_channels} channels are available."
        )

    logger.info(
        f"Processing {n_epochs} epochs, {n_channels} channels "
        f"({ch_names}), {n_times_epoch} time points per epoch."
    )

    # Handle indices parameter
    if indices is None:
        logger.info("using all connections for lower-triangular matrix")
        # Compute lower-triangular connections
        indices_use = np.tril_indices(n_channels, k=-1)
    else:
        # User provided explicit indices
        indices_use = check_indices(indices)

        # Check that we have at least one valid connection
        if len(indices_use[0]) == 0:
            raise ValueError("No valid connections specified in indices parameter.")

        # Validate that indices are within the range of channels
        max_idx = max(np.max(indices_use[0]), np.max(indices_use[1]))
        if max_idx >= n_channels:
            raise ValueError(
                f"Index {max_idx} is out of range for {n_channels} channels"
            )

        # Check that indices don't refer to the same channel (no self-connectivity)
        same_channel_mask = indices_use[0] == indices_use[1]
        if np.any(same_channel_mask):
            invalid_pairs = [
                (indices_use[0][i], indices_use[1][i])
                for i in range(len(indices_use[0]))
                if same_channel_mask[i]
            ]
            raise ValueError(
                f"Self-connectivity not supported. Found invalid pairs: {invalid_pairs}"
            )

        logger.info(f"computing connectivity for {len(indices_use[0])} connections")

    # unique signals for which we actually need to compute values for
    sig_idx = np.unique(np.r_[indices_use[0], indices_use[1]])
    # map indices to unique indices
    idx_map = [np.searchsorted(sig_idx, ind) for ind in indices_use]
    # select only needed signals
    data_for_comp = data_for_comp[:, sig_idx]

    # --- 2. Anti-aliasing filtering ---
    fdata = _apply_anti_aliasing(
        data_for_comp, sfreq, kernel, tau, anti_aliasing, is_epochs, info
    )

    # --- Time masking ---
    time_mask = _time_mask(times, tmin, tmax)
    fdata_masked = fdata[:, time_mask, :]

    # Check if time masking resulted in too few samples for symbolization
    min_samples_needed_for_one_symbol = tau * (kernel - 1) + 1
    if fdata_masked.shape[1] < min_samples_needed_for_one_symbol:
        raise ValueError(
            f"""After time masking ({tmin}-{tmax}s), data has
            {fdata_masked.shape[1]} samples per epoch, but at least
            {min_samples_needed_for_one_symbol} are needed for kernel={kernel},
            tau={tau}. Adjust tmin/tmax or check epoch length."""
        )

    # Data is all ready for symbolic transformation:
    # (n_channels, n_times, n_epochs)
    fdata_for_symb = fdata_masked

    # --- 3. Symbolic Transformation ---
    logger.info("Performing symbolic transformation...")
    try:
        sym, count = _symb(fdata_for_symb, kernel, tau)
    except MemoryError as error:
        n_symbols = math.factorial(kernel)
        memory_gb = (n_symbols**2 * 8) / (1024**3)
        raise MemoryError(
            f"Insufficient memory for kernel={kernel} (requires ~{memory_gb:.1f} GB). "
            f"Try reducing kernel size (e.g., kernel <= 7) or use fewer "
            f"channels/epochs."
        ) from error
    except Exception as e:
        raise RuntimeError(
            "Error during symbolic transformation. Please contact the "
            "MNE-Connectivity developers."
        ) from e

    n_unique_symbols = count.shape[1]
    wts = _get_weights_matrix(n_unique_symbols)

    # --- 4. wSMI/SMI Computation ---
    method_name = "wSMI" if weighted else "SMI"
    logger.info(f"""Computing {method_name} for {n_unique_symbols} unique symbols...""")
    result = _wsmi_jitted(sym, count, wts, weighted)
    # Result is (n_channels, n_channels, n_epochs)
    result = result.transpose(2, 0, 1)  # make epochs first dimension

    # --- Packaging results ---
    if indices is None:
        # Make it a lower-triangular matrix
        result = np.tril(result, k=-1)
        # Return all-to-all connectivity matrices raveled into a 1D array
        if len(picks) < n_nodes:
            # Bad channels were excluded, need to create full n_nodes x n_nodes matrix
            # and fill only the good channel entries
            con = np.zeros((n_epochs, n_nodes, n_nodes))
            con[np.ix_(range(n_epochs), picks, picks)] = result
        else:
            con = result
        con = con.reshape(n_epochs, -1)
    else:
        # Extract only requested connections
        con = result[:, idx_map[0], idx_map[1]]

    # Create connectivity object with prepared data
    con_kwargs = dict(
        names=ch_names,
        method=method_name,
        indices=indices,
        n_epochs_used=n_epochs,
        n_nodes=n_nodes,
        events=events,
        event_id=event_id,
        metadata=metadata,
    )
    if average:
        result_connectivity = Connectivity(data=np.mean(con, axis=0), **con_kwargs)
    else:
        result_connectivity = EpochConnectivity(data=con, **con_kwargs)

    logger.info(f"{method_name} computation finished.")

    return result_connectivity
