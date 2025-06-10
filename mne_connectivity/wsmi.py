# Authors: Giovanni Marraffini <giovanni.marraffini@gmail.com>
#          Laouen Belloli <laouen.belloli@gmail.com>
#          Based on the work of Jean-Remy King, Jacobo Sitt and Federico Raimondo
#
# License: BSD (3-clause)

import math
from itertools import permutations

import numba
import numpy as np
from mne import pick_types
from mne.epochs import BaseEpochs
from mne.preprocessing import compute_current_source_density
from mne.utils import _time_mask, logger, verbose
from mne.utils.check import _validate_type
from mne.utils.docs import fill_doc
from scipy.signal import butter, filtfilt

from .base import EpochTemporalConnectivity


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


def _symb_python_optimized(data, kernel, tau):
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


@numba.njit(parallel=True)  # Enabled parallel execution
def _wsmi_python_jitted(data_sym, counts, wts_matrix, weighted=True):
    """Compute raw wSMI or SMI from symbolic data (Numba-jitted).

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

    for trial_idx in numba.prange(ntrials):
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

    # Note: Original implementation only fills upper triangle, so we match that behavior
    # No mirroring to lower triangle to match original exactly

    return result


def _validate_kernel(kernel, tau, sfreq, memory_limit_gb=1.0, filter_freq=None):
    """Validate kernel and tau parameters for wSMI computation.

    Parameters
    ----------
    kernel : int
        Pattern length (symbol dimension) for symbolic analysis.
    tau : int
        Time delay (lag) between consecutive pattern elements.
    sfreq : float
        Sampling frequency of the data.
    memory_limit_gb : float
        Memory limit in GB for kernel validation. Default is 1.0.
    filter_freq : float | None
        Low-pass filter frequency in Hz to validate if provided.

    Raises
    ------
    ValueError
        If kernel or tau parameters are invalid or memory requirements exceed limits.
    """
    _validate_type(kernel, "int", "kernel")
    _validate_type(tau, "int", "tau")
    _validate_type(memory_limit_gb, "numeric", "memory_limit_gb")

    if kernel <= 1:
        raise ValueError(f"kernel (pattern length) must be > 1, got {kernel}")
    if tau <= 0:
        raise ValueError(f"tau (delay) must be > 0, got {tau}")
    if memory_limit_gb <= 0:
        raise ValueError(f"memory_limit_gb must be > 0, got {memory_limit_gb}")

    # Memory validation for large kernels
    if kernel > 7:  # Factorial grows extremely fast beyond this
        n_symbols = math.factorial(kernel)
        memory_gb = (n_symbols**2 * 8) / (1024**3)  # 8 bytes per double

        if memory_gb > memory_limit_gb:
            raise ValueError(
                f"kernel={kernel} would require ~{memory_gb:.1f} GB of memory "
                f"(factorial({kernel}) = {n_symbols} symbols). "
                f"Current limit: {memory_limit_gb} GB. "
                f"If you have enough RAM, increase 'memory_limit_gb' parameter. "
                f"Otherwise, consider kernel <= 7."
            )

    # Validate filter frequency if provided
    if filter_freq is not None:
        _validate_type(filter_freq, "numeric", "filter_freq")
        nyquist = sfreq / 2.0
        if filter_freq <= 0 or filter_freq >= nyquist:
            raise ValueError(
                f"filter_freq ({filter_freq:.2f}) must be > 0 and < Nyquist "
                f"frequency ({nyquist:.2f} Hz)"
            )


@fill_doc
@verbose
def wsmi(
    epochs,
    kernel,
    tau,
    tmin=None,
    tmax=None,
    filter_freq=None,
    csd=True,
    weighted=True,
    memory_limit_gb=1.0,
    verbose=None,
):
    """Compute weighted symbolic mutual information (wSMI).

    Parameters
    ----------
    epochs : ~mne.Epochs
        The data from which to compute connectivity.
    kernel : int
        Pattern length (symbol dimension) for symbolic analysis.
        Must be > 1. Values > 7 may require significant memory.
    tau : int
        Time delay (lag) between consecutive pattern elements.
        Must be > 0.
    tmin : float | None
        Time to start connectivity estimation. If None, uses beginning
        of epoch.
    tmax : float | None
        Time to end connectivity estimation. If None, uses end of epoch.
    filter_freq : float | None
        Low-pass filter frequency in Hz. If None, defaults to
        sfreq / (kernel * tau).
    csd : bool
        Whether to apply Current Source Density (CSD) computation
        for EEG channels. Default is True.
    weighted : bool
        Whether to compute weighted SMI (wSMI) or standard SMI.
        If True (default), computes wSMI with distance-based weights.
        If False, computes standard SMI without weights.
    memory_limit_gb : float
        Memory limit in GB for kernel validation. Default is 1.0.
        Increase if you have more RAM and want to use larger kernels.
    %(verbose)s

    Returns
    -------
    conn : instance of EpochTemporalConnectivity
        Computed connectivity measures. The connectivity object contains
        the weighted symbolic mutual information values between all channel pairs
        (if weighted=True) or the standard symbolic mutual information values
        between all channel pairs (if weighted=False).

    Notes
    -----
    The weighted Symbolic Mutual Information (wSMI) is a connectivity measure
    that quantifies non-linear statistical dependencies between time series
    based on symbolic dynamics :footcite:`KingEtAl2013`.

    The method involves:
    1. Symbolic transformation of time series using ordinal patterns
    2. Computation of mutual information between symbolic sequences
    3. Weighting based on pattern distance for enhanced sensitivity

    When weighted=False, the function computes standard Symbolic Mutual Information
    (SMI) without distance-based weighting.

    References
    ----------
    .. footbibliography::
    """
    # Input validation
    _validate_type(epochs, BaseEpochs, "epochs")
    _validate_type(csd, bool, "csd")
    _validate_type(weighted, bool, "weighted")

    # Validate all parameters early
    _validate_kernel(kernel, tau, epochs.info["sfreq"], memory_limit_gb, filter_freq)

    sfreq = epochs.info["sfreq"]
    events = epochs.events
    event_id = epochs.event_id
    metadata = epochs.metadata

    # --- 1. Preprocessing (CSD) ---
    # Apply CSD to EEG channels if requested and available
    if (
        csd
        and "eeg" in epochs
        and pick_types(epochs.info, meg=False, eeg=True).size > 0
    ):
        logger.info("Computing Current Source Density (CSD) for EEG channels.")
        epochs_temp = epochs.copy()
        if epochs_temp.info["bads"]:
            logger.info(
                f"""Interpolating {len(epochs_temp.info["bads"])} bad EEG channels
                for CSD computation."""
            )
            epochs_temp.interpolate_bads(reset_bads=True)

        epochs_csd = compute_current_source_density(epochs_temp, lambda2=1e-5)
        # Check if CSD actually produced CSD channels
        if pick_types(epochs_csd.info, csd=True).size > 0:
            epochs = epochs_csd
        else:
            logger.warning(
                """CSD computation did not result in any CSD channels.
                Using original EEG data for EEG channels."""
            )

    # Pick data channels for connectivity computation
    # MEG, EEG, CSD, SEEG, ECoG are typical data channels. Exclude bads.
    picks = pick_types(
        epochs.info,
        meg=True,
        eeg=True,
        csd=True,
        seeg=True,
        ecog=True,
        ref_meg=False,
        exclude="bads",
    )

    if len(picks) == 0:
        raise ValueError(
            """No suitable channels (MEG, EEG, CSD, SEEG, ECoG)
            found after picking logic.
            Check channel types and 'bads'."""
        )

    data_for_comp = epochs.get_data(picks=picks)
    picked_ch_names = [epochs.ch_names[i] for i in picks]
    n_epochs, n_channels_picked, n_times_epoch = data_for_comp.shape

    if n_channels_picked == 0:  # Should be caught by len(picks) == 0
        raise ValueError("No channels selected for wSMI computation after picking.")
    logger.info(
        f"Processing {n_epochs} epochs, {n_channels_picked} channels "
        f"({picked_ch_names}), {n_times_epoch} time points per epoch."
    )

    # --- 2. Filtering (match original exactly) ---
    if filter_freq is None:
        # kernel and tau are already validated in _validate_kernel
        filter_freq = np.double(sfreq) / kernel / tau  # Use np.double like original
    else:
        # filter_freq is already validated in _validate_kernel if provided
        filter_freq = float(filter_freq)
    logger.info(f"Filtering  at {filter_freq:.2f} Hz")  # Match original message format

    # Match original exactly: concatenate epochs, filter, then split back
    b, a = butter(6, 2.0 * filter_freq / np.double(sfreq), "lowpass")
    data_concatenated = np.hstack(data_for_comp)  # Concatenate epochs horizontally

    # Filter the concatenated data
    fdata_concatenated = filtfilt(b, a, data_concatenated)

    # Split back into epochs and transpose to match original format
    fdata = np.transpose(
        np.array(np.split(fdata_concatenated, n_epochs, axis=1)), [1, 2, 0]
    )

    # --- Time masking (match original exactly) ---
    time_mask = _time_mask(epochs.times, tmin, tmax)
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

    # Data is already for symbolic transformation:
    # (n_channels_picked, n_times, n_epochs)
    fdata_for_symb = fdata_masked

    # --- 3. Symbolic Transformation ---
    logger.info("Performing symbolic transformation...")
    try:
        sym, count = _symb_python_optimized(fdata_for_symb, kernel, tau)
    except Exception as e:
        logger.error(f"Error during symbolic transformation: {e}")
        raise

    n_unique_symbols = count.shape[1]
    if (
        sym.shape[0] != n_channels_picked
        or sym.shape[2] != n_epochs
        or count.shape[0] != n_channels_picked
        or count.shape[2] != n_epochs
    ):
        raise ValueError(
            f"""Symbolic transformation output has unexpected shape.
            Got sym: {sym.shape}, count: {count.shape}.
            Expected channels: {n_channels_picked}, epochs: {n_epochs}."""
        )

    wts = _get_weights_matrix(n_unique_symbols)

    # --- 4. wSMI/SMI Computation (Jitted) ---
    method_name = "wSMI" if weighted else "SMI"
    logger.info(
        f"""Computing {method_name} for {n_unique_symbols} unique symbols
        (Numba-jitted)..."""
    )
    result = _wsmi_python_jitted(sym, count, wts, weighted)
    # result is (n_channels_picked, n_channels_picked, n_epochs)

    result_epoched = result.transpose(2, 0, 1)

    # --- Packaging results ---
    if n_channels_picked > 1:
        # Extract upper triangular part (excluding diagonal) for connectivity object
        triu_inds = np.triu_indices(n_channels_picked, k=1)
        indices_list = list(zip(triu_inds[0], triu_inds[1]))

        # Extract connectivity data for upper triangular connections only
        result_conn_data = np.zeros((n_epochs, len(indices_list), 1))
        for epoch_idx in range(n_epochs):
            for conn_idx, (i, j) in enumerate(indices_list):
                result_conn_data[epoch_idx, conn_idx, 0] = result_epoched[
                    epoch_idx, i, j
                ]
    else:
        # For single channel or no channels, create empty connectivity
        logger.info(
            f"Only 1 channel selected, {method_name} connectivity will be empty."
        )
        result_conn_data = np.empty((n_epochs, 0, 1))
        indices_list = []

    # Create connectivity object with prepared data
    result_connectivity = EpochTemporalConnectivity(
        data=result_conn_data,
        names=picked_ch_names,
        times=None,
        method="wSMI" if weighted else "SMI",
        indices=indices_list,
        n_epochs_used=n_epochs,
        n_nodes=n_channels_picked,
        sfreq=sfreq,
        events=events,
        event_id=event_id,
        metadata=metadata,
    )

    logger.info(f"{method_name} computation finished.")

    return result_connectivity


# User's original footbibliography comment can remain here or at the end of the file.
# .. [1] King, J. R., Sitt, J. D., Faugeras, F., Rohaut, B., El Karoui, I.,
#        Cohen, L., ... & Dehaene, S. (2013). Information sharing in the
#        brain indexes consciousness in noncommunicative patients. Current
#        biology, 23(19), 1914-1919.
