# Authors: Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

import copy

import numpy as np
from mne.utils import logger, verbose, warn

from .base import (
    EpochSpectralConnectivity,
    SpectralConnectivity,
    SpectroTemporalConnectivity,
)
from .spectral import spectral_connectivity_epochs, spectral_connectivity_time
from .utils import fill_doc


@verbose
@fill_doc
def phase_slope_index(
    data,
    names=None,
    indices=None,
    sfreq="",
    mode="multitaper",
    fmin=None,
    fmax=np.inf,
    tmin=None,
    tmax=None,
    mt_bandwidth=None,
    mt_adaptive=False,
    mt_low_bias=True,
    cwt_freqs=None,
    cwt_n_cycles=7,
    block_size=1000,
    n_jobs=1,
    verbose=None,
):
    """Compute the Phase Slope Index (PSI) connectivity measure.

    The PSI is an effective connectivity measure, i.e., a measure which can give an
    indication of the direction of the information flow (causality). For two time
    series, and one computes the PSI between the first and the second time series as
    follows::

        indices = (np.array([0]), np.array([1]))
        psi = phase_slope_index(data, indices=indices, ...)

    A positive value means that time series 0 is ahead of time series 1 and a negative
    value means the opposite.

    The PSI is computed from the coherency (see :func:`spectral_connectivity_epochs`),
    details can be found in :footcite:`NolteEtAl2008`.

    Parameters
    ----------
    data : array_like, shape (n_epochs, n_signals, n_times) | ~mne.Epochs | generator
        Can also be a list/generator of arrays, shape ``(n_signals, n_times)``;
        list/generator of :class:`mne.SourceEstimate`; or :class:`mne.Epochs`. The
        data from which to compute connectivity. Note that it is also possible to
        combine multiple signals by providing a list of tuples, e.g., ``data = [(arr_0,
        stc_0), (arr_1, stc_1), (arr_2, stc_2)]``, corresponds to 3 epochs, and
        ``arr_*`` could be an array with the same number of time points as ``stc_*``.
    %(names)s
    indices : tuple of array_like | None
        Two array-likes with indices of connections for which to compute connectivity.
        If ``None``, all connections are computed. See Notes of
        :func:`~mne_connectivity.spectral_connectivity_epochs` for details.
    sfreq : float | None
        The sampling frequency. Default is an empty string for ``2*np.pi`` in 0.8, but
        will change to ``None`` in 0.9. Set it explicitly when ``data`` is an array-like
        to avoid a warning.
    mode : ``'multitaper'`` | ``'fourier'`` | ``'cwt_morlet'``
        Spectrum estimation mode.
    fmin : float | tuple of float
        The lower frequency of interest. Multiple bands are defined using a tuple, e.g.,
        (8., 20.) for two bands with 8 Hz and 20 Hz lower freq. If ``None`` the
        frequency corresponding to an epoch length of 5 cycles is used.
    fmax : float | tuple of float
        The upper frequency of interest. Multiple bands are defined using a tuple, e.g.,
        (13., 30.) for two bands with 13 Hz and 30 Hz upper freq.
    tmin : float | None
        Time to start connectivity estimation.
    tmax : float | None
        Time to end connectivity estimation.
    mt_bandwidth : float | None
        The bandwidth of the multitaper windowing function in Hz. Only used in
        ``'multitaper'`` mode.
    mt_adaptive : bool
        Use adaptive weights to combine the tapered spectra into PSD. Only used in
        ``'multitaper'`` mode.
    mt_low_bias : bool
        Only use tapers with more than 90 percent spectral concentration within
        bandwidth. Only used in ``'multitaper'`` mode.
    cwt_freqs : array_like
        Array-like of frequencies of interest. Only used in ``'cwt_morlet'`` mode.
    cwt_n_cycles : float | array_like
        Number of cycles. Fixed number or one per frequency. Only used in
        ``'cwt_morlet'`` mode.
    block_size : int
        How many connections to compute at once (higher numbers are faster but require
        more memory).
    n_jobs : int
        How many epochs to process in parallel.
    %(verbose)s

    Returns
    -------
    psi : instance of SpectralConnectivity or SpectroTemporalConnectivity
        Computed connectivity measure. Either a :class:`SpectralConnectivity`, or
        :class:`SpectroTemporalConnectivity` container. The shape of the connectivity
        dataset is:

        - ``(n_cons, n_bands)`` for ``'multitaper'`` or ``'fourier'`` modes
        - ``(n_cons, n_bands, n_times)`` for ``'cwt_morlet'`` mode
        - ``n_cons = n_signals ** 2`` when ``indices=None``
        - ``n_cons = len(indices[0])`` when ``indices`` is supplied
        - ``n_bands`` is the number of frequency bands defined by ``fmin`` and ``fmax``

    See Also
    --------
    mne_connectivity.spectral_connectivity_epochs
    mne_connectivity.phase_slope_index_time
    mne_connectivity.SpectralConnectivity
    mne_connectivity.SpectroTemporalConnectivity

    References
    ----------
    .. footbibliography::
    """
    logger.info("Estimating phase slope index (PSI)")

    if sfreq == "":
        sfreq = 2 * np.pi
        if isinstance(data, np.ndarray | list | tuple | set):
            warn(
                "The current default of sfreq=2*np.pi will change to sfreq=None in "
                "0.9. Set the value of sfreq explicitly for array-like inputs to avoid "
                "this warning",
                FutureWarning,
            )

    # estimate the coherency
    cohy = spectral_connectivity_epochs(
        data,
        names,
        method="cohy",
        indices=indices,
        sfreq=sfreq,
        mode=mode,
        fmin=fmin,
        fmax=fmax,
        fskip=0,
        faverage=False,
        tmin=tmin,
        tmax=tmax,
        mt_bandwidth=mt_bandwidth,
        mt_adaptive=mt_adaptive,
        mt_low_bias=mt_low_bias,
        cwt_freqs=cwt_freqs,
        cwt_n_cycles=cwt_n_cycles,
        block_size=block_size,
        n_jobs=n_jobs,
        verbose=verbose,
    )

    # extract class properties from the spectral connectivity structure
    if isinstance(cohy, SpectroTemporalConnectivity):
        times = cohy.times
    else:
        times = None
    freqs = np.array(cohy.freqs)
    names = cohy.names
    n_tapers = cohy.attrs.get("n_tapers")
    n_epochs_used = cohy.n_epochs
    n_nodes = cohy.n_nodes
    metadata = cohy.metadata
    events = cohy.events
    event_id = cohy.event_id

    logger.info(f"Computing PSI from estimated Coherency: {cohy}")
    # compute PSI in the requested bands
    if fmin is None:
        fmin = -np.inf  # set it to -inf, so we can adjust it later
    bands = list(zip(np.asarray((fmin,)).ravel(), np.asarray((fmax,)).ravel()))
    psi, freq_bands, freqs_computed = _compute_psi(
        cohy=cohy,
        freqs=freqs,
        bands=bands,
        freq_dim=-2 if mode == "cwt_morlet" else -1,
    )
    logger.info("[PSI Estimation Done]")

    # create a connectivity container
    if mode in ["multitaper", "fourier"]:
        # spectral only
        psi = SpectralConnectivity(
            data=psi,
            names=names,
            freqs=freq_bands,
            n_nodes=n_nodes,
            method="phase-slope-index",
            spec_method=mode,
            indices=indices,
            freqs_computed=freqs_computed,
            n_epochs_used=n_epochs_used,
            n_tapers=n_tapers,
            metadata=metadata,
            events=events,
            event_id=event_id,
        )
    elif mode == "cwt_morlet":
        # spectrotemporal
        psi = SpectroTemporalConnectivity(
            data=psi,
            names=names,
            freqs=freq_bands,
            times=times,
            n_nodes=n_nodes,
            method="phase-slope-index",
            spec_method=mode,
            indices=indices,
            freqs_computed=freqs_computed,
            n_epochs_used=n_epochs_used,
            n_tapers=n_tapers,
            metadata=metadata,
            events=events,
            event_id=event_id,
        )

    return psi


@verbose
def phase_slope_index_time(
    data,
    freqs=None,
    indices=None,
    sfreq=None,
    mode="cwt_morlet",
    average=False,
    fmin=None,
    fmax=None,
    fskip=0,
    sm_times=0.0,
    sm_freqs=1,
    sm_kernel="hanning",
    padding=0.0,
    mt_bandwidth=4.0,
    n_cycles=7.0,
    decim=1,
    n_jobs=1,
    verbose=None,
):
    """Compute the Phase Slope Index (PSI) connectivity measure over time.

    This function computes PSI over time from epoched data. The data may consist of a
    single epoch.

    The PSI is an effective connectivity measure, i.e., a measure which can give an
    indication of the direction of the information flow (causality). For two time
    series, one computes the PSI between the first and the second time series as
    follows::

        indices = (np.array([0]), np.array([1]))
        psi = phase_slope_index_time(data, indices=indices, ...)

    A positive value means that time series 0 is ahead of time series 1 and a negative
    value means the opposite.

    The PSI is computed from the coherency (see :func:`spectral_connectivity_time`),
    details can be found in :footcite:`NolteEtAl2008`.

    Parameters
    ----------
    data : array-like, shape (n_epochs, n_signals, n_times) | ~mne.Epochs | ~mne.time_frequency.EpochsTFR
        The data from which to compute connectivity. Can be epoched time series data as
        an array-like or :class:`mne.Epochs` object, or Fourier coefficients for each
        epoch as an :class:`mne.time_frequency.EpochsTFR` object. If time series data,
        the spectral information will be computed according to the spectral estimation
        mode (see the ``mode`` parameter). If an :class:`mne.time_frequency.EpochsTFR`
        object, existing spectral information will be used and the ``mode`` parameter
        will be ignored.

        .. versionchanged:: 0.8
           Fourier coefficients stored in an :class:`mne.time_frequency.EpochsTFR`
           object can also be passed in as data. Storing multitaper weights in
           :class:`mne.time_frequency.EpochsTFR` objects requires ``mne >= 1.10``.
    freqs : array_like | None
        Array-like of frequencies of interest for time-frequency decomposition. Only the
        frequencies within the range specified by ``fmin`` and ``fmax`` are used. If
        ``data`` is an array-like or :class:`mne.Epochs` object, the frequencies must
        be specified. If ``data`` is an :class:`mne.time_frequency.EpochsTFR` object,
        ``data.freqs`` is used and this parameter is ignored.
    indices : tuple of array_like | None
        Two array-likes with indices of connections for which to compute connectivity.
        If ``None`` (default), all connections are computed.
    sfreq : float | None
        The sampling frequency. Required if ``data`` is not an :class:`mne.Epochs` or
        :class:`mne.time_frequency.EpochsTFR` object.
    mode : ``'multitaper'`` | ``'cwt_morlet'``
        Time-frequency decomposition method (``'cwt_morlet'`` default). See
        :func:`mne.time_frequency.tfr_array_multitaper` and
        :func:`mne.time_frequency.tfr_array_morlet` for reference. Ignored if ``data``
        is an :class:`mne.time_frequency.EpochsTFR` object.
    average : bool
        Average connectivity scores over epochs. If ``True``, output will be an instance
        of :class:`SpectralConnectivity`, or :class:`EpochSpectralConnectivity` if
        ``False`` (default).
    fmin : float | tuple of float | None
        The lower frequency of interest. Multiple bands are defined using a tuple, e.g.,
        ``(8., 20.)`` for two bands with 8 Hz and 20 Hz lower bounds. If ``None``
        (default), the lowest frequency in ``freqs`` is used.
    fmax : float | tuple of float | None
        The upper frequency of interest. Multiple bands are defined using a tuple, e.g.
        ``(13., 30.)`` for two band with 13 Hz and 30 Hz upper bounds. If ``None``
        (default), the highest frequency in ``freqs`` is used.
    fskip : int
        Omit every ``(fskip + 1)``-th frequency bin to decimate in frequency domain.
        Default is 0 (no skipping).
    sm_times : float
        Amount of time to consider for the temporal smoothing, in seconds. If 0.0
        (default), no temporal smoothing is applied.
    sm_freqs : int
        Number of points for frequency smoothing. If 1 (default), no spectral smoothing
        is applied.
    sm_kernel : ``'square'`` | ``'hanning'``
        Smoothing kernel type. For ``'hanning'``, see :func:`numpy.hanning`.
    padding : float
        Amount of time to consider as padding at the beginning and end of each epoch in
        seconds (0.0 default for no padding). See Notes of
        :func:`spectral_connectivity_time` for more information.
    mt_bandwidth : float
        Product between the temporal window length (in seconds) and the full frequency
        bandwidth (in Hz; default 4.0). This product can be seen as the surface of the
        window on the time/frequency plane and controls the frequency bandwidth (thus
        the frequency resolution) and the number of good tapers. See
        :func:`mne.time_frequency.tfr_array_multitaper` documentation. Ignored if
        ``data`` is an :class:`mne.time_frequency.EpochsTFR` object.
    n_cycles : float | array_like
        Number of cycles in the wavelet, either a fixed number or one per frequency (7.0
        default). The number of cycles ``n_cycles`` and the frequencies of interest
        ``freqs`` define the temporal window length. For details, see
        :func:`mne.time_frequency.tfr_array_multitaper` and
        :func:`mne.time_frequency.tfr_array_morlet` documentation. Ignored if ``data``
        is an :class:`mne.time_frequency.EpochsTFR` object.
    decim : int
        To reduce memory usage, decimation factor after time-frequency decomposition.
        Returns ``tfr[…, ::decim]``. If 1 (default), no decimation occurs.
    n_jobs : int
        Number of connections to compute in parallel. Memory mapping must be activated.
        Please see the Notes section of :func:`spectral_connectivity_time` for details.
    %(verbose)s

    Returns
    -------
    psi : instance of EpochSpectralConnectivity or SpectralConnectivity
        Computed connectivity measure. Either a
        :class:`EpochSpectralConnectivity` or :class:`SpectralConnectivity` container
        depending on the ``average`` parameter. The shape of the connectivity dataset is
        ``([n_epochs,] n_cons, n_bands)``:

        - The epoch dimension is present when ``average=False``, and absent when
          ``average=True``.
        - When ``indices`` is ``None``, ``n_cons = n_signals ** 2``
        - When ``indices`` is specified, ``n_con = len(indices[0])``
        - ``n_bands`` is the number of frequency bands defined by ``fmin`` and ``fmax``

    See Also
    --------
    mne_connectivity.spectral_connectivity_time
    mne_connectivity.phase_slope_index
    mne_connectivity.SpectralConnectivity
    mne_connectivity.EpochSpectralConnectivity

    Notes
    -----
    .. versionadded:: 0.8

    References
    ----------
    .. footbibliography::
    """  # noqa: E501
    logger.info("Estimating phase slope index (PSI) over time")

    # Estimate the coherency
    # Always compute coherency without averaging first, so we can compute PSI for each
    # epoch, then average PSI if requested
    cohy = spectral_connectivity_time(
        data,
        freqs=freqs,
        method="cohy",
        average=False,
        indices=indices,
        sfreq=sfreq,
        fmin=fmin,
        fmax=fmax,
        fskip=fskip,
        faverage=False,
        sm_times=sm_times,
        sm_freqs=sm_freqs,
        sm_kernel=sm_kernel,
        padding=padding,
        mode=mode,
        mt_bandwidth=mt_bandwidth,
        n_cycles=n_cycles,
        decim=decim,
        n_jobs=n_jobs,
        verbose=verbose,
    )

    # extract class properties from the spectral connectivity structure
    freqs = np.array(cohy.freqs)
    names = cohy.names
    n_tapers = cohy.attrs.get("n_tapers")
    n_epochs_used = cohy.n_epochs
    n_nodes = cohy.n_nodes
    metadata = cohy.metadata
    events = cohy.events
    event_id = cohy.event_id

    logger.info(f"Computing PSI over time from estimated Coherency: {cohy}")
    # compute PSI in the requested bands
    if fmin is None:
        fmin = -np.inf
    if fmax is None:
        fmax = np.inf
    bands = list(zip(np.asarray((fmin,)).ravel(), np.asarray((fmax,)).ravel()))
    psi, freq_bands, freqs_computed = _compute_psi(
        cohy=cohy, freqs=freqs, bands=bands, freq_dim=-1
    )
    logger.info("[PSI Estimation Done]")

    # create a connectivity container
    conn_kwargs = dict(
        names=names,
        freqs=freq_bands,
        n_nodes=n_nodes,
        method="phase-slope-index",
        spec_method=mode,
        indices=indices,
        freqs_computed=freqs_computed,
        n_tapers=n_tapers,
        metadata=metadata,
        events=events,
        event_id=event_id,
    )
    if average:
        # average over epochs
        psi = SpectralConnectivity(
            data=psi.mean(axis=0), n_epochs_used=n_epochs_used, **conn_kwargs
        )
    else:
        psi = EpochSpectralConnectivity(data=psi, **conn_kwargs)

    return psi


def _compute_psi(cohy, freqs, bands, freq_dim):
    """Compute Phase Slope Index (PSI) from coherency data."""
    # Allocate space for output
    out_shape = list(cohy.shape)
    out_shape[freq_dim] = len(bands)
    psi = np.zeros(out_shape, dtype=np.float64)

    # Allocate accumulator
    acc_shape = copy.copy(out_shape)
    acc_shape.pop(freq_dim)
    acc = np.empty(acc_shape, dtype=np.complex128)

    # Create list for frequencies used and frequency bands of results
    freqs_computed = list()
    freq_bands = list()
    idx_fi = [slice(None)] * len(out_shape)
    idx_fj = [slice(None)] * len(out_shape)
    for band_idx, band in enumerate(bands):
        freq_idx = np.where((freqs > band[0]) & (freqs < band[1]))[0]
        freqs_computed.append(freqs[freq_idx])
        freq_bands.append(np.mean(freqs[freq_idx]))

        acc.fill(0.0)
        for fi, fj in zip(freq_idx, freq_idx[1:]):
            idx_fi[freq_dim] = fi
            idx_fj[freq_dim] = fj
            acc += (
                np.conj(cohy.get_data()[tuple(idx_fi)]) * cohy.get_data()[tuple(idx_fj)]
            )

        idx_fi[freq_dim] = band_idx
        psi[tuple(idx_fi)] = np.imag(acc)

    return psi, freq_bands, freqs_computed
