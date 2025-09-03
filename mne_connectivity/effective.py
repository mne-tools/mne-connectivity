# Authors: Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

import copy

import numpy as np
from mne.utils import logger, verbose

from .base import SpectralConnectivity, SpectroTemporalConnectivity
from .spectral import spectral_connectivity_epochs
from .utils import fill_doc


@verbose
@fill_doc
def phase_slope_index(
    data,
    names=None,
    indices=None,
    sfreq=2 * np.pi,
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
    data : array_like, shape (n_epochs, n_signals, n_times) | ~mne.Epochs | generator | ~mne.time_frequency.EpochsSpectrum | ~mne.time_frequency.EpochsTFR
        The data from which to compute connectivity. Can be epoched time series data as
        an array-like or :class:`mne.Epochs` object, or Fourier coefficients for each
        epoch as an :class:`mne.time_frequency.EpochsSpectrum` or
        :class:`mne.time_frequency.EpochsTFR` object. If time series data, the spectral
        information will be computed according to the spectral estimation mode (see the
        ``mode`` parameter). If an :class:`mne.time_frequency.EpochsSpectrum` or
        :class:`mne.time_frequency.EpochsTFR` object, existing spectral information
        will be used and the ``mode`` parameter will be ignored.

        Note that it is also possible to combine multiple time series signals by
        providing a list of tuples, e.g.: ::

            data = [(arr_0, stc_0), (arr_1, stc_1), (arr_2, stc_2)]

        which corresponds to 3 epochs where ``arr_*`` is an array with the same number
        of time points as ``stc_*``. Data can also be a list/generator of arrays, shape
        ``(n_signals, n_times)``, or a list/generator of :class:`mne.SourceEstimate` or
        :class:`mne.VolSourceEstimate` objects.

        .. versionchanged:: 0.8
           Fourier coefficients stored in an :class:`mne.time_frequency.EpochsSpectrum`
           or :class:`mne.time_frequency.EpochsTFR` object can also be passed in as
           data. Storing Fourier coefficients in
           :class:`mne.time_frequency.EpochsSpectrum` objects requires ``mne >= 1.8``.
           Storing multitaper weights in :class:`mne.time_frequency.EpochsTFR` objects
           requires ``mne >= 1.10``.
    %(names)s
    indices : tuple of array_like | None
        Two array-likes with indices of connections for which to compute connectivity.
        If ``None``, all connections are computed. See Notes of
        :func:`~mne_connectivity.spectral_connectivity_epochs` for details.
    sfreq : float
        The sampling frequency.
    mode : ``'multitaper'`` | ``'fourier'`` | ``'cwt_morlet'``
        Spectrum estimation mode. Ignored if ``data`` is an
        :class:`mne.time_frequency.EpochsSpectrum` or
        :class:`mne.time_frequency.EpochsTFR` object.
    fmin : float | tuple of float
        The lower frequency of interest. Multiple bands are defined using a tuple, e.g.,
        (8., 20.) for two bands with 8 Hz and 20 Hz lower freq. If ``None`` the
        frequency corresponding to an epoch length of 5 cycles is used.
    fmax : float | tuple of float
        The upper frequency of interest. Multiple bands are defined using a tuple, e.g.,
        (13., 30.) for two bands with 13 Hz and 30 Hz upper freq.
    tmin : float | None
        Time to start connectivity estimation. Ignored if ``data`` is an
        :class:`mne.time_frequency.EpochsSpectrum` object.
    tmax : float | None
        Time to end connectivity estimation. Ignored if ``data`` is an
        :class:`mne.time_frequency.EpochsSpectrum` object.
    mt_bandwidth : float | None
        The bandwidth of the multitaper windowing function in Hz. Only used in
        ``'multitaper'`` mode. Ignored if ``data`` is an
        :class:`mne.time_frequency.EpochsSpectrum` or
        :class:`mne.time_frequency.EpochsTFR` object.
    mt_adaptive : bool
        Use adaptive weights to combine the tapered spectra into PSD. Only used in
        ``'multitaper'`` mode. Ignored if ``data`` is an
        :class:`mne.time_frequency.EpochsSpectrum` or
        :class:`mne.time_frequency.EpochsTFR` object.
    mt_low_bias : bool
        Only use tapers with more than 90 percent spectral concentration within
        bandwidth. Only used in ``'multitaper'`` mode. Ignored if ``data`` is an
        :class:`mne.time_frequency.EpochsSpectrum` or
        :class:`mne.time_frequency.EpochsTFR` object.
    cwt_freqs : array_like
        Array-like of frequencies of interest. Only used in ``'cwt_morlet'`` mode.
        Ignored if ``data`` is an :class:`mne.time_frequency.EpochsSpectrum` or
        :class:`mne.time_frequency.EpochsTFR` object.
    cwt_n_cycles : float | array_like
        Number of cycles. Fixed number or one per frequency. Only used in
        ``'cwt_morlet'`` mode. Ignored if ``data`` is an
        :class:`mne.time_frequency.EpochsSpectrum` or
        :class:`mne.time_frequency.EpochsTFR` object.
    block_size : int
        How many connections to compute at once (higher numbers are faster but require
        more memory).
    n_jobs : int
        How many epochs to process in parallel.
    %(verbose)s

    Returns
    -------
    conn : instance of SpectralConnectivity or SpectroTemporalConnectivity
        Computed connectivity measure(s). Either a :class:`SpectralConnectivity`, or
        :class:`SpectroTemporalConnectivity` container. The shape of each array is:

        - ``(n_cons, n_bands)`` for ``'multitaper'`` or ``'fourier'`` modes
        - ``(n_cons, n_bands, n_times)`` for ``'cwt_morlet'`` mode
        - ``n_cons = n_signals ** 2`` when ``indices=None``
        - ``n_cons = len(indices[0])`` when ``indices`` is supplied

    See Also
    --------
    mne_connectivity.SpectralConnectivity
    mne_connectivity.SpectroTemporalConnectivity

    References
    ----------
    .. footbibliography::
    """  # noqa: E501
    logger.info("Estimating phase slope index (PSI)")
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
    freqs_ = np.array(cohy.freqs)
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
    n_bands = len(bands)

    freq_dim = -2 if isinstance(cohy, SpectroTemporalConnectivity) else -1

    # allocate space for output
    out_shape = list(cohy.shape)
    out_shape[freq_dim] = n_bands
    psi = np.zeros(out_shape, dtype=np.float64)

    # allocate accumulator
    acc_shape = copy.copy(out_shape)
    acc_shape.pop(freq_dim)
    acc = np.empty(acc_shape, dtype=np.complex128)

    # create list for frequencies used and frequency bands
    # of resulting connectivity data
    freqs = list()
    freq_bands = list()
    idx_fi = [slice(None)] * len(out_shape)
    idx_fj = [slice(None)] * len(out_shape)
    for band_idx, band in enumerate(bands):
        freq_idx = np.where((freqs_ > band[0]) & (freqs_ < band[1]))[0]
        freqs.append(freqs_[freq_idx])
        freq_bands.append(np.mean(freqs_[freq_idx]))

        acc.fill(0.0)
        for fi, fj in zip(freq_idx, freq_idx[1:]):
            idx_fi[freq_dim] = fi
            idx_fj[freq_dim] = fj
            acc += (
                np.conj(cohy.get_data()[tuple(idx_fi)]) * cohy.get_data()[tuple(idx_fj)]
            )

        idx_fi[freq_dim] = band_idx
        psi[tuple(idx_fi)] = np.imag(acc)
    logger.info("[PSI Estimation Done]")

    # create a connectivity container
    if isinstance(cohy, SpectralConnectivity):
        # spectral only
        conn = SpectralConnectivity(
            data=psi,
            names=names,
            freqs=freq_bands,
            n_nodes=n_nodes,
            method="phase-slope-index",
            spec_method=mode,
            indices=indices,
            freqs_computed=freqs,
            n_epochs_used=n_epochs_used,
            n_tapers=n_tapers,
            metadata=metadata,
            events=events,
            event_id=event_id,
        )
    else:
        # spectrotemporal
        conn = SpectroTemporalConnectivity(
            data=psi,
            names=names,
            freqs=freq_bands,
            times=times,
            n_nodes=n_nodes,
            method="phase-slope-index",
            spec_method=mode,
            indices=indices,
            freqs_computed=freqs,
            n_epochs_used=n_epochs_used,
            n_tapers=n_tapers,
            metadata=metadata,
            events=events,
            event_id=event_id,
        )

    return conn
