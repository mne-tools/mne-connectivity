# Authors: Adam Li <adam2392@gmail.com>
#          Thomas S. Binns <t.s.binns@outlook.com>
#
# License: BSD (3-clause)

import numpy as np
from mne import EpochsArray, create_info
from mne.filter import filter_data
from mne.utils import _validate_type


def make_signals_in_freq_bands(
    n_seeds,
    n_targets,
    freq_band,
    *,
    n_epochs=10,
    n_times=200,
    sfreq=100.0,
    trans_bandwidth=1.0,
    snr=0.7,
    connection_delay=5,
    connection_time=None,
    window_alpha=0.5,
    tmin=0.0,
    ch_names=None,
    ch_types="eeg",
    rng_seed=None,
):
    """Simulate signals interacting in a given frequency band.

    Parameters
    ----------
    n_seeds : int
        Number of seed channels to simulate.
    n_targets : int
        Number of target channels to simulate.
    freq_band : tuple of float
        Frequency band where the connectivity should be simulated, where the first entry
        corresponds to the lower frequency, and the second entry to the higher
        frequency.
    n_epochs : int (default 10)
        Number of epochs in the simulated data.
    n_times : int (default 200)
        Number of timepoints each epoch of the simulated data.
    sfreq : float (default 100.0)
        Sampling frequency of the simulated data, in Hz.
    trans_bandwidth : float (default 1.0)
        Transition bandwidth of the filter to apply to isolate activity in
        ``freq_band``, in Hz. These are passed to the ``l_bandwidth`` and
        ``h_bandwidth`` keyword arguments in :func:`mne.filter.create_filter`.
    snr : float (default 0.7)
        Signal-to-noise ratio of the simulated data in the range [0, 1].
    connection_delay : int (default 5)
        Number of timepoints for the delay of connectivity between the seeds and
        targets. If > 0, the target data is a delayed form of the seed data. If < 0, the
        seed data is a delayed form of the target data.
    connection_time : tuple of float | None (default None)
        Start and end times at which the interaction occurs in each epoch, in seconds.
        First entry must be greater than ``tmin``, last entry must be less than
        ``n_times / sfreq``. If a given entry is ``None``, it will be substituted for
        ``tmin`` or ``n_times / sfreq``, as appropriate. If ``None``, the connectivity
        is simulated throughout the whole epoch.

        .. versionadded:: 0.9
    window_alpha : float (default 0.5)
        Fraction of the Tukey window in the cosine tapered region, used to isolate the
        interaction to a given time in the epochs. Must be between 0 and 1. 0 is
        equivalent to a rectangular window. 1 is equivalent to a Hann window. Ignored if
        ``connection_time`` is ``None``.

        .. versionadded:: 0.9
    tmin : float (default 0.0)
        Earliest time of each epoch.
    ch_names : list of str | None (default None)
        Names of the channels in the simulated data. If ``None``, the channels are named
        according to their index and the frequency band of interaction. If specified,
        must be a list of ``n_seeds + n_targets`` channel names.
    ch_types : str | list of str (default ``'eeg'``)
        Types of the channels in the simulated data. Must be a recognised data channel
        type (see :term:`mne:data channels`). If specified as a list, must be a
        list of ``n_seeds + n_targets`` channel names.
    rng_seed : int | None (default None)
        Seed to use for the random number generator. If ``None``, no seed is specified.

    Returns
    -------
    epochs : ~mne.EpochsArray, shape (n_epochs, n_seeds + n_targets, n_times)
        The simulated data stored in an :class:`mne.EpochsArray` object. The channels
        are arranged according to seeds, then targets.

    Notes
    -----
    Signals are simulated as a single source of activity in the given frequency band and
    projected into ``n_seeds + n_targets`` white noise channels.

    By default, connectivity is simulated throughout the whole epoch. However, the
    interaction can be isolated to a given time range using the ``connection_time`` and
    ``window_alpha`` parameters, mimicking an evoked potential. E.g., to simulate a
    burst of activity with a 200 ms duration beginning at the start of each epoch::

        epochs = make_signals_in_freq_bands(
            ...,
            tmin=-0.5,  # include a pre-trial period (optional)
            connection_time=(0.0, 0.2),  # burst from 0 to 200 ms
            window_alpha=0.5,  # isolate burst with 50% cosine-tapered Tukey window
        )
    """
    from scipy.signal.windows import tukey

    n_channels = n_seeds + n_targets

    # check inputs
    if n_seeds < 1 or n_targets < 1:
        raise ValueError("Number of seeds and targets must each be at least 1.")

    _validate_type(freq_band, tuple, "freq_band")
    if len(freq_band) != 2:
        raise ValueError("Frequency band must contain two numbers.")

    if n_times < 1:
        raise ValueError("Number of timepoints must be at least 1.")

    if n_epochs < 1:
        raise ValueError("Number of epochs must be at least 1.")

    if sfreq <= 0:
        raise ValueError("Sampling frequency must be > 0.")

    if snr < 0 or snr > 1:
        raise ValueError("Signal-to-noise ratio must be between 0 and 1.")

    if np.abs(connection_delay) >= n_epochs * n_times:
        raise ValueError(
            "Connection delay must be less than the total number of timepoints."
        )

    _validate_type(connection_time, (tuple, None), "`connection_time`")
    epoch_dur = n_times / sfreq
    tmax = tmin + epoch_dur
    if connection_time is not None:
        connection_time = list(connection_time)  # convert to list to allow modification
        if len(connection_time) != 2:
            raise ValueError("`connection_time` must be a tuple with length 2.")
        for time_idx, time in enumerate(connection_time):
            _validate_type(time, ("numeric", None), "`connection_time` entries")
            if time is None:
                connection_time[time_idx] = tmin if time_idx == 0 else tmax
        if connection_time[0] < tmin or connection_time[1] > tmax:
            raise ValueError(
                f"`connection_time` {connection_time} must be within the epoch time "
                f"range [{tmin}, {tmax}]."
            )
        if connection_time[0] >= connection_time[1]:
            raise ValueError(
                "Start time of `connection_time` must be less than the end time."
            )
        connection_time = np.array(connection_time)

    _validate_type(window_alpha, float, "window_alpha")
    if window_alpha < 0 or window_alpha > 1:
        raise ValueError("`window_alpha` must be between 0 and 1.")

    # simulate data
    rng = np.random.default_rng(rng_seed)

    # simulate signal source at desired frequency band
    signal = rng.standard_normal(
        size=(1, n_epochs * n_times + np.abs(connection_delay))
    )
    signal = filter_data(
        data=signal,
        sfreq=sfreq,
        l_freq=freq_band[0],
        h_freq=freq_band[1],
        l_trans_bandwidth=trans_bandwidth,
        h_trans_bandwidth=trans_bandwidth,
        fir_design="firwin2",
    )

    # isolate interaction to a given time range per epoch
    if connection_time is not None:
        connection_time -= tmin  # convert time to be relative to time of first sample
        time_samples = (connection_time * sfreq).astype(int)  # convert time to samples
        full_window = np.zeros((n_times,))  # create empty filter the shape of one epoch

        # add Tukey window for burst of connectivity to temporal filter
        burst_window = tukey(np.squeeze(np.diff(time_samples)), window_alpha, sym=False)
        full_window[time_samples[0] : time_samples[1]] = burst_window

        # apply temporal filter to each epoch to create bursts of connectivity
        for epoch_i in range(n_epochs):
            signal[:, epoch_i * n_times : (epoch_i + 1) * n_times] *= full_window
            if epoch_i == n_epochs - 1 and connection_delay != 0:
                signal[:, (epoch_i + 1) * n_times :] *= full_window[
                    : np.abs(connection_delay)
                ]

    # simulate noise for each channel
    noise = rng.standard_normal(size=(n_channels, signal.shape[1]))

    # create data by projecting signal into each channel of noise
    data = (signal * snr) + (noise * (1 - snr))

    # shift data by desired delay and remove extra time
    if connection_delay != 0:
        if connection_delay > 0:
            delay_chans = np.arange(n_seeds, n_channels)  # delay targets
        else:
            delay_chans = np.arange(0, n_seeds)  # delay seeds
        data[delay_chans, np.abs(connection_delay) :] = data[
            delay_chans, : n_epochs * n_times
        ]
        data = data[:, : n_epochs * n_times]

    # reshape data into epochs
    data = data.reshape(n_channels, n_epochs, n_times)
    data = data.transpose((1, 0, 2))  # (epochs x channels x times)

    # store data in an MNE EpochsArray object
    if ch_names is None:
        ch_names = [
            f"{ch_i}_{freq_band[0]}_{freq_band[1]}" for ch_i in range(n_channels)
        ]
    info = create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    epochs = EpochsArray(data=data, info=info, tmin=tmin)

    return epochs
