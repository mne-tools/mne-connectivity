import numpy as np
from scipy.signal import fftconvolve

from mne.utils import logger


def _create_kernel(sm_times, sm_freqs, kernel='hanning'):
    """2D (freqs, time) smoothing kernel.

    Parameters
    ----------
    sm_times : int, array_like
        Number of points to consider for the temporal smoothing,
        if it is an array it will be considered that the kernel
        if frequence dependent.
    sm_freqs : int
        Number of points to consider for the frequency smoothing
    kernel : {'square', 'hanning'}
        Kernel type to use. Choose either 'square' or 'hanning'

    Returns
    -------
    kernel : array_like
        Smoothing kernel of shape (sm_freqs, sm_times)
    """
    scale = isinstance(sm_times, np.ndarray)

    if scale:
        # I know this piece of code is terrible ='D
        logger.info("For frequency dependent kernel sm_freqs is not used"
                    "")
        # Number of kernels
        n_kernel = len(sm_times)
        # Get the size of the biggest kernel
        max_size = sm_times.max()
        # Container for the padded kernel
        s_pad = np.zeros((n_kernel, max_size), dtype=np.float32)
        # Store kernel for each frequency
        s = []

        def __pad_kernel(s):
            for i in range(n_kernel):
                #  print(f"{s[i]}")
                pad_size = int(max_size - len(s[i]))
                # The len(s[i])%2 corrects in case the len is odd
                s_pad[i, :] = np.pad(
                    s[i], (pad_size // 2, pad_size // 2 + pad_size % 2))
            return s_pad

    if kernel == 'square':
        if not scale:
            return np.full((sm_freqs, sm_times), 1. / (sm_times * sm_freqs))
        else:
            for i in range(n_kernel):
                s += [np.ones(sm_times[i]) / sm_times[i]]
            # Pad with zeros
            return __pad_kernel(s)
    elif kernel == 'hanning':
        if not scale:
            hann_t, hann_f = np.hanning(sm_times), np.hanning(sm_freqs)
            hann = hann_f.reshape(-1, 1) * hann_t.reshape(1, -1)
            return hann / np.sum(hann)
        else:
            for i in range(n_kernel):
                hann = np.hanning(sm_times[i])
                s += [hann / np.sum(hann)]
            return __pad_kernel(s)
    else:
        raise ValueError(f"No kernel {kernel}")


def _smooth_spectra(spectra, kernel, scale=False, decim=1):
    """Smoothing spectra.

    This function assumes that the frequency and time axis are respectively
    located at positions (..., freqs, times).

    Parameters
    ----------
    spectra : array_like
        Spectra of shape (..., n_freqs, n_times)
    kernel : array_like
        Smoothing kernel of shape (sm_freqs, sm_times)
    decim : int | 1
        Decimation factor to apply after the kernel smoothing

    Returns
    -------
    sm_spectra : array_like
        Smoothed spectra of shape (..., n_freqs, n_times)
    """
    # fill potentially missing dimensions
    kernel = kernel[
        tuple([np.newaxis] * (spectra.ndim - kernel.ndim)) + (Ellipsis,)]

    # smooth the spectra
    if not scale:
        axes = (-2, -1)
    else:
        axes = -1

    spectra = fftconvolve(spectra, kernel, mode='same', axes=axes)
    # return decimated spectra
    return spectra[..., ::decim]
