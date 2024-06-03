# Authors: Thomas S. Binns <t.s.binns@outlook.com>
#
# License: BSD (3-clause)

from typing import Optional

import numpy as np
from mne import Info
from mne.decoding.mixin import TransformerMixin
from mne.fixes import BaseEstimator
from mne.time_frequency import csd_array_fourier, csd_array_morlet, csd_array_multitaper
from mne.utils import _check_option, _validate_type

from ..spectral.epochs_multivariate import (
    _CaCohEst,
    _check_rank_input,
    _EpochMeanMultivariateConEstBase,
    _MICEst,
)
from ..utils import _check_multivariate_indices, fill_doc


class _AbstractDecompositionBase(BaseEstimator, TransformerMixin):
    """ABC for multivariate connectivity signal decomposition."""

    filters_: Optional[tuple] = None
    patterns_: Optional[tuple] = None

    _indices: Optional[tuple] = None
    _rank: Optional[tuple] = None
    _conn_estimator: Optional[_EpochMeanMultivariateConEstBase] = None

    @property
    def indices(self):
        """Get ``indices`` parameter in the input format.

        :meta private:
        """
        return (self._indices[0].compressed(), self._indices[1].compressed())

    @indices.setter
    def indices(self, indices):
        """Set ``indices`` parameter using the input format."""
        self._indices = (np.array([indices[0]]), np.array([indices[1]]))

    @property
    def rank(self):
        """Get ``rank`` parameter in the input format.

        :meta private:
        """
        return (self._rank[0][0], self._rank[1][0])

    @rank.setter
    def rank(self, rank):
        """Set ``rank`` parameter using the input format."""
        self._rank = ([rank[0]], [rank[1]])

    def __init__(
        self,
        info,
        fmin,
        fmax,
        indices,
        mode="multitaper",
        mt_bandwidth=None,
        mt_adaptive=False,
        mt_low_bias=True,
        cwt_freq_resolution=1,
        cwt_n_cycles=7,
        n_components=None,
        rank=None,
        n_jobs=1,
        verbose=None,
    ):
        """Initialise instance."""
        # Validate inputs
        _validate_type(info, Info, "`info`", "mne.Info")

        _validate_type(fmin, (int, float), "`fmin`", "int or float")
        _validate_type(fmax, (int, float), "`fmax`", "int or float")
        if fmin > fmax:
            raise ValueError("`fmax` must be larger than `fmin`")
        if fmax > info["sfreq"] / 2:
            raise ValueError("`fmax` cannot be larger than the Nyquist frequency")

        _validate_type(indices, tuple, "`indices`", "tuple of array-like")
        if len(indices) != 2:
            raise ValueError("`indices` must be have length 2")
        for indices_group in indices:
            _validate_type(
                indices_group,
                (list, tuple, np.ndarray),
                "`indices`",
                "tuple of array-likes",
            )
        _indices = self._check_indices(indices, info["nchan"])

        _check_option("mode", mode, ("multitaper", "fourier", "cwt_morlet"))

        _validate_type(
            mt_bandwidth, (int, float, None), "`mt_bandwidth`", "int, float, or None"
        )
        _validate_type(mt_adaptive, bool, "`mt_adaptive`", "bool")
        _validate_type(mt_low_bias, bool, "`mt_low_bias`", "bool")

        _validate_type(
            cwt_freq_resolution, (int, float), "`cwt_freq_resolution`", "int or float"
        )
        _validate_type(
            cwt_n_cycles,
            (int, float, tuple, list, np.ndarray),
            "`cwt_n_cycles`",
            "int, float, or array-like of ints or floats",
        )

        _validate_type(n_components, (int, None), "`n_components`", "int or None")

        _validate_type(rank, (tuple, None), "`rank`", "tuple of ints or None")
        if rank is not None:
            if len(rank) != 2:
                raise ValueError("`rank` must be have length 2")
            for rank_group in rank:
                _validate_type(rank_group, int, "`rank`", "tuple of ints or None")
        _rank = self._check_rank(rank, indices)

        _validate_type(n_jobs, int, "`n_jobs`", "int")

        _validate_type(
            verbose, (bool, str, int, None), "`verbose`", "bool, str, int, or None"
        )

        # Store inputs
        self.info = info
        self.fmin = fmin
        self.fmax = fmax
        self._indices = _indices  # uses getter/setter for public parameter
        self.mode = mode
        self.mt_bandwidth = mt_bandwidth
        self.mt_adaptive = mt_adaptive
        self.mt_low_bias = mt_low_bias
        self.cwt_freq_resolution = cwt_freq_resolution
        self.cwt_n_cycles = cwt_n_cycles
        self.n_components = 1  # XXX: fixed until n_comps > 1 supported
        self._rank = _rank  # uses getter/setter for public parameter
        self.n_jobs = n_jobs
        self.verbose = verbose

    def _check_indices(self, indices, n_chans):
        """Check that the indices input is valid."""
        # convert to multivariate format and check validity
        indices = _check_multivariate_indices(([indices[0]], [indices[1]]), n_chans)

        # find whether entries of indices exceed number of channels
        max_idx = np.max(indices.compressed())
        if max_idx + 1 > n_chans:
            raise ValueError(
                "At least one entry in `indices` is greater than the number of "
                "channels in `info`"
            )

        return indices

    def _check_rank(self, rank, indices):
        """Check that the rank input is valid."""
        if rank is not None:
            # convert to multivariate format
            rank = ([rank[0]], [rank[1]])

            # find whether entries of rank exceed number of channels in indices
            if rank[0][0] > len(indices[0]) or rank[1][0] > len(indices[1]):
                raise ValueError(
                    "At least one entry in `rank` is greater than the number of "
                    "seed/target channels in `indices`"
                )

        return rank

    def fit(self, X, y=None):
        """Compute connectivity decomposition filters for epoched data.

        Parameters
        ----------
        X : array, shape=(n_epochs, n_signals, n_times)
            The input data which the connectivity decomposition filters should be fit
            to.
        y : None
            Used for scikit-learn compatibility.

        Returns
        -------
        self : instance of CaCoh | MIC
            The modified class instance.
        """
        # validate input data
        self._check_X(X, ndim=[3])
        self._get_rank_and_ncomps_from_X(X)

        # compute CSD
        csd = self._compute_csd(X)

        # instantiate connectivity estimator and add CSD information
        self._conn_estimator = self._conn_estimator(
            n_signals=X.shape[1],
            n_cons=1,
            n_freqs=1,
            n_times=0,
            n_jobs=self.n_jobs,
            store_filters=True,
        )
        self._conn_estimator.accumulate(con_idx=np.arange(csd.shape[0]), csd_xy=csd)

        # fit filters to data and compute corresponding patterns
        self._conn_estimator.compute_con(
            indices=self._indices, ranks=self._rank, n_epochs=1
        )

        # extract filters and patterns
        self._extract_filters_and_patterns()

        return self

    def _check_X(self, X, ndim):
        """Check that the input data is valid."""
        # check data is a 2/3D array
        _validate_type(X, np.ndarray, "`X`", "NumPy array")
        _check_option("`X.ndim`", X.ndim, ndim)
        n_chans = X.shape[1]
        if n_chans != self.info["nchan"]:
            raise ValueError(
                "`X` does not match Info\nExpected %i channels, got %i"
                % (n_chans, self.info["nchan"])
            )

    def _get_rank_and_ncomps_from_X(self, X):
        """Get/validate rank and n_components parameters using the data."""
        # compute rank from data if necessary / check it is valid for the indices
        self._rank = _check_rank_input(self._rank, X, self._indices)

        # set n_components if necessary / check it is valid for the rank
        if self.n_components is None:
            self.n_components = np.min(self.rank)
        elif self.n_components > np.min(self.rank):
            raise ValueError(
                "`n_components` is greater than the minimum rank of the data"
            )

    def _compute_csd(self, X):
        """Compute the cross-spectral density of the input data."""
        # XXX: fix csd returning [fmin +1 bin to fmax -1 bin] frequencies
        csd_kwargs = {
            "X": X,
            "sfreq": self.info["sfreq"],
            "n_jobs": self.n_jobs,
        }
        if self.mode == "multitaper":
            csd_kwargs.update(
                {
                    "fmin": self.fmin,
                    "fmax": self.fmax,
                    "bandwidth": self.mt_bandwidth,
                    "adaptive": self.mt_adaptive,
                    "low_bias": self.mt_low_bias,
                }
            )
            csd = csd_array_multitaper(**csd_kwargs)
        elif self.mode == "fourier":
            csd_kwargs.update({"fmin": self.fmin, "fmax": self.fmax})
            csd = csd_array_fourier(**csd_kwargs)
        else:
            csd_kwargs.update(
                {
                    "frequencies": np.arange(
                        self.fmin,
                        self.fmax + self.cwt_freq_resolution,
                        self.cwt_freq_resolution,
                    ),
                    "n_cycles": self.cwt_n_cycles,
                }
            )
            csd = csd_array_morlet(**csd_kwargs)

        csd = csd.sum(self.fmin, self.fmax).get_data(index=0)
        csd = np.reshape(csd, csd.shape[0] ** 2)

        return np.expand_dims(csd, 1)

    def _extract_filters_and_patterns(self):
        """Extract filters and patterns from the connectivity estimator."""
        # XXX: need to sort indices and transpose patterns when multiple comps supported
        self.filters_ = (
            self._conn_estimator.filters[0, 0, : len(self.indices[0]), 0],
            self._conn_estimator.filters[1, 0, : len(self.indices[1]), 0],
        )

        self.patterns_ = (
            self._conn_estimator.patterns[0, 0, : len(self.indices[0]), 0],
            self._conn_estimator.patterns[1, 0, : len(self.indices[1]), 0],
        )

        # XXX: remove once support for multiple comps implemented
        self.filters_ = (
            np.expand_dims(self.filters_[0], 1),
            np.expand_dims(self.filters_[1], 1),
        )
        self.patterns_ = (
            np.expand_dims(self.patterns_[0], 0),
            np.expand_dims(self.patterns_[1], 0),
        )

    def transform(self, X):
        """Decompose data into connectivity sources using the fitted filters.

        Parameters
        ----------
        X : array, shape=((n_epochs, ) n_signals, n_times)
            The data to be transformed by the connectivity decomposition filters.

        Returns
        -------
        X_transformed : array, shape=((n_epochs, ) n_components*2, n_times)
            The transformed data. The first ``n_components`` channels are the
            transformed seeds, and the last ``n_components`` channels are the
            transformed targets.
        """
        self._check_X(X, ndim=(2, 3))
        if self.filters_ is None:
            raise RuntimeError(
                "no filters are available, please call the `fit` method first"
            )

        # transform seed and target data
        X_seeds = self.filters_[0].T @ X[..., self.indices[0], :]
        X_targets = self.filters_[1].T @ X[..., self.indices[1], :]

        return np.concatenate((X_seeds, X_targets), axis=-2)

    def fit_transform(self, X, y=None, **fit_params):
        """Fit filters to data, then transform and return it.

        Parameters
        ----------
        X : array, shape=(n_epochs, n_signals, n_times)
            The input data which the connectivity decomposition filters should be fit to
            and subsequently transformed.
        y : None
            Used for scikit-learn compatibility.
        **fit_params : dict
            Additional fitting parameters passed to the ``fit`` method. Not used for
            this class.

        Returns
        -------
        X_transformed : array, shape=(n_epochs, n_components*2, n_times)
            The transformed data. The first ``n_components`` channels are the
            transformed seeds, and the last ``n_components`` channels are the
            transformed targets.
        """
        # use parent TransformerMixin method but with custom docstring

    def get_transformed_indices(self):
        """Get indices for the transformed data.

        Returns
        -------
        indices_transformed : tuple of array
            Indices of seeds and targets in the transformed data with the form (seeds,
            targets) to be used when passing the data to
            `~mne_connectivity.spectral_connectivity_epochs` and
            `~mne_connectivity.spectral_connectivity_time`. Entries of the indices are
            arranged such that connectivity would be computed between the first seed
            component and first target component, second seed component and second
            target component, etc...
        """
        return (
            np.arange(self.n_components),
            np.arange(self.n_components) + self.n_components,
        )


@fill_doc
class CaCoh(_AbstractDecompositionBase):
    """Decompose connectivity sources using canonical coherency (CaCoh).

    CaCoh is a multivariate approach to maximise coherency/coherence between a set of
    seed and target signals in a frequency-resolved manner :footcite:`VidaurreEtAl2019`.
    The maximisation of connectivity involves fitting spatial filters to the
    cross-spectral density of the seed and target data, alongisde which spatial patterns
    of the contributions to connectivity can be computed :footcite:`HaufeEtAl2014`.

    Once fit, the filters can be used to transform data into the underlying connectivity
    components. Connectivity can be computed on this transformed data using the
    ``"coh"`` and ``"cohy"`` methods of the
    `mne_connectivity.spectral_connectivity_epochs` and
    `mne_connectivity.spectral_connectivity_time` functions.

    The approach taken here is to optimise the connectivity in a given frequency band.
    Frequency bin-wise optimisation is offered in the ``"cacoh"`` method of the
    `mne_connectivity.spectral_connectivity_epochs` and
    `mne_connectivity.spectral_connectivity_time` functions.

    Parameters
    ----------
    %(info_decoding)s
    %(fmin_decoding)s
    %(fmax_decoding)s
    %(indices_decoding)s
    %(mode)s
    %(mt_bandwidth)s
    %(mt_adaptive)s
    %(mt_low_bias)s
    %(cwt_freq_resolution)s
    %(cwt_n_cycles)s
    %(n_components)s
    %(rank)s
    %(n_jobs)s
    %(verbose)s

    Attributes
    ----------
    %(filters_)s
    %(patterns_)s

    References
    ----------
    .. footbibliography::
    """

    _conn_estimator = _CaCohEst


@fill_doc
class MIC(_AbstractDecompositionBase):
    """Decompose connectivity sources using maximised imaginary coherency (MIC).

    MIC is a multivariate approach to maximise the imaginary part of coherency between a
    set of seed and target signals in a frequency-resolved manner
    :footcite:`EwaldEtAl2012`. The maximisation of connectivity involves fitting spatial
    filters to the cross-spectral density of the seed and target data, alongisde which
    spatial patterns of the contributions to connectivity can be computed
    :footcite:`HaufeEtAl2014`.

    Once fit, the filters can be used to transform data into the underlying connectivity
    components. Connectivity can be computed on this transformed data using the
    ``"imcoh"`` method of the `mne_connectivity.spectral_connectivity_epochs` and
    `mne_connectivity.spectral_connectivity_time` functions.

    The approach taken here is to optimise the connectivity in a given frequency band.
    Frequency bin-wise optimisation is offered in the ``"mic"`` method of the
    `mne_connectivity.spectral_connectivity_epochs` and
    `mne_connectivity.spectral_connectivity_time` functions.

    Parameters
    ----------
    %(info_decoding)s
    %(fmin_decoding)s
    %(fmax_decoding)s
    %(indices_decoding)s
    %(mode)s
    %(mt_bandwidth)s
    %(mt_adaptive)s
    %(mt_low_bias)s
    %(cwt_freq_resolution)s
    %(cwt_n_cycles)s
    %(n_components)s
    %(rank)s
    %(n_jobs)s
    %(verbose)s

    Attributes
    ----------
    %(filters_)s
    %(patterns_)s

    References
    ----------
    .. footbibliography::
    """

    _conn_estimator = _MICEst
