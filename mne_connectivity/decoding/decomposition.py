# Authors: Thomas S. Binns <t.s.binns@outlook.com>
#          Marijn van Vliet <w.m.vanvliet@gmail.com>
#          Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD (3-clause)


from sklearn.base import TransformerMixin

import numpy as np
from mne import Info
from mne._fiff.pick import pick_info
from mne.defaults import _BORDER_DEFAULT, _EXTRAPOLATE_DEFAULT, _INTERPOLATION_DEFAULT
from mne.evoked import EvokedArray
from mne.fixes import BaseEstimator
from mne.time_frequency import csd_array_fourier, csd_array_morlet, csd_array_multitaper
from mne.utils import _check_option, _validate_type
from mne.viz.utils import plt_show

from ..spectral.epochs_multivariate import (
    _CaCohEst,
    _check_n_components_input,
    _check_rank_input,
    _MICEst,
)
from ..utils import _check_multivariate_indices, fill_doc


@fill_doc
class CoherencyDecomposition(BaseEstimator, TransformerMixin):
    """Decompose connectivity sources using multivariate coherency-based methods.

    Parameters
    ----------
    %(info_decoding)s
    %(method_decoding)s
    %(indices_decoding)s
    %(mode)s
    %(fmin_decoding)s
    %(fmax_decoding)s
    %(mt_bandwidth)s
    %(mt_adaptive)s
    %(mt_low_bias)s
    %(cwt_freqs)s
    %(cwt_n_cycles)s
    %(n_components)s
    %(rank)s
    %(n_jobs)s
    %(verbose)s

    Attributes
    ----------
    %(filters_)s
    %(patterns_)s

    Notes
    -----
    The multivariate methods maximise connectivity between a set of seed and target
    signals in a frequency-resolved manner. The maximisation of connectivity involves
    fitting spatial filters to the cross-spectral density of the seed and target data,
    alongside which spatial patterns of the contributions to connectivity can be
    computed :footcite:`HaufeEtAl2014`.

    Once fit, the filters can be used to transform data into the underlying connectivity
    components. Connectivity can be computed on this transformed data using the
    bivariate coherency-based methods of the
    `mne_connectivity.spectral_connectivity_epochs` and
    `mne_connectivity.spectral_connectivity_time` functions. These bivariate methods
    are:

    * ``"cohy"`` and ``"coh"`` for CaCoh :footcite:`VidaurreEtAl2019`
    * ``"imcoh"`` for MIC :footcite:`EwaldEtAl2012`

    The approach taken here is to optimise the connectivity in a given frequency band.
    Frequency bin-wise optimisation is offered in the multivariate coherency-based
    methods of the `mne_connectivity.spectral_connectivity_epochs` and
    `mne_connectivity.spectral_connectivity_time` functions.

    References
    ----------
    .. footbibliography::
    """

    filters_: tuple | None = None
    patterns_: tuple | None = None

    _conn_estimator: _CaCohEst | _MICEst | None = None
    _indices: tuple | None = None
    _rank: tuple | None = None

    @property
    def indices(self):
        """Get ``indices`` parameter in the input format.

        :meta private:
        """
        return (self._indices[0].compressed(), self._indices[1].compressed())

    @indices.setter
    def indices(self, indices):
        """Set ``indices`` parameter using the input format."""
        self._indices = _check_multivariate_indices(
            ([indices[0]], [indices[1]]), self.info["nchan"]
        )

    @property
    def rank(self):
        """Get ``rank`` parameter in the input format.

        :meta private:
        """
        if self._rank is not None:
            return (self._rank[0][0], self._rank[1][0])
        return None

    @rank.setter
    def rank(self, rank):
        """Set ``rank`` parameter using the input format."""
        if rank is None:
            self._rank = None
        else:
            self._rank = ([rank[0]], [rank[1]])

    def __init__(
        self,
        info,
        method,
        indices,
        mode="multitaper",
        fmin=None,
        fmax=None,
        mt_bandwidth=None,
        mt_adaptive=False,
        mt_low_bias=True,
        cwt_freqs=None,
        cwt_n_cycles=7,
        n_components=None,
        rank=None,
        n_jobs=1,
        verbose=None,
    ):
        """Initialise instance."""
        # Validate inputs
        _validate_type(info, Info, "`info`", "mne.Info")

        _check_option("method", method, ("cacoh", "mic"))
        if method == "cacoh":
            _conn_estimator_class = _CaCohEst
        else:
            _conn_estimator_class = _MICEst

        _validate_type(indices, tuple, "`indices`", "tuple of array-likes")
        if len(indices) != 2:
            raise ValueError("`indices` must have length 2")
        for indices_group in indices:
            _validate_type(
                indices_group, "array-like", "`indices`", "tuple of array-likes"
            )
        _indices = self._check_indices(indices, info["nchan"])

        _check_option("mode", mode, ("multitaper", "fourier", "cwt_morlet"))
        if mode in ["multitaper", "fourier"]:
            if fmin is None or fmax is None:
                raise TypeError(
                    "`fmin` and `fmax` must not be None if `mode` is 'multitaper' or "
                    "'fourier'"
                )
            _validate_type(fmin, "numeric", "`fmin`", "int or float")
            _validate_type(fmax, "numeric", "`fmax`", "int or float")
            if fmin > fmax:
                raise ValueError("`fmax` must be larger than `fmin`")
            if fmin < 0:
                raise ValueError("`fmin` cannot be less than 0")
            if fmax > info["sfreq"] / 2:
                raise ValueError("`fmax` cannot be larger than the Nyquist frequency")
            if mode == "multitaper":
                _validate_type(
                    mt_bandwidth,
                    ("numeric", None),
                    "`mt_bandwidth`",
                    "int, float, or None",
                )
                _validate_type(mt_adaptive, bool, "`mt_adaptive`", "bool")
                _validate_type(mt_low_bias, bool, "`mt_low_bias`", "bool")
        else:
            if cwt_freqs is None:
                raise TypeError(
                    "`cwt_freqs` must not be None if `mode` is 'cwt_morlet'"
                )
            _validate_type(cwt_freqs, "array-like", "`cwt_freqs`", "array-like")
            if cwt_freqs[-1] > info["sfreq"] / 2:
                raise ValueError(
                    "last entry of `cwt_freqs` cannot be larger than the Nyquist "
                    "frequency"
                )
            _validate_type(
                cwt_n_cycles,
                ("numeric", "array-like"),
                "`cwt_n_cycles`",
                "int, float, or array-like",
            )
            if isinstance(cwt_n_cycles, tuple | list | np.ndarray) and len(
                cwt_n_cycles
            ) != len(cwt_freqs):
                raise ValueError(
                    "`cwt_n_cycles` array-like must have the same length as `cwt_freqs`"
                )

        _validate_type(
            n_components, ("int-like", None), "`n_components`", "int or None"
        )

        _validate_type(rank, (tuple, None), "`rank`", "tuple of ints or None")
        if rank is not None:
            if len(rank) != 2:
                raise ValueError("`rank` must have length 2")
            for rank_group in rank:
                _validate_type(
                    rank_group, "int-like", "`rank`", "tuple of ints or None"
                )
        _rank = self._check_rank(rank, indices)

        # n_jobs and verbose will be checked downstream

        # Store inputs
        self.info = info
        self._conn_estimator_class = _conn_estimator_class
        self._indices = _indices  # uses getter/setter for public parameter
        self.mode = mode
        self.fmin = fmin
        self.fmax = fmax
        self.mt_bandwidth = mt_bandwidth
        self.mt_adaptive = mt_adaptive
        self.mt_low_bias = mt_low_bias
        self.cwt_freqs = cwt_freqs
        self.cwt_n_cycles = cwt_n_cycles
        self.n_components = n_components
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
                "at least one entry in `indices` is greater than the number of "
                "channels in `info`"
            )

        return indices

    def _check_rank(self, rank, indices):
        """Check that the rank input is valid."""
        if rank is not None:
            # convert to multivariate format
            rank = ([rank[0]], [rank[1]])

            # make sure ranks are > 0
            if np.any(np.array(rank) <= 0):
                raise ValueError("entries of `rank` must be > 0")

            # find whether entries of rank exceed number of channels in indices
            if rank[0][0] > len(indices[0]) or rank[1][0] > len(indices[1]):
                raise ValueError(
                    "at least one entry in `rank` is greater than the number of "
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
            Ignored; exists for compatibility with scikit-learn pipelines.

        Returns
        -------
        self : instance of CoherencyDecomposition
            The modified class instance.
        """
        # validate input data
        self._check_X(X, ndim=[3])
        self._get_rank_and_ncomps_from_X(X)

        # compute CSD
        csd = self._compute_csd(X)

        # instantiate connectivity estimator and add CSD information
        self._conn_estimator = self._conn_estimator_class(
            n_signals=X.shape[1],
            n_cons=1,
            n_freqs=1,
            n_times=0,
            n_components=self.n_components,
            store_con=False,
            store_filters=True,
            n_jobs=self.n_jobs,
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
        n_chans = X.shape[-2]
        if n_chans != self.info["nchan"]:
            raise ValueError(
                f"`X` does not match Info\nExpected {n_chans} channels, got "
                f"{self.info['nchan']}"
            )

    def _get_rank_and_ncomps_from_X(self, X):
        """Get/validate rank and n_components parameters using the data."""
        # compute rank from data if necessary / check it is valid for the indices
        self._rank = _check_rank_input(self._rank, X, self._indices)

        # check n_components is valid for the rank
        self.n_components = _check_n_components_input(self.n_components, self._rank)

    def _compute_csd(self, X):
        """Compute the cross-spectral density of the input data."""
        csd_kwargs = {
            "X": X,
            "sfreq": self.info["sfreq"],
            "n_jobs": self.n_jobs,
            "verbose": self.verbose,
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
                {"frequencies": self.cwt_freqs, "n_cycles": self.cwt_n_cycles}
            )
            csd = csd_array_morlet(**csd_kwargs)

        if self.mode in ["multitaper", "fourier"]:
            fmin = self.fmin
            fmax = self.fmax
        else:
            fmin = self.cwt_freqs[0]
            fmax = self.cwt_freqs[-1]
        csd = csd.sum(fmin, fmax).get_data(index=0)
        csd = np.reshape(csd, csd.shape[0] ** 2)

        return np.expand_dims(csd, 1)

    def _extract_filters_and_patterns(self):
        """Extract filters and patterns from the connectivity estimator."""
        # shape=(seeds/targets, n_cons, n_components, n_signals, n_freqs)
        # i.e. (2, 1, n_components, n_signals, 1)

        self.filters_ = (
            self._conn_estimator.filters[0, 0, :, : len(self.indices[0]), 0].T,
            self._conn_estimator.filters[1, 0, :, : len(self.indices[1]), 0].T,
        )  # shape=(seeds/targets, n_signals, n_components)

        self.patterns_ = (
            self._conn_estimator.patterns[0, 0, :, : len(self.indices[0]), 0],
            self._conn_estimator.patterns[1, 0, :, : len(self.indices[1]), 0],
        )  # shape=(seeds/targets, n_components, n_signals)

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
            Ignored; exists for compatibility with scikit-learn pipelines.
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
        return super().fit_transform(X, y=y, **fit_params)

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
    def plot_patterns(
        self,
        info,
        components=None,
        ch_type=None,
        scalings=None,
        sensors=True,
        show_names=False,
        mask=None,
        mask_params=None,
        contours=6,
        outlines="head",
        sphere=None,
        image_interp=_INTERPOLATION_DEFAULT,
        extrapolate=_EXTRAPOLATE_DEFAULT,
        border=_BORDER_DEFAULT,
        res=64,
        size=1,
        cmap="RdBu_r",
        vlim=(None, None),
        cnorm=None,
        colorbar=True,
        cbar_fmt="%.1E",
        units="AU",
        axes=None,
        name_format=None,
        nrows=1,
        ncols="auto",
        show=True,
    ):
        """Plot topographic patterns of components.

        The patterns explain how the measured data was generated from the
        neural sources (a.k.a. the forward model) :footcite:`HaufeEtAl2014`.

        Seed and target patterns are plotted separately.

        Parameters
        ----------
        %(info_decoding_plotting)s
        %(components_topomap)s
        %(ch_type_topomap)s
        %(scalings_topomap)s
        %(sensors_topomap)s
        %(show_names_topomap)s
        %(mask_patterns_topomap)s
        %(mask_params_topomap)s
        %(contours_topomap)s
        %(outlines_topomap)s
        %(sphere_topomap)s
        %(image_interp_topomap)s
        %(extrapolate_topomap)s
        %(border_topomap)s
        %(res_topomap)s
        %(size_topomap)s
        %(cmap_topomap)s
        %(vlim_topomap)s
        %(cnorm_topomap)s
        %(colorbar_topomap)s
        %(colorbar_format_topomap)s
        %(units_topomap)s
        %(axes_topomap)s
        %(name_format_topomap)s
        %(nrows_topomap)s
        %(ncols_topomap)s
        %(show)s

        Returns
        -------
        %(figs_topomap)s
        """
        if self.patterns_ is None:
            raise RuntimeError(
                "no patterns are available, please call the `fit` method first"
            )

        return self._plot_filters_patterns(
            (self.patterns_[0].T, self.patterns_[1].T),
            info,
            components,
            ch_type,
            scalings,
            sensors,
            show_names,
            mask,
            mask_params,
            contours,
            outlines,
            sphere,
            image_interp,
            extrapolate,
            border,
            res,
            size,
            cmap,
            vlim,
            cnorm,
            colorbar,
            cbar_fmt,
            units,
            axes,
            name_format,
            nrows,
            ncols,
            show,
        )

    @fill_doc
    def plot_filters(
        self,
        info,
        components=None,
        ch_type=None,
        scalings=None,
        sensors=True,
        show_names=False,
        mask=None,
        mask_params=None,
        contours=6,
        outlines="head",
        sphere=None,
        image_interp=_INTERPOLATION_DEFAULT,
        extrapolate=_EXTRAPOLATE_DEFAULT,
        border=_BORDER_DEFAULT,
        res=64,
        size=1,
        cmap="RdBu_r",
        vlim=(None, None),
        cnorm=None,
        colorbar=True,
        cbar_fmt="%.1E",
        units="AU",
        axes=None,
        name_format=None,
        nrows=1,
        ncols="auto",
        show=True,
    ):
        """Plot topographic filters of components.

        The filters are used to extract discriminant neural sources from the measured
        data (a.k.a. the backward model). :footcite:`HaufeEtAl2014`.

        Seed and target filters are plotted separately.

        Parameters
        ----------
        %(info_decoding_plotting)s
        %(components_topomap)s
        %(ch_type_topomap)s
        %(scalings_topomap)s
        %(sensors_topomap)s
        %(show_names_topomap)s
        %(mask_filters_topomap)s
        %(mask_params_topomap)s
        %(contours_topomap)s
        %(outlines_topomap)s
        %(sphere_topomap)s
        %(image_interp_topomap)s
        %(extrapolate_topomap)s
        %(border_topomap)s
        %(res_topomap)s
        %(size_topomap)s
        %(cmap_topomap)s
        %(vlim_topomap)s
        %(cnorm_topomap)s
        %(colorbar_topomap)s
        %(colorbar_format_topomap)s
        %(units_topomap)s
        %(axes_topomap)s
        %(name_format_topomap)s
        %(nrows_topomap)s
        %(ncols_topomap)s
        %(show)s

        Returns
        -------
        %(figs_topomap)s
        """
        if self.filters_ is None:
            raise RuntimeError(
                "no filters are available, please call the `fit` method first"
            )

        return self._plot_filters_patterns(
            self.filters_,
            info,
            components,
            ch_type,
            scalings,
            sensors,
            show_names,
            mask,
            mask_params,
            contours,
            outlines,
            sphere,
            image_interp,
            extrapolate,
            border,
            res,
            size,
            cmap,
            vlim,
            cnorm,
            colorbar,
            cbar_fmt,
            units,
            axes,
            name_format,
            nrows,
            ncols,
            show,
        )

    def _plot_filters_patterns(
        self,
        plot_data,
        info,
        components,
        ch_type,
        scalings,
        sensors,
        show_names,
        mask,
        mask_params,
        contours,
        outlines,
        sphere,
        image_interp,
        extrapolate,
        border,
        res,
        size,
        cmap,
        vlim,
        cnorm,
        colorbar,
        cbar_fmt,
        units,
        axes,
        name_format,
        nrows,
        ncols,
        show,
    ):
        """Plot filters/targets for components."""
        # Sort inputs
        _validate_type(info, Info, "`info`", "mne.Info")
        if components is None:
            components = np.arange(self.n_components)

        # plot seeds and targets
        figs = []
        for group_idx, group_name in zip([0, 1], ["Seeds", "Targets"]):
            # create info for seeds/targets
            group_info = pick_info(info, self.indices[group_idx])
            with group_info._unlock():
                group_info["sfreq"] = 1.0  # 1 component per time point
            # create Evoked object
            evoked = EvokedArray(plot_data[group_idx], group_info, tmin=0)
            # then call plot_topomap
            figs.append(
                evoked.plot_topomap(
                    times=components,
                    average=None,  # do not average across independent components
                    ch_type=ch_type,
                    scalings=scalings,
                    sensors=sensors,
                    show_names=show_names,
                    mask=mask,
                    mask_params=mask_params,
                    contours=contours,
                    outlines=outlines,
                    sphere=sphere,
                    image_interp=image_interp,
                    extrapolate=extrapolate,
                    border=border,
                    res=res,
                    size=size,
                    cmap=cmap,
                    vlim=vlim,
                    cnorm=cnorm,
                    colorbar=colorbar,
                    cbar_fmt=cbar_fmt,
                    units=units,
                    axes=axes,
                    time_format=f"{self._conn_estimator.name}%01d"
                    if name_format is None
                    else name_format,
                    nrows=nrows,
                    ncols=ncols,
                    show=False,  # set Seeds/Targets suptitle first
                )
            )
            figs[-1].suptitle(group_name)  # differentiate seeds from targets
            plt_show(show=show, fig=figs[-1])

        return figs
