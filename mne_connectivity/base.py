import re
from collections.abc import Callable
from copy import copy, deepcopy

import numpy as np
import pandas as pd
import xarray as xr
from mne.utils import (
    _check_combine,
    _check_event_id,
    _check_option,
    _ensure_events,
    _on_missing,
    _validate_type,
    check_random_state,
    copy_function_doc_to_method_doc,
    object_size,
    sizeof_fmt,
    warn,
)

from mne_connectivity.utils import (
    _check_if_multivariate_indices,
    _get_full_connectivity,
    _get_unique_multivariate_nodes_and_indices,
    _prepare_xarray_mne_data_structures,
    fill_doc,
)
from mne_connectivity.viz import plot_connectivity_circle


class SpectralMixin:
    """Mixin class for spectral connectivity."""

    @property
    def freqs(self):
        """The frequency points of the connectivity data.

        If these are computed over a frequency band, it will be the median frequency of
        the frequency band.
        """
        return self.xarray.coords.get("freqs").values.tolist()

    @fill_doc
    def combine_freqs(self, method="mean", fmin=None, fmax=None):
        """Aggregate connectivity data over frequencies.

        Parameters
        ----------
        %(combine_method_freqs)s
        %(combine_fmin_fmax)s

        Returns
        -------
        %(combine_out_freqs)s

        Notes
        -----
        %(combine_median_note)s
        """
        data, fmin, fmax, cls_kwargs = _combine(
            self, method=method, vmin=fmin, vmax=fmax, combine_dim="freqs"
        )

        cls_kwargs["fmin"], cls_kwargs["fmax"] = fmin, fmax
        match type(self).__name__:
            case "SpectralConnectivity":
                con_cls = Connectivity
            case "SpectroTemporalConnectivity":
                con_cls = TemporalConnectivity
                cls_kwargs["times"] = self.times
            case "EpochSpectralConnectivity":
                con_cls = EpochConnectivity
            case "EpochSpectroTemporalConnectivity":
                con_cls = EpochTemporalConnectivity
                cls_kwargs["times"] = self.times

        return con_cls(data=data, metadata=self.metadata, **cls_kwargs)


class TimeMixin:
    """Mixin class for temporal connectivity."""

    @property
    def times(self):
        """The time points of the connectivity data."""
        return self.xarray.coords.get("times").values.tolist()

    @fill_doc
    def combine_times(self, method="mean", tmin=None, tmax=None):
        """Aggregate connectivity data over times.

        Parameters
        ----------
        %(combine_method_times)s
        %(combine_tmin_tmax)s

        Returns
        -------
        %(combine_out_times)s

        Notes
        -----
        %(combine_median_note)s
        """
        data, tmin, tmax, cls_kwargs = _combine(
            self, method=method, vmin=tmin, vmax=tmax, combine_dim="times"
        )

        cls_kwargs["tmin"], cls_kwargs["tmax"] = tmin, tmax
        match type(self).__name__:
            case "TemporalConnectivity":
                con_cls = Connectivity
            case "SpectroTemporalConnectivity":
                con_cls = SpectralConnectivity
                cls_kwargs["freqs"] = self.freqs
            case "EpochTemporalConnectivity":
                con_cls = EpochConnectivity
            case "EpochSpectroTemporalConnectivity":
                con_cls = EpochSpectralConnectivity
                cls_kwargs["freqs"] = self.freqs

        return con_cls(data=data, metadata=self.metadata, **cls_kwargs)


class EpochMixin:
    """Mixin class for per-epoch connectivity."""

    is_epoched = True

    def __init__(self, events, event_id, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Epochs should have the events array that informs user of
        # sample points at which each Epoch was taken from.
        # An empty list occurs when NetCDF stores empty arrays.
        if events is not None and np.array(events).size != 0:
            events = _ensure_events(events)
        else:
            events = np.empty((0, 3))

        event_id = _check_event_id(event_id, events)

        # see BaseEpochs init in MNE-Python
        if events is not None:
            for key, val in event_id.items():
                if val not in events[:, 2]:
                    msg = f"No matching events found for {key} (event id {val})"
                    _on_missing("warn", msg)

            # ensure metadata matches original events size
            self.selection = np.arange(len(events))

            values = list(event_id.values())
            selected = np.where(np.isin(events[:, 2], values))[0]

            events = events[selected]

        self._obj.attrs["events"] = events
        self._obj.attrs["event_id"] = event_id

    @property
    def n_epochs(self):
        """The number of epochs the connectivity data varies over."""
        return self._data.shape[0]

    @property
    def events(self):
        """The events array, for those event types found in ``event_id``."""
        return self.xarray.attrs["events"]

    @property
    def event_id(self):
        """Mapping from condition descriptions (strings) to integer event codes."""
        return self.xarray.attrs["event_id"]

    def append(self, epoch_conn):
        """Append another epoched connectivity structure, in place.

        Parameters
        ----------
        epoch_conn : instance of EpochConnectivity or EpochTemporalConnectivity or EpochSpectralConnectivity or EpochSpectroTemporalConnectivity
            The connectivity container to append. Must be of the same type as the
            current container instance.

        Returns
        -------
        self : instance of EpochConnectivity or EpochTemporalConnectivity or EpochSpectralConnectivity or EpochSpectroTemporalConnectivity
            The altered connectivity container with the new epochs appended. The
            instance type matches that of the current container instance.
        """  # noqa: E501
        if not isinstance(self, type(epoch_conn)):
            raise ValueError(
                f"The type of the epoch connectivity to append is {type(epoch_conn)}, "
                f"which does not match {type(self)}."
            )
        if hasattr(self, "times"):
            if not np.allclose(self.times, epoch_conn.times):
                raise ValueError("Epochs must have same times")
        if hasattr(self, "freqs"):
            if not np.allclose(self.freqs, epoch_conn.freqs):
                raise ValueError("Epochs must have same frequencies")

        events = list(deepcopy(self.events))
        event_id = deepcopy(self.event_id)
        metadata = copy(self.metadata)

        # compare event_id
        common_keys = list(set(event_id).intersection(set(epoch_conn.event_id)))
        for key in common_keys:
            if not event_id[key] == epoch_conn.event_id[key]:
                msg = (
                    "event_id values must be the same for identical keys "
                    'for all concatenated epochs. Key "{}" maps to {} in '
                    "some epochs and to {} in others."
                )
                raise ValueError(
                    msg.format(key, event_id[key], epoch_conn.event_id[key])
                )

        evs = epoch_conn.events.copy()
        if epoch_conn.n_epochs == 0:
            warn("Epoch Connectivity object to append was empty.")
        event_id.update(epoch_conn.event_id)
        events = np.concatenate((events, evs), axis=0)
        metadata = pd.concat([epoch_conn.metadata, metadata])

        # now combine the xarray data, altered events and event ID
        self._obj = xr.concat([self.xarray, epoch_conn.xarray], dim="epochs")
        self._obj.attrs["events"] = events
        self._obj.attrs["event_id"] = event_id
        return self

    @fill_doc
    def combine_epochs(self, method="mean"):
        """Aggregate connectivity data over epochs.

        Parameters
        ----------
        %(combine_method_epochs)s

        Returns
        -------
        %(combine_out_epochs)s

        Notes
        -----
        %(combine_median_note)s
        """
        data, _, _, cls_kwargs = _combine(
            self, method=method, vmin=None, vmax=None, combine_dim="epochs"
        )
        cls_kwargs["n_epochs_used"] = self.n_epochs
        del cls_kwargs["events"]
        del cls_kwargs["event_id"]

        match type(self).__name__:
            case "EpochConnectivity":
                con_cls = Connectivity
            case "EpochTemporalConnectivity":
                con_cls = TemporalConnectivity
                cls_kwargs["times"] = self.times
            case "EpochSpectralConnectivity":
                con_cls = SpectralConnectivity
                cls_kwargs["freqs"] = self.freqs
            case "EpochSpectroTemporalConnectivity":
                con_cls = SpectroTemporalConnectivity
                cls_kwargs["freqs"] = self.freqs
                cls_kwargs["times"] = self.times

        return con_cls(data=data, metadata=self.metadata, **cls_kwargs)


class NonEpochMixin:
    """Mixin class for non-epoched connectivity."""

    is_epoched = False

    @property
    def n_epochs_used(self):
        """Number of epochs used in computation of connectivity.

        Can be ``None``, if no epochs were used.
        """
        return self.attrs["n_epochs_used"]


class DynamicMixin:
    """Mixin class for VAR models."""

    def _check_is_var(self):
        if not re.match(r"^var(?:_dynamic)?$", self.method):
            warn(
                f"The connectivity data comes from a method ({self.method}) that is "
                "not a recognised vector autoregressive (VAR) model. The output of "
                "this may not be valid or interpretable.",
                UserWarning,
            )

    def is_stable(self):
        companion_mat = self.companion
        return np.abs(np.linalg.eigvals(companion_mat)).max() < 1.0

    def eigvals(self):
        return np.linalg.eigvals(self.companion)

    @property
    def companion(self):
        """Generate block companion matrix for a vector autoregressive (VAR) model.

        Returns the data matrix if the VAR model order (i.e., the number of lags) is 1.
        """
        from .vector_ar.utils import _block_companion

        self._check_is_var()

        if "components" in self.dims:
            raise NotImplementedError(
                "Companion matrix generation is not currently supported for VAR "
                "models with multiple components."
            )

        lags = self.attrs.get("lags")
        data = self.get_data("dense")
        if lags == 1:
            return data

        if self.is_epoched:
            n_epochs = self.n_epochs
        else:
            n_epochs = 1
            data = data[np.newaxis, ...]
        arrs = []
        for idx in range(n_epochs):
            blocks = _block_companion([data[idx, ..., jdx] for jdx in range(lags)])
            arrs.append(blocks)
        if not self.is_epoched:
            arrs = arrs[0]

        return np.asarray(arrs)

    def predict(self, data):
        """Predict samples on data using the vector autoregressive (VAR) model.

        The result of this method is used for calculating the residuals.

        Parameters
        ----------
        data : array, shape ([n_epochs,] n_signals, n_times)
            Epoched or continuous data. If the VAR model is time-varying (i.e., there
            are different models per epoch), then the data should also be epoched, with
            one epoch per VAR model. ``n_signals`` should match the number of signals in
            the VAR model.

        Returns
        -------
        predicted : array, shape ([n_epochs,] n_signals, n_times)
            Data as predicted by the VAR model, of the same shape as ``data``.

        Notes
        -----
        Residuals are obtained by ``residuals = x - var.predict(x)``.

        To compute residual covariances from epoched data::

            # compute the covariance of the residuals
            # rows are observations, columns are variables
            t = residuals.shape[0]  # n_epochs
            sampled_residuals = np.concatenate(
                np.split(residuals[:, :, lags:], t, 0), axis=2
            ).squeeze(0)
            residuals_cov = np.cov(sampled_residuals)
        """
        self._check_is_var()

        if data.ndim < 2 or data.ndim > 3:
            raise ValueError(
                "`data` must be either 2D or 3D. The data you passed in has "
                f"{data.ndim} dims."
            )

        if data.ndim == 2 and self.is_epoched:
            raise ValueError("For a time-varying VAR model, `data` must be a 3D array.")

        if "components" in self.dims:
            raise NotImplementedError(
                "Prediction is not currently supported from VAR models with multiple "
                "components."
            )

        n_var = self.n_epochs if self.is_epoched else 1
        if self.is_epoched and data.shape[0] != n_var:
            raise ValueError(
                f"The number of epochs in `data` ({data.shape[0]}) does not match the "
                f"number of VAR models ({n_var})."
            )

        if data.ndim == 2:
            data_ndim = data.ndim
            data = data[np.newaxis, ...]
        else:
            data_ndim = data.ndim
        if data.shape[1] != self.n_nodes:
            raise ValueError(
                f"The number of signals in `data` ({data.shape[1]}) does not match the "
                f"number of signals in the VAR model ({self.n_nodes})."
            )

        # prepare VAR model
        n_epochs, _, n_times = data.shape
        var_model = self.get_data(output="dense")
        lags = self.attrs.get("lags")

        # reshape the coeffs for prediction (and add epochs dim if not present)
        var_model = np.reshape(var_model, (n_var, self.n_nodes, self.n_nodes * lags))

        # predict the data by applying forward model
        predicted_data = np.zeros(data.shape)
        for lag_idx in range(1, lags + 1):
            for epo_idx in range(n_epochs):
                var_idx = epo_idx if self.is_epoched else 0
                bp = var_model[var_idx, :, (lag_idx - 1) :: lags]
                predicted_data[epo_idx, :, lags:] += np.dot(
                    bp, data[epo_idx, :, (lags - lag_idx) : (n_times - lag_idx)]
                )
        if data_ndim == 2:  # remove unwanted epochs dim
            predicted_data = predicted_data[0]

        return predicted_data

    @fill_doc
    def simulate(self, n_samples, noise_func=None, random_state=None):
        """Simulate data using the vector autoregressive (VAR) model.

        Parameters
        ----------
        n_samples : int
            Number of samples to generate.
        noise_func : callable | None
            The function used to create the generating noise process. Each call to this
            function should return an array of shape ``(n_nodes,)``. If ``None``,
            Gaussian white noise with zero mean and unit variance is used.
        %(random_state)s

        Returns
        -------
        data : array, shape (n_nodes, n_samples)
            Generated data.

        Notes
        -----
        If the VAR model is time-varying (i.e., there are different models per epoch),
        the data is simulated from the average of these models.
        """
        self._check_is_var()

        if "components" in self.dims:
            raise NotImplementedError(
                "Simulation is not currently supported from VAR models with multiple "
                "components."
            )

        var_model = self.get_data(output="dense")
        if self.is_epoched:
            var_model = var_model.mean(axis=0)

        # reshape the coeffs for simulation
        n_nodes = self.n_nodes
        lags = self.attrs.get("lags")
        var_model = np.reshape(var_model, (n_nodes, n_nodes * lags))

        # set noise function
        if noise_func is None:
            rng = check_random_state(random_state)

            def noise_func():
                return rng.normal(size=(1, n_nodes))

        n = n_samples + 10 * lags

        # simulated data
        data = np.zeros((n, n_nodes))
        res = np.zeros((n, n_nodes))

        for jdx in range(lags, n):
            e = noise_func()
            res[jdx, :] = e
            data[jdx, :] = e
            for idx in range(1, lags + 1):
                data[jdx, :] += var_model[:, (idx - 1) :: lags].dot(data[jdx - idx, :])

        return data[10 * lags :, :].transpose()


class BaseConnectivity:
    """Base class for connectivity data.

    This class should not be instantiated directly.

    Connectivity data is anything that represents "connections" between nodes as a
    ``(N, N)`` array. It can be symmetric, or asymmetric. If it is symmetric (or
    asymmetric, but the missing (e.g., upper-triangular) elements can be determined
    based on the existing (e.g., lower-triangular) elements), we can optimise memory
    demand for storage.

    Notes
    -----
    Connectivity data can be generally represented as a square matrix with values
    intending the connectivity function value between two nodes. The underlying data
    structure is an ``xarray.DataArray``, with a similar API to ``xarray``.#

    We store connectivity data as a raveled ``(n_connections, ...)`` array, where
    ``n_connections`` can be ``n_nodes * n_nodes`` if a full connectivity structure is
    computed, or a subset of the nodes (equal to the length of the indices passed in).

    Since we store connectivity data as a raveled array, one can easily optimize the
    storage of "symmetric" connectivity data by storing only the lower-triangular (or
    upper-triangular) elements, and filling these in when the user requests the full
    connectivity matrix.

    Storage optimisation will not occur for multivariate connectivity, as computing
    connectivity in the same lower-/upper-triangular manner for "symmetric" methods does
    not transfer.
    """

    # Expected to be properly defined in the child classes
    is_epoched: bool
    n_epochs_used: int | None

    def __init__(
        self,
        data,
        names,
        indices,
        method,
        n_nodes,
        metadata=None,
        **kwargs,
    ):
        _validate_type(indices, (str, tuple), "`indices`")
        if isinstance(indices, str):
            _check_option("indices", indices, ["all", "lower", "upper"], "as a string")

        # prepare metadata pandas dataframe and ensure metadata is a Pandas
        # DataFrame object
        if metadata is None:
            metadata = pd.DataFrame(dtype="float64")
        self.metadata = metadata

        # check the incoming data structure
        if "components" in kwargs:
            bad_indices = None
            if not isinstance(indices, tuple):
                bad_indices = f"'{indices}'"  # must be a str
            elif not _check_if_multivariate_indices(indices):
                bad_indices = "a tuple of non-nested arrays"  # must be non-nested
            if bad_indices:
                raise ValueError(
                    "`components` are present in `kwargs`, which is a term reserved "
                    "for multivariate connectivity methods. However, `indices` does "
                    "not match the format for multivariate methods. Expected a tuple "
                    f"of nested arrays, got {bad_indices}."
                )
        self._check_data_consistency(data, indices=indices, n_nodes=n_nodes)
        self._prepare_xarray(
            data, names=names, indices=indices, n_nodes=n_nodes, method=method, **kwargs
        )

    def __repr__(self) -> str:
        r = f"<{self.__class__.__name__} | "
        r += f"{self.method if self.method is not None else 'Unknown method'} | "

        dim_info = []
        for dim in ("epochs", "connections", "components", "freqs", "times"):
            if dim == "connections":
                dim_info.append(f"{self.n_connections} connections")
            elif dim == "freqs" and dim in self.dims:
                dim_info.append(f"{self.freqs[0]}-{self.freqs[-1]} Hz")
            elif dim == "times" and dim in self.dims:
                dim_info.append(f"{self.times[0]}-{self.times[-1]} s")
            elif dim in self.dims:  # epochs or components
                dim_info.append(f"{len(self.coords[dim])} {dim}")
        dim_info = ", ".join(dim_info) + " | "
        r += dim_info

        if not self.is_epoched and self.n_epochs_used is not None:
            r += f"nave: {self.n_epochs_used} | "

        r += f"~{sizeof_fmt(self._size)}>"

        return r

    def _prepare_xarray(self, data, names, indices, n_nodes, method, **kwargs):
        """Prepare xarray data structure."""
        # set node names
        if names is None:
            names = list(map(str, range(n_nodes)))

        # the dimensions and their coordinates
        dims = ["connections"]
        if self.is_epoched:
            dims = ["epochs"] + dims
        coords = dict()
        if self.is_epoched:
            coords["epochs"] = list(range(data.shape[0]))
        coords["connections"] = list(range(self.n_connections))
        if "components" in kwargs:
            coords["components"] = kwargs.pop("components")
            dims.append("components")
        if "freqs" in kwargs:
            coords["freqs"] = kwargs.pop("freqs")
            dims.append("freqs")
        if "times" in kwargs:
            times = kwargs.pop("times")
            if times is None:
                times = list(range(data.shape[-1]))
            coords["times"] = list(times)
            dims.append("times")

        # convert all numpy arrays to lists
        for key, val in kwargs.items():
            if isinstance(val, np.ndarray):
                kwargs[key] = val.tolist()
        kwargs["node_names"] = names

        # set method, indices and n_nodes
        if isinstance(indices, tuple):
            if all(isinstance(inds, np.ndarray) for inds in indices):
                # leave multivariate indices as arrays for easier indexing
                if all(inds.ndim > 1 for inds in indices):
                    new_indices = (indices[0], indices[1])
                else:
                    new_indices = (list(indices[0]), list(indices[1]))
            else:
                new_indices = (list(indices[0]), list(indices[1]))
            indices = new_indices
        kwargs["method"] = method
        kwargs["indices"] = indices
        kwargs["n_nodes"] = n_nodes

        # create xarray object
        xarray_obj = xr.DataArray(data=data, coords=coords, dims=dims, attrs=kwargs)
        self._obj = xarray_obj

    def _check_data_consistency(self, data, indices, n_nodes):
        """Perform data input checks."""
        if not isinstance(data, np.ndarray):
            raise TypeError("Connectivity data must be passed in as a numpy array.")

        if self.is_epoched:
            if data.ndim < 2 or data.ndim > 5:
                raise RuntimeError(
                    "Data using an epoched data structure should have at least 2 "
                    f"dimensions and at most 5 dimensions. Your data was {data.shape} "
                    "shape."
                )
        else:
            if data.ndim > 4:
                raise RuntimeError(
                    "Data not using an epoched data structure should have at least 1 "
                    f"dimensions and at most 4 dimensions. Your data was {data.shape} "
                    "shape."
                )

        # get the number of connections
        self.n_connections = data.shape[1] if self.is_epoched else data.shape[0]

        if isinstance(indices, tuple):
            # check that the indices passed in are of the same length
            if len(indices[0]) != len(indices[1]):
                raise ValueError(
                    "If indices are passed in then they must be the same length. They "
                    f"are right now {len(indices[0])} and {len(indices[1])}."
                )
            # indices length should match the data length
            if len(indices[0]) != self.n_connections:
                raise ValueError(
                    f"The number of indices, {len(indices[0])} should match the "
                    f"raveled data length passed in of {self.n_connections}."
                )

        elif indices in ["lower", "upper"]:
            expected_len = n_nodes * (n_nodes - 1) // 2
            if self.n_connections != expected_len:
                raise ValueError(
                    "If `indices` is 'lower' or 'upper', then connectivity data should "
                    "be the lower- or upper-triangular part of the connectivity "
                    f"matrix, respectively. Expected {expected_len} connections from "
                    f"the {n_nodes} nodes, but got {self.n_connections} "
                    "connections."
                )

        else:  # indices = "all"
            expected_len = n_nodes**2
            if self.n_connections != expected_len:
                raise ValueError(
                    "If `indices` is 'all', then connectivity data should be the full "
                    f"connectivity matrix. Expected {expected_len} connections from "
                    f"the {n_nodes} nodes, but got {self.n_connections} "
                    "connections."
                )

    def copy(self):
        """Return a deep copy of the connectivity container."""
        return deepcopy(self)

    @property
    def _data(self):
        """Numpy array of connectivity data."""
        return self.xarray.values

    @property
    def dims(self):
        """The dimensions of the xarray data."""
        return self.xarray.dims

    @property
    def coords(self):
        """The coordinates of the xarray data."""
        return self.xarray.coords

    @property
    def attrs(self):
        """Xarray attributes of connectivity.

        See ``xarray``'s ``attrs``.
        """
        return self.xarray.attrs

    @property
    def shape(self):
        """Shape of raveled connectivity."""
        return self.xarray.shape

    @property
    def n_nodes(self):
        """The number of nodes in the original dataset.

        Even if ``indices`` defines a subset of nodes that were computed, this should be
        the total number of nodes in the original dataset.
        """
        return self.attrs["n_nodes"]

    @property
    def method(self):
        """The method used to compute connectivity."""
        return self.attrs["method"]

    @property
    def indices(self):
        """Indices of connectivity data.

        Returns
        -------
        indices : ``'all'`` | ``'lower'`` | ``'upper'`` | tuple of list
            Either ``'all'`` for all-to-all connectivity, ``'lower'`` for
            lower-triangular connectivity, ``'upper'`` for upper-triangular
            connectivity, or a tuple of lists representing the seed and target nodes
            that connectivity was computed between.
        """
        return self.attrs["indices"]

    @property
    def names(self):
        """Node names."""
        return self.attrs["node_names"]

    @property
    def xarray(self):
        """Xarray of the connectivity data."""
        return self._obj

    @property
    def _size(self):
        """Estimate the object size."""
        size = 0
        size += object_size(self._data)
        size += object_size(self.attrs)

        # if self.metadata is not None:
        #     size += self.metadata.memory_usage(index=True).sum()
        return size

    def get_data(self, output="compact", missing="raise"):
        """Get connectivity data as a numpy array.

        Parameters
        ----------
        output : ``'compact'`` | ``'raveled'`` | ``'dense'``
            How to format the output:

            - ``'raveled'`` will represent each connectivity matrix as a
              ``(..., n_connections, ...)`` array.
            - ``'dense'`` will return each connectivity matrix as a ``(..., n_seeds,
              n_targets, ...)`` array.
            - ``'compact'`` (default) will return ``'raveled'`` if ``indices`` is
              a tuple of arrays, or ``'dense'`` if ``indices`` is ``'all'``,
              ``'lower'``, or ``'upper'``.
        missing : ``'raise'`` | float
            How to handle missing values in the dense connectivity matrix when these
            cannot be filled in (see notes for more information). If ``'raise'``, an
            error is raised. If a float, the missing values are filled with that float.
            Ignored if ``output='raveled'``. Default is ``'raise'``.

        Returns
        -------
        data : array
            The output connectivity data.
        multivariate_nodes : tuple of array
            Returned if the connectivity data is multivariate with ``self.indices``
            defined as a tuple of arrays, and ``output='dense'``. Used to map from the
            original set of indices to the dense matrix space. See notes for more
            information.

        Notes
        -----
        **Handling missing values for dense outputs**

        If ``indices`` is not ``'all'`` and ``output='dense'``, there may be missing
        values from the full connectivity matrix that need to be filled in:

        1. When ``indices`` is ``'lower'`` or ``'upper'``, the missing values will try
           to be inferred based on the existing ones, falling back to the behaviour
           determined by the ``missing`` parameter if they cannot be inferred.

        2. When ``indices`` is a tuple and ``indices`` represents a subset of the full
           connectivity matrix, the missing values will not be inferred, and the
           behaviour is determined by the ``missing`` parameter.

        3. When ``indices`` is a tuple and ``indices`` represents the full connectivity
           matrix, there are no missing values to fill in.

        **Handling dense outputs for multivariate connectivity**

        Because multivariate connectivity data can involve multiple channels per
        connection, it is not possible to represent the data in a dense matrix form
        where each row/column represent a single channel.

        Instead, the multivariate data is mapped into a new space, based on the set of
        channels that define each node. E.g., the multivariate indices::

            (
                [[0, 1], [0, 1], [2, 3]],  # seeds
                [[2, 3], [4, 5], [4, 5]]   # targets
            )

        contains the following sets of channels: ``[0, 1]`` (node 0); ``[2, 3]``
        (node 1); and ``[4, 5]`` (node 2). Based on this, the multivariate indices can
        be mapped to a new space with indices::

            (
                [0, 0, 1],  # seeds
                [1, 2, 2]   # targets
            )

        For a dense connectivity matrix, this would return a ``(3, 3)`` upper-triangular
        array.

        To ensure the mapping from the original multivariate indices to the new dense
        matrix space is traceable, ``multivariate_nodes`` is returned, which contains
        the (unmasked form) of each node in the position where it exists in the dense
        matrix space. For the above example, ``multivariate_nodes`` would be
        ``[[0, 1], [2, 3], [4, 5]]``.
        """
        _check_option("output", output, ["raveled", "dense", "compact"])
        _validate_type(missing, (str, "numeric"), "`missing`")
        if isinstance(missing, str):
            _check_option("missing", missing, ["raise"], "as a string")
        else:
            missing = float(missing)

        multivariate_nodes = None

        if output == "compact":
            if self.indices in ["all", "lower", "upper"]:
                output = "dense"
            else:
                output = "raveled"

        if output == "raveled":
            data = self._data
        else:
            # Check if indices are for multivariate connectivity
            if isinstance(self.indices, tuple) and _check_if_multivariate_indices(
                self.indices
            ):
                # Remap multivariate indices from channels to the unique nodes
                # (nodes here are considered a set of channels)
                multivariate_nodes, indices = (
                    _get_unique_multivariate_nodes_and_indices(self.indices)
                )
                n_nodes = len(multivariate_nodes)
            else:
                indices, n_nodes = self.indices, self.n_nodes

            # Move epochs from first dimension temporarily for easier reshaping
            data = self._data
            if self.is_epoched:
                data = np.moveaxis(data, 0, -1)

            # Get connectivity as a square matrix, with missing values filled
            data = _get_full_connectivity(
                data, indices, n_nodes, self.method, missing=missing
            )

            # Move epochs back to first dimension if needed
            if self.is_epoched:
                data = np.moveaxis(data, -1, 0)

        if multivariate_nodes is not None:
            return data, multivariate_nodes
        return data

    def rename_nodes(self, mapping):
        """Rename nodes.

        Parameters
        ----------
        mapping : dict
            Mapping from original node names (keys) to new node names (values).
        """
        names = copy(self.names)

        # first check and assemble clean mappings of index and name
        if isinstance(mapping, dict):
            orig_names = sorted(list(mapping.keys()))
            missing = [orig_name not in names for orig_name in orig_names]
            if any(missing):
                raise ValueError(
                    "Name(s) in mapping missing from info: "
                    f"{np.array(orig_names)[np.array(missing)]}"
                )
            new_names = [
                (names.index(name), new_name) for name, new_name in mapping.items()
            ]
        elif callable(mapping):
            new_names = [(ci, mapping(name)) for ci, name in enumerate(names)]
        else:
            raise ValueError(f"mapping must be callable or dict, not {type(mapping)}")

        # check we got all strings out of the mapping
        for new_name in new_names:
            _validate_type(new_name[1], "str", "New name mappings")

        # do the remapping locally
        for c_ind, new_name in new_names:
            names[c_ind] = new_name

        # check that all the channel names are unique
        if len(names) != len(np.unique(names)):
            raise ValueError("New channel names are not unique, renaming failed")

        # rename the new names
        self._obj.attrs["node_names"] = names

    @copy_function_doc_to_method_doc(plot_connectivity_circle)
    def plot_circle(self, **kwargs):
        plot_connectivity_circle(
            self.get_data(), node_names=self.names, indices=self.indices, **kwargs
        )

    def save(self, fname):
        """Save connectivity data to disk.

        Can later be loaded using the function
        :func:`~mne_connectivity.read_connectivity`.

        Parameters
        ----------
        fname : str | pathlib.Path
            The filepath to save the data. Data is saved as netCDF files (``.nc``
            extension).
        """
        method = self.method
        indices = self.indices
        n_nodes = self.n_nodes

        # create a copy of the old attributes
        old_attrs = copy(self.attrs)

        # assign these to xarray's attrs
        self.attrs["method"] = method
        self.attrs["indices"] = indices
        self.attrs["n_nodes"] = n_nodes

        # save the name of the connectivity structure
        self.attrs["data_structure"] = str(self.__class__.__name__)

        # get a copy of metadata into attrs as a dictionary
        self = _prepare_xarray_mne_data_structures(self)

        # netCDF does not support 'None'
        # so map these to 'n/a'
        for key, val in self.attrs.items():
            if val is None:
                self.attrs[key] = "n/a"

        # save as a netCDF file
        # note this requires the netcdf4 python library
        # and h5netcdf library.
        # The engine specified requires the ability to save
        # complex data types, which was not natively supported
        # in xarray. Therefore, h5netcdf is the only engine
        # to support that feature at this moment.
        self.xarray.to_netcdf(
            fname, mode="w", format="NETCDF4", engine="h5netcdf", invalid_netcdf=True
        )

        # re-set old attributes
        self.xarray.attrs = old_attrs


@fill_doc
class Connectivity(BaseConnectivity, DynamicMixin, NonEpochMixin):
    """Connectivity class without additional dimensions.

    The underlying data is an array of shape
    ``%(data_shape_Connectivity)s``.%(optional_components_dim)s

    Parameters
    ----------
    %(data_Connectivity)s
    %(n_nodes)s
    %(names)s
    %(indices)s
    %(method)s
    %(n_epochs_used)s
    %(connectivity_kwargs)s

    See Also
    --------
    mne_connectivity.vector_auto_regression
    """

    def __init__(
        self,
        data,
        n_nodes,
        names=None,
        indices="all",
        method=None,
        n_epochs_used=None,
        **kwargs,
    ):
        super().__init__(
            data=data,
            names=names,
            method=method,
            n_nodes=n_nodes,
            indices=indices,
            n_epochs_used=n_epochs_used,
            **kwargs,
        )


@fill_doc
class SpectralConnectivity(BaseConnectivity, SpectralMixin, NonEpochMixin):
    """Connectivity class with frequency dimension.

    The underlying data is an array of shape
    ``%(data_shape_SpectralConnectivity)s``.%(optional_components_dim)s

    Parameters
    ----------
    %(data_SpectralConnectivity)s
    %(freqs)s
    %(n_nodes)s
    %(names)s
    %(indices)s
    %(method)s
    %(spec_method)s
    %(n_epochs_used)s
    %(connectivity_kwargs)s

    See Also
    --------
    mne_connectivity.phase_slope_index
    mne_connectivity.spectral_connectivity_epochs
    mne_connectivity.spectral_connectivity_time
    """

    def __init__(
        self,
        data,
        freqs,
        n_nodes,
        names=None,
        indices="all",
        method=None,
        spec_method=None,
        n_epochs_used=None,
        **kwargs,
    ):
        super().__init__(
            data=data,
            names=names,
            method=method,
            indices=indices,
            n_nodes=n_nodes,
            freqs=freqs,
            spec_method=spec_method,
            n_epochs_used=n_epochs_used,
            **kwargs,
        )


@fill_doc
class TemporalConnectivity(BaseConnectivity, TimeMixin, DynamicMixin, NonEpochMixin):
    """Connectivity class with time dimension.

    The underlying data is an array of shape
    ``%(data_shape_TemporalConnectivity)s``.%(optional_components_dim)s

    Parameters
    ----------
    %(data_TemporalConnectivity)s
    %(times)s
    %(n_nodes)s
    %(names)s
    %(indices)s
    %(method)s
    %(n_epochs_used)s
    %(connectivity_kwargs)s
    """

    def __init__(
        self,
        data,
        times,
        n_nodes,
        names=None,
        indices="all",
        method=None,
        n_epochs_used=None,
        **kwargs,
    ):
        super().__init__(
            data=data,
            names=names,
            method=method,
            n_nodes=n_nodes,
            indices=indices,
            times=times,
            n_epochs_used=n_epochs_used,
            **kwargs,
        )


@fill_doc
class SpectroTemporalConnectivity(
    BaseConnectivity, SpectralMixin, TimeMixin, NonEpochMixin
):
    """Connectivity class with frequency and time dimensions.

    The underlying data is an array of shape
    ``%(data_shape_SpectroTemporalConnectivity)s``.%(optional_components_dim)s

    Parameters
    ----------
    %(data_SpectroTemporalConnectivity)s
    %(freqs)s
    %(times)s
    %(n_nodes)s
    %(names)s
    %(indices)s
    %(method)s
    %(spec_method)s
    %(n_epochs_used)s
    %(connectivity_kwargs)s

    See Also
    --------
    mne_connectivity.phase_slope_index
    mne_connectivity.spectral_connectivity_epochs
    """

    def __init__(
        self,
        data,
        freqs,
        times,
        n_nodes,
        names=None,
        indices="all",
        method=None,
        spec_method=None,
        n_epochs_used=None,
        **kwargs,
    ):
        super().__init__(
            data=data,
            names=names,
            method=method,
            indices=indices,
            n_nodes=n_nodes,
            freqs=freqs,
            spec_method=spec_method,
            times=times,
            n_epochs_used=n_epochs_used,
            **kwargs,
        )


@fill_doc
class EpochConnectivity(EpochMixin, BaseConnectivity, DynamicMixin):
    """Connectivity class with epoch dimension.

    This underlying data is an array of shape
    ``%(data_shape_EpochConnectivity)s``.%(optional_components_dim)s

    Parameters
    ----------
    %(data_EpochConnectivity)s
    %(n_nodes)s
    %(names)s
    %(indices)s
    %(method)s
    %(events)s
    %(event_id)s
    %(connectivity_kwargs)s

    See Also
    --------
    mne_connectivity.vector_auto_regression
    """

    def __init__(
        self,
        data,
        n_nodes,
        names=None,
        indices="all",
        method=None,
        events=None,
        event_id=None,
        **kwargs,
    ):
        super().__init__(
            data=data,
            names=names,
            method=method,
            n_nodes=n_nodes,
            indices=indices,
            events=events,
            event_id=event_id,
            **kwargs,
        )


@fill_doc
class EpochSpectralConnectivity(EpochMixin, BaseConnectivity, SpectralMixin):
    """Connectivity class with epoch and frequency dimensions.

    The underlying data is an array of shape
    ``%(data_shape_EpochSpectralConnectivity)s``.%(optional_components_dim)s

    Parameters
    ----------
    %(data_EpochSpectralConnectivity)s
    %(freqs)s
    %(n_nodes)s
    %(names)s
    %(indices)s
    %(method)s
    %(spec_method)s
    %(events)s
    %(event_id)s
    %(connectivity_kwargs)s

    See Also
    --------
    mne_connectivity.spectral_connectivity_time
    """

    def __init__(
        self,
        data,
        freqs,
        n_nodes,
        names=None,
        indices="all",
        method=None,
        spec_method=None,
        events=None,
        event_id=None,
        **kwargs,
    ):
        super().__init__(
            data=data,
            freqs=freqs,
            names=names,
            indices=indices,
            n_nodes=n_nodes,
            method=method,
            spec_method=spec_method,
            events=events,
            event_id=event_id,
            **kwargs,
        )


@fill_doc
class EpochTemporalConnectivity(EpochMixin, BaseConnectivity, TimeMixin, DynamicMixin):
    """Connectivity class with epoch and time dimensions.

    The underlying data is an array of shape
    ``%(data_shape_EpochTemporalConnectivity)s``.%(optional_components_dim)s

    Parameters
    ----------
    %(data_EpochTemporalConnectivity)s
    %(times)s
    %(n_nodes)s
    %(names)s
    %(indices)s
    %(method)s
    %(events)s
    %(event_id)s
    %(connectivity_kwargs)s

    See Also
    --------
    mne_connectivity.envelope_correlation
    mne_connectivity.vector_auto_regression
    """

    def __init__(
        self,
        data,
        times,
        n_nodes,
        names=None,
        indices="all",
        method=None,
        events=None,
        event_id=None,
        **kwargs,
    ):
        super().__init__(
            data=data,
            times=times,
            names=names,
            indices=indices,
            n_nodes=n_nodes,
            method=method,
            events=events,
            event_id=event_id,
            **kwargs,
        )


@fill_doc
class EpochSpectroTemporalConnectivity(
    EpochMixin, BaseConnectivity, SpectralMixin, TimeMixin
):
    """Connectivity class with epoch, frequency, and time dimensions.

    The underlying data is an array of shape
    ``%(data_shape_EpochSpectroTemporalConnectivity)s``.%(optional_components_dim)s

    Parameters
    ----------
    %(data_EpochSpectroTemporalConnectivity)s
    %(freqs)s
    %(times)s
    %(n_nodes)s
    %(names)s
    %(indices)s
    %(method)s
    %(spec_method)s
    %(events)s
    %(event_id)s
    %(connectivity_kwargs)s
    """

    def __init__(
        self,
        data,
        freqs,
        times,
        n_nodes,
        names=None,
        indices="all",
        method=None,
        spec_method=None,
        events=None,
        event_id=None,
        **kwargs,
    ):
        super().__init__(
            data=data,
            names=names,
            freqs=freqs,
            times=times,
            indices=indices,
            n_nodes=n_nodes,
            method=method,
            spec_method=spec_method,
            events=events,
            event_id=event_id,
            **kwargs,
        )


def _combine(
    con: BaseConnectivity,
    method: str | Callable,
    vmin: float | None,
    vmax: float | None,
    combine_dim: str,
) -> np.ndarray:
    expected_shape = tuple(
        n for n, dim in zip(con.shape, con.dims) if dim != combine_dim
    )
    dim_axis = con.dims.index(combine_dim)
    func = _check_combine(method, valid=("mean", "median"), axis=dim_axis)
    dim_vals = np.array(con.coords[combine_dim])
    vmin = -np.inf if vmin is None else vmin
    vmax = np.inf if vmax is None else vmax
    dim_indices = np.argwhere((dim_vals >= vmin) & (dim_vals <= vmax))[..., 0]
    if dim_indices.size == 0:
        raise ValueError(
            f"No values found in the {combine_dim} dimension between {vmin} and {vmax}."
        )

    if con.method in ["cohy", "imcoh", "cacoh", "mic"] and not np.allclose(
        con.xarray.data, np.abs(con.xarray.data)
    ):
        warn(
            "To ensure the interpretability of the connectivity data, consider taking "
            f"the absolute value of the connectivity data for the '{con.method}' "
            f"method before combining along the {combine_dim} dimension."
        )

    data = func(np.take(con.xarray.data, dim_indices, axis=dim_axis))
    vmin, vmax = dim_vals[dim_indices][[0, -1]]

    if data.shape != expected_shape:
        raise RuntimeError(
            f"`combine_{combine_dim}` got a method that resulted in data of shape "
            f"{data.shape}, expected {expected_shape}."
        )

    cls_kwargs = copy(con.attrs)
    cls_kwargs["names"] = cls_kwargs.pop("node_names")
    if "components" in con.dims:
        cls_kwargs["components"] = con.coords["components"].values.tolist()

    return data, vmin, vmax, cls_kwargs
