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

from mne_connectivity.utils import _prepare_xarray_mne_data_structures, fill_doc
from mne_connectivity.viz import plot_connectivity_circle


class SpectralMixin:
    """Mixin class for spectral connectivities.

    Note: In mne-connectivity, we associate the word "spectral" with time-frequency.
    Reference to eigenvalue structure is not captured in this mixin.
    """

    @property
    def freqs(self):
        """The frequency points of the connectivity data.

        If these are computed over a frequency band, it will be the median frequency of
        the frequency band.
        """
        return self.xarray.coords.get("freqs").values.tolist()


class TimeMixin:
    @property
    def times(self):
        """The time points of the connectivity data."""
        return self.xarray.coords.get("times").values.tolist()


class EpochMixin:
    def _init_epochs(self, events, event_id, on_missing="warn") -> None:
        # Epochs should have the events array that informs user of
        # sample points at which each Epoch was taken from.
        # An empty list occurs when NetCDF stores empty arrays.
        if events is not None and np.array(events).size != 0:
            events = _ensure_events(events)
        else:
            events = np.empty((0, 3))

        event_id = _check_event_id(event_id, events)
        self.event_id = event_id
        self.events = events

        # see BaseEpochs init in MNE-Python
        if events is not None:
            for key, val in self.event_id.items():
                if val not in events[:, 2]:
                    msg = f"No matching events found for {key} (event id {val})"
                    _on_missing(on_missing, msg)

            # ensure metadata matches original events size
            self.selection = np.arange(len(events))
            self.events = events
            del events

            values = list(self.event_id.values())
            selected = np.where(np.isin(self.events[:, 2], values))[0]

            self.events = self.events[selected]

    def append(self, epoch_conn):
        """Append another connectivity structure.

        Parameters
        ----------
        epoch_conn : instance of Connectivity
            The epoched Connectivity class to append.

        Returns
        -------
        self : instance of Connectivity
            The altered epoched Connectivity class.
        """
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
        self.events = events
        self.event_id = event_id
        return self

    def combine(self, combine="mean"):
        """Combine connectivity data over epochs.

        Parameters
        ----------
        combine : ``'mean'`` | ``'median'`` | callable
            How to combine correlation estimates across epochs. Default is ``'mean'``.
            If callable, it must accept one positional input. For example::

                combine = lambda data: np.median(data, axis=0)

        Returns
        -------
        conn : instance of Connectivity
            The combined connectivity data structure. Instance type reflects that of the
            input instance, without the epoch dimension.
        """  # noqa: E501
        from .io import _xarray_to_conn

        if not self.is_epoched:
            raise RuntimeError(
                "Combine only works over Epoched connectivity. It does not work with "
                f"{self}"
            )

        fun = _check_combine(combine, valid=("mean", "median"))

        # get a copy of metadata into attrs as a dictionary
        self = _prepare_xarray_mne_data_structures(self)

        # apply function over the  array
        new_xr = xr.apply_ufunc(
            fun, self.xarray, input_core_dims=[["epochs"]], vectorize=True
        )
        new_xr.attrs = self.xarray.attrs

        # map class name to its actual class
        conn_cls = {
            "EpochConnectivity": Connectivity,
            "EpochTemporalConnectivity": TemporalConnectivity,
            "EpochSpectralConnectivity": SpectralConnectivity,
            "EpochSpectroTemporalConnectivity": SpectroTemporalConnectivity,
        }
        cls_func = conn_cls[self.__class__.__name__]

        # convert new xarray to non-Epoch data structure
        conn = _xarray_to_conn(new_xr, cls_func)
        return conn


class DynamicMixin:
    def is_stable(self):
        companion_mat = self.companion
        return np.abs(np.linalg.eigvals(companion_mat)).max() < 1.0

    def eigvals(self):
        return np.linalg.eigvals(self.companion)

    @property
    def companion(self):
        """Generate block companion matrix.

        Returns the data matrix if the model is VAR(1).
        """
        from .vector_ar.utils import _block_companion

        lags = self.attrs.get("lags")
        data = self.get_data()
        if lags == 1:
            return data

        arrs = []
        for idx in range(self.n_epochs):
            blocks = _block_companion([data[idx, ..., jdx] for jdx in range(lags)])
            arrs.append(blocks)
        return arrs

    def predict(self, data):
        """Predict samples on actual data.

        The result of this function is used for calculating the residuals.

        Parameters
        ----------
        data : array, shape ([n_epochs,] n_signals, n_times)
            Epoched or continuous data set.

        Returns
        -------
        predicted : array, shape ([n_epochs,] n_signals, n_times)
            Data as predicted by the VAR model of shape same as ``data``.

        Notes
        -----
        Residuals are obtained by ``r = x - var.predict(x)``.

        To compute residual covariances::

            # compute the covariance of the residuals
            # row are observations, columns are variables
            t = residuals.shape[0]
            sampled_residuals = np.concatenate(
                np.split(residuals[:, :, lags:], t, 0), axis=2
            ).squeeze(0)
            rescov = np.cov(sampled_residuals)
        """
        if data.ndim < 2 or data.ndim > 3:
            raise ValueError(
                "Data passed in must be either 2D or 3D. The data you passed in has "
                f"{data.ndim} dims."
            )
        if data.ndim == 2 and self.is_epoched:
            raise RuntimeError(
                "If there is a VAR model over epochs, one must pass in a 3D array."
            )
        if data.ndim == 3 and not self.is_epoched:
            raise RuntimeError(
                "If there is a single VAR model, one must pass in a 2D array."
            )

        # make the data 3D
        if data.ndim == 2:
            data = data[np.newaxis, ...]

        n_epochs, _, n_times = data.shape
        var_model = self.get_data(output="dense")

        # get the model order
        lags = self.attrs.get("lags")

        # predict the data by applying forward model
        predicted_data = np.zeros(data.shape)
        # which takes less loop iterations
        if n_epochs > n_times - lags:
            for idx in range(1, lags + 1):
                for jdx in range(lags, n_times):
                    if self.is_epoched:
                        bp = var_model[jdx, :, (idx - 1) :: lags]
                    else:
                        bp = var_model[:, (idx - 1) :: lags]
                    predicted_data[:, :, jdx] += np.dot(data[:, :, jdx - idx], bp.T)
        else:
            for idx in range(1, lags + 1):
                for jdx in range(n_epochs):
                    if self.is_epoched:
                        bp = var_model[jdx, :, (idx - 1) :: lags]
                    else:
                        bp = var_model[:, (idx - 1) :: lags]
                    predicted_data[jdx, :, lags:] += np.dot(
                        bp, data[jdx, :, (lags - idx) : (n_times - idx)]
                    )

        return predicted_data

    @fill_doc
    def simulate(self, n_samples, noise_func=None, random_state=None):
        """Simulate vector autoregressive (VAR) model.

        This function generates data from the VAR model.

        Parameters
        ----------
        n_samples : int
            Number of samples to generate.
        noise_func : callable | None
            This function is used to create the generating noise process. If ``None``,
            Gaussian white noise with zero mean and unit variance is used.
        %(random_state)s

        Returns
        -------
        data : array, shape (n_samples, n_channels)
            Generated data.
        """
        var_model = self.get_data(output="dense")
        if self.is_epoched:
            var_model = var_model.mean(axis=0)

        n_nodes = self.n_nodes
        lags = self.attrs.get("lags")

        # set noise function
        if noise_func is None:
            rng = check_random_state(random_state)

            def noise_func():
                return rng.normal(size=(1, n_nodes))

        n = n_samples + 10 * lags

        # simulated data
        data = np.zeros((n, n_nodes))
        res = np.zeros((n, n_nodes))

        for jdx in range(lags):
            e = noise_func()
            res[jdx, :] = e
            data[jdx, :] = e
        for jdx in range(lags, n):
            e = noise_func()
            res[jdx, :] = e
            data[jdx, :] = e
            for idx in range(1, lags + 1):
                data[jdx, :] += var_model[:, (idx - 1) :: lags].dot(data[jdx - idx, :])

        # self.residuals = res[10 * lags:, :, :].T
        # self.rescov = sp.cov(cat_trials(self.residuals).T, rowvar=False)
        return data[10 * lags :, :].transpose()


@fill_doc
class BaseConnectivity(DynamicMixin, EpochMixin):
    """Base class for connectivity data.

    This class should not be instantiated directly, but should be used to do
    type-checking. All connectivity classes will be returned from corresponding
    connectivity computing functions.

    Connectivity data is anything that represents "connections" between nodes as a
    ``(N, N)`` array. It can be symmetric, or asymmetric (if it is symmetric, storage
    optimization will occur).

    Parameters
    ----------
    %(data)s
    %(names)s
    %(indices)s
    %(method)s
    %(n_nodes)s
    %(events)s
    %(event_id)s
    metadata : instance of pandas.DataFrame | None
        The metadata data frame that would come from the :class:`mne.Epochs` class. See
        :class:`mne.Epochs` docstring for details.
    %(connectivity_kwargs)s

    Notes
    -----
    Connectivity data can be generally represented as a square matrix with values
    intending the connectivity function value between two nodes. We optimize storage of
    symmetric connectivity data and allow support for computing connectivity data on a
    subset of nodes. We store connectivity data as a raveled ``(n_estimated_nodes,
    ...)`` where ``n_estimated_nodes`` can be ``n_nodes_in * n_nodes_out`` if a full
    connectivity structure is computed, or a subset of the nodes (equal to the length of
    the indices passed in).

    Since we store connectivity data as a raveled array, one can easily optimize the
    storage of "symmetric" connectivity data. One can use numpy to convert a full
    all-to-all connectivity into an upper triangular portion, and set
    ``indices='symmetric'``. This would reduce the RAM needed in half.

    The underlying data structure is an :class:`xarray.DataArray`, with a similar API to
    ``xarray``. We provide support for storing connectivity data in a subset of nodes.
    Thus the underlying data structure instead of a ``(n_nodes_in, n_nodes_out)`` 2D
    array would be a ``(n_nodes_in * n_nodes_out,)`` raveled 1D array. This allows us to
    optimize storage also for symmetric connectivity.
    """

    # whether or not the connectivity occurs over epochs
    is_epoched = False

    def __init__(
        self,
        data,
        names,
        indices,
        method,
        n_nodes,
        events=None,
        event_id=None,
        metadata=None,
        **kwargs,
    ):
        if isinstance(indices, str) and indices not in ["all", "symmetric"]:
            raise ValueError(
                'Indices can only be "all", "symmetric", or a list of tuples. '
                f"It cannot be {indices}."
            )

        # prepare metadata pandas dataframe and ensure metadata is a Pandas
        # DataFrame object
        if metadata is None:
            metadata = pd.DataFrame(dtype="float64")
        self.metadata = metadata

        # check the incoming data structure
        self._check_data_consistency(data, indices=indices, n_nodes=n_nodes)
        self._prepare_xarray(
            data,
            names=names,
            indices=indices,
            n_nodes=n_nodes,
            method=method,
            events=events,
            event_id=event_id,
            **kwargs,
        )

    def __repr__(self) -> str:
        r = f"<{self.__class__.__name__} | "

        if self.n_epochs is not None:
            r += f"n_epochs : {self.n_epochs}, "
        if "freqs" in self.dims:
            r += f"freq : [{self.freqs[0]}, {self.freqs[-1]}], "  # type: ignore
        if "times" in self.dims:
            r += f"time : [{self.times[0]}, {self.times[-1]}], "  # type: ignore
        r += f", nave : {self.n_epochs_used}"
        r += f", nodes, n_estimated : {self.n_nodes}, {self.n_estimated_nodes}"
        if "components" in self.dims:
            r += f", n_components : {len(self.coords['components'])}, "
        r += f", ~{sizeof_fmt(self._size)}"
        r += ">"
        return r

    def _get_num_connections(self, data):
        """Compute the number of estimated nodes' connectivity."""
        # account for epoch data structures
        if self.is_epoched:
            start_idx = 1
        else:
            start_idx = 0
        self.n_estimated_nodes = data.shape[start_idx]

    def _prepare_xarray(
        self, data, names, indices, n_nodes, method, events, event_id, **kwargs
    ):
        """Prepare xarray data structure."""
        # generate events and event_id that originate from Epochs class
        # which stores the windows of Raw that were used to generate
        # the corresponding connectivity data
        self._init_epochs(events, event_id, on_missing="warn")

        # set node names
        if names is None:
            names = list(map(str, range(n_nodes)))

        # the names of each first few dimensions of
        # the data depending if data is epoched or not
        if self.is_epoched:
            dims = ["epochs", "node_in -> node_out"]
        else:
            dims = ["node_in -> node_out"]

        # the coordinates of each dimension
        n_estimated_list = list(map(str, range(self.n_estimated_nodes)))
        coords = dict()
        if self.is_epoched:
            coords["epochs"] = list(map(str, range(data.shape[0])))
        coords["node_in -> node_out"] = n_estimated_list
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
        kwargs["events"] = self.events
        # kwargs['event_id'] = self.event_id

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

        # get the number of estimated nodes
        self._get_num_connections(data)
        if self.is_epoched:
            data_len = data.shape[1]
        else:
            data_len = data.shape[0]

        if isinstance(indices, tuple):
            # check that the indices passed in are of the same length
            if len(indices[0]) != len(indices[1]):
                raise ValueError(
                    "If indices are passed in then they must be the same length. They "
                    f"are right now {len(indices[0])} and {len(indices[1])}."
                )
            # indices length should match the data length
            if len(indices[0]) != data_len:
                raise ValueError(
                    f"The number of indices, {len(indices[0])} should match the "
                    f"raveled data length passed in of {data_len}."
                )

        elif indices == "symmetric":
            expected_len = ((n_nodes + 1) * n_nodes) // 2
            if data_len != expected_len:
                raise ValueError(
                    'If "indices" is "symmetric", then '
                    f"connectivity data should be the upper-triangular part of the "
                    f"matrix. There are {data_len} estimated connections. But there "
                    f"should be {expected_len} estimated connections."
                )

    def copy(self):
        return deepcopy(self)

    def get_epoch_annotations(self):
        pass

    @property
    def n_epochs(self):
        """The number of epochs the connectivity data varies over."""
        if self.is_epoched:
            n_epochs = self._data.shape[0]
        else:
            n_epochs = None
        return n_epochs

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
        indices : ``'all'`` | ``'symmetric'`` | tuple of list
            Either ``'all'`` for all-to-all connectivity, ``'symmetric'`` for symmetric
            connectivity, or a tuple of lists representing the node-to-nodes that
            connectivity was computed for.
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
    def n_epochs_used(self):
        """Number of epochs used in computation of connectivity.

        Can be ``None``, if there was no epochs used. This is equivalent to the number
        of epochs, if there is no combining of epochs.
        """
        return self.attrs.get("n_epochs_used")

    @property
    def _size(self):
        """Estimate the object size."""
        size = 0
        size += object_size(self._data)
        size += object_size(self.attrs)

        # if self.metadata is not None:
        #     size += self.metadata.memory_usage(index=True).sum()
        return size

    def get_data(self, output="compact"):
        """Get connectivity data as a numpy array.

        Parameters
        ----------
        output : ``'compact'`` | ``'raveled'`` | ``'dense'``
            How to format the output:

            - ``'raveled'`` will represent each connectivity matrix as a
              ``(..., n_nodes_in * n_nodes_out, ...)`` array
            - ``'dense'`` will return each connectivity matrix as a ``(..., n_nodes_in,
              n_nodes_out, ...)`` array
            - ``'compact'`` (default) will return ``'raveled'`` if ``indices`` were
              defined as a tuple of arrays, or ``'dense'`` if ``indices='all'``

            Multivariate connectivity data cannot be returned in a dense form.

        Returns
        -------
        data : array
            The output connectivity data.
        """
        _check_option("output", output, ["raveled", "dense", "compact"])

        if output == "compact":
            if self.indices in ["all", "symmetric"]:
                output = "dense"
            else:
                output = "raveled"

        if output == "raveled":
            data = self._data
        else:
            if (
                isinstance(self.indices, tuple)
                and not isinstance(self.indices[0], int)
                and not isinstance(self.indices[1], int)
            ):  # i.e. check if multivariate results based on nested indices
                # multivariate results cannot be returned in a dense form as a single
                # set of results would correspond to multiple entries in the matrix, and
                # there could also be cases where multiple results correspond to the
                # same entries in the matrix.
                raise ValueError(
                    "cannot return multivariate connectivity data in a dense form"
                )

            # get the new shape of the data array
            if self.is_epoched:
                new_shape = [self.n_epochs]
            else:
                new_shape = []

            # handle the case where model order is defined in VAR connectivity
            # and thus appends the connectivity matrices side by side, so the
            # shape is N x N * lags
            new_shape.extend([self.n_nodes, self.n_nodes])
            if "components" in self.dims:
                new_shape.append(len(self.coords["components"]))
            if "freqs" in self.dims:
                new_shape.append(len(self.coords["freqs"]))
            if "times" in self.dims:
                new_shape.append(len(self.coords["times"]))

            # handle things differently if indices is defined
            if isinstance(self.indices, tuple):
                # TODO: improve this to be more memory efficient
                # from all-to-all connectivity structure
                data = np.zeros(new_shape)
                data[:] = np.nan

                row_idx, col_idx = self.indices
                if self.is_epoched:
                    data[:, row_idx, col_idx, ...] = self._data
                else:
                    data[row_idx, col_idx, ...] = self._data
            elif self.indices == "symmetric":
                data = np.zeros(new_shape)

                # get the upper/lower triangular indices
                row_triu_inds, col_triu_inds = np.triu_indices(self.n_nodes, k=0)
                if self.is_epoched:
                    data[:, row_triu_inds, col_triu_inds, ...] = self._data
                    data[:, col_triu_inds, row_triu_inds, ...] = self._data
                else:
                    data[row_triu_inds, col_triu_inds, ...] = self._data
                    data[col_triu_inds, row_triu_inds, ...] = self._data
            else:
                data = self._data.reshape(new_shape)

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

    # def plot_matrix(self):
    #     pass

    # def plot_3d(self):
    #     pass

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
class SpectralConnectivity(BaseConnectivity, SpectralMixin):
    """Spectral connectivity class.

    This class stores connectivity data that varies over frequencies. The underlying
    data is an array of shape ``(n_connections, [n_components,] n_freqs)``, or
    ``(n_nodes, n_nodes, [n_components,] n_freqs)``. ``n_components`` is an optional
    dimension for multivariate methods where each connection has multiple components of
    connectivity.

    Parameters
    ----------
    %(data)s
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

    expected_n_dim = 2

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
            data,
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
class TemporalConnectivity(BaseConnectivity, TimeMixin):
    """Temporal connectivity class.

    This is an array of shape ``(n_connections, [n_components,] n_times)``, or
    ``(n_nodes, n_nodes, [n_components,] n_times)``. This describes how connectivity
    varies over time. It describes sample-by-sample time-varying connectivity (usually
    on the order of milliseconds). Here time (t=0) is the same for all connectivity
    measures. ``n_components`` is an optional dimension for multivariate methods where
    each connection has multiple components of connectivity.

    Parameters
    ----------
    %(data)s
    %(times)s
    %(n_nodes)s
    %(names)s
    %(indices)s
    %(method)s
    %(n_epochs_used)s
    %(connectivity_kwargs)s

    Notes
    -----
    :class:`mne_connectivity.EpochConnectivity` is a similar connectivity class to this
    one. However, that describes one connectivity snapshot for each epoch. These epochs
    might be chunks of time that have different meaning for time ``t=0``. Epochs can
    mean separate trials, where the beginning of the trial implies t=0. These epochs may
    also be discontiguous.
    """

    expected_n_dim = 2

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
            data,
            names=names,
            method=method,
            n_nodes=n_nodes,
            indices=indices,
            times=times,
            n_epochs_used=n_epochs_used,
            **kwargs,
        )


@fill_doc
class SpectroTemporalConnectivity(BaseConnectivity, SpectralMixin, TimeMixin):
    """Spectrotemporal connectivity class.

    This class stores connectivity data that varies over both frequency and time. The
    temporal part describes sample-by-sample time-varying connectivity (usually on the
    order of milliseconds). Note the difference relative to epochs.

    The underlying data is an array of shape ``(n_connections, [n_components,] n_freqs,
    n_times)``, or ``(n_nodes, n_nodes, [n_components,] n_freqs, n_times)``.
    ``n_components`` is an optional dimension for multivariate methods where each
    connection has multiple components of connectivity.

    Parameters
    ----------
    %(data)s
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
            data,
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
class EpochSpectralConnectivity(SpectralConnectivity):
    """Spectral connectivity class over epochs.

    This is an array of shape ``(n_epochs, n_connections, [n_components,] n_freqs)``, or
    ``(n_epochs, n_nodes, n_nodes, [n_components,] n_freqs)``. This describes how
    connectivity varies over frequencies for different epochs. ``n_components`` is an
    optional dimension for multivariate methods where each connection has multiple
    components of connectivity.

    Parameters
    ----------
    %(data)s
    %(freqs)s
    %(n_nodes)s
    %(names)s
    %(indices)s
    %(method)s
    %(spec_method)s
    %(connectivity_kwargs)s

    See Also
    --------
    mne_connectivity.spectral_connectivity_time
    """

    # whether or not the connectivity occurs over epochs
    is_epoched = True

    def __init__(
        self,
        data,
        freqs,
        n_nodes,
        names=None,
        indices="all",
        method=None,
        spec_method=None,
        **kwargs,
    ):
        super().__init__(
            data,
            freqs=freqs,
            names=names,
            indices=indices,
            n_nodes=n_nodes,
            method=method,
            spec_method=spec_method,
            **kwargs,
        )


@fill_doc
class EpochTemporalConnectivity(TemporalConnectivity):
    """Temporal connectivity class over epochs.

    This is an array of shape ``(n_epochs, n_connections, [n_components,] n_times)``, or
    ``(n_epochs, n_nodes, n_nodes, [n_components,] n_times)``. This describes how
    connectivity varies over time for different epochs. ``n_components`` is an optional
    dimension for multivariate methods where each connection has multiple components of
    connectivity.

    Parameters
    ----------
    %(data)s
    %(times)s
    %(n_nodes)s
    %(names)s
    %(indices)s
    %(method)s
    %(connectivity_kwargs)s

    See Also
    --------
    mne_connectivity.envelope_correlation
    mne_connectivity.vector_auto_regression
    """

    # whether or not the connectivity occurs over epochs
    is_epoched = True

    def __init__(
        self, data, times, n_nodes, names=None, indices="all", method=None, **kwargs
    ):
        super().__init__(
            data,
            times=times,
            names=names,
            indices=indices,
            n_nodes=n_nodes,
            method=method,
            **kwargs,
        )


@fill_doc
class EpochSpectroTemporalConnectivity(SpectroTemporalConnectivity):
    """Spectrotemporal connectivity class over epochs.

    This is an array of shape ``(n_epochs, n_connections, [n_components,] n_freqs,
    n_times)``, or ``(n_epochs, n_nodes, n_nodes, [n_components,] n_freqs, n_times)``.
    This describes how connectivity varies over frequencies and time for different
    epochs. ``n_components`` is an optional dimension for multivariate methods where
    each connection has multiple components of connectivity.

    Parameters
    ----------
    %(data)s
    %(freqs)s
    %(times)s
    %(n_nodes)s
    %(names)s
    %(indices)s
    %(method)s
    %(spec_method)s
    %(connectivity_kwargs)s
    """

    # whether or not the connectivity occurs over epochs
    is_epoched = True

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
        **kwargs,
    ):
        super().__init__(
            data,
            names=names,
            freqs=freqs,
            times=times,
            indices=indices,
            n_nodes=n_nodes,
            method=method,
            spec_method=spec_method,
            **kwargs,
        )


@fill_doc
class Connectivity(BaseConnectivity):
    """Connectivity class without frequency or time component.

    This is an array of shape ``(n_connections[, n_components])``, or ``(n_nodes,
    n_nodes[, n_components])``. This describes a connectivity matrix/graph that does not
    vary over time, frequency, or epochs. ``n_components`` is an optional dimension for
    multivariate methods where each connection has multiple components of connectivity.

    Parameters
    ----------
    %(data)s
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
            data,
            names=names,
            method=method,
            n_nodes=n_nodes,
            indices=indices,
            n_epochs_used=n_epochs_used,
            **kwargs,
        )


@fill_doc
class EpochConnectivity(BaseConnectivity):
    """Epoch connectivity class.

    This is an array of shape ``(n_epochs, n_connections[, n_components])``, or
    ``(n_epochs, n_nodes, n_nodes[, n_components])``. This describes how connectivity
    varies for different epochs. ``n_components`` is an optional dimension for
    multivariate methods where each connection has multiple components of connectivity.

    Parameters
    ----------
    %(data)s
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

    # whether or not the connectivity occurs over epochs
    is_epoched = True

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
            data,
            names=names,
            method=method,
            n_nodes=n_nodes,
            indices=indices,
            n_epochs_used=n_epochs_used,
            **kwargs,
        )
