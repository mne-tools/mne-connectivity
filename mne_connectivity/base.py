from copy import copy, deepcopy

import numpy as np
import xarray as xr
import pandas as pd
from mne.utils import (_check_combine, _check_option, _validate_type,
                       copy_function_doc_to_method_doc, object_size,
                       sizeof_fmt, _check_event_id, _ensure_events,
                       _on_missing, warn, check_random_state)

from mne_connectivity.utils import (
    fill_doc, _prepare_xarray_mne_data_structures)
from mne_connectivity.viz import plot_connectivity_circle


class SpectralMixin:
    """Mixin class for spectral connectivities.

    Note: In mne-connectivity, we associate the word
    "spectral" with time-frequency. Reference to
    eigenvalue structure is not captured in this mixin.
    """
    @property
    def freqs(self):
        """The frequency points of the connectivity data.

        If these are computed over a frequency band, it will
        be the median frequency of the frequency band.
        """
        return self.xarray.coords.get('freqs').values.tolist()


class TimeMixin:
    @property
    def times(self):
        """The time points of the connectivity data."""
        return self.xarray.coords.get('times').values.tolist()


class EpochMixin:
    def _init_epochs(self, events, event_id, on_missing='warn') -> None:
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
                    msg = ('No matching events found for %s '
                           '(event id %i)' % (key, val))
                    _on_missing(on_missing, msg)

            # ensure metadata matches original events size
            self.selection = np.arange(len(events))
            self.events = events
            del events

            values = list(self.event_id.values())
            selected = np.where(np.in1d(self.events[:, 2], values))[0]

            self.events = self.events[selected]

    def append(self, epoch_conn):
        """Append another connectivity structure.

        Parameters
        ----------
        epoch_conn : instance of Connectivity
            The Epoched Connectivity class to append.

        Returns
        -------
        self : instance of Connectivity
            The altered Epoched Connectivity class.
        """
        if not isinstance(self, type(epoch_conn)):
            raise ValueError(f'The type of the epoch connectivity to append '
                             f'is {type(epoch_conn)}, which does not match '
                             f'{type(self)}.')
        if hasattr(self, 'times'):
            if not np.allclose(self.times, epoch_conn.times):
                raise ValueError('Epochs must have same times')
        if hasattr(self, 'freqs'):
            if not np.allclose(self.freqs, epoch_conn.freqs):
                raise ValueError('Epochs must have same frequencies')

        events = list(deepcopy(self.events))
        event_id = deepcopy(self.event_id)
        metadata = copy(self.metadata)

        # compare event_id
        common_keys = list(set(event_id).intersection(
            set(epoch_conn.event_id)))
        for key in common_keys:
            if not event_id[key] == epoch_conn.event_id[key]:
                msg = ('event_id values must be the same for identical keys '
                       'for all concatenated epochs. Key "{}" maps to {} in '
                       'some epochs and to {} in others.')
                raise ValueError(msg.format(key, event_id[key],
                                            epoch_conn.event_id[key]))

        evs = epoch_conn.events.copy()
        if epoch_conn.n_epochs == 0:
            warn('Epoch Connectivity object to append was empty.')
        event_id.update(epoch_conn.event_id)
        events = np.concatenate((events, evs), axis=0)
        metadata = pd.concat([epoch_conn.metadata, metadata])

        # now combine the xarray data, altered events and event ID
        self._obj = xr.concat([self.xarray, epoch_conn.xarray], dim='epochs')
        self.events = events
        self.event_id = event_id
        return self

    def combine(self, combine='mean'):
        """Combine connectivity data over epochs.

        Parameters
        ----------
        combine : 'mean' | 'median' | callable
            How to combine correlation estimates across epochs.
            Default is 'mean'. If callable, it must accept one
            positional input. For example::

                combine = lambda data: np.median(data, axis=0)

        Returns
        -------
        conn : instance of Connectivity
            The combined connectivity data structure.
        """
        from .io import _xarray_to_conn

        if not self.is_epoched:
            raise RuntimeError('Combine only works over Epoched connectivity. '
                               f'It does not work with {self}')

        fun = _check_combine(combine, valid=('mean', 'median'))

        # get a copy of metadata into attrs as a dictionary
        self = _prepare_xarray_mne_data_structures(self)

        # apply function over the  array
        new_xr = xr.apply_ufunc(fun, self.xarray,
                                input_core_dims=[['epochs']],
                                vectorize=True)
        new_xr.attrs = self.xarray.attrs

        # map class name to its actual class
        conn_cls = {
            'EpochConnectivity': Connectivity,
            'EpochTemporalConnectivity': TemporalConnectivity,
            'EpochSpectralConnectivity': SpectralConnectivity,
            'EpochSpectroTemporalConnectivity': SpectroTemporalConnectivity
        }
        cls_func = conn_cls[self.__class__.__name__]

        # convert new xarray to non-Epoch data structure
        conn = _xarray_to_conn(new_xr, cls_func)
        return conn


class DynamicMixin:
    def is_stable(self):
        companion_mat = self.companion
        return np.abs(np.linalg.eigvals(companion_mat)).max() < 1.

    def eigvals(self):
        return np.linalg.eigvals(self.companion)

    @property
    def companion(self):
        """Generate block companion matrix.

        Returns the data matrix if the model is VAR(1).
        """
        from .vector_ar.utils import _block_companion

        lags = self.attrs.get('lags')
        data = self.get_data()
        if lags == 1:
            return data

        arrs = []
        for idx in range(self.n_epochs):
            blocks = _block_companion(
                [data[idx, ..., jdx]for jdx in range(lags)]
            )
            arrs.append(blocks)
        return arrs

    def predict(self, data):
        """Predict samples on actual data.

        The result of this function is used for calculating the residuals.

        Parameters
        ----------
        data : array
            Epoched or continuous data set. Has shape
            (n_epochs, n_signals, n_times) or (n_signals, n_times).

        Returns
        -------
        predicted : array
            Data as predicted by the VAR model of
            shape same as ``data``.

        Notes
        -----
        Residuals are obtained by r = x - var.predict(x).

        To compute residual covariances::

            # compute the covariance of the residuals
            # row are observations, columns are variables
            t = residuals.shape[0]
            sampled_residuals = np.concatenate(
                np.split(residuals[:, :, lags:], t, 0),
                axis=2
            ).squeeze(0)
            rescov = np.cov(sampled_residuals)
        """
        if data.ndim < 2 or data.ndim > 3:
            raise ValueError(f'Data passed in must be either 2D or 3D. '
                             f'The data you passed in has {data.ndim} dims.')
        if data.ndim == 2 and self.is_epoched:
            raise RuntimeError('If there is a VAR model over epochs, '
                               'one must pass in a 3D array.')
        if data.ndim == 3 and not self.is_epoched:
            raise RuntimeError('If there is a single VAR model, '
                               'one must pass in a 2D array.')

        # make the data 3D
        if data.ndim == 2:
            data = data[np.newaxis, ...]

        n_epochs, _, n_times = data.shape
        var_model = self.get_data(output='dense')

        # get the model order
        lags = self.attrs.get('lags')

        # predict the data by applying forward model
        predicted_data = np.zeros(data.shape)
        # which takes less loop iterations
        if n_epochs > n_times - lags:
            for idx in range(1, lags + 1):
                for jdx in range(lags, n_times):
                    if self.is_epoched:
                        bp = var_model[jdx, :, (idx - 1)::lags]
                    else:
                        bp = var_model[:, (idx - 1)::lags]
                    predicted_data[:, :,
                                   jdx] += np.dot(data[:, :, jdx - idx], bp.T)
        else:
            for idx in range(1, lags + 1):
                for jdx in range(n_epochs):
                    if self.is_epoched:
                        bp = var_model[jdx, :, (idx - 1)::lags]
                    else:
                        bp = var_model[:, (idx - 1)::lags]
                    predicted_data[jdx, :, lags:] += \
                        np.dot(
                            bp,
                            data[jdx, :, (lags - idx):(n_times - idx)]
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
        noise_func : func, optional
            This function is used to create the generating noise process. If
            set to None, Gaussian white noise with zero mean and unit variance
            is used.
        %(random_state)s

        Returns
        -------
        data : array, shape (n_samples, n_channels)
            Generated data.
        """
        var_model = self.get_data(output='dense')
        if self.is_epoched:
            var_model = var_model.mean(axis=0)

        n_nodes = self.n_nodes
        lags = self.attrs.get('lags')

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
                data[jdx, :] += \
                    var_model[:, (idx - 1)::lags].dot(
                        data[jdx - idx, :]
                )

        # self.residuals = res[10 * lags:, :, :].T
        # self.rescov = sp.cov(cat_trials(self.residuals).T, rowvar=False)
        return data[10 * lags:, :].transpose()


@fill_doc
class BaseConnectivity(DynamicMixin, EpochMixin):
    """Base class for connectivity data.

    This class should not be instantiated directly, but should be used
    to do type-checking. All connectivity classes will be returned from
    corresponding connectivity computing functions.

    Connectivity data is anything that represents "connections"
    between nodes as a (N, N) array. It can be symmetric, or
    asymmetric (if it is symmetric, storage optimization will
    occur).

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
        The metadata data frame that would come from the :class:`mne.Epochs`
        class. See :class:`mne.Epochs` docstring for details.
    %(connectivity_kwargs)s

    Notes
    -----
    Connectivity data can be generally represented as a square matrix
    with values intending the connectivity function value between two
    nodes. We optimize storage of symmetric connectivity data
    and allow support for computing connectivity data on a subset of nodes.
    We store connectivity data as a raveled ``(n_estimated_nodes, ...)``
    where ``n_estimated_nodes`` can be ``n_nodes_in * n_nodes_out`` if a
    full connectivity structure is computed, or a subset of the nodes
    (equal to the length of the indices passed in).

    Since we store connectivity data as a raveled array, one can
    easily optimize the storage of "symmetric" connectivity data.
    One can use numpy to convert a full all-to-all connectivity
    into an upper triangular portion, and set ``indices='symmetric'``.
    This would reduce the RAM needed in half.

    The underlying data structure is an ``xarray.DataArray``,
    with a similar API to ``xarray``. We provide support for storing
    connectivity data in a subset of nodes. Thus the underlying
    data structure instead of a ``(n_nodes_in, n_nodes_out)`` 2D array
    would be a ``(n_nodes_in * n_nodes_out,)`` raveled 1D array. This
    allows us to optimize storage also for symmetric connectivity.
    """
    # whether or not the connectivity occurs over epochs
    is_epoched = False

    def __init__(self, data, names, indices, method,
                 n_nodes, events=None, event_id=None,
                 metadata=None, **kwargs):

        if isinstance(indices, str) and \
                indices not in ['all', 'symmetric']:
            raise ValueError(f'Indices can only be '
                             f'"all", otherwise '
                             f'should be a list of tuples. '
                             f'It cannot be {indices}.')

        # prepare metadata pandas dataframe and ensure metadata is a Pandas
        # DataFrame object
        if metadata is None:
            metadata = pd.DataFrame(dtype='float64')
        self.metadata = metadata

        # check the incoming data structure
        self._check_data_consistency(data, indices=indices, n_nodes=n_nodes)
        self._prepare_xarray(data, names=names, indices=indices,
                             n_nodes=n_nodes, method=method, events=events,
                             event_id=event_id, **kwargs)

    def __repr__(self) -> str:
        r = f'<{self.__class__.__name__} | '

        if self.n_epochs is not None:
            r += f"n_epochs : {self.n_epochs}, "
        if 'freqs' in self.dims:
            r += "freq : [%f, %f], " % (self.freqs[0], self.freqs[-1])
        if 'times' in self.dims:
            r += "time : [%f, %f], " % (self.times[0], self.times[-1])
        r += f", nave : {self.n_epochs_used}"
        r += f', nodes, n_estimated : {self.n_nodes}, ' \
             f'{self.n_estimated_nodes}'
        r += ', ~%s' % (sizeof_fmt(self._size),)
        r += '>'
        return r

    def _get_num_connections(self, data):
        """Compute the number of estimated nodes' connectivity."""
        # account for epoch data structures
        if self.is_epoched:
            start_idx = 1
        else:
            start_idx = 0
        self.n_estimated_nodes = data.shape[start_idx]

    def _prepare_xarray(self, data, names, indices, n_nodes, method,
                        events, event_id, **kwargs):
        """Prepare xarray data structure."""
        # generate events and event_id that originate from Epochs class
        # which stores the windows of Raw that were used to generate
        # the corresponding connectivity data
        self._init_epochs(events, event_id, on_missing='warn')

        # set node names
        if names is None:
            names = list(map(str, range(n_nodes)))

        # the names of each first few dimensions of
        # the data depending if data is epoched or not
        if self.is_epoched:
            dims = ['epochs', 'node_in -> node_out']
        else:
            dims = ['node_in -> node_out']

        # the coordinates of each dimension
        n_estimated_list = list(map(str, range(self.n_estimated_nodes)))
        coords = dict()
        if self.is_epoched:
            coords['epochs'] = list(map(str, range(data.shape[0])))
        coords["node_in -> node_out"] = n_estimated_list
        if 'freqs' in kwargs:
            coords['freqs'] = kwargs.pop('freqs')
            dims.append('freqs')
        if 'times' in kwargs:
            times = kwargs.pop('times')
            if times is None:
                times = list(range(data.shape[-1]))
            coords['times'] = list(times)
            dims.append('times')

        # convert all numpy arrays to lists
        for key, val in kwargs.items():
            if isinstance(val, np.ndarray):
                kwargs[key] = val.tolist()
        kwargs['node_names'] = names

        # set method, indices and n_nodes
        if isinstance(indices, tuple):
            new_indices = (list(indices[0]), list(indices[1]))
            indices = new_indices
        kwargs['method'] = method
        kwargs['indices'] = indices
        kwargs['n_nodes'] = n_nodes
        kwargs['events'] = self.events
        # kwargs['event_id'] = self.event_id

        # create xarray object
        xarray_obj = xr.DataArray(
            data=data,
            coords=coords,
            dims=dims,
            attrs=kwargs
        )
        self._obj = xarray_obj

    def _check_data_consistency(self, data, indices, n_nodes):
        """Perform data input checks."""
        if not isinstance(data, np.ndarray):
            raise TypeError('Connectivity data must be passed in as a '
                            'numpy array.')

        if self.is_epoched:
            if data.ndim < 2 or data.ndim > 4:
                raise RuntimeError(f'Data using an epoched data '
                                   f'structure should have at least '
                                   f'2 dimensions and at most 4 '
                                   f'dimensions. Your data was '
                                   f'{data.shape} shape.')
        else:
            if data.ndim > 3:
                raise RuntimeError(f'Data not using an epoched data '
                                   f'structure should have at least '
                                   f'1 dimensions and at most 3 '
                                   f'dimensions. Your data was '
                                   f'{data.shape} shape.')

        # get the number of estimated nodes
        self._get_num_connections(data)
        if self.is_epoched:
            data_len = data.shape[1]
        else:
            data_len = data.shape[0]

        if isinstance(indices, tuple):
            # check that the indices passed in are of the same length
            if len(indices[0]) != len(indices[1]):
                raise ValueError(f'If indices are passed in '
                                 f'then they must be the same '
                                 f'length. They are right now '
                                 f'{len(indices[0])} and '
                                 f'{len(indices[1])}.')
            # indices length should match the data length
            if len(indices[0]) != data_len:
                raise ValueError(
                    f'The number of indices, {len(indices[0])} '
                    f'should match the raveled data length passed '
                    f'in of {data_len}.')

        elif isinstance(indices, str) and indices == 'symmetric':
            expected_len = ((n_nodes + 1) * n_nodes) // 2
            if data_len != expected_len:
                raise ValueError(f'If "indices" is "symmetric", then '
                                 f'connectivity data should be the '
                                 f'upper-triangular part of the matrix. There '
                                 f'are {data_len} estimated connections. '
                                 f'But there should be {expected_len} '
                                 f'estimated connections.')

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

        Even if ``indices`` defines a subset of nodes that
        were computed, this should be the total number of
        nodes in the original dataset.
        """
        return self.attrs['n_nodes']

    @property
    def method(self):
        """The method used to compute connectivity."""
        return self.attrs['method']

    @property
    def indices(self):
        """Indices of connectivity data.

        Returns
        -------
        indices : str | tuple of lists
            Either 'all' for all-to-all connectivity,
            'symmetric' for symmetric all-to-all connectivity,
            or a tuple of lists representing the node-to-nodes
            that connectivity was computed.
        """
        return self.attrs['indices']

    @property
    def names(self):
        """Node names."""
        return self.attrs['node_names']

    @property
    def xarray(self):
        """Xarray of the connectivity data."""
        return self._obj

    @property
    def n_epochs_used(self):
        """Number of epochs used in computation of connectivity.

        Can be 'None', if there was no epochs used. This is
        equivalent to the number of epochs, if there is no
        combining of epochs.
        """
        return self.attrs.get('n_epochs_used')

    @property
    def _size(self):
        """Estimate the object size."""
        size = 0
        size += object_size(self._data)
        size += object_size(self.attrs)

        # if self.metadata is not None:
        #     size += self.metadata.memory_usage(index=True).sum()
        return size

    def get_data(self, output='compact'):
        """Get connectivity data as a numpy array.

        Parameters
        ----------
        output : str, optional
            How to format the output, by default 'raveled', which
            will represent each connectivity matrix as a
            ``(n_nodes_in * n_nodes_out,)`` list. If 'dense', then
            will return each connectivity matrix as a 2D array. If 'compact'
            (default) then will return 'raveled' if ``indices`` were defined as
            a list of tuples, or ``dense`` if indices is 'all'.

        Returns
        -------
        data : np.ndarray
            The output connectivity data.
        """
        _check_option('output', output, ['raveled', 'dense', 'compact'])

        if output == 'compact':
            if isinstance(self.indices, str) and \
                self.indices in ['all', 'symmetric']:
                output = 'dense'
            else:
                output = 'raveled'

        if output == 'raveled':
            data = self._data
        else:
            # get the new shape of the data array
            if self.is_epoched:
                new_shape = [self.n_epochs]
            else:
                new_shape = []

            # handle the case where model order is defined in VAR connectivity
            # and thus appends the connectivity matrices side by side, so the
            # shape is N x N * lags
            new_shape.extend([self.n_nodes, self.n_nodes])
            if 'freqs' in self.dims:
                new_shape.append(len(self.coords['freqs']))
            if 'times' in self.dims:
                new_shape.append(len(self.coords['times']))

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
            elif isinstance(self.indices, str) and self.indices == 'symmetric':
                data = np.zeros(new_shape)

                # get the upper/lower triangular indices
                row_triu_inds, col_triu_inds = np.triu_indices(
                    self.n_nodes, k=0)
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
                    "%s" % np.array(orig_names)[np.array(missing)])
            new_names = [(names.index(name), new_name)
                         for name, new_name in mapping.items()]
        elif callable(mapping):
            new_names = [(ci, mapping(name))
                         for ci, name in enumerate(names)]
        else:
            raise ValueError('mapping must be callable or dict, not %s'
                             % (type(mapping),))

        # check we got all strings out of the mapping
        for new_name in new_names:
            _validate_type(new_name[1], 'str', 'New name mappings')

        # do the remapping locally
        for c_ind, new_name in new_names:
            names[c_ind] = new_name

        # check that all the channel names are unique
        if len(names) != len(np.unique(names)):
            raise ValueError(
                'New channel names are not unique, renaming failed')

        # rename the new names
        self._obj.attrs['node_names'] = names

    @copy_function_doc_to_method_doc(plot_connectivity_circle)
    def plot_circle(self, **kwargs):
        plot_connectivity_circle(
            self.get_data(),
            node_names=self.names,
            indices=self.indices, **kwargs)

    # def plot_matrix(self):
    #     pass

    # def plot_3d(self):
    #     pass

    def save(self, fname):
        """Save connectivity data to disk.

        Parameters
        ----------
        fname : str | pathlib.Path
            The filepath to save the data. Data is saved
            as netCDF files (``.nc`` extension).
        """
        method = self.method
        indices = self.indices
        n_nodes = self.n_nodes

        # create a copy of the old attributes
        old_attrs = copy(self.attrs)

        # assign these to xarray's attrs
        self.attrs['method'] = method
        self.attrs['indices'] = indices
        self.attrs['n_nodes'] = n_nodes

        # save the name of the connectivity structure
        self.attrs['data_structure'] = str(self.__class__.__name__)

        # get a copy of metadata into attrs as a dictionary
        self = _prepare_xarray_mne_data_structures(self)

        # netCDF does not support 'None'
        # so map these to 'n/a'
        for key, val in self.attrs.items():
            if val is None:
                self.attrs[key] = 'n/a'

        # save as a netCDF file
        # note this requires the netcdf4 python library
        # and h5netcdf library.
        # The engine specified requires the ability to save
        # complex data types, which was not natively supported
        # in xarray. Therefore, h5netcdf is the only engine
        # to support that feature at this moment.
        self.xarray.to_netcdf(fname, mode='w',
                              format='NETCDF4',
                              engine='h5netcdf',
                              invalid_netcdf=True)

        # re-set old attributes
        self.xarray.attrs = old_attrs


@fill_doc
class SpectralConnectivity(BaseConnectivity, SpectralMixin):
    """Spectral connectivity class.

    This class stores connectivity data that varies over
    frequencies. The underlying data is an array of shape
    (n_connections, n_freqs), or (n_nodes, n_nodes, n_freqs).

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
    mne_connectivity.spectral_connectivity_epochs
    """
    expected_n_dim = 2

    def __init__(self, data, freqs, n_nodes, names=None,
                 indices='all', method=None, spec_method=None,
                 n_epochs_used=None, **kwargs):
        super(SpectralConnectivity, self).__init__(
            data, names=names, method=method,
            indices=indices, n_nodes=n_nodes,
            freqs=freqs, spec_method=spec_method,
            n_epochs_used=n_epochs_used, **kwargs)


@fill_doc
class TemporalConnectivity(BaseConnectivity, TimeMixin):
    """Temporal connectivity class.

    This is an array of shape (n_connections, n_times),
    or (n_nodes, n_nodes, n_times). This describes how connectivity
    varies over time. It describes sample-by-sample time-varying
    connectivity (usually on the order of milliseconds). Here
    time (t=0) is the same for all connectivity measures.

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
    `mne_connectivity.EpochConnectivity` is a similar connectivity
    class to this one. However, that describes one connectivity snapshot
    for each epoch. These epochs might be chunks of time that have
    different meaning for time ``t=0``. Epochs can mean separate trials,
    where the beginning of the trial implies t=0. These Epochs may
    also be discontiguous.
    """
    expected_n_dim = 2

    def __init__(self, data, times, n_nodes, names=None, indices='all',
                 method=None, n_epochs_used=None, **kwargs):
        super(TemporalConnectivity, self).__init__(
            data, names=names, method=method,
            n_nodes=n_nodes, indices=indices,
            times=times, n_epochs_used=n_epochs_used,
            **kwargs)


@fill_doc
class SpectroTemporalConnectivity(BaseConnectivity, SpectralMixin, TimeMixin):
    """Spectrotemporal connectivity class.

    This class stores connectivity data that varies over both frequency
    and time. The temporal part describes sample-by-sample time-varying
    connectivity (usually on the order of milliseconds). Note the
    difference relative to Epochs.

    The underlying data is an array of shape (n_connections, n_freqs,
    n_times), or (n_nodes, n_nodes, n_freqs, n_times).

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
    """

    def __init__(self, data, freqs, times, n_nodes, names=None,
                 indices='all', method=None,
                 spec_method=None, n_epochs_used=None, **kwargs):
        super(SpectroTemporalConnectivity, self).__init__(
            data, names=names, method=method, indices=indices,
            n_nodes=n_nodes, freqs=freqs,
            spec_method=spec_method, times=times,
            n_epochs_used=n_epochs_used, **kwargs)


@fill_doc
class EpochSpectralConnectivity(SpectralConnectivity):
    """Spectral connectivity class over Epochs.

    This is an array of shape (n_epochs, n_connections, n_freqs),
    or (n_epochs, n_nodes, n_nodes, n_freqs). This describes how
    connectivity varies over frequencies for different epochs.

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
    """
    # whether or not the connectivity occurs over epochs
    is_epoched = True

    def __init__(self, data, freqs, n_nodes, names=None,
                 indices='all', method=None,
                 spec_method=None, **kwargs):
        super(EpochSpectralConnectivity, self).__init__(
            data, freqs=freqs, names=names, indices=indices,
            n_nodes=n_nodes, method=method,
            spec_method=spec_method, **kwargs)


@fill_doc
class EpochTemporalConnectivity(TemporalConnectivity):
    """Temporal connectivity class over Epochs.

    This is an array of shape (n_epochs, n_connections, n_times),
    or (n_epochs, n_nodes, n_nodes, n_times). This describes how
    connectivity varies over time for different epochs.

    Parameters
    ----------
    %(data)s
    %(times)s
    %(n_nodes)s
    %(names)s
    %(indices)s
    %(method)s
    %(connectivity_kwargs)s
    """
    # whether or not the connectivity occurs over epochs
    is_epoched = True

    def __init__(self, data, times, n_nodes, names=None,
                 indices='all', method=None, **kwargs):
        super(EpochTemporalConnectivity, self).__init__(
            data, times=times, names=names,
            indices=indices, n_nodes=n_nodes,
            method=method, **kwargs)


@fill_doc
class EpochSpectroTemporalConnectivity(SpectroTemporalConnectivity):
    """Spectrotemporal connectivity class over Epochs.

    This is an array of shape (n_epochs, n_connections, n_freqs, n_times),
    or (n_epochs, n_nodes, n_nodes, n_freqs, n_times). This describes how
    connectivity varies over frequencies and time for different epochs.

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

    def __init__(self, data, freqs, times, n_nodes,
                 names=None, indices='all', method=None,
                 spec_method=None, **kwargs):
        super(EpochSpectroTemporalConnectivity, self).__init__(
            data, names=names, freqs=freqs, times=times, indices=indices,
            n_nodes=n_nodes, method=method, spec_method=spec_method,
            **kwargs)

class BaseMultivariateConnectivity(BaseConnectivity):
    """Base class for multivariate connectivity data.

    This class should not be instantiated directly, but should be used
    to do type-checking.
    """

    _pad_val = np.inf # used to pad ragged xarray attributes before saving
    # until they are no longer ragged, at which point they can be saved with
    # HDF5 (np.inf is chosen as it should not appear in the xarray attributes;
    # if it is, an error will be raised when trying to save)

    def _add_multivariate_attrs(self, topographies, n_components, n_lags):
        """Add multivariate connectivity-specific attributes to the object."""
        if topographies is not None:
            self._check_topographies_consistency(topographies)
        self.attrs['topographies'] = topographies

        if n_components is not None:
            self._check_n_components_consistency(n_components)
        self.attrs['n_components'] = n_components

        self.attrs['n_lags'] = n_lags

    def _check_topographies_consistency(self, topographies):
        """Perform topographies input checks."""
        data = self.get_data()

        if not isinstance(topographies, np.ndarray):
            raise TypeError(
                'Topographies must be passed in as a numpy array.'
            )

        for topographies_group in topographies:
            for topo_data in topographies_group:
                if topo_data.ndim not in [2, 3]:
                    raise RuntimeError(
                        'Topographies should have either 3 or 4 dimensions '
                        '(connections, channels, frequencies, [timepoints]). '
                        f'Your topographies have {topo_data.ndim + 1} '
                        'dimensions.'
                    )
        
        if len(topographies[0]) != len(topographies[1]):
            raise ValueError(
                'If topographies are passed in then they must be the same '
                f'length. They are right now {len(topographies[0])} and '
                f'{len(topographies[1])}.'
            )
        
        group_names = ["seeds", "targets"]
        for group_i, topographies_group in enumerate(topographies):
            if len(topographies_group) != len(data):
                raise ValueError(
                    'If topographies are passed in then they must have the '
                    f'same number of connections ({len(topographies_group)}) '
                    f'as the connectivity data ({len(data)}).'
                )
            for con_i, topo_data in enumerate(topographies_group):
                if self.indices is not None and topo_data.shape[0] != \
                len(self.indices[group_i][con_i]):
                    raise ValueError(
                        'If topographies are passed in then the values for '
                        'each connection must have the same number of entries '
                        'as there are channels in the corresponding indices. '
                        f'For the {group_names[group_i]}, connection {con_i}, '
                        f'the topographies have {topo_data.shape[0]} entries, '
                        'but the indices contain '
                        f'{len(self.indices[group_i][con_i])} channels.'
                    )
                if topo_data.shape[1:] != data[con_i].shape:
                    raise ValueError(
                        'If topographies are passed in then the values for '
                        'each channel of each connection must have the same '
                        'dimensions as the connectivity data. For the '
                        f'{group_names[group_i]}, connection {con_i}, the '
                        f'topographies have shape {topo_data.shape[1:]}, but '
                        'the connectivity data has dimensions '
                        f'{data[con_i].shape}.'
                    )

    def _check_n_components_consistency(self, n_components):
        """Perform n_components input checks."""
        # useful for converting back to a tuple when re-loading after saving
        if isinstance(n_components, np.ndarray):
            n_components = tuple(copy(n_components.tolist()))
        elif isinstance(n_components, list):
            n_components = tuple(copy(n_components))

        if not isinstance(n_components, tuple):
            raise TypeError('n_components should be a tuple')

        if len(n_components) != 2:
            raise ValueError('n_components should be a tuple of two lists')

        for group in n_components:
            if not isinstance(group, list):
                raise TypeError('n_components should contain two lists')

        if len(n_components[0]) != len(n_components[1]):
            raise ValueError(
                'the seed and target portions of n_components must have an '
                'equal length'
            )

    @property
    def topographies(self):
        """Connectivity topographies."""
        return self.attrs['topographies']
    
    @property
    def n_components(self):
        """Number of components used for the SVD in the connectivity
        computation."""
        return self.attrs['n_components']

    @property
    def n_lags(self):
        """Number of lags used when computing connectivity."""
        return self.attrs['n_lags']

    def save(self, fname):
        """Save connectivity data to disk.

        Parameters
        ----------
        fname : str | pathlib.Path
            The filepath to save the data. Data is saved
            as netCDF files (``.nc`` extension).
        """
        old_attrs = deepcopy(self.attrs)
        self._pad_ragged_attrs()
        self._replace_none_n_components()
        super(BaseMultivariateConnectivity, self).save(fname)
        self.xarray.attrs = old_attrs # resets to non-padded attrs

    def _pad_ragged_attrs(self):
        """Pads ragged attributes of the connectivity object (i.e. indices and
        topographies) with np.inf until they are no longer ragged, at which
        point they can be saved using HDF5."""
        max_n_channels = self._get_max_n_channels()
        self._pad_indices(max_n_channels)
        if self.topographies is not None:
            self._pad_topographies(max_n_channels)
    
    def _get_max_n_channels(self):
        """Finds the highest number of channels involved in any one
        connection."""
        max_n_channels = 0
        for group in self.indices:
            for con in group:
                if len(con) > max_n_channels:
                    max_n_channels = len(con)
        
        return max_n_channels

    def _pad_indices(self, max_n_channels):
        """Pads indices for seeds and targets with np.inf until they are no
        longer ragged (i.e. the length of indices for each connection equals
        'max_n_channels')."""
        padded_indices = [[], []]
        for group_i, group in enumerate(self.indices):
            for con_i, con in enumerate(group):
                if np.count_nonzero(con == self._pad_val):
                    # this would break the unpadding process when re-loading the
                    # connectivity object
                    raise ValueError(
                        'the connectivity object cannot be saved with '
                        f'{self._pad_val} in the indices, as index values '
                        'should be ints'
                    )
                padded_indices[group_i].append(con)
                len_diff = max_n_channels - len(con)
                if len_diff != 0:
                    padded_indices[group_i][con_i].extend(
                        [self._pad_val for _ in range(len_diff)]
                    )
        
        self.attrs['indices'] = tuple(padded_indices)
    
    def _pad_topographies(self, max_n_channels):
        """Pads topographies for seeds and targets with np.inf until they are no
        longer ragged (i.e. the length of the first dimension of topographies
        for each connection equals 'max_n_channels')."""
        topos_dims = [2, len(self.indices[0]), max_n_channels, len(self.freqs)]
        if 'times' in self.coords:
            topos_dims.append(len(self.times))
        padded_topos = np.full(
            topos_dims, self._pad_val, dtype=self.topographies[0][0].dtype
        )

        for group_i, group in enumerate(self.topographies):
            for con_i, con in enumerate(group):
                padded_topos[group_i][con_i][:con.shape[0]] = con
                    
        self.attrs['topographies'] = padded_topos

    def _replace_none_n_components(self):
        """Replace None values in the n_components attribute with 'n/a', since
        None is not supported by netCDF."""
        n_components = [[], []]
        for group_i, group in enumerate(self.attrs['n_components']):
            for con in group:
                if con is None:
                    n_components[group_i].append('n/a')
                else:
                    n_components[group_i].append(con)
        self.attrs['n_components'] = tuple(n_components)

    def _restore_attrs(self):
        """Unpads ragged attributes of the connectivity object (i.e. indices and
        topographies) padded with np.inf and restored nested None values in
        attributes replaced with 'n/a' so that they could be saved using
        HDF5."""
        n_padded_channels = self._get_n_padded_channels()
        self._unpad_indices(n_padded_channels)
        if self.topographies is not None:
            self._unpad_topographies(n_padded_channels)
        
        self._restore_non_n_components()

    def _get_n_padded_channels(self):
        """Finds the number of channels that have been added when padding the
        seed and target indices and topographies based on the number of padded
        entries (np.inf) in each connection."""
        n_padded_channels = np.zeros((2, len(self.indices[0])), dtype=np.int32)
        for group_i, group in enumerate(self.indices):
            for con_i, con in enumerate(group):
                n_padded_channels[group_i][con_i] = np.count_nonzero(
                    con == self._pad_val
                )
        
        return n_padded_channels
    
    def _unpad_indices(self, n_padded_channels):
        """Removes entries in the indices for each connection added when the
        indices were padded with np.inf before saving."""
        unpadded_indices = [[], []]
        for group_i, group in enumerate(self.indices):
            for con_i, con in enumerate(group):
                if n_padded_channels[group_i][con_i] != 0:
                    unpadded_con = con[:-n_padded_channels[group_i][con_i]]
                else:
                    unpadded_con = con
                unpadded_indices[group_i].append(
                    [int(idx) for idx in unpadded_con]
                )
        
        self.attrs['indices'] = tuple(unpadded_indices)
    
    def _unpad_topographies(self, n_padded_channels):
        """Removes entries in the topographies for each connection added when
        the topographies were padded with np.inf before saving."""
        unpadded_topos = np.empty((2, len(self.indices[0])), dtype=object)
        for group_i, group in enumerate(self.topographies):
            for con_i, con in enumerate(group):
                if n_padded_channels[group_i][con_i] != 0:
                    unpadded_topos[group_i][con_i] = con[
                        :-n_padded_channels[group_i][con_i]
                    ]
                else:
                    unpadded_topos[group_i][con_i] = con
        
        self.attrs['topographies'] = unpadded_topos

    def _restore_non_n_components(self):
        """Restores None values in the n_components attribute with from
        'n/a'."""
        n_components = [[], []]
        for group_i, group in enumerate(self.attrs['n_components']):
            for con in group:
                if con == 'n/a':
                    n_components[group_i].append(None)
                else:
                    n_components[group_i].append(con)
        self.attrs['n_components'] = tuple(n_components)


class MultivariateSpectralConnectivity(
    SpectralConnectivity, BaseMultivariateConnectivity
):
    """Multivariate spectral connectivity class.

    This class stores multivariate connectivity data that varies over
    frequencies. The underlying data is an array of shape (n_connections,
    n_freqs).

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
    mne_connectivity.multivariate_spectral_connectivity_epochs
    """

    def __init__(
        self, data, freqs, n_nodes, names=None, indices=None, method=None,
        spec_method=None, n_epochs_used=None, topographies=None,
        n_components=None, n_lags=None, **kwargs
    ):
        super(MultivariateSpectralConnectivity, self).__init__(
            data=data, names=names, method=method, indices=indices,
            n_nodes=n_nodes, freqs=freqs, spec_method=spec_method,
            n_epochs_used=n_epochs_used, **kwargs
        )
        super(MultivariateSpectralConnectivity, self)._add_multivariate_attrs(
            topographies=topographies, n_components=n_components, n_lags=n_lags
        )


class MultivariateSpectroTemporalConnectivity(
    SpectroTemporalConnectivity, BaseMultivariateConnectivity
):
    """Multivariate spectrotemporal connectivity class.

    This class stores multivariate connectivity data that varies over both
    frequency and time. The temporal part describes sample-by-sample
    time-varying connectivity (usually on the order of milliseconds). Note the
    difference relative to Epochs.

    The underlying data is an array of shape (n_connections, n_freqs, n_times).

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
    mne_connectivity.multivariate_spectral_connectivity_epochs
    """

    def __init__(
        self, data, freqs, n_nodes, names=None, indices=None, method=None,
        spec_method=None, times=None, n_epochs_used=None, topographies=None,
        n_components=None, n_lags=None, **kwargs
    ):
        super(MultivariateSpectroTemporalConnectivity, self).__init__(
            data=data, names=names, method=method, indices=indices,
            n_nodes=n_nodes, freqs=freqs, spec_method=spec_method, times=times,
            n_epochs_used=n_epochs_used, **kwargs
        )
        super(
            MultivariateSpectroTemporalConnectivity, self
        )._add_multivariate_attrs(
            topographies=topographies, n_components=n_components, n_lags=n_lags
        )
        


@fill_doc
class Connectivity(BaseConnectivity):
    """Connectivity class without frequency or time component.

    This is an array of shape (n_connections,),
    or (n_nodes, n_nodes). This describes a connectivity matrix/graph
    that does not vary over time, frequency, or epochs.

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
    mne_connectivity.envelope_correlation
    """

    def __init__(self, data, n_nodes, names=None, indices='all',
                 method=None, n_epochs_used=None, **kwargs):
        super(Connectivity, self).__init__(data, names=names, method=method,
                                           n_nodes=n_nodes, indices=indices,
                                           n_epochs_used=n_epochs_used,
                                           **kwargs)


@fill_doc
class EpochConnectivity(BaseConnectivity):
    """Epoch connectivity class.

    This is an array of shape (n_epochs, n_connections),
    or (n_epochs, n_nodes, n_nodes). This describes how
    connectivity varies for different epochs.

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
    mne_connectivity.envelope_correlation
    """

    # whether or not the connectivity occurs over epochs
    is_epoched = True

    def __init__(self, data, n_nodes, names=None, indices='all',
                 method=None, n_epochs_used=None, **kwargs):
        super(EpochConnectivity, self).__init__(
            data, names=names, method=method,
            n_nodes=n_nodes, indices=indices,
            n_epochs_used=n_epochs_used,
            **kwargs)
