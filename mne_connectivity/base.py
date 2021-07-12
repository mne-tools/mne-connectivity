from copy import copy

import numpy as np
import xarray as xr
from mne.utils import (_check_combine, _check_option, _validate_type,
                       copy_function_doc_to_method_doc, object_size,
                       sizeof_fmt)

from mne_connectivity.utils import fill_doc
from mne_connectivity.viz import plot_connectivity_circle


class SpectralMixin:
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

        fun = _check_combine(combine, valid=('mean', 'median'))

        # apply function over the dataset
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


@fill_doc
class _Connectivity():
    """Base class for connectivity data.

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
    kwargs : dict
        Extra connectivity parameters. These may include
        ``freqs`` for spectral connectivity, and/or
        ``times`` for connectivity over time. In addition,
        these may include extra parameters that are stored
        as xarray ``attrs``.

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
                 n_nodes, **kwargs):

        if isinstance(indices, str) and \
                indices not in ['all', 'symmetric']:
            raise ValueError(f'Indices can only be '
                             f'"all", otherwise '
                             f'should be a list of tuples. '
                             f'It cannot be {indices}.')

        # check the incoming data structure
        self._check_data_consistency(data, indices=indices, n_nodes=n_nodes)
        self._prepare_xarray(data, names=names, indices=indices,
                             n_nodes=n_nodes, method=method, **kwargs)

    def __repr__(self) -> str:
        r = f'<{self.__class__.__name__} | '

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

    def _get_n_estimated_nodes(self, data):
        """Compute the number of estimated nodes' connectivity."""
        # account for epoch data structures
        if self.is_epoched:
            start_idx = 1
            self.n_epochs = data.shape[0]
        else:
            self.n_epochs = None
            start_idx = 0
        self.n_estimated_nodes = data.shape[start_idx]

    def _prepare_xarray(self, data, names, indices, n_nodes, method,
                        **kwargs):
        """Prepare xarray data structure.

        Parameters
        ----------
        data : [type]
            [description]
        names : [type]
            [description]
        """
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
        self._get_n_estimated_nodes(data)
        if self.is_epoched:
            data_len = data.shape[1]
        else:
            data_len = data.shape[0]

        # check that the indices passed in are of the same length
        if isinstance(indices, tuple):
            if len(indices[0]) != len(indices[1]):
                raise ValueError(f'If indices are passed in '
                                 f'then they must be the same '
                                 f'length. They are right now '
                                 f'{len(indices[0])} and '
                                 f'{len(indices[1])}.')
        elif indices == 'symmetric':
            expected_len = ((n_nodes + 1) * n_nodes) // 2
            if data_len != expected_len:
                raise ValueError(f'If "indices" is "symmetric", then '
                                 f'connectivity data should be the '
                                 f'upper-triangular part of the matrix. There '
                                 f'are {data_len} estimated connections. '
                                 f'But there should be {expected_len} '
                                 f'estimated connections.')

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
        """Attributes of connectivity dataset.

        See ``xarray``'s ``attrs``.
        """
        return self.xarray.attrs

    @property
    def shape(self):
        """Shape of raveled connectivity."""
        return self.xarray.shape

    @property
    def n_nodes(self):
        """The number of nodes in the dataset.

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

        Can be 'None', if there was no epochs used.
        """
        return self.attrs.get('n_epochs_used')

    @property
    def _size(self):
        """Estimate the object size."""
        size = 0
        size += object_size(self.get_data())
        size += object_size(self.attrs)
        return size

    def get_data(self, output='raveled'):
        """Get connectivity data as a numpy array.

        Parameters
        ----------
        output : str, optional
            How to format the output, by default 'raveled', which
            will represent each connectivity matrix as a
            ``(n_nodes_in * n_nodes_out,)`` list. If 'dense', then
            will return each connectivity matrix as a 2D array.
        squeeze : bool, optional
            Whether to squeeze the array or not, by default True.

        Returns
        -------
        data : np.ndarray
            The output connectivity data.
        """
        _check_option('output', output, ['raveled', 'dense'])

        if output == 'raveled':
            data = self._data
        else:
            # get the new shape of the data array
            if self.is_epoched:
                new_shape = [self.n_epochs]
            else:
                new_shape = []
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
            elif self.indices == 'symmetric':
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
        names = self.names

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

        # netCDF does not support 'None'
        # so map these to 'n/a'
        for key, val in self.attrs.items():
            if val is None:
                self.attrs[key] = 'n/a'

        # save as a netCDF file
        # note this requires the netcdf4 python library
        self.xarray.to_netcdf(fname, mode='w', format='NETCDF4',
                              engine='netcdf4')

        # re-set old attributes
        self.xarray.attrs = old_attrs


@fill_doc
class SpectralConnectivity(_Connectivity, SpectralMixin):
    """Spectral connectivity container.

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
    """

    def __init__(self, data, freqs, n_nodes, names=None,
                 indices='all', method=None, spec_method=None,
                 n_epochs_used=None, **kwargs):
        super().__init__(data, names=names, method=method,
                         indices=indices, n_nodes=n_nodes,
                         freqs=freqs, spec_method=spec_method,
                         n_epochs_used=n_epochs_used, **kwargs)


@fill_doc
class TemporalConnectivity(_Connectivity, TimeMixin):
    """Temporal connectivity container.

    Parameters
    ----------
    %(data)s
    %(times)s
    %(n_nodes)s
    %(names)s
    %(indices)s
    %(method)s
    %(n_epochs_used)s
    """

    def __init__(self, data, times, n_nodes, names=None, indices='all',
                 method=None, n_epochs_used=None, **kwargs):
        super().__init__(data, names=names, method=method,
                         n_nodes=n_nodes, indices=indices,
                         times=times, n_epochs_used=n_epochs_used,
                         **kwargs)


@fill_doc
class SpectroTemporalConnectivity(_Connectivity, SpectralMixin, TimeMixin):
    """Spectrotemporal connectivity container.

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
    """

    def __init__(self, data, freqs, times, n_nodes, names=None,
                 indices='all', method=None,
                 spec_method=None, n_epochs_used=None, **kwargs):
        super().__init__(data, names=names, method=method, indices=indices,
                         n_nodes=n_nodes, freqs=freqs,
                         spec_method=spec_method, times=times,
                         n_epochs_used=n_epochs_used, **kwargs)


@fill_doc
class EpochSpectralConnectivity(SpectralConnectivity, EpochMixin):
    """Spectral connectivity container over Epochs.

    Parameters
    ----------
    %(data)s
    %(freqs)s
    %(n_nodes)s
    %(names)s
    %(indices)s
    %(method)s
    %(spec_method)s
    """
    # whether or not the connectivity occurs over epochs
    is_epoched = True

    def __init__(self, data, freqs, n_nodes, names=None,
                 indices='all', method=None,
                 spec_method=None, **kwargs):
        super().__init__(
            data, freqs=freqs, names=names, indices=indices,
            n_nodes=n_nodes, method=method,
            spec_method=spec_method, **kwargs)


@fill_doc
class EpochTemporalConnectivity(TemporalConnectivity, EpochMixin):
    """Temporal connectivity container over Epochs.

    Parameters
    ----------
    %(data)s
    %(times)s
    %(n_nodes)s
    %(names)s
    %(indices)s
    %(method)s
    """
    # whether or not the connectivity occurs over epochs
    is_epoched = True

    def __init__(self, data, times, n_nodes, names=None,
                 indices='all', method=None, **kwargs):
        super().__init__(data, times=times, names=names,
                         indices=indices, n_nodes=n_nodes,
                         method=method, **kwargs)


@fill_doc
class EpochSpectroTemporalConnectivity(
    SpectroTemporalConnectivity, EpochMixin
):
    """Spectrotemporal connectivity container over Epochs.

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
    """
    # whether or not the connectivity occurs over epochs
    is_epoched = True

    def __init__(self, data, freqs, times, n_nodes,
                 names=None, indices='all', method=None,
                 spec_method=None, **kwargs):
        super().__init__(
            data, names=names, freqs=freqs, times=times, indices=indices,
            n_nodes=n_nodes, method=method, spec_method=spec_method,
            **kwargs)


@fill_doc
class Connectivity(_Connectivity, EpochMixin):
    """Connectivity container without frequency or time component.

    Parameters
    ----------
    %(data)s
    %(n_nodes)s
    %(names)s
    %(indices)s
    %(method)s
    %(n_epochs_used)s
    """

    def __init__(self, data, n_nodes, names=None, indices='all',
                 method=None, n_epochs_used=None, **kwargs):
        super().__init__(data, names=names, method=method,
                         n_nodes=n_nodes, indices=indices,
                         n_epochs_used=n_epochs_used,
                         **kwargs)


@fill_doc
class EpochConnectivity(_Connectivity, EpochMixin):
    """Epoch connectivity container.

    Parameters
    ----------
    %(data)s
    %(n_nodes)s
    %(names)s
    %(indices)s
    %(method)s
    %(n_epochs_used)s
    """

    # whether or not the connectivity occurs over epochs
    is_epoched = True

    def __init__(self, data, n_nodes, names=None, indices='all',
                 method=None, n_epochs_used=None, **kwargs):
        super().__init__(data, names=names, method=method,
                         n_nodes=n_nodes, indices=indices,
                         n_epochs_used=n_epochs_used,
                         **kwargs)
