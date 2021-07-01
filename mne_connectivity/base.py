import numpy as np
import xarray as xr

from mne.utils import sizeof_fmt, object_size, _validate_type


class SpectralMixin:
    @property
    def freqs(self):
        return self.xarray.coords.get('freqs').values.tolist()


class TimeMixin:
    @property
    def times(self):
        return self.xarray.coords.get('times').values.tolist()


class _Connectivity():
    # whether or not the connectivity occurs over epochs
    is_epoched = False

    def __init__(self, data, names, indices, method,
                 n_nodes, **kwargs):
        """Base class container for connectivity data.

        Connectivity data is anything that represents "connections"
        between nodes as a (N, N) array. It can be symmetric, or
        asymmetric (if it is symmetric, storage optimization will
        occur).

        The underlying data structure is an ``xarray.DataArray``,
        with a similar API. We provide support for storing
        connectivity data in a subset of nodes. Thus the underlying
        data structure instead of a ``(n_nodes_in, n_nodes_out)`` 2D array
        would be a ``(n_nodes_in * n_nodes_out,)`` raveled 1D array. This
        allows us to optimize storage also for symmetric connectivity.

        Parameters
        ----------
        data : np.ndarray ([epochs], n_estimated_nodes, [freqs], [times])
            The connectivity data that is a raveled array of
            ``(n_estimated_nodes, ...)`` shape. The
            ``n_estimated_nodes`` is equal to
            ``n_nodes_in * n_nodes_out`` if one is computing
            the full connectivity, or a subset of nodes
            equal to the length of ``indices`` passed in.
        names : list | np.ndarray | None
            The names of the nodes that we would consider in the
            connectivity data.
        indices : tuple of arrays | str | None
            The indices of relevant connectivity data. If ``'all'`` (default),
            then data is connectivity between all nodes. If ``'symmetric'``,
            then data is symmetric connectivity between all nodes. If a tuple,
            then the first list represents the "in nodes", and the second list
            represents the "out nodes". See "Notes" for more information.
        method : str
            The method name used to compute connectivity.
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
        """
        if isinstance(indices, str) and \
                indices not in ['all', 'symmetric']:
            raise ValueError(f'Indices {indices} can only be '
                             f'"all", or "symmetric", otherwise '
                             f'should be a list of tuples.')

        # check the incoming data structure
        self.method = method
        self.indices = indices
        self.n_nodes = n_nodes
        self._check_data_consistency(data)
        self._prepare_xarray(data, names=names, **kwargs)
        self._check()

    def _get_n_estimated_nodes(self, data):
        # account for epoch data structures
        if self.is_epoched:
            start_idx = 1
            self.n_epochs = data.shape[0]
        else:
            self.n_epochs = None
            start_idx = 0
        self.n_estimated_nodes = data.shape[start_idx]

        return data

    def _prepare_xarray(self, data, names, **kwargs):
        # get the number of estimated nodes
        data = self._get_n_estimated_nodes(data)

        # set node names
        if names is None:
            names = list(map(str, range(self.n_nodes)))

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

        # create xarray object
        xarray_obj = xr.DataArray(
            data=data,
            coords=coords,
            dims=dims,
            attrs=kwargs
        )
        self._obj = xarray_obj

    def _check(self):
        if len(self.names) != self.n_nodes:
            raise ValueError(f'The number of names passed in '
                             f'({len(self.names)}) '
                             f'must match the number of nodes in the '
                             f'original data ({self.n_nodes}).')

    def _check_data_consistency(self, data):
        if self.is_epoched:
            if data.ndim < 2 or data.ndim > 4:
                raise RuntimeError(f'Data using an Epoched data '
                                   f'structure should have at least '
                                   f'2 dimensions and at most 4 '
                                   f'dimensions. Your data was '
                                   f'{data.shape} shape.')
        else:
            if data.ndim > 3:
                raise RuntimeError(f'Data not using an Epoched data '
                                   f'structure should have at least '
                                   f'1 dimensions and at most 3 '
                                   f'dimensions. Your data was '
                                   f'{data.shape} shape.')

        # check that the indices passed in are of the same length
        if isinstance(self.indices, tuple):
            if len(self.indices[0]) != len(self.indices[1]):
                raise ValueError(f'If indices are passed in '
                                 f'then they must be the same '
                                 f'length. They are right now '
                                 f'{len(self.indices[0])} and '
                                 f'{len(self.indices[1])}.')

        if not isinstance(data, np.ndarray):
            raise TypeError('Connectivity data must be passed in as a '
                            'numpy array.')

    @property
    def _data(self):
        return self.xarray.values

    @property
    def dims(self):
        return self.xarray.dims

    @property
    def coords(self):
        return self.xarray.coords

    @property
    def attrs(self):
        return self.xarray.attrs

    @property
    def shape(self):
        return self.xarray.shape

    @property
    def names(self):
        return self.attrs['node_names']

    @property
    def xarray(self):
        return self._obj

    @property
    def n_epochs_used(self):
        return self.attrs.get('n_epochs_used')

    @property
    def _size(self):
        """Estimate the object size."""
        size = 0
        size += object_size(self.get_data())
        size += object_size(self.attrs)
        return size

    def get_data(self, output='raveled', squeeze=True):
        """Get connectivity data as a numpy array.

        Parameters
        ----------
        output : str, optional
            How to format the output, by default 'raveled', which
            will represent each connectivity matrix as a
            ``(n_nodes_in * n_nodes_out, 1)`` list. If 'full', then
            will return each connectivity matrix as a 2D array.
        squeeze : bool, optional
            Whether to squeeze the array or not, by default True.

        Returns
        -------
        data : np.ndarray
        """
        if output not in ['raveled', 'full']:
            raise ValueError(f'Output of data can only be one of '
                             f'"raveled, "full", not {output}.')

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
                # form all-to-all connectivity structure
                data = np.zeros(new_shape)
                data[:] = np.nan

                row_idx, col_idx = self.indices
                if self.is_epoched:
                    data[:, row_idx, col_idx, ...] = self._data
                else:
                    data[row_idx, col_idx, ...] = self._data
            else:
                data = self._data.reshape(new_shape)

        if squeeze:
            return data.squeeze()

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

    def plot_circle(self):
        pass

    def plot_matrix(self):
        pass

    def plot_3d(self):
        pass

    def save(self, fname):
        pass


class SpectralConnectivity(_Connectivity, SpectralMixin):
    def __init__(self, data, freqs, n_nodes, names=None,
                 indices=None, method=None, spec_method=None,
                 n_epochs_used=None, **kwargs):
        super().__init__(data, names=names, method=method,
                         indices=indices, n_nodes=n_nodes,
                         freqs=freqs, spec_method=spec_method,
                         n_epochs_used=n_epochs_used, **kwargs)

    def __repr__(self):  # noqa: D105
        s = ", freq : [%f, %f]" % (self.freqs[0], self.freqs[-1])
        s += ", nave : %d" % self.n_epochs_used
        s += ', nodes, n_estimated : %d, %d' % (self.n_nodes,
                                                self.n_estimated_nodes)
        s += ', ~%s' % (sizeof_fmt(self._size),)
        return "<SpectralConnectivity | %s>" % s


class TemporalConnectivity(_Connectivity, TimeMixin):
    def __init__(self, data, times, n_nodes, names=None, indices=None,
                 method=None, n_epochs_used=None, **kwargs):
        super().__init__(data, names=names, method=method,
                         n_nodes=n_nodes, indices=indices,
                         times=times, n_epochs_used=n_epochs_used,
                         **kwargs)

    def __repr__(self):  # noqa: D105
        s = "time : [%f, %f]" % (self.times[0], self.times[-1])
        s += ", nave : %d" % self.n_epochs_used
        s += ', nodes, n_estimated : %d, %d' % (self.n_nodes,
                                                self.n_estimated_nodes)
        s += ', ~%s' % (sizeof_fmt(self._size),)
        return "<TemporalConnectivity | %s>" % s


class SpectroTemporalConnectivity(_Connectivity, SpectralMixin, TimeMixin):
    def __init__(self, data, freqs, times, n_nodes, names=None,
                 indices=None, method=None,
                 spec_method=None, n_epochs_used=None, **kwargs):
        super().__init__(data, names=names, method=method, indices=indices,
                         n_nodes=n_nodes, freqs=freqs,
                         spec_method=spec_method, times=times,
                         n_epochs_used=n_epochs_used, **kwargs)

    def __repr__(self):  # noqa: D105
        s = "time : [%f, %f]" % (self.times[0], self.times[-1])
        s += ", freq : [%f, %f]" % (self.freqs[0], self.freqs[-1])
        s += ", nave : %d" % self.n_epochs_used
        s += ', nodes, n_estimated : %d, %d' % (self.n_nodes,
                                                self.n_estimated_nodes)
        s += ', ~%s' % (sizeof_fmt(self._size),)
        return "<SpectroTemporalConnectivity | %s>" % s


class EpochSpectralConnectivity(SpectralConnectivity):
    # whether or not the connectivity occurs over epochs
    is_epoched = True

    def __init__(self, data, freqs, n_nodes, names=None,
                 indices=None, method=None,
                 spec_method=None, **kwargs):
        super().__init__(
            data, freqs=freqs, names=names, indices=indices,
            n_nodes=n_nodes, method=method,
            spec_method=spec_method, **kwargs)

    def __repr__(self):  # noqa: D105
        s = "time : [%f, %f]" % (self.times[0], self.times[-1])
        s += ", freq : [%f, %f]" % (self.freqs[0], self.freqs[-1])
        s += ", n_epochs : %d" % self.n_epochs_used
        s += ', nodes, n_estimated : %d, %d' % (self.n_nodes,
                                                self.n_estimated_nodes)
        s += ', ~%s' % (sizeof_fmt(self._size),)
        return "<EpochSpectralConnectivity | %s>" % s


class EpochTemporalConnectivity(TemporalConnectivity):
    # whether or not the connectivity occurs over epochs
    is_epoched = True

    def __init__(self, data, times, n_nodes, names=None,
                 indices=None, method=None, **kwargs):
        super().__init__(data, times=times, names=names,
                         indices=indices, n_nodes=n_nodes,
                         method=method, **kwargs)

    def __repr__(self):  # noqa: D105
        s = "time : [%f, %f]" % (self.times[0], self.times[-1])
        s += ", freq : [%f, %f]" % (self.freqs[0], self.freqs[-1])
        s += ", n_epochs : %d" % self.n_epochs_used
        s += ', nodes, n_estimated : %d, %d' % (self.n_nodes,
                                                self.n_estimated_nodes)
        s += ', ~%s' % (sizeof_fmt(self._size),)
        return "<EpochTemporalConnectivity | %s>" % s


class EpochSpectroTemporalConnectivity(SpectroTemporalConnectivity):
    # whether or not the connectivity occurs over epochs
    is_epoched = True

    def __init__(self, data, freqs, times, n_nodes,
                 method, spec_method, names=None,
                 indices=None, **kwargs):
        super().__init__(
            data, names=names, freqs=freqs, times=times, indices=indices,
            n_nodes=n_nodes, method=method, spec_method=spec_method,
            **kwargs)

    def __repr__(self):  # noqa: D105
        s = "time : [%f, %f]" % (self.times[0], self.times[-1])
        s += ", freq : [%f, %f]" % (self.freqs[0], self.freqs[-1])
        s += ", n_epochs : %d" % self.n_epochs_used
        s += ', nodes, n_estimated : %d, %d' % (self.n_nodes,
                                                self.n_estimated_nodes)
        s += ', ~%s' % (sizeof_fmt(self._size),)
        return "<EpochSpectroTemporalConnectivity | %s>" % s
