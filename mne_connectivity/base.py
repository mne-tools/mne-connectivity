import numpy as np
import xarray as xr

from mne.utils import sizeof_fmt, object_size


class SpectralMixin:
    @property
    def freqs(self):
        return self.xarray.coords.get('freqs').values.squeeze().tolist()


class TimeMixin:
    @property
    def times(self):
        return self.xarray.coords.get('times').values.squeeze().tolist()


class _Connectivity():
    # whether or not the connectivity occurs over epochs
    is_epoched = False

    def __init__(self, data, names, method, indices, **kwargs):
        """Base class container for connectivity data.

        Connectivity data is anything that represents "connections"
        between nodes as a (N, N) array. It can be symmetric, or
        asymmetric (if it is symmetric, storage optimization will
        occur).

        The underlying data is stored as an ``xarray.DataArray``,
        with a similar API.

        TODO: account for symmetric connectivity structures

        Parameters
        ----------
        data : np.ndarray
            The connectivity data. The first two axes should
            have matching dimensions.
        names : list | np.ndarray | None
            The names of the nodes in the connectivity data.
        method : str
            The method name used to compute connectivity.
        indices : tuple of list | None
            The list of indices with relevant data.
        kwargs : dict
            Extra connectivity parameters. These may include 
            ``freqs`` for spectral connectivity, and/or 
            ``times`` for connectivity over time. In addition,
            these may include extra parameters that are stored 
            as xarray ``attrs``.
        """
        # check the incoming data structure
        data, names = self._check_data_consistency(data, names)
        self.names = names
        self.method = method
        self.indices = indices
        self._prepare_xarray(data, names, **kwargs)

    def _prepare_xarray(self, data, names, **kwargs):
        # the names of each dimension of the data
        if self.is_epoched:
            dims = ['epochs']
        else:
            dims = []
        dims.extend(['node_in', 'node_out'])

        # the coordinates of each dimension
        coords = {
            "node_in": names,
            "node_out": names,
        }
        if self.is_epoched:
            coords['epochs'] = list(map(str, range(data.shape[0])))
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

        # create xarray object
        xarray_obj = xr.DataArray(
            data=data,
            coords=coords,
            dims=dims,
            attrs=kwargs
        )

        self._obj = xarray_obj

    def _check_data_consistency(self, data, names):
        if self.is_epoched:
            node_in_shape = data.shape[1]
            node_out_shape = data.shape[2]
        else:
            node_in_shape = data.shape[0]
            node_out_shape = data.shape[1]

        self.n_nodes_in = node_in_shape
        self.n_nodes_out = node_out_shape
        if node_in_shape != node_out_shape:
            raise RuntimeError(f'The data shape should have '
                               f'matching first two axes. '
                               f'The current shape is {data.shape}.')

        if not isinstance(data, np.ndarray):
            raise TypeError('Connectivity data must be stored as a '
                            'numpy array.')

        if np.ndim(data) < 2 or np.ndim(data) > 4:
            raise ValueError(f'Connectivity data that is passed '
                             f'must be either 2D, 3D, or 4D, where the '
                             f'last axis is time if 3D or 4D and the '
                             f'3rd axis is frequency if 4D. The shape '
                             f'passed was {data.shape}.')

        if names is None:
            names = list(map(str, range(node_in_shape)))

        if len(names) != node_in_shape:
            raise ValueError(f'The number of names passed in ({len(names)}) '
                             f'must match the number of nodes in the '
                             f'connectivity data ({node_in_shape}).')
        return data, names

    @property
    def dims(self):
        return self.xarray.dims

    @property
    def attrs(self):
        return self.xarray.attrs

    @property
    def shape(self):
        return self.xarray.shape

    @property
    def data(self):
        return self.xarray.values

    @property
    def xarray(self):
        return self._obj

    @property
    def n_epochs(self):
        return self.attrs.get('n_epochs_used')

    @property
    def _size(self):
        """Estimate the object size."""
        size = 0
        size += object_size(self.data)
        size += object_size(self.attrs)
        return size

    def get_data(self, return_nans=False, squeeze=True):
        if return_nans:
            data = self.data

        if self.indices is not None:
            row_idx, col_idx = self.indices
            data = self.data[row_idx, col_idx, ...]
        else:
            data = self.data

        if squeeze:
            return data.squeeze()
        return data

    def rename_nodes(self, from_mapping, to_mapping='auto',
                     allow_duplicates=False):
        pass

    def plot_circle(self):
        pass

    def plot_matrix(self):
        pass

    def plot_3d(self):
        pass

    def save(self, fname):
        pass


class SpectralConnectivity(_Connectivity, SpectralMixin):
    def __init__(self, data, freqs, names=None, indices=None, method=None,
                 spec_method=None, n_epochs_used=None, **kwargs):
        super().__init__(data, names=names, method=method, indices=indices,
                         freqs=freqs, spec_method=spec_method,
                         n_epochs_used=n_epochs_used, **kwargs)

    def __repr__(self):  # noqa: D105
        s = ", freq : [%f, %f]" % (self.freqs[0], self.freqs[-1])
        s += ", nave : %d" % self.n_epochs
        s += ', nodes : %d, %d' % (self.n_nodes_in, self.n_nodes_out)
        s += ', ~%s' % (sizeof_fmt(self._size),)
        return "<SpectralConnectivity | %s>" % s


class TemporalConnectivity(_Connectivity, TimeMixin):
    def __init__(self, data, times, names=None, indices=None,
                 method=None, n_epochs_used=None, **kwargs):
        super().__init__(data, names=names, method=method, indices=indices,
                         times=times, n_epochs_used=n_epochs_used,
                         **kwargs)

    def __repr__(self):  # noqa: D105
        s = "time : [%f, %f]" % (self.times[0], self.times[-1])
        s += ", nave : %d" % self.n_epochs
        s += ', nodes : %d, %d' % (self.n_nodes_in, self.n_nodes_out)
        s += ', ~%s' % (sizeof_fmt(self._size),)
        return "<TemporalConnectivity | %s>" % s


class SpectroTemporalConnectivity(_Connectivity, SpectralMixin, TimeMixin):
    def __init__(self, data, freqs, times, names=None, indices=None,
                 method=None, spec_method=None, n_epochs_used=None, **kwargs):
        super().__init__(data, names=names, method=method, indices=indices,
                         freqs=freqs, spec_method=spec_method,
                         times=times, n_epochs_used=n_epochs_used, **kwargs)

    def __repr__(self):  # noqa: D105
        s = "time : [%f, %f]" % (self.times[0], self.times[-1])
        s += ", freq : [%f, %f]" % (self.freqs[0], self.freqs[-1])
        s += ", nave : %d" % self.n_epochs
        s += ', nodes : %d, %d' % (self.n_nodes_in, self.n_nodes_out)
        s += ', ~%s' % (sizeof_fmt(self._size),)
        return "<SpectroTemporalConnectivity | %s>" % s


class EpochSpectralConnectivity(SpectralConnectivity):
    # whether or not the connectivity occurs over epochs
    is_epoched = True

    def __init__(self, data, freqs, names=None, indices=None, method=None,
                 spec_method=None, **kwargs):
        super().__init__(
            data, freqs=freqs, names=names, indices=indices, method=method,
            spec_method=spec_method, **kwargs)

    def __repr__(self):  # noqa: D105
        s = "time : [%f, %f]" % (self.times[0], self.times[-1])
        s += ", freq : [%f, %f]" % (self.freqs[0], self.freqs[-1])
        s += ", n_epochs : %d" % self.n_epochs
        s += ', nodes : %d, %d' % (self.n_nodes_in, self.n_nodes_out)
        s += ', ~%s' % (sizeof_fmt(self._size),)
        return "<EpochSpectralConnectivity | %s>" % s


class EpochTemporalConnectivity(TemporalConnectivity):
    # whether or not the connectivity occurs over epochs
    is_epoched = True

    def __init__(self, data, times, names=None, indices=None, method=None, **kwargs):
        super().__init__(data, times=times, names=names, indices=indices,
                         method=method, **kwargs)

    def __repr__(self):  # noqa: D105
        s = "time : [%f, %f]" % (self.times[0], self.times[-1])
        s += ", freq : [%f, %f]" % (self.freqs[0], self.freqs[-1])
        s += ", n_epochs : %d" % self.n_epochs
        s += ', nodes : %d, %d' % (self.n_nodes_in, self.n_nodes_out)
        s += ', ~%s' % (sizeof_fmt(self._size),)
        return "<EpochTemporalConnectivity | %s>" % s


class EpochSpectroTemporalConnectivity(SpectroTemporalConnectivity):
    # whether or not the connectivity occurs over epochs
    is_epoched = True

    def __init__(self, data, names, freqs, times, indices,
                 method, spec_method, **kwargs):
        super().__init__(
            data, names=names, freqs=freqs, times=times, indices=indices,
            method=method, spec_method=spec_method, **kwargs)

    def __repr__(self):  # noqa: D105
        s = "time : [%f, %f]" % (self.times[0], self.times[-1])
        s += ", freq : [%f, %f]" % (self.freqs[0], self.freqs[-1])
        s += ", n_epochs : %d" % self.n_epochs
        s += ', nodes : %d, %d' % (self.n_nodes_in, self.n_nodes_out)
        s += ', ~%s' % (sizeof_fmt(self._size),)
        return "<EpochSpectroTemporalConnectivity | %s>" % s
