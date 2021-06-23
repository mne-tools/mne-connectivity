import numpy as np

from mne.utils import sizeof_fmt, object_size


class _Connectivity():
    def __init__(self, data, names, method, indices):
        """Base class container for connectivity data.

        Connectivity data is anything represents "connections"
        between nodes as a (N, N) array. It can be symmetric, or
        asymmetric (if it is symmetric, storage optimization will
        occur). In addition,

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
        """
        self.method = method
        self.indices = indices

        # check the incoming data structure
        data = self._check_data_consistency(data, names)

        self.names = names
        self.data = data

    def _check_data_consistency(self, data, names):
        if self.dims[0] != 'epochs':
            node_in_shape = data.shape[0]
            node_out_shape = data.shape[1]
        else:
            node_in_shape = data.shape[1]
            node_out_shape = data.shape[2]
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
        return data

    @property
    def dims(self):
        return ('node_in', 'node_out')

    @property
    def shape(self):
        return self.data.shape

    @property
    def _size(self):
        """Estimate the object size."""
        size = 0
        if hasattr(self, 'data'):
            size += object_size(self.data)
        return size

    def get_data(self, return_nans=False):
        if return_nans:
            return self.data

        if self.indices is not None:
            row_idx, col_idx = self.indices
            return self.data[row_idx, col_idx, ...]
        else:
            return self.data

    def plot_circle(self):
        pass

    def plot_matrix(self):
        pass

    def plot_3d(self):
        pass

    def save(self, fname):
        pass


class SpectralConnectivity(_Connectivity):
    def __init__(self, data, freqs, names=None, indices=None, method=None,
                 spec_method=None, n_epochs=None):
        super().__init__(data, names, method, indices)

        self.freqs = freqs
        self.spec_method = spec_method
        self.n_epochs = n_epochs

    @property
    def dims(self):
        return ('node_in', 'node_out', 'freqs')

    def __repr__(self):  # noqa: D105
        s = ", freq : [%f, %f]" % (self.freqs[0], self.freqs[-1])
        s += ", nave : %d" % self.n_epochs
        s += ', nodes : %d, %d' % (self.n_nodes_in, self.n_nodes_out)
        s += ', ~%s' % (sizeof_fmt(self._size),)
        return "<SpectralConnectivity | %s>" % s


class TemporalConnectivity(_Connectivity):
    def __init__(self, data, times, names=None, indices=None,
                 method=None, n_epochs=None):
        super().__init__(data, names, method, indices)

        self.times = times
        self.n_epochs = n_epochs

    @property
    def dims(self):
        return ('node_in', 'node_out', 'times')

    def __repr__(self):  # noqa: D105
        s = "time : [%f, %f]" % (self.times[0], self.times[-1])
        s += ", nave : %d" % self.n_epochs
        s += ', nodes : %d, %d' % (self.n_nodes_in, self.n_nodes_out)
        s += ', ~%s' % (sizeof_fmt(self._size),)
        return "<TemporalConnectivity | %s>" % s


class SpectroTemporalConnectivity(_Connectivity):
    def __init__(self, data, freqs, times, names=None, indices=None,
                 method=None, spec_method=None, n_epochs=None):
        super().__init__(data, names, method, indices)

        self.freqs = freqs
        self.spec_method = spec_method
        self.times = times
        self.n_epochs = n_epochs

    @property
    def dims(self):
        return ('node_in', 'node_out', 'freqs', 'times')

    def __repr__(self):  # noqa: D105
        s = "time : [%f, %f]" % (self.times[0], self.times[-1])
        s += ", freq : [%f, %f]" % (self.freqs[0], self.freqs[-1])
        s += ", nave : %d" % self.n_epochs
        s += ', nodes : %d, %d' % (self.n_nodes_in, self.n_nodes_out)
        s += ', ~%s' % (sizeof_fmt(self._size),)
        return "<SpectroTemporalConnectivity | %s>" % s


class EpochSpectralConnectivity(SpectralConnectivity):
    def __init__(self, data, freqs, names=None, indices=None, method=None,
                 spec_method=None):
        super().__init__(
            data, names, freqs, indices=indices, method=method,
            spec_method=spec_method)

    @property
    def dims(self):
        return ('epochs', 'node_in', 'node_out', 'freqs')


class EpochTemporalConnectivity(TemporalConnectivity):
    def __init__(self, data, times, names=None, indices=None,
                 method=None):
        super().__init__(data, names, times, indices=indices,
                         method=method)

    @property
    def dims(self):
        return ('epochs', 'node_in', 'node_out', 'times')


class EpochSpectroTemporalConnectivity(SpectroTemporalConnectivity):
    def __init__(self, data, names, freqs, times, indices,
                 method, spec_method, n_epochs):
        super().__init__(
            data, names, freqs, times, indices=indices,
            method=method, spec_method=spec_method)

    @property
    def dims(self):
        return ('epochs', 'node_in', 'node_out', 'freqs', 'times')
