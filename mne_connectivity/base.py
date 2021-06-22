import numpy as np
from scipy.signal import spectral


class _Connectivity():
    def __init__(self, data, names, method):
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
        names : list | np.ndarray
            The names of the nodes in the connectivity data. 
        method : str
            The method name used to compute connectivity.
        """
        # check the incoming data structure
        data = self._check_data_consistency(data)

        self.data = data
        self.names = names
        self.method = method

    def _check_data_consistency(self, data):
        if data.shape[0] != data.shape[1]:
            raise RuntimeError(f'The data shape should have '
                               f'matching first two axes. '
                               f'The current shape is {data.shape}.')
        
        if not isinstance(data, np.ndarray):
            raise TypeError('Connectivity data must be stored as a '
                            'numpy array.')

        if np.ndim(data) < 2 or np.ndim(data) > 4:
            raise ValueError(f'Connectivity data that is passed '
                              'must be either 2D, 3D, or 4D, where the '
                              'last axis is time if 3D or 4D and the '
                              '3rd axis is frequency if 4D')
        return data

    @property
    def shape(self):
        return self.data.shape

    @property
    def frequency_axis(self):
        raise NotImplementedError(
            'The frequency axis is not defined here. Please '
            'double check that you are using the right type of '
            'connectivity container.')

    @property
    def time_axis(self):
        raise NotImplementedError(
            'The time axis is not defined here. Please '
            'double check that you are using the right type of '
            'connectivity container.')

    def plot_circle(self):
        pass

    def plot_matrix(self):
        pass

    def plot_3d(self):
        pass

    def save(self, fname):
        pass


class SpectralConnectivity(_Connectivity):
    def __init__(self, data, names, freqs, method=None, spec_method=None, n_epochs=None):
        super().__init__(data, names, method)

        self.freqs = freqs
        self.spec_method = spec_method
        self.n_epochs = n_epochs

    @property
    def frequency_axis(self):
        return 2

class TemporalConnectivity(_Connectivity):
    def __init__(self, data, names, times, method=None, n_epochs=None):
        super().__init__(data, names, method)

        self.times = times
        self.n_epochs = n_epochs

    @property
    def time_axis(self):
        return 2

class SpectroTemporalConnectivity(_Connectivity):
    def __init__(self, data, names, freqs, times, method=None, spec_method=None, n_epochs=None):
        super().__init__(data, names, method)

        self.freqs = freqs
        self.spec_method = spec_method
        self.times = times
        self.n_epochs = n_epochs

    @property
    def frequency_axis(self):
        return 2
    
    @property
    def frequency_axis(self):
        return 3
