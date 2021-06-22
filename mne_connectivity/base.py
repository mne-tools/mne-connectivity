import numpy as np


class _Connectivity():
    def __init__(self, data, names):
        """Base class container for connectivity data.

        Connectivity data is anything represents "connections" 
        between nodes as a (N, N) array. It can be symmetric, or 
        asymmetric (if it is symmetric, storage optimization will 
        occur). In addition, 

        Parameters
        ----------
        data : [type]
            [description]
        names : [type]
            [description]
        """
        # check the incoming data structure
        data = self._check_data_consistency(data)

        self.data = data
        self.names = names

    def _check_data_consistency(self, data):
        if not isinstance(data, np.ndarray):
            raise TypeError('Connectivity data must be stored as a '
                            'numpy array.')

        if np.ndim(data) < 2 or np.ndim > 4:
            raise ValueError(f'Connectivity data that is passed '
                              'must be either 2D, 3D, or 4D, where the '
                              'last axis is time if 3D or 4D and the '
                              '3rd axis is frequency if 4D')

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
    def __init__(self, data, names, freqs):
        super().__init__(data, names)

        self.freqs = freqs
        
    @property
    def frequency_axis(self):
        return 2

class TemporalConnectivity(_Connectivity):
    def __init__(self, data, names, times):
        super().__init__(data, names)

        self.times = times

    @property
    def time_axis(self):
        return 2

class SpectroTemporalConnectivity(_Connectivity):
    def __init__(self, data, names, freqs, times):
        super().__init__(data, names)

        self.freqs = freqs
        self.times = times

    @property
    def frequency_axis(self):
        return 2
    
    @property
    def frequency_axis(self):
        return 3
