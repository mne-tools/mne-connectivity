import numpy as np


class Connectivity():

    def __init__(self, data, names):
        # check the incoming data structure
        self._check_data_consistency(data)

        self.data = data
        self.names = names

    def _check_data_consistency(self, data):
        if np.ndim(data) < 2 or np.ndim > 3:
            raise ValueError(f'Connectivity data that is passed
                             must be either 2D, or 3D, where the
                             last axis is time if 3D.')

    @property
    def time_resolved(self):
        if np.ndim(self.data) == 2:
            return False
        else:
            return True
