import xarray as xr

from .base import (
    EpochConnectivity, Connectivity,
    TemporalConnectivity, SpectralConnectivity,
    SpectroTemporalConnectivity, EpochTemporalConnectivity,
    EpochSpectralConnectivity, EpochSpectroTemporalConnectivity
)


def read_connectivity(fname):
    """Read connectivity data from netCDF file.
    Parameters
    ----------
    fname : str | pathlib.Path
        The filepath.
    Returns
    -------
    conn : instance of Connectivity
        A connectivity class.
    """
    # open up a data-array using xarray
    conn_da = xr.open_dataarray(fname)

    # map 'n/a' to 'None'
    for key, val in conn_da.attrs.items():
        if not isinstance(val, list):
            if val == 'n/a':
                conn_da.attrs[key] = None

    # get the data
    data = conn_da.values

    # get the dimensions
    coords = conn_da.coords

    # attach times and frequencies
    if 'times' in coords:
        conn_da.attrs['times'] = coords.get('times')
    if 'freqs' in coords:
        conn_da.attrs['freqs'] = coords.get('freqs')

    # get the names
    names = conn_da.attrs['node_names']

    # get the name of the class
    data_structure_name = conn_da.attrs.pop('data_structure')

    conn_cls = {
        'Connectivity': Connectivity,
        'TemporalConnectivity': TemporalConnectivity,
        'SpectralConnectivity': SpectralConnectivity,
        'SpectroTemporalConnectivity': SpectroTemporalConnectivity,
        'EpochConnectivity': EpochConnectivity,
        'EpochTemporalConnectivity': EpochTemporalConnectivity,
        'EpochSpectralConnectivity': EpochSpectralConnectivity,
        'EpochSpectroTemporalConnectivity': EpochSpectroTemporalConnectivity
    }

    cls_func = conn_cls[data_structure_name]

    conn = cls_func(
        data=data, names=names, **conn_da.attrs
    )
    return conn
