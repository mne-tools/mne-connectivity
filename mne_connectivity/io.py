import xarray as xr

from .base import (Connectivity, EpochConnectivity, EpochSpectralConnectivity,
                   EpochSpectroTemporalConnectivity, EpochTemporalConnectivity,
                   SpectralConnectivity, SpectroTemporalConnectivity,
                   TemporalConnectivity)


def _xarray_to_conn(array, cls_func):
    """Create connectivity class from xarray.

    Parameters
    ----------
    array : xarray.DataArray
        Xarray containing the connectivity data.
    cls_func : Connectivity class
        The function of the connectivity class to use.

    Returns
    -------
    conn : instance of Connectivity class
        An instantiated connectivity class.
    """
    # get the data
    data = array.values

    # get the dimensions
    coords = array.coords

    # attach times and frequencies
    if 'times' in coords:
        array.attrs['times'] = coords.get('times')
    if 'freqs' in coords:
        array.attrs['freqs'] = coords.get('freqs')

    # get the names
    names = array.attrs['node_names']

    # create the connectivity class
    conn = cls_func(
        data=data, names=names, **array.attrs
    )
    return conn


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
    # get the name of the class
    data_structure_name = conn_da.attrs.pop('data_structure')

    # map class name to its actual class
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

    # get the data as a new connectivity container
    conn = _xarray_to_conn(conn_da, cls_func)
    return conn
