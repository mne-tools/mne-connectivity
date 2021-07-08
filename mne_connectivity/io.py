import xarray as xr

from .base import (
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

    # get the data
    data = conn_da.values

    # get the names
    names = conn_da.attrs['node_names']

    # get the name of the class
    data_structure_name = conn_da.attrs.pop('data_structure')

    conn_cls = {
        'TemporalConnectivity': TemporalConnectivity,
        'SpectralConnectivity': SpectralConnectivity,
        'SpectroTemporalConnectivity': SpectroTemporalConnectivity,
        'EpochTemporalConnectivity': EpochTemporalConnectivity,
        'EpochSpectralConnectivity': EpochSpectralConnectivity,
        'EpochSpectroTemporalConnectivity': EpochSpectroTemporalConnectivity
    }

    cls_func = conn_cls[data_structure_name]

    conn = cls_func(
        data=data, names=names, **conn_da.attrs
    )
    return conn
