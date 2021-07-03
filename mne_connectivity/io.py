import xarray as xr

from .base import (
    TemporalConnectivity, SpectralConnectivity,
    SpectroTemporalConnectivity, EpochTemporalConnectivity,
    EpochSpectralConnectivity, EpochSpectroTemporalConnectivity
)


def read_connectivity(fname):
    # open up a data-array using xarray
    conn_da = xr.open_dataarray(fname)

    # get the data
    data = conn_da.values

    # get the names
    names = conn_da.attrs['node_names']

    # get the name of the class
    data_structure_name = conn_da.attrs.pop('data_structure')
    if data_structure_name == 'TemporalConnectivity':
        conn = TemporalConnectivity(
            data=data, names=names, **conn_da.attrs
        )
