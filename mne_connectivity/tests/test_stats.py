import numpy as np
from numpy.testing import assert_array_equal

from mne_connectivity.stats import portmanteau

def test_whiteness():
    # simulate some data
    np.random.seed(91)
    data = np.random.randn(80, 15, 100)     # gaussian white noise
    data0 = data.copy()

    # run hypothesis test
    pvalue = portmanteau(data, max_lag=20, random_state=1)

    # make sure we don't modify the input
    assert_array_equal(data, data0)    
    # test should be non-significant for white noise
    assert pvalue > 0.01

    # create cross-correlation at lag 3
    data[:, 1, 3:] = data[:, 0, :-3]          
    pvalue = portmanteau(data, max_lag=20, random_state=1)

    # now test should be significant
    assert pvalue < 0.01
