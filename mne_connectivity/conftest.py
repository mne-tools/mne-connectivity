# -*- coding: utf-8 -*-
# Author: Adam Li <adam2392@gmail.com>
#
# License: BSD-3-Clause

import pytest
import os
import warnings
from distutils.version import LooseVersion


@pytest.fixture(autouse=True)
def close_all():
    """Close all matplotlib plots, regardless of test status."""
    # This adds < 1 µS in local testing, and we have ~2500 tests, so ~2 ms max
    import matplotlib.pyplot as plt
    yield
    plt.close('all')


@pytest.fixture(scope='session')
def matplotlib_config():
    """Configure matplotlib for viz tests."""
    import matplotlib
    from matplotlib import cbook
    # Allow for easy interactive debugging with a call like:
    #
    #     $ MNE_MPL_TESTING_BACKEND=Qt5Agg pytest mne/viz/tests/test_raw.py -k annotation -x --pdb  # noqa: E501
    #
    try:
        want = os.environ['MNE_MPL_TESTING_BACKEND']
    except KeyError:
        want = 'agg'  # don't pop up windows
    with warnings.catch_warnings(record=True):  # ignore warning
        warnings.filterwarnings('ignore')
        matplotlib.use(want, force=True)
    import matplotlib.pyplot as plt
    assert plt.get_backend() == want
    # overwrite some params that can horribly slow down tests that
    # users might have changed locally (but should not otherwise affect
    # functionality)
    plt.ioff()
    plt.rcParams['figure.dpi'] = 100
    try:
        from traits.etsconfig.api import ETSConfig
    except Exception:
        pass
    else:
        ETSConfig.toolkit = 'qt4'

    # Make sure that we always reraise exceptions in handlers
    orig = cbook.CallbackRegistry

    class CallbackRegistryReraise(orig):
        def __init__(self, exception_handler=None):
            args = ()
            if LooseVersion(matplotlib.__version__) >= LooseVersion('2.1'):
                args += (exception_handler,)
            super(CallbackRegistryReraise, self).__init__(*args)

    cbook.CallbackRegistry = CallbackRegistryReraise
