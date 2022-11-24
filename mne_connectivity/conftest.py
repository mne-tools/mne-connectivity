# -*- coding: utf-8 -*-
# Author: Adam Li <adam2392@gmail.com>
#
# License: BSD-3-Clause

from contextlib import contextmanager
import pytest
import os
import gc
import warnings

from mne.utils import _check_qt_version


def pytest_configure(config):
    """Configure pytest options."""
    # Fixtures
    for fixture in ('matplotlib_config',):
        config.addinivalue_line('usefixtures', fixture)

    warning_lines = r"""
    error::
    ignore:.*`np.bool` is a deprecated alias.*:DeprecationWarning
    ignore:.*String decoding changed with h5py.*:FutureWarning
    ignore:.*SelectableGroups dict interface is deprecated.*:DeprecationWarning
    ignore:.*Converting `np.character` to a dtype is deprecated.*:DeprecationWarning
    ignore:.*distutils Version classes are deprecated.*:DeprecationWarning
    ignore:.*`np.MachAr` is deprecated.*:DeprecationWarning
    ignore:.*You are writing invalid netcdf features to file.*:UserWarning
    # for the persistence of metadata and Raw Annotations within mne-python
    # Epochs class
    ignore:.*There were no Annotations stored in.*:RuntimeWarning
    always::ResourceWarning
    # pydarkstyle
    ignore:.*Setting theme='dark' is not yet supported.*:RuntimeWarning
    """  # noqa: E501
    for warning_line in warning_lines.split('\n'):
        warning_line = warning_line.strip()
        if warning_line and not warning_line.startswith('#'):
            config.addinivalue_line('filterwarnings', warning_line)


@pytest.fixture(autouse=True)
def close_all():
    """Close all matplotlib plots, regardless of test status."""
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

    # Make sure that we always reraise exceptions in handlers
    orig = cbook.CallbackRegistry

    class CallbackRegistryReraise(orig):
        def __init__(self, exception_handler=None, signals=None):
            super(CallbackRegistryReraise, self).__init__(exception_handler)

    cbook.CallbackRegistry = CallbackRegistryReraise


@pytest.fixture
def garbage_collect():
    """Garbage collect on exit."""
    yield
    gc.collect()


@pytest.fixture(params=["pyvistaqt"])
def renderer(request, garbage_collect):
    """Yield the 3D backends."""
    with _use_backend(request.param, interactive=False) as renderer:
        yield renderer


@pytest.fixture(params=["pyvistaqt"])
def renderer_pyvistaqt(request, garbage_collect):
    """Yield the PyVista backend."""
    with _use_backend(request.param, interactive=False) as renderer:
        yield renderer


@pytest.fixture(params=["notebook"])
def renderer_notebook(request):
    """Yield the 3D notebook renderer."""
    with _use_backend(request.param, interactive=False) as renderer:
        yield renderer


@pytest.fixture(scope="module", params=["pyvistaqt"])
def renderer_interactive_pyvistaqt(request):
    """Yield the interactive PyVista backend."""
    with _use_backend(request.param, interactive=True) as renderer:
        yield renderer


@pytest.fixture(scope="module", params=["pyvistaqt"])
def renderer_interactive(request):
    """Yield the interactive 3D backends."""
    with _use_backend(request.param, interactive=True) as renderer:
        if renderer._get_3d_backend() == 'mayavi':
            with warnings.catch_warnings(record=True):
                try:
                    from surfer import Brain  # noqa: 401 analysis:ignore
                except Exception:
                    pytest.skip('Requires PySurfer')
        yield renderer


@contextmanager
def _use_backend(backend_name, interactive):
    from mne.viz.backends.renderer import _use_test_3d_backend
    _check_skip_backend(backend_name)
    with _use_test_3d_backend(backend_name, interactive=interactive):
        from mne.viz.backends import renderer
        try:
            yield renderer
        finally:
            renderer.backend._close_all()


def _check_skip_backend(name):
    from mne.viz.backends.tests._utils import (has_pyvista, has_imageio_ffmpeg,
                                               has_pyvistaqt)
    if name in ('pyvistaqt', 'notebook'):
        if not has_pyvista():
            pytest.skip("Test skipped, requires pyvista.")
        if not has_imageio_ffmpeg():
            pytest.skip("Test skipped, requires imageio-ffmpeg")
    if name == 'pyvistaqt' and not _check_qt_version():
        pytest.skip("Test skipped, requires Python Qt bindings.")
    if name == 'pyvistaqt' and not has_pyvistaqt():
        pytest.skip("Test skipped, requires pyvistaqt")
