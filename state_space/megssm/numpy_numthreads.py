import contextlib
import ctypes
from ctypes.util import find_library

# heavily based on:
# https://stackoverflow.com/questions/29559338/set-max-number-of-threads-at-runtime-on-numpy-openblas

# Prioritize hand-compiled OpenBLAS library over version in /usr/lib/
# from Ubuntu repos
try_paths = [find_library('openblas')]
openblas_lib = None
mkl_rt = None
#if openblas_lib is None:
    #raise EnvironmentError('Could not locate an OpenBLAS shared library', 2)


def set_num_threads(n):
    """Set the current number of threads used by the OpenBLAS server."""
    if mkl_rt:
        pass
        #mkl_rt.mkl_set_num_threads(ctypes.byref(ctypes.c_int(n)))
    elif openblas_lib:
        openblas_lib.openblas_set_num_threads(int(n))


# At the time of writing these symbols were very new:
# https://github.com/xianyi/OpenBLAS/commit/65a847c
try:
    if mkl_rt: #False: #mkl_rt:
        def get_num_threads():
            return mkl_rt.mkl_get_max_threads()
    elif openblas_lib:
        # do this to throw exception if it doesn't exist
        openblas_lib.openblas_get_num_threads()
        def get_num_threads():
            """Get the current number of threads used by the OpenBLAS server."""
            return openblas_lib.openblas_get_num_threads()
    else:
        def get_num_threads():
            return -1
except AttributeError:
    def get_num_threads():
        """Dummy function (symbol not present in %s), returns -1."""
        return -1
    pass

try:
    if False: #mkl_rt:
        def get_num_procs():
            # this returns number of procs
            return mkl_rt.mkl_get_max_threads()
    elif openblas_lib:
        # do this to throw exception if it doesn't exist
        openblas_lib.openblas_get_num_procs()
        def get_num_procs():
            """Get the total number of physical processors"""
            return openblas_lib.openblas_get_num_procs()
except AttributeError:
    def get_num_procs():
        """Dummy function (symbol not present), returns -1."""
        return -1
    pass


@contextlib.contextmanager
def numpy_num_threads(n):
    """Temporarily changes the number of OpenBLAS threads.

    Example usage:

        print("Before: {}".format(get_num_threads()))
        with num_threads(n):
            print("In thread context: {}".format(get_num_threads()))
        print("After: {}".format(get_num_threads()))
    """
    old_n = get_num_threads()
    set_num_threads(n)
    try:
        yield
    finally:
        set_num_threads(old_n)
