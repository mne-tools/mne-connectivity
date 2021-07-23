# -*- coding: utf-8 -*-
"""The documentation functions."""
# Authors: Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD (3-clause)

from mne.externals.doccer import indentcount_lines

##############################################################################
# Define our standard documentation entries

docdict = dict()

# Connectivity
docdict['data'] = """
data : np.ndarray ([epochs], n_estimated_nodes, [freqs], [times])
    The connectivity data that is a raveled array of
    ``(n_estimated_nodes, ...)`` shape. The
    ``n_estimated_nodes`` is equal to
    ``n_nodes_in * n_nodes_out`` if one is computing
    the full connectivity, or a subset of nodes
    equal to the length of ``indices`` passed in.
"""

docdict['names'] = """
names : list | np.ndarray | None
    The names of the nodes of the dataset used to compute
    connectivity. If 'None' (default), then names will be
    a list of integers from 0 to ``n_nodes``. If a list
    of names, then it must be equal in length to ``n_nodes``.
"""

docdict['indices'] = """
indices : tuple of arrays | str | None
    The indices of relevant connectivity data. If ``'all'`` (default),
    then data is connectivity between all nodes. If ``'symmetric'``,
    then data is symmetric connectivity between all nodes. If a tuple,
    then the first list represents the "in nodes", and the second list
    represents the "out nodes". See "Notes" for more information.
"""

docdict['n_nodes'] = """
n_nodes : int
    The number of nodes in the dataset used to compute connectivity.
    This should be equal to the number of signals in the original
    dataset.
"""

# Downstream container variables
docdict['freqs'] = """
freqs : list | np.ndarray
    The frequencies at which the connectivity data is computed over.
    If the frequencies are "frequency bands" (i.e. gamma band), then
    these are the median of those bands.
"""

docdict['times'] = """
times : list | np.ndarray
    The times at which the connectivity data is computed over.
"""

docdict['method'] = """
method : str, optional
    The method name used to compute connectivity.
"""

docdict['spec_method'] = """
spec_method : str, optional
    The type of method used to compute spectral analysis,
    by default None.
"""

docdict['n_epochs_used'] = """
n_epochs_used : int, optional
    The number of epochs used in the computation of connectivity,
    by default None.
"""

# Verbose
docdict['verbose'] = """
verbose : bool, str, int, or None
    If not None, override default verbose level (see :func:`mne.verbose`
    for more info). If used, it should be passed as a
    keyword-argument only."""

# Parallelization
docdict['n_jobs'] = """
n_jobs : int
    The number of jobs to run in parallel (default 1).
    Requires the joblib package.
"""

docdict_indented = {}


def fill_doc(f):
    """Fill a docstring with docdict entries.

    Parameters
    ----------
    f : callable
        The function to fill the docstring of. Will be modified in place.

    Returns
    -------
    f : callable
        The function, potentially with an updated ``__doc__``.
    """
    docstring = f.__doc__
    if not docstring:
        return f
    lines = docstring.splitlines()
    # Find the minimum indent of the main docstring, after first line
    if len(lines) < 2:
        icount = 0
    else:
        icount = indentcount_lines(lines[1:])
    # Insert this indent to dictionary docstrings
    try:
        indented = docdict_indented[icount]
    except KeyError:
        indent = ' ' * icount
        docdict_indented[icount] = indented = {}
        for name, dstr in docdict.items():
            lines = dstr.splitlines()
            try:
                newlines = [lines[0]]
                for line in lines[1:]:
                    newlines.append(indent + line)
                indented[name] = '\n'.join(newlines)
            except IndexError:
                indented[name] = dstr
    try:
        f.__doc__ = docstring % indented
    except (TypeError, ValueError, KeyError) as exp:
        funcname = f.__name__
        funcname = docstring.split('\n')[0] if funcname is None else funcname
        raise RuntimeError('Error documenting %s:\n%s'
                           % (funcname, str(exp)))
    return f
