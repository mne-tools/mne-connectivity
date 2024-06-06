"""The documentation functions."""
# Authors: Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD (3-clause)

try:  # 1.0+
    from mne.utils.docs import _indentcount_lines
except ImportError:
    from mne.externals.doccer import indentcount_lines as _indentcount_lines  # noqa

##############################################################################
# Define our standard documentation entries

docdict = dict()

# Connectivity
docdict["data"] = """
data : np.ndarray ([epochs], n_estimated_nodes, [freqs], [times])
    The connectivity data that is a raveled array of
    ``(n_estimated_nodes, ...)`` shape. The
    ``n_estimated_nodes`` is equal to
    ``n_nodes_in * n_nodes_out`` if one is computing
    the full connectivity, or a subset of nodes
    equal to the length of ``indices`` passed in.
"""

docdict["names"] = """
names : list | np.ndarray | None
    The names of the nodes of the dataset used to compute
    connectivity. If 'None' (default), then names will be
    a list of integers from 0 to ``n_nodes``. If a list
    of names, then it must be equal in length to ``n_nodes``.
"""

docdict["indices"] = """
indices : tuple of arrays | str | None
    The indices of relevant connectivity data. If ``'all'`` (default),
    then data is connectivity between all nodes. If ``'symmetric'``,
    then data is symmetric connectivity between all nodes. If a tuple,
    then the first list represents the "in nodes", and the second list
    represents the "out nodes". See "Notes" for more information.
"""

docdict["n_nodes"] = """
n_nodes : int
    The number of nodes in the dataset used to compute connectivity.
    This should be equal to the number of signals in the original
    dataset.
"""

docdict["connectivity_kwargs"] = """
**kwargs : dict
        Extra connectivity parameters. These may include
        ``freqs`` for spectral connectivity, and/or
        ``times`` for connectivity over time. In addition,
        these may include extra parameters that are stored
        as xarray ``attrs``.
"""

docdict["mode"] = """
mode : str (default "multitaper")
    The cross-spectral density computation method. Can be ``"multitaper"``,
    ``"fourier"``, or ``"cwt_morlet"``.
"""

docdict["mt_bandwidth"] = """
mt_bandwidth : int | float | None (default None)
    The bandwidth of the multitaper windowing function in Hz to use when computing the
    cross-spectral density. Only used if ``mode="multitaper"``.
"""

docdict["mt_adaptive"] = """
mt_adaptive : bool (default False)
    Whether to use adaptive weights when combining the tapered spectra in the
    cross-spectral density. Only used if ``mode="multitaper"``.
"""

docdict["mt_low_bias"] = """
mt_low_bias : bool (default True)
    Whether to use tapers with over 90 percent spectral concentration within the
    bandwidth when computing the cross-spectral density. Only used if
    ``mode="multitaper"``.
"""

docdict["cwt_freqs"] = """
cwt_freqs : array of int or float | None (default None)
    The frequencies of interest in Hz. Must not be ``None`` and only used if
    ``mode="cwt_morlet"``.
"""

docdict["cwt_n_cycles"] = """
cwt_n_cycles : int | float | array of int or float (default 7)
    The number of cycles to use when constructing the Morlet wavelets. Fixed number or
    one per frequency. Only used if ``mode=cwt_morlet``.
"""

docdict["coh"] = "'coh' : Coherence"
docdict["cohy"] = "'cohy' : Coherency"
docdict["imcoh"] = "'imcoh' : Imaginary part of Coherency"
docdict["cacoh"] = "'cacoh' : Canonical Coherency (CaCoh)"
docdict["mic"] = "'mic' : Maximised Imaginary part of Coherency (MIC)"
docdict["mim"] = "'mim' : Multivariate Interaction Measure (MIM)"
docdict["plv"] = "'plv' : Phase-Locking Value (PLV)"
docdict["ciplv"] = "'ciplv' : Corrected Imaginary PLV (ciPLV)"
docdict["ppc"] = "'ppc' : Pairwise Phase Consistency (PPC)"
docdict["pli"] = "'pli' : Phase Lag Index (PLI)"
docdict["pli2_unbiased"] = "'pli2_unbiased' : Unbiased estimator of squared PLI"
docdict["dpli"] = "'dpli' : Directed PLI (DPLI)"
docdict["wpli"] = "'wpli' : Weighted PLI (WPLI)"
docdict["wpli2_debiased"] = "'wpli2_debiased' : Debiased estimator of squared WPLI"
docdict["gc"] = "'gc' : State-space Granger Causality (GC)"
docdict["gc_tr"] = "'gc_tr' : State-space GC on time-reversed signals"

# Downstream container variables
docdict["freqs"] = """
freqs : list | np.ndarray
    The frequencies at which the connectivity data is computed over.
    If the frequencies are "frequency bands" (i.e. gamma band), then
    these are the median of those bands.
"""

docdict["times"] = """
times : list | np.ndarray
    The times at which the connectivity data is computed over.
"""

docdict["method"] = """
method : str, optional
    The method name used to compute connectivity.
"""

docdict["spec_method"] = """
spec_method : str, optional
    The type of method used to compute spectral analysis,
    by default None.
"""

docdict["n_epochs_used"] = """
n_epochs_used : int, optional
    The number of epochs used in the computation of connectivity,
    by default None.
"""

docdict["events"] = """
events : array of int, shape (n_events, 3)
    The events typically returned by the read_events function.
    If some events don't match the events of interest as specified
    by event_id, they will be marked as 'IGNORED' in the drop log.
"""

docdict["event_id"] = """
event_id : int | list of int | dict | None
    The id of the event to consider. If dict,
    the keys can later be used to access associated events. Example:
    dict(auditory=1, visual=3). If int, a dict will be created with
    the id as string. If a list, all events with the IDs specified
    in the list are used. If None, all events will be used with
    and a dict is created with string integer names corresponding
    to the event id integers.
"""

# Verbose
docdict["verbose"] = """
verbose : bool, str, int, or None
    If not None, override default verbose level (see :func:`mne.verbose`
    for more info). If used, it should be passed as a
    keyword-argument only."""

# Parallelization
docdict["n_jobs"] = """
n_jobs : int
    The number of jobs to run in parallel (default 1).
    Requires the joblib package.
"""

# Random state
docdict["random_state"] = """
random_state : None | int | instance of ~numpy.random.RandomState
    If ``random_state`` is an :class:`int`, it will be used as a seed for
    :class:`~numpy.random.RandomState`. If ``None``, the seed will be
    obtained from the operating system (see
    :class:`~numpy.random.RandomState` for details). Default is
    ``None``.
"""

# Decoding
docdict["info_decoding"] = """
info : mne.Info
    Information about the data which will be decomposed and transformed, such as that
    coming from an :class:`mne.Epochs` object. The number of channels must match the
    subsequent input data.
"""

docdict["method_decoding"] = """
method : str
    The multivariate method to use for the decomposition. Can be:

    * ``"cacoh"`` - Canonical Coherency (CaCoh) :footcite:`VidaurreEtAl2019`
    * ``"mic"`` - Maximised Imaginary part of Coherency (MIC) :footcite:`EwaldEtAl2012`
"""

docdict["fmin_decoding"] = """
fmin : int | float | None (default None)
    The lowest frequency of interest in Hz. Must not be ``None`` and only used if
    ``mode in ["multitaper", "fourier"]``.
"""

docdict["fmax_decoding"] = """
fmax : int | float | None (default None)
    The highest frequency of interest in Hz. Must not be ``None`` and only used if
    ``mode in ["multitaper", "fourier"]``.
"""

docdict["indices_decoding"] = """
indices : tuple of array
    A tuple of two arrays, containing the indices of the seed and target channels in the
    input data, respectively. The indices of only a single connection (i.e. between one
    group of seeds and one group of targets) is supported.
"""

docdict["n_components"] = """
n_components : int | None (default None)
    The number of connectivity components (sources) to extract from the data. If `None`,
    the number of components equal to the minimum rank of the seeds and targets is
    extracted (see the ``rank`` parameter). If an `int`, the number of components must
    be <= the minimum rank of the seeds and targets. E.g. if the seed channels had a
    rank of 5 and the target channels had a rank of 3, ``n_components`` must be <= 3.
"""

docdict["rank"] = """
rank : tuple of int | None (default None)
    A tuple of two ints, containing the degree of rank subspace projection to apply to
    the seed and target data, respectively, before filters are fit. If `None`, the rank
    of the seed and target data is used. If a tuple of ints, the entries must be <= the
    rank of the seed and target data. The minimum rank of the seeds and targets
    determines the maximum number of connectivity components (sources) which can be
    extracted from the data (see the ``n_components`` parameter). Specifying ranks below
    that of the data may reduce the degree of overfitting when computing the filters.
"""

docdict["filters_"] = """
filters_ : tuple of array, shape=(n_signals, n_components)
    A tuple of two arrays containing the spatial filters for transforming the seed and
    target data, respectively.
"""

docdict["patterns_"] = """
patterns_ : tuple of array, shape=(n_components, n_signals)
    A tuple of two arrays containing the spatial patterns corresponding to the spatial
    filters for the seed and target data, respectively.
"""


docdict_indented = dict()  # type: ignore


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
        icount = _indentcount_lines(lines[1:])
    # Insert this indent to dictionary docstrings
    try:
        indented = docdict_indented[icount]
    except KeyError:
        indent = " " * icount
        docdict_indented[icount] = indented = {}
        for name, dstr in docdict.items():
            lines = dstr.splitlines()
            try:
                newlines = [lines[0]]
                for line in lines[1:]:
                    newlines.append(indent + line)
                indented[name] = "\n".join(newlines)
            except IndexError:
                indented[name] = dstr
    try:
        f.__doc__ = docstring % indented
    except (TypeError, ValueError, KeyError) as exp:
        funcname = f.__name__
        funcname = docstring.split("\n")[0] if funcname is None else funcname
        raise RuntimeError("Error documenting %s:\n%s" % (funcname, str(exp)))
    return f
