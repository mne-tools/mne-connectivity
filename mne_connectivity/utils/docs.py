"""The documentation functions."""
# Authors: Eric Larson <larson.eric.d@gmail.com>
#          Thomas S. Binns <t.s.binns@outlook.com>
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
data : np.ndarray ([epochs], n_estimated_nodes, [components], [freqs], [times])
    The connectivity data that is a raveled array of ``(n_estimated_nodes, ...)`` shape.
    The ``n_estimated_nodes`` is equal to ``n_nodes_in * n_nodes_out`` if one is
    computing the full connectivity, or a subset of nodes equal to the length of
    ``indices`` passed in.
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
    Extra connectivity parameters. These may include ``freqs`` for spectral
    connectivity, ``times`` for connectivity over time, or ``components`` for
    multivariate connectivity with multiple components per connection. In addition,
    these may include extra parameters that are stored as xarray ``attrs``.
"""

docdict["mode"] = """
mode : str (default 'multitaper')
    The cross-spectral density computation method. Can be ``'multitaper'``,
    ``'fourier'``, or ``'cwt_morlet'``.
"""

docdict["mt_bandwidth"] = """
mt_bandwidth : int | float | None (default None)
    The bandwidth of the multitaper windowing function in Hz to use when computing the
    cross-spectral density. Only used if ``mode='multitaper'``.
"""

docdict["mt_adaptive"] = """
mt_adaptive : bool (default False)
    Whether to use adaptive weights when combining the tapered spectra in the
    cross-spectral density. Only used if ``mode='multitaper'``.
"""

docdict["mt_low_bias"] = """
mt_low_bias : bool (default True)
    Whether to use tapers with over 90 percent spectral concentration within the
    bandwidth when computing the cross-spectral density. Only used if
    ``mode='multitaper'``.
"""

docdict["cwt_freqs"] = """
cwt_freqs : array of int or float | None (default None)
    The frequencies of interest in Hz. Must not be `None` and only used if
    ``mode='cwt_morlet'``.
"""

docdict["cwt_n_cycles"] = """
cwt_n_cycles : int | float | array of int or float (default 7)
    The number of cycles to use when constructing the Morlet wavelets. Fixed number or
    one per frequency. Only used if ``mode='cwt_morlet'``.
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

# Decoding initialisation
docdict["info_decoding"] = """
info : mne.Info
    Information about the data which will be decomposed and transformed, such as that
    coming from an :class:`mne.Epochs` object. The number of channels must match the
    subsequent input data.
"""

docdict["method_decoding"] = """
method : str
    The multivariate method to use for the decomposition. Can be:

    * ``'cacoh'`` - Canonical Coherency (CaCoh) :footcite:`VidaurreEtAl2019`
    * ``'mic'`` - Maximised Imaginary part of Coherency (MIC) :footcite:`EwaldEtAl2012`
"""

docdict["fmin_decoding"] = """
fmin : int | float | None (default None)
    The lowest frequency of interest in Hz. Must not be `None` and only used if
    ``mode in ['multitaper', 'fourier']``.
"""

docdict["fmax_decoding"] = """
fmax : int | float | None (default None)
    The highest frequency of interest in Hz. Must not be `None` and only used if
    ``mode in ['multitaper', 'fourier']``.
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

# Decoding attrs
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

# Decoding plotting
docdict["info_decoding_plotting"] = """
info : mne.Info
    Information about the sensors of the data which has been decomposed, such as that
    coming from an :class:`mne.Epochs` object.
"""

# Topomaps
docdict["components_topomap"] = """
components : int | array of int | None (default None)
    The components to plot. If `None`, all components are shown.
"""

docdict["ch_type_topomap"] = """
ch_type : 'mag' | 'grad' | 'planar1' | 'planar2' | 'eeg' | None (default None)
    The channel type to plot. For ``'grad'``, the gradiometers are collected in pairs
    and the RMS for each pair is plotted. If `None`, the first available channel type
    from the order shown above is used.
"""

docdict["scalings_topomap"] = """
scalings : dict | float | None (default None)
    The scalings of the channel types to be applied for plotting. If `None`, uses
    ``dict(eeg=1e6, grad=1e13, mag=1e15)``.
"""

docdict["sensors_topomap"] = """
sensors : bool | str (default True)
    Whether to add markers for sensor locations. If `str`, should be a valid
    matplotlib format string (e.g., ``'r+'`` for red plusses; see the Notes section of
    :meth:`~matplotlib.axes.Axes.plot`). If `True`, black circles are used.
"""

docdict["show_names_topomap"] = """
show_names : bool | callable (default False)
    Whether to show channel names next to each sensor marker. If `callable`, channel
    names will be formatted using the callable; e.g., to delete the prefix 'MEG ' from
    all channel names, pass the function ``lambda x: x.replace('MEG ', '')``. If
    ``mask`` is not `None`, only non-masked sensor names will be shown.
"""

docdict["mask_filters_topomap"] = """
mask : array of bool, shape=(n_channels, n_filters) | None (default None)
    An array specifying channel-filter combinations to highlight with a distinct
    plotting style. Array elements set to `True` will be plotted with the parameters
    given in ``mask_params``. If `None`, no combinations will be highlighted.
"""
docdict["mask_patterns_topomap"] = """
mask : array of bool, shape=(n_channels, n_patterns) | None (default None)
    An array specifying channel-pattern combinations to highlight with a distinct
    plotting style. Array elements set to `True` will be plotted with the parameters
    given in ``mask_params``. If `None`, no combinations will be highlighted.
"""

docdict["mask_params_topomap"] = """
mask_params : dict | None (default None)
    The plotting parameters for distinct combinations given in ``mask``.
    Default `None` equals::

        dict(marker='o', markerfacecolor='w', markeredgecolor='k',
             linewidth=0, markersize=4)
"""

docdict["contours_topomap"] = """
contours : int | array (default 6)
    The number of contour lines to draw. If ``0``, no contours will be drawn. If a
    positive integer, that number of contour levels are chosen using the matplotlib tick
    locator (may sometimes be inaccurate, use array for accuracy). If an array-like, the
    values are used as the contour levels. The values should be in µV for EEG, fT for
    magnetometers and fT/m for gradiometers. If ``colorbar=True``, the colorbar will
    have ticks corresponding to the contour levels.
"""

docdict["outlines_topomap"] = """
outlines : 'head' | dict | None (default 'head')
    The outlines to be drawn. If 'head', the default head scheme will be drawn. If dict,
    each key refers to a tuple of x and y positions, the values in 'mask_pos' will serve
    as image mask. Alternatively, a matplotlib patch object can be passed for advanced
    masking options, either directly or as a function that returns patches (required for
    multi-axis plots). If `None`, nothing will be drawn.
"""

docdict["sphere_topomap"] = """
sphere : float | array | mne.bem.ConductorModel | None  | 'auto' | 'eeglab' (default None)
    The sphere parameters to use for the head outline. Can be array-like of shape (4,)
    to give the X/Y/Z origin and radius in meters, or a single float to give just the
    radius (origin assumed 0, 0, 0). Can also be an instance of a spherical
    :class:`~mne.bem.ConductorModel` to use the origin and radius from that object. If
    ``'auto'`` the sphere is fit to digitization points. If ``'eeglab'`` the head circle
    is defined by EEG electrodes ``'Fpz'``, ``'Oz'``, ``'T7'``, and ``'T8'`` (if
    ``'Fpz'`` is not present, it will be approximated from the coordinates of ``'Oz'``).
    `None` is equivalent to ``'auto'`` when enough extra digitization points are
    available, and (0, 0, 0, 0.95) otherwise.
"""  # noqa E501

docdict["image_interp_topomap"] = """
image_interp : str (default 'cubic')
    The image interpolation to be used. Options are ``'cubic'`` to use
    :class:`scipy.interpolate.CloughTocher2DInterpolator`, ``'nearest'`` to use
    :class:`scipy.spatial.Voronoi`, or ``'linear'`` to use
    :class:`scipy.interpolate.LinearNDInterpolator`.
"""

docdict["extrapolate_topomap"] = """
extrapolate : str
    The extrapolation options. Can be one of:

    - ``'box'``
        Extrapolate to four points placed to form a square encompassing all data points,
        where each side of the square is three times the range of the data in the
        respective dimension.
    - ``'local'`` (default for MEG sensors)
        Extrapolate only to nearby points (approximately to points closer than median
        inter-electrode distance). This will also set the mask to be polygonal based on
        the convex hull of the sensors.
    - ``'head'`` (default for non-MEG sensors)
        Extrapolate out to the edges of the clipping circle. This will be on the head
        circle when the sensors are contained within the head circle, but it can extend
        beyond the head when sensors are plotted outside the head circle.
"""

docdict["border_topomap"] = """
border : float | 'mean' (default 'mean')
    The value to extrapolate to on the topomap borders. If ``'mean'``, each extrapolated
    point has the average value of its neighbours.
"""

docdict["res_topomap"] = """
res : int (default 64)
    The resolution of the topomap image (number of pixels along each side).
"""

docdict["size_topomap"] = """
size : int | float (default 1)
    The side length of each subplot in inches.
"""

docdict["cmap_topomap"] = """
cmap : str | matplotlib.colors.Colormap | (matplotlib.colors.Colormap, bool) | 'interactive' | None (default 'RdBu_r')
    The colormap to use. If a `str`, should be a valid matplotlib colormap. If a
    `tuple`, the first value is `matplotlib.colors.Colormap` object to use and the
    second value is a boolean defining interactivity. In interactive mode the colors are
    adjustable by clicking and dragging the colorbar with left and right mouse button.
    Left mouse button moves the scale up and down and right mouse button adjusts the
    range. Hitting space bar resets the range. Up and down arrows can be used to change
    the colormap. If `None`, ``'Reds'`` is used for data that is either all positive or
    all negative, and ``'RdBu_r'`` is used otherwise. ``'interactive'`` is equivalent to
    ``(None, True)``.

    .. warning::  Interactive mode works smoothly only for a small amount
        of topomaps. Interactive mode is disabled by default for more than
        2 topomaps.
"""  # noqa E501

docdict["vlim_topomap"] = """
vlim : tuple of length 2 (default (None, None))
    The lower and upper colormap bounds, respectively. If both entries are `None`, sets
    bounds to ``(min(data), max(data))``. If one entry is `None`, the corresponding
    boundary is set at the min/max of the data.
"""

docdict["cnorm_topomap"] = """
cnorm : matplotlib.colors.Normalize | None (default None)
    How to normalize the colormap. If `None`, standard linear normalization is used. If
    not `None`, ``vlim`` is ignored. See the :ref:`Matplotlib docs
    <matplotlib:colormapnorms>` for more details on colormap normalization.
"""

docdict["colorbar_topomap"] = """
colorbar : bool (default True)
    Whether to plot a colorbar in the rightmost column of the figure.
"""

docdict["colorbar_format_topomap"] = r"""
cbar_fmt : str (default '%.1E')
    The formatting string for colorbar tick labels. See :ref:`formatspec` for details.
"""

docdict["units_topomap"] = """
units : str (default 'AU')
    The units for the colorbar label. Ignored if ``colorbar=False``.
"""

docdict["axes_topomap"] = """
axes : length-2 tuple of list of matplotlib.axes.Axes | None (default None)
    The axes to plot to. If `None`, a new figure will be created with the correct number
    of axes. If not `None`, there must be two lists containing the axes for the seeds
    and targets, respectively. In each of these two lists, the number of axes must match
    ``components`` if ``colorbar=False``, or ``components * 2`` if ``colorbar=True``.
"""

docdict["name_format_topomap"] = r"""
name_format : str | None (default None)
    The string format for axes titles. If `None`, uses ``f"{method}%01d_{group}"``,
    i.e., the method name followed by the component number and the group being plotted
    (seeds or targets). If not `None`, it must contain a formatting specifier for the
    component number, and the group will be appended to the end.
"""

docdict["nrows_topomap"] = """
nrows : int | 'auto' (default 'auto')
    The number of rows of components to plot. If ``'auto'``, the necessary number will
    be inferred.
"""

docdict["ncols_topomap"] = """
ncols : int | 'auto' (default 'auto')
    The number of columns of components to plot. If ``'auto'``, the necessary number
    will be inferred. If ``nrows='auto'`` and ``ncols='auto'``, becomes ``nrows=1,
    ncols='auto'``.
"""

docdict["figs_topomap"] = """
figs : list of matplotlib.figure.Figure
    The seed and target figures, respectively.
"""

docdict["show"] = """
show : bool (default True)
    Whether to show the figure.
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
        raise RuntimeError(f"Error documenting {funcname}:\n{str(exp)}")
    return f
