"""The documentation functions."""
# Authors: Eric Larson <larson.eric.d@gmail.com>
#          Thomas S. Binns <t.s.binns@outlook.com>
#
# License: BSD (3-clause)

from functools import partial

from mne.utils.docs import _indentcount_lines

##############################################################################
# Define our standard documentation entries

docdict = dict()

# Connectivity
docdict["data"] = """
data : array, shape ([epochs,] n_estimated_nodes[, components, freqs, times])
    The connectivity data that is a raveled array of ``(..., n_estimated_nodes, ...)``
    shape. The ``n_estimated_nodes`` is equal to ``n_nodes_in * n_nodes_out`` if one has
    computed the full connectivity, or a subset of nodes equal to the length of the
    arrays in ``indices`` passed in.
"""

docdict["names"] = """
names : array_like | None
    The names of the nodes of the dataset used to compute connectivity. If ``None``
    (default), then names will be a list of integers from 0 to ``n_nodes``. If a list
    of names, then it must be equal in length to ``n_nodes``.
"""

indices_base = """
indices : tuple of array_like{other_types}
    The indices of channels to compute connectivity between. If a tuple, then must
    contain two array-likes, where the first array represents the channel indices of the
    seeds, and the second array represents the channel indices of the targets.
    {multivar_indices}
    See the notes section for more information.
    {other_types_description}
    {default}
"""
multivar_indices_extra = (
    "For multivariate methods, the seed and target indices should consist of nested "
    "arrays containing the channel indices for each multivariate connection."
)
indices_with_str = partial(
    indices_base.format,
    other_types=" | ``'lower'`` | ``'upper'`` | ``'all'``",
    other_types_description=(
        "If ``'lower'``, compute the lower-triangular part of the connectivity matrix. "
        "If ``'upper'``, compute the upper-triangular part of the connectivity matrix. "
        "If ``'all'``, compute all connections."
    ),
    default="Default is ``'lower'``.",
)
docdict["indices_with_str_only_bivar"] = indices_with_str(multivar_indices="")
docdict["indices_with_str_with_multivar"] = indices_with_str(
    multivar_indices=multivar_indices_extra
)

docdict["n_nodes"] = """
n_nodes : int
    The number of nodes in the dataset used to compute connectivity. This should be
    equal to the number of signals in the original dataset.
"""

docdict["connectivity_kwargs"] = """
**kwargs : dict
    Extra connectivity parameters. These may include ``freqs`` for spectral
    connectivity, ``times`` for connectivity over time, or ``components`` for
    multivariate connectivity with multiple components per connection. In addition,
    these may include extra parameters that are stored as xarray ``attrs``.
"""

docdict["mode"] = """
mode : ``'multitaper'`` | ``'fourier'`` | ``'cwt_morlet'``
    The cross-spectral density computation method.
"""

docdict["mt_bandwidth"] = """
mt_bandwidth : float | None (default None)
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
cwt_freqs : array_like | None (default None)
    The frequencies of interest in Hz. Must not be `None` and only used if
    ``mode='cwt_morlet'``.
"""

docdict["cwt_n_cycles"] = """
cwt_n_cycles : float | array_like (default 7.0)
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

docdict["tri_indices_efficiency_note"] = """
If connectivity for all possible connections is desired, we can save time and memory by
only computing and storing the lower- (or upper-) triangular part of the connectivity
matrix. When the full set of connectivity values is needed, the missing values can be
inferred based on what has been computed. See the notes section of the
:meth:`~Connectivity.get_data` method in the connectivity containers for more
information.
"""

docdict["tuple_bivar_indices_note"] = """
If one is only interested in the connectivity between some signals, the ``indices``
parameter can be specified as a tuple of array-likes. For example, to compute the
connectivity between the signal with index 0 and the signals with indices 2, 3, and 4 (a
total of 3 connections) one can use the following::

    indices = (np.array([0, 0, 0]),  # seed indices
               np.array([2, 3, 4]))  # target indices
"""

docdict["tuple_multivar_indices_note"] = """
For multivariate methods, ``indices`` must be specified as a tuple, where the seed and
target indices for each connection are nested array-likes. For example, to compute the
connectivity between signals (0, 1) → (2, 3) and (0, 1) → (4, 5), ``indices`` should be
specified as::

    indices = (np.array([[0, 1], [0, 1]]),  # seeds
               np.array([[2, 3], [4, 5]]))  # targets

More information on working with multivariate indices and handling connections where the
number of seeds and targets are not equal can be found in the
:doc:`../auto_examples/handling_ragged_arrays` example.
"""

# Downstream container variables
docdict["indices"] = """
indices : tuple of array_like | ``'lower'`` | ``'upper'`` | ``'all'``
    The indices of channels connectivity was computed between.
    If a tuple, then contains two array-likes, where the first array represents the
    channel indices of the seeds, and the second array represents the channel indices of
    the targets.
    For multivariate methods, the seed and target indices should consist of nested
    arrays containing the channel indices for each multivariate connection.
    If ``'lower'``, connectivity represents the lower-triangular part of the
    connectivity matrix.
    If ``'upper'``, connectivity represents the upper-triangular part of the
    connectivity matrix.
    If ``'all'``, connectivity represents all connections.
    Indices for multivarate methods should only be specified as a tuple of arrays.
"""

docdict["freqs"] = """
freqs : list | array
    The frequencies at which the connectivity data is computed over. If the frequencies
    are "frequency bands" (i.e. gamma band), then these are the median of those bands.
"""

docdict["times"] = """
times : list | array
    The times at which the connectivity data is computed over.
"""

docdict["method"] = """
method : str | None
    The method name used to compute connectivity (default ``None``).
"""

docdict["spec_method"] = """
spec_method : str | None
    The type of method used to compute spectral analysis (default ``None``).
"""

docdict["n_epochs_used"] = """
n_epochs_used : int | None
    The number of epochs used in the computation of connectivity (default ``None``).
"""

docdict["events"] = """
events : array of int, shape (n_events, 3)
    The events typically returned by the read_events function. If some events don't
    match the events of interest as specified by ``event_id``, they will be marked as
    'IGNORED' in the drop log.
"""

docdict["event_id"] = """
event_id : int | list of int | dict | None
    The ID of the event to consider. If a dict, the keys can later be used to access
    associated events. Example: ``dict(auditory=1, visual=3)``. If an int, a dict will
    be created with the ID as string. If a list, all events with the IDs specified in
    the list are used. If ``None``, all events will be used and a dict is created with
    string integer names corresponding to the event ID integers.
"""

# Verbose
docdict["verbose"] = """
verbose : bool | str | int | None
    If not ``None``, override default verbose level (see :func:`mne.verbose` for more
    info). If used, it should be passed as a keyword-argument only."""

# Parallelization
docdict["n_jobs"] = """
n_jobs : int
    The number of jobs to run in parallel (default 1). Requires the joblib package.
"""

# Random state
docdict["random_state"] = """
random_state : None | int | instance of ~numpy.random.RandomState
    If ``random_state`` is an int, it will be used as a seed for
    :class:`numpy.random.RandomState`. If ``None``, the seed will be obtained from the
    operating system (see :class:`numpy.random.RandomState` for details). Default is
    ``None``.
"""

# Visualisation
docdict["viz_info"] = """
info : mne.Info | None
    The :class:`mne.Info` object with information about the sensors and methods of
    measurement. Used to split the figures by channel types and identify bad channels.
    If ``None`` (default), all channels are assumed to be good ``'misc'`` channels.
"""

docdict["viz_picks"] = """
picks : str | array_like | slice | None
    Channels to include in the plot. Connections involving these channels will be
    included, based on ``selection``. Slices and lists of integers will be interpreted
    as channel indices. In lists, channel type strings (e.g., ``['meg', 'eeg']``) will
    pick channels of those types, channel name strings (e.g.,
    ``['MEG0111', 'MEG2623']``) will pick the given channels. Can also be the string
    values ``'all'`` to pick all channels, or ``'data'`` to pick data channels. None
    (default) will pick any good channels. Note that channels in ``info['bads']`` will
    be included if their names or indices are explicitly provided.
"""

viz_selection_template = """
selection : ``'seeds'`` | ``'targets'`` | ``'both'``
    What the ``picks`` parameter will be applied to. If ``'seeds'``, only connections
    within the seed channels matchinng ``picks`` will be included. If ``'targets'``,
    only connections within the target channels matching ``picks`` will be included. If
    ``'both'``, connections will be included if either the seed or target channels match
    ``picks``. Ignored if ``picks`` is ``None``.{}
"""
docdict["viz_selection"] = viz_selection_template.format("")
docdict["viz_selection_line"] = docdict["viz_selection"].format(
    " This also controls the channels which can be selected in the interactive circle "
    "plot."
)

docdict["viz_exclude"] = """
exclude : list of str | ``'bads'``
    Channel names to exclude from plotting. All connections involving these channels
    will be excluded. If ``'bads'`` (default), channels in ``info['bads']`` are
    excluded.
"""

viz_combine_template = """
combine : ``'mean'`` | callable | None
    How to aggregate across connections. ``'mean'`` uses :func:`numpy.mean`. If
    :func:`callable`, it must operate on an array of shape {} and return an array of
    shape {}. If ``None``, plot {}. Defaults to {}.
"""
docdict["viz_combine_line_spectral"] = viz_combine_template.format(
    "``(n_connections, n_freqs)``",
    "``(n_freqs,)``",
    "each connection individually",
    "``None``",
)
docdict["viz_combine_line_temporal"] = viz_combine_template.format(
    "``(n_connections, n_times)``",
    "``(n_times,)``",
    "each connection individually",
    "``None``",
)
docdict["viz_combine_image_spectrotemporal"] = viz_combine_template.format(
    "``(n_connections, n_freqs, n_times)``",
    "``(n_freqs, n_times)``",
    "one figure per connection",
    "``'mean'``",
)

docdict["viz_ci"] = """
ci : float | ``'sd'`` | ``'range'`` | None
    Type of confidence band drawn around the aggregated data when ``combine`` is not
    ``None``. If ``'sd'`` (default) the band spans ±1 the standard deviation across
    connections. If ``'range'`` the band spans the range across connections at each bin.
    If a float, it indicates the (bootstrapped) confidence interval to display, and must
    satisfy ``0 < ci <= 100``. If ``None``, no band is drawn.
"""

docdict["viz_tmin_tmax"] = """
tmin, tmax : float | None
    First and last times to plot, in seconds. If ``None`` take the first/last time
    in the data, respectively. Default is ``None``.
"""

docdict["viz_fmin_fmax"] = """
fmin, fmax : float | None
    First and last frequencies to plot, in Hz. If ``None`` take the first/last frequency
    in the data, respectively. Default is ``None``.
"""

docdict["viz_colors_line"] = """
colors : ``'auto'`` | ``'global'`` | ``'relative'``
    How to color the connections. If ``'global'``, the same colormap is used across all
    connections. This is recommended if the connectivity indices do not correspond to
    a full or symmetric matrix. If ``'relative'``, the connections for each channel
    span the full colormap. This is recommended if the connectivity indices correspond
    to a full or symmetric matrix. If ``'auto'`` (default), the coloring is set to
    ``'relative'`` if the connectivity indices correspond to a lower-triangular matrix
    and ``interactive`` is ``True``, or ``'global'`` otherwise.
"""

docdict["viz_cmap_line"] = """
cmap : str | matplotlib.colors.Colormap
    Colormap to use for coloring the connections. If a str, must be a recognised
    Matplotlib colormap name. Default is ``'turbo'``.
"""

docdict["viz_yscale_image"] = """
yscale : ``'linear'`` | ``'log'`` | ``'auto'``
    The scale of the y-axis (frequencies). ``'linear'`` gives a linear y-axis. ``'log'``
    gives a log-spaced y-axis. ``'auto'`` (default) detects if frequencies are
    log-spaced, and if so, sets the y-axis to ``'log'``, otherwise is ``'linear'``.
    Default is ``'auto'``.
"""

docdict["viz_node_aliases"] = """
node_aliases : dict | None
    Mapping of node indices to node names. Keys should be seed or target indices
    found in ``con.indices``, that is, integers for bivariate connectivity, and arrays
    of integers for multivariate connectivity. If ``None`` and plotting results for
    bivariate connectivity, node names will be taken from ``con.names``. If ``None``
    and plotting results for multivariate connectivity, node names will be generated
    as ``'node {idx}'``, where ``idx`` is the order of the node in the unique set of
    indices, as determined by ``np.unique([*con.indices[0], *con.indices[1]])``.
"""

docdict["viz_vmin_vmax"] = """
vmin, vmax : float | None
    Lower and upper bounds of the colormap, respectively. If both entries are ``None``
    and there are both positive and negative values in the data, the bounds are set at ±
    the maximum absolute value of the data (yielding a colormap with midpoint at 0). If
    both entries are ``None`` and the data is all positive or all negative, the bounds
    are set at the min/max of the data, respectively. Providing ``None`` for just one
    entry will set the corresponding boundary at the min/max of the data. Defaults to
    ``None``.
"""

docdict["viz_cnorm"] = """
cnorm : matplotlib.colors.Normalize | None
    How to normalize the colormap. If ``None`` (default), standard linear normalization
    is performed. If not ``None``, ``vmin`` and ``vmax`` will be ignored. See
    :ref:`Matplotlib docs <matplotlib:colormapnorms>` for more details on colormap
    normalization.
"""

docdict["viz_cbar"] = """
colorbar : bool
    Whether to display a colorbar for each figure. Defaults to ``True``.
"""

docdict["viz_cmap"] = """
cmap : str | matplotlib.colors.Colormap | None
    The colormap to use for coloring the connectivity values. If a str, must be a valid
    Matplotlib colormap name.If ``None`` (default), ``'RdBu_r'`` is used for data that
    has positive and negative values, ``Reds`` is used for data that is all positive,
    and ``Blues_r`` is used for data that is all negative.
"""

docdict["viz_node_labels_matrix"] = """
node_labels : ``'ticks'`` | ``'names'`` | None
    How to label the nodes in the matrix along the x- and y-axes. If ``'ticks'``
    (default), the indices of the nodes are shown at evenly spaced intervals. Note that
    for many nodes, not all may have labels. If ``'names'``, each node's name is shown.
    Note that for many nodes, this can lead to overlapping labels. If ``None``, no
    labels are shown.
"""

docdict["viz_mask"] = """
mask : numpy.ndarray | None
    An array of boolean values, of the same shape as the data. Data that corresponds to
    ``False`` entries in the mask are plotted differently, as determined by
    ``mask_style``, ``mask_alpha``, and ``mask_cmap``. Useful for, e.g., highlighting
    areas of statistical significance. Default is ``None``, for no masking.
"""

docdict["viz_mask_style"] = """
mask_style : None | ``'contour'`` | ``'mask'`` | ``'both'``
    How to distinguish the masked/unmasked regions of the plot. If ``'contour'``, a line
    is drawn around the areas where ``mask`` is True. If ``'mask'``, areas where
    ``mask`` is ``False`` will be (partially) transparent, as determined by
    ``mask_alpha``. If ``'both'``, both a contour and transparency are used. Default is
    ``None``, which is silently ignored if ``mask`` is ``None``, and is interpreted like
    ``'both'`` otherwise.
"""

docdict["viz_mask_cmap"] = """
mask_cmap : matplotlib.colors.Colormap | str | None
    Colormap to use for masked areas of the plot. If a str, must be a valid Matplotlib
    colormap name. If ``None``, ``cmap`` is used for both masked and unmasked areas.
    Ignored if mask is ``None``. Default is ``'Greys'``.
"""

docdict["viz_mask_alpha"] = """
mask_alpha : float
    Relative opacity of the masked region versus the unmasked region, given as a float
    between 0 and 1 (0 means masked areas are not visible at all). Defaults to ``0.1``.
"""

docdict["viz_highlight"] = """
highlight : array_like of float, shape (2,) | array_like of float, shape (n, 2) | None
    Segments of the data to highlight by means of a light-yellow background color. The
    data periods to highlight must be specified as array-like objects in the form of
    ``(start, end)`` in the units of the data. Multiple periods can be specified by
    passing an array-like object of individual periods (e.g., for 3 periods, the shape
    of the passed object would be ``(3, 2)``. If ``None`` (default), no highlighting is
    applied.
"""

docdict["viz_interactive"] = """
interactive : bool
    Whether to make the plot interactive. This enables functionality like clicking on
    a connection to show its name. Defaults to ``True``.
"""

docdict["viz_show"] = """
show : bool
    Whether to show the figure(s). Defaults to ``True``.
"""

viz_figures_base = """
fig : instance of matplotlib.figure.Figure | list of instance of matplotlib.figure.Figure
    The figure(s) containing the connectivity plot(s).{}
"""  # noqa E501

docdict["viz_figure_image"] = viz_figures_base.format(
    " One figure is returned per connection."
)

docdict["viz_figures_line"] = viz_figures_base.format(
    " One figure is returned per channel types in the seeds and targets."
)

docdict["viz_figures_matrix"] = viz_figures_base.format(
    " One figure is returned per channel types in the seeds and targets. For "
    "multivariate connectivity with multiple components, one figure is also returned "
    "per component."
)

viz_components_note_base = """
Plotting for multivariate connectivity with multiple components is handled by treating
each component of the multivariate connections as a separate connection.{}
"""

docdict["viz_components_extra_con_note"] = viz_components_note_base.format(
    " The names of the nodes are differentiated by the addition of the component "
    "number to the node name, e.g., ``'node 0 (0)', 'node 0 (1)', ...``."
)

docdict["viz_components_extra_fig_note"] = viz_components_note_base.format(
    " The connections for each component are plotted on a separate figure."
)

docdict["viz_circle_line_note"] = """
The circle plot acts as an overview of the channels and their connections in the line
plot. If ``interactive`` is ``True``, left-clicking on a channel in the circle plot will
show the connections only for that channel (the exact behaviour is determined by
``selection``). Right-click to return to the original view. The circle plot is not shown
if the plotted connections correspond to only a single node in the figure.
"""

# Decoding initialisation
docdict["info_decoding"] = """
info : mne.Info
    Information about the data which will be decomposed and transformed, such as that
    coming from an :class:`mne.Epochs` object. The number of channels must match the
    subsequent input data.
"""

docdict["method_decoding"] = """
method : ``'cacoh'`` | ``'mic'``
    The multivariate method to use for the decomposition. Can be:

    * ``'cacoh'`` - Canonical Coherency (CaCoh) :footcite:`VidaurreEtAl2019`
    * ``'mic'`` - Maximised Imaginary part of Coherency (MIC) :footcite:`EwaldEtAl2012`
"""

docdict["fmin_decoding"] = """
fmin : float | None (default None)
    The lowest frequency of interest in Hz. Must not be ``None`` and only used if
    ``mode in ['multitaper', 'fourier']``.
"""

docdict["fmax_decoding"] = """
fmax : float | None (default None)
    The highest frequency of interest in Hz. Must not be ``None`` and only used if
    ``mode in ['multitaper', 'fourier']``.
"""

docdict["indices_decoding"] = """
indices : tuple of array_like
    A tuple of two array-likes, containing the indices of the seed and target channels
    in the input data, respectively. The indices of only a single connection (i.e.
    between one group of seeds and one group of targets) is supported, such that the
    array-likes have shape ``(n_seed_channels,)`` and ``(n_target_channels,)``.
"""

docdict["n_components"] = """
n_components : int | None (default None)
    The number of connectivity components (sources) to extract from the data. If
    ``None``, the number of components equal to the minimum rank of the seeds and
    targets is extracted (see the ``rank`` parameter). If an int, the number of
    components must be <= the minimum rank of the seeds and targets. E.g. if the seed
    channels had a ank of 5 and the target channels had a rank of 3, ``n_components``
    must be <= 3.
"""

docdict["rank"] = """
rank : tuple of int | None (default None)
    A tuple of two ints, containing the degree of rank subspace projection to apply to
    the seed and target data, respectively, before filters are fit. If ``None``, the
    rank of the seed and target data is used. If a tuple, the entries must be <= the
    rank of the seed and target data. The minimum rank of the seeds and targets
    determines the maximum number of connectivity components (sources) which can be
    extracted from the data (see the ``n_components`` parameter). Specifying ranks below
    that of the data may reduce the degree of overfitting when computing the filters.
"""

# Decoding attrs
docdict["filters_"] = """
filters_ : tuple of length 2
    A tuple of two arrays containing the spatial filters for transforming the seed and
    target data, respectively. The arrays have shape ``(n_components, n_seeds)`` and
    ``(n_components, n_targets)``.
"""

docdict["patterns_"] = """
patterns_ : tuple of length 2
    A tuple of two arrays containing the spatial patterns corresponding to the spatial
    filters for the seed and target data, respectively. The arrays have shape
    ``(n_components, n_seeds)`` and ``(n_components, n_targets)``.
"""

# Decoding plotting
docdict["info_decoding_plotting"] = """
info : mne.Info
    Information about the sensors of the data which has been decomposed, such as that
    coming from an :class:`mne.Epochs` object.
"""

# Topomaps
docdict["components_topomap"] = """
components : int | array_like of int | None (default None)
    The components to plot. If ``None``, all components are shown.
"""

docdict["ch_type_topomap"] = """
ch_type : ``'mag'`` | ``'grad'`` | ``'planar1'`` | ``'planar2'`` | ``'eeg'`` | None (default None)
    The channel type to plot. For ``'grad'``, the gradiometers are collected in pairs
    and the RMS for each pair is plotted. If ``None``, the first available channel type
    from the order shown above is used.
"""  # noqa E501

docdict["scalings_topomap"] = """
scalings : dict | float | None (default None)
    The scalings of the channel types to be applied for plotting. If ``None``, uses
    ``dict(eeg=1e6, grad=1e13, mag=1e15)``.
"""

docdict["sensors_topomap"] = """
sensors : bool | str (default True)
    Whether to add markers for sensor locations. If a str, should be a valid matplotlib
    format string (e.g., ``'r+'`` for red plusses; see the Notes section of
    :meth:`matplotlib.axes.Axes.plot`). If ``True``, black circles are used.
"""

docdict["show_names_topomap"] = """
show_names : bool | callable (default False)
    Whether to show channel names next to each sensor marker. If a callable, channel
    names will be formatted using the callable; e.g., to delete the prefix 'MEG ' from
    all channel names, pass the function ``lambda x: x.replace('MEG ', '')``. If
    ``mask`` is not ``None``, only non-masked sensor names will be shown.
"""

docdict["mask_filters_topomap"] = """
mask : array of bool, shape (n_channels, n_filters) | None (default None)
    An array specifying channel-filter combinations to highlight with a distinct
    plotting style. Array elements set to ``True`` will be plotted with the parameters
    given in ``mask_params``. If ``None``, no combinations will be highlighted.
"""
docdict["mask_patterns_topomap"] = """
mask : array of bool, shape (n_channels, n_patterns) | None (default None)
    An array specifying channel-pattern combinations to highlight with a distinct
    plotting style. Array elements set to ``True`` will be plotted with the parameters
    given in ``mask_params``. If ``None``, no combinations will be highlighted.
"""

docdict["mask_params_topomap"] = """
mask_params : dict | None (default None)
    The plotting parameters for distinct combinations given in ``mask``.
    Default ``None`` equals::

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
outlines : ``'head'`` | dict | None (default ``'head'``)
    The outlines to be drawn. If ``'head'``, the default head scheme will be drawn. If a
    dict, each key refers to a tuple of x and y positions, the values in ``'mask_pos'``
    will serve as an image mask. Alternatively, a matplotlib patch object can be passed
    for advanced masking options, either directly or as a function that returns patches
    (required for multi-axis plots). If ``None``, nothing will be drawn.
"""

docdict["sphere_topomap"] = """
sphere : float | array | mne.bem.ConductorModel | None  | ``'auto'`` | ``'eeglab'`` (default None)
    The sphere parameters to use for the head outline. Can be an array-like of shape
    (4,) to give the X/Y/Z origin and radius in meters, or a single float to give just
    the radius (origin assumed 0, 0, 0). Can also be an instance of a spherical
    :class:`mne.bem.ConductorModel` to use the origin and radius from that object. If
    ``'auto'`` the sphere is fit to digitization points. If ``'eeglab'`` the head circle
    is defined by EEG electrodes ``'Fpz'``, ``'Oz'``, ``'T7'``, and ``'T8'`` (if
    ``'Fpz'`` is not present, it will be approximated from the coordinates of ``'Oz'``).
    ``None`` is equivalent to ``'auto'`` when enough extra digitization points are
    available, and (0, 0, 0, 0.95) otherwise.
"""  # noqa E501

docdict["image_interp_topomap"] = """
image_interp : ``'cubic'`` | ``'nearest'`` | ``'linear'`` (default ``'cubic'``)
    The image interpolation to be used. Options are ``'cubic'`` to use
    :class:`scipy.interpolate.CloughTocher2DInterpolator`, ``'nearest'`` to use
    :class:`scipy.spatial.Voronoi`, or ``'linear'`` to use
    :class:`scipy.interpolate.LinearNDInterpolator`.
"""

docdict["extrapolate_topomap"] = """
extrapolate : ``'box'`` | ``'local'`` | ``'head'``
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
border : float | ``'mean'`` (default ``'mean'``)
    The value to extrapolate to on the topomap borders. If ``'mean'``, each extrapolated
    point has the average value of its neighbours.
"""

docdict["res_topomap"] = """
res : int (default 64)
    The resolution of the topomap image (number of pixels along each side).
"""

docdict["size_topomap"] = """
size : float (default 1.0)
    The side length of each subplot in inches.
"""

docdict["cmap_topomap"] = """
cmap : str | matplotlib.colors.Colormap | (matplotlib.colors.Colormap, bool) | ``'interactive'`` | None (default ``'RdBu_r'``)
    The colormap to use. If a str, should be a valid matplotlib colormap. If a tuple,
    the first value is :class:`matplotlib.colors.Colormap` object to use and the
    second value is a boolean defining interactivity. In interactive mode, the colors
    are adjustable by clicking and dragging the colorbar with left and right mouse
    button. Left mouse button moves the scale up and down and right mouse button adjusts
    the range. Hitting space bar resets the range. Up and down arrows can be used to
    change the colormap. If ``None``, ``'Reds'`` is used for data that is either all
    positive or all negative, and ``'RdBu_r'`` is used otherwise. ``'interactive'`` is
    equivalent to ``(None, True)``.

    .. warning::  Interactive mode works smoothly only for a small amount
        of topomaps. Interactive mode is disabled by default for more than
        2 topomaps.
"""  # noqa E501

docdict["vlim_topomap"] = """
vlim : tuple of float or None (default (None, None))
    The lower and upper colormap bounds, respectively. If both entries are ``None``,
    sets bounds to ``(min(data), max(data))``. If one entry is ``None``, the
    corresponding boundary is set at the min/max of the data.
"""

docdict["cnorm_topomap"] = """
cnorm : matplotlib.colors.Normalize | None (default None)
    How to normalize the colormap. If ``None``, standard linear normalization is used.
    If not ``None``, ``vlim`` is ignored. See the :ref:`Matplotlib docs
    <matplotlib:colormapnorms>` for more details on colormap normalization.
"""

docdict["colorbar_topomap"] = """
colorbar : bool (default True)
    Whether to plot a colorbar in the rightmost column of the figure.
"""

docdict["colorbar_format_topomap"] = r"""
cbar_fmt : str (default ``'%.1E'``)
    The formatting string for colorbar tick labels. See :ref:`formatspec` for details.
"""

docdict["units_topomap"] = """
units : str (default ``'AU'``)
    The units for the colorbar label. Ignored if ``colorbar=False``.
"""

docdict["axes_topomap"] = """
axes : tuple of length 2 | None (default None)
    The axes to plot to. If ``None``, a new figure will be created with the correct
    number of axes. If not ``None``, there must be two lists containing the axes for the
    seeds and targets, respectively. In each of these two lists, the number of axes must
    match ``components`` if ``colorbar=False``, or ``components * 2`` if
    ``colorbar=True``.
"""

docdict["name_format_topomap"] = r"""
name_format : str | None (default None)
    The string format for axes titles. If ``None``, uses ``f"{method}%01d ({group})"``,
    i.e., the method name followed by the component number and the group being plotted
    (seeds or targets). If not ``None``, it must contain a formatting specifier for the
    component number, and the group will be appended to the end.
"""

docdict["nrows_topomap"] = """
nrows : int | ``'auto'`` (default ``'auto'``)
    The number of rows of components to plot. If ``'auto'``, the necessary number will
    be inferred.
"""

docdict["ncols_topomap"] = """
ncols : int | ``'auto'`` (default ``'auto'``)
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
        raise RuntimeError(f"Error documenting {funcname}:\n{exp}")
    return f
