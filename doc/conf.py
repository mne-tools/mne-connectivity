"""Configure details for documentation with sphinx."""
from datetime import date
import os
import sys
import warnings

import sphinx_gallery  # noqa: F401
from sphinx_gallery.sorting import ExampleTitleSortKey

import mne

sys.path.insert(0, os.path.abspath(".."))
import mne_connectivity  # noqa: E402

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
curdir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(curdir, "..")))
sys.path.append(os.path.abspath(os.path.join(curdir, "..", "mne_connectivity")))
sys.path.append(os.path.abspath(os.path.join(curdir, "sphinxext")))

# -- General configuration ------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
needs_sphinx = "4.0"

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx_gallery.gen_gallery",
    "sphinxcontrib.bibtex",
    "sphinx_issues",
    "numpydoc",
    "sphinx_copybutton",
]

# configure sphinx-issues
issues_github_path = "mne-tools/mne-connectivity"

# configure sphinx-copybutton
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True

# generate autosummary even if no references
# -- sphinx.ext.autosummary
autosummary_generate = True

autodoc_default_options = {"inherited-members": None}
autodoc_typehints = "signature"

# prevent jupyter notebooks from being run even if empty cell
# nbsphinx_execute = 'never'
# nbsphinx_allow_errors = True

error_ignores = {
    # These we do not live by:
    "GL01",  # Docstring should start in the line immediately after the quotes
    "EX01",
    "EX02",  # examples failed (we test them separately)
    "ES01",  # no extended summary
    "SA01",  # no see also
    "YD01",  # no yields section
    "SA04",  # no description in See Also
    "PR04",  # Parameter "shape (n_channels" has no type
    "RT02",  # The first line of the Returns section should contain only the type, unless multiple values are being returned  # noqa
    # XXX should also verify that | is used rather than , to separate params
    # XXX should maybe also restore the parameter-desc-length < 800 char check
}

# -- numpydoc
# Below is needed to prevent errors
numpydoc_xref_param_type = True
numpydoc_class_members_toctree = False
numpydoc_attributes_as_param_list = True
numpydoc_use_blockquotes = True
numpydoc_xref_ignore = {
    # words
    "instance",
    "instances",
    "of",
    "default",
    "shape",
    "or",
    "with",
    "length",
    "pair",
    "matplotlib",
    "optional",
    "kwargs",
    "in",
    "dtype",
    "object",
    "self.verbose",
    # shapes
    "n_times",
    "obj",
    "n_chan",
    "n_epochs",
    "n_picks",
    "n_ch_groups",
    "n_node_names",
    "n_tapers",
    "n_signals",
    "n_components",
    "n_step",
    "n_freqs",
    "epochs",
    "freqs",
    "times",
    "components",
    "arrays",
    "lists",
    "func",
    "n_nodes",
    "n_estimated_nodes",
    "n_samples",
    "n_channels",
    "n_patterns",
    "n_filters",
    "Renderer",
    "n_ytimes",
    "n_ychannels",
    "n_events",
    "n_cons",
    "max_n_chans",
    "n_unique_seeds",
    "n_unique_targets",
    "variable",
}
numpydoc_xref_aliases = {
    # Python
    "file-like": ":term:`file-like <python:file object>`",
    # Matplotlib
    "colormap": ":doc:`colormap <matplotlib:tutorials/colors/colormaps>`",
    "color": ":doc:`color <matplotlib:api/colors_api>`",
    "collection": ":doc:`collections <matplotlib:api/collections_api>`",
    "Axes": "matplotlib.axes.Axes",
    "Figure": "matplotlib.figure.Figure",
    "Axes3D": "mpl_toolkits.mplot3d.axes3d.Axes3D",
    "PolarAxes": "matplotlib.projections.polar.PolarAxes",
    "ColorbarBase": "matplotlib.colorbar.ColorbarBase",
    # sklearn
    "MetadataRequest": "sklearn.utils.metadata_routing.MetadataRequest",
    "estimator": "sklearn.base.BaseEstimator",
    # joblib
    "joblib.Parallel": "joblib.Parallel",
    # nibabel
    "Nifti1Image": "nibabel.nifti1.Nifti1Image",
    "Nifti2Image": "nibabel.nifti2.Nifti2Image",
    "SpatialImage": "nibabel.spatialimages.SpatialImage",
    # MNE
    "Label": "mne.Label",
    "Forward": "mne.Forward",
    "Evoked": "mne.Evoked",
    "Info": "mne.Info",
    "SourceSpaces": "mne.SourceSpaces",
    "SourceMorph": "mne.SourceMorph",
    "Epochs": "mne.Epochs",
    "Layout": "mne.channels.Layout",
    "EvokedArray": "mne.EvokedArray",
    "BiHemiLabel": "mne.BiHemiLabel",
    "AverageTFR": "mne.time_frequency.AverageTFR",
    "EpochsTFR": "mne.time_frequency.EpochsTFR",
    "Raw": "mne.io.Raw",
    "ICA": "mne.preprocessing.ICA",
    # MNE-Connectivity
    "Connectivity": "mne_connectivity.Connectivity",
    # dipy
    "dipy.align.AffineMap": "dipy.align.imaffine.AffineMap",
    "dipy.align.DiffeomorphicMap": "dipy.align.imwarp.DiffeomorphicMap",
}
numpydoc_validate = True
numpydoc_validation_checks = {"all"} | set(error_ignores)
numpydoc_validation_exclude = {  # set of regex
    # dict subclasses
    r"\.clear",
    r"\.get$",
    r"\.copy$",
    r"\.fromkeys",
    r"\.items",
    r"\.keys",
    r"\.pop",
    r"\.popitem",
    r"\.setdefault",
    r"\.update",
    r"\.values",
    # list subclasses
    r"\.append",
    r"\.count",
    r"\.extend",
    r"\.index",
    r"\.insert",
    r"\.remove",
    r"\.sort",
    # we currently don't document these properly (probably okay)
    r"\.__getitem__",
    r"\.__contains__",
    r"\.__hash__",
    r"\.__mul__",
    r"\.__sub__",
    r"\.__add__",
    r"\.__iter__",
    r"\.__div__",
    r"\.__neg__",
    r"plot_circle",
}


default_role = "py:obj"

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = ".rst"

# The master toctree document.
master_doc = "index"

# General information about the project.
project = "MNE-Connectivity"
td = date.today()
copyright = "2021-%s, MNE Developers. Last updated on %s" % (td.year, td.isoformat())

author = "Adam Li"

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
# The full version, including alpha/beta/rc tags.
release = mne_connectivity.__version__
# The short X.Y version.
version = ".".join(release.split(".")[:2])

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

# HTML options (e.g., theme)
# see: https://sphinx-bootstrap-theme.readthedocs.io/en/latest/README.html
# Clean up sidebar: Do not show "Source" link
html_show_sourcelink = False
html_copy_source = False

html_theme = "pydata_sphinx_theme"

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]
html_static_path = ["_static"]
html_css_files = ["style.css"]
html_sidebars = {
    "whats_new": [],
    "install": [],
}

switcher_version_match = "dev" if "dev" in release else version

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
html_theme_options = {
    "icon_links": [
        dict(
            name="GitHub",
            url="https://github.com/mne-tools/mne-connectivity",
            icon="fab fa-github-square",
        ),
    ],
    "use_edit_page_button": False,
    "navigation_with_keys": False,
    "show_toc_level": 1,
    "navbar_end": ["theme-switcher", "version-switcher", "navbar-icon-links"],
    "switcher": {
        "json_url": "https://mne.tools/mne-connectivity/dev/_static/versions.json",
        "version_match": switcher_version_match,
    },
    "back_to_top_button": False,
}
# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "mne": ("https://mne.tools/dev", None),
    "mne-bids": ("https://mne.tools/mne-bids/dev/", None),
    "numpy": ("https://numpy.org/devdocs", None),
    "scipy": ("https://scipy.github.io/devdocs", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/dev", None),
    "sklearn": ("https://scikit-learn.org/stable", None),
    "pyvista": ("https://docs.pyvista.org", None),
    "joblib": ("https://joblib.readthedocs.io/en/latest", None),
    "nibabel": ("https://nipy.org/nibabel", None),
    "nilearn": ("http://nilearn.github.io/stable", None),
    "dipy": ("https://docs.dipy.org/stable", None),
}
intersphinx_timeout = 5

# Resolve binder filepath_prefix. From the docs:
# "A prefix to append to the filepath in the Binder links. You should use this
# if you will store your built documentation in a sub-folder of a repository,
# instead of in the root."
# we will store dev docs in a `dev` subdirectory and all other docs in a
# directory "v" + version_str. E.g., "v0.3"
if "dev" in version:
    filepath_prefix = "dev"
else:
    filepath_prefix = "v{}".format(version)

os.environ["_MNE_BUILDING_DOC"] = "true"
scrapers = ("matplotlib",)
try:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        import pyvista
    pyvista.OFF_SCREEN = False
    pyvista.BUILDING_GALLERY = True
except Exception:
    pass
else:
    scrapers += ("pyvista",)
if "pyvista" in scrapers:
    import mne.viz._brain

    brain_scraper = mne.viz._brain._BrainScraper()
    scrapers = list(scrapers)
    scrapers.insert(scrapers.index("pyvista"), brain_scraper)
    scrapers = tuple(scrapers)

sphinx_gallery_conf = {
    "doc_module": "mne_connectivity",
    "reference_url": {
        "mne_connectivity": None,
    },
    "backreferences_dir": "generated",
    "plot_gallery": "True",  # Avoid annoying Unicode/bool default warning
    "within_subsection_order": ExampleTitleSortKey,
    "examples_dirs": ["../examples"],
    "gallery_dirs": ["auto_examples"],
    "filename_pattern": "^((?!sgskip).)*$",
    "matplotlib_animations": True,
    "compress_images": ("images", "thumbnails"),
    "image_scrapers": scrapers,
    "expected_failing_examples": ["../examples/granger_causality.py"],
    "show_signature": False,
}

# sphinxcontrib-bibtex
bibtex_bibfiles = ["./references.bib"]
bibtex_style = "unsrt"
bibtex_footbibliography_header = ""


# Enable nitpicky mode - which ensures that all references in the docs
# resolve.

nitpicky = True
nitpick_ignore = []

suppress_warnings = [
    "config.cache",  # our rebuild is okay
]


def fix_sklearn_inherited_docstrings(app, what, name, obj, options, lines):
    """Fix sklearn docstrings because they use autolink and we do not."""
    if (
        name.startswith("mne_connectivity.decoding.")
    ) and name.endswith(
        (
            ".get_metadata_routing",
            ".fit",
            ".fit_transform",
            ".set_output",
            ".transform",
        )
    ):
        if ":Parameters:" in lines:
            loc = lines.index(":Parameters:")
        else:
            loc = lines.index(":Returns:")
        lines.insert(loc, "")
        lines.insert(loc, ".. default-role:: autolink")
        lines.insert(loc, "")


def setup(app):
    """Set up the Sphinx app."""
    app.connect("autodoc-process-docstring", fix_sklearn_inherited_docstrings)
