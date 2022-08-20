# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os

from recommonmark.transform import AutoStructify
from sphinx_gallery.sorting import ExplicitOrder

curdir = os.path.dirname(__file__)

# -- Project information -----------------------------------------------------

project = "pycrostates"
copyright = "2021, Victor Férat"
author = "Victor Férat"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# If your documentation needs a minimal Sphinx version, state it here.
needs_sphinx = "4.0"

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.doctest",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "nbsphinx",
    "sphinxcontrib.bibtex",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx_gallery.gen_gallery",
    "recommonmark",
    "numpydoc",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# Sphinx will warn about all references where the target cannot be found.
nitpicky = True
nitpick_ignore = []

# The document name of the “root” document, that is, the document that contains
# the root toctree directive.
root_doc = "index"

# A list of ignored prefixes for module index sorting.
modindex_common_prefix = ["pycrostates."]

# The style name to use for Pygments highlighting of source code. If not set,
# either the theme’s default style or 'sphinx' is selected for HTML output.
pygments_style = "default"

# -- autodoc-autosummary -----------------------------------------------------
# autodoc
autodoc_typehints = 'none'
autodoc_member_order = "groupwise"

# autosummary
autosummary_generate = True
autodoc_default_options = {"inherited-members": None}

# -- autosectionlabels -------------------------------------------------------
autosectionlabel_prefix_document = True

# -- intersphinx -------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/devdocs", None),
    "scipy": ("https://scipy.github.io/devdocs", None),
    "matplotlib": ("https://matplotlib.org", None),
    "mne": ("https://mne.tools/stable/", None),
    "joblib": ("https://joblib.readthedocs.io/en/latest", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
}
intersphinx_timeout = 5

# -- numpydoc ----------------------------------------------------------------

# https://numpydoc.readthedocs.io/en/latest/validation.html#validation-checks
error_ignores = {
    "GL01",  # docstring should start in the line immediately after the quotes
    "EX01",  # section 'Examples' not found
    "ES01",  # no extended summary found
    "SA01",  # section 'See Also' not found
    "RT02",  # The first line of the Returns section should contain only the type, unless multiple values are being returned  # noqa
}

numpydoc_class_members_toctree = False
numpydoc_attributes_as_param_list = True
numpydoc_xref_param_type = True

numpydoc_xref_aliases = {
    # Python
    "bool": ":class:`python:bool`",
    "Path": "pathlib.Path",
    # MNE
    "DigMontage": "mne.channels.DigMontage",
    "Epochs": "mne.Epochs",
    "Evoked": "mne.Evoked",
    "Info": "mne.Info",
    "Projection": "mne.Projection",
    "Raw": "mne.io.Raw",
    # Pycrostates:
    "ChData": "pycrostates.io.ChData",
    "ChInfo": "pycrostates.io.ChInfo",
    # Matplotlib
    "Axes": "matplotlib.axes.Axes",
    "Figure": "matplotlib.figure.Figure",
}
numpydoc_xref_ignore = {
    "instance",
    "of",
    "shape",
    "n_channels",
    "n_clusters",
    "n_epochs",
    "n_samples",
}

numpydoc_validate = True
numpydoc_validation_checks = {"all"} | set(error_ignores)
numpydoc_validation_exclude = {  # regex to ignore during docstring check
    r"\.__getitem__",
    r"\.__contains__",
    r"\.__hash__",
    r"\.__mul__",
    r"\.__sub__",
    r"\.__add__",
    r"\.__iter__",
    r"\.__div__",
    r"\.__neg__",
    # dict subclasses
    r"\.clear",
    r"\.get$",
    r"\.fromkeys",
    r"\.items",
    r"\.keys",
    r"\.pop",
    r"\.popitem",
    r"\.setdefault",
    r"\.update",
    r"\.values",
    # copy methods
    r"\.copy",
}

# -- sphinxcontrib-bibtex ----------------------------------------------------
bibtex_bibfiles = ['../references.bib']

# -- sphinx-gallery ----------------------------------------------------------
sphinx_gallery_conf = {
    "examples_dirs": os.path.abspath(
        os.path.join(curdir, "..", "..", "tutorials")
    ),
    "gallery_dirs": "auto_tutorials",
    "subsection_order": ExplicitOrder(
        [
            "../../tutorials/preprocessing",
            "../../tutorials/clustering",
            "../../tutorials/group_level_analysis",
            "../../tutorials/metrics",
        ]
    ),
    "reference_url": {"pycrostates": None},  # current lib uses None
    "backreferences_dir": "generated/backreferences",
    "doc_module": ("pycrostates",),
}

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "pydata_sphinx_theme"

html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/vferat/pycrostates",
            "icon": "fab fa-github-square",
        },
        {
            "name": "Slack",
            "url": "https://pycrostates.slack.com",
            "icon": "fab fa-slack",
        }
    ],
    "external_links": [
        {"name": "MNE", "url": "https://mne.tools/stable/index.html"}
    ],
}
# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = [os.path.abspath(os.path.join(curdir, "../_static"))]
html_css_files = [
    'style.css',
]
# If true, links to the reST sources are added to the pages.
html_show_sourcelink = False
html_copy_source = False
# variables to pass to HTML templating engine
html_context = {
    'build_dev_html': bool(int(os.environ.get('BUILD_DEV_HTML', False))),
    'default_mode': 'auto',
    'pygment_light_style': 'tango',
    'pygment_dark_style': 'native',
}


def append_attr_meth_examples(app, what, name, obj, options, lines):
    """Append SG examples backreferences to method and attr docstrings."""
    # NumpyDoc nicely embeds method and attribute docstrings for us, but it
    # does not respect the autodoc templates that would otherwise insert
    # the .. include:: lines, so we need to do it.
    # Eventually this could perhaps live in SG.
    if what in ("attribute", "method"):
        size = os.path.getsize(
            os.path.join(
                os.path.dirname(__file__),
                "generated",
                "backreferences",
                "%s.examples" % (name,),
            )
        )
        if size > 0:
            lines += """
.. _sphx_glr_backreferences_{1}:
.. rubric:: Examples using ``{0}``:
.. minigallery:: {1}
""".format(
                name.split(".")[-1], name
            ).split(
                "\n"
            )


# -- Auto-convert markdown pages to demo --------------------------------------
def setup(app):
    app.connect("autodoc-process-docstring", append_attr_meth_examples)
    app.add_transform(AutoStructify)
