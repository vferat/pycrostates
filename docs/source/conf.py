# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html


import inspect
import sys
from datetime import date
from importlib import import_module
from pathlib import Path
from typing import Dict, Optional

from sphinx_gallery.sorting import ExplicitOrder, FileNameSortKey

import pycrostates

# -- project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "pycrostates"
author = "Victor Férat"
copyright = f"{date.today().year}, {author}"
release = pycrostates.__version__
package = pycrostates.__name__
gh_url = "https://github.com/vferat/pycrostates"

# -- general configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# If your documentation needs a minimal Sphinx version, state it here.
needs_sphinx = "5.0"

# The document name of the “root” document, that is, the document that contains
# the root toctree directive.
root_doc = "index"

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.linkcode",
    "sphinx.ext.mathjax",
    "numpydoc",
    "sphinxcontrib.bibtex",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx_gallery.gen_gallery",
    "sphinx_issues",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

# Sphinx will warn about all references where the target cannot be found.
nitpicky = True
nitpick_ignore = [
    ("py:class", "None.  Remove all items from D."),
    ("py:class", "a set-like object providing a view on D's items"),
    ("py:class", "a set-like object providing a view on D's keys"),
    ("py:class", "an object providing a view on D's values"),
    (
        "py:class",
        "v, remove specified key and return the corresponding value.",
    ),
]

# A list of ignored prefixes for module index sorting.
modindex_common_prefix = [f"{package}."]

# The name of a reST role (builtin or Sphinx extension) to use as the default
# role, that is, for text marked up `like this`. This can be set to 'py:obj' to
# make `filter` a cross-reference to the Python function “filter”.
default_role = "py:obj"

# -- options for HTML output -------------------------------------------------
html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "logo": {
        "image_light": "img/Pycrostates_logo_black.png",
        "image_dark": "img/Pycrostates_logo_white.png",
    },
    "icon_links": [
        {
            "name": "GitHub",
            "url": gh_url,
            "icon": "fab fa-github-square",
        },
        {
            "name": "Slack",
            "url": "https://pycrostates.slack.com",
            "icon": "fab fa-slack",
        },
    ],
    "external_links": [
        {"name": "MNE", "url": "https://mne.tools/stable/index.html"}
    ],
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = [str(Path(__file__).parent.parent / "_static")]
html_css_files = ["style.css"]
# If true, links to the reST sources are added to the pages.
html_show_sourcelink = False
html_copy_source = False
html_show_sphinx = False

# variables to pass to HTML templating engine
html_context = {
    "pygment_light_style": "tango",
    "pygment_dark_style": "native",
}

# -- autodoc -----------------------------------------------------------------
# autodoc
autodoc_typehints = "none"
autodoc_member_order = "groupwise"
autodoc_warningiserror = True
autoclass_content = "class"
autodoc_default_options = {"inherited-members": None}

# -- autosummary -------------------------------------------------------------
autosummary_generate = True

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

# -- sphinx-issues -----------------------------------------------------------
issues_github_path = gh_url.split("http://github.com/")[-1]

# -- numpydoc ----------------------------------------------------------------
numpydoc_class_members_toctree = False
numpydoc_attributes_as_param_list = True

# x-ref
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
    "RawSegmentation": "pycrostates.segmentation.RawSegmentation",
    "EpochsSegmentation": "pycrostates.segmentation.EpochsSegmentation",
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

# validate
# https://numpydoc.readthedocs.io/en/latest/validation.html#validation-checks
error_ignores = {
    "GL01",  # docstring should start in the line immediately after the quotes
    "EX01",  # section 'Examples' not found
    "ES01",  # no extended summary found
    "SA01",  # section 'See Also' not found
    "RT02",  # The first line of the Returns section should contain only the type, unless multiple values are being returned  # noqa
    "SA04",  # Missing description for See Also "{reference_name}" reference'Missing description for See Also "{reference_name}" reference  # noqa
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
    # segmentation classes
    # TODO: the __init__.py methods should be refactored from *args, **kwargs
    # to the actual list of arguments.
    r"pycrostates.segmentation.EpochsSegmentation",
    r"pycrostates.segmentation.RawSegmentation",
}

# -- sphinxcontrib-bibtex ----------------------------------------------------
bibtex_bibfiles = ["../references.bib"]

# -- sphinx.ext.linkcode -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/linkcode.html

gh_urls = {
    package: gh_url,
    "mne": "https://github.com/mne-tools/mne-python",
}


def linkcode_resolve(domain: str, info: Dict[str, str]) -> Optional[str]:
    """Determine the URL corresponding to a Python object.

    Parameters
    ----------
    domain : str
        One of 'py', 'c', 'cpp', 'javascript'.
    info : dict
        With keys "module" and "fullname".

    Returns
    -------
    url : str | None
        The code URL. If None, no link is added.
    """
    if domain != "py":
        return None  # only document python objects

    module = info["module"].split(".")[0]
    if module not in gh_urls:
        raise RuntimeError(
            "This module is not configured in 'linkcode_resolve'. Please edit "
            "the documentation 'conf.py' file."
        )

    # retrieve pyobject and file
    try:
        pyobject = import_module(info["module"])
        for elt in info["fullname"].split("."):
            pyobject = getattr(pyobject, elt)
        while hasattr(pyobject, '__wrapped__'):
            pyobject = pyobject.__wrapped__
        fname = inspect.getsourcefile(pyobject).replace("\\", "/")
    except Exception:
        # Either the object could not be loaded or the file was not found.
        # For instance, properties will raise.
        return None

    # retrieve start/stop lines
    source, start_line = inspect.getsourcelines(pyobject)
    lines = "L%d-L%d" % (start_line, start_line + len(source) - 1)

    # create URL
    if "dev" in release:
        branch = "main"
    else:
        return None  # alternatively, link to a maint/version branch

    while module in fname:
        fname = fname.split(module)[-1]
    assert fname[0] == "/"  # sanity-check, file separator
    fname = fname[1:]
    return f"{gh_urls[module]}/blob/{branch}/{package}/{fname}#{lines}"


# -- sphinx-gallery ----------------------------------------------------------
sphinx_gallery_conf = {
    "backreferences_dir": "generated/backreferences",
    "doc_module": ("pycrostates",),
    "examples_dirs": [str(Path(__file__).parent.parent.parent / "tutorials")],
    "exclude_implicit_doc": {
        r"pycrostates.CHData",
        r"pycrostates.ChData.get_data",
    },
    "gallery_dirs": ["generated/auto_tutorials"],
    "line_numbers": False,  # messes with style
    "plot_gallery": True,
    "reference_url": dict(pycrostates=None),  # documented lib uses None
    "remove_config_comments": True,
    "show_memory": sys.platform == "linux",
    "subsection_order": ExplicitOrder(
        [
            "../../tutorials/preprocessing",
            "../../tutorials/clustering",
            "../../tutorials/group_level_analysis",
            "../../tutorials/metrics",
        ]
    ),
    "within_subsection_order": FileNameSortKey,
}

def append_attr_meth_examples(app, what, name, obj, options, lines):
    """Append SG examples backreferences to method and attr docstrings."""
    # NumpyDoc nicely embeds method and attribute docstrings for us, but it
    # does not respect the autodoc templates that would otherwise insert
    # the .. include:: lines, so we need to do it.
    # Eventually this could perhaps live in SG.
    if what in ("attribute", "method"):
        backrefs = Path(__file__).parent / "generated" / "backreferences"
        size = (backrefs / f"{name}.examples").stat().st_size
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
