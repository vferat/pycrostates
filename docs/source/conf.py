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
import sys
import pydata_sphinx_theme
from sphinx_gallery.sorting import ExplicitOrder

curdir = os.path.dirname(__file__)

# -- Project information -----------------------------------------------------

project = 'pycrostates'
copyright = '2021, Victor Férat'
author = 'Victor Férat'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
   'sphinx.ext.autodoc',
   'sphinx.ext.napoleon',
   'sphinx.ext.autosummary',
   'sphinx.ext.doctest',
   'sphinx.ext.coverage',
   'sphinx.ext.mathjax',
   'sphinx.ext.viewcode',
   'sphinx.ext.intersphinx',
   'nbsphinx',
   'sphinx_gallery.gen_gallery',
   'recommonmark',
   'numpydoc']


# sphinx
master_doc = 'index'

# pygments style
pygments_style = 'default'

# A list of ignored prefixes for module index sorting.
modindex_common_prefix = ['pycrostates.']

# autosummary
autosummary_generate = True
autodoc_default_options = {'inherited-members': None}

# intersphinx_mapping
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/devdocs', None),
    'scipy': ('https://scipy.github.io/devdocs', None),
    'matplotlib': ('https://matplotlib.org', None),
    'mne': ('https://mne.tools/stable/', None),
    'joblib': ('https://joblib.readthedocs.io/en/latest', None),
    'pandas': ('https://pandas.pydata.org/pandas-docs/stable/', None),
    'sklearn': ('https://scikit-learn.org/stable/', None)
}

# numpy doc
numpydoc_class_members_toctree = False
numpydoc_attributes_as_param_list = True
numpydoc_xref_param_type = True

# sphinx_gallery_conf
sphinx_gallery_conf = {
     'examples_dirs': os.path.abspath(os.path.join(curdir, '..', '..', 'tutorials')),   # path to example scripts
     'gallery_dirs': 'auto_tutorials',  # path to where to save gallery generated output
     'subsection_order': ExplicitOrder(['../../tutorials/clustering',
                                       '../../tutorials/backfitting',
                                       '../../tutorials/group_level_analysis']),
     'reference_url': {
         # The module you locally document uses None
        'pycrostates': None,
     },
     'backreferences_dir'  : 'generated/backreferences',
     'doc_module': ('pycrostates',),


}

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'pydata_sphinx_theme'

html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/vferat/pycrostates",
            "icon": "fab fa-github-square",
        }],
    "external_links": [
       {"name": "mne", "url": "https://mne.tools/stable/index.html"}],

}
# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = [ os.path.abspath(os.path.join(curdir, '../_static'))]

def append_attr_meth_examples(app, what, name, obj, options, lines):
    """Append SG examples backreferences to method and attr docstrings."""
    # NumpyDoc nicely embeds method and attribute docstrings for us, but it
    # does not respect the autodoc templates that would otherwise insert
    # the .. include:: lines, so we need to do it.
    # Eventually this could perhaps live in SG.
    if what in ('attribute', 'method'):
        size = os.path.getsize(os.path.join(
            os.path.dirname(__file__), 'generated', 'backreferences', '%s.examples' % (name,)))
        if size > 0:
            lines += """
.. _sphx_glr_backreferences_{1}:
.. rubric:: Examples using ``{0}``:
.. minigallery:: {1}
""".format(name.split('.')[-1], name).split('\n')



# -- Auto-convert markdown pages to demo --------------------------------------
from recommonmark.transform import AutoStructify

def setup(app):
    app.connect('autodoc-process-docstring', append_attr_meth_examples)
    app.add_transform(AutoStructify)

