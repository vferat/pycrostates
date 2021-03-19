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
   'm2r2']


# sphinx
master_doc = 'index'

# sphinx m2r
source_suffix = ['.rst', '.md']

# sphinx autosummary
autosummary_generate = True
autodoc_default_options = {'inherited-members': None}

# sphinx_gallery_conf
sphinx_gallery_conf = {
     'examples_dirs': os.path.abspath(os.path.join(curdir, '..', '..', 'tutorials')),   # path to example scripts
     'gallery_dirs': 'auto_tutorials',  # path to where to save gallery generated output
}

# intersphinx_mapping
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/devdocs', None),
    'scipy': ('https://scipy.github.io/devdocs', None),
    'matplotlib': ('https://matplotlib.org', None),
    'mne': ('https://mne.tools/stable/', None),
    'joblib': ('https://joblib.readthedocs.io/en/latest', None),
    'pandas': ('https://pandas.pydata.org/pandas-docs/stable/', None),
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
html_static_path = [ os.path.abspath(os.path.join(curdir, '..', '_static'))]