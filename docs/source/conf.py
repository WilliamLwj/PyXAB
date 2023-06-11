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

sys.path.insert(0, os.path.abspath('../..'))
sys.path.insert(0, os.path.abspath('../../PyXAB'))

# -- Project information -----------------------------------------------------

project = 'PyXAB'
copyright = '2023, Wenjie Li'
author = 'Wenjie Li'

# The full version, including alpha/beta/rc tags
release = '0.2.4'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx_panels',
    'sphinx_gallery.gen_gallery',
]

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
use_rtd_scheme = False
try:
    import sphinx_rtd_theme

    extensions.extend(["sphinx_rtd_theme"])
    use_rtd_scheme = True
except ImportError:
    print("sphinx_rtd_theme was not installed, using alabaster as fallback!")

html_theme = "sphinx_rtd_theme" if use_rtd_scheme else "alabaster"


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = []


sphinx_gallery_conf = {
     'examples_dirs': ['../examples', '../contribute_examples'],  # path to your example scripts
     'gallery_dirs':  ['getting_started/auto_examples',  'info/auto_examples'] # path to where to save gallery generated output
}