# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path information -----------------------------------------------------
import os
import sys

sys.path.insert(0, os.path.abspath('..'))
from orthority.version import __version__

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'orthority'
copyright = '2023, Leftfield Geospatial'
author = 'Leftfield Geospatial'
release = __version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx_click',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# -- Options for autodoc -----------------------------------------------------
# see https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html
# autodoc_mock_imports = ['rasterio', 'click']
autosummary_generate = True
autoclass_content = 'class'
autodoc_class_signature = 'separated'
autodoc_member_order = 'bysource'
autodoc_typehints = 'both'


# -- Options for autosectionlabel ----------------------------------------------------
# Make sure the target is unique
autosectionlabel_prefix_document = True
autosectionlabel_maxdepth = 3  # avoid duplicate section labels for CLI examples
