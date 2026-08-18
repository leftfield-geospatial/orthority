# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
from orthority.version import __version__

project = 'Orthority'
copyright = 'Leftfield Geospatial'
author = 'Leftfield Geospatial'
release = __version__

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.intersphinx',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosectionlabel',
    'sphinx_click',
    'sphinxarg.ext',
    'sphinx_copybutton',
]

exclude_patterns = ['_build', 'scripts']

# -- Options for HTML output -------------------------------------------------
html_theme = 'furo'

# -- Options for autodoc -----------------------------------------------------
autodoc_member_order = 'bysource'
autodoc_typehints = 'description'

# -- Options for intersphinx ---------------------------------------------------
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable', None),
    'rasterio': ('https://rasterio.readthedocs.io/en/stable/', None),
    'gdal': ('https://gdal.org/', None),
    'fsspec': ('https://filesystem-spec.readthedocs.io/en/latest/', None),
    'affine': ('https://affine.readthedocs.io/en/latest/', None),
}

# -- Options for pygments -----------------------------------------------------
highlight_language = 'none'

# -- Options for autosectionlabel ----------------------------------------------------
autosectionlabel_prefix_document = True
