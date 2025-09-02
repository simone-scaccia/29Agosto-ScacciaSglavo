import os, sys
sys.path.insert(0, os.path.abspath("../src"))
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'SS MultiAgentSystem'
copyright = '2025, Sglavo Scaccia'
author = 'Sglavo Scaccia'
release = '1.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# conf.py
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_rtd_theme",
    "sphinx_autodoc_typehints"  # if you use Google/NumPy style docstrings
]

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = True
napoleon_include_special_with_doc = True

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "member-order": "bysource",
    "special-members": "__call__",
    "exclude-members": "__weakref__"
}

# Show both signature and
# annotations if decoration stripped hints
autodoc_typehints = "description"
autodoc_typehints_format = "short"
autodoc_member_order = "bysource"
# If a decorator confused the
# signature, let Sphinx take it from the
# docstring’s first line
autodoc_docstring_signature = True


templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'

html_theme_options = {
    "navigation_depth": 4,
    "collapse_navigation": False,
}

templates_path = ['_templates']

exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

source_suffix = '.rst'

master_doc = 'index'
