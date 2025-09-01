import os, sys
sys.path.insert(0, os.path.abspath("../src"))

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'plot_gen'
copyright = '2025, sim_gio'
author = 'sim_gio'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = []

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",  # per Google/NumPy style
    "sphinx.ext.autodoc.typehints",
]
napoleon_google_docstring = True
napoleon_numpy_docstring = False


# # Make sure the sidebar actually contains a local toc

# html_sidebars = {

#     "**": [

#         "about.html",

#         "searchbox.html",

#         "localtoc.html",   # ← this renders the section menu

#         "relations.html",

#     ]

# }

 

# # Nice-to-have options

# html_theme_options = {

#     "sidebar_includehidden": True,  # include hidden toctree entries

#     "fixed_sidebar": True,          # sticky sidebar

# }
