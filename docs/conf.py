# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import datetime
import sys

sys.path.append('..')

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'fastDFE'
year = datetime.datetime.now().year
copyright = f'{year}, Janek Sendrowski'
author = 'Janek Sendrowski'
release = '1.3.3'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.viewcode',
    'sphinx_autodoc_typehints',
    'sphinx_copybutton',
    'autodocsumm',  # per-class method-summary table at the top of each class
    'myst_nb',
    'sphinxcontrib.bibtex',
    'sphinx_book_theme'
]

# Page-level ``.. autosummary::`` blocks render an inline class table linking
# to the autoclass docs on the same page; no stub pages need generating.
autosummary_generate = False

bibtex_bibfiles = ['refs.bib']

typehints_use_signature = True
typehints_fully_qualified = False

pygments_style = 'default'

# disable notebook execution
nb_execution_mode = 'off'

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

autodoc_default_options = {
    'members': True,
    'inherited-members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'show-inheritance': True,
    # autodocsumm: prepend a compact summary table (names + one-line descriptions)
    # before the full docs -- a class list after each module docstring and a method
    # list after each class docstring. Sections are ;;-separated; restricting to
    # Classes and Methods skips the Attributes table (it duplicates the per-attribute
    # docs below). Signatures are dropped to keep the tables to one line per entry.
    'autosummary': True,
    'autosummary-sections': 'Classes;;Methods',
    'autosummary-nosignatures': True
}

add_module_names = False

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_book_theme'
html_theme_options = {
    'search_bar_text': 'Search...',
    'repository_url': 'https://github.com/Sendrowski/fastdfe',
    'repository_branch': 'master',
    'use_repository_button': True,
    'use_edit_page_button': False,
    'use_issues_button': False
}
html_static_path = ['_static']
html_css_files = ["custom.css"]
html_logo = "logo.png"
html_favicon = "favicon.ico"
