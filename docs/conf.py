# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import sys

sys.path.append('..')

from fastdfe import __version__

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'fastDFE'
author = 'Janek Sendrowski'
release = __version__
html_show_copyright = False

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx_autodoc_typehints',
    'sphinx_copybutton',
    'sphinx_paramlinks',  # anchors on :param: entries, so arguments are linkable
    'autodocsumm',  # per-class method-summary table at the top of each class
    'myst_nb',
    'sphinx_design',
    'sphinxcontrib.bibtex',
    'sphinx_book_theme'
]

# Page-level ``.. autosummary::`` blocks render an inline class table linking
# to the autoclass docs on the same page; no stub pages need generating.
autosummary_generate = False

bibtex_bibfiles = ['refs.bib']

typehints_use_signature = True
typehints_fully_qualified = False

# Silence unresolved ``plt`` forward refs in plot-method type annotations (matplotlib.pyplot
# is not in the documented modules' import namespace at autodoc time)
suppress_warnings = [
    'sphinx_autodoc_typehints.forward_reference',
]

# Resolve cross-references to the standalone sfsutils package (VCF parsing, spectra,
# annotation, filtration) and to standard-library / scientific-stack types in
# autodoc'd signatures against their published documentation.
intersphinx_mapping = {
    'sfsutils': ('https://sfsutils.readthedocs.io/en/latest/', None),
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
    'tskit': ('https://tskit.dev/tskit/docs/stable', None),
    'zarr': ('https://zarr.readthedocs.io/en/stable', None),
    'cyvcf2': ('https://brentp.github.io/cyvcf2', None),
}


pygments_style = 'default'

# disable notebook execution
nb_execution_mode = 'off'

# merge consecutive stdout/stderr chunks from one cell into a single output block
nb_merge_streams = True

templates_path = ['_templates']
# 'jupyter_execute' is a myst-nb build artifact. 'source' holds the User Guide sources, which docs/split_page.py and
# docs/merge_notebooks.py turn into the pages.
exclude_patterns = ['_build', 'jupyter_execute', 'outputs', 'source', 'Thumbs.db', '.DS_Store']

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
    # sphinx-book-theme puts the search in the primary sidebar and clears this in its
    # theme.conf, but pydata only honours that when the key is set here, so it re-adds a
    # second search field to the header. Clear it explicitly.
    'navbar_persistent': [],
    'search_bar_text': 'Search...',
    'repository_url': 'https://github.com/Sendrowski/fastdfe',
    'repository_branch': 'master',
    'use_repository_button': True,
    'use_edit_page_button': False,
    'use_issues_button': False,
    'use_download_button': False
}
html_static_path = ['_static']
html_css_files = ["custom.css"]
html_js_files = ["language-tabs.js"]
html_logo = "logo.png"
html_favicon = "favicon.ico"
