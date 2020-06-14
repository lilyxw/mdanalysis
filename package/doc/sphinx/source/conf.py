# -*- coding: utf-8 -*-
#
# MDAnalysis documentation build configuration file, created by
# sphinx-quickstart on Mon Sep 27 09:39:55 2010.
#
# This file is execfile()d with the current directory set to its containing dir.
#
# Note that not all possible configuration values are present in this
# autogenerated file.
#
# All configuration values have a default; values that are commented out
# serve to show the default.

import sys
import os
import platform
import datetime
import msmb_theme  # for little versions pop-up
# https://sphinx-rtd-theme.readthedocs.io/en/stable/
import sphinx_rtd_theme

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown
# here.

# make sure sphinx always uses the current branch
sys.path.insert(0, os.path.abspath('../../..'))

# -- General configuration -----------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be extensions
# coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
extensions = ['sphinx.ext.autodoc', 'sphinx.ext.intersphinx',
              'sphinx.ext.mathjax', 'sphinx.ext.viewcode',
              'sphinx.ext.napoleon', 'sphinx.ext.todo',
              'sphinx_sitemap',
              'sphinx_rtd_theme']

mathjax_path = 'https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.0/MathJax.js?config=TeX-AMS-MML_HTMLorMML'

# for sitemap with https://github.com/jdillard/sphinx-sitemap
# NOTE: This sitemap is only correct for the DEVELOPMENT doccs. The RELEASE docs
#       are served from https://www.mdanalysis.org/docs/ and the sitemap.xml
#       is manually fixed when deploying the release docs with the
#       maintainer/deploy_master_docs.sh script
site_url = "https://mdadocs.minium.com.au"

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix of source filenames.
source_suffix = '.rst'

# The encoding of source files.
#source_encoding = 'utf-8-sig'

# The master toctree document.
# 'index' has the advantage that it is immediately picked up by the webserver
master_doc = 'index'

# General information about the project.
# (take the list from AUTHORS)
# Ordering: (1) Naveen (2) Elizabeth, then all contributors in alphabetical order
#           (last) Oliver
author_list = __import__('MDAnalysis').__authors__
authors = u', '.join(author_list[:-1]) + u', and ' + author_list[-1]
project = u'MDAnalysis'
now = datetime.datetime.now()
copyright = u'2005-{}, '.format(now.year) + authors

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# Dynamically calculate the version
packageversion = __import__('MDAnalysis').__version__
# The short X.Y version.
# commented out version because it wasn't used except by sitemap
# version = '.'.join(packageversion.split('.')[:2])
# The full version, including alpha/beta/rc tags.
release = packageversion

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#language = None

# There are two options for replacing |today|: either, you set today to some
# non-false value, then it is used:
#today = ''
# Else, today_fmt is used as the format for a strftime call.
#today_fmt = '%B %d, %Y'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = []

# The reST default role (used for this markup: `text`) to use for all documents.
#default_role = None

# If true, '()' will be appended to :func: etc. cross-reference text.
#add_function_parentheses = True

# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
#add_module_names = True

# If true, sectionauthor and moduleauthor directives will be shown in the
# output. They are ignored by default.
#show_authors = False

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'default'

# A list of ignored prefixes for module index sorting.
#modindex_common_prefix = []

#to include decorated objects like __init__
autoclass_content = 'both'

# -- Options for HTML output ---------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = 'msmb_theme'

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
# styles/fonts to match http://mdanalysis.org (see public/css)
#
# /* MDAnalysis orange: #FF9200 */
# /* MDAnalysis gray: #808080 */
# /* MDAnalysis white: #FFFFFF */
# /* MDAnalysis black: #000000 */

color = {'orange': '#FF9200',
         'gray': '#808080',
         'white': '#FFFFFF',
         'black': '#000000',}

html_theme_options = {
    'canonical_url': '',
    'logo_only': True,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    'style_nav_header_background': 'white',
    # Toc options
    'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False,
}

html_context = {
    'versions_json_url': 'https://mdadocs.minium.com.au/versions.json'
}

# Add any paths that contain custom themes here, relative to this directory.
html_theme_path = [
    msmb_theme.get_html_theme_path(),
    sphinx_rtd_theme.get_html_theme_path()
]

# The name for this set of Sphinx documents.  If None, it defaults to
# "<project> v<release> documentation".
#html_title = None

# A shorter title for the navigation bar.  Default is the same as html_title.
#html_short_title = None

# The name of an image file (relative to this directory) to place at the top
# of the sidebar. --- use theme
html_logo = "_static/logos/mdanalysis-logo-thin.png"

# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
html_favicon = "_static/logos/mdanalysis-logo.ico"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# For RTD theme: custom.css to override theme defaults.
html_static_path = ['_static']
html_css_files = ['custom.css']

# If not '', a 'Last updated on:' timestamp is inserted at every page bottom,
# using the given strftime format.
#html_last_updated_fmt = '%b %d, %Y'

# If true, SmartyPants will be used to convert quotes and dashes to
# typographically correct entities.
#html_use_smartypants = True

# Custom sidebar templates, maps document names to template names.
# alabaster sidebars
html_sidebars = {
    '**': [
        'about.html',
        'navigation.html',
        'relations.html',
        'searchbox.html',
    ]
}

# Additional templates that should be rendered to pages, maps page names to
# template names.
#html_additional_pages = {}

# If false, no module index is generated.
#html_domain_indices = True

# If false, no index is generated.
#html_use_index = True

# If true, the index is split into individual pages for each letter.
#html_split_index = False

# If true, links to the reST sources are added to the pages.
#html_show_sourcelink = True

# If true, "Created using Sphinx" is shown in the HTML footer. Default is True.
#html_show_sphinx = True

# If true, "(C) Copyright ..." is shown in the HTML footer. Default is True.
#html_show_copyright = True

# If true, an OpenSearch description file will be output, and all pages will
# contain a <link> tag referring to it.  The value of this option must be the
# base URL from which the finished HTML is served.
html_use_opensearch = 'https://docs.mdanalysis.org'

# This is the file name suffix for HTML files (e.g. ".xhtml").
#html_file_suffix = None

# Output file base name for HTML help builder.
htmlhelp_basename = 'MDAnalysisdoc'


# -- Options for LaTeX output --------------------------------------------------

# The paper size ('letter' or 'a4').
#latex_paper_size = 'letter'

# The font size ('10pt', '11pt' or '12pt').
#latex_font_size = '10pt'

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title, author, documentclass [howto/manual]).
latex_documents = [
  ('MDAnalysis.tex', u'MDAnalysis Documentation',
   authors, 'manual'),
]

# The name of an image file (relative to this directory) to place at the top of
# the title page.
#latex_logo = None

# For "manual" documents, if this is true, then toplevel headings are parts,
# not chapters.
#latex_use_parts = False

# If true, show page references after internal links.
#latex_show_pagerefs = False

# If true, show URL addresses after external links.
#latex_show_urls = False

# Additional stuff for the LaTeX preamble.
#latex_preamble = ''

# Documents to append as an appendix to all manuals.
#latex_appendices = []

# If false, no module index is generated.
#latex_domain_indices = True


# -- Options for manual page output --------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    ('mdanalysis', u'MDAnalysis Documentation',
     [authors], 1)
]


# -- Options for Epub output ---------------------------------------------------

# Bibliographic Dublin Core info.
epub_title = u'MDAnalysis'
epub_author = authors
epub_publisher = 'Arizona State University, Tempe, Arizona, USA'
epub_copyright = u'2015, '+authors

# The language of the text. It defaults to the language option
# or en if the language is not set.
#epub_language = ''

# The scheme of the identifier. Typical schemes are ISBN or URL.
#epub_scheme = ''

# The unique identifier of the text. This can be a ISBN number
# or the project homepage.
#epub_identifier = ''

# A unique identification for the text.
#epub_uid = ''

# HTML files that should be inserted before the pages created by sphinx.
# The format is a list of tuples containing the path and title.
#epub_pre_files = []

# HTML files shat should be inserted after the pages created by sphinx.
# The format is a list of tuples containing the path and title.
#epub_post_files = []

# A list of files that should not be packed into the epub file.
#epub_exclude_files = []

# The depth of the table of contents in toc.ncx.
#epub_tocdepth = 3

# Allow duplicate toc entries.
#epub_tocdup = True


# Configuration for intersphinx: refer to the Python standard library
# and other packages used by MDAnalysis
intersphinx_mapping = {'https://docs.python.org/': None,
                       'https://docs.scipy.org/doc/numpy/': None,
                       'https://docs.scipy.org/doc/scipy/reference/': None,
                       'https://matplotlib.org': None,
                       'https://networkx.github.io/documentation/stable/': None,
                       'https://www.mdanalysis.org/GridDataFormats/': None,
                       'https://gsd.readthedocs.io/en/stable/': None,
                       'https://parmed.github.io/ParmEd/html/': None,
                       }
