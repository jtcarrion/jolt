"""Sphinx configuration for the JOLT documentation."""

import jolt

project = "JOLT"
author = "Jackson Carrion"
copyright = "2026, Jackson Carrion"
release = jolt.__version__
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinxarg.ext",
    "sphinx_copybutton",
]
napoleon_numpy_docstring = True
napoleon_google_docstring = False

html_theme = "pydata_sphinx_theme"
html_title = "JOLT"
html_theme_options = {"github_url": "https://github.com/jtcarrion/jolt"}
exclude_patterns = ["_build"]
