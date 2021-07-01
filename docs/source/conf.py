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

sys.path.insert(0, os.path.abspath(".."))
# sys.path.insert(0, os.path.abspath('../modnet'))

autodoc_mock_imports = [
    "numpy",
    "pandas",
    "sklearn",
    "tensorflow",
    "pymatgen",
    "matminer",
    "tqdm",
    "pytest",
    "joblib",
]

autodoc_member_order = "bysource"
autodoc_typehints = "description"
autoclass_content = "both"
default_role = "any"

# -- Project information -----------------------------------------------------

project = "modnet"
copyright = "2021, Pierre-Paul De Breuck, Matthew L. Evans"
author = "Pierre-Paul De Breuck, Matthew L. Evans"

# The full version, including alpha/beta/rc tags
from modnet import __version__

release = __version__
version = __version__


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "../../modnet/tests"]

# -- Run API doc -------------------------------------------------------------

# Force regenerates the API docs on sphinx builds


def run_apidoc(_):
    import subprocess
    import glob

    output_path = os.path.abspath(os.path.dirname(__file__))
    excludes = glob.glob(os.path.join(output_path, "../../modnet/tests"))
    module = os.path.join(output_path, "../../modnet")
    cmd_path = "sphinx-apidoc"
    command = [cmd_path, "-e", "-o", output_path, module, " ".join(excludes), "--force"]
    subprocess.check_call(command)


def setup(app):
    app.connect("builder-inited", run_apidoc)


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
import sphinx_rtd_theme

html_theme = "sphinx_rtd_theme"
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

intersphinx_mapping = {
    "python": ("http://docs.python.org/", None),
    "pandas": ("http://pandas.pydata.org/pandas-docs/dev", None),
    "pd": ("http://pandas.pydata.org/pandas-docs/dev", None),
}
