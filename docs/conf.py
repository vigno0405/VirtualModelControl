"""Sphinx configuration."""

import os
import sys
from importlib.metadata import version as _version
from pathlib import Path

DOCS = Path(__file__).parent
sys.path.insert(0, str(DOCS / "_ext"))
# The executed pages import the docs' own helpers (schematics, page setup) from here.
os.environ["PYTHONPATH"] = os.pathsep.join(
    [str(DOCS), *filter(None, [os.environ.get("PYTHONPATH")])]
)

project = "virtualmodelcontrol"
author = "Lorenzo Vignoli"
copyright = "2026, Lorenzo Vignoli"
release = _version("virtualmodelcontrol")
version = ".".join(release.split(".")[:2])

extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx_copybutton",
    "sphinx_design",
    "video",
]

myst_enable_extensions = ["amsmath", "colon_fence", "deflist", "dollarmath", "tasklist"]
myst_heading_anchors = 3

# Every code cell in the docs runs at each build, and an error fails the build.
nb_execution_mode = "force"
nb_execution_raise_on_error = True
nb_execution_timeout = 300
nb_render_markdown_format = "myst"  # tables computed by a page render as tables

autosummary_generate = True
autodoc_typehints = "description"
autodoc_member_order = "bysource"
autodoc_type_aliases = {"ArrayLike": "numpy.typing.ArrayLike"}
napoleon_google_docstring = False
napoleon_numpy_docstring = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}

html_theme = "furo"
html_title = "virtualmodelcontrol"
exclude_patterns = ["_build", "jupyter_execute", "schematics", "_ext"]

# Colours from the figures' palette: navy for links, red for accents.
_brand = {"color-brand-primary": "#3C5488", "color-brand-content": "#3C5488"}
html_theme_options = {
    "light_css_variables": _brand,
    "dark_css_variables": {"color-brand-primary": "#8491B4", "color-brand-content": "#8491B4"},
    "source_repository": "https://github.com/vigno0405/VirtualModelControl",
    "source_branch": "main",
    "source_directory": "docs/",
    "top_of_page_buttons": [],
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/vigno0405/VirtualModelControl",
            "html": "GitHub",
            "class": "",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/virtualmodelcontrol/",
            "html": "PyPI",
            "class": "",
        },
    ],
}
html_copy_source = False
html_show_sourcelink = False
templates_path = ["_templates"]
html_static_path = ["_static"]
html_css_files = ["custom.css"]
add_module_names = False
