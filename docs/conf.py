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
    "sphinx_immaterial",
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

html_theme = "sphinx_immaterial"
html_title = "virtualmodelcontrol"
html_logo = "_static/logo-light.svg"  # the header is navy in both colour schemes
html_favicon = "_static/favicon.svg"
# Tooltips of API objects without their synopsis, which the theme joins with an em dash.
object_description_options = [("py:.*", {"generate_synopses": None})]
exclude_patterns = ["_build", "jupyter_execute", "schematics", "_ext"]

# The layout of Material for MkDocs: the sections of the site as tabs under the header, the
# pages of the current section in the left sidebar, the outline of the page on the right.
html_theme_options = {
    "repo_url": "https://github.com/vigno0405/VirtualModelControl",
    "repo_name": "VirtualModelControl",
    "icon": {"repo": "fontawesome/brands/github"},
    "font": {"text": "Inter", "code": "JetBrains Mono"},
    "features": [
        "navigation.tabs",
        "navigation.tabs.sticky",
        "navigation.top",
        "search.highlight",
        "toc.follow",
        "content.code.copy",
    ],
    "palette": [
        {
            "media": "(prefers-color-scheme: light)",
            "scheme": "default",
            "primary": "custom",  # colours in _static/custom.css
            "accent": "custom",
            "toggle": {"icon": "material/weather-night", "name": "Dark mode"},
        },
        {
            "media": "(prefers-color-scheme: dark)",
            "scheme": "slate",
            "primary": "custom",
            "accent": "custom",
            "toggle": {"icon": "material/weather-sunny", "name": "Light mode"},
        },
    ],
    "globaltoc_collapse": True,
    "toc_title_is_page_title": True,
    "social": [
        {
            "icon": "fontawesome/brands/github",
            "link": "https://github.com/vigno0405/VirtualModelControl",
        },
        {
            "icon": "fontawesome/brands/python",
            "link": "https://pypi.org/project/virtualmodelcontrol/",
        },
    ],
}
html_copy_source = False
html_show_sourcelink = False
templates_path = ["_templates"]
html_static_path = ["_static"]
html_css_files = ["custom.css"]
add_module_names = False
