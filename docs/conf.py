"""Sphinx configuration."""

from importlib.metadata import version as _version

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
    "sphinxcontrib.mermaid",
]

myst_enable_extensions = ["amsmath", "colon_fence", "deflist", "dollarmath"]
myst_heading_anchors = 3

# Every code cell in the docs runs at each build, and an error fails the build.
nb_execution_mode = "force"
nb_execution_raise_on_error = True
nb_execution_timeout = 300

autosummary_generate = True
autodoc_typehints = "description"
autodoc_member_order = "bysource"
napoleon_google_docstring = False
napoleon_numpy_docstring = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}

html_theme = "pydata_sphinx_theme"
html_title = "virtualmodelcontrol"
exclude_patterns = ["_build", "jupyter_execute"]

html_theme_options = {
    "github_url": "https://github.com/vigno0405/VirtualModelControl",
    "icon_links": [
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/virtualmodelcontrol/",
            "icon": "fa-brands fa-python",
        }
    ],
    "navbar_align": "left",
    "header_links_before_dropdown": 6,
    "show_toc_level": 2,
    "navigation_with_keys": False,
    "footer_start": ["copyright"],
    "footer_end": [],
}
html_context = {"default_mode": "light"}
html_copy_source = False
html_show_sourcelink = False
templates_path = ["_templates"]
html_static_path = ["_static"]
html_css_files = ["custom.css"]
add_module_names = False
