"""Run first, hidden, on every executed page: SVG figures in the lab style, sized for the page."""

import matplotlib.pyplot as plt
from IPython import get_ipython

from virtualmodelcontrol import viz

viz.use_style(usetex=False)  # Computer Modern without needing LaTeX on the build machine
plt.rcParams["figure.figsize"] = (6.4, 4.4)  # [in]: 18 pt text stays readable on a phone
get_ipython().run_line_magic("config", "InlineBackend.figure_formats = ['svg']")
