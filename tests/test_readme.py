"""Every python block in README.md runs, in order, as a reader would paste them."""

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

README = Path(__file__).parents[1] / "README.md"


def test_readme_python_blocks_run():
    blocks = re.findall(r"```python\n(.*?)```", README.read_text(encoding="utf-8"), flags=re.DOTALL)
    assert blocks, "the README should show some python"
    namespace: dict = {}
    for block in blocks:
        exec(compile(block, str(README), "exec"), namespace)
