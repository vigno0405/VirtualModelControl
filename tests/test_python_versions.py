"""Everything written in Python compiles on the oldest Python the library supports.

A syntax of a newer Python is the usual surprise on a student's machine (an f-string that reuses
its own quotes is legal on 3.12 and a SyntaxError on 3.10). The tests import only some modules and
the docs build runs only some snippets, so this reads them all: every ``.py`` file of the
repository and every Python code block of the Markdown pages. It fails on the 3.10 job of the CI,
which is where it matters; a newer interpreter at least catches the warnings that are going to be
errors.
"""

import re
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SKIP = {
    ".git",
    ".venv",
    "venv",
    "_build",
    "build",
    "dist",
    "node_modules",
    ".claude",
    "__pycache__",
}
FENCE = re.compile(r"^(`{3,})\s*(\{code-cell\}\s*python|python|py)\s*$(.*?)^\1\s*$", re.S | re.M)


def sources():
    for path in sorted(ROOT.rglob("*")):
        if SKIP & set(path.relative_to(ROOT).parts):
            continue
        if path.suffix == ".py":
            yield str(path.relative_to(ROOT)), path.read_text(encoding="utf-8")
        elif path.suffix == ".md":
            for i, block in enumerate(FENCE.finditer(path.read_text(encoding="utf-8"))):
                body = block.group(3)
                if block.group(2).startswith(
                    "{"
                ):  # a MyST cell: its options (:tags: ...) are not code
                    body = "\n".join(line for line in body.splitlines() if not line.startswith(":"))
                yield f"{path.relative_to(ROOT)} (block {i + 1})", body


def test_every_python_file_and_docs_snippet_compiles_on_the_oldest_supported_python():
    problems, count = [], 0
    with warnings.catch_warnings():
        warnings.simplefilter("error", SyntaxWarning)
        warnings.simplefilter("error", DeprecationWarning)
        for name, text in sources():
            count += 1
            try:
                compile(text, name, "exec")
            except (SyntaxError, SyntaxWarning, DeprecationWarning) as exc:
                problems.append(f"{name}: {type(exc).__name__}: {exc}")
    assert count > 400, "the repository was not found where this test expects it"
    assert not problems, "\n".join(problems)
