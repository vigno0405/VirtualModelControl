"""House rules of the documentation: code fits the page, and no em dashes anywhere public."""

import re
from pathlib import Path

ROOT = Path(__file__).parents[1]
PAGES = sorted((ROOT / "docs").rglob("*.md"))
WIDTH = 76  # characters of code that fit the docs' column without scrolling


def visible_code(text):
    """Lines of the code blocks a reader sees (hidden cells excluded)."""
    for block in re.findall(r"```(\{code-cell\} python|python)\n(.*?)```", text, flags=re.S):
        body = block[1]
        if re.match(r":tags: \[.*remove-(cell|input).*\]", body):
            continue
        yield from (line for line in body.splitlines() if not line.startswith(":tags:"))


def test_code_lines_fit_the_page():
    long = [
        f"{page.relative_to(ROOT)}: {line}"
        for page in PAGES
        if "_build" not in page.parts
        for line in visible_code(page.read_text())
        if len(line) > WIDTH
    ]
    assert not long, "\n".join(long)


def test_no_em_dashes_in_public_text():
    files = [
        *PAGES,
        *ROOT.glob("*.md"),
        *(ROOT / "src").rglob("*.py"),
        *(ROOT / "docs").rglob("*.py"),
    ]
    hits = [
        str(f.relative_to(ROOT))
        for f in files
        if "_build" not in f.parts and "\u2014" in f.read_text()
    ]
    assert not hits, hits


FIGURE_TEXT = re.compile(r"(set_[xyz]?label|set_title|suptitle|\.text|annotate)\(|\b(label|title)=")


def figure_code():
    """Code that draws figures: the pages' cells, the docs' scripts and schematics, and viz."""
    for page in PAGES:
        if "_build" not in page.parts:
            for block in re.findall(r"```\{code-cell\} python\n(.*?)```", page.read_text(), re.S):
                yield page, block.splitlines()
    for f in [
        *(ROOT / "docs").rglob("*.py"),
        *(ROOT / "src" / "virtualmodelcontrol" / "viz").rglob("*.py"),
    ]:
        if "_build" not in f.parts:
            yield f, f.read_text().splitlines()


def test_figure_text_renders_in_the_figure_font():
    """Without LaTeX, figures use Computer Modern (cmr10), which lacks ·, °, Δ and the like:
    outside $...$ math, figure text must be ASCII."""
    hits = []
    for f, lines in figure_code():
        for line in lines:
            code = line.split(" # ")[0]
            if FIGURE_TEXT.search(code) and re.search(
                r"[^\x00-\x7f]", re.sub(r"\$[^$]*\$", "", code)
            ):
                hits.append(f"{f.relative_to(ROOT)}: {line.strip()}")
    assert not hits, "\n".join(hits)
