"""A ``video`` directive: embeds an MP4 that the page's own code wrote, looping and muted."""

from __future__ import annotations

import html
from pathlib import Path
from typing import ClassVar

from docutils import nodes
from docutils.parsers.rst import directives
from sphinx.application import Sphinx
from sphinx.util import logging
from sphinx.util.docutils import SphinxDirective

LIMIT = 2_000_000  # [bytes] per video, so pages stay light on a phone
logger = logging.getLogger(__name__)


class Video(SphinxDirective):
    """``video <path>`` (relative to the page) with an optional ``:caption:``."""

    required_arguments = 1
    option_spec: ClassVar = {"caption": directives.unchanged}

    def run(self) -> list[nodes.Node]:
        rel, path = self.env.relfn2path(self.arguments[0], self.env.docname)
        if not Path(path).is_file():
            logger.warning(f"video not found: {path}", location=self.get_location())
            return []
        size = Path(path).stat().st_size
        if size > LIMIT:
            logger.warning(f"video over 2 MB ({size} bytes): {path}", location=self.get_location())
        self.env.note_dependency(rel)
        dest = self.env.dlfiles.add_file(self.env.docname, rel)
        src = "../" * self.env.docname.count("/") + "_downloads/" + Path(dest).as_posix()
        caption = self.options.get("caption")
        figcaption = f"<figcaption>{html.escape(caption)}</figcaption>" if caption else ""
        markup = (
            '<figure class="video">'
            f'<video src="{src}" controls autoplay muted loop playsinline preload="metadata">'
            "</video>"
            f"{figcaption}</figure>"
        )
        return [nodes.raw("", markup, format="html")]


def setup(app: Sphinx) -> dict[str, bool]:
    app.add_directive("video", Video)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
