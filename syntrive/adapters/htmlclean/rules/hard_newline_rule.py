from __future__ import annotations

import logging
import re

from bs4 import BeautifulSoup, NavigableString

logger = logging.getLogger(__name__)

_INTERIOR_NEWLINE_RE = re.compile(r"[^\S\n]*\r?\n\s*")
_BLOCK_DESCENDANTS = ["p", "div", "ul", "ol", "li", "table", "blockquote",
                      "h1", "h2", "h3", "h4", "h5", "h6"]


class HardNewlineToBrRule:
    name = "hard_newline_to_br"

    def apply(self, soup: BeautifulSoup) -> tuple[BeautifulSoup, int]:
        removed_chars = 0
        converted_paragraphs = 0

        for para in soup.find_all("p"):
            if para.find_parent("pre") is not None or para.find(_BLOCK_DESCENDANTS) is not None:
                continue
            children = list(para.children)
            para_changed = False
            for index, node in enumerate(children):
                if not isinstance(node, NavigableString):
                    continue
                text = str(node)
                core = text.strip()
                if "\n" not in core:
                    continue
                pieces = [p for p in _INTERIOR_NEWLINE_RE.split(core) if p]
                if len(pieces) < 2:
                    continue

                lead = " " if text[: len(text) - len(text.lstrip())] and index > 0 else ""
                trail = " " if text[len(text.rstrip()):] and index < len(children) - 1 else ""

                new_nodes: list = [NavigableString(lead + pieces[0])]
                for piece in pieces[1:]:
                    new_nodes.append(soup.new_tag("br"))
                    new_nodes.append(NavigableString(" " + piece))
                new_nodes[-1] = NavigableString(str(new_nodes[-1]) + trail)

                for new in new_nodes:
                    node.insert_before(new)
                node.extract()
                removed_chars += max(0, len(text) - sum(len(str(n)) for n in new_nodes if isinstance(n, NavigableString)))
                para_changed = True

            if para_changed:
                converted_paragraphs += 1

        if converted_paragraphs:
            logger.debug(
                "HardNewlineToBrRule: converted %d paragraph(s), %d whitespace char(s) removed",
                converted_paragraphs, removed_chars,
            )
        return soup, removed_chars
