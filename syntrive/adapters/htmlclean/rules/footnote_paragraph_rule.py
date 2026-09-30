from __future__ import annotations

import logging
import re
from typing import Optional

from bs4 import BeautifulSoup, NavigableString, Tag

logger = logging.getLogger(__name__)

_FILEPOS_ID_RE = re.compile(r'^filepos\d+$')
_FOOTNOTE_HREF_RE = re.compile(r'^[^#]+\.html#filepos\d+$')


class FootnoteParagraphRemovalRule:
    name = "footnote_paragraph_removal"

    def apply(self, soup: BeautifulSoup) -> tuple[BeautifulSoup, int]:
        removed_chars = 0
        to_remove: list[Tag] = []

        for p in soup.find_all("p"):
            if _is_footnote_paragraph(p):
                text = p.get_text()
                removed_chars += len(text)
                to_remove.append(p)
                logger.debug(
                    "FootnoteParagraphRemovalRule: removing p id=%r text=%r",
                    p.get("id"), text[:60],
                )

        for element in to_remove:
            element.decompose()

        if to_remove:
            logger.info(
                "FootnoteParagraphRemovalRule: removed %d footnote paragraph(s), "
                "%d chars",
                len(to_remove), removed_chars,
            )

        return soup, removed_chars


def _is_footnote_paragraph(p: Tag) -> bool:
    p_id = str(p.get("id", ""))
    if not _FILEPOS_ID_RE.match(p_id):
        return False

    first_child = _first_element_child(p)
    if first_child is None or first_child.name != "a":
        return False

    href = str(first_child.get("href", ""))
    return bool(_FOOTNOTE_HREF_RE.match(href))


def _first_element_child(tag: Tag) -> Optional[Tag]:
    for child in tag.children:
        if isinstance(child, Tag):
            return child
        if isinstance(child, NavigableString) and child.strip():
            return None
    return None
