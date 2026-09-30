from __future__ import annotations

import logging
import re

from bs4 import BeautifulSoup, Tag

logger = logging.getLogger(__name__)

_ANNOTATION_RE = re.compile(
    r'^[\s\d'
    r'①-⑳'
    r'⑴-⒇'
    r'⒈-⒛'
    r'⓫-⓿'
    r'*†‡§¶#]+$'
)

_MAX_ANNOTATION_LEN = 5
_MAX_LINKED_ANNOTATION_LEN = 10

_CFS_ID_RE = re.compile(r'^cfs_\d+$')
_BRACKET_NUM_TEXT_RE = re.compile(r'^\[\d+\]\s*$')


class AnnotationRemovalRule:
    name = "annotation_removal"

    def apply(self, soup: BeautifulSoup) -> tuple[BeautifulSoup, int]:
        removed_chars = 0
        to_remove: list[Tag] = []

        for sup in soup.find_all("sup"):
            text = sup.get_text(strip=True)
            has_link = bool(sup.find("a"))
            if _is_sup_annotation(text, has_link):
                removed_chars += len(sup.get_text())
                to_remove.append(sup)
                logger.debug(
                    "AnnotationRemovalRule: removing <sup> text=%r has_link=%s",
                    text, has_link,
                )

        for span in soup.find_all("span"):
            if _is_calibre_span_annotation(span):
                removed_chars += len(span.get_text())
                to_remove.append(span)
                logger.debug(
                    "AnnotationRemovalRule: removing calibre span id=%r text=%r",
                    span.get("id"), span.get_text(),
                )

        for element in to_remove:
            element.decompose()

        if to_remove:
            logger.info(
                "AnnotationRemovalRule: removed %d annotation marker(s), %d chars",
                len(to_remove), removed_chars,
            )

        return soup, removed_chars


def _is_sup_annotation(text: str, has_link: bool) -> bool:
    stripped = text.strip()
    if not stripped:
        return False

    if len(stripped) <= _MAX_ANNOTATION_LEN and _ANNOTATION_RE.match(stripped):
        return True

    if has_link and len(stripped) <= _MAX_LINKED_ANNOTATION_LEN:
        return True

    return False


def _is_calibre_span_annotation(span: Tag) -> bool:
    span_id = str(span.get("id", ""))
    if not _CFS_ID_RE.match(span_id):
        return False
    return bool(_BRACKET_NUM_TEXT_RE.match(span.get_text()))
