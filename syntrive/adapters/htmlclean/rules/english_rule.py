from __future__ import annotations

import logging

from bs4 import BeautifulSoup, NavigableString, Tag

logger = logging.getLogger(__name__)

ASCII_THRESHOLD = 0.70

_BLOCK_TAGS = frozenset({"p", "div", "blockquote", "section", "article"})

_CONTAINER_CHILD_TAGS = frozenset({"p", "div", "ul", "ol", "blockquote", "section", "article"})


class EnglishBlockRule:
    name = "english_block"

    def __init__(self, target_language: str = "en") -> None:
        self._enabled = not target_language.lower().startswith("en")

    def apply(self, soup: BeautifulSoup) -> tuple[BeautifulSoup, int]:
        if not self._enabled:
            return soup, 0

        removed_chars = 0
        to_remove: list[Tag] = []

        for tag in soup.find_all(_BLOCK_TAGS):
            text = tag.get_text()
            if not text.strip():
                continue
            if any(isinstance(c, Tag) and c.name in _CONTAINER_CHILD_TAGS for c in tag.children):
                continue
            ratio = _ascii_ratio(text)
            if ratio > ASCII_THRESHOLD:
                removed_chars += len(text)
                to_remove.append(tag)
                logger.debug(
                    "EnglishBlockRule: removing <%s> ascii_ratio=%.2f len=%d",
                    tag.name, ratio, len(text),
                )

        for tag in to_remove:
            tag.decompose()

        if removed_chars:
            logger.info(
                "EnglishBlockRule: removed %d chars from %d block(s)",
                removed_chars, len(to_remove),
            )
        return soup, removed_chars


def _ascii_ratio(text: str) -> float:
    if not text:
        return 0.0
    ascii_count = sum(1 for c in text if c.isascii() and c.isprintable())
    return ascii_count / len(text)
