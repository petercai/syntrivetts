from __future__ import annotations

import logging
import re

from bs4 import BeautifulSoup, NavigableString, Tag

logger = logging.getLogger(__name__)

_PAREN_RE = re.compile(
    r'[（(]\s*[A-Za-z][^（(）)]{0,100}?[）)]'
)

_SKIP_TAGS = frozenset({"script", "style", "head", "meta", "title", "code", "pre"})

_ASCII_THRESHOLD = 0.70


class InlineEnglishRule:
    name = "inline_english"

    def __init__(self, target_language: str = "en") -> None:
        self._enabled = not target_language.lower().startswith("en")

    def apply(self, soup: BeautifulSoup) -> tuple[BeautifulSoup, int]:
        if not self._enabled:
            return soup, 0

        removed_chars = 0

        for node in soup.find_all(string=True):
            if isinstance(node.parent, Tag) and node.parent.name in _SKIP_TAGS:
                continue

            original = str(node)
            cleaned, n_removed = _strip_english_parens(original)

            if n_removed > 0:
                removed_chars += n_removed
                node.replace_with(NavigableString(cleaned))
                logger.debug(
                    "InlineEnglishRule: stripped %d chars from %r -> %r",
                    n_removed, original[:60], cleaned[:60],
                )

        if removed_chars:
            logger.info(
                "InlineEnglishRule: removed %d chars of inline English parentheticals",
                removed_chars,
            )

        return soup, removed_chars


def _strip_english_parens(text: str) -> tuple[str, int]:
    result = _PAREN_RE.sub(_maybe_remove, text)
    return result, len(text) - len(result)


def _maybe_remove(m: re.Match) -> str:
    span = m.group(0)
    return "" if _is_english_paren(span) else span


def _is_english_paren(text: str) -> bool:
    inner = text[1:-1].strip()
    if not inner:
        return False

    if _has_cjk(inner):
        return False

    non_space = [c for c in inner if not c.isspace()]
    if not non_space:
        return False

    ascii_ratio = sum(1 for c in non_space if c.isascii()) / len(non_space)
    return ascii_ratio >= _ASCII_THRESHOLD


def _has_cjk(text: str) -> bool:
    return any('一' <= c <= '鿿' for c in text)
