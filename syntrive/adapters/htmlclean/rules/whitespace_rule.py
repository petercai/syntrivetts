from __future__ import annotations

import re

from bs4 import BeautifulSoup, NavigableString


class WhitespaceRule:
    name = "whitespace"

    _WS_RE = re.compile(r"[ \t\xa0]{2,}")

    def apply(self, soup: BeautifulSoup) -> tuple[BeautifulSoup, int]:
        removed_chars = 0

        for node in soup.find_all(string=True):
            original = node.string
            if original is None:
                continue
            cleaned = self._WS_RE.sub(" ", original)
            if cleaned != original:
                removed_chars += len(original) - len(cleaned)
                node.replace_with(NavigableString(cleaned))

        return soup, removed_chars
