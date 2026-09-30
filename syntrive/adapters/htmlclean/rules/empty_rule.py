from __future__ import annotations

from bs4 import BeautifulSoup

_REMOVABLE_TAGS = frozenset({
    "p", "div", "span", "blockquote", "li",
    "h1", "h2", "h3", "h4", "h5", "h6",
    "td", "th",
})


class EmptyElementRule:
    name = "empty_element"

    def apply(self, soup: BeautifulSoup) -> tuple[BeautifulSoup, int]:
        removed_chars = 0
        changed = True
        while changed:
            changed = False
            for tag in soup.find_all(_REMOVABLE_TAGS):
                if not tag.get_text(strip=True):
                    tag.decompose()
                    changed = True
                    break

        return soup, removed_chars
