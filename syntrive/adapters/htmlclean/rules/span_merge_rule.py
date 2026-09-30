from __future__ import annotations

import logging

from bs4 import BeautifulSoup, NavigableString, Tag

logger = logging.getLogger(__name__)

_BLOCK_TAGS = frozenset({
    "p", "li", "dt", "dd", "blockquote",
    "h1", "h2", "h3", "h4", "h5", "h6",
})


class SpanMergeRule:
    name = "span_merge"

    def apply(self, soup: BeautifulSoup) -> tuple[BeautifulSoup, int]:
        merged_blocks = 0

        for block in soup.find_all(_BLOCK_TAGS):
            child_tags = [c for c in block.children if isinstance(c, Tag)]

            if any(c.name != "span" for c in child_tags):
                continue

            span_count = len(child_tags)

            if span_count == 0:
                continue

            has_text_nodes = any(
                isinstance(c, NavigableString) and c.strip()
                for c in block.children
            )

            if span_count < 2 and not has_text_nodes:
                continue

            merged_text = block.get_text()
            if not merged_text.strip():
                continue

            block.clear()
            block.append(NavigableString(merged_text))
            merged_blocks += 1

        if merged_blocks:
            logger.debug(
                "SpanMergeRule: merged span fragments in %d block(s)", merged_blocks
            )

        return soup, 0
