from __future__ import annotations

import logging

from bs4 import BeautifulSoup, NavigableString, Tag

logger = logging.getLogger(__name__)


class ImageCaptionRule:
    name = "image_caption"

    def apply(self, soup: BeautifulSoup) -> tuple[BeautifulSoup, int]:
        chars_removed = 0
        caption_blocks_found = 0

        for div in soup.find_all("div"):
            if not div.find("img"):
                continue

            caption_ps = [
                child for child in div.children
                if isinstance(child, Tag) and child.name == "p"
            ]
            if not caption_ps:
                continue

            non_p_text = _non_p_text(div)
            if non_p_text:
                logger.debug(
                    "ImageCaptionRule: div has non-<p> text=%r — skipping (not a "
                    "pure image container)",
                    non_p_text[:60],
                )
                continue

            for p in caption_ps:
                caption_text = p.get_text()
                chars_removed += len(caption_text)
                logger.debug(
                    "ImageCaptionRule: removing caption %r (%d chars)",
                    caption_text[:80], len(caption_text),
                )
                p.decompose()

            caption_blocks_found += 1

        if caption_blocks_found:
            logger.info(
                "ImageCaptionRule: removed captions from %d image block(s), "
                "%d chars total",
                caption_blocks_found, chars_removed,
            )

        return soup, chars_removed


def _non_p_text(div: Tag) -> str:
    parts: list[str] = []
    for child in div.children:
        if isinstance(child, NavigableString):
            parts.append(str(child))
        elif isinstance(child, Tag) and child.name != "p":
            parts.append(child.get_text())
    return "".join(parts).strip()
