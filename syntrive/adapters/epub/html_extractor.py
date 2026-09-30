from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import unquote

logger = logging.getLogger(__name__)

_SPLIT_ATTR = "data-syntrive-anchor"


@dataclass(frozen=True)
class ChapterHtmlNode:
    toc_id: str
    title: str
    href: str
    anchor: Optional[str]
    raw_html: str
    order: int


class EpubHtmlExtractor:
    def __init__(self, epub_path: Path) -> None:
        self._epub_path = epub_path

    def extract(self) -> list[ChapterHtmlNode]:
        try:
            import ebooklib
            from ebooklib import epub
        except ImportError:
            raise RuntimeError("ebooklib is required: pip install EbookLib") from None

        logger.info("EpubHtmlExtractor: reading %s", self._epub_path)
        book = epub.read_epub(str(self._epub_path))

        toc_entries: list[tuple[str, str]] = _flatten_toc(book.toc)
        if not toc_entries:
            logger.warning(
                "EpubHtmlExtractor: no TOC entries found in %s — raw toc has %d items",
                self._epub_path, len(book.toc),
            )
            for i, item in enumerate(book.toc[:10]):
                logger.debug(
                    "EpubHtmlExtractor: toc[%d] type=%s repr=%r",
                    i, type(item).__name__, str(item)[:120],
                )
            return []
        logger.info("EpubHtmlExtractor: found %d TOC entries", len(toc_entries))
        for idx, (title, href) in enumerate(toc_entries):
            logger.debug("EpubHtmlExtractor: toc[%d] title=%r href=%r", idx, title, href)

        file_groups: dict[str, list[tuple[int, str, str, Optional[str]]]] = {}
        for idx, (title, href) in enumerate(toc_entries):
            file_href, anchor = _split_href(href)
            file_groups.setdefault(file_href, []).append((idx, title, href, anchor))

        try:
            import ebooklib
            epub_item_names = [
                (item.get_name() or item.file_name or "")
                for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT)
            ]
            logger.debug(
                "EpubHtmlExtractor: EPUB document items (%d): %s",
                len(epub_item_names), epub_item_names,
            )
        except Exception:
            pass

        chapters: list[ChapterHtmlNode] = []
        for file_href, entries in file_groups.items():
            html_content = _get_item_content(book, file_href)
            if html_content is None:
                logger.warning(
                    "EpubHtmlExtractor: file not in EPUB: %r "
                    "(check that href in TOC matches an EPUB document item name)",
                    file_href,
                )
                continue

            chapters.extend(_extract_entries(file_href, entries, html_content))

        result = sorted(chapters, key=lambda c: c.order)
        logger.info("EpubHtmlExtractor: extracted %d chapters total", len(result))
        return result


def _flatten_toc(toc) -> list[tuple[str, str]]:
    result: list[tuple[str, str]] = []
    for item in toc:
        if isinstance(item, tuple) and len(item) >= 2:
            head = item[0]
            children = item[1]
            if hasattr(head, "href") and head.href:
                result.append((head.title or "", head.href))
            if isinstance(children, (list, tuple)):
                result.extend(_flatten_toc(children))
        elif hasattr(item, "href") and item.href:
            result.append((item.title or "", item.href))
    return result


def _split_href(href: str) -> tuple[str, Optional[str]]:
    if "#" in href:
        file_part, fragment = href.split("#", 1)
        return unquote(file_part), unquote(fragment) or None
    return unquote(href), None


def _get_item_content(book, file_href: str) -> Optional[str]:
    try:
        import ebooklib
    except ImportError:
        return None

    file_name_only = Path(file_href).name
    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        item_name = item.get_name() or item.file_name or ""
        if item_name == file_href or Path(item_name).name == file_name_only:
            return item.get_content().decode("utf-8", errors="replace")
    return None


def _extract_entries(
    file_href: str,
    entries: list[tuple[int, str, str, Optional[str]]],
    html_content: str,
) -> list[ChapterHtmlNode]:
    if len(entries) == 1:
        idx, title, _, anchor = entries[0]
        return [ChapterHtmlNode(
            toc_id=f"ch_{idx + 1:04d}",
            title=title,
            href=file_href,
            anchor=anchor,
            raw_html=html_content,
            order=idx,
        )]

    anchor_ids = [e[3] for e in entries if e[3]]
    slices = _split_html_at_anchors(html_content, anchor_ids) if anchor_ids else {}

    nodes = []
    for idx, title, _, anchor in entries:
        slice_html = slices.get(anchor, html_content) if anchor else html_content
        nodes.append(ChapterHtmlNode(
            toc_id=f"ch_{idx + 1:04d}",
            title=title,
            href=file_href,
            anchor=anchor,
            raw_html=slice_html,
            order=idx,
        ))
    return nodes


def _split_html_at_anchors(html: str, anchor_ids: list[str]) -> dict[str, str]:
    from bs4 import BeautifulSoup

    if not anchor_ids:
        return {}

    soup = BeautifulSoup(html, "html.parser")

    found_anchors = []
    for aid in anchor_ids:
        elem = soup.find(id=aid)
        if elem and hasattr(elem, "attrs"):
            elem.attrs[_SPLIT_ATTR] = aid
            found_anchors.append(aid)
        else:
            logger.debug("EpubHtmlExtractor: anchor '%s' not found in HTML", aid)

    if not found_anchors:
        return {}

    html_str = str(soup)
    slices: dict[str, str] = {}

    for i, aid in enumerate(found_anchors):
        marker = f'{_SPLIT_ATTR}="{aid}"'
        start_marker_pos = html_str.find(marker)
        if start_marker_pos == -1:
            continue

        tag_start = html_str.rfind("<", 0, start_marker_pos)
        if tag_start == -1:
            tag_start = start_marker_pos

        end_pos = len(html_str)
        for next_aid in found_anchors[i + 1:]:
            next_marker = f'{_SPLIT_ATTR}="{next_aid}"'
            np = html_str.find(next_marker)
            if np != -1:
                next_tag_start = html_str.rfind("<", 0, np)
                end_pos = next_tag_start if next_tag_start != -1 else np
                break

        slices[aid] = html_str[tag_start:end_pos]

    return slices
