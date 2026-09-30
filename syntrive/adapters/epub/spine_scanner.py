from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import unquote

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SpineEntry:
    href: str
    spine_pos: int
    has_toc_ref: bool
    raw_html: str = field(default="", compare=False)


class SpineOrderScanner:
    def __init__(self, epub_path: Path) -> None:
        self._epub_path = epub_path

    def scan(self, toc_hrefs: set[str] | None = None) -> list[SpineEntry]:
        try:
            from ebooklib import epub as eblib
        except ImportError:
            raise RuntimeError("ebooklib is required: pip install EbookLib") from None

        logger.info("SpineOrderScanner: reading spine from %s", self._epub_path)
        book = eblib.read_epub(str(self._epub_path), {"ignore_ncx": True})

        toc_coverage = toc_hrefs or set()
        entries: list[SpineEntry] = []

        for pos, (item_id, _) in enumerate(book.spine):
            item = book.get_item_with_id(item_id)
            if item is None:
                logger.debug(
                    "SpineOrderScanner: spine item id=%r not found in manifest", item_id
                )
                continue

            raw_href = item.get_name() or item.file_name or ""
            file_href = unquote(raw_href)

            raw_html = _load_item_html(item, file_href)

            has_ref = _is_covered(file_href, toc_coverage)
            entries.append(SpineEntry(
                href=file_href,
                spine_pos=pos,
                has_toc_ref=has_ref,
                raw_html=raw_html,
            ))
            logger.debug(
                "SpineOrderScanner: pos=%d href=%r toc_ref=%s html_len=%d",
                pos, file_href, has_ref, len(raw_html),
            )

        non_toc_count = sum(1 for e in entries if not e.has_toc_ref)
        logger.info(
            "SpineOrderScanner: %d spine items, %d with TOC ref, %d without TOC ref",
            len(entries),
            len(entries) - non_toc_count,
            non_toc_count,
        )
        return entries


def _is_covered(file_href: str, toc_hrefs: set[str]) -> bool:
    if file_href in toc_hrefs:
        return True
    basename = Path(file_href).name
    return any(Path(th).name == basename for th in toc_hrefs)


def _load_item_html(item, file_href: str) -> str:
    try:
        content_bytes = item.get_content()
        if not content_bytes:
            return ""
        return content_bytes.decode("utf-8", errors="replace")
    except Exception as exc:
        logger.warning(
            "SpineOrderScanner: could not load HTML content for %r: %s", file_href, exc
        )
        return ""
