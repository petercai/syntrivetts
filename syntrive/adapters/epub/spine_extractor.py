from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SpineRawEntry:
    href: str
    spine_pos: int
    has_toc_ref: bool
    raw_path: Path


class SpineExtractor:
    def __init__(self, epub_path: Path) -> None:
        self._epub_path = epub_path

    def extract(
        self,
        raw_dir: Path,
        toc_hrefs: set[str] | None = None,
    ) -> list[SpineRawEntry]:
        try:
            from ebooklib import epub as eblib
        except ImportError:
            raise RuntimeError("ebooklib is required: pip install EbookLib") from None

        logger.info("SpineExtractor: reading %s", self._epub_path)
        book = eblib.read_epub(str(self._epub_path), {"ignore_ncx": True})

        coverage = toc_hrefs or set()
        entries: list[SpineRawEntry] = []

        for pos, (item_id, _) in enumerate(book.spine):
            item = book.get_item_with_id(item_id)
            if item is None:
                logger.debug("SpineExtractor: spine id=%r not in manifest — skipped", item_id)
                continue

            raw_href = unquote(item.get_name() or item.file_name or "")
            if not raw_href:
                logger.debug("SpineExtractor: spine pos=%d has empty href — skipped", pos)
                continue

            basename = Path(raw_href).name
            if not basename:
                continue

            html = _decode_item_html(item, raw_href)
            out_path = raw_dir / basename
            try:
                out_path.write_text(html, encoding="utf-8")
            except Exception as exc:
                logger.warning(
                    "SpineExtractor: could not write raw/%s: %s", basename, exc
                )
                continue

            has_ref = _is_toc_covered(raw_href, coverage)
            entries.append(SpineRawEntry(
                href=raw_href,
                spine_pos=pos,
                has_toc_ref=has_ref,
                raw_path=out_path,
            ))
            logger.debug(
                "SpineExtractor: pos=%d href=%r toc_ref=%s → raw/%s (%d bytes)",
                pos, raw_href, has_ref, basename, len(html),
            )

        non_toc_count = sum(1 for e in entries if not e.has_toc_ref)
        logger.info(
            "SpineExtractor: %d spine files written to raw/ (%d with TOC ref, %d without)",
            len(entries),
            len(entries) - non_toc_count,
            non_toc_count,
        )
        return entries


def _is_toc_covered(href: str, toc_hrefs: set[str]) -> bool:
    if href in toc_hrefs:
        return True
    name = Path(href).name
    return any(Path(th).name == name for th in toc_hrefs)


def _decode_item_html(item, href: str) -> str:
    try:
        data = item.get_content()
        return data.decode("utf-8", errors="replace") if data else ""
    except Exception as exc:
        logger.warning(
            "SpineExtractor: could not read content for %r: %s", href, exc
        )
        return ""
