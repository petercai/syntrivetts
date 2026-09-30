from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import NamedTuple

logger = logging.getLogger(__name__)

_SML_TOKEN_RE = re.compile(r'‡[^‡]+‡')

AUDIT_SECTION_HEADING = "## File Size & Character Count (audit)"


class AuditTocMeta(NamedTuple):
    sequence_number: str = ""
    chapter_number: str = ""
    volume_name: str = ""
    volume_number: str = ""


def file_size_bytes(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0


def file_char_count(path: Path, *, strip_html: bool = False, strip_sml: bool = False) -> int:
    if not path.exists():
        return 0
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return 0
    if strip_html:
        from bs4 import BeautifulSoup
        text = BeautifulSoup(text, "html.parser").get_text()
    if strip_sml:
        text = _SML_TOKEN_RE.sub("", text)
    return len(text)


def build_audit_table_lines(
    rows: list[tuple[str, str, Path]],
    *,
    strip_html: bool = False,
    strip_sml: bool = False,
    toc_meta: list[AuditTocMeta] | None = None,
) -> list[str]:
    if not rows:
        return []

    if toc_meta is not None and len(toc_meta) != len(rows):
        logger.warning(
            "build_audit_table_lines: toc_meta length %d != rows length %d; "
            "omitting all TOC-heading columns",
            len(toc_meta), len(rows),
        )
        toc_meta = None

    show_volume = toc_meta is not None and any(m.volume_name for m in toc_meta)
    show_volume_number = toc_meta is not None and any(m.volume_number for m in toc_meta)

    header_cols: list[str] = []
    if show_volume:
        header_cols.append("volume")
    header_cols.extend(["chapter", "file"])
    if toc_meta is not None:
        header_cols.extend(["sequence_number", "chapter_number"])
    if show_volume_number:
        header_cols.append("volume_number")
    header_cols.extend(["size_bytes", "char_count"])

    lines = [
        "",
        "---",
        "",
        AUDIT_SECTION_HEADING,
        "",
        "| " + " | ".join(header_cols) + " |",
        "|" + "---|" * len(header_cols),
    ]
    for idx, (title, link_target, out_path) in enumerate(rows):
        safe_title = title.replace("[", "\\[").replace("]", "\\]")
        size = file_size_bytes(out_path)
        chars = file_char_count(out_path, strip_html=strip_html, strip_sml=strip_sml)
        meta = toc_meta[idx] if toc_meta is not None else None

        cells: list[str] = []
        if show_volume:
            cells.append(str(meta.volume_name))
        cells.extend([f"[{safe_title}]({link_target})", link_target])
        if meta is not None:
            cells.extend([str(meta.sequence_number), str(meta.chapter_number)])
        if show_volume_number:
            cells.append(str(meta.volume_number))
        cells.extend([str(size), str(chars)])
        lines.append("| " + " | ".join(cells) + " |")
    return lines
