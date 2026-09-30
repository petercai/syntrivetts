from __future__ import annotations

import logging
from pathlib import Path

from syntrive.adapters.html.para_id_mapper import ParaIdMap

logger = logging.getLogger(__name__)

_PARA_MAP_EXT = ".para_map.yaml"
_PARA_MAP_BACKUP_PREFIX = ".bk"


class HtmlArtifactWriter:
    def __init__(self, transcript_html_dir: Path) -> None:
        self._root = transcript_html_dir

    def write_raw(self, chapter_id: str, raw_html: str) -> str:
        raw_path = self._root / "raw" / f"{chapter_id}.html"
        raw_path.write_text(raw_html, encoding="utf-8")
        logger.debug("HtmlArtifactWriter: wrote raw/%s.html", chapter_id)
        return str(raw_path)

    def write_cleaned(
        self,
        chapter_id: str,
        cleaned_html: str,
        para_map: ParaIdMap,
    ) -> dict[str, str]:
        cleaned_path = self._root / "cleaned" / f"{chapter_id}.html"
        map_path = self._root / "manifest" / f"{chapter_id}{_PARA_MAP_EXT}"

        cleaned_path.write_text(cleaned_html, encoding="utf-8")
        _backup_if_exists(map_path)
        map_path.write_text(para_map.to_yaml(), encoding="utf-8")

        logger.debug("HtmlArtifactWriter: wrote cleaned/manifest for %s", chapter_id)
        return {
            "cleaned_html": str(cleaned_path),
            "para_map": str(map_path),
        }

    def read_raw(self, chapter_id: str) -> str | None:
        raw_path = self._root / "raw" / f"{chapter_id}.html"
        if not raw_path.exists():
            return None
        return raw_path.read_text(encoding="utf-8")

    def write(
        self,
        chapter_id: str,
        raw_html: str,
        cleaned_html: str,
        para_map: ParaIdMap,
    ) -> dict[str, str]:
        raw_path = self._root / "raw" / f"{chapter_id}.html"
        raw_path.write_text(raw_html, encoding="utf-8")
        cleaned = self.write_cleaned(chapter_id, cleaned_html, para_map)

        logger.debug(
            "HtmlArtifactWriter: wrote raw/cleaned/manifest for %s", chapter_id
        )
        return {
            "raw_html": str(raw_path),
            **cleaned,
        }

    def para_map_path(self, chapter_id: str) -> Path:
        return self._root / "manifest" / f"{chapter_id}{_PARA_MAP_EXT}"


def _backup_if_exists(path: Path) -> Optional[Path]:  # type: ignore[name-defined]
    if not path.exists():
        return None

    n = 1
    while True:
        backup = path.with_name(path.name + f"{_PARA_MAP_BACKUP_PREFIX}{n}")
        if not backup.exists():
            path.rename(backup)
            logger.info(
                "HtmlArtifactWriter: backed up %s -> %s", path.name, backup.name
            )
            return backup
        n += 1


from typing import Optional  # noqa: E402
