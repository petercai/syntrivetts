from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from syntrive.adapters.image.watermark import watermark_image

logger = logging.getLogger(__name__)

MARK_KEY = "----:com.syntrive:cover_watermark"
MARK_VALUE = b"SyntriveTTS/1"

WATERMARKED, ALREADY, NO_COVER, FAILED = "watermarked", "already", "no_cover", "failed"


@dataclass(frozen=True)
class CoverWatermarkResult:
    path: Path
    status: str
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.status != FAILED


def is_marked(tags) -> bool:
    return bool(tags) and MARK_KEY in tags


def watermark_m4a_cover(path: Path, *, dry_run: bool = False) -> CoverWatermarkResult:
    from mutagen.mp4 import MP4, MP4Cover, MP4FreeForm

    path = Path(path)
    try:
        audio = MP4(path)
        tags = audio.tags
        if is_marked(tags):
            return _done(path, ALREADY)
        covers = list(tags.get("covr", [])) if tags else []
        if not covers:
            return _done(path, NO_COVER)
        if dry_run:
            return _done(path, WATERMARKED, dry_run=True)
        first = covers[0]
        stamped = watermark_image(bytes(first))
        tags["covr"] = [MP4Cover(stamped, imageformat=first.imageformat), *covers[1:]]
        tags[MARK_KEY] = [MP4FreeForm(MARK_VALUE)]
        audio.save()
    except Exception as exc:  # noqa: BLE001 -- one bad file must not stop a pipeline or a repo walk
        logger.warning("cover_watermark_failed: path=%s error=%s", path, exc)
        return CoverWatermarkResult(path, FAILED, str(exc))
    return _done(path, WATERMARKED, cover_bytes=len(stamped))


def _done(path: Path, status: str, **fields) -> CoverWatermarkResult:
    logger.info("cover_watermark: path=%s status=%s %s", path, status,
                " ".join(f"{k}={v}" for k, v in fields.items()))
    return CoverWatermarkResult(path, status)


def find_repo_m4a_files(repo_dir: Path) -> list[Path]:
    return sorted(p for p in Path(repo_dir).glob("*/audiobooks/**/*.m4a") if p.is_file())


def watermark_repo(repo_dir: Path, *, dry_run: bool = False) -> list[CoverWatermarkResult]:
    files = find_repo_m4a_files(repo_dir)
    results = [watermark_m4a_cover(p, dry_run=dry_run) for p in files]
    counts = {s: sum(r.status == s for r in results) for s in (WATERMARKED, ALREADY, NO_COVER, FAILED)}
    logger.info("cover_watermark_repo: repo_dir=%s files=%d dry_run=%s %s", repo_dir, len(files), dry_run,
                " ".join(f"{k}={v}" for k, v in counts.items()))
    return results
