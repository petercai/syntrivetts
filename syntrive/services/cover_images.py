from __future__ import annotations

import logging
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable, Optional, Sequence
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

MAX_IMAGE_BYTES = 20 * 1024 * 1024
URL_TIMEOUT_SECONDS = 15
_USER_AGENT = "SyntriveTTS-cover-import/1.0"
_SAFE_STEM = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass(frozen=True)
class ImportOutcome:
    source: str
    added: Optional[str]
    error: Optional[str] = None


@dataclass(frozen=True)
class ImportResult:
    outcomes: tuple[ImportOutcome, ...]

    @property
    def added(self) -> tuple[str, ...]:
        return tuple(o.added for o in self.outcomes if o.added)

    @property
    def refused(self) -> tuple[ImportOutcome, ...]:
        return tuple(o for o in self.outcomes if o.error)


def image_extension(data: bytes) -> Optional[str]:
    if data.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return ".gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return ".webp"
    if data[:2] == b"BM":
        return ".bmp"
    return None


def _stem(source: str) -> str:
    name = PurePosixPath(urlparse(source).path or source).name or "image"
    stem = _SAFE_STEM.sub("-", Path(name).stem).strip("-.")[:40]
    return stem or "image"


def _free_name(images_dir: Path, stem: str, ext: str) -> Path:
    candidate = images_dir / f"imported-{stem}{ext}"
    n = 2
    while candidate.exists():
        candidate = images_dir / f"imported-{stem}-{n}{ext}"
        n += 1
    return candidate


def _download(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(request, timeout=URL_TIMEOUT_SECONDS) as response:  # noqa: S310 - scheme checked by caller
        return response.read(MAX_IMAGE_BYTES + 1)


def _check(data: bytes) -> tuple[Optional[str], Optional[str]]:
    if not data:
        return None, "empty"
    if len(data) > MAX_IMAGE_BYTES:
        return None, "too_large"
    ext = image_extension(data)
    return (ext, None) if ext else (None, "not_image")


def import_cover_images(
    db_path: Path,
    job_id: int,
    uploads: Sequence[tuple[str, bytes]] = (),
    urls: Iterable[str] = (),
) -> ImportResult:
    from syntrive.db.models import Job
    from syntrive.db.path_utils import resolve_abs
    from syntrive.db.session import get_db_session
    from syntrive.services.job_service import JobService

    with get_db_session(db_path) as db:
        job = db.get(Job, job_id)
        if job is None:
            raise LookupError(f"Job {job_id} not found.")
        images_dir = resolve_abs(db_path, job.process_dir) / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    def store(source: str, data: bytes) -> ImportOutcome:
        ext, error = _check(data)
        if error:
            logger.info("cover_import_refused: job_id=%d source=%s reason=%s bytes=%d", job_id, source, error, len(data))
            return ImportOutcome(source, None, error)
        target = _free_name(images_dir, _stem(source), ext)
        target.write_bytes(data)
        logger.info("cover_import_added: job_id=%d source=%s file=%s bytes=%d", job_id, source, target.name, len(data))
        return ImportOutcome(source, f"images/{target.name}")

    def fetch(url: str) -> ImportOutcome:
        if urlparse(url).scheme not in ("http", "https"):
            logger.info("cover_import_refused: job_id=%d source=%s reason=bad_url", job_id, url)
            return ImportOutcome(url, None, "bad_url")
        try:
            return store(url, _download(url))
        except (urllib.error.URLError, OSError, ValueError) as exc:
            logger.warning("cover_import_refused: job_id=%d source=%s reason=download_failed error=%s", job_id, url, exc)
            return ImportOutcome(url, None, "download_failed")

    outcomes = [store(name or "upload", data) for name, data in uploads]
    outcomes += [fetch(url.strip()) for url in urls if url.strip()]
    if any(o.added for o in outcomes):
        JobService(db_path).sync_book_images_from_dir(job_id)
    result = ImportResult(tuple(outcomes))
    logger.info("cover_import_done: job_id=%d added=%d refused=%d", job_id, len(result.added), len(result.refused))
    return result
