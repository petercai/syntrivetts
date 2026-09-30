from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import PurePosixPath
from typing import Any, Optional

CONTRACT_SCHEMA_VERSION = 1
REPO_MANIFEST_FILENAME = "repo.json"
JOB_MANIFEST_FILENAME = "job.json"
PUBLISH_TOC_FILENAME = "0_toc_publish.md"

REPO_SCHEMA_ID = "syntrive.repo"
JOB_SCHEMA_ID = "syntrive.job"

_VOLATILE_KEYS = frozenset({"generated_at", "content_sha256"})


class FactsAnchor(str, Enum):
    REPO = "repo"
    PROCESS = "process"


@dataclass(frozen=True)
class BookFacts:
    id: int
    title: str
    author: Optional[str] = None
    publisher: Optional[str] = None
    language: Optional[str] = None
    publish_date: Optional[str] = None
    original_filename: Optional[str] = None
    cover: Optional[str] = None


@dataclass(frozen=True)
class ImageFacts:
    name: Optional[str]
    path: Optional[str]
    image_type: Optional[str]


@dataclass(frozen=True)
class ChapterFacts:
    chapter_id: str
    sequence_number: Optional[str]
    volume_number: Optional[str] = None
    volume: Optional[str] = None
    chapter_number: Optional[str] = None
    chapter_name: Optional[str] = None
    title: Optional[str] = None
    transcript_path: Optional[str] = None
    chapter_audio_path: Optional[str] = None
    chapter_audio_seconds: Optional[float] = None
    synthesis_status: Optional[str] = None


@dataclass(frozen=True)
class JobFacts:
    id: int
    process_dir: str
    status: Optional[str] = None
    current_step: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    archived_at: Optional[datetime] = None
    source_path: Optional[str] = None
    cover_path: Optional[str] = None
    audiobooks_dir: Optional[str] = None
    sentence_audio_dir: Optional[str] = None
    chapter_audio_dir: Optional[str] = None
    transcript_dir: Optional[str] = None


@dataclass(frozen=True)
class JobSnapshot:
    job: JobFacts
    book: BookFacts
    chapters: tuple[ChapterFacts, ...] = ()
    images: tuple[ImageFacts, ...] = ()


@dataclass(frozen=True)
class PathIssue:
    field: str
    value: str
    reason: str


@dataclass(frozen=True)
class BuiltManifest:
    data: dict[str, Any]
    issues: tuple[PathIssue, ...] = field(default_factory=tuple)


def _is_absolute_like(value: str) -> bool:
    if value.startswith(("/", "\\")):
        return True
    return len(value) >= 2 and value[1] == ":"


def to_process_relative(
    value: Optional[str], anchor: FactsAnchor, process_dir: str, field_name: str
) -> tuple[Optional[str], Optional[PathIssue]]:
    if not value:
        return None, None
    normalised = value.replace("\\", "/")
    if _is_absolute_like(normalised):
        return None, PathIssue(field_name, value, "absolute path stored in DB")
    if anchor is FactsAnchor.PROCESS:
        return PurePosixPath(normalised).as_posix(), None

    candidate = PurePosixPath(normalised)
    base = PurePosixPath(process_dir.replace("\\", "/"))
    if candidate == base:
        return ".", None
    try:
        return candidate.relative_to(base).as_posix(), None
    except ValueError:
        return None, PathIssue(field_name, value, f"not under process_dir {process_dir!r}")


def _iso(ts: Optional[datetime]) -> Optional[str]:
    if ts is None:
        return None
    return ts.replace(microsecond=0).isoformat() + "Z"


def content_fingerprint(manifest: dict[str, Any]) -> str:
    stable = {k: v for k, v in manifest.items() if k not in _VOLATILE_KEYS}
    payload = json.dumps(stable, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _envelope(schema_id: str, generator: str, generated_at: datetime) -> dict[str, Any]:
    return {
        "schema": schema_id,
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "generator": generator,
        "generated_at": _iso(generated_at),
    }


def _sequence_key(chapter: ChapterFacts) -> tuple[str, str]:
    return (chapter.sequence_number or "", chapter.chapter_id)


def build_job_manifest(
    snapshot: JobSnapshot, *, generator: str, generated_at: datetime
) -> BuiltManifest:
    job, book = snapshot.job, snapshot.book
    issues: list[PathIssue] = []

    def rel(value: Optional[str], anchor: FactsAnchor, name: str) -> Optional[str]:
        path, issue = to_process_relative(value, anchor, job.process_dir, name)
        if issue is not None:
            issues.append(issue)
        return path

    sequenced = sorted((c for c in snapshot.chapters if c.sequence_number), key=_sequence_key)
    chapters = [
        {
            "sequence": c.sequence_number,
            "chapter_id": c.chapter_id,
            "volume_number": c.volume_number or None,
            "volume": c.volume or None,
            "chapter_number": c.chapter_number or None,
            "chapter_name": c.chapter_name or None,
            "title": c.title or None,
            "transcript_path": rel(c.transcript_path, FactsAnchor.PROCESS, f"chapters[{c.chapter_id}].transcript_path"),
            "audio_path": rel(c.chapter_audio_path, FactsAnchor.REPO, f"chapters[{c.chapter_id}].audio_path"),
            "audio_seconds": c.chapter_audio_seconds,
            "synthesis_status": c.synthesis_status,
        }
        for c in sequenced
    ]
    with_audio = [c for c in chapters if c["audio_path"]]

    data: dict[str, Any] = {
        **_envelope(JOB_SCHEMA_ID, generator, generated_at),
        "job": {
            "id": job.id,
            "status": job.status,
            "current_step": job.current_step,
            "created_at": _iso(job.created_at),
            "updated_at": _iso(job.updated_at),
            "archived_at": _iso(job.archived_at),
        },
        "book": {
            "id": book.id,
            "title": book.title,
            "author": book.author or None,
            "publisher": book.publisher or None,
            "language": book.language or None,
            "publish_date": book.publish_date or None,
            "original_filename": book.original_filename or None,
            "cover": rel(job.cover_path or book.cover, FactsAnchor.PROCESS, "book.cover"),
        },
        "paths": {
            "source": rel(job.source_path, FactsAnchor.REPO, "paths.source"),
            "audiobooks_dir": rel(job.audiobooks_dir, FactsAnchor.REPO, "paths.audiobooks_dir"),
            "chapter_audio_dir": rel(job.chapter_audio_dir, FactsAnchor.REPO, "paths.chapter_audio_dir"),
            "sentence_audio_dir": rel(job.sentence_audio_dir, FactsAnchor.REPO, "paths.sentence_audio_dir"),
            "transcript_dir": rel(job.transcript_dir, FactsAnchor.REPO, "paths.transcript_dir"),
            "publish_toc": PUBLISH_TOC_FILENAME,
        },
        "images": [
            {
                "name": img.name,
                "path": rel(img.path, FactsAnchor.PROCESS, f"images[{img.name}].path"),
                "type": img.image_type,
            }
            for img in snapshot.images
        ],
        "chapters": chapters,
        "summary": {
            "chapter_count": len(chapters),
            "unsequenced_chapter_count": len(snapshot.chapters) - len(chapters),
            "chapters_with_audio": len(with_audio),
            "total_audio_seconds": round(sum(c["audio_seconds"] or 0.0 for c in with_audio), 3),
        },
    }
    data["content_sha256"] = content_fingerprint(data)
    return BuiltManifest(data=data, issues=tuple(issues))


@dataclass(frozen=True)
class RepoJobEntry:
    job: JobFacts
    book: BookFacts


def build_repo_manifest(
    entries: tuple[RepoJobEntry, ...], *, db_filename: str, generator: str, generated_at: datetime
) -> BuiltManifest:
    issues: list[PathIssue] = []
    jobs = []
    for entry in sorted(entries, key=lambda e: e.job.id):
        process_dir = entry.job.process_dir.replace("\\", "/")
        if _is_absolute_like(process_dir):
            issues.append(PathIssue(f"jobs[{entry.job.id}].process_dir", entry.job.process_dir, "absolute path stored in DB"))
            continue
        jobs.append(
            {
                "job_id": entry.job.id,
                "book_id": entry.book.id,
                "title": entry.book.title,
                "author": entry.book.author or None,
                "language": entry.book.language or None,
                "process_dir": PurePosixPath(process_dir).as_posix(),
                "job_manifest": (PurePosixPath(process_dir) / JOB_MANIFEST_FILENAME).as_posix(),
                "status": entry.job.status,
                "current_step": entry.job.current_step,
                "updated_at": _iso(entry.job.updated_at),
                "archived": entry.job.archived_at is not None,
                "archived_at": _iso(entry.job.archived_at),
            }
        )

    data: dict[str, Any] = {
        **_envelope(REPO_SCHEMA_ID, generator, generated_at),
        "db_file": db_filename,
        "jobs": jobs,
    }
    data["content_sha256"] = content_fingerprint(data)
    return BuiltManifest(data=data, issues=tuple(issues))
