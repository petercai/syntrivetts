from __future__ import annotations

import codecs
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

from syntrive.io.paths import is_portable_stored_path

logger = logging.getLogger(__name__)


DEFAULT_BACKUP_FILENAME = "tts-reference-voice.sql"

_REFERENCE_VOICE_COLUMNS = (
    "id", "path", "name", "gender", "language", "accent", "description",
    "sample_rate", "bit_depth", "channels", "duration_seconds", "created_at",
)
_REFERENCE_VOICE_TAG_COLUMNS = ("id", "reference_voice_id", "tag")


def sql_literal(value: object) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (int, float)):
        return repr(value)
    return "'" + str(value).replace("'", "''") + "'"


def insert_sql(table: str, columns: tuple[str, ...], row: object) -> str:
    values = ", ".join(sql_literal(getattr(row, col)) for col in columns)
    return f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({values});"


def backup_reference_voices(db_path: Path, backup_path: Path) -> tuple[int, int, int]:
    from syntrive.db.models import ReferenceVoice, ReferenceVoiceTag
    from syntrive.db.session import get_db_session

    with get_db_session(db_path) as db:
        all_voices = db.query(ReferenceVoice).order_by(ReferenceVoice.id).all()
        portable = [v for v in all_voices if is_portable_stored_path(v.path)]
        skipped = len(all_voices) - len(portable)
        voice_ids = {v.id for v in portable}
        tags = [
            t for t in db.query(ReferenceVoiceTag).order_by(ReferenceVoiceTag.id).all()
            if t.reference_voice_id in voice_ids
        ]

        lines = [
            "-- SyntriveTTS reference_voices/reference_voice_tags backup",
            f"-- generated: {datetime.utcnow().isoformat()}Z",
            f"-- source db: {db_path}",
            f"-- {len(portable)} voice(s), {skipped} absolute-path voice(s) skipped "
            f"(not portable across machines), {len(tags)} tag(s)",
            "",
        ]
        lines += [insert_sql("reference_voices", _REFERENCE_VOICE_COLUMNS, v) for v in portable]
        lines += [insert_sql("reference_voice_tags", _REFERENCE_VOICE_TAG_COLUMNS, t) for t in tags]

    backup_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info(
        "backup_reference_voices: db=%s file=%s voices=%d skipped_absolute=%d tags=%d",
        db_path, backup_path, len(portable), skipped, len(tags),
    )
    return len(portable), skipped, len(tags)


def restore_reference_voices(db_path: Path, backup_path: Path) -> tuple[int, int]:
    if not backup_path.is_file():
        raise FileNotFoundError(f"backup file not found: {backup_path}")

    from sqlalchemy import text as sa_text

    from syntrive.db.session import get_db_session

    statements = [
        line.strip() for line in backup_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("--")
    ]

    voice_inserts = tag_inserts = 0
    with get_db_session(db_path) as db:
        for stmt in statements:
            db.execute(sa_text(stmt))
            upper = stmt.upper()
            if upper.startswith("INSERT INTO REFERENCE_VOICE_TAGS"):
                tag_inserts += 1
            elif upper.startswith("INSERT INTO REFERENCE_VOICES"):
                voice_inserts += 1

    logger.info(
        "restore_reference_voices: db=%s file=%s voices=%d tags=%d",
        db_path, backup_path, voice_inserts, tag_inserts,
    )
    return voice_inserts, tag_inserts


@dataclass(frozen=True)
class JobSummary:
    job_id: int
    process_dir: str
    stage: str
    status: str


@dataclass(frozen=True)
class BookSummary:
    book_id: int
    title: str
    author: Optional[str]
    jobs: tuple[JobSummary, ...]


@dataclass(frozen=True)
class JobConflict:
    export_job_id: int
    target_job_id: int
    book_title: str
    process_dir: str


@dataclass(frozen=True)
class ExportResult:
    output_path: Path
    book_count: int
    job_count: int
    reference_voice_count: int
    process_dir_reminders: tuple[str, ...]
    unbound_tts_voice_warnings: tuple[tuple[int, str], ...]


@dataclass(frozen=True)
class ImportResult:
    created_job_count: int
    overwritten_job_count: int
    skipped_job_count: int
    reference_voice_resolved_count: int
    reference_voice_unresolved_count: int


def list_books_with_jobs(db_path: Path) -> list[BookSummary]:
    from syntrive.db.models import Book
    from syntrive.db.session import get_db_session

    with get_db_session(db_path) as db:
        books = db.query(Book).order_by(Book.id).all()
        return [
            BookSummary(
                book_id=book.id,
                title=book.title,
                author=book.author,
                jobs=tuple(
                    JobSummary(job_id=j.id, process_dir=j.process_dir, stage=j.stage, status=j.status)
                    for j in sorted(book.jobs, key=lambda j: j.id)
                ),
            )
            for book in books
        ]


def expand_book_selection(
    summaries: Sequence[BookSummary], *, book_ids: Sequence[int] = (), job_ids: Sequence[int] = (),
) -> list[int]:
    book_id_set = set(book_ids)
    resolved = set(job_ids)
    for summary in summaries:
        if summary.book_id in book_id_set:
            resolved.update(j.job_id for j in summary.jobs)
    return sorted(resolved)


def copy_row(source_obj, model_cls, **overrides):
    data = {
        col.name: getattr(source_obj, col.name)
        for col in model_cls.__table__.columns
        if col.name != "id" and col.name not in overrides
    }
    data.update(overrides)
    return model_cls(**data)


def _select_jobs(session, job_ids: Optional[Sequence[int]]):
    from syntrive.db.models import Job

    if job_ids is None:
        return session.query(Job).order_by(Job.id).all()
    jobs = []
    for jid in job_ids:
        job = session.get(Job, jid)
        if job is None:
            raise ValueError(f"job id not found: {jid}")
        jobs.append(job)
    return jobs


_INFLIGHT_SYNTHESIS_STATUSES = frozenset({"queued", "synthesizing", "combining"})


def _portable_chapter_overrides(chapter) -> dict:
    overrides: dict = {"synthesis_batch_id": None}
    if chapter.synthesis_status in _INFLIGHT_SYNTHESIS_STATUSES:
        overrides["synthesis_status"] = "pending"
    return overrides


def _copy_job_children(dst, source_job, new_job_id: int, *, tts_voice_reference_resolver) -> None:
    from syntrive.db.models import (
        OutputConfig, StageEvent, ThresholdBlockEvent, TranscriptChapter, TtsConfig, TtsVoice,
        WorkflowStepEvent,
    )

    tts_config = source_job.tts_config
    if tts_config is not None:
        new_tts_config = copy_row(tts_config, TtsConfig, job_id=new_job_id)
        dst.add(new_tts_config)
        dst.flush()
        for voice in tts_config.tts_voices:
            new_ref_id = (
                tts_voice_reference_resolver(source_job.id, voice)
                if voice.reference_voice_id is not None else None
            )
            dst.add(copy_row(
                voice, TtsVoice, tts_config_id=new_tts_config.id, reference_voice_id=new_ref_id,
            ))

    output_config = source_job.output_config
    if output_config is not None:
        dst.add(copy_row(output_config, OutputConfig, job_id=new_job_id))
    for row in source_job.stage_events:
        dst.add(copy_row(row, StageEvent, job_id=new_job_id))
    detached = reset = 0
    for row in source_job.transcript_chapters:
        overrides = _portable_chapter_overrides(row)
        detached += row.synthesis_batch_id is not None
        reset += "synthesis_status" in overrides
        dst.add(copy_row(row, TranscriptChapter, job_id=new_job_id, **overrides))
    logger.info(
        "_copy_job_children: source_job_id=%s new_job_id=%s chapters_detached_from_batch=%d "
        "chapters_reset_to_pending=%d",
        source_job.id, new_job_id, detached, reset,
    )
    for row in source_job.threshold_blocks:
        dst.add(copy_row(row, ThresholdBlockEvent, job_id=new_job_id))
    for row in source_job.workflow_step_events:
        dst.add(copy_row(row, WorkflowStepEvent, job_id=new_job_id))


def export_jobs(
    source_db_path: Path,
    output_path: Path,
    job_ids: Sequence[int],
    *,
    include_reference_voices: bool = False,
) -> ExportResult:
    from syntrive.db.models import Book, BookImage, Job
    from syntrive.db.session import ensure_db_schema, get_db_session, seal_database_file

    if not job_ids:
        raise ValueError("job_ids must not be empty")

    ensure_db_schema(output_path)

    book_id_map: dict[int, int] = {}
    process_dir_reminders: list[str] = []
    unbound_warnings: list[tuple[int, str]] = []

    with get_db_session(source_db_path) as src, get_db_session(output_path) as dst:
        jobs = _select_jobs(src, job_ids)

        reference_voice_id_map = _copy_all_reference_voices(src, dst) if include_reference_voices else {}

        def resolve_export_reference(job_id: int, voice) -> Optional[int]:
            if not include_reference_voices:
                unbound_warnings.append((job_id, voice.name))
                logger.warning(
                    "export_jobs: reference_voices not included -- "
                    "job_id=%d tts_voice=%s loses its voice binding",
                    job_id, voice.name,
                )
                return None
            return reference_voice_id_map.get(voice.reference_voice_id)

        for job in jobs:
            book = job.book
            if book.id not in book_id_map:
                new_book = copy_row(book, Book)
                dst.add(new_book)
                dst.flush()
                for img in book.images:
                    dst.add(copy_row(img, BookImage, book_id=new_book.id))
                book_id_map[book.id] = new_book.id

            new_job = copy_row(job, Job, book_id=book_id_map[book.id])
            dst.add(new_job)
            dst.flush()
            _copy_job_children(dst, job, new_job.id, tts_voice_reference_resolver=resolve_export_reference)
            process_dir_reminders.append(job.process_dir)

    seal_database_file(output_path)
    logger.info(
        "export_jobs: source=%s output=%s books=%d jobs=%d reference_voices=%d",
        source_db_path, output_path, len(book_id_map), len(jobs), len(reference_voice_id_map),
    )
    return ExportResult(
        output_path=output_path,
        book_count=len(book_id_map),
        job_count=len(jobs),
        reference_voice_count=len(reference_voice_id_map),
        process_dir_reminders=tuple(process_dir_reminders),
        unbound_tts_voice_warnings=tuple(unbound_warnings),
    )


def _copy_all_reference_voices(src, dst) -> dict[int, int]:
    from syntrive.db.models import ReferenceVoice, ReferenceVoiceTag

    id_map: dict[int, int] = {}
    for rv in src.query(ReferenceVoice).order_by(ReferenceVoice.id).all():
        new_rv = copy_row(rv, ReferenceVoice)
        dst.add(new_rv)
        dst.flush()
        for tag in rv.tags:
            dst.add(copy_row(tag, ReferenceVoiceTag, reference_voice_id=new_rv.id))
        id_map[rv.id] = new_rv.id
    return id_map


def find_job_conflicts(
    export_db_path: Path, target_db_path: Path, job_ids: Optional[Sequence[int]] = None,
) -> list[JobConflict]:
    from syntrive.db.models import Book, Job
    from syntrive.db.session import ensure_db_schema, get_db_session

    ensure_db_schema(target_db_path)
    conflicts: list[JobConflict] = []
    with get_db_session(export_db_path) as src, get_db_session(target_db_path) as dst:
        for job in _select_jobs(src, job_ids):
            book = job.book
            target_book = (
                dst.query(Book).filter(Book.title == book.title, Book.author == book.author).first()
            )
            if target_book is None:
                continue
            existing_job = (
                dst.query(Job)
                .filter(Job.book_id == target_book.id, Job.process_dir == job.process_dir)
                .first()
            )
            if existing_job is not None:
                conflicts.append(JobConflict(
                    export_job_id=job.id, target_job_id=existing_job.id,
                    book_title=book.title, process_dir=job.process_dir,
                ))
    return conflicts


def import_jobs(
    export_db_path: Path,
    target_db_path: Path,
    job_ids: Optional[Sequence[int]] = None,
    *,
    overwrite_job_ids: frozenset = frozenset(),
    skip_job_ids: frozenset = frozenset(),
) -> ImportResult:
    from syntrive.db.models import BookImage, Job
    from syntrive.db.repository import JobRepository
    from syntrive.db.session import ensure_db_schema, get_db_session

    ensure_db_schema(target_db_path)

    created = overwritten = skipped = 0
    ref_resolved = ref_unresolved = 0
    book_id_map: dict[int, int] = {}

    with get_db_session(export_db_path) as src, get_db_session(target_db_path) as dst:
        jobs = _select_jobs(src, job_ids)
        repo = JobRepository(dst)
        reference_voice_id_map = _import_all_reference_voices(src, dst)

        def resolve_import_reference(_job_id: int, voice) -> Optional[int]:
            nonlocal ref_resolved, ref_unresolved
            new_ref_id = reference_voice_id_map.get(voice.reference_voice_id)
            if new_ref_id is not None:
                ref_resolved += 1
            else:
                ref_unresolved += 1
            return new_ref_id

        for job in jobs:
            book = job.book
            if book.id not in book_id_map:
                target_book = repo.find_or_create_book(
                    title=book.title, author=book.author, publisher=book.publisher,
                    language=book.language, publish_date=book.publish_date,
                    original_filename=book.original_filename,
                )
                if not target_book.cover and book.cover:
                    target_book.cover = book.cover
                if not target_book.extraction_mode and book.extraction_mode:
                    target_book.extraction_mode = book.extraction_mode
                existing_image_names = {img.name for img in target_book.images}
                for img in book.images:
                    if img.name not in existing_image_names:
                        dst.add(copy_row(img, BookImage, book_id=target_book.id))
                        existing_image_names.add(img.name)
                dst.flush()
                book_id_map[book.id] = target_book.id

            target_book_id = book_id_map[book.id]
            existing_job = (
                dst.query(Job)
                .filter(Job.book_id == target_book_id, Job.process_dir == job.process_dir)
                .first()
            )
            if existing_job is not None:
                if job.id in skip_job_ids:
                    skipped += 1
                    continue
                if job.id not in overwrite_job_ids:
                    raise ValueError(
                        f"export job id {job.id} conflicts with existing target job id "
                        f"{existing_job.id} but was not resolved via overwrite_job_ids/"
                        "skip_job_ids -- call find_job_conflicts() first"
                    )
                dst.delete(existing_job)
                dst.flush()
                overwritten += 1
            else:
                created += 1

            new_job = copy_row(job, Job, book_id=target_book_id)
            dst.add(new_job)
            dst.flush()
            _copy_job_children(dst, job, new_job.id, tts_voice_reference_resolver=resolve_import_reference)

    logger.info(
        "import_jobs: export=%s target=%s created=%d overwritten=%d skipped=%d "
        "ref_voices_resolved=%d ref_voices_unresolved=%d",
        export_db_path, target_db_path, created, overwritten, skipped, ref_resolved, ref_unresolved,
    )
    return ImportResult(
        created_job_count=created, overwritten_job_count=overwritten, skipped_job_count=skipped,
        reference_voice_resolved_count=ref_resolved, reference_voice_unresolved_count=ref_unresolved,
    )


def _import_all_reference_voices(src, dst) -> dict[int, int]:
    from syntrive.db.models import ReferenceVoice, ReferenceVoiceTag

    id_map: dict[int, int] = {}
    for rv in src.query(ReferenceVoice).order_by(ReferenceVoice.id).all():
        existing = dst.query(ReferenceVoice).filter(ReferenceVoice.path == rv.path).first()
        if existing is not None:
            id_map[rv.id] = existing.id
            continue
        new_rv = copy_row(rv, ReferenceVoice)
        dst.add(new_rv)
        dst.flush()
        for tag in rv.tags:
            dst.add(copy_row(tag, ReferenceVoiceTag, reference_voice_id=new_rv.id))
        id_map[rv.id] = new_rv.id
    return id_map


def default_export_filename() -> str:
    return f"syntrivetts-export-{datetime.now().strftime('%Y-%m-%d-%H%M%S')}.db"


def find_export_files(repo_dir: Path) -> list[Path]:
    candidates = [p for p in repo_dir.glob("*.db") if p.name != "syntrivetts.db"]
    return sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)


LIVE_DB_NAME = "syntrivetts.db"
BACKUP_GLOB = "*.sql"


@dataclass(frozen=True)
class ExportFile:
    path: Path
    size_bytes: int
    modified: datetime
    books: tuple[BookSummary, ...]
    error: str = ""

    @property
    def job_count(self) -> int:
        return sum(len(b.jobs) for b in self.books)


@dataclass(frozen=True)
class BackupFile:
    path: Path
    size_bytes: int
    modified: datetime
    voice_count: int
    tag_count: int


def _stat(path: Path) -> tuple[int, datetime]:
    info = path.stat()
    return info.st_size, datetime.fromtimestamp(info.st_mtime)


_SUMMARY_SQL = (
    "SELECT b.id, b.title, b.author, j.id, j.process_dir, j.stage, j.status "
    "FROM books b LEFT JOIN jobs j ON j.book_id = b.id ORDER BY b.id, j.id"
)


def read_books_readonly(path: Path) -> tuple[BookSummary, ...]:
    import sqlite3
    from contextlib import closing

    uri = f"{path.resolve().as_uri()}?mode=ro&immutable=1"
    with closing(sqlite3.connect(uri, uri=True)) as conn:
        rows = conn.execute(_SUMMARY_SQL).fetchall()
    books: dict[int, tuple[str, Optional[str], list[JobSummary]]] = {}
    for book_id, title, author, job_id, process_dir, stage, status in rows:
        entry = books.setdefault(book_id, (title, author, []))
        if job_id is not None:
            entry[2].append(JobSummary(job_id=job_id, process_dir=process_dir, stage=stage, status=status))
    return tuple(BookSummary(book_id=b, title=t, author=a, jobs=tuple(j)) for b, (t, a, j) in books.items())


def describe_export_file(path: Path) -> ExportFile:
    size, modified = _stat(path)
    try:
        books = read_books_readonly(path)
    except Exception as exc:  # noqa: BLE001 -- a foreign / corrupt .db must not break the listing
        logger.warning("export_file_unreadable: path=%s error=%s", path, exc)
        return ExportFile(path, size, modified, (), error=str(exc))
    return ExportFile(path, size, modified, books)


def list_export_files(repo_dir: Path) -> list[ExportFile]:
    files = [describe_export_file(p) for p in find_export_files(repo_dir)]
    logger.info("export_files_listed: repo_dir=%s files=%d", repo_dir, len(files))
    return files


def _count_inserts(path: Path) -> tuple[int, int]:
    voices = tags = 0
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        upper = line.lstrip().upper()
        if upper.startswith("INSERT INTO REFERENCE_VOICE_TAGS"):
            tags += 1
        elif upper.startswith("INSERT INTO REFERENCE_VOICES"):
            voices += 1
    return voices, tags


def list_backup_files(repo_dir: Path) -> list[BackupFile]:
    files = []
    for path in repo_dir.glob(BACKUP_GLOB):
        size, modified = _stat(path)
        files.append(BackupFile(path, size, modified, *_count_inserts(path)))
    return sorted(files, key=lambda f: f.modified, reverse=True)


def default_backup_filename() -> str:
    return f"tts-reference-voice-{datetime.now().strftime('%Y-%m-%d-%H%M%S')}.sql"


def backup_to_repo(db_path: Path) -> tuple[Path, int, int, int]:
    path = _free_name(db_path.parent, default_backup_filename())
    voices, skipped, tags = backup_reference_voices(db_path, path)
    return path, voices, skipped, tags


def export_to_repo(db_path: Path, job_ids: Sequence[int], *, include_reference_voices: bool) -> ExportResult:
    output = _free_name(db_path.parent, default_export_filename())
    return export_jobs(db_path, output, job_ids, include_reference_voices=include_reference_voices)


@dataclass(frozen=True)
class ImportPlan:
    export: ExportFile
    conflicts: tuple[JobConflict, ...]

    @property
    def conflict_ids(self) -> frozenset:
        return frozenset(c.export_job_id for c in self.conflicts)


def plan_import(export_path: Path, target_db_path: Path) -> ImportPlan:
    export = describe_export_file(export_path)
    conflicts = () if export.error else tuple(find_job_conflicts(export_path, target_db_path))
    logger.info("import_planned: export=%s target=%s jobs=%d conflicts=%d",
                export_path.name, target_db_path, export.job_count, len(conflicts))
    return ImportPlan(export, conflicts)


def import_into_repo(
    export_path: Path, target_db_path: Path, job_ids: Sequence[int], *,
    overwrite_job_ids: frozenset = frozenset(), holder_kind: str,
) -> ImportResult:
    from syntrive.services.contract_export_service import refresh_manifests_best_effort
    from syntrive.services.job_lease import hold_job_leases

    conflicts = find_job_conflicts(export_path, target_db_path, list(job_ids))
    overwrite = frozenset(c.export_job_id for c in conflicts if c.export_job_id in overwrite_job_ids)
    skip = frozenset(c.export_job_id for c in conflicts) - overwrite
    replaced = [c.target_job_id for c in conflicts if c.export_job_id in overwrite]
    with hold_job_leases(target_db_path, replaced, holder_kind=holder_kind, operation="db_import"):
        result = import_jobs(export_path, target_db_path, list(job_ids),
                             overwrite_job_ids=overwrite, skip_job_ids=skip)
    refresh_manifests_best_effort(target_db_path, reason="db:import")
    return result


REPO_FILE_SUFFIXES = (".db", ".sql")
SQLITE_HEADER = b"SQLite format 3\x00"
UPLOAD_MAX_BYTES = 2 * 2**30
_WAL_SIDECARS = ("-wal", "-shm", "-journal")


class RepoFileError(ValueError):
    def __init__(self, code: str, name: str) -> None:
        super().__init__(f"{code}: {name}")
        self.code = code
        self.name = name


def _check_name(name: str) -> None:
    if not name or name != Path(name).name or name in (".", "..") or "/" in name or "\\" in name:
        raise RepoFileError("bad_name", name)
    if Path(name).suffix.lower() not in REPO_FILE_SUFFIXES:
        raise RepoFileError("bad_kind", name)
    if name.lower() == LIVE_DB_NAME:
        raise RepoFileError("live_db", name)


def repo_file(repo_dir: Path, name: str) -> Path:
    _check_name(name)
    path = repo_dir / name
    if not path.is_file():
        raise RepoFileError("not_found", name)
    return path


def delete_repo_file(repo_dir: Path, name: str) -> Path:
    path = repo_file(repo_dir, name)
    path.unlink()
    removed = [side for side in _WAL_SIDECARS if Path(f"{path}{side}").exists()]
    for side in removed:
        Path(f"{path}{side}").unlink(missing_ok=True)
    logger.info("repo_file_deleted: repo_dir=%s name=%s sidecars=%s", repo_dir, name, removed)
    return path


def _free_name(repo_dir: Path, name: str) -> Path:
    candidate = repo_dir / name
    stem, suffix = Path(name).stem, Path(name).suffix
    n = 2
    while candidate.exists() or candidate.name.lower() == LIVE_DB_NAME:
        candidate = repo_dir / f"{stem} ({n}){suffix}"
        n += 1
    return candidate


def save_upload(repo_dir: Path, filename: str, stream, *, max_bytes: int = UPLOAD_MAX_BYTES) -> Path:
    name = Path(filename or "").name
    if Path(name).suffix.lower() not in REPO_FILE_SUFFIXES:
        raise RepoFileError("bad_kind", name)
    is_db = Path(name).suffix.lower() == ".db"
    part = repo_dir / f".upload-{datetime.now().strftime('%Y%m%d%H%M%S%f')}.part"
    size, head = 0, b""
    text = codecs.getincrementaldecoder("utf-8")()
    try:
        with part.open("wb") as out:
            while chunk := stream.read(1024 * 1024):
                size += len(chunk)
                if size > max_bytes:
                    raise RepoFileError("too_large", name)
                head = (head + chunk)[:len(SQLITE_HEADER)] if len(head) < len(SQLITE_HEADER) else head
                if not is_db:
                    try:
                        text.decode(chunk)
                    except UnicodeDecodeError as exc:
                        raise RepoFileError("not_text", name) from exc
                out.write(chunk)
        if is_db and head != SQLITE_HEADER:
            raise RepoFileError("not_sqlite", name)
        if not is_db:
            try:
                text.decode(b"", final=True)
            except UnicodeDecodeError as exc:
                raise RepoFileError("not_text", name) from exc
        target = _free_name(repo_dir, name)
        part.replace(target)
    finally:
        part.unlink(missing_ok=True)
    logger.info("repo_file_uploaded: repo_dir=%s name=%s saved_as=%s bytes=%d", repo_dir, name, target.name, size)
    return target


def list_repo_files(repo_dir: Path) -> list:
    return sorted([*list_export_files(repo_dir), *list_backup_files(repo_dir)], key=lambda f: f.modified, reverse=True)


def repo_file_count(repo_dir: Path) -> int:
    return len(find_export_files(repo_dir)) + len(list(repo_dir.glob(BACKUP_GLOB)))
