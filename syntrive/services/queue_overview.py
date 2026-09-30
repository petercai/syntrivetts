from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable, Mapping, Optional

logger = logging.getLogger(__name__)

RUNNER_HOLDER_KIND = "tts_batch"
EVENT_LOG = Path("logs") / "tts_batch.log"
_TAIL_BYTES = 128 * 1024
_PROGRESS_EVENTS = frozenset({"synth_batch_start", "chapter_start", "line_start", "line_ok", "line_failed",
                              "line_retry", "chapter_merged", "batch_aborted", "batch_stopped"})
STATUSES = ("pending", "queued", "synthesizing", "combining", "done")


@dataclass(frozen=True)
class RunnerInfo:
    job_id: int
    operation: str
    hostname: str
    pid: int
    since: float

    @property
    def batch_id(self) -> Optional[int]:
        head, _, tail = self.operation.partition(":")
        return int(tail) if head == "batch" and tail.isdigit() else None


@dataclass(frozen=True)
class LiveProgress:
    at: str
    event: str
    batch_id: Optional[int]
    job_id: Optional[int]
    book: Optional[str]
    book_seq: Optional[int]
    book_total: Optional[int]
    chapter_id: Optional[str]
    chapter_seq: Optional[int]
    chapter_total: Optional[int]
    line_idx: Optional[int]
    line_total: Optional[int]
    text: Optional[str]
    error: Optional[str]

    @property
    def fraction(self) -> Optional[float]:
        if self.line_idx is None or not self.line_total:
            return None
        return min(1.0, max(0.0, self.line_idx / self.line_total))


@dataclass(frozen=True)
class QueueRow:
    batch_id: int
    job_id: int
    book_title: str
    chapter_db_id: int
    chapter_id: str
    chapter_name: str
    sequence: str
    status: str
    paused: bool
    created_at: Optional[datetime]
    started_at: Optional[datetime]
    finished_at: Optional[datetime]
    error: Optional[str]
    failed_line: Optional[int]
    audio_seconds: Optional[float]

    @property
    def stuck(self) -> bool:
        return bool(self.error) and self.status != "done"


@dataclass(frozen=True)
class BookQueueSummary:
    job_id: int
    title: str
    chapters: int
    counts: Mapping[str, int] = field(default_factory=dict)
    schedulable: int = 0
    paused: int = 0
    audio_seconds: float = 0.0

    @property
    def done(self) -> int:
        return self.counts.get("done", 0)


@dataclass(frozen=True)
class QueueOverview:
    runner: Optional[RunnerInfo]
    progress: Optional[LiveProgress]
    live: bool
    queue: tuple[QueueRow, ...]
    recent: tuple[QueueRow, ...]
    books: tuple[BookQueueSummary, ...]
    process: Optional[object] = None

    @property
    def running(self) -> bool:
        return self.runner is not None or bool(self.process is not None and self.process.alive)

    @property
    def stuck(self) -> tuple[QueueRow, ...]:
        return tuple(r for r in self.queue if r.stuck)

    @property
    def runnable_count(self) -> int:
        return sum(1 for r in self.queue if not r.paused)


def parse_events(lines: Iterable[str]) -> list[dict]:
    events = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if isinstance(record, dict) and "event" in record:
            events.append(record)
    return events


def _int(value: object) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def summarize(events: list[dict]) -> Optional[LiveProgress]:
    progress = [e for e in events if e.get("event") in _PROGRESS_EVENTS]
    if not progress:
        return None
    last = progress[-1]
    line = next(
        (e for e in reversed(progress) if e.get("event") == "line_start"
         and e.get("chapter_id") == last.get("chapter_id") and e.get("job_id", last.get("job_id")) == last.get("job_id")),
        {},
    )
    return LiveProgress(
        at=str(last.get("ts", "")), event=str(last["event"]),
        batch_id=_int(last.get("batch_id", line.get("batch_id"))), job_id=_int(last.get("job_id", line.get("job_id"))),
        book=line.get("book"), book_seq=_int(line.get("book_seq")), book_total=_int(line.get("book_total")),
        chapter_id=last.get("chapter_id") or line.get("chapter_id"),
        chapter_seq=_int(line.get("chapter_seq")), chapter_total=_int(line.get("chapter_total")),
        line_idx=_int(last.get("line_idx", line.get("line_idx"))), line_total=_int(line.get("line_total")),
        text=line.get("text"), error=last.get("error"),
    )


def read_event_tail(repo_dir: Path, max_bytes: int = _TAIL_BYTES) -> list[dict]:
    path = repo_dir / EVENT_LOG
    if not path.is_file():
        return []
    try:
        with path.open("rb") as fh:
            size = fh.seek(0, 2)
            fh.seek(max(0, size - max_bytes))
            chunk = fh.read().decode("utf-8", errors="replace")
    except OSError as exc:
        logger.warning("queue_event_log_unreadable: path=%s error=%s", path, exc)
        return []
    return parse_events(chunk.splitlines())


def live_runner(db_path: Path, now: Optional[float] = None) -> Optional[RunnerInfo]:
    from syntrive.services.job_lease import HolderIdentity, decide_acquisition, list_leases, pid_alive

    now = time.time() if now is None else now
    me = HolderIdentity.current("check")
    for lease in list_leases(db_path):
        if lease.holder_kind != RUNNER_HOLDER_KIND:
            continue
        if decide_acquisition(lease, me, now, pid_alive).allowed:
            continue
        return RunnerInfo(lease.job_id, lease.operation or "", lease.hostname, lease.pid, lease.acquired_at)
    return None


def _rows(db_path: Path) -> tuple[list[QueueRow], dict[int, BookQueueSummary]]:
    from syntrive.db.models import Book, Job, SynthesisBatch, TranscriptChapter
    from syntrive.db.session import get_db_session

    rows: list[QueueRow] = []
    books: dict[int, dict] = {}
    with get_db_session(db_path) as db:
        query = (
            db.query(TranscriptChapter, Job, Book, SynthesisBatch)
            .join(Job, Job.id == TranscriptChapter.job_id)
            .join(Book, Book.id == Job.book_id)
            .outerjoin(SynthesisBatch, SynthesisBatch.synthesis_batch_id == TranscriptChapter.synthesis_batch_id)
            .filter(TranscriptChapter.sequence_number.isnot(None), Job.archived_at.is_(None))
            .order_by(Job.id, TranscriptChapter.sequence_number)
        )
        for chapter, job, book, batch in query:
            status = chapter.synthesis_status or "pending"
            summary = books.setdefault(job.id, {"title": book.title or "", "chapters": 0, "counts": {}, "paused": 0, "seconds": 0.0})
            summary["chapters"] += 1
            summary["counts"][status] = summary["counts"].get(status, 0) + 1
            summary["seconds"] += chapter.chapter_audio_seconds or 0.0
            if batch is None:
                continue
            paused = batch.paused_at is not None
            summary["paused"] += int(paused and status != "done")
            rows.append(QueueRow(
                batch_id=batch.synthesis_batch_id, job_id=job.id, book_title=book.title or "",
                chapter_db_id=chapter.id, chapter_id=chapter.chapter_id, chapter_name=chapter.chapter_name or chapter.title or chapter.chapter_id,
                sequence=chapter.sequence_number or "", status=status, paused=paused,
                created_at=batch.created_at, started_at=batch.synth_started_at, finished_at=batch.synth_finished_at,
                error=batch.synth_error, failed_line=batch.failed_line_index, audio_seconds=chapter.chapter_audio_seconds,
            ))
    summaries = {
        job_id: BookQueueSummary(job_id, s["title"], s["chapters"], dict(s["counts"]), 0, s["paused"], s["seconds"])
        for job_id, s in books.items()
    }
    return rows, summaries


def read_queue(db_path: Path, *, recent: int = 10) -> QueueOverview:
    from dataclasses import replace

    from syntrive.services.synthesis_service import list_schedulable_books

    rows, summaries = _rows(db_path)
    for book in list_schedulable_books(db_path):
        if book.job_id in summaries:
            summaries[book.job_id] = replace(summaries[book.job_id], schedulable=len(book.chapters))

    open_rows = [r for r in rows if r.status != "done"]
    queue = tuple(sorted(open_rows, key=lambda r: (r.paused, r.batch_id)))
    done = sorted((r for r in rows if r.status == "done"), key=lambda r: (r.finished_at or datetime.min), reverse=True)

    from syntrive.services.batch_runner import runner_process

    runner = live_runner(db_path)
    progress = summarize(read_event_tail(db_path.parent))
    live = runner is not None and progress is not None and (runner.batch_id is None or progress.batch_id == runner.batch_id)
    overview = QueueOverview(
        runner=runner, progress=progress, live=live, queue=queue, recent=tuple(done[:recent]),
        books=tuple(summaries[k] for k in sorted(summaries)), process=runner_process(db_path.parent),
    )
    logger.debug(
        "queue_overview_read: runner=%s live=%s queued=%d paused=%d stuck=%d books=%d",
        runner.operation if runner else None, live, overview.runnable_count,
        len(queue) - overview.runnable_count, len(overview.stuck), len(overview.books),
    )
    return overview


@dataclass(frozen=True)
class QueueChapter:
    chapter_db_id: int
    sequence: str
    title: str
    lines: Optional[int]
    status: str
    batch_id: Optional[int]
    paused: bool
    reason: Optional[str]
    audio_seconds: Optional[float]
    error: Optional[str]
    failed_line: Optional[int]

    @property
    def ready(self) -> bool:
        return self.reason is None

    @property
    def action(self) -> Optional[str]:
        if self.ready:
            return "enqueue"
        if self.batch_id is not None and self.status != "done":
            return "resume" if self.paused else "pause"
        return None


@dataclass(frozen=True)
class BookQueue:
    job_id: int
    title: str
    book_reason: Optional[str]
    chapters: tuple[QueueChapter, ...]

    def ready_ids(self, limit: Optional[int] = None) -> tuple[int, ...]:
        ids = tuple(c.chapter_db_id for c in self.chapters if c.ready)
        return ids if limit is None else ids[:limit]


def read_book_queue(db_path: Path, job_id: int) -> Optional[BookQueue]:
    from syntrive.db.models import Book, Job, SynthesisBatch, TranscriptChapter
    from syntrive.db.path_utils import resolve_abs
    from syntrive.db.session import get_db_session
    from syntrive.orchestration.jobs import book_readiness, chapter_readiness

    with get_db_session(db_path) as db:
        job = db.get(Job, job_id)
        if job is None:
            return None
        book = db.get(Book, job.book_id)
        chapters = (
            db.query(TranscriptChapter).filter(TranscriptChapter.job_id == job_id)
            .order_by(TranscriptChapter.sequence_number.asc()).all()
        )
        batch_ids = {c.synthesis_batch_id for c in chapters if c.synthesis_batch_id is not None}
        batches = {b.synthesis_batch_id: b for b in db.query(SynthesisBatch).filter(SynthesisBatch.synthesis_batch_id.in_(batch_ids))} if batch_ids else {}
        book_reason = book_readiness(job, chapters)
        process_dir = resolve_abs(db_path, job.process_dir)

        def row(c) -> QueueChapter:
            batch = batches.get(c.synthesis_batch_id)
            reason = chapter_readiness(c, process_dir)
            if reason is None and book_reason is not None:
                reason = book_reason
            return QueueChapter(
                chapter_db_id=c.id, sequence=c.sequence_number or "", title=c.chapter_name or c.title or c.chapter_id,
                lines=c.transcript_lines, status=c.synthesis_status or "pending", batch_id=c.synthesis_batch_id,
                paused=bool(batch is not None and batch.paused_at is not None), reason=reason,
                audio_seconds=c.chapter_audio_seconds, error=batch.synth_error if batch is not None else None,
                failed_line=batch.failed_line_index if batch is not None else None,
            )

        result = BookQueue(job_id, (book.title if book else "") or "", book_reason,
                           tuple(row(c) for c in chapters if c.sequence_number))
    logger.debug("book_queue_read: job_id=%d chapters=%d ready=%d book_reason=%s",
                 job_id, len(result.chapters), len(result.ready_ids()), book_reason)
    return result


def open_batch_count(db_path: Path) -> int:
    from sqlalchemy import func

    from syntrive.db.models import TranscriptChapter
    from syntrive.db.session import get_db_session

    with get_db_session(db_path) as db:
        return int(
            db.query(func.count(func.distinct(TranscriptChapter.synthesis_batch_id)))
            .filter(TranscriptChapter.synthesis_batch_id.isnot(None), TranscriptChapter.synthesis_status != "done")
            .scalar() or 0
        )


@dataclass(frozen=True)
class NowState:
    runner: Optional[RunnerInfo]
    process: Optional[object]
    progress: Optional[LiveProgress]
    live: bool

    @property
    def running(self) -> bool:
        return self.runner is not None or bool(self.process is not None and self.process.alive)


def read_now(db_path: Path) -> NowState:
    from syntrive.services.batch_runner import runner_process

    runner = live_runner(db_path)
    progress = summarize(read_event_tail(db_path.parent))
    live = runner is not None and progress is not None and (runner.batch_id is None or progress.batch_id == runner.batch_id)
    return NowState(runner, runner_process(db_path.parent), progress, live)


@dataclass(frozen=True)
class HistoryRow:
    batch_id: int
    job_id: int
    book_title: str
    chapters: tuple[str, ...]
    started_at: Optional[datetime]
    finished_at: Optional[datetime]
    audio_seconds: float
    outcome: str
    error: Optional[str]
    failed_line: Optional[int]

    @property
    def run_seconds(self) -> Optional[float]:
        if self.started_at is None or self.finished_at is None:
            return None
        return (self.finished_at - self.started_at).total_seconds()


def history_outcome(*, finished: bool, all_done: bool, error: Optional[str], running: bool) -> str:
    if running:
        return "running"
    if error:
        return "failed"
    return "done" if finished and all_done else "stopped"


def read_history(db_path: Path, job_id: Optional[int] = None, limit: int = 50) -> tuple[HistoryRow, ...]:
    from sqlalchemy import func

    from syntrive.db.models import Book, Job, SynthesisBatch, TranscriptChapter
    from syntrive.db.session import get_db_session

    runner = live_runner(db_path)
    with get_db_session(db_path) as db:
        query = (
            db.query(SynthesisBatch, TranscriptChapter, Job, Book)
            .join(TranscriptChapter, TranscriptChapter.synthesis_batch_id == SynthesisBatch.synthesis_batch_id)
            .join(Job, Job.id == TranscriptChapter.job_id)
            .join(Book, Book.id == Job.book_id)
            .filter(SynthesisBatch.synth_started_at.isnot(None))
        )
        if job_id is not None:
            query = query.filter(Job.id == job_id)
        grouped: dict[int, dict] = {}
        order = func.coalesce(SynthesisBatch.synth_finished_at, SynthesisBatch.synth_started_at).desc()
        for batch, chapter, job, book in query.order_by(order, TranscriptChapter.sequence_number):
            entry = grouped.setdefault(batch.synthesis_batch_id, {"batch": batch, "job": job, "book": book, "chapters": []})
            entry["chapters"].append(chapter)
        rows = []
        for batch_id, entry in list(grouped.items())[:limit]:
            batch, chapters = entry["batch"], entry["chapters"]
            rows.append(HistoryRow(
                batch_id=batch_id, job_id=entry["job"].id, book_title=entry["book"].title or "",
                chapters=tuple(f"{c.sequence_number or ''} {c.chapter_name or c.title or c.chapter_id}".strip() for c in chapters),
                started_at=batch.synth_started_at, finished_at=batch.synth_finished_at,
                audio_seconds=sum(c.chapter_audio_seconds or 0.0 for c in chapters),
                outcome=history_outcome(
                    finished=batch.synth_finished_at is not None,
                    all_done=all(c.synthesis_status == "done" for c in chapters),
                    error=batch.synth_error, running=runner is not None and runner.batch_id == batch_id,
                ),
                error=batch.synth_error, failed_line=batch.failed_line_index,
            ))
    logger.debug("queue_history_read: job_id=%s rows=%d", job_id, len(rows))
    return tuple(rows)
