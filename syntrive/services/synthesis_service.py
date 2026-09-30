from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

from syntrive.db.models import SynthesisBatch, TranscriptChapter
from syntrive.db.session import get_db_session
from syntrive.orchestration import jobs as jobs_module
from syntrive.orchestration import workflow as workflow_module
from syntrive.orchestration.jobs import BatchStatusReport, PausableBook, SchedulableBook
from syntrive.orchestration.workflow import BatchRunResult, EnqueueResult, PauseResult
from syntrive.services.job_lease import JobLeaseConflict, find_blocking_lease, hold_job_leases

logger = logging.getLogger(__name__)


def list_schedulable_books(db_path: Path) -> List[SchedulableBook]:
    with get_db_session(db_path) as db:
        return jobs_module.schedulable_books(db, db_path.parent)


def run_redo_detection(db_path: Path) -> int:
    repo_dir = db_path.parent
    rolled_back = 0
    with get_db_session(db_path) as db:
        done_chapters = db.query(TranscriptChapter).filter_by(synthesis_status="done").all()
        for chapter in done_chapters:
            if workflow_module.detect_and_rollback_redo(db, chapter, repo_dir):
                rolled_back += 1
    if rolled_back:
        logger.info("run_redo_detection: rolled_back=%d", rolled_back)
    return rolled_back


def enqueue(db_path: Path, chapter_db_ids: List[int], note: Optional[str] = None) -> EnqueueResult:
    with get_db_session(db_path) as db:
        job_ids = [
            row.job_id for row in
            db.query(TranscriptChapter.job_id).filter(TranscriptChapter.id.in_(chapter_db_ids)).distinct()
        ]
    blocked = find_blocking_lease(db_path, job_ids, tolerate_kinds=frozenset({"tts_batch"}))
    if blocked is not None:
        logger.warning("enqueue_refused: job_id=%d reason=job_lease chapters=%d", blocked.job_id, len(chapter_db_ids))
        return EnqueueResult(ok=False, error=str(blocked))
    with get_db_session(db_path) as db:
        return workflow_module.enqueue_chapters(db, chapter_db_ids, note)


def list_pausable_books(db_path: Path) -> List[PausableBook]:
    with get_db_session(db_path) as db:
        return jobs_module.pausable_books(db)


class _PauseInputError(RuntimeError):
    pass


def set_pause_state(
    db_path: Path, *,
    pause_batch_ids: Optional[List[int]] = None,
    resume_batch_ids: Optional[List[int]] = None,
) -> PauseResult:
    pause_ids = list(pause_batch_ids or [])
    resume_ids = list(resume_batch_ids or [])
    if not pause_ids and not resume_ids:
        return PauseResult(ok=False, error="set_pause_state: nothing to pause or resume")

    flipped = 0
    already: list = []
    skipped: list = []
    try:
        with get_db_session(db_path) as db:
            for ids, paused in ((pause_ids, True), (resume_ids, False)):
                if not ids:
                    continue
                result = workflow_module.set_batches_paused(db, ids, paused=paused)
                if not result.ok:
                    logger.warning("set_pause_state: aborting -- %s", result.error)
                    raise _PauseInputError(result.error)
                flipped += result.flipped
                already.extend(result.already_in_state)
                skipped.extend(result.skipped_done)
    except _PauseInputError as exc:
        return PauseResult(ok=False, error=str(exc))

    logger.info(
        "set_pause_state: pause=%s resume=%s flipped=%d already=%s skipped_done=%s",
        pause_ids, resume_ids, flipped, already, skipped,
    )
    return PauseResult(
        ok=True, flipped=flipped,
        already_in_state=tuple(already), skipped_done=tuple(skipped),
    )


def run_next_batch(
    db_path: Path, *, on_event: Optional[Callable[[str, dict], None]] = None,
    progress_plan: Optional[dict] = None,
) -> Optional[BatchRunResult]:
    repo_dir = db_path.parent
    with get_db_session(db_path) as db:
        batch = jobs_module.find_earliest_active_batch(db)
        if batch is None:
            return None
        batch_id = batch.synthesis_batch_id
        return workflow_module.run_batch(
            db, batch_id, repo_dir, on_event=on_event, progress_plan=progress_plan,
        )


def run_one_batch(
    db_path: Path, batch_id: int, *,
    on_event: Optional[Callable[[str, dict], None]] = None,
    progress_plan: Optional[dict] = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> Optional[BatchRunResult]:
    repo_dir = db_path.parent
    with get_db_session(db_path) as db:
        batch_row = (
            db.query(SynthesisBatch)
            .filter_by(synthesis_batch_id=batch_id)
            .first()
        )
        if batch_row is None:
            logger.warning("run_one_batch: unknown batch_id=%s", batch_id)
            return None
        if batch_row.paused_at is not None:
            logger.warning(
                "run_one_batch: batch_id=%s is paused (paused_at=%s) -- running anyway "
                "(explicit --batch-id)", batch_id, batch_row.paused_at,
            )
        logger.info("run_one_batch: batch_id=%s repo=%s", batch_id, repo_dir)
        return workflow_module.run_batch(
            db, batch_id, repo_dir, on_event=on_event, progress_plan=progress_plan,
            should_stop=should_stop,
        )


@dataclass(frozen=True)
class SkippedBatch:
    batch_id: int
    reason: str


@dataclass(frozen=True)
class MultiBatchRunResult:
    ok: bool
    batches_run: int = 0
    chapters_completed: int = 0
    last_result: Optional[BatchRunResult] = None
    failed_results: List[BatchRunResult] = field(default_factory=list)
    batch_not_found: bool = False
    skipped: List[SkippedBatch] = field(default_factory=list)


DEFAULT_MAX_CONSECUTIVE_BATCH_FAILURES = 3
_MAX_CONSECUTIVE_FAILURES_ENV = "SYNTRIVE_MAX_CONSECUTIVE_BATCH_FAILURES"


def _max_consecutive_batch_failures() -> int:
    raw = os.environ.get(_MAX_CONSECUTIVE_FAILURES_ENV)
    try:
        return max(0, int(raw)) if raw else DEFAULT_MAX_CONSECUTIVE_BATCH_FAILURES
    except ValueError:
        logger.warning("synthesis_config_invalid: var=%s value=%r", _MAX_CONSECUTIVE_FAILURES_ENV, raw)
        return DEFAULT_MAX_CONSECUTIVE_BATCH_FAILURES


def _record_batch_crash(db_path: Path, batch_id: int, error: str) -> None:
    try:
        with get_db_session(db_path) as db:
            batch = db.query(SynthesisBatch).filter_by(synthesis_batch_id=batch_id).first()
            if batch is not None:
                batch.synth_error = error[:2000] or "batch crashed"
                batch.failed_line_index = None
        logger.info("run_active_batches_crash_recorded: batch_id=%s error=%s", batch_id, error[:200])
    except Exception:  # noqa: BLE001 -- recording the crash must never crash the queue
        logger.exception("run_active_batches_crash_record_failed: batch_id=%s", batch_id)


def _batch_job_ids(db_path: Path, batch_id: int) -> List[int]:
    with get_db_session(db_path) as db:
        return sorted({chapter.job_id for chapter in jobs_module.batch_chapters(db, batch_id)})


def run_active_batches(
    db_path: Path, *, on_event: Optional[Callable[[str, dict], None]] = None,
    only_batch_id: Optional[int] = None,
) -> MultiBatchRunResult:
    rolled_back = run_redo_detection(db_path)
    logger.info(
        "run_active_batches_redo_preflight: rolled_back=%d only_batch_id=%s",
        rolled_back, only_batch_id,
    )
    with get_db_session(db_path) as db:
        progress_plan = jobs_module.build_progress_plan(db)
        batch_ids = [only_batch_id] if only_batch_id is not None else jobs_module.active_batch_ids(db)

    batches_run = 0
    chapters_completed = 0
    failed_results: List[BatchRunResult] = []
    skipped: List[SkippedBatch] = []
    max_consecutive_failures = _max_consecutive_batch_failures()
    consecutive_failures = 0
    for position, batch_id in enumerate(batch_ids):
        if max_consecutive_failures and consecutive_failures >= max_consecutive_failures:
            logger.error(
                "run_active_batches_circuit_open: consecutive_failures=%d skipped=%d -- stopping; "
                "fix the cause (see the preceding failure reasons) and re-run to resume",
                consecutive_failures, len(batch_ids) - position,
            )
            break
        try:
            job_ids = _batch_job_ids(db_path, batch_id)
            with hold_job_leases(
                db_path, job_ids, holder_kind="tts_batch", operation=f"batch:{batch_id}",
            ) as handles:
                result = run_one_batch(
                    db_path, batch_id, on_event=on_event, progress_plan=progress_plan,
                    should_stop=lambda: any(handle.lost for handle in handles),
                )
            if result is not None and result.stopped:
                logger.error(
                    "run_active_batches_lease_lost: batch_id=%s job_ids=%s chapters_completed=%d -- "
                    "another entry point took over the job lease; stopped before the next line, "
                    "batch left resumable",
                    batch_id, job_ids, result.chapters_completed,
                )
                chapters_completed += result.chapters_completed
                skipped.append(SkippedBatch(batch_id=batch_id, reason=f"job lease lost mid-batch: {result.error}"))
                continue
        except JobLeaseConflict as exc:
            logger.warning(
                "run_active_batches_batch_skipped: batch_id=%s job_id=%s reason=job_lease_conflict "
                "holder=%s:%s operation=%s -- left queued for the next run",
                batch_id, exc.job_id, exc.holder.holder_kind, exc.holder.holder_id[:8], exc.holder.operation,
            )
            skipped.append(SkippedBatch(batch_id=batch_id, reason=str(exc)))
            continue
        except Exception as exc:  # noqa: BLE001 -- queue-level resilience boundary (round 40)
            logger.exception(
                "run_active_batches_batch_crashed: batch_id=%s error=%s -- continuing with the "
                "remaining %d queued batch(es)",
                batch_id, exc, len(batch_ids) - position - 1,
            )
            batches_run += 1
            failed_results.append(BatchRunResult(ok=False, error=str(exc)))
            _record_batch_crash(db_path, batch_id, str(exc))
            consecutive_failures += 1
            continue
        if result is None:
            if only_batch_id is not None:
                logger.warning(
                    "run_active_batches: requested batch_id=%s does not exist", only_batch_id,
                )
                return MultiBatchRunResult(ok=False, batch_not_found=True, skipped=skipped)
            continue
        batches_run += 1
        chapters_completed += result.chapters_completed
        logger.info(
            "run_active_batches: batch_done batches_run=%d chapters_completed=%d ok=%s",
            batches_run, chapters_completed, result.ok,
        )
        consecutive_failures = 0 if result.ok else consecutive_failures + 1
        if not result.ok:
            logger.warning(
                "run_active_batches_batch_failed: batch_id=%s failed_chapter=%s error=%s "
                "-- continuing with the remaining %d queued batch(es)",
                batch_id, result.failed_chapter_id, result.error,
                len(batch_ids) - position - 1,
            )
            failed_results.append(result)

    if chapters_completed or rolled_back:
        from syntrive.services.contract_export_service import refresh_manifests_best_effort

        refresh_manifests_best_effort(db_path, reason="tts_batch")

    if failed_results:
        logger.warning(
            "run_active_batches_done_with_failures: batches_run=%d chapters_completed=%d failed=%d skipped=%d",
            batches_run, chapters_completed, len(failed_results), len(skipped),
        )
        return MultiBatchRunResult(
            ok=False, batches_run=batches_run, chapters_completed=chapters_completed,
            last_result=failed_results[-1], failed_results=failed_results, skipped=skipped,
        )
    logger.info(
        "run_active_batches_done: batches_run=%d chapters_completed=%d skipped=%d ok=True",
        batches_run, chapters_completed, len(skipped),
    )
    return MultiBatchRunResult(
        ok=True, batches_run=batches_run, chapters_completed=chapters_completed, skipped=skipped,
    )


def get_status(db_path: Path, batch_id: Optional[int] = None) -> List[BatchStatusReport]:
    with get_db_session(db_path) as db:
        ids = [batch_id] if batch_id is not None else jobs_module.all_batch_ids(db)
        reports = [jobs_module.batch_status(db, bid) for bid in ids]
        return [r for r in reports if r is not None]
