from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Optional

from syntrive.workflow.engine import (
    _ORDERED_STEPS,
    StepAction,
    WorkflowEngine,
    WorkflowStep,
    step_from_db_value,
)

logger = logging.getLogger(__name__)


class StepMode(str, Enum):
    RUN = "run"
    TUI = "tui"
    QUEUE = "queue"


PIPELINE_STEPS: tuple[WorkflowStep, ...] = tuple(s for s in _ORDERED_STEPS if s != WorkflowStep.DONE)

STEP_MODES: dict[WorkflowStep, StepMode] = {
    WorkflowStep.BOOTSTRAP: StepMode.RUN,
    WorkflowStep.TRANSCRIPT_EXTRACT: StepMode.RUN,
    WorkflowStep.TRANSCRIPT_MERGE: StepMode.RUN,
    WorkflowStep.TRANSCRIPT_CLEAN: StepMode.RUN,
    WorkflowStep.TRANSCRIPT_TEXT: StepMode.RUN,
    WorkflowStep.TRANSCRIPT_REVIEW: StepMode.TUI,
    WorkflowStep.TTS_CONFIG: StepMode.RUN,
    WorkflowStep.SYNTHESIS: StepMode.QUEUE,
    WorkflowStep.COMBINE: StepMode.TUI,
}


class StepState(str, Enum):
    DONE = "done"
    CURRENT = "current"
    LATER = "later"


@dataclass(frozen=True)
class PipelineStep:
    step: WorkflowStep
    number: int
    state: StepState
    mode: StepMode


@dataclass(frozen=True)
class PipelineView:
    steps: tuple[PipelineStep, ...]
    current: Optional[WorkflowStep]

    @property
    def finished(self) -> bool:
        return self.current is None

    @property
    def current_number(self) -> int:
        if self.current is None:
            return len(self.steps)
        return PIPELINE_STEPS.index(self.current) + 1

    def get(self, step: WorkflowStep) -> Optional[PipelineStep]:
        return next((s for s in self.steps if s.step == step), None)


def build_pipeline_view(current_step: Optional[str], status: Optional[str]) -> PipelineView:
    resolved = step_from_db_value(current_step)
    finished = resolved == WorkflowStep.DONE or status == "completed"
    current_idx = len(PIPELINE_STEPS) if finished else PIPELINE_STEPS.index(resolved)

    def state_of(idx: int) -> StepState:
        if idx < current_idx:
            return StepState.DONE
        return StepState.CURRENT if idx == current_idx else StepState.LATER

    steps = tuple(
        PipelineStep(step=s, number=i + 1, state=state_of(i), mode=STEP_MODES[s])
        for i, s in enumerate(PIPELINE_STEPS)
    )
    return PipelineView(steps=steps, current=None if finished else resolved)


class OnExisting(str, Enum):
    ASK = "ask"
    OVERWRITE = "overwrite"
    KEEP = "keep"


class RunDisposition(str, Enum):
    DONE = "done"
    KEPT = "kept"
    NEEDS_CONFIRM = "needs_confirm"
    REFUSED = "refused"
    FAILED = "failed"


@dataclass(frozen=True)
class StepRunResult:
    job_id: int
    step: WorkflowStep
    disposition: RunDisposition
    artifacts: dict = field(default_factory=dict)
    notes: Optional[str] = None
    error: Optional[str] = None
    existing_desc: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.disposition in (RunDisposition.DONE, RunDisposition.KEPT)


class StepRunError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class StepOptions:
    text_formats: Optional[frozenset[str]] = None
    text_primary: Optional[str] = None
    transcript_path: Optional[str] = None


DECISION_OPTIONS: dict[WorkflowStep, str] = {WorkflowStep.TRANSCRIPT_REVIEW: "transcript_path"}


def check_options(step: WorkflowStep, options: StepOptions) -> None:
    from syntrive.pipeline.text_extraction_stage import TEXT_EXTRACTION_FORMATS, TRANSCRIPT_PATH_CHOICES

    def bad(message: str) -> StepRunError:
        return StepRunError("bad_options", message)

    text_given = options.text_formats is not None or options.text_primary is not None
    if text_given and step != WorkflowStep.TRANSCRIPT_TEXT:
        raise bad("Text formats apply to 'transcript_text' only.")
    if options.transcript_path is not None and step != WorkflowStep.TRANSCRIPT_REVIEW:
        raise bad("A transcript path applies to 'transcript_review' only.")
    formats = options.text_formats if options.text_formats is not None else frozenset(TEXT_EXTRACTION_FORMATS)
    if not formats or not formats <= set(TEXT_EXTRACTION_FORMATS):
        raise bad(f"Text formats must be a non-empty subset of {TEXT_EXTRACTION_FORMATS}.")
    if options.text_primary is not None and options.text_primary not in formats:
        raise bad("The primary text format must be one of the generated formats.")
    if options.transcript_path is not None and options.transcript_path not in TRANSCRIPT_PATH_CHOICES:
        raise bad(f"Transcript path must be one of {TRANSCRIPT_PATH_CHOICES}.")


def _apply_options(engine: WorkflowEngine, options: StepOptions) -> None:
    from syntrive.pipeline.text_extraction_stage import TEXT_EXTRACTION_FORMATS

    if options.text_formats is not None:
        for name in TEXT_EXTRACTION_FORMATS:
            engine.set_text_extraction_format(name, name in options.text_formats)
    if options.text_primary is not None:
        engine.set_text_extraction_primary(options.text_primary)
    if options.transcript_path is not None:
        engine.set_transcript_path_selection(options.transcript_path)


def load_job(db_path: Path, job_id: int):
    from sqlalchemy.orm import joinedload

    from syntrive.db.models import Job
    from syntrive.db.session import get_db_session

    with get_db_session(db_path) as db:
        job = db.query(Job).options(joinedload(Job.book)).filter(Job.id == job_id).first()
        if job is not None:
            db.expunge_all()
        return job


def check_runnable(job, step: WorkflowStep, options: StepOptions = StepOptions()) -> None:
    if job is None:
        raise StepRunError("not_found", "Job not found.")
    if job.archived_at is not None:
        raise StepRunError("archived", "This book is archived and read-only. Restore it first.")
    view = build_pipeline_view(job.current_step, job.status)
    if view.current != step:
        raise StepRunError("not_current", f"'{step.value}' is not the current step. Restart from it first.")
    decision_option = DECISION_OPTIONS.get(step)
    decided = decision_option is not None and getattr(options, decision_option) is not None
    if STEP_MODES[step] != StepMode.RUN and not decided:
        raise StepRunError("needs_decision", f"'{step.value}' needs a decision this entry point cannot take yet.")
    check_options(step, options)


def existing_output(db_path: Path, job_id: int, step: WorkflowStep) -> Optional[str]:
    job = load_job(db_path, job_id)
    if job is None:
        return None
    engine = WorkflowEngine(job=job, db_path=db_path)
    return engine.artifact_description(step) if engine.check_artifacts_exist(step) else None


def run_step(
    db_path: Path,
    job_id: int,
    step: WorkflowStep,
    *,
    on_existing: OnExisting = OnExisting.ASK,
    holder_kind: str = "webui",
    options: StepOptions = StepOptions(),
) -> StepRunResult:
    job = load_job(db_path, job_id)
    try:
        check_runnable(job, step, options)
    except StepRunError as exc:
        logger.info("step_run_refused: job_id=%d step=%s code=%s", job_id, step.value, exc.code)
        return StepRunResult(job_id, step, RunDisposition.REFUSED, error=str(exc))

    engine = WorkflowEngine(job=job, db_path=db_path, holder_kind=holder_kind)
    _apply_options(engine, options)

    if on_existing == OnExisting.KEEP and engine.check_artifacts_exist(step):
        engine.record_step(step, StepAction.SKIPPED, notes=f"{holder_kind}: kept existing output")
        engine.advance_job_step(step)
        logger.info("step_run_kept: job_id=%d step=%s", job_id, step.value)
        return StepRunResult(job_id, step, RunDisposition.KEPT)

    outcome = engine.execute_step(step, force_overwrite=on_existing == OnExisting.OVERWRITE)
    if outcome.needs_overwrite_confirm:
        return StepRunResult(
            job_id, step, RunDisposition.NEEDS_CONFIRM, existing_desc=outcome.existing_artifact_desc,
        )
    if not outcome.success:
        logger.warning("step_run_failed: job_id=%d step=%s error=%s", job_id, step.value, outcome.error)
        return StepRunResult(job_id, step, RunDisposition.FAILED, error=outcome.error)

    engine.record_step(step, StepAction.CONFIRMED, artifacts_summary=outcome.artifacts)
    engine.advance_job_step(step)
    logger.info("step_run_done: job_id=%d step=%s artifacts=%s", job_id, step.value, outcome.artifacts)
    return StepRunResult(job_id, step, RunDisposition.DONE, artifacts=dict(outcome.artifacts or {}), notes=outcome.notes)


@dataclass(frozen=True)
class StepRun:
    job_id: int
    step: WorkflowStep
    started_at: float
    finished_at: Optional[float] = None
    result: Optional[StepRunResult] = None

    @property
    def running(self) -> bool:
        return self.finished_at is None

    @property
    def seconds(self) -> float:
        return (self.finished_at or time.time()) - self.started_at


class RunAlreadyActive(Exception):
    pass


Runner = Callable[..., StepRunResult]


class StepRunRegistry:
    def __init__(self, runner: Runner = run_step) -> None:
        self._runner = runner
        self._lock = threading.Lock()
        self._runs: dict[tuple[str, int], StepRun] = {}

    @staticmethod
    def _key(db_path: Path, job_id: int) -> tuple[str, int]:
        return (str(Path(db_path).resolve()), job_id)

    def get(self, db_path: Path, job_id: int) -> Optional[StepRun]:
        with self._lock:
            return self._runs.get(self._key(db_path, job_id))

    def any_running(self) -> bool:
        with self._lock:
            return any(run.running for run in self._runs.values())

    def start(
        self, db_path: Path, job_id: int, step: WorkflowStep, *,
        on_existing: OnExisting, holder_kind: str, options: StepOptions = StepOptions(),
    ) -> StepRun:
        key = self._key(db_path, job_id)
        with self._lock:
            current = self._runs.get(key)
            if current is not None and current.running:
                raise RunAlreadyActive(f"'{current.step.value}' is already running for this book.")
            run = StepRun(job_id=job_id, step=step, started_at=time.time())
            self._runs[key] = run

        def work() -> None:
            try:
                result = self._runner(db_path, job_id, step, on_existing=on_existing, holder_kind=holder_kind, options=options)
            except Exception as exc:
                logger.error("step_run_crashed: job_id=%d step=%s", job_id, step.value, exc_info=True)
                result = StepRunResult(job_id, step, RunDisposition.FAILED, error=str(exc))
            with self._lock:
                self._runs[key] = StepRun(job_id, step, run.started_at, time.time(), result)
            logger.info(
                "step_run_finished: job_id=%d step=%s disposition=%s seconds=%.1f",
                job_id, step.value, result.disposition.value, time.time() - run.started_at,
            )

        logger.info(
            "step_run_started: job_id=%d step=%s on_existing=%s holder=%s options=%s",
            job_id, step.value, on_existing.value, holder_kind, options,
        )
        threading.Thread(target=work, name=f"step-run-{job_id}-{step.value}", daemon=True).start()
        return run
