from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Optional

from syntrive.workflow.engine import MERGE_MODE_LABELS, WorkflowEngine, WorkflowStep

logger = logging.getLogger(__name__)

MERGE_MODES: tuple[str, ...] = tuple(MERGE_MODE_LABELS)
_TRANSCRIPT_FILE_RE = re.compile(r"^ch_\d{4}\.txt$")
_CHAPTER_ID_RE = re.compile(r"^ch_\d{4}$")
_MARKER_RE = re.compile(r"‡([^‡\s]+)‡")
_PREVIEW_CHARS = 20_000


class DecisionError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class DecisionResult:
    changed: bool
    rerun_from: Optional[WorkflowStep]


class _Unchanged:
    def __repr__(self) -> str:
        return "UNCHANGED"


UNCHANGED = _Unchanged()


def _job(db_path: Path, job_id: int):
    from syntrive.services.pipeline_service import load_job

    job = load_job(db_path, job_id)
    if job is None:
        raise DecisionError("not_found", f"Job {job_id} not found.")
    return job


def _process_dir(db_path: Path, job) -> Path:
    from syntrive.db.path_utils import resolve_abs

    return resolve_abs(db_path, job.process_dir)


def _merged_toc(db_path: Path, job) -> Path:
    return _process_dir(db_path, job) / "transcript_html" / "merged" / "0_toc.md"


def _rerun_from(db_path: Path, job, step: WorkflowStep, changed: bool) -> Optional[WorkflowStep]:
    if not changed:
        return None
    return step if WorkflowEngine(job=job, db_path=db_path).check_artifacts_exist(step) else None


def _saved(what: str, job_id: int, holder_kind: str, result: DecisionResult, detail: object) -> DecisionResult:
    logger.info(
        "step_decision_saved: job_id=%d what=%s holder=%s changed=%s rerun_from=%s detail=%s",
        job_id, what, holder_kind, result.changed, result.rerun_from.value if result.rerun_from else None, detail,
    )
    return result


@dataclass(frozen=True)
class MergeSettings:
    override: Optional[str]
    detected: Optional[str]
    choices: tuple[str, ...]
    reset_per_volume: bool


def read_merge_settings(db_path: Path, job_id: int) -> MergeSettings:
    from syntrive.adapters.epub.merged_toc_writer import MergedTocReader

    job = _job(db_path, job_id)
    toc = _merged_toc(db_path, job)
    detected = (MergedTocReader().read(toc)[0].merge_mode or None) if toc.is_file() else None
    return MergeSettings(
        override=job.merge_mode_override,
        detected=detected,
        choices=MERGE_MODES,
        reset_per_volume=bool(job.chapter_number_reset_per_volume),
    )


def save_merge_settings(
    db_path: Path,
    job_id: int,
    *,
    override: object = UNCHANGED,
    reset_per_volume: object = UNCHANGED,
    holder_kind: str,
) -> DecisionResult:
    from syntrive.db.models import Job
    from syntrive.db.session import get_db_session
    from syntrive.services.job_lease import hold_job_lease

    if override is not UNCHANGED and override is not None and override not in MERGE_MODES:
        raise DecisionError("invalid_merge_mode", f"Unknown merge mode {override!r}; expected one of {MERGE_MODES}.")
    job = _job(db_path, job_id)

    with hold_job_lease(db_path, job_id, holder_kind=holder_kind, operation="decide:merge_settings"):
        with get_db_session(db_path) as db:
            row = db.get(Job, job_id)
            before = (row.merge_mode_override, bool(row.chapter_number_reset_per_volume))
            if override is not UNCHANGED:
                row.merge_mode_override = override
            if reset_per_volume is not UNCHANGED:
                row.chapter_number_reset_per_volume = bool(reset_per_volume)
            after = (row.merge_mode_override, bool(row.chapter_number_reset_per_volume))
    changed = before != after
    result = DecisionResult(changed, _rerun_from(db_path, job, WorkflowStep.TRANSCRIPT_MERGE, changed))
    return _saved("merge_settings", job_id, holder_kind, result, {"override": after[0], "reset_per_volume": after[1]})


@dataclass(frozen=True)
class ChapterChoice:
    chapter_id: str
    title: str
    heading_level: int
    volume: str
    excluded: bool
    size_bytes: int
    char_count: int
    source_files: tuple[str, ...]


@dataclass(frozen=True)
class ChapterSelection:
    available: bool
    chapters: tuple[ChapterChoice, ...]
    orphans: tuple[ChapterChoice, ...]

    @property
    def excluded_count(self) -> int:
        return sum(1 for c in self.chapters if c.excluded)


def _choice(entry, merged_dir: Path) -> ChapterChoice:
    from syntrive.adapters.toc_audit import file_char_count, file_size_bytes

    html = merged_dir / f"{entry.chapter_id}.html"
    exists = html.is_file()
    return ChapterChoice(
        chapter_id=entry.chapter_id,
        title=entry.title,
        heading_level=entry.heading_level,
        volume=entry.volume_name,
        excluded=entry.excluded,
        size_bytes=file_size_bytes(html) if exists else 0,
        char_count=file_char_count(html, strip_html=True) if exists else 0,
        source_files=tuple(entry.source_files),
    )


def read_chapter_selection(db_path: Path, job_id: int) -> ChapterSelection:
    from syntrive.adapters.epub.merged_toc_writer import MergedTocReader

    toc = _merged_toc(db_path, _job(db_path, job_id))
    if not toc.is_file():
        return ChapterSelection(available=False, chapters=(), orphans=())
    _, entries = MergedTocReader().read(toc)
    return ChapterSelection(
        available=True,
        chapters=tuple(_choice(e, toc.parent) for e in entries if not e.is_discard),
        orphans=tuple(_choice(e, toc.parent) for e in entries if e.is_discard),
    )


def read_chapter_text(db_path: Path, job_id: int, chapter_id: str, *, limit: int = 4_000) -> tuple[str, bool]:
    from bs4 import BeautifulSoup

    if not _CHAPTER_ID_RE.match(chapter_id):
        raise DecisionError("unknown_chapter", f"Not a chapter id: {chapter_id!r}.")
    path = _merged_toc(db_path, _job(db_path, job_id)).parent / f"{chapter_id}.html"
    if not path.is_file():
        raise DecisionError("unknown_chapter", f"merged/{chapter_id}.html does not exist.")
    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="replace"), "html.parser")
    lines = [line.strip() for line in soup.get_text("\n").splitlines() if line.strip()]
    text = "\n".join(lines)
    return text[:limit], len(text) > limit


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_bytes(text.encode("utf-8"))
    os.replace(tmp, path)


def save_exclusions(db_path: Path, job_id: int, excluded: Iterable[str], *, holder_kind: str) -> DecisionResult:
    from syntrive.adapters.epub.merged_toc_writer import set_excluded_flags
    from syntrive.services.job_lease import hold_job_lease

    job = _job(db_path, job_id)
    toc = _merged_toc(db_path, job)
    if not toc.is_file():
        raise DecisionError("not_merged", "Chapters can be chosen after “Merge chapters” has run.")
    wanted = frozenset(excluded)
    known = {c.chapter_id for c in read_chapter_selection(db_path, job_id).chapters}
    unknown = sorted(wanted - known)
    if unknown:
        raise DecisionError("unknown_chapter", f"Not a chapter of merged/0_toc.md: {', '.join(unknown)}")

    with hold_job_lease(db_path, job_id, holder_kind=holder_kind, operation="decide:exclusions"):
        with toc.open(encoding="utf-8", newline="") as fh:
            text, changed_ids = set_excluded_flags(fh.read(), wanted)
        if changed_ids:
            _atomic_write(toc, text)
    result = DecisionResult(bool(changed_ids), _rerun_from(db_path, job, WorkflowStep.TRANSCRIPT_CLEAN, bool(changed_ids)))
    return _saved("exclusions", job_id, holder_kind, result, {"excluded": len(wanted), "flipped": list(changed_ids)})


@dataclass(frozen=True)
class RuleChoice:
    name: str
    label: str
    description: str
    enabled: bool
    enabled_by_default: bool
    sort_order: int


@dataclass(frozen=True)
class CleanResult:
    chapter_id: str
    title: str
    raw_chars: Optional[int]
    cleaned_chars: Optional[int]
    deletion_ratio: Optional[float]
    threshold_blocked: bool


def read_cleaning_rules(db_path: Path, job_id: int) -> tuple[RuleChoice, ...]:
    from syntrive.db.models import CleaningRule, JobCleaningRuleConfig
    from syntrive.db.session import get_db_session

    _job(db_path, job_id)
    with get_db_session(db_path) as db:
        overrides = {c.rule_name: c.enabled for c in db.query(JobCleaningRuleConfig).filter_by(job_id=job_id)}
        return tuple(
            RuleChoice(
                name=r.name, label=r.display_name or r.name, description=r.description or "",
                enabled=overrides.get(r.name, r.enabled_by_default), enabled_by_default=bool(r.enabled_by_default),
                sort_order=r.sort_order,
            )
            for r in db.query(CleaningRule).order_by(CleaningRule.sort_order)
        )


def read_clean_results(db_path: Path, job_id: int) -> tuple[CleanResult, ...]:
    from syntrive.db.models import TranscriptChapter
    from syntrive.db.session import get_db_session

    with get_db_session(db_path) as db:
        rows = (
            db.query(TranscriptChapter)
            .filter(TranscriptChapter.job_id == job_id, TranscriptChapter.cleaned_chars.isnot(None))
            .order_by(TranscriptChapter.chapter_id)
        )
        return tuple(
            CleanResult(
                chapter_id=r.chapter_id, title=r.chapter_name or r.title or r.chapter_id,
                raw_chars=r.raw_chars, cleaned_chars=r.cleaned_chars, deletion_ratio=r.deletion_ratio,
                threshold_blocked=bool(r.threshold_blocked),
            )
            for r in rows
        )


def save_cleaning_rules(db_path: Path, job_id: int, enabled: Mapping[str, bool], *, holder_kind: str) -> DecisionResult:
    from syntrive.db.models import JobCleaningRuleConfig
    from syntrive.db.session import get_db_session
    from syntrive.services.job_lease import hold_job_lease

    job = _job(db_path, job_id)
    current = {r.name: r.enabled for r in read_cleaning_rules(db_path, job_id)}
    unknown = sorted(set(enabled) - set(current))
    if unknown:
        raise DecisionError("unknown_rule", f"Unknown cleaning rule(s): {', '.join(unknown)}")
    flips = {name: bool(on) for name, on in enabled.items() if current[name] != bool(on)}

    if flips:
        with hold_job_lease(db_path, job_id, holder_kind=holder_kind, operation="decide:cleaning_rules"):
            with get_db_session(db_path) as db:
                rows = {c.rule_name: c for c in db.query(JobCleaningRuleConfig).filter_by(job_id=job_id)}
                for name, on in flips.items():
                    if name in rows:
                        rows[name].enabled = on
                    else:
                        db.add(JobCleaningRuleConfig(job_id=job_id, rule_name=name, enabled=on))
    result = DecisionResult(bool(flips), _rerun_from(db_path, job, WorkflowStep.TRANSCRIPT_CLEAN, bool(flips)))
    return _saved("cleaning_rules", job_id, holder_kind, result, flips)


@dataclass(frozen=True)
class TranscriptFile:
    file_name: str
    title: str
    volume: str
    lines: int
    exists: bool


@dataclass(frozen=True)
class TranscriptFormat:
    name: str
    available: bool
    files: tuple[TranscriptFile, ...]


@dataclass(frozen=True)
class TranscriptReview:
    formats: tuple[TranscriptFormat, ...]
    selected: Optional[str]
    default: str


def _text_dir(db_path: Path, job, fmt: str) -> Path:
    return _process_dir(db_path, job) / "transcript_text" / fmt


def read_transcript_review(db_path: Path, job_id: int) -> TranscriptReview:
    from syntrive.adapters.text.toc_writer import TextTocReader
    from syntrive.pipeline.text_extraction_stage import (
        DEFAULT_TRANSCRIPT_PATH_SELECTION,
        TRANSCRIPT_PATH_CHOICES,
        _count_transcript_lines,
        _output_id,
    )

    job = _job(db_path, job_id)

    def one(fmt: str) -> TranscriptFormat:
        folder = _text_dir(db_path, job, fmt)
        toc = folder / "0_toc.md"
        if not toc.is_file():
            return TranscriptFormat(fmt, False, ())
        files = []
        for entry in TextTocReader().read(toc):
            if entry.excluded:
                continue
            path = folder / f"{_output_id(entry)}.txt"
            files.append(TranscriptFile(
                file_name=path.name, title=entry.title, volume=entry.volume_name,
                lines=_count_transcript_lines(path) if path.is_file() else 0, exists=path.is_file(),
            ))
        return TranscriptFormat(fmt, True, tuple(files))

    transcript_dir = (job.transcript_dir or "").rstrip("/")
    selected = next((f for f in TRANSCRIPT_PATH_CHOICES if transcript_dir.endswith(f"/{f}")), None)
    return TranscriptReview(tuple(one(f) for f in TRANSCRIPT_PATH_CHOICES), selected, DEFAULT_TRANSCRIPT_PATH_SELECTION)


@dataclass(frozen=True)
class TextPiece:
    text: str
    marker: bool


def split_markers(text: str) -> tuple[tuple[TextPiece, ...], ...]:
    lines = []
    for line in text.splitlines():
        if not line.strip():
            continue
        pieces, pos = [], 0
        for match in _MARKER_RE.finditer(line):
            if match.start() > pos:
                pieces.append(TextPiece(line[pos:match.start()], False))
            pieces.append(TextPiece(match.group(1), True))
            pos = match.end()
        if pos < len(line):
            pieces.append(TextPiece(line[pos:], False))
        lines.append(tuple(pieces))
    return tuple(lines)


def read_transcript_text(db_path: Path, job_id: int, fmt: str, file_name: str, *, limit: int = _PREVIEW_CHARS) -> tuple[str, bool]:
    from syntrive.pipeline.text_extraction_stage import TRANSCRIPT_PATH_CHOICES

    if fmt not in TRANSCRIPT_PATH_CHOICES:
        raise DecisionError("unknown_format", f"Unknown transcript format {fmt!r}.")
    if not _TRANSCRIPT_FILE_RE.match(file_name):
        raise DecisionError("unknown_file", f"Not a transcript file name: {file_name!r}.")
    path = _text_dir(db_path, _job(db_path, job_id), fmt) / file_name
    if not path.is_file():
        raise DecisionError("unknown_file", f"{fmt}/{file_name} does not exist.")
    with path.open(encoding="utf-8", errors="replace") as fh:
        text = fh.read(limit + 1)
    return text[:limit], len(text) > limit
