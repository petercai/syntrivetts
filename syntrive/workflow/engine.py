import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class WorkflowStep(str, Enum):
    BOOTSTRAP = "bootstrap"
    TRANSCRIPT_EXTRACT = "transcript_extract"
    TRANSCRIPT_MERGE = "transcript_merge"
    TRANSCRIPT_CLEAN = "transcript_clean"
    TRANSCRIPT_TEXT = "transcript_text"
    TRANSCRIPT_REVIEW = "transcript_review"
    TTS_CONFIG = "tts_config"
    SYNTHESIS = "synthesis"
    COMBINE = "combine"
    DONE = "done"


class StepAction(str, Enum):
    CONFIRMED = "confirmed"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"
    BACK = "back"
    PAUSED = "paused"
    RESET = "reset"


@dataclass
class StepOutcome:
    success: bool
    artifacts: dict[str, str] = field(default_factory=dict)
    error: Optional[str] = None
    notes: Optional[str] = None
    needs_overwrite_confirm: bool = False
    existing_artifact_desc: Optional[str] = None


_TRANSITIONS: dict[WorkflowStep, list[WorkflowStep]] = {
    WorkflowStep.BOOTSTRAP: [WorkflowStep.TRANSCRIPT_EXTRACT],
    WorkflowStep.TRANSCRIPT_EXTRACT: [WorkflowStep.TRANSCRIPT_MERGE],
    WorkflowStep.TRANSCRIPT_MERGE: [WorkflowStep.TRANSCRIPT_CLEAN],
    WorkflowStep.TRANSCRIPT_CLEAN: [WorkflowStep.TRANSCRIPT_TEXT],
    WorkflowStep.TRANSCRIPT_TEXT: [WorkflowStep.TRANSCRIPT_REVIEW],
    WorkflowStep.TRANSCRIPT_REVIEW: [WorkflowStep.TTS_CONFIG],
    WorkflowStep.TTS_CONFIG: [WorkflowStep.SYNTHESIS],
    WorkflowStep.SYNTHESIS: [WorkflowStep.COMBINE, WorkflowStep.DONE],
    WorkflowStep.COMBINE: [WorkflowStep.DONE],
    WorkflowStep.DONE: [],
}

_STEP_LABELS: dict[WorkflowStep, str] = {
    WorkflowStep.BOOTSTRAP: "Bootstrap",
    WorkflowStep.TRANSCRIPT_EXTRACT: "Spine Extraction to raw/",
    WorkflowStep.TRANSCRIPT_MERGE: "Chapter Merge to merged/",
    WorkflowStep.TRANSCRIPT_CLEAN: "HTML Cleaning & Tagging",
    WorkflowStep.TRANSCRIPT_TEXT: "Transcript Text Extraction (multi-format)",
    WorkflowStep.TRANSCRIPT_REVIEW: "Manual Review, Fix & Normalize",
    WorkflowStep.TTS_CONFIG: "Configure TTS Parameters",
    WorkflowStep.SYNTHESIS: "TTS Synthesis",
    WorkflowStep.COMBINE: "Combine Audiobook",
    WorkflowStep.DONE: "Done",
}

_ORDERED_STEPS = [
    WorkflowStep.BOOTSTRAP,
    WorkflowStep.TRANSCRIPT_EXTRACT,
    WorkflowStep.TRANSCRIPT_MERGE,
    WorkflowStep.TRANSCRIPT_CLEAN,
    WorkflowStep.TRANSCRIPT_TEXT,
    WorkflowStep.TRANSCRIPT_REVIEW,
    WorkflowStep.TTS_CONFIG,
    WorkflowStep.SYNTHESIS,
    WorkflowStep.COMBINE,
    WorkflowStep.DONE,
]

_LEGACY_STEP_VALUES: dict[str, WorkflowStep] = {
    "transcript": WorkflowStep.TRANSCRIPT_EXTRACT,
    "review_html": WorkflowStep.TRANSCRIPT_CLEAN,
    "transcript_normalize": WorkflowStep.TRANSCRIPT_REVIEW,
}


def step_from_db_value(value: Optional[str]) -> WorkflowStep:
    if not value:
        return WorkflowStep.BOOTSTRAP
    if value in _LEGACY_STEP_VALUES:
        return _LEGACY_STEP_VALUES[value]
    try:
        return WorkflowStep(value)
    except ValueError:
        return WorkflowStep.BOOTSTRAP


MERGE_MODE_LABELS: dict[str, str] = {
    "flat": "flat — single file (flat TOC or continuous chapters)",
    "volume_split": "volume_split — one file per volume (chapter numbering resets)",
    "volume_continuous": "volume_continuous — single file despite hierarchical TOC",
}

TEXT_EXTRACTION_FORMAT_LABELS: dict[str, str] = {
    "raw": "Plain text (raw/) — direct HTML→text, no SML tokens",
    "tts_script": "TTS script (tts_script/) — inline ‡break‡/‡pause‡/‡voice:N‡, default TTS input",
}

TRANSCRIPT_PATH_LABELS: dict[str, str] = {
    "raw": "Plain text (raw/) — direct HTML→text, no SML tokens",
    "tts_script": "TTS script (tts_script/) — inline ‡break‡/‡pause‡/‡voice:N‡ (default)",
}


class WorkflowEngine:
    def __init__(
        self,
        job,
        db_path: Path,
        holder_kind: str = "cli",
    ) -> None:
        self._job = job
        self._db_path = db_path
        self._holder_kind = holder_kind
        from syntrive.pipeline.text_extraction_stage import (
            DEFAULT_TEXT_EXTRACTION_FORMATS,
            DEFAULT_TEXT_EXTRACTION_PRIMARY,
            DEFAULT_TRANSCRIPT_PATH_SELECTION,
        )
        self._text_extraction_formats: set[str] = set(DEFAULT_TEXT_EXTRACTION_FORMATS)
        self._text_extraction_primary: str = DEFAULT_TEXT_EXTRACTION_PRIMARY
        self._transcript_path_selection: str = DEFAULT_TRANSCRIPT_PATH_SELECTION


    def run(self, console=None) -> None:
        start_step = self.resolve_start_step()
        current = start_step

        while current != WorkflowStep.DONE:
            action, next_step = self._cli_run_one_step(current, console)

            if action == StepAction.PAUSED:
                logger.info("Progress saved. Resume any time with 'syntrive resume'.")
                return

            if action == StepAction.BACK:
                prev = self.get_previous_step(current)
                if prev is None:
                    logger.warning("Already at the first step.")
                else:
                    current = prev
                continue

            if action in (StepAction.CANCELLED, StepAction.SKIPPED):
                logger.warning(
                    "Step '%s' %s. Workflow paused.", _STEP_LABELS[current], action.value
                )
                return

            if next_step is None:
                logger.info("Progress saved.")
                return

            current = next_step

        logger.info("All pipeline steps complete! Your audiobook is ready.")

    def execute_step(
        self, step: WorkflowStep, force_overwrite: bool = False
    ) -> StepOutcome:
        if not force_overwrite and self.check_artifacts_exist(step):
            desc = self.artifact_description(step)
            logger.info("Step %s: artifacts exist, overwrite confirmation required", step.value)
            return StepOutcome(
                success=False,
                needs_overwrite_confirm=True,
                existing_artifact_desc=desc,
            )

        from syntrive.services.job_lease import JobLeaseConflict, hold_job_lease

        try:
            with hold_job_lease(
                self._db_path, self._job.id,
                holder_kind=self._holder_kind, operation=f"step:{step.value}",
            ):
                return self._dispatch_handler(step)
        except JobLeaseConflict as exc:
            logger.warning("Step %s blocked by job lease: %s", step.value, exc)
            return StepOutcome(success=False, error=str(exc))
        except Exception as exc:
            logger.error("Step %s failed: %s", step, exc, exc_info=True)
            return StepOutcome(success=False, error=str(exc))

    def record_step(
        self,
        step: WorkflowStep,
        action: StepAction,
        artifacts_summary: Optional[dict] = None,
        notes: Optional[str] = None,
    ) -> None:
        from syntrive.db.session import get_db_session
        from syntrive.db.repository import JobRepository

        try:
            with get_db_session(self._db_path) as db:
                repo = JobRepository(db)
                job = repo.get_job(self._job.id)
                if job:
                    repo.record_workflow_action(
                        job=job,
                        step_name=step.value,
                        action=action.value,
                        artifacts_summary=artifacts_summary,
                        user_notes=notes,
                    )
        except Exception as exc:
            logger.warning("Failed to record workflow action: %s", exc)

    def advance_job_step(self, completed_step: WorkflowStep) -> None:
        from syntrive.db.session import get_db_session
        from syntrive.db.repository import JobRepository

        transitions = _TRANSITIONS.get(completed_step, [])
        next_step = transitions[0] if transitions else WorkflowStep.DONE
        new_status = "completed" if next_step == WorkflowStep.DONE else "running"

        try:
            with get_db_session(self._db_path) as db:
                repo = JobRepository(db)
                job = repo.get_job(self._job.id)
                if job:
                    job.current_step = next_step.value
                    job.stage = completed_step.value
                    job.status = new_status
                    logger.info(
                        "Job %d: completed=%s -> next=%s status=%s",
                        job.id, completed_step.value, next_step.value, new_status,
                    )
        except Exception as exc:
            logger.warning("Failed to update job step: %s", exc)

        from syntrive.services.contract_export_service import refresh_manifests_best_effort

        refresh_manifests_best_effort(
            self._db_path, reason=f"step:{completed_step.value}", job_id=self._job.id,
        )

    def get_previous_step(self, step: WorkflowStep) -> Optional[WorkflowStep]:
        idx = _ORDERED_STEPS.index(step)
        return _ORDERED_STEPS[idx - 1] if idx > 0 else None

    def get_next_step(self, step: WorkflowStep) -> Optional[WorkflowStep]:
        idx = _ORDERED_STEPS.index(step)
        if idx >= len(_ORDERED_STEPS) - 1:
            return None
        nxt = _ORDERED_STEPS[idx + 1]
        return nxt if nxt != WorkflowStep.DONE else None

    def describe_tasks(self, step: WorkflowStep) -> list[str]:
        if step == WorkflowStep.BOOTSTRAP:
            if self._is_bootstrap_done():
                return [
                    "Extract EPUB metadata, including images (re-extraction)",
                    "Update Book record in database with latest metadata",
                    "Ensure AI-skill pyproject.toml is present in process_dir (skip if already present)",
                ]
            return [
                "Create process directory structure",
                "Copy EPUB into process_dir",
                "Copy AI-skill pyproject.toml template into process_dir",
                "Extract EPUB metadata, including images",
                "Write Book + Job records to database",
            ]

        if step == WorkflowStep.TRANSCRIPT_EXTRACT:
            return self._describe_extract_tasks()

        if step == WorkflowStep.TRANSCRIPT_MERGE:
            return self._describe_merge_tasks()

        if step == WorkflowStep.TRANSCRIPT_CLEAN:
            return self._describe_clean_tasks()

        if step == WorkflowStep.TRANSCRIPT_TEXT:
            return self._describe_text_tasks()

        if step == WorkflowStep.TRANSCRIPT_REVIEW:
            return self._describe_review_tasks()

        if step == WorkflowStep.TTS_CONFIG:
            return self._describe_tts_config_tasks()

        descriptions: dict[WorkflowStep, list[str]] = {
            WorkflowStep.SYNTHESIS: [
                "Run TTS engine on all transcript chapters",
                "Generate per-sentence audio files",
                "Combine sentences into chapter-level audio",
            ],
            WorkflowStep.COMBINE: [
                "Combine all chapter audio into final audiobook",
                "Apply metadata (title, author, cover)",
                "Export in selected format (m4a/mp3/m4b)",
            ],
        }
        return descriptions.get(step, ["Execute step"])

    def _describe_extract_tasks(self) -> list[str]:
        return [
            "Read EPUB TOC (chapter anchors, hierarchy) for toc_hrefs",
            "Write every spine file to raw/{basename}.html (SpineExtractor)",
            "Write raw/0_toc.md — TOC + orphan spine files listing for ops review",
            "Review raw/0_toc.md to understand which files are TOC-covered vs orphan",
        ]

    def _describe_merge_tasks(self) -> list[str]:
        tasks = [
            "Re-extract TOC chapters with per-anchor HTML content (EpubHtmlExtractor)",
            "Scan EPUB spine with raw HTML content (SpineOrderScanner)",
        ]

        override = self._get_job_merge_mode_override()
        if override:
            tasks.append(
                f"Merge mode: [bold]{MERGE_MODE_LABELS.get(override, override)}[/bold] (OVERRIDDEN)"
                " — Press M to change"
            )
        else:
            tasks.append(
                "Detect merge mode: flat / volume_split / volume_continuous (auto)"
                " — Press M to set override before running"
            )

        reset_val = self.get_chapter_number_reset_per_volume()
        if reset_val:
            ch_num_desc = (
                "Chapter numbering: per-volume reset (vol1→0001, vol2→0001, ...)"
                " — Press V to change"
            )
        else:
            ch_num_desc = (
                "Chapter numbering: global sequential (vol1→0001, vol2→N+1, ...)"
                " — Press V to enable per-volume reset"
            )
        tasks.append(ch_num_desc)

        tasks += [
            "Route Q7 pre/post-book orphans (cover, copyright) → ch_9NNN discard files",
            "Route Q8 inter-volume gaps: text → next vol; images → ch_9NNN discard files",
            "Write merged/{ch_NNNN}.html — one file per TOC chapter",
            "Write merged/{ch_9NNN}.html — discard candidates for ops review",
            "Write merged/0_toc.md — chapter index (set excluded=y to skip in cleaning)",
        ]
        return tasks

    def _describe_text_tasks(self) -> list[str]:
        opts = self.get_text_extraction_options()
        enabled = [name for name, on in opts["formats"].items() if on]
        primary = opts["primary"]

        tasks = [
            "Read cleaned/0_toc.md — chapter index (skip entries with excluded=y)",
            "Fetch book.extraction_mode (none/html_tag/auto/dialogue_detect) from DB",
            "Press X to choose which output format(s) to generate and which is primary",
        ]
        for name in enabled:
            mark = " (primary → job.transcript_dir)" if name == primary else ""
            tasks.append(f"  ✓ {TEXT_EXTRACTION_FORMAT_LABELS.get(name, name)}{mark}")
        skipped = [name for name in TEXT_EXTRACTION_FORMAT_LABELS if name not in enabled]
        for name in skipped:
            tasks.append(f"  ✗ {TEXT_EXTRACTION_FORMAT_LABELS.get(name, name)} (skipped)")
        return tasks

    def _describe_review_tasks(self) -> list[str]:
        language = self._get_book_language()
        skill_name = "tts-text-normalize-cn" if language == "zh" else "tts-text-normalize-en"
        selection = self._transcript_path_selection
        selection_label = TRANSCRIPT_PATH_LABELS.get(selection, selection)
        return [
            "Manually review the transcript_text/{raw,tts_script} output from step 5/9",
            "Press C (global) to open this job's process_dir in VS Code for hand-editing",
            f"Detect book language (current: {language})",
            f"Invoke the '{skill_name}' AI skill to normalize "
            "transcript_text/tts_script/*.txt (join/split + Roman-numeral "
            "conversion — see syntrive/shared/skills/)",
            f"transcript_path selection: [bold]{selection_label}[/bold] — Press T to change",
            "On confirm (Y): sets job.transcript_dir and every chapter's "
            "TranscriptChapter.transcript_path/volume/volume_number/sequence_number/"
            "chapter_name/chapter_number from the selected format's 0_toc.md",
        ]

    def _describe_tts_config_tasks(self) -> list[str]:
        summary = self.get_tts_config_summary()
        if summary is None:
            return ["No TTS configuration saved yet."]
        summary_str = ", ".join(f"{k}={v}" for k, v in summary.items())
        return [f"Current TTS configuration: {summary_str}"]

    def _describe_clean_tasks(self) -> list[str]:
        tasks = []

        detected = self.get_detected_merge_mode()
        override = self._get_job_merge_mode_override()
        if detected or override:
            mode_str = override or detected or "unknown"
            suffix = " (OVERRIDDEN)" if override else " (auto-detected)"
            tasks.append(
                f"Merge mode used: {mode_str}{suffix}"
                " — re-run TRANSCRIPT_MERGE to change"
            )

        active_rules = self.get_active_cleaning_rules()
        if active_rules:
            rules_str = ", ".join(active_rules)
            tasks.append(
                f"Active cleaning rules ({len(active_rules)}): {rules_str}"
                " — Press R to manage rules"
            )
        else:
            tasks.append("Cleaning rules: loading from DB (fallback to hardcoded chain)")

        tasks += [
            "Read merged/0_toc.md — chapter index (skip entries with excluded=y)",
            "Read merged/{ch_NNNN}.html — assembled chapter HTML for each entry",
            "Apply cleaning rules: span merge, table summarize, English removal, numeral conversion",
            "Run threshold guard (log warning if deletion > 35%; chapter is always processed)",
            "Tag paragraphs with stable content-hash IDs",
            "Write cleaned HTML to cleaned/{ch_NNNN}.html",
            "Write paragraph map to manifest/{ch_NNNN}.para_map.yaml",
            "Write cleaned/0_toc.md — chapter index (TOC entries only, no orphan table)",
            "Persist TranscriptChapter records to database",
        ]
        return tasks

    def check_artifacts_exist(self, step: WorkflowStep) -> bool:
        from syntrive.db.path_utils import resolve_abs
        process_dir = resolve_abs(self._db_path, self._job.process_dir)

        checkers = {
            WorkflowStep.BOOTSTRAP: lambda: (
                (process_dir / "images").exists()
                and any((process_dir / "images").iterdir())
            ),
            WorkflowStep.TRANSCRIPT_EXTRACT: lambda: (
                (process_dir / "transcript_html" / "raw" / "0_toc.md").exists()
            ),
            WorkflowStep.TRANSCRIPT_MERGE: lambda: (
                (process_dir / "transcript_html" / "merged" / "0_toc.md").exists()
            ),
            WorkflowStep.TRANSCRIPT_CLEAN: lambda: (
                (process_dir / "transcript_html" / "cleaned").exists()
                and any((process_dir / "transcript_html" / "cleaned").glob("*.html"))
            ),
            WorkflowStep.TRANSCRIPT_TEXT: lambda: any(
                (process_dir / "transcript_text" / fmt).exists()
                and any((process_dir / "transcript_text" / fmt).glob("*.txt"))
                for fmt in ("raw", "tts_script")
            ),
            WorkflowStep.TRANSCRIPT_REVIEW: lambda: self._transcript_review_artifacts_exist(),
            WorkflowStep.SYNTHESIS: lambda: (
                (process_dir / "audio").exists()
                and any((process_dir / "audio").iterdir())
            ),
            WorkflowStep.COMBINE: lambda: (
                (process_dir / "audiobooks").exists()
                and any((process_dir / "audiobooks").iterdir())
            ),
        }
        checker = checkers.get(step)
        if checker is None:
            return False
        try:
            return checker()
        except Exception:
            return False

    def _transcript_review_artifacts_exist(self) -> bool:
        from syntrive.db.session import get_db_session
        from syntrive.db.models import TranscriptChapter

        try:
            with get_db_session(self._db_path) as db:
                return (
                    db.query(TranscriptChapter)
                    .filter_by(job_id=self._job.id)
                    .filter(TranscriptChapter.transcript_path.isnot(None))
                    .first()
                    is not None
                )
        except Exception as exc:
            logger.debug("_transcript_review_artifacts_exist: %s", exc)
            return False

    def resolve_start_step(self) -> WorkflowStep:
        current = getattr(self._job, "current_step", None)
        if current in _LEGACY_STEP_VALUES:
            logger.info(
                "WorkflowEngine: migrating legacy step %r -> %r for job=%d",
                current, _LEGACY_STEP_VALUES[current].value, self._job.id,
            )
        return step_from_db_value(current)

    def get_detected_merge_mode(self) -> Optional[str]:
        try:
            from syntrive.adapters.epub.merged_toc_writer import MergedTocReader
            from syntrive.db.path_utils import resolve_abs

            process_dir = resolve_abs(self._db_path, self._job.process_dir)
            toc_path = process_dir / "transcript_html" / "merged" / "0_toc.md"
            if not toc_path.exists():
                return None
            meta, _ = MergedTocReader().read(toc_path)
            return meta.merge_mode or None
        except Exception as exc:
            logger.debug("WorkflowEngine.get_detected_merge_mode: %s", exc)
        return None

    def get_tts_config_summary(self) -> Optional[dict[str, object]]:
        try:
            from syntrive.db.session import get_db_session
            from syntrive.db.repository import JobRepository
            from syntrive.db.models import ReferenceVoice, TtsVoice

            with get_db_session(self._db_path) as db:
                repo = JobRepository(db)
                job = repo.get_job(self._job.id)
                config = job.tts_config if job else None
                if config is None:
                    return None

                narrator_voice = "(unset)"
                narrator = db.query(TtsVoice).filter_by(tts_config_id=config.id, voice_id=0).first()
                if narrator is not None and narrator.reference_voice_id is not None:
                    ref = db.query(ReferenceVoice).filter_by(id=narrator.reference_voice_id).first()
                    if ref is not None:
                        narrator_voice = ref.name

                return {
                    "engine": config.engine or "(unset)",
                    "device": config.device or "(unset)",
                    "offline_mode": config.offline_mode,
                    "language": config.language or "(unset)",
                    "fine_tuned_model": config.fine_tuned_model or "internal",
                    "voice": narrator_voice,
                    "temperature": config.temperature,
                    "speed": config.speed,
                }
        except Exception as exc:
            logger.debug("WorkflowEngine.get_tts_config_summary: %s", exc)
            return None

    def get_active_cleaning_rules(self) -> list[str]:
        return [r["display_name"] for r in self.get_all_cleaning_rules_with_status() if r["enabled"]]

    def get_all_cleaning_rules_with_status(self) -> list[dict]:
        from syntrive.services.step_decisions import read_cleaning_rules

        try:
            return [
                {
                    "name": r.name,
                    "display_name": r.label,
                    "description": r.description,
                    "sort_order": r.sort_order,
                    "enabled": r.enabled,
                }
                for r in read_cleaning_rules(self._db_path, self._job.id)
            ]
        except Exception as exc:
            logger.debug("WorkflowEngine.get_all_cleaning_rules_with_status: %s", exc)
            return []

    def set_merge_mode_override(self, mode: Optional[str]) -> None:
        from syntrive.services.step_decisions import save_merge_settings

        try:
            save_merge_settings(self._db_path, self._job.id, override=mode, holder_kind=self._holder_kind)
        except Exception as exc:
            logger.warning("WorkflowEngine.set_merge_mode_override: %s", exc)

    def set_chapter_number_reset_per_volume(self, value: bool) -> None:
        from syntrive.services.step_decisions import save_merge_settings

        try:
            save_merge_settings(self._db_path, self._job.id, reset_per_volume=value, holder_kind=self._holder_kind)
        except Exception as exc:
            logger.warning("WorkflowEngine.set_chapter_number_reset_per_volume: %s", exc)

    def get_chapter_number_reset_per_volume(self) -> bool:
        from syntrive.services.step_decisions import read_merge_settings

        try:
            return read_merge_settings(self._db_path, self._job.id).reset_per_volume
        except Exception as exc:
            logger.debug("WorkflowEngine.get_chapter_number_reset_per_volume: %s", exc)
            return False

    def set_cleaning_rule_enabled(self, rule_name: str, enabled: bool) -> None:
        from syntrive.services.step_decisions import save_cleaning_rules

        try:
            save_cleaning_rules(self._db_path, self._job.id, {rule_name: enabled}, holder_kind=self._holder_kind)
        except Exception as exc:
            logger.warning("WorkflowEngine.set_cleaning_rule_enabled: %s", exc)

    def get_text_extraction_options(self) -> dict[str, object]:
        from syntrive.pipeline.text_extraction_stage import TEXT_EXTRACTION_FORMATS

        return {
            "formats": {
                name: (name in self._text_extraction_formats)
                for name in TEXT_EXTRACTION_FORMATS
            },
            "primary": self._text_extraction_primary,
        }

    def set_text_extraction_format(self, name: str, enabled: bool) -> None:
        from syntrive.pipeline.text_extraction_stage import TEXT_EXTRACTION_FORMATS

        if name not in TEXT_EXTRACTION_FORMATS:
            logger.warning("WorkflowEngine.set_text_extraction_format: unknown format %r", name)
            return

        if enabled:
            self._text_extraction_formats.add(name)
        else:
            self._text_extraction_formats.discard(name)
            if self._text_extraction_primary == name and self._text_extraction_formats:
                self._text_extraction_primary = sorted(self._text_extraction_formats)[0]
                logger.info(
                    "WorkflowEngine: primary format was disabled -- falling back to %r",
                    self._text_extraction_primary,
                )
        logger.info(
            "WorkflowEngine: set_text_extraction_format %s=%s (active=%s)",
            name, enabled, sorted(self._text_extraction_formats),
        )

    def set_text_extraction_primary(self, name: str) -> None:
        from syntrive.pipeline.text_extraction_stage import TEXT_EXTRACTION_FORMATS

        if name not in TEXT_EXTRACTION_FORMATS:
            logger.warning("WorkflowEngine.set_text_extraction_primary: unknown format %r", name)
            return

        self._text_extraction_formats.add(name)
        self._text_extraction_primary = name
        logger.info("WorkflowEngine: set_text_extraction_primary=%r", name)

    def get_transcript_path_selection(self) -> str:
        return self._transcript_path_selection

    def set_transcript_path_selection(self, name: str) -> None:
        from syntrive.pipeline.text_extraction_stage import TRANSCRIPT_PATH_CHOICES

        if name not in TRANSCRIPT_PATH_CHOICES:
            logger.warning(
                "WorkflowEngine.set_transcript_path_selection: unknown selection %r", name
            )
            return

        self._transcript_path_selection = name
        logger.info("WorkflowEngine: set_transcript_path_selection=%r", name)

    def _cli_run_one_step(
        self, step: WorkflowStep, console
    ) -> tuple[StepAction, Optional[WorkflowStep]]:
        step_index = _ORDERED_STEPS.index(step) + 1
        total = len(_ORDERED_STEPS) - 1

        if console is not None:
            console.show_step_summary(
                step_name=_STEP_LABELS[step],
                step_number=step_index,
                total_steps=total,
                tasks=self.describe_tasks(step),
            )
        else:
            logger.info("STEP %d/%d: %s", step_index, total, _STEP_LABELS[step])

        choices = ["[Y] Confirm", "[B] Go back", "[P] Pause & save", "[Q] Quit"]
        logger.info("Options: %s", "  ".join(choices))
        raw = input("Your choice [Y/B/P/Q]: ").strip().upper()

        if raw == "B":
            return StepAction.BACK, None
        if raw == "P":
            self.record_step(step, StepAction.PAUSED)
            return StepAction.PAUSED, None
        if raw == "Q":
            self.record_step(step, StepAction.PAUSED, notes="user quit")
            return StepAction.PAUSED, None
        if raw not in ("Y", ""):
            logger.warning("Unknown input '%s'. Treating as confirm.", raw)

        outcome = self.execute_step(step)

        if outcome.needs_overwrite_confirm:
            logger.warning(
                "%s\nOverwrite? [Y/S/N] Y=Overwrite S=Skip N=Cancel: ",
                outcome.existing_artifact_desc,
            )
            choice = input().strip().upper()
            if choice == "Y":
                outcome = self.execute_step(step, force_overwrite=True)
            elif choice == "S":
                self.record_step(step, StepAction.SKIPPED, notes="user skipped existing artifact")
                self.advance_job_step(step)
                transitions = _TRANSITIONS.get(step, [])
                real = [s for s in transitions if s != WorkflowStep.DONE]
                return StepAction.SKIPPED, real[0] if real else None
            else:
                return StepAction.CANCELLED, None

        if not outcome.success:
            if console is not None:
                console.show_step_error(_STEP_LABELS[step], outcome.error or "Unknown error")
            else:
                logger.error("Step failed: %s", outcome.error)
            self.record_step(step, StepAction.CANCELLED, notes=outcome.error)
            return StepAction.CANCELLED, None

        if console is not None:
            console.show_step_result(_STEP_LABELS[step], outcome.artifacts, outcome.notes)
        self.record_step(step, StepAction.CONFIRMED, artifacts_summary=outcome.artifacts)
        self.advance_job_step(step)

        transitions = _TRANSITIONS.get(step, [])
        if not transitions:
            return StepAction.CONFIRMED, WorkflowStep.DONE

        next_labels = [_STEP_LABELS[s] for s in transitions]
        if console is not None:
            chosen_label = console.ask_next_step(next_labels)
        else:
            chosen_label = next_labels[0]

        if chosen_label == "quit":
            return StepAction.CONFIRMED, None

        chosen_step = transitions[next_labels.index(chosen_label)]
        return StepAction.CONFIRMED, chosen_step

    def _dispatch_handler(self, step: WorkflowStep) -> StepOutcome:
        handlers = {
            WorkflowStep.BOOTSTRAP: self._handle_bootstrap,
            WorkflowStep.TRANSCRIPT_EXTRACT: self._handle_transcript_extract,
            WorkflowStep.TRANSCRIPT_MERGE: self._handle_transcript_merge,
            WorkflowStep.TRANSCRIPT_CLEAN: self._handle_transcript_clean,
            WorkflowStep.TRANSCRIPT_TEXT: self._handle_transcript_text,
            WorkflowStep.TRANSCRIPT_REVIEW: self._handle_transcript_review,
            WorkflowStep.TTS_CONFIG: self._handle_tts_config,
            WorkflowStep.SYNTHESIS: self._handle_synthesis,
            WorkflowStep.COMBINE: self._handle_combine,
        }
        handler = handlers.get(step)
        if handler is None:
            return StepOutcome(success=False, error=f"No handler for step: {step}")
        return handler()

    def _handle_bootstrap(self) -> StepOutcome:
        from syntrive.bootstrap import extract_epub_images, update_book_metadata, copy_shared_pyproject
        from syntrive.db.session import get_db_session
        from syntrive.db.repository import JobRepository
        from syntrive.db.models import Book
        from syntrive.db.path_utils import resolve_abs

        epub_path = resolve_abs(self._db_path, self._job.epub_path)
        process_dir = resolve_abs(self._db_path, self._job.process_dir)
        images_dir = process_dir / "images"
        is_start_over = self._is_bootstrap_done()

        if is_start_over:
            logger.info(
                "Bootstrap (start-over): re-extracting metadata+images for job %d", self._job.id
            )
            update_book_metadata(
                epub_path=epub_path,
                db_path=self._db_path,
                book_id=self._job.book_id,
            )

        logger.info(
            "Bootstrap (%s): extracting images for job %d",
            "start-over" if is_start_over else "new book",
            self._job.id,
        )
        image_records = extract_epub_images(epub_path, images_dir, process_dir)

        pyproject_path = copy_shared_pyproject(process_dir)

        cover_rel_path: Optional[str] = None
        try:
            with get_db_session(self._db_path) as db:
                repo = JobRepository(db)
                book = db.query(Book).filter_by(id=self._job.book_id).first()
                if book is not None:
                    for rec in image_records:
                        repo.upsert_book_image(
                            book=book,
                            name=rec.name,
                            path=rec.rel_path,
                            image_type=rec.image_type,
                        )
                    detected = next(
                        (r for r in image_records if r.image_type == "cover"), None
                    )
                    if detected is not None:
                        cover_rel_path = detected.rel_path
                        repo.update_book_cover(book, cover_rel_path)
                    elif is_start_over and book.cover is None and image_records:
                        cover_rel_path = image_records[0].rel_path
                        repo.update_book_cover(book, cover_rel_path)
                else:
                    logger.warning(
                        "_handle_bootstrap: book_id=%d not found, skipping image DB write",
                        self._job.book_id,
                    )
        except Exception as exc:
            logger.warning("Bootstrap: failed to persist image records: %s", exc)

        n_images = len(image_records)
        pyproject_flag = 1 if pyproject_path else 0
        if is_start_over:
            return StepOutcome(
                success=True,
                artifacts={"images": n_images, "metadata": 1, "pyproject": pyproject_flag},
                notes=(
                    f"Re-extracted {n_images} image(s) and updated Book metadata. "
                    f"Cover: {cover_rel_path or 'not detected'}"
                ),
            )
        return StepOutcome(
            success=True,
            artifacts={"epub": 1, "db": 1, "images": n_images, "pyproject": pyproject_flag},
            notes=(
                f"Job {self._job.id} — {self._job.epub_path} "
                f"({n_images} image(s), cover: {cover_rel_path or 'not detected'})"
            ),
        )

    def _handle_transcript_extract(self) -> StepOutcome:
        from syntrive.pipeline.transcript_html_stage import TranscriptHtmlStage

        stage = TranscriptHtmlStage(
            job=self._job,
            db_path=self._db_path,
        )
        result = stage.run_extract()
        return StepOutcome(
            success=result.success,
            artifacts=result.artifacts,
            notes=result.notes,
            error=result.error,
        )

    def _handle_transcript_merge(self) -> StepOutcome:
        from syntrive.pipeline.transcript_html_stage import TranscriptHtmlStage

        stage = TranscriptHtmlStage(
            job=self._job,
            db_path=self._db_path,
        )
        result = stage.run_merge()
        return StepOutcome(
            success=result.success,
            artifacts=result.artifacts,
            notes=result.notes,
            error=result.error,
        )

    def _handle_transcript_text(self) -> StepOutcome:
        from syntrive.pipeline.text_extraction_stage import TextExtractionStage

        stage = TextExtractionStage(job=self._job, db_path=self._db_path)
        result = stage.run_extract(
            formats=self._text_extraction_formats,
            primary_format=self._text_extraction_primary,
        )
        return StepOutcome(
            success=result.success,
            artifacts=result.artifacts,
            notes=result.notes,
            error=result.error,
        )

    def _handle_transcript_review(self) -> StepOutcome:
        from syntrive.pipeline.text_extraction_stage import TextExtractionStage

        language = self._get_book_language()
        skill_name = "tts-text-normalize-cn" if language == "zh" else "tts-text-normalize-en"
        selection = self._transcript_path_selection

        stage = TextExtractionStage(job=self._job, db_path=self._db_path)
        result = stage.persist_transcript_paths(selection)

        logger.info(
            "_handle_transcript_review: job=%d lang=%s selection=%s result=%s",
            self._job.id, language, selection, result,
        )

        if not result["success"]:
            return StepOutcome(success=False, error=result["error"])

        skipped_note = f", {result['skipped']} skipped" if result["skipped"] else ""
        notes = (
            "Manual review + AI-skill normalization checkpoint. "
            f"Use the '{skill_name}' AI skill to normalize transcript_text/tts_script/*.txt "
            f"before proceeding. transcript_path selection: {selection} "
            f"({result['updated']} chapter(s) updated{skipped_note}). "
            f"job.transcript_dir -> {result['transcript_dir']}."
        )
        return StepOutcome(
            success=True,
            artifacts={"updated": result["updated"], "skipped": result["skipped"]},
            notes=notes,
        )

    def _get_book_language(self) -> str:
        from syntrive.db.session import get_db_session
        from syntrive.db.models import Book
        try:
            with get_db_session(self._db_path) as db:
                book = db.query(Book).filter_by(id=self._job.book_id).first()
                return (book.language if book and book.language else "en")
        except Exception as exc:
            logger.warning("_get_book_language: failed (defaulting to 'en'): %s", exc)
            return "en"


    def _handle_transcript_clean(self) -> StepOutcome:
        from syntrive.pipeline.transcript_html_stage import TranscriptHtmlStage

        stage = TranscriptHtmlStage(
            job=self._job,
            db_path=self._db_path,
        )
        result = stage.run_clean()
        return StepOutcome(
            success=result.success,
            artifacts=result.artifacts,
            notes=result.notes,
            error=result.error,
        )

    def ensure_tts_config(self) -> bool:
        from syntrive.adapters.tts.device import detect_best_device
        from syntrive.db.session import get_db_session
        from syntrive.db.repository import JobRepository
        from syntrive.db.models import Book, TtsConfig

        with get_db_session(self._db_path) as db:
            repo = JobRepository(db)
            job = repo.get_job(self._job.id)
            if job is None or job.tts_config is not None:
                return False

            book = db.query(Book).filter_by(id=job.book_id).first()
            language = book.language if book else None
            config = TtsConfig(
                job_id=self._job.id,
                language=language,
                device=detect_best_device(),
                engine="cosyvoice",
            )
            db.add(config)
            db.flush()
            logger.info(
                "TtsConfig auto-created for job %d: language=%s device=%s engine=%s",
                self._job.id, config.language, config.device, config.engine,
            )
            return True

    def ensure_tts_voices(self) -> None:
        self._ensure_narrator_tts_voice()
        self._reconcile_role_tts_voices()

    def _ensure_narrator_tts_voice(self) -> None:
        from syntrive.db.models import TtsVoice
        from syntrive.db.repository import JobRepository
        from syntrive.db.session import get_db_session

        with get_db_session(self._db_path) as db:
            repo = JobRepository(db)
            job = repo.get_job(self._job.id)
            if job is None or job.tts_config is None:
                return
            tts_config = job.tts_config
            if db.query(TtsVoice).filter_by(tts_config_id=tts_config.id, voice_id=0).first():
                return

            db.add(TtsVoice(
                tts_config_id=tts_config.id, name="narrator", voice_id=0,
                reference_voice_id=None, excluded=False,
            ))
            logger.info("TtsVoice narrator row created for job %d", self._job.id)

    def _reconcile_role_tts_voices(self) -> None:
        import sys
        from pathlib import Path as _Path

        from syntrive.db.models import TtsVoice
        from syntrive.db.path_utils import resolve_abs
        from syntrive.db.repository import JobRepository
        from syntrive.db.session import get_db_session

        skill_script_dir = (
            _Path(__file__).resolve().parent.parent
            / "shared" / "skills" / "tts-voice-config" / "script"
        )
        if str(skill_script_dir) not in sys.path:
            sys.path.insert(0, str(skill_script_dir))
        import _voice_config as vc

        with get_db_session(self._db_path) as db:
            repo = JobRepository(db)
            job = repo.get_job(self._job.id)
            if job is None or job.tts_config is None or not job.transcript_dir:
                return

            yaml_path = resolve_abs(self._db_path, job.transcript_dir) / "voice_config.yaml"
            yaml_roles = {e.name: e.voice for e in vc.parse(yaml_path) if e.voice is not None}
            if not yaml_roles:
                return

            existing_by_name = {
                row.name: row
                for row in db.query(TtsVoice)
                .filter(TtsVoice.tts_config_id == job.tts_config.id, TtsVoice.voice_id != 0)
                .all()
            }

            created = deleted = updated = 0
            for name, voice_id in yaml_roles.items():
                row = existing_by_name.get(name)
                if row is None:
                    db.add(TtsVoice(
                        tts_config_id=job.tts_config.id, name=name, voice_id=voice_id,
                        reference_voice_id=None, excluded=False,
                    ))
                    created += 1
                elif row.voice_id != voice_id:
                    row.voice_id = voice_id
                    updated += 1

            for name, row in existing_by_name.items():
                if name not in yaml_roles:
                    db.delete(row)
                    deleted += 1

            if created or deleted or updated:
                logger.info(
                    "TtsVoice roles reconciled against voice_config.yaml: job=%d "
                    "created=%d deleted=%d voice_id_updated=%d",
                    self._job.id, created, deleted, updated,
                )

    def _handle_tts_config(self) -> StepOutcome:
        from syntrive.db.session import get_db_session
        from syntrive.db.repository import JobRepository
        from syntrive.db.models import TtsVoice

        created = self.ensure_tts_config()
        self.ensure_tts_voices()

        with get_db_session(self._db_path) as db:
            repo = JobRepository(db)
            job = repo.get_job(self._job.id)
            if job is None:
                return StepOutcome(success=False, error=f"Job {self._job.id} not found")

            config = job.tts_config
            engine = config.engine
            device = config.device
            language = config.language
            offline_mode = config.offline_mode
            fine_tuned_model = config.fine_tuned_model

            narrator = db.query(TtsVoice).filter_by(tts_config_id=config.id, voice_id=0).first()
            narrator_voice_bound = narrator is not None and narrator.reference_voice_id is not None

        if not engine or not narrator_voice_bound:
            return StepOutcome(
                success=False,
                error=(
                    "No TTS engine/narrator voice selected. Choose a narrator voice before "
                    "continuing (TUI: press E, Voices section; WebUI: step 7, Cast tab)."
                ),
            )

        logger.info(
            "TTS config confirmed for job %d: engine=%s device=%s language=%s "
            "narrator_voice_bound=%s fine_tuned_model=%s offline_mode=%s (created=%s)",
            self._job.id, engine, device, language, narrator_voice_bound,
            fine_tuned_model, offline_mode, created,
        )
        return StepOutcome(
            success=True,
            artifacts={"config": 1},
            notes=(
                f"TTS config: engine={engine}, device={device}, language={language}, "
                f"narrator voice bound, offline_mode={offline_mode}"
            ),
        )

    def _handle_synthesis(self) -> StepOutcome:
        cover = self._get_book_cover()
        if not cover:
            msg = (
                "Audiobook cover is not set. "
                "Please set a cover image first: press M -> select Cover Image -> Ctrl+S."
            )
            logger.warning(
                "_handle_synthesis: blocked — book_id=%d has no cover set", self._job.book_id
            )
            return StepOutcome(success=False, error=msg)

        logger.info("TTS Synthesis implementation pending (iteration 6). cover=%s", cover)
        return StepOutcome(success=True, artifacts={"flac": 0}, notes="Pending iteration 6.")

    def _handle_combine(self) -> StepOutcome:
        logger.info("Combine stage implementation pending (iteration 6).")
        return StepOutcome(success=True, artifacts={"m4a": 0}, notes="Pending iteration 6.")

    def _get_book_cover(self) -> Optional[str]:
        try:
            from syntrive.db.session import get_db_session
            from syntrive.db.models import Book

            with get_db_session(self._db_path) as db:
                book = db.query(Book).filter_by(id=self._job.book_id).first()
                return book.cover if book else None
        except Exception as exc:
            logger.warning("_get_book_cover: failed for book_id=%d: %s", self._job.book_id, exc)
            return None

    def _get_job_merge_mode_override(self) -> Optional[str]:
        try:
            from syntrive.db.session import get_db_session
            from syntrive.db.models import Job

            with get_db_session(self._db_path) as db:
                job = db.query(Job).filter_by(id=self._job.id).first()
                return job.merge_mode_override if job else None
        except Exception:
            return None

    def _is_bootstrap_done(self) -> bool:
        from syntrive.db.path_utils import resolve_abs
        images_dir = resolve_abs(self._db_path, self._job.process_dir) / "images"
        try:
            return images_dir.exists() and any(images_dir.iterdir())
        except Exception:
            return False

    def artifact_description(self, step: WorkflowStep) -> str:
        from syntrive.db.path_utils import resolve_abs
        process_dir = resolve_abs(self._db_path, self._job.process_dir)

        descriptions = {
            WorkflowStep.BOOTSTRAP: (
                f"EPUB metadata and images already extracted in {process_dir / 'images'}.\n"
                "Overwrite will re-extract metadata and images from the EPUB file.\n"
                "This will UPDATE the Book record in the database."
            ),
            WorkflowStep.TRANSCRIPT_EXTRACT: (
                f"Spine files already exist in "
                f"{process_dir / 'transcript_html' / 'raw'} (raw/0_toc.md found).\n"
                "Overwrite will re-scan the EPUB spine and rewrite raw/ + raw/0_toc.md."
            ),
            WorkflowStep.TRANSCRIPT_MERGE: (
                f"Merged chapter files already exist in "
                f"{process_dir / 'transcript_html' / 'merged'} (merged/0_toc.md found).\n"
                "Overwrite will re-run HtmlMerger and rewrite all merged/ files.\n"
                "WARNING: This will discard any manual edits to merged/0_toc.md."
            ),
            WorkflowStep.TRANSCRIPT_CLEAN: (
                f"Cleaned HTML files already exist in "
                f"{process_dir / 'transcript_html' / 'cleaned'}.\n"
                "Overwrite will re-apply all cleaning rules to the merged HTML files."
            ),
            WorkflowStep.TRANSCRIPT_TEXT: (
                f"Plain text files already exist in "
                f"{process_dir / 'transcript_text' / 'raw'}.\n"
                "Overwrite will re-extract plain text from all cleaned HTML chapters."
            ),
            WorkflowStep.TRANSCRIPT_REVIEW: (
                "A transcript_path selection has already been persisted for this job "
                "(job.transcript_dir + TranscriptChapter.transcript_path set).\n"
                "Overwrite will re-persist the current transcript_path selection, "
                "overwriting those DB values."
            ),
            WorkflowStep.SYNTHESIS: (
                f"Audio files already exist in {process_dir / 'audio'}"
            ),
            WorkflowStep.COMBINE: (
                f"Audiobook files already exist in {process_dir / 'audiobooks'}"
            ),
        }
        return descriptions.get(step, f"Artifacts for step '{_STEP_LABELS[step]}' already exist")
