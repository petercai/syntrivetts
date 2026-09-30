from __future__ import annotations

import logging
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

def _count_transcript_lines(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return 0
    return sum(1 for ln in lines if ln.strip())


def _renumber_entries_for_text_output(entries: list, reset_per_vol: bool) -> list:
    import copy as _copy

    has_volume_structure = any(e.volume_number for e in entries)
    vol_counter: dict[str, int] = {}
    chapter_counter = 0
    renumbered: list = []
    for i, entry in enumerate(entries):
        out = _copy.copy(entry)
        out.sequence_number = f"{i + 1:04d}"

        is_chapter = (not has_volume_structure) or bool(entry.volume_number)
        if not is_chapter:
            out.chapter_number = ""
        elif reset_per_vol and entry.volume_number:
            vol_counter[entry.volume_number] = vol_counter.get(entry.volume_number, 0) + 1
            out.chapter_number = f"{vol_counter[entry.volume_number]:04d}"
        else:
            chapter_counter += 1
            out.chapter_number = f"{chapter_counter:04d}"
        renumbered.append(out)
    return renumbered


def _output_id(entry) -> str:
    return f"ch_{entry.sequence_number}" if entry.sequence_number else entry.chapter_id


@dataclass
class TextExtractionResult:
    success: bool
    artifacts: dict[str, int] = dc_field(default_factory=dict)
    notes: Optional[str] = None
    error: Optional[str] = None


TEXT_EXTRACTION_FORMATS = ("raw", "tts_script")
REVIEW_REPORT_NAME = "review_unpunctuated.md"
DEFAULT_TEXT_EXTRACTION_FORMATS = frozenset({"raw", "tts_script"})
DEFAULT_TEXT_EXTRACTION_PRIMARY = "tts_script"

TRANSCRIPT_PATH_CHOICES = ("raw", "tts_script")
DEFAULT_TRANSCRIPT_PATH_SELECTION = "tts_script"


class TextExtractionStage:
    def __init__(self, job, db_path: Path) -> None:
        from syntrive.db.path_utils import resolve_abs

        self._job = job
        self._db_path = db_path
        self._process_dir = resolve_abs(db_path, job.process_dir)
        self._cleaned_dir = self._process_dir / "transcript_html" / "cleaned"
        self._text_root = self._process_dir / "transcript_text"
        self._raw_dir = self._text_root / "raw"
        self._script_dir = self._text_root / "tts_script"
        self._review_findings: list = []

    def run_extract(
        self,
        formats: Optional[set[str]] = None,
        primary_format: Optional[str] = None,
    ) -> TextExtractionResult:
        formats = set(formats) if formats else set(DEFAULT_TEXT_EXTRACTION_FORMATS)
        formats &= set(TEXT_EXTRACTION_FORMATS)
        if not formats:
            formats = set(DEFAULT_TEXT_EXTRACTION_FORMATS)
        if not primary_format or primary_format not in formats:
            primary_format = (
                DEFAULT_TEXT_EXTRACTION_PRIMARY
                if DEFAULT_TEXT_EXTRACTION_PRIMARY in formats
                else sorted(formats)[0]
            )

        logger.info(
            "TextExtractionStage.run_extract: start -- job=%d formats=%s primary=%s",
            self._job.id, sorted(formats), primary_format,
        )

        try:
            toc_path = self._cleaned_dir / "0_toc.md"
            if not toc_path.exists():
                msg = (
                    f"cleaned/0_toc.md not found at {toc_path} -- "
                    "run TRANSCRIPT_CLEAN first"
                )
                logger.error("TextExtractionStage.run_extract: %s", msg)
                return TextExtractionResult(success=False, error=msg)

            language, extraction_mode = self._fetch_book_details()
            logger.info(
                "TextExtractionStage.run_extract: language=%s extraction_mode=%s",
                language, extraction_mode,
            )

            _meta, raw_entries = self._read_cleaned_toc(toc_path)

            keep_entries = [e for e in raw_entries if not e.is_discard and not e.excluded]
            skipped_upfront = len(raw_entries) - len(keep_entries)
            reset_per_vol = self._get_chapter_number_reset_per_volume()
            entries = _renumber_entries_for_text_output(keep_entries, reset_per_vol)
            logger.info(
                "TextExtractionStage.run_extract: renumbered %d/%d entries "
                "(reset_per_vol=%s, %d excluded/discard dropped)",
                len(entries), len(raw_entries), reset_per_vol, skipped_upfront,
            )

            artifacts: dict[str, int] = {}
            notes_parts: list[str] = []
            skipped_count = 0

            if "raw" in formats:
                self._raw_dir.mkdir(parents=True, exist_ok=True)
                self._cleanup_stale_output_files(self._raw_dir, entries, ext="txt")
                raw_count, skipped_count = self._run_raw_extraction_loop(
                    entries, language=language
                )
                self._write_text_format_toc(self._raw_dir, entries, ext="txt", strip_sml=False)
                artifacts["raw"] = raw_count
                notes_parts.append(f"{raw_count} raw .txt file(s)")

            if "tts_script" in formats:
                self._script_dir.mkdir(parents=True, exist_ok=True)
                self._cleanup_stale_output_files(self._script_dir, entries, ext="txt")
                script_count, skipped_count = self._run_script_extraction_loop(
                    entries, language=language, extraction_mode=extraction_mode
                )
                self._write_text_format_toc(self._script_dir, entries, ext="txt", strip_sml=True)
                artifacts["tts_script"] = script_count
                notes_parts.append(f"{script_count} tts_script .txt file(s)")
                review_count = self._write_review_report()
                if review_count:
                    notes_parts.append(
                        f"REVIEW NEEDED: {review_count} paragraph(s) without terminal punctuation "
                        f"(see tts_script/{REVIEW_REPORT_NAME})"
                    )

            artifacts["skipped"] = skipped_count + skipped_upfront
            transcript_dir_rel = self._set_transcript_dir(primary_format)
            artifacts_str = ", ".join(notes_parts) if notes_parts else "no formats generated"
            notes = (
                f"Generated: {artifacts_str} (extraction_mode={extraction_mode}). "
                f"job.transcript_dir -> {transcript_dir_rel} (primary={primary_format})."
            )
            if skipped_upfront:
                notes += f" {skipped_upfront} chapter(s) had excluded=y and were omitted."
            if skipped_count:
                notes += f" {skipped_count} chapter(s) had a missing cleaned HTML file."

            success = any(v > 0 for k, v in artifacts.items() if k in formats)
            logger.info(
                "TextExtractionStage.run_extract: complete -- "
                "job=%d formats=%s primary=%s artifacts=%s",
                self._job.id, sorted(formats), primary_format, artifacts,
            )
            return TextExtractionResult(
                success=success,
                artifacts=artifacts,
                notes=notes,
            )

        except Exception as exc:
            logger.error(
                "TextExtractionStage.run_extract: failed -- job=%d error=%s",
                self._job.id, exc, exc_info=True,
            )
            return TextExtractionResult(success=False, error=str(exc))

    def _fetch_book_details(self) -> tuple[str, str]:
        from syntrive.db.session import get_db_session
        from syntrive.db.models import Book

        try:
            with get_db_session(self._db_path) as db:
                book = db.query(Book).filter_by(id=self._job.book_id).first()
                lang = (book.language if book and book.language else "en")
                mode = (book.extraction_mode if book and book.extraction_mode else "none")
                logger.debug(
                    "TextExtractionStage: book_id=%d language=%s extraction_mode=%s",
                    self._job.book_id, lang, mode,
                )
                return lang, mode
        except Exception as exc:
            logger.warning(
                "TextExtractionStage._fetch_book_details: failed (defaulting to en/none): %s",
                exc,
            )
            return "en", "none"

    def _get_chapter_number_reset_per_volume(self) -> bool:
        try:
            from syntrive.db.session import get_db_session
            from syntrive.db.models import Job

            with get_db_session(self._db_path) as db:
                job = db.query(Job).filter_by(id=self._job.id).first()
                if job is not None:
                    return bool(job.chapter_number_reset_per_volume)
        except Exception as exc:
            logger.debug(
                "TextExtractionStage: could not read chapter_number_reset_per_volume: %s", exc
            )
        return False

    def _read_cleaned_toc(self, toc_path: Path):
        from syntrive.adapters.epub.merged_toc_writer import MergedTocReader
        meta, entries = MergedTocReader().read(toc_path)
        logger.debug(
            "TextExtractionStage: read cleaned/0_toc.md -- %d entries", len(entries)
        )
        return meta, entries

    def _run_raw_extraction_loop(
        self, entries: list, *, language: str
    ) -> tuple[int, int]:
        from syntrive.adapters.text.extractor import extract_text_from_html

        written = 0
        skipped = 0
        for entry in entries:
            if entry.is_discard:
                continue
            if entry.excluded:
                skipped += 1
                continue

            chapter_id: str = entry.chapter_id
            html_path = self._cleaned_dir / f"{chapter_id}.html"
            if not html_path.exists():
                logger.warning(
                    "TextExtractionStage(raw): cleaned HTML not found for %s -- skipping",
                    chapter_id,
                )
                skipped += 1
                continue

            text = extract_text_from_html(
                html_path.read_text(encoding="utf-8"), language=language
            )
            (self._raw_dir / f"{_output_id(entry)}.txt").write_text(text, encoding="utf-8")
            written += 1

        logger.debug("TextExtractionStage(raw): wrote %d file(s), skipped %d", written, skipped)
        return written, skipped

    def _run_script_extraction_loop(
        self, entries: list, *, language: str, extraction_mode: str
    ) -> tuple[int, int]:
        from syntrive.adapters.text.extractor import extract_tts_script_from_html

        role_voice_map: dict[str, int] = {}
        self._review_findings = []
        written = 0
        skipped = 0
        for entry in entries:
            if entry.is_discard:
                continue
            if entry.excluded:
                skipped += 1
                continue

            chapter_id: str = entry.chapter_id
            html_path = self._cleaned_dir / f"{chapter_id}.html"
            if not html_path.exists():
                logger.warning(
                    "TextExtractionStage(tts_script): cleaned HTML not found for %s -- skipping",
                    chapter_id,
                )
                skipped += 1
                continue

            chapter_review: list = []
            script = extract_tts_script_from_html(
                html_path.read_text(encoding="utf-8"),
                language=language,
                extraction_mode=extraction_mode,
                role_voice_map=role_voice_map,
                review_sink=chapter_review,
            )
            self._review_findings.extend(
                (_output_id(entry), getattr(entry, "title", "") or "", item)
                for item in chapter_review
            )
            (self._script_dir / f"{_output_id(entry)}.txt").write_text(script, encoding="utf-8")
            written += 1

        logger.debug(
            "TextExtractionStage(tts_script): wrote %d file(s), skipped %d, roles=%s",
            written, skipped, role_voice_map,
        )
        return written, skipped

    def _write_review_report(self) -> int:
        report_path = self._script_dir / REVIEW_REPORT_NAME
        findings = self._review_findings
        if not findings:
            if report_path.exists():
                report_path.unlink()
            return 0

        rows = [
            "# Paragraphs without terminal punctuation (manual review)",
            "",
            "Each paragraph below got **no** `‡break‡`, so tts-text-normalize joins it with the",
            "next paragraph. Check that the merge is right; fix the source HTML otherwise.",
            "",
            "| Chapter | Title | Element id | Preview |",
            "|---|---|---|---|",
        ]
        for chapter_id, title, item in findings:
            title_cell = title.replace("|", "\\|")
            preview_cell = item.preview.replace("|", "\\|")
            rows.append(f"| {chapter_id} | {title_cell} | {item.element_id} | {preview_cell} |")
        report_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
        logger.warning(
            "TextExtractionStage: %d paragraph(s) without terminal punctuation -- review %s",
            len(findings), report_path,
        )
        return len(findings)

    def _cleanup_stale_output_files(self, dir_path: Path, entries: list, *, ext: str) -> int:
        if not dir_path.exists():
            return 0
        expected = {f"{_output_id(e)}.{ext}" for e in entries}
        removed = 0
        for p in dir_path.glob(f"ch_*.{ext}"):
            if p.name not in expected:
                p.unlink()
                removed += 1
                logger.info(
                    "TextExtractionStage: removed stale %s (no longer in current 0_toc.md)",
                    p,
                )
        if removed:
            logger.info(
                "TextExtractionStage: cleaned up %d stale file(s) in %s", removed, dir_path.name
            )
        return removed

    def _write_text_format_toc(
        self, dir_path: Path, entries: list, *, ext: str, strip_sml: bool = False,
    ) -> None:
        from syntrive.adapters.text.toc_writer import TextTocWriter

        toc_path = dir_path / "0_toc.md"
        TextTocWriter().write(entries, toc_path, ext=ext, strip_sml=strip_sml)
        logger.info(
            "TextExtractionStage: wrote %s/0_toc.md -- %d entries",
            dir_path.name, len(entries),
        )

    def _set_transcript_dir(self, primary_format: str) -> str:
        from syntrive.db.path_utils import to_repo_relative
        from syntrive.db.session import get_db_session
        from syntrive.db.models import Job

        format_dirs = {
            "raw": self._raw_dir,
            "tts_script": self._script_dir,
        }
        target_dir = format_dirs[primary_format]
        rel_path = to_repo_relative(self._db_path, target_dir)

        try:
            with get_db_session(self._db_path) as db:
                job = db.query(Job).filter_by(id=self._job.id).first()
                if job is not None:
                    job.transcript_dir = rel_path
                    logger.info(
                        "TextExtractionStage: set job.transcript_dir=%r (primary=%s) for job=%d",
                        rel_path, primary_format, self._job.id,
                    )
        except Exception as exc:
            logger.warning(
                "TextExtractionStage._set_transcript_dir: DB update failed (non-fatal): %s",
                exc,
            )
        return rel_path

    def persist_transcript_paths(self, selection: str) -> dict[str, object]:
        if selection not in TRANSCRIPT_PATH_CHOICES:
            logger.warning(
                "TextExtractionStage.persist_transcript_paths: unknown selection %r "
                "-- falling back to %r",
                selection, DEFAULT_TRANSCRIPT_PATH_SELECTION,
            )
            selection = DEFAULT_TRANSCRIPT_PATH_SELECTION

        format_dirs = {"raw": self._raw_dir, "tts_script": self._script_dir}
        target_dir = format_dirs[selection]
        toc_path = target_dir / "0_toc.md"
        if not toc_path.exists():
            msg = (
                f"{target_dir.name}/0_toc.md not found at {toc_path} -- "
                f"run TRANSCRIPT_TEXT (step 5/9) with the '{selection}' format enabled first"
            )
            logger.error("TextExtractionStage.persist_transcript_paths: %s", msg)
            return {
                "success": False, "updated": 0, "skipped": 0,
                "transcript_dir": None, "error": msg,
            }

        logger.info(
            "TextExtractionStage.persist_transcript_paths: start -- job=%d selection=%s",
            self._job.id, selection,
        )

        from syntrive.adapters.text.toc_writer import TextTocReader

        entries = TextTocReader().read(toc_path)

        updated = 0
        skipped = 0
        try:
            from syntrive.db.session import get_db_session
            from syntrive.db.models import TranscriptChapter

            with get_db_session(self._db_path) as db:
                for entry in entries:
                    file_path = target_dir / f"ch_{entry.sequence_number}.txt"
                    if not file_path.exists():
                        skipped += 1
                        logger.warning(
                            "TextExtractionStage.persist_transcript_paths: %s not found "
                            "for chapter %s -- skipping",
                            file_path, entry.chapter_id,
                        )
                        continue

                    try:
                        rel_path = file_path.relative_to(self._process_dir).as_posix()
                    except ValueError:
                        rel_path = file_path.as_posix()

                    record = (
                        db.query(TranscriptChapter)
                        .filter_by(job_id=self._job.id, chapter_id=entry.chapter_id)
                        .first()
                    )
                    if record is None:
                        skipped += 1
                        logger.warning(
                            "TextExtractionStage.persist_transcript_paths: "
                            "TranscriptChapter not found for job_id=%d chapter_id=%s "
                            "-- transcript_path not persisted",
                            self._job.id, entry.chapter_id,
                        )
                        continue

                    record.transcript_path = rel_path
                    record.transcript_lines = _count_transcript_lines(file_path)
                    record.volume = entry.volume_name
                    record.volume_number = entry.volume_number
                    record.sequence_number = entry.sequence_number
                    record.chapter_name = entry.title
                    record.chapter_number = entry.chapter_number or ""
                    updated += 1
                    logger.debug(
                        "TextExtractionStage: updated TranscriptChapter %s "
                        "transcript_path=%s transcript_lines=%d volume=%s "
                        "volume_number=%s sequence_number=%s chapter_name=%s "
                        "chapter_number=%s",
                        entry.chapter_id, rel_path, record.transcript_lines,
                        record.volume, record.volume_number, record.sequence_number,
                        record.chapter_name, record.chapter_number,
                    )
        except Exception as exc:
            msg = f"DB update failed: {exc}"
            logger.error(
                "TextExtractionStage.persist_transcript_paths: %s", msg, exc_info=True,
            )
            return {
                "success": False, "updated": updated, "skipped": skipped,
                "transcript_dir": None, "error": msg,
            }

        transcript_dir_rel = self._set_transcript_dir(selection)
        logger.info(
            "TextExtractionStage.persist_transcript_paths: complete -- job=%d "
            "selection=%s updated=%d skipped=%d job.transcript_dir=%s",
            self._job.id, selection, updated, skipped, transcript_dir_rel,
        )
        return {
            "success": True, "updated": updated, "skipped": skipped,
            "transcript_dir": transcript_dir_rel, "error": None,
        }
