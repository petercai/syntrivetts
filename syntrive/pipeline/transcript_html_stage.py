from __future__ import annotations

import logging
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class TranscriptStageResult:
    success: bool
    artifacts: dict[str, int] = dc_field(default_factory=dict)
    notes: Optional[str] = None
    error: Optional[str] = None


class TranscriptHtmlStage:
    _OUTPUT_SUBDIRS = ("raw", "cleaned", "index", "manifest", "merged")

    def __init__(
        self,
        job,
        db_path: Path,
    ) -> None:
        from syntrive.db.path_utils import resolve_abs
        from syntrive.adapters.htmlclean.engine import HtmlCleaningEngine

        self._job = job
        self._db_path = db_path
        self._process_dir = resolve_abs(db_path, job.process_dir)
        self._output_dir = self._process_dir / "transcript_html"
        self._language: str = "en"
        self._group_plan = None
        self._engine = HtmlCleaningEngine(db_path=db_path, job_id=job.id)

    def run_extract(self) -> TranscriptStageResult:
        logger.info(
            "TranscriptHtmlStage.run_extract: start — job=%d epub=%s",
            self._job.id, self._job.epub_path,
        )
        self._record_stage_start()
        self._language = self._get_book_language()

        try:
            self._ensure_output_dirs()
            self._ensure_numeral_rules_template()

            raw_dir = self._output_dir / "raw"

            chapters = self._step_extract()
            if not chapters:
                msg = "No chapters extracted from EPUB TOC"
                logger.warning("TranscriptHtmlStage.run_extract: %s", msg)
                self._record_stage_failed(msg)
                return TranscriptStageResult(success=False, error=msg)

            self._group_plan = self._step_plan_hierarchy()

            raw_entries = self._step_extract_to_raw(chapters, raw_dir)
            raw_count = len(raw_entries)

            self._step_write_raw_toc(chapters, raw_entries, raw_dir)

            logger.info(
                "TranscriptHtmlStage.run_extract: complete — job=%d raw=%d",
                self._job.id, raw_count,
            )
            self._record_stage_complete({"raw": raw_count})
            return TranscriptStageResult(
                success=raw_count > 0,
                artifacts={"raw": raw_count},
                notes=(
                    f"{raw_count} spine file(s) written to raw/. "
                    "Review raw/0_toc.md before proceeding to merge."
                ),
            )

        except Exception as exc:
            logger.error(
                "TranscriptHtmlStage.run_extract: failed — job=%d error=%s",
                self._job.id, exc, exc_info=True,
            )
            self._record_stage_failed(str(exc))
            return TranscriptStageResult(success=False, error=str(exc))

    def run_merge(self) -> TranscriptStageResult:
        logger.info(
            "TranscriptHtmlStage.run_merge: start — job=%d",
            self._job.id,
        )
        self._record_stage_start()
        self._language = self._get_book_language()

        try:
            self._ensure_output_dirs()
            merged_dir = self._output_dir / "merged"

            chapters = self._step_extract()
            self._group_plan = self._step_plan_hierarchy()

            if not chapters:
                msg = "No chapters extracted from EPUB TOC"
                logger.warning("TranscriptHtmlStage.run_merge: %s", msg)
                self._record_stage_failed(msg)
                return TranscriptStageResult(success=False, error=msg)

            chapter_files, discard_count = self._step_merge_to_chapters(chapters, merged_dir)
            regular_count = len(chapter_files) - discard_count

            logger.info(
                "TranscriptHtmlStage.run_merge: complete — job=%d chapters=%d discard=%d",
                self._job.id, regular_count, discard_count,
            )
            self._record_stage_complete({"chapters": regular_count, "discard": discard_count})
            return TranscriptStageResult(
                success=regular_count > 0,
                artifacts={"chapters": regular_count, "discard": discard_count},
                notes=(
                    f"{regular_count} chapter(s) merged into merged/. "
                    f"{discard_count} discard file(s) (ch_9NNN) for ops review. "
                    "Edit merged/0_toc.md (excluded=y) to skip chapters before cleaning."
                ),
            )

        except Exception as exc:
            logger.error(
                "TranscriptHtmlStage.run_merge: failed — job=%d error=%s",
                self._job.id, exc, exc_info=True,
            )
            self._record_stage_failed(str(exc))
            return TranscriptStageResult(success=False, error=str(exc))

    def run_clean(self) -> TranscriptStageResult:
        logger.info(
            "TranscriptHtmlStage.run_clean: start — job=%d",
            self._job.id,
        )
        self._record_stage_start()
        self._language = self._get_book_language()

        try:
            self._ensure_output_dirs()
            self._ensure_numeral_rules_template()

            from syntrive.adapters.epub.merged_toc_writer import MergedTocReader
            toc_path = self._output_dir / "merged" / "0_toc.md"
            _, entries = MergedTocReader().read(toc_path)

            if not entries:
                msg = (
                    "merged/0_toc.md is missing or empty — "
                    "run TRANSCRIPT_MERGE before TRANSCRIPT_CLEAN"
                )
                logger.warning("TranscriptHtmlStage.run_clean: %s", msg)
                self._record_stage_failed(msg)
                return TranscriptStageResult(success=False, error=msg)

            clean_entries = [
                e for e in entries if not e.excluded and not e.is_discard
            ]
            skipped = len(entries) - len(clean_entries)
            if skipped:
                logger.info(
                    "TranscriptHtmlStage.run_clean: %d entries excluded/discard — "
                    "processing %d",
                    skipped, len(clean_entries),
                )

            if not clean_entries:
                msg = (
                    f"All {len(entries)} entries are excluded or discarded in "
                    "merged/0_toc.md — nothing to clean"
                )
                logger.warning("TranscriptHtmlStage.run_clean: %s", msg)
                self._record_stage_failed(msg)
                return TranscriptStageResult(success=False, error=msg)

            self._group_plan = self._step_plan_hierarchy()
            counts = self._run_merged_clean_loop(clean_entries)
            self._write_cleaned_toc(entries)

            logger.info(
                "TranscriptHtmlStage.run_clean: complete — job=%d artifacts=%s",
                self._job.id, counts,
            )
            self._record_stage_complete(counts)
            return TranscriptStageResult(
                success=counts.get("html", 0) > 0,
                artifacts=counts,
                notes=(
                    f"{counts.get('html', 0)} chapter(s) cleaned and tagged. "
                    f"{counts.get('high_deletion', 0)} had >35% deletion "
                    "(see log warnings; processed anyway)."
                ),
            )

        except Exception as exc:
            logger.error(
                "TranscriptHtmlStage.run_clean: failed — job=%d error=%s",
                self._job.id, exc, exc_info=True,
            )
            self._record_stage_failed(str(exc))
            return TranscriptStageResult(success=False, error=str(exc))

    def run(self) -> TranscriptStageResult:
        logger.info(
            "TranscriptHtmlStage.run: start — job=%d epub=%s",
            self._job.id, self._job.epub_path,
        )
        self._record_stage_start()

        for phase_fn, phase_name in (
            (self.run_extract, "extract"),
            (self.run_merge, "merge"),
            (self.run_clean, "clean"),
        ):
            result = phase_fn()
            if not result.success:
                self._record_stage_failed(result.error or phase_name)
                return result

        logger.info("TranscriptHtmlStage.run: all phases complete — job=%d", self._job.id)
        self._record_stage_complete(result.artifacts)
        return result

    def _step_extract(self) -> list:
        from syntrive.adapters.epub.html_extractor import EpubHtmlExtractor
        from syntrive.db.path_utils import resolve_abs

        extractor = EpubHtmlExtractor(resolve_abs(self._db_path, self._job.epub_path))
        chapters = extractor.extract()
        logger.info("TranscriptHtmlStage: step extract — %d chapters from TOC", len(chapters))
        return chapters

    def _step_plan_hierarchy(self):
        from syntrive.adapters.epub.toc_planner import TocHierarchyPlanner
        from syntrive.db.path_utils import resolve_abs

        try:
            plan = TocHierarchyPlanner(resolve_abs(self._db_path, self._job.epub_path)).plan()
            logger.info(
                "TranscriptHtmlStage: step plan — TOC mode=%s groups=%d",
                plan.mode, len(plan.groups),
            )
            return plan
        except Exception as exc:
            logger.warning(
                "TranscriptHtmlStage: TocHierarchyPlanner failed (%s) — using flat mode", exc
            )
            from syntrive.adapters.epub.toc_planner import HtmlGroupPlan, GroupEntry
            return HtmlGroupPlan(
                mode="flat",
                groups=[GroupEntry(group_id="main", title="", href_set=frozenset())],
            )

    def _step_extract_to_raw(self, chapters: list, raw_dir: Path) -> list:
        from syntrive.adapters.epub.spine_extractor import SpineExtractor
        from syntrive.adapters.epub.html_merger import toc_hrefs_from_chapters
        from syntrive.db.path_utils import resolve_abs

        epub_abs = resolve_abs(self._db_path, self._job.epub_path)
        toc_hrefs = toc_hrefs_from_chapters(chapters)
        extractor = SpineExtractor(epub_abs)
        entries = extractor.extract(raw_dir, toc_hrefs=toc_hrefs)
        logger.info(
            "TranscriptHtmlStage: step extract_to_raw — %d spine files in raw/", len(entries)
        )
        return entries

    def _step_write_raw_toc(self, chapters: list, raw_entries: list, raw_dir: Path) -> None:
        from syntrive.adapters.epub.toc_md_writer import TocMdWriter

        sorted_chapters = sorted(chapters, key=lambda c: c.order)
        toc_entries = [
            (ch.title, ch.href, self._heading_level_for_order(ch.order))
            for ch in sorted_chapters
        ]
        toc_href_basenames = {ch.href.split("/")[-1].lower() for ch in chapters}
        orphan_hrefs = [
            e.href for e in raw_entries
            if e.href.split("/")[-1].lower() not in toc_href_basenames
        ]
        TocMdWriter().write(
            toc_entries, orphan_hrefs, raw_dir / "0_toc.md",
            book_title=self._get_book_title(),
        )
        logger.info(
            "TranscriptHtmlStage: step write_raw_toc — %d TOC, %d orphans",
            len(toc_entries), len(orphan_hrefs),
        )

    def _heading_level_for_order(self, order: int) -> int:
        if self._group_plan is None:
            return 2
        depth = self._group_plan.get_node_info(order).depth
        return min(depth + 2, 6)

    def _step_merge_to_chapters(
        self, chapters: list, merged_dir: Path
    ) -> tuple[list, int]:
        from syntrive.adapters.epub.html_merger import HtmlMerger, toc_hrefs_from_chapters
        from syntrive.adapters.epub.spine_scanner import SpineOrderScanner
        from syntrive.adapters.epub.merged_toc_writer import (
            MergedTocWriter, MergedTocEntry, MergedTocMeta,
        )
        from syntrive.db.path_utils import resolve_abs

        epub_abs = resolve_abs(self._db_path, self._job.epub_path)
        override = self._get_merge_mode_override()
        if override:
            logger.info(
                "TranscriptHtmlStage: merge_mode_override=%r applied", override
            )

        try:
            toc_hrefs = toc_hrefs_from_chapters(chapters)
            spine_entries = SpineOrderScanner(epub_abs).scan(toc_hrefs)
        except Exception as exc:
            logger.warning(
                "TranscriptHtmlStage: SpineOrderScanner failed (%s) — spine excluded", exc
            )
            spine_entries = None

        book_title = self._get_book_title()
        chapter_files = HtmlMerger().merge(
            chapters=chapters,
            spine_entries=spine_entries,
            group_plan=self._group_plan,
            book_title=book_title,
            override=override,
        )

        written = 0
        for cf in chapter_files:
            try:
                out_path = merged_dir / f"{cf.chapter_id}.html"
                out_path.write_text(cf.body_html, encoding="utf-8")
                written += 1
                logger.debug(
                    "TranscriptHtmlStage: merged/%s.html (%d bytes)",
                    cf.chapter_id, len(cf.body_html.encode("utf-8")),
                )
            except Exception as exc:
                logger.warning(
                    "TranscriptHtmlStage: could not write merged/%s.html: %s",
                    cf.chapter_id, exc,
                )

        reset_per_vol = self._get_chapter_number_reset_per_volume()
        merge_mode = chapter_files[0].merge_mode if chapter_files else "flat"

        toc_entries = _build_toc_entries(
            chapter_files,
            group_plan=self._group_plan,
            reset_per_vol=reset_per_vol and (merge_mode == "volume_split"),
        )
        MergedTocWriter().write(
            toc_entries,
            MergedTocMeta(book_title=book_title, merge_mode=merge_mode),
            merged_dir / "0_toc.md",
        )

        discard_count = sum(1 for cf in chapter_files if cf.is_discard)
        logger.info(
            "TranscriptHtmlStage: step merge_to_chapters — %d files written "
            "(%d chapters, %d discard) mode=%s",
            written, written - discard_count, discard_count, merge_mode,
        )
        return chapter_files, discard_count

    def _run_merged_clean_loop(self, entries: list) -> dict[str, int]:
        from syntrive.adapters.html.para_id_mapper import ParagraphIdMapper
        from syntrive.adapters.html.artifact_writer import HtmlArtifactWriter
        from syntrive.pipeline.threshold_guard import ThresholdGuard
        from bs4 import BeautifulSoup

        merged_dir = self._output_dir / "merged"
        mapper = ParagraphIdMapper()
        writer = HtmlArtifactWriter(self._output_dir)
        numeral_rules = self._load_global_numeral_rules() or None

        html_count = 0
        high_deletion_count = 0

        for entry in entries:
            chapter_id = entry.chapter_id
            merged_file = merged_dir / f"{chapter_id}.html"

            if not merged_file.exists():
                logger.warning(
                    "TranscriptHtmlStage: merged/%s.html not found — "
                    "was run_merge() completed successfully?",
                    chapter_id,
                )
                continue

            try:
                raw_html = merged_file.read_text(encoding="utf-8")

                cleaned_html, stats = self._engine.run(
                    raw_html,
                    language=self._language,
                    numeral_rules=numeral_rules,
                )

                entry.applied_rules = sorted(stats.removals_by_rule.keys())

                guard = ThresholdGuard(
                    job_id=self._job.id,
                    chapter_id=chapter_id,
                    db_path=self._db_path,
                )
                if guard.check(
                    raw_chars=stats.raw_chars,
                    cleaned_chars=stats.cleaned_chars,
                    chapter_title=entry.title,
                    rule_distribution=stats.removals_by_rule,
                ):
                    high_deletion_count += 1

                cleaned_soup = BeautifulSoup(cleaned_html, "html.parser")
                para_map = mapper.tag(cleaned_soup, chapter_id=chapter_id)
                tagged_html = str(cleaned_soup)

                writer.write_cleaned(
                    chapter_id=chapter_id,
                    cleaned_html=tagged_html,
                    para_map=para_map,
                )

                html_count += 1
                self._persist_chapter_from_entry(entry, raw_html, stats)

            except Exception as exc:
                logger.error(
                    "TranscriptHtmlStage: chapter=%s failed: %s",
                    chapter_id, exc, exc_info=True,
                )

        return {"html": html_count, "high_deletion": high_deletion_count}

    def _write_cleaned_toc(self, entries: list) -> None:
        from syntrive.adapters.epub.merged_toc_writer import (
            MergedTocWriter, MergedTocMeta, apply_toc_reset_flags,
        )

        processable = [e for e in entries if not e.is_discard and not e.excluded]

        has_resets = any(
            e.sequence_reset or e.chapter_reset or e.chapter_clear for e in processable
        )
        if has_resets:
            logger.info(
                "TranscriptHtmlStage: applying TOC reset flags from merged/0_toc.md"
                " before writing cleaned/0_toc.md"
            )
            processable = apply_toc_reset_flags(processable)

        cleaned_toc_path = self._output_dir / "cleaned" / "0_toc.md"
        MergedTocWriter().write(
            processable, MergedTocMeta(book_title=self._get_book_title()), cleaned_toc_path
        )
        logger.info(
            "TranscriptHtmlStage: wrote cleaned/0_toc.md — %d chapters "
            "(%d excluded/discard omitted)",
            len(processable),
            len(entries) - len(processable),
        )

    def _get_merge_mode_override(self) -> str | None:
        try:
            from syntrive.db.session import get_db_session
            from syntrive.db.models import Job

            with get_db_session(self._db_path) as db:
                job = db.query(Job).filter_by(id=self._job.id).first()
                if job is not None:
                    return job.merge_mode_override
        except Exception as exc:
            logger.debug("TranscriptHtmlStage: could not read merge_mode_override: %s", exc)
        return None

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
                "TranscriptHtmlStage: could not read chapter_number_reset_per_volume: %s", exc
            )
        return False

    def _load_global_numeral_rules(self) -> list:
        from syntrive.adapters.html.para_id_mapper import (
            load_global_numeral_rules, NUMERAL_RULES_FILENAME,
        )

        rules_path = self._output_dir / NUMERAL_RULES_FILENAME
        rules = load_global_numeral_rules(rules_path)
        if rules:
            logger.info(
                "TranscriptHtmlStage: applying %d global numeral rule(s) to all chapters",
                len(rules),
            )
        else:
            logger.debug(
                "TranscriptHtmlStage: no numeral rules active (edit %s to add rules)",
                rules_path.name,
            )
        return rules

    def _get_book_title(self) -> str:
        try:
            from syntrive.db.session import get_db_session
            from syntrive.db.models import Book

            with get_db_session(self._db_path) as db:
                book = db.query(Book).filter_by(id=self._job.book_id).first()
                if book and book.title:
                    return book.title
        except Exception as exc:
            logger.debug("TranscriptHtmlStage: could not query book title: %s", exc)
        return ""

    def _ensure_numeral_rules_template(self) -> None:
        from syntrive.adapters.html.para_id_mapper import (
            create_numeral_rules_template, NUMERAL_RULES_FILENAME,
        )
        create_numeral_rules_template(self._output_dir / NUMERAL_RULES_FILENAME)

    def _persist_chapter_from_entry(self, entry, raw_html: str, stats) -> None:
        try:
            from syntrive.db.session import get_db_session
            from syntrive.db.models import TranscriptChapter
            from syntrive.db.path_utils import to_repo_relative

            chapter_id = entry.chapter_id
            source_href = entry.source_files[0] if entry.source_files else ""

            raw_path = to_repo_relative(
                self._db_path,
                self._output_dir / "raw" / f"{chapter_id}.html",
            )
            cleaned_path = to_repo_relative(
                self._db_path,
                self._output_dir / "cleaned" / f"{chapter_id}.html",
            )
            map_path = to_repo_relative(
                self._db_path,
                self._output_dir / "manifest" / f"{chapter_id}.para_map.yaml",
            )

            group_id, _ = (
                self._group_plan.get_group_for_href(source_href)
                if self._group_plan and source_href
                else ("main", "")
            )

            with get_db_session(self._db_path) as db:
                existing = (
                    db.query(TranscriptChapter)
                    .filter_by(job_id=self._job.id, chapter_id=chapter_id)
                    .first()
                )
                if existing:
                    row = existing
                else:
                    row = TranscriptChapter(
                        job_id=self._job.id,
                        chapter_id=chapter_id,
                    )
                    db.add(row)

                row.title = entry.title
                row.source_href = source_href
                row.group_id = group_id
                row.raw_html_path = raw_path
                row.cleaned_html_path = cleaned_path
                row.para_id_map_path = map_path
                row.raw_chars = stats.raw_chars
                row.cleaned_chars = stats.cleaned_chars
                row.deletion_ratio = stats.deletion_ratio
                row.threshold_blocked = False
                row.synthesis_status = "pending"

        except Exception as exc:
            logger.warning(
                "TranscriptHtmlStage: could not persist chapter record: %s", exc
            )

    def _get_book_language(self) -> str:
        try:
            from syntrive.db.session import get_db_session
            from syntrive.db.models import Book
            from syntrive.adapters.epub.lang import normalize_language

            with get_db_session(self._db_path) as db:
                book = db.query(Book).filter_by(id=self._job.book_id).first()
                if book and book.language:
                    normalized = normalize_language(book.language)
                    if normalized:
                        return normalized
                    logger.warning(
                        "TranscriptHtmlStage: unrecognised book language=%r for job=%d"
                        " — defaulting to 'en'",
                        book.language, self._job.id,
                    )
        except Exception as exc:
            logger.debug("TranscriptHtmlStage: could not query book language: %s", exc)
        return "en"

    def _ensure_output_dirs(self) -> None:
        for sub in self._OUTPUT_SUBDIRS:
            (self._output_dir / sub).mkdir(parents=True, exist_ok=True)
        logger.debug("TranscriptHtmlStage: output dirs ready at %s", self._output_dir)

    def _record_stage_start(self) -> None:
        self._record_event("started", None, None)

    def _record_stage_complete(self, artifacts: dict) -> None:
        self._record_event("completed", None, artifacts)

    def _record_stage_failed(self, error: str) -> None:
        self._record_event("failed", error, None)

    def _record_event(
        self,
        status: str,
        error_message: Optional[str],
        artifact_paths: Optional[dict],
    ) -> None:
        try:
            from syntrive.db.session import get_db_session
            from syntrive.db.repository import JobRepository

            with get_db_session(self._db_path) as db:
                repo = JobRepository(db)
                job = repo.get_job(self._job.id)
                if job is None:
                    return
                if status == "started":
                    repo.record_stage_start(job, "transcript")
                elif status == "completed":
                    repo.record_stage_complete(
                        job, "transcript", artifact_paths=artifact_paths
                    )
                else:
                    repo.record_stage_failed(
                        job, "transcript",
                        error_message=error_message or "unknown error",
                    )
        except Exception as exc:
            logger.warning("TranscriptHtmlStage: could not record stage event: %s", exc)


def _basename(href: str) -> str:
    return Path(href).name


def _derive_volume_info(cf, group_plan) -> tuple[str, str]:
    if group_plan is None or not cf.source_hrefs:
        return "", ""
    for href in cf.source_hrefs:
        group_id, group_title = group_plan.get_group_for_href(href)
        if group_id != "main":
            vol_num = group_id.split("_", 1)[1] if "_" in group_id else group_id
            return vol_num, group_title
    return "", ""


def _build_toc_entries(
    chapter_files: list,
    group_plan,
    reset_per_vol: bool,
) -> list:
    from syntrive.adapters.epub.merged_toc_writer import MergedTocEntry

    vol_chapter_counter: dict[str, int] = {}
    entries: list[MergedTocEntry] = []

    for cf in chapter_files:
        vol_num, vol_name = _derive_volume_info(cf, group_plan)
        seq_num = cf.chapter_id.split("_", 1)[1]

        if reset_per_vol and vol_num:
            vol_chapter_counter[vol_num] = vol_chapter_counter.get(vol_num, 0) + 1
            chapter_num = f"{vol_chapter_counter[vol_num]:04d}"
        else:
            chapter_num = seq_num

        entries.append(MergedTocEntry(
            chapter_id=cf.chapter_id,
            title=cf.title,
            source_files=[_basename(h) for h in cf.source_hrefs],
            excluded=cf.is_discard or cf.default_excluded,
            is_discard=cf.is_discard,
            volume_name=vol_name,
            volume_number=vol_num,
            chapter_number=chapter_num,
            heading_level=cf.heading_level,
        ))

    return entries
