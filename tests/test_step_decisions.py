from __future__ import annotations

from pathlib import Path

import pytest

from syntrive.adapters.epub.merged_toc_writer import MergedTocMeta, MergedTocReader, MergedTocWriter, set_excluded_flags
from syntrive.bootstrap import bootstrap
from syntrive.db.models import Job, TranscriptChapter
from syntrive.db.session import get_db_session
from syntrive.services import step_decisions as sd
from syntrive.services.job_lease import HolderIdentity, JobLeaseConflict, acquire_lease
from syntrive.services.pipeline_service import OnExisting, RunDisposition, StepOptions, run_step
from syntrive.workflow.engine import WorkflowEngine, WorkflowStep

EBOOKS = Path(__file__).resolve().parent.parent / "ebooks"
FOREIGN = HolderIdentity(holder_id="f" * 32, holder_kind="tts_batch", pid=4242, hostname="other-host")


def _run(db: Path, job_id: int, step: WorkflowStep, **kw):
    result = run_step(db, job_id, step, on_existing=OnExisting.OVERWRITE, holder_kind="test", **kw)
    assert result.disposition == RunDisposition.DONE, result.error
    return result


def _process_dir(db: Path, job_id: int) -> Path:
    with get_db_session(db) as s:
        return db.parent / s.get(Job, job_id).process_dir


@pytest.fixture()
def merged(tmp_path):
    job = bootstrap(tmp_path, EBOOKS / "Jan-Eyre-5.epub")
    db = tmp_path / "syntrivetts.db"
    for step in (WorkflowStep.BOOTSTRAP, WorkflowStep.TRANSCRIPT_EXTRACT, WorkflowStep.TRANSCRIPT_MERGE):
        _run(db, job.id, step)
    return db, job.id


class TestSetExcludedFlags:
    TEXT = (
        "# Book\r\n\r\n## [One](ch_0001.html)\r\n  - chapter_name: One\r\n  - excluded: n\r\n"
        "  - chapter_reset: y\r\n### [Two](ch_0002.html)\r\n  - excluded: y\r\n\r\n---\r\n\r\n"
        "| orphan_files | source_files |\r\n|---|---|\r\n| [ch_9000.html](ch_9000.html) | x |\r\n"
    )

    def test_flips_only_excluded_lines_and_keeps_line_endings(self):
        new, changed = set_excluded_flags(self.TEXT, frozenset({"ch_0001"}))
        assert changed == ("ch_0001", "ch_0002")
        assert new.replace("excluded: y", "X").replace("excluded: n", "X") == \
            self.TEXT.replace("excluded: y", "X").replace("excluded: n", "X")
        assert "  - excluded: y\r\n  - chapter_reset: y" in new
        assert new.count("\r\n") == self.TEXT.count("\r\n")

    def test_no_change_returns_same_text(self):
        new, changed = set_excluded_flags(self.TEXT, frozenset({"ch_0002"}))
        assert changed == () and new == self.TEXT


class TestMergeModeRoundTrip:
    def test_writer_records_mode_and_reader_reads_it(self, tmp_path):
        toc = tmp_path / "0_toc.md"
        MergedTocWriter().write([], MergedTocMeta(book_title="B", merge_mode="volume_split"), toc)
        assert MergedTocReader().read(toc)[0].merge_mode == "volume_split"

    def test_older_file_without_comment_is_unknown_not_flat(self, tmp_path):
        toc = tmp_path / "0_toc.md"
        toc.write_text("# B\n\n## [One](ch_0001.html)\n  - excluded: n\n", encoding="utf-8")
        assert MergedTocReader().read(toc)[0].merge_mode == ""


class TestMergeSettings:
    def test_read_after_merge_reports_the_detected_mode(self, merged):
        db, job_id = merged
        settings = sd.read_merge_settings(db, job_id)
        assert settings.detected in sd.MERGE_MODES
        assert settings.override is None and settings.reset_per_volume is False

    def test_save_partial_update_and_rerun_hint(self, merged):
        db, job_id = merged
        result = sd.save_merge_settings(db, job_id, override="flat", holder_kind="test")
        assert result.changed and result.rerun_from == WorkflowStep.TRANSCRIPT_MERGE
        again = sd.save_merge_settings(db, job_id, reset_per_volume=True, holder_kind="test")
        settings = sd.read_merge_settings(db, job_id)
        assert again.changed and settings.override == "flat" and settings.reset_per_volume is True
        assert sd.save_merge_settings(db, job_id, override="flat", holder_kind="test").changed is False
        with pytest.raises(sd.DecisionError) as err:
            sd.save_merge_settings(db, job_id, override="sideways", holder_kind="test")
        assert err.value.code == "invalid_merge_mode"

    def test_engine_setters_use_the_service(self, merged):
        db, job_id = merged
        with get_db_session(db) as s:
            job = s.get(Job, job_id)
            s.expunge_all()
        engine = WorkflowEngine(job=job, db_path=db, holder_kind="tui")
        engine.set_merge_mode_override("volume_continuous")
        engine.set_chapter_number_reset_per_volume(True)
        settings = sd.read_merge_settings(db, job_id)
        assert settings.override == "volume_continuous" and engine.get_chapter_number_reset_per_volume() is True


class TestChapterSelection:
    def test_exclusion_edits_the_toc_and_clean_skips_the_chapter(self, merged):
        db, job_id = merged
        selection = sd.read_chapter_selection(db, job_id)
        assert selection.available and len(selection.chapters) >= 2
        assert all(c.char_count > 0 for c in selection.chapters if c.size_bytes)
        target = selection.chapters[-1].chapter_id
        toc = _process_dir(db, job_id) / "transcript_html" / "merged" / "0_toc.md"
        before = toc.read_text(encoding="utf-8").splitlines()

        keep_excluded = {c.chapter_id for c in selection.chapters if c.excluded}
        result = sd.save_exclusions(db, job_id, keep_excluded | {target}, holder_kind="test")
        assert result.changed and result.rerun_from is None

        after = toc.read_text(encoding="utf-8").splitlines()
        diff = [(a, b) for a, b in zip(before, after) if a != b]
        assert len(before) == len(after) and len(diff) == 1 and diff[0][1].strip() == "- excluded: y"
        assert sd.read_chapter_selection(db, job_id).excluded_count == len(keep_excluded) + 1

        _run(db, job_id, WorkflowStep.TRANSCRIPT_CLEAN)
        cleaned = _process_dir(db, job_id) / "transcript_html" / "cleaned"
        assert not (cleaned / f"{target}.html").exists()
        assert any(cleaned.glob("ch_0*.html"))
        stale = sd.save_exclusions(db, job_id, keep_excluded, holder_kind="test")
        assert stale.changed and stale.rerun_from == WorkflowStep.TRANSCRIPT_CLEAN

    def test_refusals(self, merged, tmp_path):
        db, job_id = merged
        with pytest.raises(sd.DecisionError) as err:
            sd.save_exclusions(db, job_id, {"ch_4242"}, holder_kind="test")
        assert err.value.code == "unknown_chapter"
        fresh = bootstrap(tmp_path / "other", EBOOKS / "English-2.epub")
        with pytest.raises(sd.DecisionError) as err:
            sd.save_exclusions(tmp_path / "other" / "syntrivetts.db", fresh.id, set(), holder_kind="test")
        assert err.value.code == "not_merged"

    def test_foreign_lease_blocks_the_submit(self, merged):
        db, job_id = merged
        acquire_lease(db, job_id, holder=FOREIGN, operation="batch:1")
        target = sd.read_chapter_selection(db, job_id).chapters[0].chapter_id
        with pytest.raises(JobLeaseConflict):
            sd.save_exclusions(db, job_id, {target}, holder_kind="webui")
        assert not sd.read_chapter_selection(db, job_id).chapters[0].excluded


class TestCleaningRules:
    def test_save_read_back_and_results_after_clean(self, merged):
        db, job_id = merged
        rules = sd.read_cleaning_rules(db, job_id)
        assert rules and [r.sort_order for r in rules] == sorted(r.sort_order for r in rules)
        first = rules[0]
        result = sd.save_cleaning_rules(db, job_id, {first.name: not first.enabled}, holder_kind="test")
        assert result.changed and result.rerun_from is None
        assert sd.read_cleaning_rules(db, job_id)[0].enabled is (not first.enabled)
        assert sd.save_cleaning_rules(db, job_id, {first.name: not first.enabled}, holder_kind="test").changed is False
        with pytest.raises(sd.DecisionError):
            sd.save_cleaning_rules(db, job_id, {"no_such_rule": True}, holder_kind="test")

        assert sd.read_clean_results(db, job_id) == ()
        _run(db, job_id, WorkflowStep.TRANSCRIPT_CLEAN)
        results = sd.read_clean_results(db, job_id)
        assert results and all(r.cleaned_chars is not None for r in results)


class TestRunOptions:
    def test_text_formats_then_review_with_the_chosen_path(self, merged):
        db, job_id = merged
        _run(db, job_id, WorkflowStep.TRANSCRIPT_CLEAN)
        _run(db, job_id, WorkflowStep.TRANSCRIPT_TEXT, options=StepOptions(text_formats=frozenset({"raw"}), text_primary="raw"))
        text_dir = _process_dir(db, job_id) / "transcript_text"
        assert any((text_dir / "raw").glob("ch_*.txt"))
        assert not any((text_dir / "tts_script").glob("ch_*.txt"))

        review = sd.read_transcript_review(db, job_id)
        raw = next(f for f in review.formats if f.name == "raw")
        assert raw.available and raw.files and all(f.exists and f.lines > 0 for f in raw.files)
        assert not next(f for f in review.formats if f.name == "tts_script").available
        text, truncated = sd.read_transcript_text(db, job_id, "raw", raw.files[0].file_name, limit=40)
        assert 0 < len(text) <= 40 and truncated is True

        refused = run_step(db, job_id, WorkflowStep.TRANSCRIPT_REVIEW, holder_kind="test")
        assert refused.disposition == RunDisposition.REFUSED and "decision" in refused.error
        _run(db, job_id, WorkflowStep.TRANSCRIPT_REVIEW, options=StepOptions(transcript_path="raw"))
        with get_db_session(db) as s:
            paths = [c.transcript_path for c in s.query(TranscriptChapter).filter_by(job_id=job_id) if c.transcript_path]
            current = s.get(Job, job_id).current_step
        assert paths and all("/raw/" in p for p in paths)
        assert current == "tts_config"
        assert sd.read_transcript_review(db, job_id).selected == "raw"

    @pytest.mark.parametrize(
        "step, options",
        [
            (WorkflowStep.TRANSCRIPT_CLEAN, StepOptions(text_formats=frozenset({"raw"}))),
            (WorkflowStep.TRANSCRIPT_CLEAN, StepOptions(transcript_path="raw")),
        ],
    )
    def test_options_for_another_step_are_refused(self, merged, step, options):
        db, job_id = merged
        result = run_step(db, job_id, step, holder_kind="test", options=options)
        assert result.disposition == RunDisposition.REFUSED and "only" in result.error

    def test_bad_text_options_are_refused_before_running(self, merged):
        db, job_id = merged
        _run(db, job_id, WorkflowStep.TRANSCRIPT_CLEAN)
        for options in (
            StepOptions(text_formats=frozenset()),
            StepOptions(text_formats=frozenset({"pdf"})),
            StepOptions(text_formats=frozenset({"raw"}), text_primary="tts_script"),
        ):
            result = run_step(db, job_id, WorkflowStep.TRANSCRIPT_TEXT, holder_kind="test", options=options)
            assert result.disposition == RunDisposition.REFUSED, options
        assert not (_process_dir(db, job_id) / "transcript_text" / "raw").exists() or not any(
            (_process_dir(db, job_id) / "transcript_text" / "raw").glob("ch_*.txt")
        )

    def test_transcript_preview_refuses_path_tricks(self, merged):
        db, job_id = merged
        for fmt, name in (("raw", "../0_toc.md"), ("raw", "ch_0001.html"), ("html", "ch_0001.txt")):
            with pytest.raises(sd.DecisionError):
                sd.read_transcript_text(db, job_id, fmt, name)
