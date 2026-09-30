from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from syntrive.services.book_catalog import BookRow, filter_rows, tab_counts
from syntrive.services.pipeline_service import (
    PIPELINE_STEPS,
    STEP_MODES,
    OnExisting,
    RunAlreadyActive,
    RunDisposition,
    StepMode,
    StepRunRegistry,
    StepRunResult,
    StepState,
    build_pipeline_view,
)
from syntrive.workflow.engine import WorkflowStep, step_from_db_value


class TestStepFromDbValue:
    @pytest.mark.parametrize(
        "value, expected",
        [
            (None, WorkflowStep.BOOTSTRAP),
            ("", WorkflowStep.BOOTSTRAP),
            ("garbage", WorkflowStep.BOOTSTRAP),
            ("transcript", WorkflowStep.TRANSCRIPT_EXTRACT),
            ("review_html", WorkflowStep.TRANSCRIPT_CLEAN),
            ("transcript_normalize", WorkflowStep.TRANSCRIPT_REVIEW),
            ("tts_config", WorkflowStep.TTS_CONFIG),
            ("done", WorkflowStep.DONE),
        ],
    )
    def test_maps_current_and_legacy_values(self, value, expected):
        assert step_from_db_value(value) == expected


class TestPipelineView:
    def test_nine_steps_every_one_has_a_mode(self):
        assert len(PIPELINE_STEPS) == 9
        assert WorkflowStep.DONE not in PIPELINE_STEPS
        assert set(STEP_MODES) == set(PIPELINE_STEPS)

    def test_current_step_splits_done_and_later(self):
        view = build_pipeline_view("transcript_clean", "running")
        assert view.current == WorkflowStep.TRANSCRIPT_CLEAN
        assert view.current_number == 4
        assert [s.state for s in view.steps] == [StepState.DONE] * 3 + [StepState.CURRENT] + [StepState.LATER] * 5
        assert [s.number for s in view.steps] == list(range(1, 10))

    def test_new_job_starts_at_import(self):
        view = build_pipeline_view(None, "pending")
        assert view.current == WorkflowStep.BOOTSTRAP
        assert view.steps[0].state == StepState.CURRENT

    @pytest.mark.parametrize("current, status", [("done", "completed"), ("combine", "completed")])
    def test_finished_job_fills_the_track(self, current, status):
        view = build_pipeline_view(current, status)
        assert view.finished and view.current is None
        assert view.current_number == 9
        assert all(s.state == StepState.DONE for s in view.steps)

    def test_modes_match_the_design(self):
        run_steps = tuple(s for s in PIPELINE_STEPS if STEP_MODES[s] == StepMode.RUN)
        assert run_steps == PIPELINE_STEPS[:5] + (WorkflowStep.TTS_CONFIG,)
        assert STEP_MODES[WorkflowStep.SYNTHESIS] == StepMode.QUEUE


def _row(job_id: int, title: str, *, author: str = "", current: str = "bootstrap", status: str = "pending", archived: bool = False) -> BookRow:
    return BookRow(
        job_id=job_id, book_id=job_id, title=title, author=author, language="en",
        process_dir=Path("."), cover_path=None, status=status, archived=archived,
        chapter_count=0, audio_chapter_count=0, updated_at=None,
        pipeline=build_pipeline_view(current, status),
    )


class TestCatalogFilters:
    ROWS = (
        _row(1, "1984", author="Orwell, George"),
        _row(2, "被讨厌的勇气", author="岸见一郎"),
        _row(3, "Done Book", current="done", status="completed"),
        _row(4, "Old Book", archived=True),
    )

    def test_tabs_partition_the_rows(self):
        assert tab_counts(self.ROWS) == {"active": 2, "done": 1, "archived": 1}
        assert [r.job_id for r in filter_rows(self.ROWS, tab="archived")] == [4]

    @pytest.mark.parametrize("query, expected", [("orwell", [1]), ("勇气", [2]), ("  ", [1, 2]), ("zzz", [])])
    def test_search_matches_title_or_author(self, query, expected):
        assert [r.job_id for r in filter_rows(self.ROWS, tab="active", query=query)] == expected


class TestStepRunRegistry:
    def test_one_run_per_job_and_result_kept(self, tmp_path):
        release = threading.Event()
        calls = []

        def runner(db_path, job_id, step, *, on_existing, holder_kind, options):
            calls.append((job_id, step, on_existing, holder_kind))
            release.wait(5)
            return StepRunResult(job_id, step, RunDisposition.DONE, artifacts={"n": 1})

        registry = StepRunRegistry(runner=runner)
        db = tmp_path / "syntrivetts.db"
        run = registry.start(db, 7, WorkflowStep.TRANSCRIPT_EXTRACT, on_existing=OnExisting.ASK, holder_kind="webui")
        assert run.running and registry.any_running()
        with pytest.raises(RunAlreadyActive):
            registry.start(db, 7, WorkflowStep.TRANSCRIPT_EXTRACT, on_existing=OnExisting.ASK, holder_kind="webui")
        registry.start(db, 8, WorkflowStep.BOOTSTRAP, on_existing=OnExisting.KEEP, holder_kind="webui")

        release.set()
        deadline = time.time() + 5
        while registry.any_running() and time.time() < deadline:
            time.sleep(0.02)
        finished = registry.get(db, 7)
        assert not finished.running
        assert finished.result.disposition == RunDisposition.DONE and finished.result.artifacts == {"n": 1}
        assert (7, WorkflowStep.TRANSCRIPT_EXTRACT, OnExisting.ASK, "webui") in calls
        registry.start(db, 7, WorkflowStep.TRANSCRIPT_EXTRACT, on_existing=OnExisting.ASK, holder_kind="webui")

    def test_a_crashing_runner_still_settles_as_failed(self, tmp_path):
        def runner(*args, **kwargs):
            raise RuntimeError("boom")

        registry = StepRunRegistry(runner=runner)
        db = tmp_path / "syntrivetts.db"
        registry.start(db, 1, WorkflowStep.BOOTSTRAP, on_existing=OnExisting.ASK, holder_kind="webui")
        deadline = time.time() + 5
        while registry.any_running() and time.time() < deadline:
            time.sleep(0.02)
        result = registry.get(db, 1).result
        assert result.disposition == RunDisposition.FAILED and "boom" in result.error
