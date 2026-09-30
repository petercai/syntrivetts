from __future__ import annotations

import json
from pathlib import Path

import pytest

from syntrive.bootstrap import bootstrap
from syntrive.db.models import SynthesisBatch
from syntrive.db.session import get_db_session
from syntrive.services import queue_overview as qo
from syntrive.services import synthesis_service
from syntrive.services.job_lease import HolderIdentity, acquire_lease
from syntrive.services.pipeline_service import OnExisting, RunDisposition, StepOptions, run_step
from syntrive.workflow.engine import WorkflowStep

EBOOKS = Path(__file__).resolve().parent.parent / "ebooks"
RUNNER = HolderIdentity(holder_id="r" * 32, holder_kind="tts_batch", pid=4242, hostname="gpu-box")


def _event(name: str, **fields) -> str:
    return json.dumps({"event": name, "ts": f"2026-09-27T12:00:{fields.pop('sec', 0):02d}+00:00", **fields}, ensure_ascii=False)


class TestPure:
    def test_parse_skips_garbage_and_partial_lines(self):
        lines = ['{"event": "line_start", "line_idx": 1', "", "not json", "[1, 2]", '{"no_event": 1}', _event("chapter_start", batch_id=3)]
        assert [e["event"] for e in qo.parse_events(lines)] == ["chapter_start"]

    def test_summarize_takes_the_latest_position_with_line_details(self):
        events = qo.parse_events([
            _event("synth_batch_start", batch_id=7, chapter_count=1),
            _event("line_start", batch_id=7, job_id=1, chapter_id="ch_0002", book="Jan Eyre", book_seq=1, book_total=2,
                   chapter_seq=2, chapter_total=5, line_idx=3, line_total=40, text="Reader, I married him."),
            _event("line_ok", batch_id=7, job_id=1, chapter_id="ch_0002", line_idx=3, sec=1),
        ])
        progress = qo.summarize(events)
        assert progress.event == "line_ok" and progress.batch_id == 7 and progress.book == "Jan Eyre"
        assert (progress.line_idx, progress.line_total, progress.chapter_total) == (3, 40, 5)
        assert progress.text == "Reader, I married him." and progress.fraction == pytest.approx(3 / 40)
        assert qo.summarize([]) is None

    def test_abort_carries_its_error(self):
        progress = qo.summarize(qo.parse_events([_event("batch_aborted", batch_id=2, job_id=1, chapter_id="ch_0001", error="engine crashed")]))
        assert progress.event == "batch_aborted" and progress.error == "engine crashed"


@pytest.fixture()
def repo(tmp_path):
    job = bootstrap(tmp_path, EBOOKS / "Jan-Eyre-5.epub")
    db = tmp_path / "syntrivetts.db"
    for step in (WorkflowStep.BOOTSTRAP, WorkflowStep.TRANSCRIPT_EXTRACT, WorkflowStep.TRANSCRIPT_MERGE,
                 WorkflowStep.TRANSCRIPT_CLEAN, WorkflowStep.TRANSCRIPT_TEXT):
        assert run_step(db, job.id, step, on_existing=OnExisting.OVERWRITE, holder_kind="test").disposition == RunDisposition.DONE
    assert run_step(db, job.id, WorkflowStep.TRANSCRIPT_REVIEW, on_existing=OnExisting.OVERWRITE, holder_kind="test",
                    options=StepOptions(transcript_path="tts_script")).disposition == RunDisposition.DONE
    return {"db": db, "job_id": job.id, "repo": tmp_path}


def test_empty_queue_before_anything_is_enqueued(repo):
    overview = qo.read_queue(repo["db"])
    assert overview.queue == () and overview.runner is None and overview.progress is None
    book = overview.books[0]
    assert book.counts == {"pending": book.chapters} and book.schedulable == book.chapters


def test_queue_order_pause_stuck_runner_and_live_progress(repo):
    db, job_id = repo["db"], repo["job_id"]
    chapters = synthesis_service.list_schedulable_books(db)[0].chapters
    assert len(chapters) >= 3
    result = synthesis_service.enqueue(db, [c.chapter_db_id for c in chapters[:3]])
    assert result.ok
    first, second, third = result.batch_ids
    assert synthesis_service.set_pause_state(db, pause_batch_ids=[first]).ok
    with get_db_session(db) as s:
        batch = s.get(SynthesisBatch, third)
        batch.synth_error, batch.failed_line_index = "engine crashed", 12

    log = repo["repo"] / "logs" / "tts_batch.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text("\n".join([
        _event("synth_batch_start", batch_id=second, chapter_count=1),
        _event("line_start", batch_id=second, job_id=job_id, chapter_id=chapters[1].chapter_id, book="Jan Eyre",
               book_seq=1, book_total=1, chapter_seq=1, chapter_total=2, line_idx=5, line_total=20, text="第五行"),
    ]) + "\n", encoding="utf-8")
    acquire_lease(db, job_id, holder=RUNNER, operation=f"batch:{second}")

    overview = qo.read_queue(db)
    assert [r.batch_id for r in overview.queue] == [second, third, first]
    assert overview.queue[-1].paused and overview.runnable_count == 2
    assert [r.batch_id for r in overview.stuck] == [third] and overview.stuck[0].failed_line == 12
    assert overview.runner.batch_id == second and overview.runner.hostname == "gpu-box"
    assert overview.live and overview.progress.line_idx == 5 and overview.progress.text == "第五行"

    book = overview.books[0]
    assert book.counts["queued"] == 3 and book.paused == 1
    assert book.schedulable == book.chapters - 3


def test_runner_on_this_host_with_a_dead_pid_is_not_live(repo):
    import socket

    ghost = HolderIdentity(holder_id="g" * 32, holder_kind="tts_batch", pid=999_999, hostname=socket.gethostname())
    acquire_lease(repo["db"], repo["job_id"], holder=ghost, operation="batch:1")
    assert qo.live_runner(repo["db"]) is None
