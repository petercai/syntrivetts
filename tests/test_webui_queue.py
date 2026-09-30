from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from syntrive.bootstrap import bootstrap  # noqa: E402
from syntrive.db.models import SynthesisBatch, TranscriptChapter  # noqa: E402
from syntrive.db.session import get_db_session  # noqa: E402
from syntrive.services import batch_runner, queue_overview  # noqa: E402
from syntrive.services.job_lease import HolderIdentity, acquire_lease  # noqa: E402
from syntrive.services.pipeline_service import OnExisting, RunDisposition, StepOptions, run_step  # noqa: E402
from syntrive.webui.app import create_app  # noqa: E402
from syntrive.webui.shared.state import ServerInfo, open_web_repo  # noqa: E402
from syntrive.workflow.engine import WorkflowStep  # noqa: E402

EBOOKS = Path(__file__).resolve().parent.parent / "ebooks"
HX = {"hx-request": "true", "accept-language": "en"}
RUNNER = HolderIdentity(holder_id="r" * 32, holder_kind="tts_batch", pid=4242, hostname="gpu-box")


@pytest.fixture()
def env(tmp_path):
    job = bootstrap(tmp_path, EBOOKS / "Jan-Eyre-5.epub")
    db = tmp_path / "syntrivetts.db"
    for step in (WorkflowStep.BOOTSTRAP, WorkflowStep.TRANSCRIPT_EXTRACT, WorkflowStep.TRANSCRIPT_MERGE,
                 WorkflowStep.TRANSCRIPT_CLEAN, WorkflowStep.TRANSCRIPT_TEXT):
        assert run_step(db, job.id, step, on_existing=OnExisting.OVERWRITE, holder_kind="test").disposition == RunDisposition.DONE
    assert run_step(db, job.id, WorkflowStep.TRANSCRIPT_REVIEW, on_existing=OnExisting.OVERWRITE, holder_kind="test",
                    options=StepOptions(transcript_path="tts_script")).disposition == RunDisposition.DONE
    app = create_app(open_web_repo(tmp_path), ServerInfo(port=1, host="127.0.0.1", started_at="t0", version="test"), register=False)
    with TestClient(app) as client:
        yield {"client": client, "db": db, "job_id": job.id, "repo": tmp_path}


def _transcript(env, chapter_db_id: int) -> Path:
    with get_db_session(env["db"]) as s:
        chapter = s.get(TranscriptChapter, chapter_db_id)
        job_dir = env["repo"] / chapter.job.process_dir
        return job_dir / chapter.transcript_path


class TestReadiness:
    def test_reasons_come_from_the_real_files(self, env):
        db, job_id = env["db"], env["job_id"]
        chapters = queue_overview.read_book_queue(db, job_id).chapters
        assert all(c.ready for c in chapters) and len(chapters) >= 3
        edited, removed = chapters[0].chapter_db_id, chapters[1].chapter_db_id
        path = _transcript(env, edited)
        path.write_text(path.read_text(encoding="utf-8") + "\nan extra line added after step 6\n", encoding="utf-8")
        _transcript(env, removed).unlink()

        by_id = {c.chapter_db_id: c for c in queue_overview.read_book_queue(db, job_id).chapters}
        assert by_id[edited].reason == "line_mismatch" and by_id[edited].action is None
        assert by_id[removed].reason == "missing_file"
        assert all(by_id[c.chapter_db_id].ready for c in chapters[2:])


class TestPage:
    def test_books_and_default_preselection(self, env):
        html = env["client"].get("/queue", headers={"accept-language": "en"}).text
        book = queue_overview.read_book_queue(env["db"], env["job_id"])
        assert "Synthesis queue" in html and 'class="book-item"' in html
        assert html.count('data-action="enqueue"') == len(book.ready_ids())
        chapter_boxes = html.split('name="chapter"')[1:]
        assert sum("checked" in b.split(">", 1)[0] for b in chapter_boxes) == min(5, len(book.ready_ids()))
        assert f"Queue {min(5, len(book.ready_ids()))} selected" in html
        assert 'hx-trigger="every 2s"' not in html

    def test_enqueue_pause_resume_through_the_routes(self, env):
        client, db, job_id = env["client"], env["db"], env["job_id"]
        ready = queue_overview.read_book_queue(db, job_id).ready_ids()
        picked = list(ready[:2])
        queued = client.post("/api/v1/queue/enqueue", data={"book": job_id, "chapter": picked}, headers=HX)
        assert "notice=queued%3A2" in queued.headers["hx-redirect"]
        rows = {c.chapter_db_id: c for c in queue_overview.read_book_queue(db, job_id).chapters}
        assert all(rows[i].status == "queued" and rows[i].batch_id for i in picked)

        paused = client.post("/api/v1/queue/pause", data={"book": job_id, "chapter": picked[:1]}, headers=HX)
        assert "notice=paused%3A1" in paused.headers["hx-redirect"]
        with get_db_session(db) as s:
            assert s.get(SynthesisBatch, rows[picked[0]].batch_id).paused_at is not None
        resumed = client.post("/api/v1/queue/resume", data={"book": job_id, "chapter": picked[:1]}, headers=HX)
        assert "notice=resumed%3A1" in resumed.headers["hx-redirect"]
        with get_db_session(db) as s:
            assert s.get(SynthesisBatch, rows[picked[0]].batch_id).paused_at is None

        again = client.post("/api/v1/queue/enqueue", data={"book": job_id, "chapter": picked}, headers=HX)
        assert again.status_code == 422 and again.headers["hx-retarget"] == "#flash"
        nav = client.get("/queue").text
        assert 'nav-count is-accent">2<' in nav


class TestNow:
    def test_live_runner_strip_polls_and_idle_does_not(self, env):
        client, db, job_id = env["client"], env["db"], env["job_id"]
        ready = queue_overview.read_book_queue(db, job_id).ready_ids()
        client.post("/api/v1/queue/enqueue", data={"book": job_id, "chapter": list(ready[:1])}, headers=HX)
        batch_id = queue_overview.read_book_queue(db, job_id).chapters[0].batch_id
        log = env["repo"] / "logs" / "tts_batch.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(json.dumps({
            "event": "line_start", "ts": "2026-09-27T12:00:00+00:00", "batch_id": batch_id, "job_id": job_id,
            "chapter_id": "ch_0001", "book": "Jan Eyre", "book_seq": 1, "book_total": 1, "chapter_seq": 1,
            "chapter_total": 1, "line_idx": 7, "line_total": 28, "text": "‡voice:0‡Reader, I married him.‡break‡",
        }) + "\n", encoding="utf-8")

        idle = client.get("/api/v1/queue/now", headers=HX).text
        assert "No runner is working" in idle and "every 2s" not in idle

        acquire_lease(db, job_id, holder=RUNNER, operation=f"batch:{batch_id}")
        live = client.get("/api/v1/queue/now", headers=HX).text
        assert 'hx-trigger="every 2s"' in live and "line 7 / 28" in live and "gpu-box" in live
        assert 'class="chip chip-voice"' in live and "‡" not in live
        refused = client.post("/api/v1/queue/start", data={}, headers=HX)
        assert refused.status_code == 409 and "already synthesizing" in refused.text


class TestStartRunner:
    def test_detached_start_record_and_single_instance(self, env):
        db = env["db"]
        command = [sys.executable, "-c", "import time; print('runner up', flush=True); time.sleep(2)"]
        first = batch_runner.start_runner(db, command)
        assert first.started and first.pid
        record = json.loads((env["repo"] / ".syntrive" / "runner.json").read_text(encoding="utf-8"))
        assert record["pid"] == first.pid and record["command"] == command
        second = batch_runner.start_runner(db, command)
        assert not second.started and second.reason == "process_alive"
        assert queue_overview.read_now(db).running

        deadline = time.time() + 15
        while batch_runner.runner_process(env["repo"]).alive and time.time() < deadline:
            time.sleep(0.2)
        assert not batch_runner.runner_process(env["repo"]).alive
        assert "runner up" in (env["repo"] / "logs" / "tts_batch.console.log").read_text(encoding="utf-8")
        third = batch_runner.start_runner(db, [sys.executable, "-c", "pass"])
        assert third.started


class TestOfflineStart:
    def test_offline_flag_matches_tts_task_and_is_recorded(self, env):
        argv = batch_runner.default_command(env["repo"], offline=True)
        assert argv[-1] == "--offline" and argv[1].endswith("tts_batch.py")
        assert "--offline" not in batch_runner.default_command(env["repo"])
        started = batch_runner.start_runner(env["db"], [sys.executable, "-c", "import sys", "--offline"])
        assert started.started
        assert batch_runner.runner_process(env["repo"]).offline is True


class TestHistory:
    def test_outcomes_newest_first_and_per_book(self, env):
        from datetime import datetime, timedelta

        db, job_id = env["db"], env["job_id"]
        ready = queue_overview.read_book_queue(db, job_id).ready_ids()
        result = queue_overview_enqueue(db, list(ready[:3]))
        done_id, failed_id, stopped_id = result
        t0 = datetime(2026, 9, 27, 12, 0, 0)
        with get_db_session(db) as s:
            done = s.get(SynthesisBatch, done_id)
            done.synth_started_at, done.synth_finished_at = t0, t0 + timedelta(minutes=4)
            for c in s.query(TranscriptChapter).filter_by(synthesis_batch_id=done_id):
                c.synthesis_status, c.chapter_audio_seconds = "done", 183.5
            failed = s.get(SynthesisBatch, failed_id)
            failed.synth_started_at, failed.synth_error, failed.failed_line_index = t0 + timedelta(minutes=5), "engine crashed", 9
            s.get(SynthesisBatch, stopped_id).synth_started_at = t0 + timedelta(minutes=6)

        rows = queue_overview.read_history(db)
        assert [r.batch_id for r in rows] == [stopped_id, failed_id, done_id]
        by_id = {r.batch_id: r for r in rows}
        assert by_id[done_id].outcome == "done" and by_id[done_id].run_seconds == 240 and by_id[done_id].audio_seconds == 183.5
        assert by_id[failed_id].outcome == "failed" and by_id[failed_id].failed_line == 9
        assert by_id[stopped_id].outcome == "stopped"
        acquire_lease(db, job_id, holder=RUNNER, operation=f"batch:{stopped_id}")
        assert {r.batch_id: r.outcome for r in queue_overview.read_history(db, job_id)}[stopped_id] == "running"
        assert queue_overview.read_history(db, job_id + 999) == ()

    def test_pause_resume_buttons_follow_the_selection_contract(self, env):
        html = env["client"].get("/queue").text
        assert 'data-applies="pause"' in html and 'data-applies="resume"' in html


def queue_overview_enqueue(db, chapter_ids):
    from syntrive.services import synthesis_service

    result = synthesis_service.enqueue(db, chapter_ids)
    assert result.ok
    return result.batch_ids


class TestOfflineDefault:
    def test_environment_locks_dotenv_defaults(self, tmp_path):
        dotenv = tmp_path / ".env"
        assert batch_runner.offline_default({"SYNTRIVE_TTS_OFFLINE": "1"}, dotenv).locked
        dotenv.write_text("# local\nexport SYNTRIVE_TTS_OFFLINE='1'  # offline box\nOTHER=x\n", encoding="utf-8")
        d = batch_runner.offline_default({}, dotenv)
        assert (d.checked, d.locked, d.source) == (True, False, "dotenv")
        dotenv.write_text("SYNTRIVE_TTS_OFFLINE=1\nSYNTRIVE_TTS_OFFLINE=0\n", encoding="utf-8")
        assert batch_runner.offline_default({}, dotenv) == batch_runner.OfflineDefault(False, False, None)
        assert batch_runner.offline_default({}, tmp_path / "missing.env").source is None

    def test_page_shows_the_state(self, env, monkeypatch):
        monkeypatch.setenv("SYNTRIVE_TTS_OFFLINE", "1")
        html = env["client"].get("/queue", headers={"accept-language": "en"}).text
        box = html.split('name="offline"', 1)[1].split(">", 1)[0]
        assert "checked" in box and "disabled" in box and "Forced: the WebUI server runs" in html
        monkeypatch.delenv("SYNTRIVE_TTS_OFFLINE")
        box = env["client"].get("/queue").text.split('name="offline"', 1)[1].split(">", 1)[0]
        assert "disabled" not in box


class TestHistoryView:
    def test_history_tab_lists_the_books_runs(self, env):
        from datetime import datetime, timedelta

        client, db, job_id = env["client"], env["db"], env["job_id"]
        ids = queue_overview_enqueue(db, list(queue_overview.read_book_queue(db, job_id).ready_ids()[:2]))
        t0 = datetime(2026, 9, 27, 12, 0, 0)
        with get_db_session(db) as s:
            done = s.get(SynthesisBatch, ids[0])
            done.synth_started_at, done.synth_finished_at = t0, t0 + timedelta(seconds=372)
            for c in s.query(TranscriptChapter).filter_by(synthesis_batch_id=ids[0]):
                c.synthesis_status, c.chapter_audio_seconds = "done", 588
            failed = s.get(SynthesisBatch, ids[1])
            failed.synth_started_at, failed.synth_error, failed.failed_line_index = t0 + timedelta(minutes=7), "engine timeout", 17

        chapters_view = client.get("/queue", params={"book": job_id}).text
        assert "History (2)" in chapters_view and 'data-queue-form' in chapters_view
        html = client.get("/queue", params={"book": job_id, "view": "history"}, headers={"accept-language": "en"}).text
        assert 'data-queue-form' not in html and 'aria-selected="true">History (2)' in html
        assert html.index(f"#{ids[1]}") < html.index(f"#{ids[0]}")
        assert "6:12" in html and "9:48" in html and "h-done" in html and "h-failed" in html
        assert "Failed at line 17: engine timeout" in html and "stays queued" in html
        assert client.get("/queue", params={"book": job_id, "view": "bogus"}).status_code == 422
