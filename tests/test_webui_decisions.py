from __future__ import annotations

import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("multipart")

from fastapi.testclient import TestClient  # noqa: E402

from syntrive.bootstrap import bootstrap  # noqa: E402
from syntrive.db.models import Job, TranscriptChapter  # noqa: E402
from syntrive.db.session import get_db_session  # noqa: E402
from syntrive.services import step_decisions as sd  # noqa: E402
from syntrive.services.pipeline_service import OnExisting, RunDisposition, run_step  # noqa: E402
from syntrive.webui.app import create_app  # noqa: E402
from syntrive.webui.shared.state import ServerInfo, open_web_repo  # noqa: E402
from syntrive.workflow.engine import WorkflowStep  # noqa: E402

EBOOKS = Path(__file__).resolve().parent.parent / "ebooks"
HX = {"hx-request": "true", "accept-language": "en"}


def _run(db: Path, job_id: int, step: WorkflowStep) -> None:
    result = run_step(db, job_id, step, on_existing=OnExisting.OVERWRITE, holder_kind="test")
    assert result.disposition == RunDisposition.DONE, result.error


def _job(db: Path, job_id: int) -> Job:
    with get_db_session(db) as s:
        job = s.get(Job, job_id)
        s.expunge_all()
        return job


def _wait(client: TestClient, job_id: int, timeout: float = 60) -> str:
    deadline = time.time() + timeout
    while time.time() < deadline:
        response = client.get(f"/api/v1/pipeline/{job_id}/run", headers=HX)
        if response.status_code == 204:
            return response.headers.get("hx-redirect", "")
        time.sleep(0.1)
    raise AssertionError("run did not settle")


@pytest.fixture()
def env(tmp_path):
    job = bootstrap(tmp_path, EBOOKS / "Jan-Eyre-5.epub")
    db = tmp_path / "syntrivetts.db"
    for step in (WorkflowStep.BOOTSTRAP, WorkflowStep.TRANSCRIPT_EXTRACT, WorkflowStep.TRANSCRIPT_MERGE):
        _run(db, job.id, step)
    app = create_app(open_web_repo(tmp_path), ServerInfo(port=1, host="127.0.0.1", started_at="t0", version="test"), register=False)
    with TestClient(app) as client:
        yield {"client": client, "db": db, "job_id": job.id}


def _page(env, step: str, **params) -> str:
    response = env["client"].get(f"/pipeline/{env['job_id']}", params={"step": step, **params}, headers={"accept-language": "en"})
    assert response.status_code == 200, response.text[:500]
    return response.text


class TestMergePage:
    def test_settings_and_chapter_checklist_render_real_data(self, env):
        html = _page(env, "transcript_merge")
        selection = sd.read_chapter_selection(env["db"], env["job_id"])
        assert 'name="override"' in html and 'name="reset_per_volume"' in html
        assert "data-selection" in html and "Save selection" in html
        for chapter in selection.chapters:
            assert f'value="{chapter.chapter_id}"' in html
        assert "/static/pipeline/pipeline.js" in html

    def test_preview_drawer_and_bad_ids(self, env):
        client, job_id = env["client"], env["job_id"]
        first = sd.read_chapter_selection(env["db"], job_id).chapters[0]
        drawer = client.get(f"/api/v1/pipeline/{job_id}/chapter-text", params={"chapter_id": first.chapter_id}, headers=HX)
        assert drawer.status_code == 200 and f"merged/{first.chapter_id}.html" in drawer.text and "<p>" in drawer.text
        bad = client.get(f"/api/v1/pipeline/{job_id}/chapter-text", params={"chapter_id": "../0_toc"}, headers=HX)
        assert bad.status_code == 422 and bad.headers["hx-retarget"] == "#flash"

    def test_save_merge_settings(self, env):
        client, job_id = env["client"], env["job_id"]
        saved = client.post(f"/api/v1/pipeline/{job_id}/merge-settings", data={"override": "flat", "reset_per_volume": "1"}, headers=HX)
        assert saved.status_code == 200 and "notice=stale%3Atranscript_merge" in saved.headers["hx-redirect"]
        settings = sd.read_merge_settings(env["db"], job_id)
        assert settings.override == "flat" and settings.reset_per_volume is True
        bad = client.post(f"/api/v1/pipeline/{job_id}/merge-settings", data={"override": "sideways"}, headers=HX)
        assert bad.status_code == 422 and "Unknown merge mode" in bad.text


class TestSelectionThenClean:
    def test_exclusion_clean_stale_notice_and_rerun(self, env):
        client, db, job_id = env["client"], env["db"], env["job_id"]
        chapters = sd.read_chapter_selection(db, job_id).chapters
        dropped = chapters[-1].chapter_id
        keep = [c.chapter_id for c in chapters if not c.excluded and c.chapter_id != dropped]

        saved = client.post(f"/api/v1/pipeline/{job_id}/exclusions", data={"include": keep}, headers=HX)
        assert saved.headers["hx-redirect"].endswith("notice=saved")
        assert next(c for c in sd.read_chapter_selection(db, job_id).chapters if c.chapter_id == dropped).excluded

        clean_page = _page(env, "transcript_clean")
        assert f"{len(keep)} of {len(chapters)} chapters are cleaned" in clean_page
        assert 'role="switch" name="rule"' in clean_page and "No results yet" in clean_page

        client.post(f"/api/v1/pipeline/{job_id}/run", data={"step": "transcript_clean"}, headers=HX)
        _wait(client, job_id)
        cleaned = db.parent / _job(db, job_id).process_dir / "transcript_html" / "cleaned"
        assert not (cleaned / f"{dropped}.html").exists()
        assert "Last clean" in _page(env, "transcript_clean") and "−" in _page(env, "transcript_clean")

        stale = client.post(f"/api/v1/pipeline/{job_id}/exclusions", data={"include": keep + [dropped]}, headers=HX)
        assert "notice=stale%3Atranscript_clean" in stale.headers["hx-redirect"]
        notice_page = _page(env, "transcript_merge", notice="stale:transcript_clean")
        assert "/rerun" in notice_page and "must run again" in notice_page

        rerun = client.post(f"/api/v1/pipeline/{job_id}/rerun", data={"step": "transcript_clean"}, headers=HX)
        assert rerun.headers["hx-redirect"].startswith(f"/pipeline/{job_id}?step=transcript_clean")
        _wait(client, job_id)
        assert (cleaned / f"{dropped}.html").exists()

    def test_rules_save_marks_clean_stale(self, env):
        client, db, job_id = env["client"], env["db"], env["job_id"]
        _run(db, job_id, WorkflowStep.TRANSCRIPT_CLEAN)
        rules = sd.read_cleaning_rules(db, job_id)
        on = [r.name for r in rules if r.enabled][1:]
        response = client.post(f"/api/v1/pipeline/{job_id}/cleaning-rules", data={"rule": on}, headers=HX)
        assert "notice=stale%3Atranscript_clean" in response.headers["hx-redirect"]
        assert [r.name for r in sd.read_cleaning_rules(db, job_id) if r.enabled] == on


class TestTextAndReview:
    def _to_text(self, env) -> None:
        _run(env["db"], env["job_id"], WorkflowStep.TRANSCRIPT_CLEAN)

    def test_text_options_render_and_run_raw_only(self, env):
        self._to_text(env)
        client, db, job_id = env["client"], env["db"], env["job_id"]
        html = _page(env, "transcript_text")
        assert 'name="text_format" value="raw"' in html and 'name="text_primary" value="tts_script" checked' in html

        empty = client.post(f"/api/v1/pipeline/{job_id}/run", data={"step": "transcript_text", "text_options": "1"}, headers=HX)
        assert empty.status_code == 422 and "at least one format" in empty.text

        client.post(f"/api/v1/pipeline/{job_id}/run", data={
            "step": "transcript_text", "text_options": "1", "text_format": ["raw"], "text_primary": "raw"}, headers=HX)
        _wait(client, job_id)
        text_dir = db.parent / _job(db, job_id).process_dir / "transcript_text"
        assert any((text_dir / "raw").glob("ch_*.txt")) and not any((text_dir / "tts_script").glob("ch_*.txt"))

        review = _page(env, "transcript_review")
        assert "Use raw for synthesis and continue" in review and 'name="transcript_path" value="raw"' in review
        first = sd.read_transcript_review(db, job_id).formats[0].files[0]
        reader = client.get(f"/api/v1/pipeline/{job_id}/transcript", params={"fmt": "raw", "file": first.file_name}, headers=HX)
        assert reader.status_code == 200 and f"raw/{first.file_name}" in reader.text

        client.post(f"/api/v1/pipeline/{job_id}/run", data={"step": "transcript_review", "transcript_path": "raw"}, headers=HX)
        _wait(client, job_id)
        with get_db_session(db) as s:
            paths = [c.transcript_path for c in s.query(TranscriptChapter).filter_by(job_id=job_id) if c.transcript_path]
        assert paths and all("/raw/" in p for p in paths)
        assert _job(db, job_id).current_step == "tts_config"

    def test_reader_draws_markers_as_chips(self, env):
        self._to_text(env)
        client, job_id = env["client"], env["job_id"]
        client.post(f"/api/v1/pipeline/{job_id}/run", data={"step": "transcript_text"}, headers=HX)
        _wait(client, job_id)
        files = sd.read_transcript_review(env["db"], job_id).formats[1].files
        html = "".join(client.get(f"/api/v1/pipeline/{job_id}/transcript", params={"fmt": "tts_script", "file": f.file_name}).text for f in files)
        assert 'class="chip' in html and "‡" not in html

    def test_review_without_a_decision_is_refused(self, env):
        self._to_text(env)
        client, job_id = env["client"], env["job_id"]
        client.post(f"/api/v1/pipeline/{job_id}/run", data={"step": "transcript_text"}, headers=HX)
        _wait(client, job_id)
        refused = client.post(f"/api/v1/pipeline/{job_id}/run", data={"step": "transcript_review"}, headers=HX)
        assert refused.status_code == 422


def test_split_markers_is_pure():
    lines = sd.split_markers("‡voice:0‡Hello.‡break‡\n\nNext‡pause‡ line")
    assert [[(p.text, p.marker) for p in line] for line in lines] == [
        [("voice:0", True), ("Hello.", False), ("break", True)],
        [("Next", False), ("pause", True), (" line", False)],
    ]
