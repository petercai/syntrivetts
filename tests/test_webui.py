from __future__ import annotations

import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("jinja2")
pytest.importorskip("multipart")

from fastapi.testclient import TestClient  # noqa: E402

from syntrive.bootstrap import bootstrap  # noqa: E402
from syntrive.db.models import Job, WorkflowStepEvent  # noqa: E402
from syntrive.db.session import get_db_session  # noqa: E402
from syntrive.io import server_registry  # noqa: E402
from syntrive.services.job_lease import HolderIdentity, acquire_lease, get_lease  # noqa: E402
from syntrive.webui.app import create_app  # noqa: E402
from syntrive.webui.shared.state import ServerInfo, open_web_repo  # noqa: E402

EBOOKS = Path(__file__).resolve().parent.parent / "ebooks"
HX = {"hx-request": "true"}
FOREIGN = HolderIdentity(holder_id="f" * 32, holder_kind="tts_batch", pid=4242, hostname="other-host")


def _job(db_path: Path, job_id: int) -> Job:
    with get_db_session(db_path) as db:
        job = db.get(Job, job_id)
        db.expunge_all()
        return job


def _wait_settled(client: TestClient, job_id: int, timeout: float = 60) -> str:
    deadline = time.time() + timeout
    while time.time() < deadline:
        response = client.get(f"/api/v1/pipeline/{job_id}/run", headers=HX)
        if response.status_code == 204:
            return response.headers.get("hx-redirect") or response.headers.get("hx-refresh")
        assert response.status_code == 200 and "every 2s" in response.text
        time.sleep(0.1)
    raise AssertionError("step run did not settle")


@pytest.fixture()
def repo(tmp_path):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    job = bootstrap(repo_dir, EBOOKS / "English-2.epub")
    return {"repo": open_web_repo(repo_dir), "job_id": job.id}


@pytest.fixture()
def client(repo):
    info = ServerInfo(port=1, host="127.0.0.1", started_at="t0", version="test")
    app = create_app(repo["repo"], info, register=False)
    with TestClient(app) as c:
        yield c


class TestShell:
    def test_health_answers_for_the_resolved_repo(self, client, repo):
        body = client.get("/api/v1/health").json()
        assert body["repo_dir"] == str(repo["repo"].repo_dir)
        assert body["version"] == "test"

    def test_root_redirects_to_books(self, client):
        response = client.get("/", follow_redirects=False)
        assert response.status_code == 303 and response.headers["location"] == "/books"

    def test_books_page_lists_the_book_with_sidebar_and_request_id(self, client, repo):
        response = client.get("/books")
        assert response.status_code == 200
        assert f'href="/pipeline/{repo["job_id"]}"' in response.text
        assert "1/9" in response.text and "Import book" in response.text
        assert 'class="nav-link"' in response.text and "Maintenance" in response.text
        assert len(response.headers["x-request-id"]) == 8
        mark = response.text.split('<a class="mark"', 1)[1].split("</a>", 1)[0]
        assert 'd="M2 11h3M7 6v10M11 3v16M15 7v8M19 10v2"' in mark and "<rect" not in mark
        favicon = client.get("/static/shared/favicon.svg")
        assert favicon.status_code == 200 and "M16 8v16" in favicon.text and 'fill="#3F8F58"' not in favicon.text

    def test_language_follows_browser_then_cookie(self, client):
        assert "添加书籍" in client.get("/books", headers={"accept-language": "zh-CN,zh"}).text
        response = client.post("/api/v1/repo/lang", data={"lang": "en"}, headers=HX)
        assert response.status_code == 204 and response.headers["hx-refresh"] == "true"
        assert "Add a book" in client.get("/books", headers={"accept-language": "zh-CN"}).text

    def test_unknown_book_is_a_404_page(self, client):
        response = client.get("/pipeline/9999")
        assert response.status_code == 404 and "Back to books" in response.text


class TestBooks:
    def test_upload_rejects_non_epub_into_flash(self, client):
        response = client.post("/api/v1/books", files={"file": ("notes.txt", b"hi", "text/plain")}, headers=HX)
        assert response.status_code == 422
        assert response.headers["hx-retarget"] == "#flash"
        assert "EPUB" in response.text

    def test_upload_bootstraps_a_new_book(self, client, repo):
        data = (EBOOKS / "English-1.epub").read_bytes()
        response = client.post("/api/v1/books", files={"file": ("English-1.epub", data, "application/epub+zip")}, headers=HX)
        assert response.status_code == 200
        new_id = int(response.headers["hx-redirect"].rsplit("/", 1)[1])
        assert new_id != repo["job_id"]
        job = _job(repo["repo"].db_path, new_id)
        assert (repo["repo"].repo_dir / job.epub_path).is_file()
        assert not any((repo["repo"].repo_dir / ".syntrive" / "uploads").glob("*/*"))

    def test_archive_hides_then_restore_brings_back(self, client, repo):
        job_id = repo["job_id"]
        response = client.post(f"/api/v1/books/{job_id}/archive", data={"tab": "active"}, headers=HX)
        assert response.status_code == 200
        assert 'hx-swap-oob="innerHTML"' in response.text
        assert f'href="/pipeline/{job_id}"' not in response.text
        assert _job(repo["repo"].db_path, job_id).archived_at is not None

        archived = client.get("/api/v1/books/rows", params={"tab": "archived"}).text
        assert f'href="/pipeline/{job_id}"' in archived
        refused = client.post(f"/api/v1/pipeline/{job_id}/run", data={"step": "bootstrap"}, headers=HX)
        assert refused.status_code == 422 and "archived" in refused.text

        client.post(f"/api/v1/books/{job_id}/restore", data={"tab": "archived"}, headers=HX)
        assert _job(repo["repo"].db_path, job_id).archived_at is None

    def test_hard_delete_needs_the_exact_title(self, client, repo):
        job_id = repo["job_id"]
        job = _job(repo["repo"].db_path, job_id)
        process_dir = repo["repo"].repo_dir / job.process_dir
        dialog = client.get(f"/api/v1/books/{job_id}/delete", headers=HX)
        assert dialog.status_code == 200 and "<dialog" in dialog.text and str(process_dir) in dialog.text

        wrong = client.post(f"/api/v1/books/{job_id}/delete", data={"confirm_title": "nope"}, headers=HX)
        assert wrong.status_code == 422 and process_dir.is_dir()

        title = dialog.text.split('data-match="', 1)[1].split('"', 1)[0]
        ok = client.post(f"/api/v1/books/{job_id}/delete", data={"confirm_title": title}, headers=HX)
        assert ok.status_code == 200 and ok.headers["hx-redirect"] == "/books"
        assert not process_dir.exists() and _job(repo["repo"].db_path, job_id) is None

    def test_delete_under_a_foreign_lease_is_409_with_unlock(self, client, repo):
        job_id = repo["job_id"]
        db_path = repo["repo"].db_path
        acquire_lease(db_path, job_id, holder=FOREIGN, operation="batch:1")
        title = client.get(f"/api/v1/books/{job_id}/delete").text.split('data-match="', 1)[1].split('"', 1)[0]
        response = client.post(f"/api/v1/books/{job_id}/delete", data={"confirm_title": title}, headers=HX)
        assert response.status_code == 409
        assert f"/api/v1/leases/{job_id}/unlock" in response.text and "tts_batch" in response.text
        assert _job(db_path, job_id) is not None

        unlock = client.post(f"/api/v1/leases/{job_id}/unlock", headers=HX)
        assert unlock.status_code == 204 and get_lease(db_path, job_id) is None


class TestPipeline:
    def test_run_steps_in_order_with_poll_and_result(self, client, repo):
        job_id = repo["job_id"]
        db_path = repo["repo"].db_path

        started = client.post(f"/api/v1/pipeline/{job_id}/run", data={"step": "bootstrap"}, headers=HX)
        assert started.status_code == 200
        redirect = _wait_settled(client, job_id)
        assert redirect == f"/pipeline/{job_id}?step=bootstrap"
        assert _job(db_path, job_id).current_step == "transcript_extract"

        page = client.get(redirect).text
        assert "result-ok" in page and "Continue to Extract chapters" in page

        client.post(f"/api/v1/pipeline/{job_id}/run", data={"step": "transcript_extract"}, headers=HX)
        _wait_settled(client, job_id)
        job = _job(db_path, job_id)
        assert job.current_step == "transcript_merge"
        assert (repo["repo"].repo_dir / job.process_dir / "transcript_html" / "raw" / "0_toc.md").is_file()

        with get_db_session(db_path) as db:
            actions = [(e.step_name, e.action) for e in db.query(WorkflowStepEvent).filter_by(job_id=job_id)]
        assert ("bootstrap", "confirmed") in actions and ("transcript_extract", "confirmed") in actions
        assert "Extract chapters" in client.get(f"/pipeline/{job_id}?tab=activity").text

    def test_only_the_current_runnable_step_may_run(self, client, repo):
        job_id = repo["job_id"]
        not_current = client.post(f"/api/v1/pipeline/{job_id}/run", data={"step": "transcript_merge"}, headers=HX)
        assert not_current.status_code == 422 and "current step" in not_current.text
        unknown = client.post(f"/api/v1/pipeline/{job_id}/run", data={"step": "bogus"}, headers=HX)
        assert unknown.status_code == 422

    def test_restart_then_existing_output_asks_and_keep_advances(self, client, repo):
        job_id = repo["job_id"]
        db_path = repo["repo"].db_path
        client.post(f"/api/v1/pipeline/{job_id}/run", data={"step": "bootstrap"}, headers=HX)
        _wait_settled(client, job_id)

        refused = client.post(f"/api/v1/pipeline/{job_id}/restart", data={"step": "transcript_merge"}, headers=HX)
        assert refused.status_code == 422
        restarted = client.post(f"/api/v1/pipeline/{job_id}/restart", data={"step": "bootstrap"}, headers=HX)
        assert restarted.headers["hx-redirect"] == f"/pipeline/{job_id}?step=bootstrap"
        assert _job(db_path, job_id).current_step == "bootstrap"

        ask = client.post(f"/api/v1/pipeline/{job_id}/run", data={"step": "bootstrap"}, headers=HX)
        assert ask.status_code == 200 and "Keep it and continue" in ask.text
        assert _job(db_path, job_id).current_step == "bootstrap"

        client.post(f"/api/v1/pipeline/{job_id}/run", data={"step": "bootstrap", "on_existing": "keep"}, headers=HX)
        _wait_settled(client, job_id)
        assert _job(db_path, job_id).current_step == "transcript_extract"
        with get_db_session(db_path) as db:
            assert db.query(WorkflowStepEvent).filter_by(job_id=job_id, step_name="bootstrap", action="skipped").count() == 1

    def test_foreign_lease_blocks_the_run_before_it_starts(self, client, repo):
        job_id = repo["job_id"]
        acquire_lease(repo["repo"].db_path, job_id, holder=FOREIGN, operation="batch:1")
        response = client.post(f"/api/v1/pipeline/{job_id}/run", data={"step": "bootstrap"}, headers=HX)
        assert response.status_code == 409 and "Unlock" in response.text
        assert client.app.state.runs.get(repo["repo"].db_path, job_id) is None

    def test_decision_steps_explain_where_to_continue(self, client, repo):
        page = client.get(f"/pipeline/{repo['job_id']}?step=combine").text
        assert "syntrive.py -r" in page
        page = client.get(f"/pipeline/{repo['job_id']}?step=synthesis").text
        assert "tts_task.py" in page and "tts_batch.py" in page


class TestRepoSwitch:
    def test_switch_moves_the_discovery_record(self, repo, tmp_path):
        other_dir = tmp_path / "other"
        other_dir.mkdir()
        bootstrap(other_dir, EBOOKS / "English-1.epub")
        first = repo["repo"]
        info = ServerInfo(port=1, host="127.0.0.1", started_at="t0", version="test")
        app = create_app(first, info, register=True)
        with TestClient(app) as c:
            assert server_registry.read_record(first.repo_dir, "webui") is not None
            bad = c.post("/api/v1/repo/switch", data={"repo_dir": str(tmp_path)}, headers=HX)
            assert bad.status_code == 422 and app.state.web == first

            ok = c.post("/api/v1/repo/switch", data={"repo_dir": str(other_dir)}, headers=HX)
            assert ok.headers["hx-redirect"] == "/books"
            assert app.state.web.repo_dir == other_dir.resolve()
            assert server_registry.read_record(first.repo_dir, "webui") is None
            assert server_registry.read_record(other_dir, "webui").port == 1
            assert c.get("/api/v1/health").json()["repo_dir"] == str(other_dir.resolve())
        assert server_registry.read_record(other_dir, "webui") is None


class TestLanToken:
    def test_token_required_then_cookie(self, repo):
        info = ServerInfo(port=1, host="0.0.0.0", started_at="t0", version="test", token="s3cret-token")
        app = create_app(repo["repo"], info, register=False)
        with TestClient(app) as c:
            assert c.get("/books").status_code == 401
            assert c.get("/books?token=wrong").status_code == 401
            assert c.get("/api/v1/health").status_code == 200
            first = c.get("/books?token=s3cret-token", follow_redirects=False)
            assert first.status_code == 303 and "token" not in first.headers["location"]
            assert c.get("/books").status_code == 200
