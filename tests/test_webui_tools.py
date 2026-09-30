from __future__ import annotations

from pathlib import Path
from urllib.parse import unquote_plus

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("multipart")

from fastapi.testclient import TestClient  # noqa: E402

from syntrive.bootstrap import bootstrap  # noqa: E402
from syntrive.db.models import Job, ReferenceVoice  # noqa: E402
from syntrive.db.session import get_db_session  # noqa: E402
from syntrive.services import db_tools_service as dbt  # noqa: E402
from syntrive.services.job_lease import HolderIdentity, acquire_lease, release_lease  # noqa: E402
from syntrive.webui.app import create_app  # noqa: E402
from syntrive.webui.shared.state import ServerInfo, open_web_repo  # noqa: E402

EBOOKS = Path(__file__).resolve().parent.parent / "ebooks"
HX = {"hx-request": "true", "accept-language": "en"}
EN = {"accept-language": "en"}


@pytest.fixture()
def env(tmp_path):
    source = tmp_path / "source"
    source_job = bootstrap(source, EBOOKS / "Jan-Eyre-5.epub")
    with get_db_session(source / "syntrivetts.db") as db:
        db.add(ReferenceVoice(path="voices/en/adult/male/adam.wav", name="adam", gender="male", language="en"))
    target = tmp_path / "target"
    target_job = bootstrap(target, EBOOKS / "Jan-Eyre-5.epub")
    app = create_app(open_web_repo(target), ServerInfo(port=0, host="127.0.0.1", started_at="now", version="t"), register=False)
    return TestClient(app), source, source_job.id, target, target_job.id


def _export_into(source: Path, job_id: int, target: Path) -> str:
    out = dbt.export_to_repo(source / "syntrivetts.db", [job_id], include_reference_voices=True).output_path
    (target / out.name).write_bytes(out.read_bytes())
    return out.name


def test_page_ledger_and_sidebar(env):
    client, source, source_job, target, _ = env
    empty = client.get("/tools", headers=EN).text
    assert "No export or backup files yet." in empty and ">Maintenance<" in empty
    name = _export_into(source, source_job, target)
    (target / "garbage.db").write_bytes(b"not a database")
    page = client.get("/tools", headers=EN).text
    assert name in page and "1 books · 1 jobs" in page and "Not a readable export file" in page
    assert "syntrivetts.db<" not in page.replace("next to syntrivetts.db<", "")
    assert '>2</span>' in page


def test_export_from_dialog_writes_a_self_contained_file(env):
    client, _, _, target, target_job = env
    dialog = client.get("/api/v1/tools/export", headers=HX).text
    assert f'name="job" value="{target_job}"' in dialog and "PROCESSING-Jan-Eyre-5" in dialog
    assert client.post("/api/v1/tools/export", headers=HX, data={}).status_code == 422
    r = client.post("/api/v1/tools/export", headers=HX, data={"job": [str(target_job)], "include_voices": "1"})
    assert r.status_code == 200 and "exported" in r.headers["HX-Redirect"]
    (export,) = [p for p in target.glob("syntrivetts-export-*.db")]
    alone = target.parent / "alone.db"
    alone.write_bytes(export.read_bytes())
    assert dbt.describe_export_file(alone).job_count == 1


def test_import_plan_skip_and_overwrite(env):
    client, source, source_job, target, target_job = env
    name = _export_into(source, source_job, target)
    dialog = client.get(f"/api/v1/tools/files/{name}/import", headers=HX).text
    assert f"exists here · job #{target_job}" in dialog and f'name="mode_{source_job}"' in dialog

    skipped = client.post(f"/api/v1/tools/files/{name}/import", headers=HX, data={"job": [str(source_job)]})
    assert skipped.status_code == 200 and "imported%3A0%3A0%3A1" in skipped.headers["HX-Redirect"]
    done = client.post(f"/api/v1/tools/files/{name}/import", headers=HX,
                       data={"job": [str(source_job)], f"mode_{source_job}": "overwrite"})
    assert done.status_code == 200 and "imported%3A0%3A1%3A0" in done.headers["HX-Redirect"]
    with get_db_session(target / "syntrivetts.db") as db:
        assert db.query(Job).count() == 1
        assert [rv.name for rv in db.query(ReferenceVoice)] == ["adam"]
    assert client.post(f"/api/v1/tools/files/{name}/import", headers=HX, data={}).status_code == 422


def test_overwrite_refused_under_another_holders_lease(env):
    client, source, source_job, target, target_job = env
    name = _export_into(source, source_job, target)
    db_path = target / "syntrivetts.db"
    other = HolderIdentity.current("tui")
    assert acquire_lease(db_path, target_job, holder=other, operation="editing").ok
    try:
        r = client.post(f"/api/v1/tools/files/{name}/import", headers=HX,
                        data={"job": [str(source_job)], f"mode_{source_job}": "overwrite"})
        assert r.status_code == 409 and f"/api/v1/leases/{target_job}/unlock" in r.text
        with get_db_session(db_path) as db:
            assert db.get(Job, target_job) is not None and db.query(ReferenceVoice).count() == 0
    finally:
        release_lease(db_path, target_job, other.holder_id)


def test_backup_upload_restore_and_collision(env):
    client, source, _, target, _ = env
    path, voices, _, _ = dbt.backup_to_repo(source / "syntrivetts.db")
    assert voices == 1
    up = client.post("/api/v1/tools/upload", headers=HX, files={"file": (path.name, path.read_bytes(), "application/sql")})
    assert up.status_code == 200 and "uploaded" in up.headers["HX-Redirect"]
    dialog = client.get(f"/api/v1/tools/files/{path.name}/restore", headers=HX).text
    assert "1 voices · 0 tags" in dialog
    ok = client.post(f"/api/v1/tools/files/{path.name}/restore", headers=HX)
    assert ok.status_code == 200 and "restored%3A1%3A0" in ok.headers["HX-Redirect"]
    again = client.post(f"/api/v1/tools/files/{path.name}/restore", headers=HX)
    assert again.status_code == 409 and "Nothing was changed" in again.text
    with get_db_session(target / "syntrivetts.db") as db:
        assert db.query(ReferenceVoice).count() == 1

    second = client.post("/api/v1/tools/upload", headers=HX, files={"file": (path.name, path.read_bytes(), "application/sql")})
    assert f"{Path(path.name).stem} (2).sql" in unquote_plus(second.headers["HX-Redirect"])
    assert client.post("/api/v1/tools/backup", headers=HX).status_code == 200
    assert len(list(target.glob("*.sql"))) == 3


def test_upload_refusals(env):
    client, *_ = env
    assert client.post("/api/v1/tools/upload", headers=HX, files={"file": ("x.db", b"hello", "application/octet-stream")}).status_code == 422
    assert client.post("/api/v1/tools/upload", headers=HX, files={"file": ("x.txt", b"hello", "text/plain")}).status_code == 422
    assert client.post("/api/v1/tools/upload", headers=HX, files={"file": ("x.sql", b"\xff\xfe\x00bad", "application/sql")}).status_code == 422
    target = env[3]
    assert not list(target.glob(".upload-*.part"))
    live = (target / "syntrivetts.db").read_bytes()
    r = client.post("/api/v1/tools/upload", headers=HX, files={"file": ("syntrivetts.db", live, "application/octet-stream")})
    assert r.status_code == 200 and (target / "syntrivetts (2).db").is_file()


def test_download_delete_and_name_guards(env):
    client, source, source_job, target, _ = env
    name = _export_into(source, source_job, target)
    Path(f"{target / name}-wal").write_bytes(b"")
    got = client.get(f"/api/v1/tools/files/{name}")
    assert got.status_code == 200 and got.content[:16] == dbt.SQLITE_HEADER
    assert "attachment" in got.headers["content-disposition"]

    assert client.get("/api/v1/tools/files/syntrivetts.db", headers=HX).status_code == 422
    assert client.post("/api/v1/tools/files/syntrivetts.db/delete", headers=HX).status_code == 422
    assert client.get("/api/v1/tools/files/..%5Csyntrivetts.db", headers=HX).status_code == 422
    assert client.get("/api/v1/tools/files/notes.txt", headers=HX).status_code == 422
    assert client.get("/api/v1/tools/files/missing.db", headers=HX).status_code == 404

    gone = client.post(f"/api/v1/tools/files/{name}/delete", headers=HX)
    assert gone.status_code == 200 and not (target / name).exists() and not Path(f"{target / name}-wal").exists()
    assert (target / "syntrivetts.db").is_file()
