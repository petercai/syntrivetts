from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

pytest.importorskip("fastapi")
pytest.importorskip("multipart")

from fastapi.testclient import TestClient  # noqa: E402

from syntrive.adapters.tts import model_catalog  # noqa: E402
from syntrive.bootstrap import bootstrap  # noqa: E402
from syntrive.db.models import Job, ReferenceVoice, TtsConfig, TtsVoice  # noqa: E402
from syntrive.db.session import get_db_session  # noqa: E402
from syntrive.io import paths  # noqa: E402
from syntrive.services import model_library as ml  # noqa: E402
from syntrive.services import voice_library as vl  # noqa: E402
from syntrive.webui.app import create_app  # noqa: E402
from syntrive.webui.shared.state import ServerInfo, open_web_repo  # noqa: E402

EBOOKS = Path(__file__).resolve().parent.parent / "ebooks"
HX = {"hx-request": "true", "accept-language": "en"}
EN = {"accept-language": "en"}


def _wav(path: Path, seconds: float = 0.5) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.zeros(int(seconds * 16000), dtype=np.float32), 16000, subtype="PCM_16")
    return path


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.delenv("SYNTRIVE_TTS_OFFLINE", raising=False)
    root = tmp_path / "project"
    _wav(root / "voices" / "en" / "adult" / "male" / "adam.wav", 1.25)
    _wav(root / "voices" / "zh" / "adult" / "male" / "yunjian.wav", 2.0)
    _wav(root / "voices" / "zh" / "adult" / "female" / "yunxia.wav")
    monkeypatch.setattr(paths, "PROJECT_ROOT", root)
    monkeypatch.setattr(ml, "_DEVICE_CONFIG_PATH", tmp_path / ".model_manager.json")
    repo = tmp_path / "repo"
    job = bootstrap(repo, EBOOKS / "Jan-Eyre-5.epub")
    db = repo / "syntrivetts.db"
    vl.import_voices(db, list(vl.scan_new_voices(db).new))
    ids = {r.voice.name: r.voice.db_id for r in vl.list_voices(db)}
    with get_db_session(db) as s:
        cfg = s.query(TtsConfig).filter_by(job_id=job.id).one_or_none()
        if cfg is None:
            cfg = TtsConfig(job_id=job.id, engine="cosyvoice")
            s.add(cfg)
            s.flush()
        s.add(TtsVoice(tts_config_id=cfg.id, name="narrator", voice_id=0, reference_voice_id=ids["yunjian"]))
        title = s.get(Job, job.id).book.title
    app = create_app(open_web_repo(repo), ServerInfo(port=0, host="127.0.0.1", started_at="now", version="t"), register=False)
    return TestClient(app), db, root, ids, title


def test_voices_page_filters_and_sidebar(env):
    client, _, _, ids, title = env
    page = client.get("/voices", headers=EN).text
    assert page.count('class="v-row"') == 3
    assert page.index(">yunjian<") < page.index(">adam<")
    assert title in page
    assert ">Library<" in page and ">3<" in page

    zh = client.get("/voices?lang=zh", headers=EN).text
    assert ">yunxia<" in zh and ">adam<" not in zh
    assert ">zh · 2<" in zh and ">en · 1<" in zh
    assert client.get("/voices?flag=in_use", headers=EN).text.count('class="v-row"') == 1
    assert client.get("/voices?q=ADA", headers=EN).text.count('class="v-row"') == 1
    assert client.get("/voices?flag=bogus", headers=HX).status_code == 422

    inspector = client.get(f"/voices?lang=zh&voice={ids['yunjian']}", headers=EN).text
    assert 'id="inspector-title">yunjian<' in inspector
    assert "In use: can&#39;t delete" in inspector or "In use: can't delete" in inspector

    audio = client.get(f"/api/v1/voices/{ids['adam']}/audio")
    assert audio.status_code == 200 and audio.content[:4] == b"RIFF"
    assert client.get("/api/v1/voices/9999/audio", headers=HX).status_code == 404


def test_save_voice_fields_and_reference_text(env):
    client, db, root, ids, _ = env
    r = client.post(f"/api/v1/voices/{ids['adam']}", headers=HX, data={
        "gender": "male", "language": "eng", "accent": "british", "tags": "adult, Warm, adult",
        "ref_text": "Hello there.", "back": "lang=en&q=ad&notice=x&voice=1"})
    assert r.status_code == 200
    target = r.headers["HX-Redirect"]
    assert target.startswith("/voices?") and "lang=en" in target and "q=ad" in target
    assert f"voice={ids['adam']}" in target and "notice=saved" in target and "notice=x" not in target
    row = next(x for x in vl.list_voices(db) if x.voice.name == "adam")
    assert (row.voice.language, row.voice.accent, set(row.voice.tags)) == ("en", "british", {"adult", "warm"})
    assert (root / "voices/en/adult/male/adam.txt").read_text(encoding="utf-8").strip() == "Hello there."

    same = client.post(f"/api/v1/voices/{ids['adam']}", headers=HX, data={
        "gender": "male", "language": "en", "accent": "british", "tags": "adult, warm", "ref_text": "Hello there."})
    assert "notice=unchanged" in same.headers["HX-Redirect"]
    client.post(f"/api/v1/voices/{ids['adam']}", headers=HX, data={
        "gender": "male", "language": "en", "accent": "british", "tags": "adult, warm", "ref_text": ""})
    assert not (root / "voices/en/adult/male/adam.txt").exists()

    assert client.post(f"/api/v1/voices/{ids['adam']}", headers=HX, data={"gender": "robot"}).status_code == 422
    assert client.post(f"/api/v1/voices/{ids['adam']}", headers=HX, data={"language": "english!"}).status_code == 422


def test_delete_refused_in_use_then_deleted(env):
    client, db, _, ids, title = env
    refused = client.post(f"/api/v1/voices/{ids['yunjian']}/delete", headers=HX)
    assert refused.status_code == 409 and title in refused.text
    ok = client.post(f"/api/v1/voices/{ids['yunxia']}/delete", headers=HX, data={"back": "lang=zh"})
    assert ok.status_code == 200 and "lang=zh" in ok.headers["HX-Redirect"] and "deleted" in ok.headers["HX-Redirect"]
    with get_db_session(db) as s:
        assert {rv.name for rv in s.query(ReferenceVoice)} == {"adam", "yunjian"}


def test_import_chosen_with_batch_fields_and_relink(env):
    client, db, root, ids, _ = env
    _wav(root / "voices" / "en" / "teen" / "female" / "bella.wav")
    _wav(root / "voices" / "en" / "teen" / "female" / "cara.wav")
    ext = root.parent / "outside"
    _wav(ext / "dora.wav")

    page = client.get("/voices", params={"mode": "import", "external": str(ext)}, headers=EN).text
    assert page.count('name="path"') == 3 and "Import 3" in page
    assert client.get("/voices", params={"mode": "import", "external": str(ext / "nope")}, headers=HX).status_code == 422

    r = client.post("/api/v1/voices/import", headers=HX, data={
        "path": ["voices/en/teen/female/bella.wav", str((ext / "dora.wav").resolve().as_posix()), "voices/../../evil.wav"],
        "external": str(ext), "language": "en", "tags": "calm"})
    assert r.status_code == 200, r.text
    assert "imported%3A2" in r.headers["HX-Redirect"]
    rows = {x.voice.name: x.voice for x in vl.list_voices(db)}
    assert set(rows) == {"adam", "yunjian", "yunxia", "bella", "dora"}
    assert rows["dora"].language == "en" and rows["bella"].extra_tags == ("calm",)
    assert client.post("/api/v1/voices/import", headers=HX, data={}).status_code == 422

    (root / "voices/en/adult/male/adam.wav").rename(root / "voices/en/adult/male/adam_moved.wav")
    missing_page = client.get("/voices?mode=import", headers=EN).text
    assert 'value="voices/en/adult/male/adam_moved.wav"' in missing_page
    assert "No matching file found" in missing_page and ">Relink<" not in missing_page
    assert client.post(f"/api/v1/voices/{ids['adam']}/relink", headers=HX).status_code == 409
    moved = root / "voices/en/elder/male"
    moved.mkdir(parents=True)
    (root / "voices/en/adult/male/adam_moved.wav").rename(moved / "adam.wav")
    assert ">Relink<" in client.get("/voices?mode=import", headers=EN).text
    ok = client.post(f"/api/v1/voices/{ids['adam']}/relink", headers=HX)
    assert ok.status_code == 200 and "relinked" in ok.headers["HX-Redirect"]
    assert next(x for x in vl.list_voices(db) if x.voice.name == "adam").voice.stored_path == "voices/en/elder/male/adam.wav"


def test_update_durations(env):
    client, db, _, _, _ = env
    with get_db_session(db) as s:
        for rv in s.query(ReferenceVoice):
            rv.duration_seconds = None
    r = client.post("/api/v1/voices/durations", headers=HX)
    assert r.status_code == 200 and "durations%3A3%3A0" in r.headers["HX-Redirect"]
    assert {x.voice.name: x.voice.duration_seconds for x in vl.list_voices(db)}["adam"] == 1.25


def test_models_page_mirrors_catalog(env):
    client = env[0]
    page = client.get("/models", headers=EN).text
    groups = ml.read_model_catalog()
    assert page.count('class="m-row"') == sum(len(g.rows) for g in groups)
    for g in groups:
        assert f"<strong>{g.label}</strong>" in page
    done, total = ml.model_counts()
    assert f">{done}/{total}<" in page
    assert "No model action since this server started." in page
    assert "Offline mode" not in page

    engine = model_catalog.engines()[0]
    spec = model_catalog.models_for(engine)[0]
    dialog = client.get(f"/api/v1/models/{engine}/{spec.model_id}/confirm", headers=HX).text
    check = ml.disk_check(float(spec.approx_gb or 0))
    assert "<dialog" in dialog and f"~{spec.approx_gb} GB" in dialog
    assert ("This leaves less than 5 GB free." in dialog) == check.low
    assert client.get(f"/api/v1/models/{engine}/no-such-model/confirm", headers=HX).status_code == 404
    assert client.post(f"/api/v1/models/{engine}/no-such-model/remove", headers=HX).status_code == 404
    assert client.post("/api/v1/models/asr/provision", headers=HX).status_code == 404


def test_files_panel_for_a_downloaded_model(env):
    client = env[0]
    row = next((r for g in ml.read_model_catalog() for r in g.rows if r.kind == "model" and r.downloaded), None)
    if row is None:
        pytest.skip("no model downloaded on this machine")
    panel = client.get(f"/api/v1/models/{row.engine}/{row.model_id}/files", headers=HX).text
    assert panel.count('class="num ellipsis"') == len(ml.major_files(row.engine, row.model_id)) or "No files on disk yet." in panel


def test_offline_environment_locks_network_actions(env, monkeypatch):
    client = env[0]
    monkeypatch.setenv("SYNTRIVE_TTS_OFFLINE", "1")
    page = client.get("/models", headers=EN).text
    assert "Offline mode: SYNTRIVE_TTS_OFFLINE=1" in page
    assert 'title="Offline mode (server environment)"' in page
    engine = next(e for e in model_catalog.engines() if model_catalog.has_venv(e))
    spec = model_catalog.models_for(engine)[0]
    for action in ("download", "update", "check"):
        assert client.post(f"/api/v1/models/{engine}/{spec.model_id}/{action}", headers=HX).status_code == 409
    assert client.post(f"/api/v1/models/{engine}/provision", headers=HX).status_code == 409
    assert client.get(f"/api/v1/models/{engine}/{spec.model_id}/confirm", headers=HX).status_code == 409
    assert getattr(client.app.state, "model_tasks").views() == []


def test_device_round_trip(env, tmp_path):
    client = env[0]
    r = client.post("/api/v1/models/device", headers=HX, data={"device": "cuda"})
    assert r.status_code == 200 and "device" in r.headers["HX-Redirect"]
    assert ml.compute_device()[0] == "cuda"
    assert '<option value="cuda" selected>' in client.get("/models", headers=EN).text
    assert client.post("/api/v1/models/device", headers=HX, data={"device": "tpu"}).status_code == 422


def test_activity_polls_then_refreshes(env):
    client = env[0]
    client.get("/models")
    tasks: ml.ModelTasks = client.app.state.model_tasks
    engine = model_catalog.engines()[0]
    spec = model_catalog.models_for(engine)[0]
    release = threading.Event()

    def work(log):
        log("step one")
        release.wait(10)
        return None

    view = tasks.submit(f"model:{engine}/{spec.model_id}", f"download {engine}/{spec.model_id}", work)
    assert view is not None
    page = client.get("/models", headers=EN).text
    assert "working…" in page and 'hx-trigger="every 2s"' in page
    strip = client.get("/api/v1/models/activity?busy=1", headers=HX)
    assert strip.status_code == 200 and "running" in strip.text and "step one" in strip.text
    assert client.post(f"/api/v1/models/{engine}/{spec.model_id}/remove", headers=HX).status_code == 409

    release.set()
    assert tasks.wait_idle(10)
    done = client.get("/api/v1/models/activity?busy=1", headers=HX)
    assert done.headers.get("HX-Refresh") == "true"
    idle = client.get("/api/v1/models/activity", headers=HX).text
    assert "done" in idle and 'hx-trigger="every 2s"' not in idle


@pytest.mark.real_env
def test_transcribe_fills_text_box_without_saving(env):
    client, _, root, ids, _ = env
    r = client.post(f"/api/v1/voices/{ids['adam']}/transcribe", headers=HX)
    assert r.status_code in (200, 409)
    if r.status_code == 200:
        assert 'id="ref-text"' in r.text and 'data-generated="true"' in r.text
    assert not (root / "voices/en/adult/male/adam.txt").exists()
