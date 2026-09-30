from __future__ import annotations

import functools
import http.server
import struct
import threading
import time
import zlib
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

pytest.importorskip("fastapi")
pytest.importorskip("multipart")

from fastapi.testclient import TestClient  # noqa: E402

from syntrive.bootstrap import bootstrap  # noqa: E402
from syntrive.db.models import Job, ReferenceVoice  # noqa: E402
from syntrive.db.session import get_db_session  # noqa: E402
from syntrive.services import book_catalog  # noqa: E402
from syntrive.services import tts_settings as ts  # noqa: E402
from syntrive.services.cover_images import image_extension  # noqa: E402
from syntrive.services.pipeline_service import OnExisting, RunDisposition, StepOptions, run_step  # noqa: E402
from syntrive.webui.app import create_app  # noqa: E402
from syntrive.webui.shared.state import ServerInfo, open_web_repo  # noqa: E402
from syntrive.workflow.engine import WorkflowStep  # noqa: E402

EBOOKS = Path(__file__).resolve().parent.parent / "ebooks"
HX = {"hx-request": "true", "accept-language": "en"}


def _png(width: int = 2, height: int = 3) -> bytes:
    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    raw = b"".join(b"\x00" + b"\xff\x80\x00" * width for _ in range(height))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def _job(db: Path, job_id: int) -> Job:
    with get_db_session(db) as s:
        job = s.get(Job, job_id)
        s.expunge_all()
        return job


def _wait(client: TestClient, job_id: int, timeout: float = 60) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if client.get(f"/api/v1/pipeline/{job_id}/run", headers=HX).status_code == 204:
            return
        time.sleep(0.1)
    raise AssertionError("run did not settle")


@pytest.fixture()
def env(tmp_path):
    job = bootstrap(tmp_path, EBOOKS / "Jan-Eyre-5.epub")
    db = tmp_path / "syntrivetts.db"
    for step in (WorkflowStep.BOOTSTRAP, WorkflowStep.TRANSCRIPT_EXTRACT, WorkflowStep.TRANSCRIPT_MERGE,
                 WorkflowStep.TRANSCRIPT_CLEAN, WorkflowStep.TRANSCRIPT_TEXT):
        assert run_step(db, job.id, step, on_existing=OnExisting.OVERWRITE, holder_kind="test").disposition == RunDisposition.DONE
    assert run_step(db, job.id, WorkflowStep.TRANSCRIPT_REVIEW, on_existing=OnExisting.OVERWRITE, holder_kind="test",
                    options=StepOptions(transcript_path="tts_script")).disposition == RunDisposition.DONE
    wav = tmp_path / "voice-a.wav"
    sf.write(wav, np.zeros(8000, dtype="float32"), 16000)
    with get_db_session(db) as s:
        ref = ReferenceVoice(path=wav.as_posix(), name="Voice A", gender="female", language="en")
        s.add(ref)
        s.flush()
        ref_id = ref.id
    app = create_app(open_web_repo(tmp_path), ServerInfo(port=1, host="127.0.0.1", started_at="t0", version="test"), register=False)
    with TestClient(app) as client:
        yield {"client": client, "db": db, "job_id": job.id, "ref_id": ref_id, "wav": wav, "repo": tmp_path}


def _form_from_settings(settings: ts.TtsSettings, **overrides) -> dict:
    data = {}
    for p in ts.PARAMS:
        value = overrides.get(p.field, settings.values[p.field])
        if p.input_type == ts.InputType.CHECKBOX:
            if value:
                data[p.field] = "1"
        else:
            data[p.field] = "" if value is None else str(value)
    for voice in settings.voices:
        data[f"voice_{voice.id}"] = str(overrides.get(f"voice_{voice.id}", voice.reference_voice_id or ""))
    return data


class TestStep7:
    def test_page_prepares_rows_and_renders_cast(self, env):
        html = env["client"].get(f"/pipeline/{env['job_id']}", headers={"accept-language": "en"}).text
        settings = ts.read_tts_settings(env["db"], env["job_id"])
        assert settings.exists and any(v.is_narrator for v in settings.voices)
        assert 'data-subtab="cast"' in html and 'data-subtab="tuning"' in html
        assert 'name="engine"' in html and 'id="model-select"' in html
        assert "Voice A [female/en]" in html and f"/voices/{env['ref_id']}/audio" in html
        assert 'name="retry_badcase_ratio_threshold"' in html

    def test_model_options_and_voice_audio(self, env):
        client, job_id = env["client"], env["job_id"]
        models = client.get(f"/api/v1/pipeline/{job_id}/tts-models", params={"engine": "indextts"}, headers=HX)
        assert models.status_code == 200 and 'id="model-select"' in models.text and "Engine default" in models.text
        assert client.get(f"/api/v1/pipeline/{job_id}/tts-models", params={"engine": "bark"}, headers=HX).status_code == 422
        audio = client.get(f"/api/v1/pipeline/{job_id}/voices/{env['ref_id']}/audio")
        assert audio.status_code == 200 and audio.content == env["wav"].read_bytes()
        assert client.get(f"/api/v1/pipeline/{job_id}/voices/424242/audio", headers=HX).status_code == 422

    def test_save_round_trip_and_refusal(self, env):
        client, db, job_id = env["client"], env["db"], env["job_id"]
        client.get(f"/pipeline/{job_id}")
        settings = ts.read_tts_settings(db, job_id)
        narrator = next(v for v in settings.voices if v.is_narrator)
        form = _form_from_settings(settings, speed=1.2, **{f"voice_{narrator.id}": env["ref_id"]})
        saved = client.post(f"/api/v1/pipeline/{job_id}/tts-settings", data=form, headers=HX)
        assert saved.headers["hx-redirect"].endswith("notice=saved")
        after = ts.read_tts_settings(db, job_id)
        assert after.values["speed"] == 1.2
        assert next(v for v in after.voices if v.is_narrator).reference_voice_id == env["ref_id"]

        bad = client.post(f"/api/v1/pipeline/{job_id}/tts-settings", data=_form_from_settings(after, speed=9), headers=HX)
        assert bad.status_code == 422 and "Speed" in bad.text

    def test_continue_runs_step_7_and_requires_a_narrator_voice(self, env):
        client, db, job_id = env["client"], env["db"], env["job_id"]
        client.get(f"/pipeline/{job_id}")
        settings = ts.read_tts_settings(db, job_id)
        client.post(f"/api/v1/pipeline/{job_id}/tts-settings", data={**_form_from_settings(settings), "continue": "1"}, headers=HX)
        _wait(client, job_id)
        assert _job(db, job_id).current_step == "tts_config"
        assert "Choose a narrator voice" in client.get(f"/pipeline/{job_id}").text

        narrator = next(v for v in settings.voices if v.is_narrator)
        form = {**_form_from_settings(settings, **{f"voice_{narrator.id}": env["ref_id"]}), "continue": "1"}
        client.post(f"/api/v1/pipeline/{job_id}/tts-settings", data=form, headers=HX)
        _wait(client, job_id)
        assert _job(db, job_id).current_step == "synthesis"


class TestMetadata:
    def test_tab_renders_and_saves_author_and_cover(self, env):
        client, db, job_id = env["client"], env["db"], env["job_id"]
        html = client.get(f"/pipeline/{job_id}", params={"tab": "metadata"}, headers={"accept-language": "en"}).text
        meta = book_catalog.read_book_metadata(db, job_id)
        assert 'name="author"' in html and 'name="cover"' in html and "Add cover images" in html
        assert all(f"/images/{img.name}" in html for img in meta.images)

        response = client.post(f"/api/v1/pipeline/{job_id}/metadata", headers=HX,
                               data={"title": meta.title, "author": "Charlotte Brontë", "language": meta.language, "cover": meta.cover_rel})
        assert response.headers["hx-redirect"].endswith("notice=saved")
        assert book_catalog.read_book_metadata(db, job_id).author == "Charlotte Brontë"
        empty = client.post(f"/api/v1/pipeline/{job_id}/metadata", headers=HX, data={"title": " ", "author": "", "language": "en"})
        assert empty.status_code == 422
        bad_cover = client.post(f"/api/v1/pipeline/{job_id}/metadata", headers=HX,
                                data={"title": meta.title, "language": "en", "cover": "../../secret.jpg"})
        assert bad_cover.status_code == 422

    def test_import_uploads_and_urls(self, env, tmp_path):
        client, db, job_id = env["client"], env["db"], env["job_id"]
        served = tmp_path / "served"
        served.mkdir()
        jpeg = next(i for i in book_catalog.read_book_metadata(db, job_id).images if image_extension(i.path.read_bytes()) == ".jpg")
        (served / "remote-cover.jpg").write_bytes(jpeg.path.read_bytes())
        handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(served))
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            port = server.server_address[1]
            response = client.post(
                f"/api/v1/pipeline/{job_id}/images", headers=HX,
                files=[("files", ("my cover.png", _png(), "image/png")), ("files", ("notes.txt", b"hello", "text/plain"))],
                data={"urls": f"http://127.0.0.1:{port}/remote-cover.jpg\nftp://example.com/x.jpg\nhttp://127.0.0.1:{port}/missing.jpg"},
            )
        finally:
            server.shutdown()
        assert response.status_code == 200 and 'id="cover-gallery"' in response.text
        assert response.text.count("✓") == 2 and response.text.count("✗") == 3
        names = {i.name for i in book_catalog.read_book_metadata(db, job_id).images}
        assert {"imported-my-cover.png", "imported-remote-cover.jpg"} <= names
        served_png = client.get(f"/api/v1/pipeline/{job_id}/images/imported-my-cover.png")
        assert served_png.status_code == 200 and served_png.content == _png()
        assert client.get(f"/api/v1/pipeline/{job_id}/images/..%2Fsyntrivetts.db", headers=HX).status_code == 404
        assert client.post(f"/api/v1/pipeline/{job_id}/images", data={"urls": " "}, headers=HX).status_code == 422
