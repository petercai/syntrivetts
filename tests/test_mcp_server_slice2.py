from __future__ import annotations

import asyncio
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

pytest.importorskip("mcp")

from mcp import Client  # noqa: E402
from PIL import Image  # noqa: E402

from syntrive.adapters import ffmpeg as ffmpeg_adapter  # noqa: E402
from syntrive.bootstrap import bootstrap  # noqa: E402
from syntrive.db.models import Job, ReferenceVoice, TtsConfig, TtsVoice  # noqa: E402
from syntrive.db.session import get_db_session  # noqa: E402
from syntrive.io import paths  # noqa: E402
from syntrive.mcp.server import create_server  # noqa: E402
from syntrive.services import cover_watermark as cw  # noqa: E402
from syntrive.services import model_library as ml  # noqa: E402

EBOOKS = Path(__file__).resolve().parent.parent / "ebooks"


def _wav(path: Path, seconds: float = 0.5) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.zeros(int(seconds * 16000), dtype=np.float32), 16000, subtype="PCM_16")
    return path


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.delenv("SYNTRIVE_TTS_OFFLINE", raising=False)
    root = tmp_path / "project"
    for rel in ("voices/en/adult/male/adam.wav", "voices/zh/adult/male/yunjian.wav", "voices/zh/adult/female/yunxia.wav"):
        _wav(root / rel)
    monkeypatch.setattr(paths, "PROJECT_ROOT", root)
    repo = tmp_path / "repo"
    job = bootstrap(repo, EBOOKS / "Jan-Eyre-5.epub")
    server, state = create_server(repo)
    return server, state, repo, root, job.id


def run(server, body):
    async def main():
        async with Client(server) as client:
            return await body(client)
    return asyncio.run(main())


def data(result):
    assert not result.is_error, result.content[0].text
    return result.structured_content


def error(result):
    assert result.is_error, result.structured_content
    return result.content[0].text


def _cast(db: Path, job_id: int, voice_id: int) -> None:
    with get_db_session(db) as s:
        cfg = s.query(TtsConfig).filter_by(job_id=job_id).one_or_none()
        if cfg is None:
            cfg = TtsConfig(job_id=job_id, engine="cosyvoice")
            s.add(cfg)
            s.flush()
        s.add(TtsVoice(tts_config_id=cfg.id, name="narrator", voice_id=0, reference_voice_id=voice_id))


def test_voice_tools(env):
    server, _, repo, root, job_id = env
    db = repo / "syntrivetts.db"

    async def scan_import(c):
        scan = data(await c.call_tool("voice_scan", {}))
        imported = data(await c.call_tool("voice_import", {
            "stored_paths": ["voices/zh/adult/male/yunjian.wav", "voices/zh/adult/female/yunxia.wav", "voices/../evil.wav"],
            "tags": "calm"}))
        bad = error(await c.call_tool("voice_import", {"stored_paths": ["voices/en/adult/male/adam.wav"], "gender": "robot"}))
        return scan, imported, bad

    scan, imported, bad = run(server, scan_import)
    assert scan["evidence"]["new"] == 3 and scan["evidence"]["missing"] == 0
    assert imported["data"]["imported"] == 2 and imported["data"]["ignored"] == 1 and imported["warnings"]
    assert "invalid: gender must be male, female or empty" in bad

    ids = {rv.name: rv.id for rv in _names(db)}
    _cast(db, job_id, ids["yunjian"])

    async def edit_delete(c):
        listed = data(await c.call_tool("voice_list", {"lang": "zh"}))
        in_use = data(await c.call_tool("voice_list", {"flag": "in_use"}))
        got = data(await c.call_tool("voice_get", {"voice_id": ids["yunxia"]}))
        upd = data(await c.call_tool("voice_update", {"voice_id": ids["yunxia"], "accent": "beijing",
                                                     "tags": ["adult", "Warm"], "ref_text": "你好。"}))
        same = data(await c.call_tool("voice_update", {"voice_id": ids["yunxia"], "accent": "beijing"}))
        blocked = error(await c.call_tool("voice_delete", {"voice_id": ids["yunjian"]}))
        preview = data(await c.call_tool("voice_delete", {"voice_id": ids["yunxia"]}))
        deleted = data(await c.call_tool("voice_delete", {"voice_id": ids["yunxia"], "dry_run": False,
                                                         "confirm_token": preview["evidence"]["confirm_token"]}))
        missing = error(await c.call_tool("voice_get", {"voice_id": 9999}))
        return listed, in_use, got, upd, same, blocked, preview, deleted, missing

    listed, in_use, got, upd, same, blocked, preview, deleted, missing = run(server, edit_delete)
    assert [v["name"] for v in listed["data"]] == ["yunjian", "yunxia"]
    assert listed["evidence"]["facets"]["lang:zh"] == 2
    assert [v["name"] for v in in_use["data"]] == ["yunjian"] and in_use["data"][0]["used_by"]
    assert got["data"]["ref_text"] == "" and got["data"]["sample_rate"] == 16000
    assert upd["data"]["changed"] and upd["data"]["accent"] == "beijing" and set(upd["data"]["tags"]) == {"adult", "warm"}
    assert (root / "voices/zh/adult/female/yunxia.txt").read_text(encoding="utf-8").strip() == "你好。"
    assert same["data"]["changed"] is False
    assert blocked.startswith("Error executing tool voice_delete: conflict:") and "Jan Eyre" in blocked
    assert preview["dry_run"] is True and deleted["data"]["deleted"] is True
    assert "not_found" in missing
    assert {rv.name for rv in _names(db)} == {"yunjian"}

    (root / "voices/zh/adult/male/yunjian.wav").rename(_moved := root / "voices/zh/elder/male/yunjian.wav") if (
        (root / "voices/zh/elder/male").mkdir(parents=True) or True) else None

    async def relink(c):
        scan2 = data(await c.call_tool("voice_scan", {}))
        return scan2, data(await c.call_tool("voice_relink", {"voice_id": ids["yunjian"]}))

    scan2, relinked = run(server, relink)
    assert scan2["evidence"]["missing"] == 1
    assert relinked["data"]["new_path"] == "voices/zh/elder/male/yunjian.wav" and _moved.is_file()


def _names(db: Path):
    with get_db_session(db) as s:
        rows = s.query(ReferenceVoice).all()
        s.expunge_all()
        return rows


def test_model_tools_without_downloading(env, monkeypatch):
    server = env[0]
    downloaded = next((r for g in ml.read_model_catalog() for r in g.rows if r.kind == "model" and r.downloaded), None)
    missing_row = next((r for g in ml.read_model_catalog() for r in g.rows if r.kind == "model" and not r.downloaded), None)

    async def body(c):
        catalog = data(await c.call_tool("model_catalog", {}))
        one = data(await c.call_tool("model_catalog", {"engine": catalog["data"][0]["engine"]}))
        unknown_engine = error(await c.call_tool("model_catalog", {"engine": "nope"}))
        unknown = error(await c.call_tool("model_download", {"engine": "xtts", "item_id": "nope"}))
        tasks = data(await c.call_tool("model_tasks", {}))
        no_task = error(await c.call_tool("model_task_status", {"task_id": 42}))
        out = {"catalog": catalog, "one": one, "unknown_engine": unknown_engine, "unknown": unknown, "tasks": tasks,
               "no_task": no_task}
        if downloaded is not None:
            out["files"] = data(await c.call_tool("model_files", {"engine": downloaded.engine, "model_id": downloaded.model_id}))
            out["remove_preview"] = data(await c.call_tool("model_remove", {"engine": downloaded.engine, "item_id": downloaded.model_id}))
        if missing_row is not None:
            out["remove_nothing"] = error(await c.call_tool("model_remove", {"engine": missing_row.engine, "item_id": missing_row.model_id}))
        monkeypatch.setenv("SYNTRIVE_TTS_OFFLINE", "1")
        target = missing_row or downloaded
        out["offline_download"] = error(await c.call_tool("model_download", {"engine": target.engine, "item_id": target.model_id}))
        out["offline_check"] = error(await c.call_tool("model_check_update", {"engine": target.engine, "model_id": target.model_id}))
        out["offline_provision"] = error(await c.call_tool("model_provision", {"engine": target.engine}))
        out["tasks_after"] = data(await c.call_tool("model_tasks", {}))
        return out

    out = run(server, body)
    engines = {g["engine"] for g in out["catalog"]["data"]}
    assert engines == {g.engine for g in ml.read_model_catalog()} and len(out["one"]["data"]) == 1
    assert "not_found" in out["unknown_engine"] and "not_found" in out["unknown"] and "not_found" in out["no_task"]
    assert out["tasks"]["data"] == []
    if downloaded is not None:
        assert out["remove_preview"]["dry_run"] is True and out["remove_preview"]["evidence"]["confirm_token"]
        assert isinstance(out["files"]["data"], list)
    if missing_row is not None:
        assert "refused:" in out["remove_nothing"]
    for key in ("offline_download", "offline_check", "offline_provision"):
        assert "refused: Offline mode is on" in out[key]
    assert out["tasks_after"]["data"] == []


def test_db_tools(env):
    server, _, repo, root, job_id = env

    async def body(c):
        exported = data(await c.call_tool("db_export", {"job_ids": [job_id]}))
        backup = data(await c.call_tool("db_backup_voices", {}))
        files = data(await c.call_tool("db_files", {}))
        name = exported["data"]["file"]
        plan = data(await c.call_tool("db_import", {"file": name}))
        plan_over = data(await c.call_tool("db_import", {"file": name, "overwrite_job_ids": [job_id]}))
        wrong_token = error(await c.call_tool("db_import", {"file": name, "dry_run": False,
                                                           "confirm_token": plan_over["evidence"]["confirm_token"]}))
        done = data(await c.call_tool("db_import", {"file": name, "overwrite_job_ids": [job_id], "dry_run": False,
                                                   "confirm_token": plan_over["evidence"]["confirm_token"]}))
        restore_preview = data(await c.call_tool("db_restore_voices", {"file": backup["data"]["file"]}))
        restore = data(await c.call_tool("db_restore_voices", {"file": backup["data"]["file"], "dry_run": False,
                                                              "confirm_token": restore_preview["evidence"]["confirm_token"]}))
        live = error(await c.call_tool("db_delete_file", {"file": "syntrivetts.db"}))
        bad = error(await c.call_tool("db_import", {"file": "../x.db"}))
        del_preview = data(await c.call_tool("db_delete_file", {"file": name}))
        deleted = data(await c.call_tool("db_delete_file", {"file": name, "dry_run": False,
                                                           "confirm_token": del_preview["evidence"]["confirm_token"]}))
        return exported, backup, files, plan, plan_over, wrong_token, done, restore, live, bad, deleted

    exported, backup, files, plan, plan_over, wrong_token, done, restore, live, bad, deleted = run(server, body)
    name = exported["data"]["file"]
    assert exported["data"]["jobs"] == 1 and exported["data"]["folders_to_copy"] == ["PROCESSING-Jan-Eyre-5"]
    assert {f["name"] for f in files["data"]} == {name, backup["data"]["file"]}
    assert plan["data"]["jobs"][0]["action"] == "skip" and plan_over["data"]["jobs"][0]["action"] == "overwrite"
    assert "refused: confirm_token mismatch" in wrong_token
    assert (done["data"]["overwritten"], done["data"]["added"]) == (1, 0)
    assert restore["data"]["voices"] == 0
    assert "refused:" in live and "invalid:" in bad
    assert deleted["data"]["deleted"] == name and not (repo / name).exists()
    with get_db_session(repo / "syntrivetts.db") as s:
        assert s.query(Job).count() == 1


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not found on PATH")
def test_audio_watermark_covers(env, tmp_path):
    server, _, repo, _, _ = env
    cover = tmp_path / "cover.jpg"
    Image.new("RGB", (400, 600), (230, 230, 230)).save(cover, "JPEG")
    wav = tmp_path / "tone_raw.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "sine=duration=0.5", str(wav)], check=True)
    m4a = repo / "PROCESSING-Jan-Eyre-5" / "audiobooks" / "ch0001.m4a"
    m4a.parent.mkdir(parents=True, exist_ok=True)
    assert ffmpeg_adapter.to_m4a(str(wav), str(m4a), metadata={"album": "Jan Eyre"}, cover_path=str(cover)).ok
    before = m4a.read_bytes()

    async def body(c):
        preview = data(await c.call_tool("audio_watermark_covers", {}))
        untouched = m4a.read_bytes() == before
        done = data(await c.call_tool("audio_watermark_covers", {"dry_run": False,
                                                                "confirm_token": preview["evidence"]["confirm_token"]}))
        again = data(await c.call_tool("audio_watermark_covers", {}))
        return preview, untouched, done, again

    preview, untouched, done, again = run(server, body)
    assert preview["data"]["would_watermark"] == ["PROCESSING-Jan-Eyre-5/audiobooks/ch0001.m4a"] and untouched
    assert done["data"]["counts"]["watermarked"] == 1
    assert again["data"]["would_watermark"] == [] and again["data"]["counts"]["already"] == 1
    from mutagen.mp4 import MP4
    assert cw.is_marked(MP4(m4a).tags)
