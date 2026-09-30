from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from syntrive.bootstrap import bootstrap
from syntrive.db.models import Job, ReferenceVoice, ReferenceVoiceTag, TtsConfig, TtsVoice
from syntrive.db.session import get_db_session
from syntrive.io import paths
from syntrive.services import voice_library as vl

EBOOKS = Path(__file__).resolve().parent.parent / "ebooks"


def _wav(path: Path, seconds: float = 0.5, rate: int = 16000) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.zeros(int(seconds * rate), dtype=np.float32), rate, subtype="PCM_16")
    return path


@pytest.fixture()
def lib(tmp_path, monkeypatch):
    root = tmp_path / "project"
    _wav(root / "voices" / "en" / "adult" / "male" / "adam.wav", seconds=1.25)
    _wav(root / "voices" / "zh" / "adult" / "female" / "yunjian.wav")
    _wav(root / "voices" / "fr" / "adult" / "male" / "pierre.wav")
    _wav(root / "voices" / "__sessions" / "cache.wav")
    ft = root / "models" / "tts" / vl.FINETUNED_MODELS_DIRNAME / "en" / "narrator_x"
    (ft / "model.pth").parent.mkdir(parents=True, exist_ok=True)
    (ft / "model.pth").write_bytes(b"0")
    _wav(ft / "ref.wav")
    monkeypatch.setattr(paths, "PROJECT_ROOT", root)
    repo = tmp_path / "repo"
    job = bootstrap(repo, EBOOKS / "Jan-Eyre-5.epub")
    return root, repo / "syntrivetts.db", job.id


def _cast(db_path: Path, job_id: int, rv_id: int) -> None:
    with get_db_session(db_path) as db:
        cfg = db.query(TtsConfig).filter_by(job_id=job_id).one_or_none()
        if cfg is None:
            cfg = TtsConfig(job_id=job_id, engine="xtts")
            db.add(cfg)
            db.flush()
        db.add(TtsVoice(tts_config_id=cfg.id, name="narrator", voice_id=0, reference_voice_id=rv_id))


def test_scan_import_list_roundtrip(lib):
    root, db, job_id = lib
    scan = vl.scan_new_voices(db)
    names = sorted(c.name for c in scan.new)
    assert names == ["adam", "ref", "yunjian"]
    assert scan.skipped == 0 and scan.missing == ()
    assert "pierre" in {c.name for c in vl.scan_new_voices(db, lan_filter="all").new}

    ref = next(c for c in scan.new if c.name == "ref")
    assert ref.fine_tuned and ref.language == "en"
    adam = next(c for c in scan.new if c.name == "adam")
    assert (adam.gender, adam.language, adam.extra_tags, adam.duration_seconds) == ("male", "en", ("adult",), 1.25)
    assert adam.stored_path == "voices/en/adult/male/adam.wav"

    chosen = [vl.apply_batch_tags(adam, accent="british", tags="adult,warm")]
    assert vl.import_voices(db, chosen) == 1
    assert vl.import_voices(db, chosen) == 0

    rescan = vl.scan_new_voices(db)
    assert sorted(c.name for c in rescan.new) == ["ref", "yunjian"] and rescan.skipped == 1

    (rows,) = [vl.list_voices(db)]
    assert len(rows) == 1
    row = rows[0]
    assert row.voice.name == "adam" and row.voice.accent == "british"
    assert set(row.voice.tags) == {"adult", "warm"}
    assert row.file_present and not row.has_ref_text and not row.in_use
    assert row.folder == "voices/en/adult/male"

    vl.write_ref_text(row.voice.abs_path, "Hello there.")
    assert vl.list_voices(db)[0].has_ref_text


def test_usage_blocks_delete_and_names_the_book(lib):
    root, db, job_id = lib
    vl.import_voices(db, list(vl.scan_new_voices(db).new))
    rows = {r.voice.name: r for r in vl.list_voices(db)}
    _cast(db, job_id, rows["yunjian"].voice.db_id)

    with get_db_session(db) as s:
        title = s.get(Job, job_id).book.title
    rows = {r.voice.name: r for r in vl.list_voices(db)}
    assert rows["yunjian"].used_by == (title,)
    assert rows["yunjian"].in_use and not rows["adam"].in_use

    assert vl.delete_reference_voice(db, rows["yunjian"].voice.db_id) is False
    assert vl.delete_reference_voice(db, rows["adam"].voice.db_id) is True
    with get_db_session(db) as s:
        assert {rv.name for rv in s.query(ReferenceVoice)} == {"yunjian", "ref"}
        assert s.query(ReferenceVoiceTag).count() >= 1


def test_update_and_missing_file_relink(lib):
    root, db, _ = lib
    vl.import_voices(db, list(vl.scan_new_voices(db).new))
    adam = next(r for r in vl.list_voices(db) if r.voice.name == "adam").voice

    assert vl.update_reference_voice(db, adam.db_id, gender="male", language="eng", accent="", tags=("elder",)) is True
    again = next(r for r in vl.list_voices(db) if r.voice.name == "adam").voice
    assert (again.language, again.extra_tags) == ("en", ("elder",))
    assert vl.update_reference_voice(db, adam.db_id, gender="male", language="en", accent="", tags=("elder",)) is False

    moved = root / "voices" / "en" / "elder" / "adam.wav"
    moved.parent.mkdir(parents=True)
    adam.abs_path.rename(moved)
    scan = vl.scan_new_voices(db)
    (missing,) = scan.missing
    assert missing.name == "adam" and missing.replacement == "voices/en/elder/adam.wav"
    assert not next(r for r in vl.list_voices(db) if r.voice.name == "adam").file_present

    assert vl.relink_reference_voice(db, missing.db_id, missing.replacement) is True
    assert next(r for r in vl.list_voices(db) if r.voice.name == "adam").file_present
    assert vl.scan_new_voices(db).missing == ()
