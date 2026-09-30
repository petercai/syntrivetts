from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from syntrive.bootstrap import bootstrap
from syntrive.db.models import ReferenceVoice
from syntrive.db.session import get_db_session
from syntrive.services import book_catalog
from syntrive.services import tts_settings as ts
from syntrive.services.job_lease import HolderIdentity, JobLeaseConflict, acquire_lease
from syntrive.services.pipeline_service import OnExisting, run_step
from syntrive.workflow.engine import WorkflowEngine, WorkflowStep

EBOOKS = Path(__file__).resolve().parent.parent / "ebooks"
FOREIGN = HolderIdentity(holder_id="f" * 32, holder_kind="tts_batch", pid=4242, hostname="other-host")


@pytest.fixture()
def repo(tmp_path):
    job = bootstrap(tmp_path, EBOOKS / "Jan-Eyre-5.epub")
    db = tmp_path / "syntrivetts.db"
    run_step(db, job.id, WorkflowStep.BOOTSTRAP, on_existing=OnExisting.OVERWRITE, holder_kind="test")
    wav = tmp_path / "narrator.wav"
    sf.write(wav, np.zeros(1600, dtype="float32"), 16000)
    with get_db_session(db) as s:
        ref = ReferenceVoice(path=wav.as_posix(), name="Narrator A", gender="female", language="en")
        s.add(ref)
        s.flush()
        ref_id = ref.id
    return {"db": db, "job": job, "ref_id": ref_id, "wav": wav}


def _configure(repo) -> None:
    engine = WorkflowEngine(job=repo["job"], db_path=repo["db"], holder_kind="test")
    engine.ensure_tts_config()
    engine.ensure_tts_voices()


class TestRead:
    def test_before_step_7_everything_reads_as_defaults(self, repo):
        settings = ts.read_tts_settings(repo["db"], repo["job"].id)
        assert settings.exists is False and settings.voices == ()
        assert settings.values == ts.DEFAULTS | {"model": ""}
        assert settings.model_choices

    def test_after_step_7_entry_narrator_and_language(self, repo):
        _configure(repo)
        settings = ts.read_tts_settings(repo["db"], repo["job"].id)
        assert settings.exists and settings.values["language"] == "en"
        narrator = next(v for v in settings.voices if v.is_narrator)
        assert narrator.reference_voice_id is None
        assert [r.id for r in settings.references] == [repo["ref_id"]]
        assert settings.references[0].label.startswith("Narrator A [female/en]")

    def test_unknown_job(self, repo):
        with pytest.raises(ts.SettingsError) as err:
            ts.read_tts_settings(repo["db"], 999)
        assert err.value.code == "not_found"


class TestSave:
    def test_round_trip_with_coercion_and_voice_binding(self, repo):
        _configure(repo)
        db, job_id = repo["db"], repo["job"].id
        narrator = next(v for v in ts.read_tts_settings(db, job_id).voices if v.is_narrator)
        result = ts.save_tts_settings(
            db, job_id,
            {"speed": "1.25", "output_split": "by-duration", "output_split_minutes": "45", "denoise": "false"},
            [ts.VoiceBinding(narrator.id, repo["ref_id"], False)],
            holder_kind="test",
        )
        assert set(result.changed_fields) >= {"speed", "output_split", "output_split_minutes", "denoise"}
        assert result.changed_voices == (narrator.id,)

        settings = ts.read_tts_settings(db, job_id)
        assert settings.values["speed"] == 1.25 and settings.values["output_split_minutes"] == 45
        assert settings.values["denoise"] is False and settings.values["output_split"] == "by-duration"
        assert next(v for v in settings.voices if v.is_narrator).reference_voice_id == repo["ref_id"]
        again = ts.save_tts_settings(db, job_id, {"speed": 1.25}, [ts.VoiceBinding(narrator.id, repo["ref_id"], False)], holder_kind="test")
        assert again.changed is False

    def test_save_creates_the_config_rows_when_missing(self, repo):
        result = ts.save_tts_settings(repo["db"], repo["job"].id, {"device": "cpu", "temperature": 0.1}, holder_kind="test")
        assert "temperature" in result.changed_fields
        assert ts.read_tts_settings(repo["db"], repo["job"].id).exists

    @pytest.mark.parametrize(
        "changes, code, field",
        [
            ({"engine": "bark"}, "bad_choice", "engine"),
            ({"speed": 3}, "bad_number", "speed"),
            ({"speed": "fast"}, "bad_number", "speed"),
            ({"voice_dir": "x"}, "unknown_field", "voice_dir"),
            ({"model": "no-such-model"}, "bad_model", "model"),
        ],
    )
    def test_refusals_change_nothing(self, repo, changes, code, field):
        _configure(repo)
        before = ts.read_tts_settings(repo["db"], repo["job"].id).values
        with pytest.raises(ts.SettingsError) as err:
            ts.save_tts_settings(repo["db"], repo["job"].id, changes, holder_kind="test")
        assert (err.value.code, err.value.field) == (code, field)
        assert ts.read_tts_settings(repo["db"], repo["job"].id).values == before

    def test_voice_and_reference_refusals(self, repo):
        _configure(repo)
        db, job_id = repo["db"], repo["job"].id
        narrator = next(v for v in ts.read_tts_settings(db, job_id).voices if v.is_narrator)
        with pytest.raises(ts.SettingsError) as err:
            ts.save_tts_settings(db, job_id, {}, [ts.VoiceBinding(424242, None, False)], holder_kind="test")
        assert err.value.code == "unknown_voice"
        with pytest.raises(ts.SettingsError) as err:
            ts.save_tts_settings(db, job_id, {}, [ts.VoiceBinding(narrator.id, 424242, False)], holder_kind="test")
        assert err.value.code == "unknown_reference"

    def test_foreign_lease_blocks_the_save(self, repo):
        _configure(repo)
        acquire_lease(repo["db"], repo["job"].id, holder=FOREIGN, operation="batch:1")
        with pytest.raises(JobLeaseConflict):
            ts.save_tts_settings(repo["db"], repo["job"].id, {"speed": 1.1}, holder_kind="webui")
        assert ts.read_tts_settings(repo["db"], repo["job"].id).values["speed"] != 1.1


def test_gating_and_reference_file(repo):
    assert ts.field_enabled("output_split_minutes", {"output_split": "by-duration"})
    assert not ts.field_enabled("output_split_minutes", {"output_split": "by-chapter"})
    assert ts.reference_voice_file(repo["db"], repo["ref_id"]) == repo["wav"]
    repo["wav"].unlink()
    with pytest.raises(ts.SettingsError) as err:
        ts.reference_voice_file(repo["db"], repo["ref_id"])
    assert err.value.code == "missing_file"


def test_book_metadata_lists_title_language_and_images(repo):
    meta = book_catalog.read_book_metadata(repo["db"], repo["job"].id)
    assert meta.title and meta.language == "en"
    assert meta.images and all(img.exists for img in meta.images)
    assert meta.cover_rel in {img.rel_path for img in meta.images}
    assert book_catalog.read_book_metadata(repo["db"], 999) is None
