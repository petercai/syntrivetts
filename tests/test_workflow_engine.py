from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest


def _mock_job(process_dir: Path) -> SimpleNamespace:
    return SimpleNamespace(
        id=1,
        book_id=1,
        process_dir=str(process_dir),
        epub_path=str(process_dir / "book.epub"),
        current_step="bootstrap",
        stage="init",
        status="pending",
        created_at=None,
    )


def _ebook(filename: str) -> Path:
    path = (Path("ebooks") / filename).resolve()
    if not path.is_file():
        pytest.skip(f"Test EPUB not found (place in ebooks/): {path}")
    return path


class TestCheckArtifactsExist:
    def test_bootstrap_no_images_dir_returns_false(self, tmp_path: Path) -> None:
        from syntrive.workflow.engine import WorkflowEngine, WorkflowStep

        engine = WorkflowEngine(job=_mock_job(tmp_path), db_path=tmp_path / "db.sqlite")
        assert engine.check_artifacts_exist(WorkflowStep.BOOTSTRAP) is False

    def test_bootstrap_empty_images_dir_returns_false(self, tmp_path: Path) -> None:
        from syntrive.workflow.engine import WorkflowEngine, WorkflowStep

        (tmp_path / "images").mkdir()
        engine = WorkflowEngine(job=_mock_job(tmp_path), db_path=tmp_path / "db.sqlite")
        assert engine.check_artifacts_exist(WorkflowStep.BOOTSTRAP) is False

    def test_bootstrap_with_image_file_returns_true(self, tmp_path: Path) -> None:
        from syntrive.workflow.engine import WorkflowEngine, WorkflowStep

        images = tmp_path / "images"
        images.mkdir()
        (images / "cover.jpg").write_bytes(b"fake-image")
        engine = WorkflowEngine(job=_mock_job(tmp_path), db_path=tmp_path / "db.sqlite")
        assert engine.check_artifacts_exist(WorkflowStep.BOOTSTRAP) is True

    def test_transcript_no_raw_dir_returns_false(self, tmp_path: Path) -> None:
        from syntrive.workflow.engine import WorkflowEngine, WorkflowStep

        engine = WorkflowEngine(job=_mock_job(tmp_path), db_path=tmp_path / "db.sqlite")
        assert engine.check_artifacts_exist(WorkflowStep.TRANSCRIPT) is False

    def test_transcript_with_html_files_returns_true(self, tmp_path: Path) -> None:
        from syntrive.workflow.engine import WorkflowEngine, WorkflowStep

        raw = tmp_path / "transcript_html" / "raw"
        raw.mkdir(parents=True)
        (raw / "ch01.html").write_text("<html/>", encoding="utf-8")
        engine = WorkflowEngine(job=_mock_job(tmp_path), db_path=tmp_path / "db.sqlite")
        assert engine.check_artifacts_exist(WorkflowStep.TRANSCRIPT) is True

    def test_tts_config_has_no_file_artifacts(self, tmp_path: Path) -> None:
        from syntrive.workflow.engine import WorkflowEngine, WorkflowStep

        engine = WorkflowEngine(job=_mock_job(tmp_path), db_path=tmp_path / "db.sqlite")
        assert engine.check_artifacts_exist(WorkflowStep.TTS_CONFIG) is False

    def test_review_html_has_no_file_artifacts(self, tmp_path: Path) -> None:
        from syntrive.workflow.engine import WorkflowEngine, WorkflowStep

        engine = WorkflowEngine(job=_mock_job(tmp_path), db_path=tmp_path / "db.sqlite")
        assert engine.check_artifacts_exist(WorkflowStep.REVIEW_HTML) is False

    def test_synthesis_with_audio_file_returns_true(self, tmp_path: Path) -> None:
        from syntrive.workflow.engine import WorkflowEngine, WorkflowStep

        audio = tmp_path / "audio"
        audio.mkdir()
        (audio / "ch01.flac").write_bytes(b"fake-audio")
        engine = WorkflowEngine(job=_mock_job(tmp_path), db_path=tmp_path / "db.sqlite")
        assert engine.check_artifacts_exist(WorkflowStep.SYNTHESIS) is True

    def test_combine_with_audiobook_file_returns_true(self, tmp_path: Path) -> None:
        from syntrive.workflow.engine import WorkflowEngine, WorkflowStep

        books = tmp_path / "audiobooks"
        books.mkdir()
        (books / "book.m4b").write_bytes(b"fake-audiobook")
        engine = WorkflowEngine(job=_mock_job(tmp_path), db_path=tmp_path / "db.sqlite")
        assert engine.check_artifacts_exist(WorkflowStep.COMBINE) is True


class TestStepNavigation:
    def test_get_next_step_returns_following_step(self, tmp_path: Path) -> None:
        from syntrive.workflow.engine import WorkflowEngine, WorkflowStep

        engine = WorkflowEngine(job=_mock_job(tmp_path), db_path=tmp_path / "db.sqlite")
        assert engine.get_next_step(WorkflowStep.BOOTSTRAP) == WorkflowStep.TRANSCRIPT_EXTRACT
        assert engine.get_next_step(WorkflowStep.TRANSCRIPT_CLEAN) == WorkflowStep.TRANSCRIPT_TEXT

    def test_get_next_step_at_last_actionable_step_returns_none(self, tmp_path: Path) -> None:
        from syntrive.workflow.engine import WorkflowEngine, WorkflowStep

        engine = WorkflowEngine(job=_mock_job(tmp_path), db_path=tmp_path / "db.sqlite")
        assert engine.get_next_step(WorkflowStep.COMBINE) is None

    def test_get_previous_step_returns_prior_step(self, tmp_path: Path) -> None:
        from syntrive.workflow.engine import WorkflowEngine, WorkflowStep

        engine = WorkflowEngine(job=_mock_job(tmp_path), db_path=tmp_path / "db.sqlite")
        assert engine.get_previous_step(WorkflowStep.TRANSCRIPT_TEXT) == WorkflowStep.TRANSCRIPT_CLEAN

    def test_get_previous_step_at_first_step_returns_none(self, tmp_path: Path) -> None:
        from syntrive.workflow.engine import WorkflowEngine, WorkflowStep

        engine = WorkflowEngine(job=_mock_job(tmp_path), db_path=tmp_path / "db.sqlite")
        assert engine.get_previous_step(WorkflowStep.BOOTSTRAP) is None


class TestTextExtractionFormatOptions:
    def test_defaults_are_raw_and_tts_script_with_tts_script_primary(
        self, tmp_path: Path
    ) -> None:
        from syntrive.workflow.engine import WorkflowEngine

        engine = WorkflowEngine(job=_mock_job(tmp_path), db_path=tmp_path / "db.sqlite")
        opts = engine.get_text_extraction_options()

        assert opts["formats"] == {"raw": True, "tts_script": True}
        assert opts["primary"] == "tts_script"

    def test_disabling_primary_falls_back_to_another_enabled_format(
        self, tmp_path: Path
    ) -> None:
        from syntrive.workflow.engine import WorkflowEngine

        engine = WorkflowEngine(job=_mock_job(tmp_path), db_path=tmp_path / "db.sqlite")
        engine.set_text_extraction_format("tts_script", False)
        opts = engine.get_text_extraction_options()

        assert opts["formats"]["tts_script"] is False
        assert opts["primary"] == "raw"

    def test_set_primary_auto_enables_the_format(self, tmp_path: Path) -> None:
        from syntrive.workflow.engine import WorkflowEngine

        engine = WorkflowEngine(job=_mock_job(tmp_path), db_path=tmp_path / "db.sqlite")
        engine.set_text_extraction_format("raw", False)
        engine.set_text_extraction_primary("raw")
        opts = engine.get_text_extraction_options()

        assert opts["primary"] == "raw"
        assert opts["formats"]["raw"] is True

    def test_unknown_format_name_is_ignored(self, tmp_path: Path) -> None:
        from syntrive.workflow.engine import WorkflowEngine

        engine = WorkflowEngine(job=_mock_job(tmp_path), db_path=tmp_path / "db.sqlite")
        engine.set_text_extraction_format("bogus", True)
        opts = engine.get_text_extraction_options()

        assert "bogus" not in opts["formats"]

    def test_transcript_text_step_label_reflects_multi_format(self) -> None:
        from syntrive.workflow.engine import _STEP_LABELS, WorkflowStep

        assert "multi-format" in _STEP_LABELS[WorkflowStep.TRANSCRIPT_TEXT].lower()


class TestTranscriptPathSelection:
    def test_default_selection_is_tts_script(self, tmp_path: Path) -> None:
        from syntrive.workflow.engine import WorkflowEngine

        engine = WorkflowEngine(job=_mock_job(tmp_path), db_path=tmp_path / "db.sqlite")
        assert engine.get_transcript_path_selection() == "tts_script"

    def test_set_transcript_path_selection_updates_state(self, tmp_path: Path) -> None:
        from syntrive.workflow.engine import WorkflowEngine

        engine = WorkflowEngine(job=_mock_job(tmp_path), db_path=tmp_path / "db.sqlite")
        engine.set_transcript_path_selection("raw")
        assert engine.get_transcript_path_selection() == "raw"

    def test_unknown_selection_is_ignored(self, tmp_path: Path) -> None:
        from syntrive.workflow.engine import WorkflowEngine

        engine = WorkflowEngine(job=_mock_job(tmp_path), db_path=tmp_path / "db.sqlite")
        engine.set_transcript_path_selection("bogus")
        assert engine.get_transcript_path_selection() == "tts_script"

    def test_transcript_review_step_label_reflects_manual_review(self) -> None:
        from syntrive.workflow.engine import _STEP_LABELS, WorkflowStep

        label = _STEP_LABELS[WorkflowStep.TRANSCRIPT_REVIEW].lower()
        assert "review" in label


class TestTranscriptReviewAiSkillHandoff:
    def test_task_description_has_no_code_based_pipeline_mentions(self, tmp_path: Path) -> None:
        from syntrive.workflow.engine import WorkflowEngine, WorkflowStep

        engine = WorkflowEngine(job=_mock_job(tmp_path), db_path=tmp_path / "db.sqlite")
        tasks = engine.describe_tasks(WorkflowStep.TRANSCRIPT_REVIEW)
        tasks_str = " ".join(tasks).lower()
        assert "code-based" not in tasks_str

    def test_task_description_points_to_ai_skill(self, tmp_path: Path) -> None:
        from syntrive.workflow.engine import WorkflowEngine, WorkflowStep

        engine = WorkflowEngine(job=_mock_job(tmp_path), db_path=tmp_path / "db.sqlite")
        tasks = engine.describe_tasks(WorkflowStep.TRANSCRIPT_REVIEW)
        tasks_str = " ".join(tasks)
        assert "tts-text-normalize-en" in tasks_str

    def test_task_description_has_no_p_key_sync_panel_mention(self, tmp_path: Path) -> None:
        from syntrive.workflow.engine import WorkflowEngine, WorkflowStep

        engine = WorkflowEngine(job=_mock_job(tmp_path), db_path=tmp_path / "db.sqlite")
        tasks = engine.describe_tasks(WorkflowStep.TRANSCRIPT_REVIEW)
        tasks_str = " ".join(tasks).lower()
        assert "sync panel" not in tasks_str
        assert "p key" not in tasks_str

    def test_task_description_shows_default_transcript_path_selection(
        self, tmp_path: Path
    ) -> None:
        from syntrive.workflow.engine import WorkflowEngine, WorkflowStep

        engine = WorkflowEngine(job=_mock_job(tmp_path), db_path=tmp_path / "db.sqlite")
        tasks = engine.describe_tasks(WorkflowStep.TRANSCRIPT_REVIEW)
        tasks_str = " ".join(tasks)
        assert "transcript_path selection" in tasks_str
        assert "tts_script" in engine.get_transcript_path_selection()

    def test_handle_transcript_review_fails_gracefully_without_cleaned_toc(
        self, tmp_path: Path
    ) -> None:
        from syntrive.workflow.engine import WorkflowEngine

        engine = WorkflowEngine(job=_mock_job(tmp_path), db_path=tmp_path / "db.sqlite")
        outcome = engine._handle_transcript_review()

        assert outcome.success is False
        assert "tts_script/0_toc.md" in (outcome.error or "")


class TestExecuteStep:
    def test_returns_overwrite_confirm_when_artifacts_exist(self, tmp_path: Path) -> None:
        from syntrive.workflow.engine import WorkflowEngine, WorkflowStep

        images = tmp_path / "images"
        images.mkdir()
        (images / "cover.jpg").write_bytes(b"fake")

        engine = WorkflowEngine(job=_mock_job(tmp_path), db_path=tmp_path / "db.sqlite")
        outcome = engine.execute_step(WorkflowStep.BOOTSTRAP, force_overwrite=False)

        assert outcome.needs_overwrite_confirm is True
        assert outcome.success is False
        assert outcome.existing_artifact_desc is not None

    def test_force_overwrite_calls_handler_despite_existing_artifacts(
        self, tmp_path: Path
    ) -> None:
        from syntrive.workflow.engine import WorkflowEngine, WorkflowStep, StepOutcome

        images = tmp_path / "images"
        images.mkdir()
        (images / "cover.jpg").write_bytes(b"fake")

        engine = WorkflowEngine(job=_mock_job(tmp_path), db_path=tmp_path / "db.sqlite")
        fake = StepOutcome(success=True, notes="patched")
        with patch.object(engine, "_dispatch_handler", return_value=fake) as mock_h:
            outcome = engine.execute_step(WorkflowStep.BOOTSTRAP, force_overwrite=True)
            mock_h.assert_called_once_with(WorkflowStep.BOOTSTRAP)
            assert outcome.success is True

    def test_no_artifacts_calls_handler_directly(self, tmp_path: Path) -> None:
        from syntrive.workflow.engine import WorkflowEngine, WorkflowStep, StepOutcome

        engine = WorkflowEngine(job=_mock_job(tmp_path), db_path=tmp_path / "db.sqlite")
        fake = StepOutcome(success=True, notes="patched")
        with patch.object(engine, "_dispatch_handler", return_value=fake) as mock_h:
            outcome = engine.execute_step(WorkflowStep.BOOTSTRAP)
            mock_h.assert_called_once_with(WorkflowStep.BOOTSTRAP)
            assert outcome.success is True

    def test_handler_exception_returns_failure_outcome(self, tmp_path: Path) -> None:
        from syntrive.workflow.engine import WorkflowEngine, WorkflowStep

        engine = WorkflowEngine(job=_mock_job(tmp_path), db_path=tmp_path / "db.sqlite")
        with patch.object(engine, "_dispatch_handler", side_effect=RuntimeError("boom")):
            outcome = engine.execute_step(WorkflowStep.BOOTSTRAP)
            assert outcome.success is False
            assert "boom" in (outcome.error or "")

    def test_no_overwrite_confirm_for_steps_without_artifacts(self, tmp_path: Path) -> None:
        from syntrive.workflow.engine import WorkflowEngine, WorkflowStep, StepOutcome

        engine = WorkflowEngine(job=_mock_job(tmp_path), db_path=tmp_path / "db.sqlite")
        fake = StepOutcome(success=True, notes="patched")
        with patch.object(engine, "_dispatch_handler", return_value=fake):
            outcome = engine.execute_step(WorkflowStep.TTS_CONFIG)
            assert outcome.needs_overwrite_confirm is False


class TestJobServiceReset:
    def test_reset_to_start_sets_bootstrap_and_pending(self, tmp_path: Path) -> None:
        ebook = _ebook("1984.epub")
        from syntrive.bootstrap import bootstrap
        from syntrive.services.job_service import JobService
        from syntrive.db.session import get_db_session
        from syntrive.db.models import Job

        db_path = tmp_path / "syntrivetts.db"
        job = bootstrap(repo_dir=tmp_path, ebook_path=ebook)

        with get_db_session(db_path) as db:
            j = db.query(Job).filter(Job.id == job.id).first()
            j.current_step = "synthesis"
            j.status = "running"

        JobService(db_path).reset_job_to_start(job.id)

        with get_db_session(db_path) as db:
            j = db.query(Job).filter(Job.id == job.id).first()
            assert j.current_step == "bootstrap"
            assert j.status == "pending"

    def test_reset_to_step_sets_chosen_step(self, tmp_path: Path) -> None:
        ebook = _ebook("1984.epub")
        from syntrive.bootstrap import bootstrap
        from syntrive.services.job_service import JobService
        from syntrive.workflow.engine import WorkflowStep
        from syntrive.db.session import get_db_session
        from syntrive.db.models import Job

        db_path = tmp_path / "syntrivetts.db"
        job = bootstrap(repo_dir=tmp_path, ebook_path=ebook)

        with get_db_session(db_path) as db:
            j = db.query(Job).filter(Job.id == job.id).first()
            j.current_step = "combine"
            j.status = "running"

        JobService(db_path).reset_job_to_step(job.id, WorkflowStep.TRANSCRIPT)

        with get_db_session(db_path) as db:
            j = db.query(Job).filter(Job.id == job.id).first()
            assert j.current_step == WorkflowStep.TRANSCRIPT.value
            assert j.status == "pending"

    def test_reset_preserves_disk_artifacts(self, tmp_path: Path) -> None:
        ebook = _ebook("1984.epub")
        from syntrive.bootstrap import bootstrap
        from syntrive.services.job_service import JobService
        from pathlib import Path as Pth

        db_path = tmp_path / "syntrivetts.db"
        job = bootstrap(repo_dir=tmp_path, ebook_path=ebook)

        sentinel = Pth(job.process_dir) / "images" / "sentinel.jpg"
        sentinel.parent.mkdir(parents=True, exist_ok=True)
        sentinel.write_bytes(b"keep-me")

        JobService(db_path).reset_job_to_start(job.id)

        assert sentinel.exists(), "reset must not delete any artifact files"

    def test_reset_nonexistent_job_does_not_raise(self, tmp_path: Path) -> None:
        from syntrive.services.job_service import JobService

        db_path = tmp_path / "syntrivetts.db"
        JobService(db_path).reset_job_to_start(99999)


class TestEnsureTtsConfig:
    @pytest.mark.parametrize("language", ["en", "zh"])
    def test_ensure_tts_config_defaults_engine_to_cosyvoice_for_any_language(
        self, tmp_path: Path, language: str
    ) -> None:
        ebook = _ebook("English-1.epub")
        from syntrive.bootstrap import bootstrap
        from syntrive.workflow.engine import WorkflowEngine
        from syntrive.db.models import Book, TtsConfig
        from syntrive.db.session import get_db_session

        db_path = tmp_path / "syntrivetts.db"
        job = bootstrap(repo_dir=tmp_path, ebook_path=ebook)

        with get_db_session(db_path) as db:
            book = db.query(Book).filter_by(id=job.book_id).first()
            book.language = language

        engine = WorkflowEngine(job, db_path)
        created = engine.ensure_tts_config()
        assert created is True

        with get_db_session(db_path) as db:
            cfg = db.query(TtsConfig).filter_by(job_id=job.id).first()
            assert cfg is not None
            assert cfg.language == language
            assert cfg.engine == "cosyvoice"


class TestEnsureTtsVoices:
    def test_ensure_narrator_creates_row_with_null_reference_by_default(
        self, tmp_path: Path
    ) -> None:
        ebook = _ebook("1984-Orwell_George.epub")
        from syntrive.bootstrap import bootstrap
        from syntrive.workflow.engine import WorkflowEngine
        from syntrive.db.models import TtsVoice
        from syntrive.db.session import get_db_session

        db_path = tmp_path / "syntrivetts.db"
        job = bootstrap(repo_dir=tmp_path, ebook_path=ebook)

        engine = WorkflowEngine(job, db_path)
        engine.ensure_tts_config()
        engine.ensure_tts_voices()

        with get_db_session(db_path) as db:
            from syntrive.db.models import TtsConfig
            cfg = db.query(TtsConfig).filter_by(job_id=job.id).first()
            narrator = db.query(TtsVoice).filter_by(tts_config_id=cfg.id, voice_id=0).first()
            assert narrator is not None
            assert narrator.name == "narrator"
            assert narrator.reference_voice_id is None

    def test_ensure_narrator_is_idempotent(self, tmp_path: Path) -> None:
        ebook = _ebook("1984-Orwell_George.epub")
        from syntrive.bootstrap import bootstrap
        from syntrive.workflow.engine import WorkflowEngine
        from syntrive.db.models import TtsVoice
        from syntrive.db.session import get_db_session

        db_path = tmp_path / "syntrivetts.db"
        job = bootstrap(repo_dir=tmp_path, ebook_path=ebook)

        engine = WorkflowEngine(job, db_path)
        engine.ensure_tts_config()
        engine.ensure_tts_voices()
        engine.ensure_tts_voices()
        engine.ensure_tts_voices()

        with get_db_session(db_path) as db:
            rows = db.query(TtsVoice).filter_by(voice_id=0).all()
            assert len(rows) == 1

    def test_reconcile_creates_updates_deletes_roles_by_name(self, tmp_path: Path) -> None:
        from syntrive.bootstrap import bootstrap
        from syntrive.workflow.engine import WorkflowEngine
        from syntrive.db.models import TtsVoice
        from syntrive.db.session import get_db_session

        db_path = tmp_path / "syntrivetts.db"
        job = bootstrap(repo_dir=tmp_path, ebook_path=_ebook("1984-Orwell_George.epub"))

        transcript_dir = tmp_path / "transcript_text" / "tts_script"
        transcript_dir.mkdir(parents=True)
        (transcript_dir / "voice_config.yaml").write_text(
            "roles:\n"
            "  - name: 哲人\n"
            "    voice: 1\n"
            "  - name: 青年\n"
            "    voice: 2\n",
            encoding="utf-8",
        )

        with get_db_session(db_path) as db:
            from syntrive.db.models import Job
            j = db.query(Job).filter_by(id=job.id).first()
            j.transcript_dir = "transcript_text/tts_script"

        engine = WorkflowEngine(job, db_path)
        engine.ensure_tts_config()
        engine.ensure_tts_voices()

        with get_db_session(db_path) as db:
            from syntrive.db.models import TtsConfig, ReferenceVoice
            cfg = db.query(TtsConfig).filter_by(job_id=job.id).first()
            rows = {r.name: r for r in db.query(TtsVoice).filter_by(tts_config_id=cfg.id).all()}
            assert set(rows) == {"narrator", "哲人", "青年"}
            assert rows["哲人"].voice_id == 1
            assert rows["青年"].voice_id == 2

            ref = ReferenceVoice(path="voices/zh/adult/female/x.wav", name="x", gender="female", language="zh")
            db.add(ref)
            db.flush()
            rows["哲人"].reference_voice_id = ref.id
            rows["青年"].excluded = True

        (transcript_dir / "voice_config.yaml").write_text(
            "roles:\n"
            "  - name: 哲人\n"
            "    voice: 3\n"
            "  - name: 青年\n"
            "    voice: 2\n"
            "  - name: 老者\n"
            "    voice: 4\n",
            encoding="utf-8",
        )
        engine.ensure_tts_voices()

        with get_db_session(db_path) as db:
            cfg = db.query(TtsConfig).filter_by(job_id=job.id).first()
            rows = {r.name: r for r in db.query(TtsVoice).filter_by(tts_config_id=cfg.id).all()}
            assert set(rows) == {"narrator", "哲人", "青年", "老者"}
            assert rows["哲人"].voice_id == 3
            assert rows["哲人"].reference_voice_id is not None
            assert rows["青年"].excluded is True
            assert rows["老者"].reference_voice_id is None

        (transcript_dir / "voice_config.yaml").write_text(
            "roles:\n"
            "  - name: 哲人\n"
            "    voice: 3\n"
            "  - name: 老者\n"
            "    voice: 4\n",
            encoding="utf-8",
        )
        engine.ensure_tts_voices()

        with get_db_session(db_path) as db:
            cfg = db.query(TtsConfig).filter_by(job_id=job.id).first()
            rows = {r.name: r for r in db.query(TtsVoice).filter_by(tts_config_id=cfg.id).all()}
            assert set(rows) == {"narrator", "哲人", "老者"}

    def test_handle_tts_config_gate_requires_narrator_reference_voice(self, tmp_path: Path) -> None:
        from syntrive.bootstrap import bootstrap
        from syntrive.workflow.engine import WorkflowEngine
        from syntrive.db.models import ReferenceVoice, TtsConfig, TtsVoice
        from syntrive.db.session import get_db_session

        db_path = tmp_path / "syntrivetts.db"
        job = bootstrap(repo_dir=tmp_path, ebook_path=_ebook("1984-Orwell_George.epub"))
        engine = WorkflowEngine(job, db_path)

        outcome = engine._handle_tts_config()
        assert outcome.success is False
        assert "narrator voice" in outcome.error.lower()

        with get_db_session(db_path) as db:
            cfg = db.query(TtsConfig).filter_by(job_id=job.id).first()
            ref = ReferenceVoice(path="voices/en/adult/male/y.wav", name="y", gender="male", language="en")
            db.add(ref)
            db.flush()
            db.query(TtsVoice).filter_by(tts_config_id=cfg.id, voice_id=0).update(
                {"reference_voice_id": ref.id}
            )

        outcome2 = engine._handle_tts_config()
        assert outcome2.success is True
