from __future__ import annotations

import os
import subprocess
import sys

from pathlib import Path

import syntrive.tui.tts_editor_gui as teg


class TestLayoutTabFields:
    def test_normal_fields_pair_up_two_per_row(self):
        params = [p for p in teg._PARAMS if p.tab == teg.TAB_OUTPUT]
        rows = teg.layout_tab_fields(params)
        assert [p.field for p in rows[0]] == ["output_format", "output_split"]
        assert [p.field for p in rows[1]] == ["output_split_minutes"]

    def test_row_break_after_field_gets_its_own_row(self):
        params = [p for p in teg._PARAMS if p.tab == teg.TAB_BASIC]
        rows = teg.layout_tab_fields(params)
        device_row = next(r for r in rows if r[0].field == "device")
        assert len(device_row) == 1

    def test_full_width_fields_get_their_own_row(self, monkeypatch):
        synthetic = teg.TtsParam("synthetic_wide", "Synthetic", teg.InputType.TEXT, teg.TAB_BASIC)
        monkeypatch.setattr(teg, "_FULL_WIDTH_FIELDS", frozenset({"synthetic_wide"}))
        params = [p for p in teg._PARAMS if p.tab == teg.TAB_BASIC] + [synthetic]
        rows = teg.layout_tab_fields(params)
        wide_row = next(r for r in rows if r[0].field == "synthetic_wide")
        assert len(wide_row) == 1

    def test_tuning_tab_has_no_dangling_partial_row_bug(self):
        params = [p for p in teg._PARAMS if p.tab == teg.TAB_TUNING]
        rows = teg.layout_tab_fields(params)
        assert sum(len(r) for r in rows) == len(params)
        assert len(rows[-1]) == 1


class TestEngineChoices:
    def _engine_param(self) -> teg.TtsParam:
        return next(p for p in teg._PARAMS if p.field == "engine")

    def test_bark_is_no_longer_offered(self):
        assert "bark" not in self._engine_param().choices

    def test_two_experimental_engines_are_offered(self):
        assert {"qwen3tts", "voxcpm"} <= set(self._engine_param().choices)

    def test_choice_set_matches_design_fr1_exactly(self):
        assert set(self._engine_param().choices) == {
            "cosyvoice", "indextts", "qwen3tts", "voxcpm",
        }

    def test_default_engine_is_cosyvoice(self):
        assert self._engine_param().default == "cosyvoice"

    def test_fine_tuned_model_param_replaced_by_model(self):
        fields = {p.field for p in teg._PARAMS}
        assert "model" in fields
        assert "fine_tuned_model" not in fields
        model_param = next(p for p in teg._PARAMS if p.field == "model")
        assert model_param.input_type == teg.InputType.SELECT
        assert model_param.tab == teg.TAB_BASIC

    def test_engine_choices_exact_tuple_and_default(self):
        engine_param = self._engine_param()
        assert engine_param.choices == ("cosyvoice", "indextts", "qwen3tts", "voxcpm")
        assert engine_param.default == "cosyvoice"


class TestSpeedParameter:
    def test_speed_step_is_0_05(self):
        speed_param = next(p for p in teg._PARAMS if p.field == "speed")
        assert speed_param.step == 0.05


class TestModelChoicesForEngine:
    def test_every_engine_has_at_least_one_model(self):
        for engine in ("xtts", "cosyvoice", "indextts", "voxcpm", "f5tts", "qwen3tts"):
            assert teg.model_choices_for(engine), engine

    def test_returns_model_id_display_pairs(self):
        pairs = teg.model_choices_for("cosyvoice")
        ids = [mid for mid, _ in pairs]
        assert "cosyvoice2-0.5b" in ids
        assert all(isinstance(mid, str) and isinstance(disp, str) for mid, disp in pairs)

    def test_not_downloaded_models_are_marked(self):
        pairs = dict((d, m) for m, d in teg.model_choices_for("qwen3tts"))
        assert any("(not downloaded)" in d for d in pairs)

    def test_xtts_list_includes_internal(self):
        ids = [mid for mid, _ in teg.model_choices_for("xtts")]
        assert "internal" in ids


class TestScanFinetunedModelDirs:
    def test_finds_checkpoint_dirs_by_direct_checkpoint_file(self, tmp_path: Path):
        base = tmp_path / "finetuned"
        good_dir = base / "snapshots" / "abc" / "SomeModel"
        good_dir.mkdir(parents=True)
        (good_dir / "model.safetensors").write_bytes(b"\x00")

        found = teg._scan_finetuned_model_dirs(base)
        assert list(found) == ["SomeModel"]
        assert found["SomeModel"] == good_dir

    def test_reserved_dirname_itself_is_not_treated_as_a_model(self, tmp_path: Path):
        base = tmp_path / "finetuned"
        blobs_dir = base / "blobs"
        blobs_dir.mkdir(parents=True)
        (blobs_dir / "model.safetensors").write_bytes(b"\x00")

        found = teg._scan_finetuned_model_dirs(base)
        assert found == {}

    def test_missing_base_dir_returns_empty(self, tmp_path: Path):
        assert teg._scan_finetuned_model_dirs(tmp_path / "does_not_exist") == {}


class TestVoicePreviewBehavior:
    def test_voice_selection_change_stops_current_preview(self):
        gui = object.__new__(teg.TtsParamsGui)
        called = {"stop": 0}

        def _stop():
            called["stop"] += 1

        gui._stop_voice_play = _stop
        gui._on_voice_row_selection_changed(101)

        assert called["stop"] == 1


class TestRunTtsEditorGuiSubprocess:
    def _seed_job(self, db_path: Path) -> int:
        from syntrive.db.models import Book, Job
        from syntrive.db.session import ensure_schema, get_db_session, make_engine

        ensure_schema(make_engine(db_path))
        with get_db_session(db_path) as db:
            book = Book(title="Test Book", language="en")
            db.add(book)
            db.flush()
            job = Job(
                book_id=book.id, process_dir=str(db_path.parent),
                epub_path="book.epub", stage="init", status="pending", current_step="bootstrap",
            )
            db.add(job)
            db.flush()
            return job.id

    def _run_subprocess(self, db_path: Path, job_id: int, autoclick: str) -> subprocess.CompletedProcess:
        env = dict(os.environ, SYNTRIVE_TTS_EDITOR_TEST_AUTOCLICK=autoclick)
        module_path = Path(teg.__file__).resolve()
        return subprocess.run(
            [sys.executable, str(module_path), str(db_path), str(job_id)],
            env=env, capture_output=True, text=True, timeout=30,
        )

    def test_save_path_exits_zero_and_persists(self, tmp_path: Path):
        db_path = tmp_path / "syntrivetts.db"
        job_id = self._seed_job(db_path)

        result = self._run_subprocess(db_path, job_id, "save")
        assert result.returncode == 0, result.stderr

        from syntrive.db.session import get_db_session
        from syntrive.db.repository import JobRepository

        with get_db_session(db_path) as db:
            job = JobRepository(db).get_job(job_id)
            assert job.tts_config is not None
            assert job.tts_config.engine == "cosyvoice"

    def test_cancel_path_exits_nonzero_and_does_not_persist(self, tmp_path: Path):
        db_path = tmp_path / "syntrivetts.db"
        job_id = self._seed_job(db_path)

        result = self._run_subprocess(db_path, job_id, "cancel")
        assert result.returncode != 0, result.stderr

        from syntrive.db.session import get_db_session
        from syntrive.db.repository import JobRepository

        with get_db_session(db_path) as db:
            job = JobRepository(db).get_job(job_id)
            assert job.tts_config is None

    def test_run_tts_editor_gui_wrapper_matches_subprocess_exit_code(self, tmp_path: Path, monkeypatch):
        db_path = tmp_path / "syntrivetts.db"
        job_id = self._seed_job(db_path)
        monkeypatch.setenv("SYNTRIVE_TTS_EDITOR_TEST_AUTOCLICK", "save")

        saved = teg.run_tts_editor_gui(job_id=job_id, db_path=db_path)
        assert saved is True
