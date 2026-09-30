from __future__ import annotations

import pytest

from syntrive.adapters.tts import model_catalog

_ALL_ENGINES = ("xtts", "cosyvoice", "indextts", "voxcpm", "f5tts", "qwen3tts")
_NON_TTS_GROUPS = ("asr",)


class TestRegistryIntegrity:
    def test_every_engine_is_registered(self):
        assert set(model_catalog.engines()) == set(_ALL_ENGINES) | set(_NON_TTS_GROUPS)

    def test_has_venv_flag_separates_tts_engines_from_asr(self):
        assert all(model_catalog.has_venv(e) for e in _ALL_ENGINES)
        assert not any(model_catalog.has_venv(e) for e in _NON_TTS_GROUPS)

    def test_asr_default_is_faster_whisper_small_under_models_tts(self):
        spec = model_catalog.default_model("asr")
        assert spec.model_id == "faster-whisper-small"
        assert spec.hf_repo == "Systran/faster-whisper-small"
        assert spec.local_path().name == "models--Systran--faster-whisper-small"

    @pytest.mark.parametrize("engine", _ALL_ENGINES)
    def test_engine_has_at_least_one_model(self, engine):
        assert model_catalog.models_for(engine)

    @pytest.mark.parametrize("engine", _ALL_ENGINES)
    def test_exactly_one_default_per_engine(self, engine):
        defaults = [s for s in model_catalog.models_for(engine) if s.is_default]
        assert len(defaults) == 1

    @pytest.mark.parametrize("engine", _ALL_ENGINES)
    def test_model_ids_are_unique_within_engine(self, engine):
        ids = [s.model_id for s in model_catalog.models_for(engine)]
        assert len(ids) == len(set(ids))

    @pytest.mark.parametrize("engine", _ALL_ENGINES)
    def test_local_path_is_under_models_tts(self, engine):
        for spec in model_catalog.models_for(engine):
            p = spec.local_path()
            assert p.parent.name == "tts"
            assert p.parent.parent.name == "models"


class TestResolve:
    def test_resolve_known_id(self):
        assert model_catalog.resolve("cosyvoice", "cosyvoice-300m-sft").model_id == "cosyvoice-300m-sft"

    def test_resolve_none_returns_engine_default(self):
        assert model_catalog.resolve("qwen3tts", None).model_id == "qwen3-tts-1.7b-base"

    def test_resolve_unknown_id_degrades_to_default_not_error(self):
        assert model_catalog.resolve("indextts", "does-not-exist").model_id == "indextts-2.5"


class TestRunnerKwargs:
    def test_xtts_gives_model_repo(self):
        assert model_catalog.runner_kwargs("xtts", "internal") == {"model_repo": "coqui/XTTS-v2"}

    def test_cosyvoice_gives_a_local_model_dir(self):
        kw = model_catalog.runner_kwargs("cosyvoice", "cosyvoice-300m-sft")
        assert set(kw) == {"model_dir"}
        assert kw["model_dir"].replace("\\", "/").endswith("models/tts/CosyVoice-300M-SFT")

    def test_indextts_2_5_local_dir(self):
        kw = model_catalog.runner_kwargs("indextts", "indextts-2.5")
        assert kw["model_dir"].replace("\\", "/").endswith("models/tts/IndexTTS-2.5")

    def test_f5tts_gives_the_f5_model_name(self):
        assert model_catalog.runner_kwargs("f5tts", "e2tts-base") == {"model": "E2TTS_Base"}

    def test_qwen3tts_gives_the_hf_repo(self):
        assert model_catalog.runner_kwargs("qwen3tts", "qwen3-tts-0.6b-base") == {
            "model": "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
        }

    def test_none_model_id_still_resolves_to_the_default_kwargs(self):
        assert model_catalog.runner_kwargs("cosyvoice", None)["model_dir"].endswith("Fun-CosyVoice3-0.5B")


class TestChoicesForGui:
    def test_marks_missing_models_not_downloaded(self):
        pairs = model_catalog.model_choices_for("indextts")
        assert all("(not downloaded)" in disp for _mid, disp in pairs)

    def test_extra_labels_are_appended_verbatim(self):
        pairs = model_catalog.model_choices_for("xtts", extra_labels=("RosamundPike", "BryanCranston"))
        assert pairs[-2:] == [("RosamundPike", "RosamundPike"), ("BryanCranston", "BryanCranston")]


class TestBundledExtraRepos:
    def _spec(self):
        spec = model_catalog.resolve("f5tts", "f5tts-v1-base")
        assert spec.extra_repos, "f5tts-v1-base must carry the vocos extra_repo"
        return spec

    def test_extra_repo_dirs_point_under_models_tts(self):
        spec = self._spec()
        dirs = model_catalog.extra_repo_dirs(spec)
        assert [d.name for d in dirs] == ["models--charactr--vocos-mel-24khz"]

    def test_is_downloaded_false_when_primary_present_but_bundle_missing(self, tmp_path, monkeypatch):
        registry = model_catalog._registry
        monkeypatch.setattr(registry, "TTS_CACHE_DIR", tmp_path)
        monkeypatch.setattr(registry.manifest, "is_healthy", lambda engine, item: None)
        spec = self._spec()

        (spec.local_path() / "snapshots" / "abc").mkdir(parents=True)
        (spec.local_path() / "snapshots" / "abc" / "F5TTS_v1_Base").mkdir()
        assert model_catalog.missing_extra_repos(spec) == ("charactr/vocos-mel-24khz",)
        assert model_catalog.is_downloaded(spec) is False

        vocos = model_catalog.extra_repo_dirs(spec)[0] / "snapshots" / "def"
        vocos.mkdir(parents=True)
        (vocos / "pytorch_model.bin").write_bytes(b"x")
        assert model_catalog.missing_extra_repos(spec) == ()
        assert model_catalog.is_downloaded(spec) is True

    def test_bundle_dirs_is_primary_plus_each_extra(self, tmp_path, monkeypatch):
        monkeypatch.setattr(model_catalog._registry, "TTS_CACHE_DIR", tmp_path)
        spec = self._spec()
        dirs = model_catalog.bundle_dirs(spec)
        assert dirs[0] == spec.local_path()
        assert dirs[1].name == "models--charactr--vocos-mel-24khz"


class TestApplyOfflineEnvIfCached:
    _OFFLINE_VARS = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")

    @pytest.fixture(autouse=True)
    def _isolate(self, tmp_path, monkeypatch):
        import os

        registry = model_catalog._registry
        monkeypatch.setattr(registry, "TTS_CACHE_DIR", tmp_path)
        monkeypatch.setattr(registry.manifest, "is_healthy", lambda engine, item: None)
        for var in (*self._OFFLINE_VARS, "HF_HUB_DISABLE_TELEMETRY"):
            monkeypatch.delenv(var, raising=False)
        self._cache_dir = tmp_path
        self._os = os

    def _lay_down_f5tts(self, *, with_vocos: bool):
        spec = model_catalog.resolve("f5tts", "f5tts-v1-base")
        (spec.local_path() / "snapshots" / "abc" / "F5TTS_v1_Base").mkdir(parents=True)
        if with_vocos:
            vocos = model_catalog.extra_repo_dirs(spec)[0] / "snapshots" / "def"
            vocos.mkdir(parents=True)
            (vocos / "pytorch_model.bin").write_bytes(b"x")
        return spec

    def test_fully_cached_forces_offline_env_and_returns_true(self):
        self._lay_down_f5tts(with_vocos=True)
        engaged = model_catalog.apply_offline_env_if_cached(
            "f5tts", "f5tts-v1-base", log=lambda _m: None
        )
        assert engaged is True
        for var in self._OFFLINE_VARS:
            assert self._os.environ[var] == "1"

    def test_missing_bundled_repo_is_a_no_op(self):
        self._lay_down_f5tts(with_vocos=False)
        engaged = model_catalog.apply_offline_env_if_cached(
            "f5tts", "f5tts-v1-base", log=lambda _m: None
        )
        assert engaged is False
        for var in self._OFFLINE_VARS:
            assert var not in self._os.environ

    def test_nothing_on_disk_is_a_no_op(self):
        engaged = model_catalog.apply_offline_env_if_cached(
            "voxcpm", "voxcpm2", log=lambda _m: None
        )
        assert engaged is False
        assert "HF_HUB_OFFLINE" not in self._os.environ

    def test_does_not_clobber_an_explicit_operator_value(self, monkeypatch):
        monkeypatch.setenv("HF_HUB_OFFLINE", "0")
        self._lay_down_f5tts(with_vocos=True)
        engaged = model_catalog.apply_offline_env_if_cached("f5tts", "f5tts-v1-base")
        assert engaged is True
        assert self._os.environ["HF_HUB_OFFLINE"] == "0"


class TestResolvedSnapshotDir:
    def test_none_when_absent(self, tmp_path):
        assert model_catalog.resolved_snapshot_dir(tmp_path / "nope") is None

    def test_none_when_empty(self, tmp_path):
        (tmp_path / "empty").mkdir()
        assert model_catalog.resolved_snapshot_dir(tmp_path / "empty") is None

    def test_flat_local_dir_layout_returns_the_dir_itself(self, tmp_path):
        flat = tmp_path / "models--charactr--vocos-mel-24khz"
        flat.mkdir()
        (flat / "config.yaml").write_text("x")
        assert model_catalog.resolved_snapshot_dir(flat) == flat

    def test_hf_cache_layout_prefers_refs_main(self, tmp_path):
        base = tmp_path / "models--charactr--vocos-mel-24khz"
        (base / "snapshots" / "aaa").mkdir(parents=True)
        (base / "snapshots" / "bbb").mkdir(parents=True)
        (base / "refs").mkdir()
        (base / "refs" / "main").write_text("aaa\n")
        assert model_catalog.resolved_snapshot_dir(base) == base / "snapshots" / "aaa"

    def test_hf_cache_layout_falls_back_to_newest_sha_without_refs(self, tmp_path):
        import os
        import time

        base = tmp_path / "models--x--y"
        old = base / "snapshots" / "old"
        new = base / "snapshots" / "new"
        old.mkdir(parents=True)
        new.mkdir(parents=True)
        past = time.time() - 100
        os.utime(old, (past, past))
        assert model_catalog.resolved_snapshot_dir(base) == new
