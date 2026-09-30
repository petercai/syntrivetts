from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_ENGINES_DIR = _REPO_ROOT / "engines"

if str(_ENGINES_DIR) not in sys.path:
    sys.path.insert(0, str(_ENGINES_DIR))

from _shared import catalog  # noqa: E402

_ALL_ENGINES = ("xtts", "cosyvoice", "indextts", "voxcpm", "f5tts", "qwen3tts")
_NON_TTS_GROUPS = ("asr",)
_ALL_GROUPS = _ALL_ENGINES + _NON_TTS_GROUPS


@pytest.fixture()
def raw() -> dict:
    return catalog.load_raw()


class TestSchema:
    def test_committed_catalog_is_valid(self, raw):
        assert catalog.validate(raw) == []

    def test_every_engine_present_with_one_default(self, raw):
        assert set(catalog.engine_names(raw)) == set(_ALL_GROUPS)
        for engine in _ALL_GROUPS:
            models = catalog.models(raw, engine)
            assert models, engine
            assert sum(1 for m in models if m.get("default")) == 1, engine

    def test_asr_group_is_venv_less_and_needs_no_runner_keys(self, raw):
        assert catalog.has_venv(raw, "asr") is False
        assert all(catalog.has_venv(raw, e) for e in _ALL_ENGINES)
        asr = catalog.engine_meta(raw, "asr")
        assert "runner_key" not in asr and "local_dir_engine" not in asr
        assert catalog.validate(raw) == []

    def test_validate_still_requires_runner_keys_for_a_venv_engine(self, raw):
        bad = copy.deepcopy(raw)
        bad["engines"]["voxcpm"].pop("runner_key")
        assert any("runner_key" in p for p in catalog.validate(bad))

    def test_validate_rejects_non_boolean_venv(self, raw):
        bad = copy.deepcopy(raw)
        bad["engines"]["asr"]["venv"] = "no"
        assert any("'venv' must be true or false" in p for p in catalog.validate(bad))

    def test_asr_model_matches_faster_whisper_download_allow_list(self, raw):
        (model,) = catalog.models(raw, "asr")
        assert model["hf_repo"] == "Systran/faster-whisper-small"
        assert model["download"] == "hf_cache"
        assert {"model.bin", "tokenizer.json", "config.json"} <= set(model["allow_patterns"])

    def test_validate_catches_missing_hf_repo(self, raw):
        bad = copy.deepcopy(raw)
        bad["engines"]["voxcpm"]["models"][0].pop("hf_repo")
        problems = catalog.validate(bad)
        assert any("hf_repo" in p for p in problems)

    def test_validate_catches_duplicate_id(self, raw):
        bad = copy.deepcopy(raw)
        first = bad["engines"]["cosyvoice"]["models"][0]
        bad["engines"]["cosyvoice"]["models"].append(dict(first))
        problems = catalog.validate(bad)
        assert any("duplicate id" in p for p in problems)

    def test_validate_catches_unknown_download_kind(self, raw):
        bad = copy.deepcopy(raw)
        bad["engines"]["f5tts"]["models"][0]["download"] = "magnet"
        problems = catalog.validate(bad)
        assert any("download" in p for p in problems)

    def test_validate_catches_zero_or_two_defaults(self, raw):
        bad = copy.deepcopy(raw)
        bad["engines"]["qwen3tts"]["models"][1]["default"] = True
        problems = catalog.validate(bad)
        assert any("exactly one" in p for p in problems)


class TestSnapshotParity:
    def test_committed_json_matches_a_fresh_dump_of_the_yaml(self):
        import yaml

        with catalog.CATALOG_YAML.open(encoding="utf-8") as fh:
            yaml_raw = yaml.safe_load(fh)
        fresh = catalog.dump_json(yaml_raw)
        committed = catalog.CATALOG_JSON.read_text(encoding="utf-8")
        assert fresh == committed, "run `python tools/model_catalog.py sync` and commit"

    def test_json_snapshot_is_pure_stdlib_loadable(self):
        data = json.loads(catalog.CATALOG_JSON.read_text(encoding="utf-8"))
        assert set(data["engines"]) == set(_ALL_GROUPS)

    def test_yaml_and_json_produce_identical_registry(self, monkeypatch):
        from _shared import model_registry

        yaml_specs = {e: model_registry.models_for(e) for e in model_registry.engines()}

        monkeypatch.setattr(catalog, "source", lambda: "json")
        model_registry.reload()
        try:
            json_specs = {e: model_registry.models_for(e) for e in model_registry.engines()}
        finally:
            monkeypatch.undo()
            model_registry.reload()

        assert yaml_specs == json_specs


class TestGlossary:
    def test_per_model_override_beats_shared_glossary(self, raw):
        role = catalog.glossary_for(raw, "cosyvoice", "cosyvoice2-0.5b", "flow.pt")
        assert role and "flow-matching" in role.lower()

    def test_shared_glossary_fallback_by_extension(self, raw):
        role = catalog.glossary_for(raw, "voxcpm", "voxcpm2", "model.safetensors")
        assert role and "safetensors" in role.lower()

    def test_unknown_file_returns_none(self, raw):
        assert catalog.glossary_for(raw, "voxcpm", "voxcpm2", "mystery.xyz") is None

    def test_wetext_resource_is_listed_under_cosyvoice(self, raw):
        ids = [r["id"] for r in catalog.resources_for(raw, "cosyvoice")]
        assert "wetext" in ids


class TestCli:
    def _tool(self):
        script = _REPO_ROOT / "tools" / "model_catalog.py"
        spec = importlib.util.spec_from_file_location("_model_catalog_tool", script)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_validate_ok(self, capsys):
        assert self._tool().cmd_validate() == 0
        assert "catalog_valid" in capsys.readouterr().out

    def test_sync_check_passes_on_clean_tree(self, capsys):
        assert self._tool().cmd_sync(check_only=True) == 0

    def test_list_shows_resources(self, capsys):
        assert self._tool().cmd_list() == 0
        assert "wetext" in capsys.readouterr().out
