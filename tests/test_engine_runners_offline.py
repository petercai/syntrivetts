from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_ENGINES_DIR = _REPO_ROOT / "engines"


def _load_runner(engine_name: str) -> ModuleType:
    module_name = f"_test_engine_runner_{engine_name}"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, _ENGINES_DIR / engine_name / "runner.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_OFFLINE_ENV_VARS = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY")

_MODEL_ENV_KEYS = (
    "HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "HF_DATASETS_CACHE", "TORCH_HOME",
    "TOKENIZERS_PARALLELISM", "HF_HUB_DISABLE_TELEMETRY",
    "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "PYTORCH_ENABLE_MPS_FALLBACK", "COQUI_TOS_AGREED",
)


@pytest.fixture(autouse=True)
def _clean_offline_env(monkeypatch):
    for var in ("SYNTRIVE_TTS_OFFLINE", *_OFFLINE_ENV_VARS, *_MODEL_ENV_KEYS):
        monkeypatch.delenv(var, raising=False)


@pytest.mark.parametrize("engine_name", ["xtts", "indextts", "voxcpm", "cosyvoice", "f5tts", "qwen3tts"])
class TestReleaseAcceleratorMemoryAcrossEngines:
    def test_is_the_shared_symbol(self, engine_name):
        runner = _load_runner(engine_name)
        assert runner.release_accelerator_memory.__module__ == "_shared.runtime_env"

    def test_calls_cuda_empty_cache_when_cuda_available(self, engine_name, monkeypatch):
        runner = _load_runner(engine_name)
        import torch

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        calls = []
        monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append("cuda"))
        messages = []

        runner.release_accelerator_memory(log=messages.append, reason="test_reason")

        assert calls == ["cuda"]
        assert any("accelerator_memory_released" in m and "cuda" in m for m in messages)

    def test_calls_mps_empty_cache_when_only_mps_available(self, engine_name, monkeypatch):
        runner = _load_runner(engine_name)
        import torch

        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
        calls = []
        monkeypatch.setattr(torch.mps, "empty_cache", lambda: calls.append("mps"))
        messages = []

        runner.release_accelerator_memory(log=messages.append, reason="test_reason")

        assert calls == ["mps"]
        assert any("accelerator_memory_released" in m and "mps" in m for m in messages)

    def test_noop_when_neither_backend_available(self, engine_name, monkeypatch):
        runner = _load_runner(engine_name)
        import torch

        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
        messages = []

        runner.release_accelerator_memory(log=messages.append, reason="test_reason")

        assert messages == []

    def test_never_raises_when_empty_cache_itself_fails(self, engine_name, monkeypatch):
        runner = _load_runner(engine_name)
        import torch

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

        def _boom():
            raise RuntimeError("driver gone")

        monkeypatch.setattr(torch.cuda, "empty_cache", _boom)
        messages = []

        runner.release_accelerator_memory(log=messages.append, reason="test_reason")

        assert any("accelerator_memory_release_failed" in m for m in messages)


@pytest.mark.parametrize("engine_name", ["xtts", "indextts", "voxcpm", "cosyvoice", "f5tts", "qwen3tts"])
class TestApplyOfflineModeAcrossEngines:
    def test_is_the_shared_symbol(self, engine_name):
        runner = _load_runner(engine_name)
        assert runner.apply_offline_mode.__module__ == "_shared.runtime_env"

    def test_returns_false_and_sets_nothing_when_flag_unset(self, engine_name):
        runner = _load_runner(engine_name)

        assert runner.apply_offline_mode(log=lambda _m: None) is False
        for var in _OFFLINE_ENV_VARS:
            assert var not in __import__("os").environ

    def test_returns_true_and_sets_hf_env_vars_when_flag_set(self, engine_name, monkeypatch):
        monkeypatch.setenv("SYNTRIVE_TTS_OFFLINE", "1")
        runner = _load_runner(engine_name)

        assert runner.apply_offline_mode(log=lambda _m: None) is True
        import os

        for var in _OFFLINE_ENV_VARS:
            assert os.environ[var] == "1"

    def test_does_not_clobber_a_pre_existing_env_var_value(self, engine_name, monkeypatch):
        monkeypatch.setenv("SYNTRIVE_TTS_OFFLINE", "1")
        monkeypatch.setenv("HF_HUB_OFFLINE", "0")
        runner = _load_runner(engine_name)

        runner.apply_offline_mode(log=lambda _m: None)

        import os

        assert os.environ["HF_HUB_OFFLINE"] == "0"


@pytest.mark.parametrize("engine_name", ["f5tts", "qwen3tts", "voxcpm"])
class TestAutoOfflineWhenCached:
    _LIB_MODULE = {"f5tts": "f5_tts.api", "qwen3tts": "qwen_tts", "voxcpm": "voxcpm"}

    def test_hook_is_the_shared_registry_symbol(self, engine_name):
        runner = _load_runner(engine_name)
        assert (
            runner.model_registry.apply_offline_env_if_cached.__module__
            == "_shared.model_registry"
        )

    def test_load_model_consults_the_hook_with_the_engine_name(self, engine_name, monkeypatch):
        import os
        from unittest.mock import MagicMock

        runner = _load_runner(engine_name)

        lib_name = self._LIB_MODULE[engine_name]
        monkeypatch.setitem(sys.modules, lib_name, MagicMock())
        if "." in lib_name:
            monkeypatch.setitem(sys.modules, lib_name.split(".")[0], MagicMock())

        calls: list = []

        def _spy(engine, model_id, log=None):
            calls.append((engine, model_id))
            return False

        monkeypatch.setattr(runner.model_registry, "apply_offline_env_if_cached", _spy)

        try:
            runner._load_model({})
        except Exception:
            pass

        assert calls and calls[0][0] == engine_name
        for var in _OFFLINE_ENV_VARS:
            assert var not in os.environ


@pytest.mark.parametrize("engine_name", ["xtts", "indextts", "voxcpm", "cosyvoice", "f5tts", "qwen3tts"])
class TestConfigureModelEnvAcrossEngines:
    def test_is_the_shared_symbol(self, engine_name):
        runner = _load_runner(engine_name)
        assert runner.configure_model_env.__module__ == "_shared.runtime_env"
        assert runner.TTS_CACHE_DIR.name == "tts"
        assert runner.TTS_CACHE_DIR.parent.name == "models"

    def test_fills_every_default_when_env_is_clean(self, engine_name):
        import os

        runner = _load_runner(engine_name)
        runner.configure_model_env(log=lambda _m: None)

        tts_cache = str(runner.TTS_CACHE_DIR)
        for cache_var in ("HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE",
                          "HF_DATASETS_CACHE", "TORCH_HOME"):
            assert os.environ[cache_var] == tts_cache
        assert os.environ["TOKENIZERS_PARALLELISM"] == "false"
        assert os.environ["HF_HUB_DISABLE_TELEMETRY"] == "1"
        assert os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] == "1"
        assert os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] == "1"
        assert os.environ["COQUI_TOS_AGREED"] == "1"

    def test_never_clobbers_an_operator_set_value(self, engine_name, monkeypatch):
        import os

        monkeypatch.setenv("HF_HOME", "/operator/custom/hf")
        monkeypatch.setenv("COQUI_TOS_AGREED", "0")
        runner = _load_runner(engine_name)

        runner.configure_model_env(log=lambda _m: None)

        assert os.environ["HF_HOME"] == "/operator/custom/hf"
        assert os.environ["COQUI_TOS_AGREED"] == "0"
        assert os.environ["TORCH_HOME"] == str(runner.TTS_CACHE_DIR)


class TestCosyvoiceWetextOfflineCtx:
    def test_offline_false_is_a_no_op(self):
        runner = _load_runner("cosyvoice")

        with runner._wetext_offline_ctx(False):
            pass

        assert "wetext.wetext" not in sys.modules

    def test_offline_true_wraps_and_restores_snapshot_download(self, monkeypatch):
        runner = _load_runner("cosyvoice")

        calls = []

        def fake_snapshot_download(model_id, *args, **kwargs):
            calls.append((model_id, kwargs))
            return "/fake/local/path"

        fake_wetext_wetext = ModuleType("wetext.wetext")
        fake_wetext_wetext.snapshot_download = fake_snapshot_download
        fake_wetext_pkg = ModuleType("wetext")
        monkeypatch.setitem(sys.modules, "wetext", fake_wetext_pkg)
        monkeypatch.setitem(sys.modules, "wetext.wetext", fake_wetext_wetext)

        with runner._wetext_offline_ctx(True):
            import wetext.wetext as wt

            wt.snapshot_download("some-model")
            assert calls == [("some-model", {"local_files_only": True})]

        import wetext.wetext as wt_after

        assert wt_after.snapshot_download is fake_snapshot_download

    def test_offline_true_but_wetext_not_installed_does_not_raise(self, monkeypatch):
        runner = _load_runner("cosyvoice")
        monkeypatch.delitem(sys.modules, "wetext.wetext", raising=False)
        monkeypatch.delitem(sys.modules, "wetext", raising=False)

        with runner._wetext_offline_ctx(True):
            pass


class TestXttsResolveHfFile:
    def _runner_with_patched_download(self, monkeypatch, side_effect):
        import huggingface_hub

        runner = _load_runner("xtts")
        calls = []

        def fake_hf_hub_download(**kwargs):
            calls.append(kwargs)
            result = side_effect(kwargs, len(calls))
            if isinstance(result, Exception):
                raise result
            return result

        monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_hf_hub_download)
        return runner, calls

    def test_cache_hit_returns_local_path_with_no_second_call(self, monkeypatch):
        runner, calls = self._runner_with_patched_download(
            monkeypatch, lambda kwargs, n: "/models/tts/cached/config.json"
        )

        path = runner._resolve_hf_file("coqui/XTTS-v2", "config.json", "/models/tts", offline=False)

        assert path == "/models/tts/cached/config.json"
        assert len(calls) == 1
        assert calls[0]["local_files_only"] is True

    def test_online_cache_miss_falls_back_to_a_real_download(self, monkeypatch):
        from huggingface_hub.errors import LocalEntryNotFoundError

        def side_effect(kwargs, n):
            if kwargs.get("local_files_only"):
                return LocalEntryNotFoundError("not cached")
            return "/models/tts/freshly/downloaded/model.pth"

        runner, calls = self._runner_with_patched_download(monkeypatch, side_effect)

        path = runner._resolve_hf_file("coqui/XTTS-v2", "model.pth", "/models/tts", offline=False)

        assert path == "/models/tts/freshly/downloaded/model.pth"
        assert len(calls) == 2
        assert calls[0]["local_files_only"] is True
        assert "local_files_only" not in calls[1]

    def test_offline_cache_miss_raises_actionable_error_without_downloading(self, monkeypatch):
        from huggingface_hub.errors import LocalEntryNotFoundError

        runner, calls = self._runner_with_patched_download(
            monkeypatch, lambda kwargs, n: LocalEntryNotFoundError("not cached")
        )

        with pytest.raises(RuntimeError, match="offline mode.*not in the local cache"):
            runner._resolve_hf_file("coqui/XTTS-v2", "vocab.json", "/models/tts", offline=True)

        assert len(calls) == 1


class TestF5ttsCacheLocation:
    @pytest.fixture(autouse=True)
    def _isolate_from_local_models(self, monkeypatch):
        runner = _load_runner("f5tts")
        monkeypatch.setattr(
            runner.model_registry, "apply_offline_env_if_cached", lambda *a, **k: False
        )
        monkeypatch.setattr(runner, "_resolve_local_vocoder", lambda _spec: None)
        monkeypatch.setattr(runner, "_resolve_local_checkpoint", lambda _spec, _name: None)

    def test_default_cache_dir_is_repo_models_tts(self):
        runner = _load_runner("f5tts")

        assert runner._DEFAULT_CACHE_DIR == _REPO_ROOT / "models" / "tts"

    def test_load_model_passes_hf_cache_dir_to_F5TTS(self, monkeypatch):
        runner = _load_runner("f5tts")

        captured = {}

        class FakeF5TTS:
            def __init__(self, model, device, hf_cache_dir, **kwargs):
                captured["model"] = model
                captured["device"] = device
                captured["hf_cache_dir"] = hf_cache_dir
                self.target_sample_rate = 24000

        fake_api = ModuleType("f5_tts.api")
        fake_api.F5TTS = FakeF5TTS
        fake_pkg = ModuleType("f5_tts")
        monkeypatch.setitem(sys.modules, "f5_tts", fake_pkg)
        monkeypatch.setitem(sys.modules, "f5_tts.api", fake_api)

        model = runner._load_model({"device": "cpu"})

        assert model.target_sample_rate == 24000
        assert captured["hf_cache_dir"] == str(_REPO_ROOT / "models" / "tts")

    def test_config_cache_dir_overrides_the_default(self, monkeypatch):
        runner = _load_runner("f5tts")
        captured = {}

        class FakeF5TTS:
            def __init__(self, model, device, hf_cache_dir, **kwargs):
                captured["hf_cache_dir"] = hf_cache_dir
                self.target_sample_rate = 22050

        fake_api = ModuleType("f5_tts.api")
        fake_api.F5TTS = FakeF5TTS
        fake_pkg = ModuleType("f5_tts")
        monkeypatch.setitem(sys.modules, "f5_tts", fake_pkg)
        monkeypatch.setitem(sys.modules, "f5_tts.api", fake_api)

        runner._load_model({"device": "cpu", "cache_dir": "/custom/hf/cache"})

        assert captured["hf_cache_dir"] == "/custom/hf/cache"


class TestCosyvoiceRefTextGuard:
    def test_zero_shot_without_ref_text_raises_before_touching_the_model(self):
        runner = _load_runner("cosyvoice")

        with pytest.raises(RuntimeError, match="requires ref_text"):
            runner._synthesize_speech("hello", "/some/voice.wav", None)

    def test_zero_shot_with_empty_ref_text_also_raises(self):
        runner = _load_runner("cosyvoice")

        with pytest.raises(RuntimeError, match="requires ref_text"):
            runner._synthesize_speech("hello", "/some/voice.wav", "")
