from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import torch

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


@pytest.fixture(autouse=True)
def _reset_module_state(monkeypatch):
    runner = _load_runner("voxcpm")
    monkeypatch.setattr(
        runner.model_registry, "apply_offline_env_if_cached", lambda *a, **k: False
    )
    for var in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        monkeypatch.delenv(var, raising=False)
    runner._model = None
    runner._config = {}
    runner._sample_rate = runner.DEFAULT_SAMPLE_RATE
    yield
    runner._model = None
    runner._config = {}
    runner._sample_rate = runner.DEFAULT_SAMPLE_RATE


class _FakeTtsModel:
    sample_rate = 16000


class _FakeVoxCPM:
    def __init__(self):
        self.from_pretrained_kwargs: dict = {}
        self.tts_model = _FakeTtsModel()
        self.generate_calls: list = []

    @classmethod
    def from_pretrained(cls, **kwargs):
        instance = cls()
        instance.from_pretrained_kwargs = kwargs
        return instance

    def generate(self, **kwargs):
        self.generate_calls.append(kwargs)
        return np.zeros(10, dtype="float32")


def _patch_voxcpm_class(monkeypatch, fake_class=_FakeVoxCPM):
    fake_module = ModuleType("voxcpm")
    fake_module.VoxCPM = fake_class
    monkeypatch.setitem(sys.modules, "voxcpm", fake_module)


class TestLoadModel:
    def test_default_model_device_and_safe_load_flags(self, monkeypatch):
        runner = _load_runner("voxcpm")
        _patch_voxcpm_class(monkeypatch)

        model = runner._load_model({})

        assert isinstance(model, _FakeVoxCPM)
        kw = model.from_pretrained_kwargs
        assert kw["hf_model_id"] == "openbmb/VoxCPM2"
        assert kw["device"] == "cpu"
        assert kw["load_denoiser"] is False
        assert kw["optimize"] is False
        assert kw["local_files_only"] is False
        assert kw["cache_dir"] == str(runner._DEFAULT_CACHE_DIR)

    def test_model_and_device_and_optimize_overrides(self, monkeypatch):
        runner = _load_runner("voxcpm")
        _patch_voxcpm_class(monkeypatch)

        model = runner._load_model({"model_name": "openbmb/VoxCPM-0.5B", "device": "cuda", "optimize": True})

        kw = model.from_pretrained_kwargs
        assert kw["hf_model_id"] == "openbmb/VoxCPM-0.5B"
        assert kw["device"] == "cuda"
        assert kw["optimize"] is True

    def test_registry_model_id_resolves_to_the_hf_repo(self, monkeypatch):
        runner = _load_runner("voxcpm")
        _patch_voxcpm_class(monkeypatch)

        model = runner._load_model({"model": "voxcpm2"})

        assert model.from_pretrained_kwargs["hf_model_id"] == "openbmb/VoxCPM2"

    def test_offline_mode_forces_local_files_only(self, monkeypatch):
        monkeypatch.setenv("SYNTRIVE_TTS_OFFLINE", "1")
        runner = _load_runner("voxcpm")
        _patch_voxcpm_class(monkeypatch)

        model = runner._load_model({})

        assert model.from_pretrained_kwargs["local_files_only"] is True

    def test_cache_dir_override(self, monkeypatch):
        runner = _load_runner("voxcpm")
        _patch_voxcpm_class(monkeypatch)

        model = runner._load_model({"cache_dir": "/custom/hf/cache"})

        assert model.from_pretrained_kwargs["cache_dir"] == "/custom/hf/cache"

    def test_get_model_reads_real_sample_rate_and_only_loads_once(self, monkeypatch):
        runner = _load_runner("voxcpm")
        _patch_voxcpm_class(monkeypatch)
        load_calls = []
        original = runner._load_model
        monkeypatch.setattr(runner, "_load_model", lambda config: (load_calls.append(1), original(config))[1])

        first = runner._get_model({})
        second = runner._get_model({})

        assert first is second
        assert len(load_calls) == 1
        assert runner._sample_rate == 16000


class TestSynthesizeSpeech:
    def test_missing_voice_path_raises_before_touching_the_model(self):
        runner = _load_runner("voxcpm")

        with pytest.raises(RuntimeError, match="requires voice_path"):
            runner._synthesize_speech("hello", None, "some ref text")

    def test_empty_voice_path_also_raises(self):
        runner = _load_runner("voxcpm")

        with pytest.raises(RuntimeError, match="requires voice_path"):
            runner._synthesize_speech("hello", "", "some ref text")

    def test_ref_text_present_uses_prompt_path_continuation_cloning(self, monkeypatch):
        runner = _load_runner("voxcpm")
        fake_model = _FakeVoxCPM()
        monkeypatch.setattr(runner, "_get_model", lambda config: fake_model)

        wav_tensor = runner._synthesize_speech("你好", "/voices/a.wav", "参考文本")

        call = fake_model.generate_calls[0]
        assert call["text"] == "你好"
        assert call["prompt_wav_path"] == "/voices/a.wav"
        assert call["prompt_text"] == "参考文本"
        assert "reference_wav_path" not in call
        assert wav_tensor.shape == (1, 10)

    def test_ref_text_absent_uses_reference_wav_path_token_cloning(self, monkeypatch):
        runner = _load_runner("voxcpm")
        fake_model = _FakeVoxCPM()
        monkeypatch.setattr(runner, "_get_model", lambda config: fake_model)

        runner._synthesize_speech("hello", "/voices/a.wav", None)
        runner._synthesize_speech("hello", "/voices/a.wav", "")

        for call in fake_model.generate_calls:
            assert call["reference_wav_path"] == "/voices/a.wav"
            assert "prompt_wav_path" not in call
            assert "prompt_text" not in call

    def test_only_config_keys_actually_present_are_forwarded_and_cast(self, monkeypatch):
        runner = _load_runner("voxcpm")
        runner._config = {"cfg_value": "1.5", "inference_timesteps": "20", "normalize": True}
        fake_model = _FakeVoxCPM()
        monkeypatch.setattr(runner, "_get_model", lambda config: fake_model)

        runner._synthesize_speech("hello", "/voices/a.wav", "ref")

        call = fake_model.generate_calls[0]
        assert call["cfg_value"] == 1.5 and isinstance(call["cfg_value"], float)
        assert call["inference_timesteps"] == 20 and isinstance(call["inference_timesteps"], int)
        assert call["normalize"] is True
        assert "denoise" not in call
        assert "retry_badcase" not in call


class TestHandleInit:
    def test_success_returns_ok_with_features_and_real_samplerate(self, monkeypatch):
        runner = _load_runner("voxcpm")
        _patch_voxcpm_class(monkeypatch)

        response = runner.handle_init({"request_id": "r1", "config": {"device": "cpu"}})

        assert response == {
            "request_id": "r1",
            "status": "OK",
            "supported_features": ["zero_shot"],
            "samplerate": 16000,
        }
        assert runner._config == {"device": "cpu"}

    def test_model_load_failure_returns_error_without_raising(self, monkeypatch):
        runner = _load_runner("voxcpm")

        def _boom(config):
            raise RuntimeError("snapshot_download failed")

        monkeypatch.setattr(runner, "_load_model", _boom)

        response = runner.handle_init({"request_id": "r2", "config": {}})

        assert response["request_id"] == "r2"
        assert response["status"] == "ERROR"
        assert "snapshot_download failed" in response["error"]


class TestHandleSynthesize:
    def test_success_writes_wav_and_reports_duration(self, monkeypatch, tmp_path):
        runner = _load_runner("voxcpm")
        runner._sample_rate = 48000
        speech = torch.zeros(1, 4800)
        monkeypatch.setattr(runner, "_synthesize_speech", lambda text, voice_path, ref_text: speech)
        output_file = tmp_path / "out.wav"

        response = runner.handle_synthesize({
            "request_id": "r3",
            "text": "hello",
            "voice_path": "/voices/a.wav",
            "ref_text": None,
            "leading_silence_ms": 0,
            "output_file": str(output_file),
        })

        assert response["status"] == "OK"
        assert response["samplerate"] == 48000
        assert response["duration_seconds"] == pytest.approx(0.1)
        assert output_file.exists()

    def test_leading_silence_is_prepended_at_the_model_sample_rate(self, monkeypatch, tmp_path):
        runner = _load_runner("voxcpm")
        runner._sample_rate = 48000
        speech = torch.zeros(1, 4800)
        monkeypatch.setattr(runner, "_synthesize_speech", lambda text, voice_path, ref_text: speech)
        output_file = tmp_path / "out.wav"

        response = runner.handle_synthesize({
            "request_id": "r4",
            "text": "hello",
            "voice_path": "/voices/a.wav",
            "ref_text": None,
            "leading_silence_ms": 500,
            "output_file": str(output_file),
        })

        assert response["duration_seconds"] == pytest.approx(0.6)

    def test_empty_text_and_no_silence_still_writes_a_valid_empty_wav(self, tmp_path):
        runner = _load_runner("voxcpm")
        output_file = tmp_path / "out.wav"

        response = runner.handle_synthesize({
            "request_id": "r5",
            "text": "",
            "voice_path": None,
            "leading_silence_ms": 0,
            "output_file": str(output_file),
        })

        assert response["status"] == "OK"
        assert response["duration_seconds"] == 0.0
        assert output_file.exists()

    def test_synthesis_failure_returns_error_without_crashing_the_worker(self, monkeypatch, tmp_path):
        runner = _load_runner("voxcpm")

        def _boom(text, voice_path, ref_text):
            raise RuntimeError("requires voice_path")

        monkeypatch.setattr(runner, "_synthesize_speech", _boom)
        output_file = tmp_path / "out.wav"

        response = runner.handle_synthesize({
            "request_id": "r6",
            "text": "hello",
            "voice_path": None,
            "leading_silence_ms": 0,
            "output_file": str(output_file),
        })

        assert response["request_id"] == "r6"
        assert response["status"] == "ERROR"
        assert "requires voice_path" in response["error"]

    def test_synthesis_failure_releases_accelerator_memory(self, monkeypatch, tmp_path):
        runner = _load_runner("voxcpm")
        calls = []
        monkeypatch.setattr(runner, "release_accelerator_memory", lambda log=None, reason="": calls.append(reason))

        def _boom(text, voice_path, ref_text):
            raise RuntimeError("CUDA out of memory")

        monkeypatch.setattr(runner, "_synthesize_speech", _boom)
        output_file = tmp_path / "out.wav"

        response = runner.handle_synthesize({
            "request_id": "r7",
            "text": "hello",
            "voice_path": "/voices/a.wav",
            "ref_text": None,
            "leading_silence_ms": 0,
            "output_file": str(output_file),
        })

        assert response["status"] == "ERROR"
        assert calls == ["synthesize_failed"]

    def test_success_releases_accelerator_memory(self, monkeypatch, tmp_path):
        runner = _load_runner("voxcpm")
        runner._sample_rate = 48000
        calls = []
        monkeypatch.setattr(runner, "release_accelerator_memory", lambda log=None, reason="": calls.append(reason))
        speech = torch.zeros(1, 4800)
        monkeypatch.setattr(runner, "_synthesize_speech", lambda text, voice_path, ref_text: speech)
        output_file = tmp_path / "out.wav"

        response = runner.handle_synthesize({
            "request_id": "r9",
            "text": "hello",
            "voice_path": "/voices/a.wav",
            "ref_text": None,
            "leading_silence_ms": 0,
            "output_file": str(output_file),
        })

        assert response["status"] == "OK"
        assert calls == ["synthesize_ok"]
