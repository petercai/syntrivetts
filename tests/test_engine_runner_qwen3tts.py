from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

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
    runner = _load_runner("qwen3tts")
    monkeypatch.setattr(
        runner.model_registry, "apply_offline_env_if_cached", lambda *a, **k: False
    )
    monkeypatch.setattr(
        runner.model_registry, "resolved_snapshot_dir", lambda *a, **k: None
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


class TestStdoutIsReservedForNdjson:
    def test_respond_writes_to_the_captured_real_handle_not_current_stdout(self, monkeypatch):
        import io

        runner = _load_runner("qwen3tts")
        real = io.StringIO()
        monkeypatch.setattr(runner, "_REAL_STDOUT", real)
        monkeypatch.setattr(sys, "stdout", io.StringIO())

        runner._respond({"request_id": "r1", "status": "OK"})

        assert '"request_id": "r1"' in real.getvalue()
        assert sys.stdout.getvalue() == ""

    def test_main_points_stdout_at_stderr_for_the_request_loop(self, monkeypatch):
        import io

        runner = _load_runner("qwen3tts")
        monkeypatch.setattr(sys, "stdin", io.StringIO(""))
        monkeypatch.setattr(sys, "stdout", io.StringIO())
        monkeypatch.setattr(runner, "_REAL_STDOUT", sys.stdout)
        original_stderr = sys.stderr

        runner.main()

        assert sys.stdout is original_stderr


class _FakeQwen3TTSModel:
    load_calls = []

    def __init__(self):
        self.generate_voice_clone_calls = []
        self._sr = 24000

    @classmethod
    def from_pretrained(cls, model_name, device_map=None, dtype=None):
        instance = cls()
        cls.load_calls.append({"model_name": model_name, "device_map": device_map, "dtype": dtype})
        return instance

    def generate_voice_clone(self, **kwargs):
        self.generate_voice_clone_calls.append(kwargs)
        import numpy as np

        return [np.zeros(10, dtype="float32")], self._sr


def _patch_qwen_tts_module(monkeypatch, fake_class=_FakeQwen3TTSModel):
    fake_class.load_calls = []
    fake_module = ModuleType("qwen_tts")
    fake_module.Qwen3TTSModel = fake_class
    monkeypatch.setitem(sys.modules, "qwen_tts", fake_module)


class TestLoadModel:
    def test_default_model_cpu_device_uses_float32(self, monkeypatch):
        runner = _load_runner("qwen3tts")
        _patch_qwen_tts_module(monkeypatch)

        runner._load_model({})

        call = _FakeQwen3TTSModel.load_calls[0]
        assert call["model_name"] == "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
        assert call["device_map"] == "cpu"
        assert call["dtype"] is torch.float32

    def test_cuda_device_uses_bfloat16(self, monkeypatch):
        runner = _load_runner("qwen3tts")
        _patch_qwen_tts_module(monkeypatch)

        runner._load_model({"device": "cuda"})

        call = _FakeQwen3TTSModel.load_calls[0]
        assert call["device_map"] == "cuda"
        assert call["dtype"] is torch.bfloat16

    def test_model_name_raw_override(self, monkeypatch):
        runner = _load_runner("qwen3tts")
        _patch_qwen_tts_module(monkeypatch)

        runner._load_model({"model_name": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"})

        assert _FakeQwen3TTSModel.load_calls[0]["model_name"] == "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"

    def test_model_registry_id_resolves_to_the_hf_repo(self, monkeypatch):
        runner = _load_runner("qwen3tts")
        _patch_qwen_tts_module(monkeypatch)

        runner._load_model({"model": "qwen3-tts-1.7b-customvoice"})

        assert _FakeQwen3TTSModel.load_calls[0]["model_name"] == "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"

    def test_get_model_only_loads_once(self, monkeypatch):
        runner = _load_runner("qwen3tts")
        _patch_qwen_tts_module(monkeypatch)

        first = runner._get_model({})
        second = runner._get_model({})

        assert first is second
        assert len(_FakeQwen3TTSModel.load_calls) == 1


class TestResolveLocalModelDir:
    def _spec(self):
        runner = _load_runner("qwen3tts")
        return runner.model_registry.resolve("qwen3tts", "qwen3-tts-1.7b-base")

    def _make_complete_dir(self, path):
        (path / "config.json").write_text("{}", encoding="utf-8")
        (path / "model.safetensors").write_bytes(b"x")
        (path / "speech_tokenizer").mkdir()
        (path / "speech_tokenizer" / "config.json").write_text("{}", encoding="utf-8")

    def test_complete_local_dir_is_returned(self, monkeypatch, tmp_path):
        runner = _load_runner("qwen3tts")
        self._make_complete_dir(tmp_path)
        monkeypatch.setattr(
            runner.model_registry, "resolved_snapshot_dir", lambda _p: tmp_path
        )

        assert runner._resolve_local_model_dir(self._spec()) == tmp_path

    def test_missing_speech_tokenizer_falls_back_to_none(self, monkeypatch, tmp_path):
        runner = _load_runner("qwen3tts")
        (tmp_path / "config.json").write_text("{}", encoding="utf-8")
        (tmp_path / "model.safetensors").write_bytes(b"x")
        monkeypatch.setattr(
            runner.model_registry, "resolved_snapshot_dir", lambda _p: tmp_path
        )

        assert runner._resolve_local_model_dir(self._spec()) is None

    def test_no_snapshot_dir_returns_none(self, monkeypatch):
        runner = _load_runner("qwen3tts")
        monkeypatch.setattr(
            runner.model_registry, "resolved_snapshot_dir", lambda _p: None
        )

        assert runner._resolve_local_model_dir(self._spec()) is None

    def test_none_spec_returns_none(self):
        runner = _load_runner("qwen3tts")

        assert runner._resolve_local_model_dir(None) is None


class TestLoadModelLocalDir:
    def test_local_dir_is_passed_to_from_pretrained_instead_of_repo_id(self, monkeypatch, tmp_path):
        runner = _load_runner("qwen3tts")
        _patch_qwen_tts_module(monkeypatch)
        monkeypatch.setattr(runner, "_resolve_local_model_dir", lambda _spec: tmp_path)

        runner._load_model({})

        assert _FakeQwen3TTSModel.load_calls[0]["model_name"] == str(tmp_path)

    def test_offline_and_no_local_copy_raises_actionable_error(self, monkeypatch):
        runner = _load_runner("qwen3tts")
        _patch_qwen_tts_module(monkeypatch)
        monkeypatch.setattr(runner, "_resolve_local_model_dir", lambda _spec: None)
        monkeypatch.setattr(runner, "apply_offline_mode", lambda log=None: True)

        with pytest.raises(RuntimeError, match="offline: Qwen3-TTS has no local copy"):
            runner._load_model({})

    def test_raw_model_name_override_still_wins_over_local_dir(self, monkeypatch, tmp_path):
        runner = _load_runner("qwen3tts")
        _patch_qwen_tts_module(monkeypatch)
        monkeypatch.setattr(runner, "_resolve_local_model_dir", lambda _spec: tmp_path)

        runner._load_model({"model_name": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"})

        assert (
            _FakeQwen3TTSModel.load_calls[0]["model_name"]
            == "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
        )


class TestSynthesizeSpeech:
    def test_missing_voice_path_raises_before_touching_the_model(self):
        runner = _load_runner("qwen3tts")

        with pytest.raises(RuntimeError, match="requires voice_path"):
            runner._synthesize_speech("hello", None, "some ref text")

    def test_missing_ref_text_raises_before_touching_the_model(self):
        runner = _load_runner("qwen3tts")

        with pytest.raises(RuntimeError, match="requires ref_text"):
            runner._synthesize_speech("hello", "/some/voice.wav", None)

    def test_empty_ref_text_also_raises(self):
        runner = _load_runner("qwen3tts")

        with pytest.raises(RuntimeError, match="requires ref_text"):
            runner._synthesize_speech("hello", "/some/voice.wav", "")

    def test_returns_tensor_and_real_sample_rate(self, monkeypatch):
        runner = _load_runner("qwen3tts")
        fake_model = _FakeQwen3TTSModel()
        monkeypatch.setattr(runner, "_get_model", lambda config: fake_model)

        wav_tensor, sr = runner._synthesize_speech("hello", "/voices/a.wav", "reference text")

        assert wav_tensor.shape == (1, 10)
        assert sr == 24000
        call = fake_model.generate_voice_clone_calls[0]
        assert call["ref_audio"] == "/voices/a.wav"
        assert call["ref_text"] == "reference text"
        assert call["text"] == "hello"

    def test_defaults_to_english_when_language_not_configured(self, monkeypatch):
        runner = _load_runner("qwen3tts")
        runner._config = {}
        fake_model = _FakeQwen3TTSModel()
        monkeypatch.setattr(runner, "_get_model", lambda config: fake_model)

        runner._synthesize_speech("hello", "/voices/a.wav", "reference text")

        assert fake_model.generate_voice_clone_calls[0]["language"] == "English"

    def test_language_override(self, monkeypatch):
        runner = _load_runner("qwen3tts")
        runner._config = {"language": "Chinese"}
        fake_model = _FakeQwen3TTSModel()
        monkeypatch.setattr(runner, "_get_model", lambda config: fake_model)

        runner._synthesize_speech("你好", "/voices/a.wav", "参考文本")

        assert fake_model.generate_voice_clone_calls[0]["language"] == "Chinese"

    def test_iso_639_1_code_from_tts_config_is_mapped_to_the_full_name(self, monkeypatch):
        runner = _load_runner("qwen3tts")
        runner._config = {"language": "zh"}
        fake_model = _FakeQwen3TTSModel()
        monkeypatch.setattr(runner, "_get_model", lambda config: fake_model)

        runner._synthesize_speech("你好", "/voices/a.wav", "参考文本")

        assert fake_model.generate_voice_clone_calls[0]["language"] == "Chinese"


class TestHandleInit:
    def test_success_returns_ok_with_placeholder_samplerate(self, monkeypatch):
        runner = _load_runner("qwen3tts")
        monkeypatch.setattr(runner, "_load_model", lambda config: _FakeQwen3TTSModel())

        response = runner.handle_init({"request_id": "r1", "config": {"device": "cpu"}})

        assert response == {
            "request_id": "r1",
            "status": "OK",
            "supported_features": ["zero_shot"],
            "samplerate": 24000,
        }

    def test_model_load_failure_returns_error_without_raising(self, monkeypatch):
        runner = _load_runner("qwen3tts")

        def _boom(config):
            raise RuntimeError("checkpoint download failed")

        monkeypatch.setattr(runner, "_load_model", _boom)

        response = runner.handle_init({"request_id": "r2", "config": {}})

        assert response["request_id"] == "r2"
        assert response["status"] == "ERROR"
        assert "checkpoint download failed" in response["error"]


class TestHandleSynthesize:
    def test_success_writes_wav_and_reports_real_sample_rate(self, monkeypatch, tmp_path):
        runner = _load_runner("qwen3tts")
        speech = torch.zeros(1, 4800)
        monkeypatch.setattr(runner, "_synthesize_speech", lambda text, voice_path, ref_text: (speech, 48000))
        output_file = tmp_path / "out.wav"

        response = runner.handle_synthesize({
            "request_id": "r3",
            "text": "hello",
            "voice_path": "/voices/a.wav",
            "ref_text": "reference",
            "leading_silence_ms": 0,
            "output_file": str(output_file),
        })

        assert response["status"] == "OK"
        assert response["samplerate"] == 48000
        assert response["duration_seconds"] == pytest.approx(0.1)
        assert output_file.exists()

    def test_leading_silence_uses_the_real_sample_rate_learned_from_this_call(self, monkeypatch, tmp_path):
        runner = _load_runner("qwen3tts")
        assert runner._sample_rate == runner.DEFAULT_SAMPLE_RATE == 24000
        speech = torch.zeros(1, 4800)
        monkeypatch.setattr(runner, "_synthesize_speech", lambda text, voice_path, ref_text: (speech, 48000))
        output_file = tmp_path / "out.wav"

        response = runner.handle_synthesize({
            "request_id": "r4",
            "text": "hello",
            "voice_path": "/voices/a.wav",
            "ref_text": "reference",
            "leading_silence_ms": 500,
            "output_file": str(output_file),
        })

        assert response["duration_seconds"] == pytest.approx(0.6)
        assert response["samplerate"] == 48000

    def test_empty_text_and_no_silence_still_writes_a_valid_empty_wav(self, tmp_path):
        runner = _load_runner("qwen3tts")
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
        runner = _load_runner("qwen3tts")

        def _boom(text, voice_path, ref_text):
            raise RuntimeError("requires ref_text")

        monkeypatch.setattr(runner, "_synthesize_speech", _boom)
        output_file = tmp_path / "out.wav"

        response = runner.handle_synthesize({
            "request_id": "r6",
            "text": "hello",
            "voice_path": "/voices/a.wav",
            "ref_text": None,
            "leading_silence_ms": 0,
            "output_file": str(output_file),
        })

        assert response["request_id"] == "r6"
        assert response["status"] == "ERROR"
        assert "requires ref_text" in response["error"]

    def test_synthesis_failure_releases_accelerator_memory(self, monkeypatch, tmp_path):
        runner = _load_runner("qwen3tts")
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
        runner = _load_runner("qwen3tts")
        calls = []
        monkeypatch.setattr(runner, "release_accelerator_memory", lambda log=None, reason="": calls.append(reason))
        speech = torch.zeros(1, 4800)
        monkeypatch.setattr(runner, "_synthesize_speech", lambda text, voice_path, ref_text: (speech, 48000))
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
