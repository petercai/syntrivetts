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
def _reset_module_state():
    runner = _load_runner("xtts")
    runner._model = None
    runner._config = {}
    runner._latent_cache = {}
    yield
    runner._model = None
    runner._config = {}
    runner._latent_cache = {}


class _FakeXttsConfig:
    def __init__(self):
        self.loaded_json_path = None

    def load_json(self, path):
        self.loaded_json_path = path


class _FakeXtts:
    def __init__(self):
        self.load_checkpoint_calls = []
        self.device_calls = []

    @classmethod
    def init_from_config(cls, config):
        instance = cls()
        instance.config = config
        return instance

    def load_checkpoint(self, config, **kwargs):
        self.load_checkpoint_calls.append(kwargs)

    def to(self, device):
        self.device_calls.append(("to", device))

    def cuda(self):
        self.device_calls.append(("cuda", None))


def _patch_model_loading(monkeypatch):
    calls = []

    def fake_download(repo_id, filename, cache_dir=None, **kwargs):
        calls.append((repo_id, filename, cache_dir))
        return f"/fake/{filename}"

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)
    monkeypatch.setattr("TTS.tts.configs.xtts_config.XttsConfig", _FakeXttsConfig)
    monkeypatch.setattr("TTS.tts.models.xtts.Xtts", _FakeXtts)
    return calls


class TestLoadModel:
    def test_downloads_the_three_stock_files_from_the_default_repo(self, monkeypatch):
        runner = _load_runner("xtts")
        calls = _patch_model_loading(monkeypatch)

        model = runner._load_model({"device": "cpu"})

        assert calls == [
            ("coqui/XTTS-v2", "config.json", str(runner._DEFAULT_CACHE_DIR)),
            ("coqui/XTTS-v2", "model.pth", str(runner._DEFAULT_CACHE_DIR)),
            ("coqui/XTTS-v2", "vocab.json", str(runner._DEFAULT_CACHE_DIR)),
        ]
        assert isinstance(model, _FakeXtts)
        assert model.load_checkpoint_calls == [
            {
                "checkpoint_path": "/fake/model.pth",
                "vocab_path": "/fake/vocab.json",
                "use_deepspeed": False,
                "eval": True,
            }
        ]

    def test_model_repo_and_model_sub_override_the_default(self, monkeypatch):
        runner = _load_runner("xtts")
        calls = _patch_model_loading(monkeypatch)

        runner._load_model({
            "model_repo": "drewThomasson/fineTunedTTSModels",
            "model_sub": "xtts-v2/eng/AiExplained/",
        })

        filenames = [c[1] for c in calls]
        repos = {c[0] for c in calls}
        assert repos == {"drewThomasson/fineTunedTTSModels"}
        assert filenames == [
            "xtts-v2/eng/AiExplained/config.json",
            "xtts-v2/eng/AiExplained/model.pth",
            "xtts-v2/eng/AiExplained/vocab.json",
        ]

    def test_cpu_device_default_calls_to_not_cuda(self, monkeypatch):
        runner = _load_runner("xtts")
        _patch_model_loading(monkeypatch)

        model = runner._load_model({})

        assert model.device_calls == [("to", "cpu")]

    def test_cuda_device_calls_cuda_not_to(self, monkeypatch):
        runner = _load_runner("xtts")
        _patch_model_loading(monkeypatch)

        model = runner._load_model({"device": "cuda"})

        assert model.device_calls == [("cuda", None)]

    def test_use_deepspeed_flag_is_forwarded(self, monkeypatch):
        runner = _load_runner("xtts")
        _patch_model_loading(monkeypatch)

        model = runner._load_model({"use_deepspeed": True})

        assert model.load_checkpoint_calls[0]["use_deepspeed"] is True


class TestGetModelCaching:
    def test_get_model_only_loads_once(self, monkeypatch):
        runner = _load_runner("xtts")
        calls = _patch_model_loading(monkeypatch)

        first = runner._get_model({"device": "cpu"})
        second = runner._get_model({"device": "cpu"})

        assert first is second
        assert len(calls) == 3


class TestConditioningLatentsCache:
    class _FakeModel:
        def __init__(self):
            self.calls = 0

        def get_conditioning_latents(self, audio_path):
            self.calls += 1
            return (f"latent-{self.calls}", f"embedding-{self.calls}")

    def test_same_voice_path_is_cached_different_voice_path_is_not(self):
        runner = _load_runner("xtts")
        model = self._FakeModel()

        first = runner._get_conditioning_latents(model, "/voices/a.wav")
        second = runner._get_conditioning_latents(model, "/voices/a.wav")
        third = runner._get_conditioning_latents(model, "/voices/b.wav")

        assert first == second
        assert third != first
        assert model.calls == 2


class TestSynthesizeSpeech:
    class _FakeModel:
        def __init__(self, wav=None):
            self.inference_calls = []
            self._wav = wav if wav is not None else [0.0] * 10

        def get_conditioning_latents(self, audio_path):
            return ("latent", "embedding")

        def inference(self, **kwargs):
            self.inference_calls.append(kwargs)
            return {"wav": self._wav}

    def test_missing_voice_path_raises_before_touching_the_model(self):
        runner = _load_runner("xtts")

        with pytest.raises(RuntimeError, match="requires voice_path"):
            runner._synthesize_speech("hello", None)

    def test_empty_voice_path_also_raises(self):
        runner = _load_runner("xtts")

        with pytest.raises(RuntimeError, match="requires voice_path"):
            runner._synthesize_speech("hello", "")

    def test_only_config_keys_actually_present_are_forwarded(self, monkeypatch):
        runner = _load_runner("xtts")
        runner._config = {"language": "zh", "temperature": 0.3}
        fake_model = self._FakeModel()
        monkeypatch.setattr(runner, "_get_model", lambda config: fake_model)

        wav_tensor = runner._synthesize_speech("你好", "/voices/a.wav")

        call = fake_model.inference_calls[0]
        assert call["language"] == "zh"
        assert call["temperature"] == 0.3
        assert "top_k" not in call
        assert "speed" not in call
        assert wav_tensor.shape == (1, 10)

    def test_defaults_to_english_when_language_not_configured(self, monkeypatch):
        runner = _load_runner("xtts")
        runner._config = {}
        fake_model = self._FakeModel()
        monkeypatch.setattr(runner, "_get_model", lambda config: fake_model)

        runner._synthesize_speech("hello", "/voices/a.wav")

        assert fake_model.inference_calls[0]["language"] == "en"

    def test_raises_when_inference_returns_no_waveform(self, monkeypatch):
        runner = _load_runner("xtts")

        class _NoWavModel:
            def get_conditioning_latents(self, audio_path):
                return ("l", "e")

            def inference(self, **kwargs):
                return {}

        monkeypatch.setattr(runner, "_get_model", lambda config: _NoWavModel())

        with pytest.raises(RuntimeError, match="no waveform"):
            runner._synthesize_speech("hello", "/voices/a.wav")


class TestReplacePeriodsForXtts:
    def test_replaces_every_period_with_an_em_dash(self):
        runner = _load_runner("xtts")
        assert runner._replace_periods_for_xtts("Hello. World.") == "Hello — World —"

    def test_noop_when_no_period_present(self):
        runner = _load_runner("xtts")
        assert runner._replace_periods_for_xtts("PART one") == "PART one"

    def test_noop_for_empty_text(self):
        runner = _load_runner("xtts")
        assert runner._replace_periods_for_xtts("") == ""


class TestEstimateExpectedSeconds:
    def test_scales_with_text_length(self):
        runner = _load_runner("xtts")
        short = runner._estimate_expected_seconds("hi")
        long = runner._estimate_expected_seconds("hi " * 50)
        assert long > short

    def test_floors_at_minimum_for_very_short_text(self):
        runner = _load_runner("xtts")
        assert runner._estimate_expected_seconds("hi") == runner._MIN_EXPECTED_SECONDS


class TestBadcaseDetectionIsLogOnly:
    class _FixedWavModel:
        def __init__(self, wav):
            self._wav = wav
            self.call_count = 0

        def get_conditioning_latents(self, audio_path):
            return ("latent", "embedding")

        def inference(self, **kwargs):
            self.call_count += 1
            return {"wav": self._wav}

    def test_a_bad_ratio_is_accepted_not_retried_or_raised(self, monkeypatch):
        runner = _load_runner("xtts")
        bad_wav = [0.0] * (5 * 24000)
        model = self._FixedWavModel(bad_wav)
        monkeypatch.setattr(runner, "_get_model", lambda config: model)

        wav_tensor = runner._synthesize_speech("PART one", "/voices/a.wav")

        assert model.call_count == 1
        assert wav_tensor.shape == (1, len(bad_wav))

    def test_a_good_ratio_produces_no_detection_log_but_still_returns_audio(self, monkeypatch):
        runner = _load_runner("xtts")
        good_wav = [0.0] * int(0.1 * 24000)
        model = self._FixedWavModel(good_wav)
        monkeypatch.setattr(runner, "_get_model", lambda config: model)

        wav_tensor = runner._synthesize_speech("PART one", "/voices/a.wav")

        assert model.call_count == 1
        assert wav_tensor.shape == (1, len(good_wav))


class TestHandleInit:
    def test_success_returns_ok_with_features_and_samplerate(self, monkeypatch):
        runner = _load_runner("xtts")
        monkeypatch.setattr(runner, "_load_model", lambda config: object())

        response = runner.handle_init({"request_id": "r1", "config": {"device": "cpu"}})

        assert response == {
            "request_id": "r1",
            "status": "OK",
            "supported_features": ["zero_shot"],
            "samplerate": 24000,
        }
        assert runner._config == {"device": "cpu"}

    def test_model_load_failure_returns_error_without_raising(self, monkeypatch):
        runner = _load_runner("xtts")

        def _boom(config):
            raise RuntimeError("checkpoint missing")

        monkeypatch.setattr(runner, "_load_model", _boom)

        response = runner.handle_init({"request_id": "r2", "config": {}})

        assert response["request_id"] == "r2"
        assert response["status"] == "ERROR"
        assert "checkpoint missing" in response["error"]


class TestHandleSynthesize:
    def test_success_writes_wav_and_reports_duration(self, monkeypatch, tmp_path):
        runner = _load_runner("xtts")
        speech = torch.zeros(1, 2400)
        monkeypatch.setattr(runner, "_synthesize_speech", lambda text, voice_path: speech)
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
        assert response["samplerate"] == 24000
        assert response["duration_seconds"] == pytest.approx(0.1)
        assert output_file.exists()

    def test_leading_silence_is_prepended(self, monkeypatch, tmp_path):
        runner = _load_runner("xtts")
        speech = torch.zeros(1, 2400)
        monkeypatch.setattr(runner, "_synthesize_speech", lambda text, voice_path: speech)
        output_file = tmp_path / "out.wav"

        response = runner.handle_synthesize({
            "request_id": "r4",
            "text": "hello",
            "voice_path": "/voices/a.wav",
            "leading_silence_ms": 500,
            "output_file": str(output_file),
        })

        assert response["duration_seconds"] == pytest.approx(0.6)

    def test_empty_text_and_no_silence_still_writes_a_valid_empty_wav(self, tmp_path):
        runner = _load_runner("xtts")
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
        runner = _load_runner("xtts")

        def _boom(text, voice_path):
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
        runner = _load_runner("xtts")
        calls = []
        monkeypatch.setattr(runner, "release_accelerator_memory", lambda log=None, reason="": calls.append(reason))

        def _boom(text, voice_path):
            raise RuntimeError("CUDA out of memory")

        monkeypatch.setattr(runner, "_synthesize_speech", _boom)
        output_file = tmp_path / "out.wav"

        response = runner.handle_synthesize({
            "request_id": "r7",
            "text": "hello",
            "voice_path": "/voices/a.wav",
            "leading_silence_ms": 0,
            "output_file": str(output_file),
        })

        assert response["status"] == "ERROR"
        assert calls == ["synthesize_failed"]

    def test_success_releases_accelerator_memory(self, monkeypatch, tmp_path):
        runner = _load_runner("xtts")
        calls = []
        monkeypatch.setattr(runner, "release_accelerator_memory", lambda log=None, reason="": calls.append(reason))
        speech = torch.zeros(1, 2400)
        monkeypatch.setattr(runner, "_synthesize_speech", lambda text, voice_path: speech)
        output_file = tmp_path / "out.wav"

        response = runner.handle_synthesize({
            "request_id": "r9",
            "text": "hello",
            "voice_path": "/voices/a.wav",
            "leading_silence_ms": 0,
            "output_file": str(output_file),
        })

        assert response["status"] == "OK"
        assert calls == ["synthesize_ok"]
