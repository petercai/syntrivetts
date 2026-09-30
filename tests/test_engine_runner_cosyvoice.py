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
    runner = _load_runner("cosyvoice")
    runner._model = None
    runner._config = {}
    yield
    runner._model = None
    runner._config = {}


class _FakeCosyVoiceModel:
    def __init__(self, model_dir=None, available_spks=None):
        self.model_dir = model_dir
        self.sample_rate = 22050
        self._available_spks = available_spks if available_spks is not None else []
        self.zero_shot_calls: list = []
        self.sft_calls: list = []

    def list_available_spks(self):
        return list(self._available_spks)

    def inference_zero_shot(self, text, ref_text, voice_path, stream=False, speed=1.0):
        self.zero_shot_calls.append({"text": text, "ref_text": ref_text, "voice_path": voice_path, "stream": stream, "speed": speed})
        yield {"tts_speech": torch.zeros(1, 8)}

    def inference_sft(self, text, spk, stream=False, speed=1.0):
        self.sft_calls.append({"text": text, "spk": spk, "stream": stream, "speed": speed})
        yield {"tts_speech": torch.zeros(1, 6)}


def _patch_automodel(monkeypatch, fake_factory):
    pkg = ModuleType("cosyvoice")
    cli = ModuleType("cosyvoice.cli")
    mod = ModuleType("cosyvoice.cli.cosyvoice")
    mod.AutoModel = fake_factory
    pkg.cli = cli
    cli.cosyvoice = mod
    monkeypatch.setitem(sys.modules, "cosyvoice", pkg)
    monkeypatch.setitem(sys.modules, "cosyvoice.cli", cli)
    monkeypatch.setitem(sys.modules, "cosyvoice.cli.cosyvoice", mod)


class TestLoadModel:
    def test_constructs_automodel_with_the_given_model_dir(self, monkeypatch):
        runner = _load_runner("cosyvoice")
        captured = {}

        def fake_automodel(model_dir=None):
            captured["model_dir"] = model_dir
            return _FakeCosyVoiceModel(model_dir=model_dir)

        _patch_automodel(monkeypatch, fake_automodel)

        model = runner._load_model("/some/checkpoint/dir")

        assert captured["model_dir"] == "/some/checkpoint/dir"
        assert isinstance(model, _FakeCosyVoiceModel)


class TestGetModel:
    def test_default_model_dir_is_the_registry_default_checkpoint(self, monkeypatch):
        runner = _load_runner("cosyvoice")
        _patch_automodel(monkeypatch, lambda model_dir=None: _FakeCosyVoiceModel(model_dir=model_dir))

        model = runner._get_model({})

        assert model.model_dir == str(runner._DEFAULT_MODEL_DIR)
        assert model.model_dir.replace("\\", "/").endswith("models/tts/Fun-CosyVoice3-0.5B")

    def test_config_model_dir_overrides_everything(self, monkeypatch):
        runner = _load_runner("cosyvoice")
        _patch_automodel(monkeypatch, lambda model_dir=None: _FakeCosyVoiceModel(model_dir=model_dir))

        model = runner._get_model({"model_dir": "/custom/ckpt"})

        assert model.model_dir == "/custom/ckpt"

    def test_registry_model_id_selects_a_local_checkpoint_dir(self, monkeypatch):
        runner = _load_runner("cosyvoice")
        _patch_automodel(monkeypatch, lambda model_dir=None: _FakeCosyVoiceModel(model_dir=model_dir))

        model = runner._get_model({"model": "cosyvoice-300m-sft"})

        assert model.model_dir.replace("\\", "/").endswith("models/tts/CosyVoice-300M-SFT")

    def test_only_loads_once(self, monkeypatch):
        runner = _load_runner("cosyvoice")
        load_calls = []
        _patch_automodel(
            monkeypatch,
            lambda model_dir=None: (load_calls.append(1), _FakeCosyVoiceModel(model_dir=model_dir))[1],
        )

        first = runner._get_model({})
        second = runner._get_model({})

        assert first is second
        assert len(load_calls) == 1


class TestSynthesizeSpeech:
    def test_zero_shot_without_ref_text_raises_before_touching_the_model(self):
        runner = _load_runner("cosyvoice")

        with pytest.raises(RuntimeError, match="requires ref_text"):
            runner._synthesize_speech("hello", "/voices/a.wav", None)

    def test_voice_path_present_runs_zero_shot_cloning(self, monkeypatch):
        runner = _load_runner("cosyvoice")
        fake_model = _FakeCosyVoiceModel()
        monkeypatch.setattr(runner, "_get_model", lambda config: fake_model)

        wav = runner._synthesize_speech("你好", "/voices/a.wav", "参考文本")

        assert fake_model.zero_shot_calls == [
            {"text": "你好", "ref_text": "参考文本", "voice_path": "/voices/a.wav", "stream": False, "speed": 1.0}
        ]
        assert fake_model.sft_calls == []
        assert wav.shape == (1, 8)

    def test_no_voice_path_falls_back_to_sft_with_the_first_available_speaker(self, monkeypatch):
        runner = _load_runner("cosyvoice")
        fake_model = _FakeCosyVoiceModel(available_spks=["中文女", "中文男"])
        monkeypatch.setattr(runner, "_get_model", lambda config: fake_model)

        wav = runner._synthesize_speech("hello", None, None)

        assert fake_model.sft_calls == [{"text": "hello", "spk": "中文女", "stream": False, "speed": 1.0}]
        assert fake_model.zero_shot_calls == []
        assert wav.shape == (1, 6)

    def test_no_voice_path_and_no_sft_speakers_fails_loudly(self, monkeypatch):
        runner = _load_runner("cosyvoice")
        fake_model = _FakeCosyVoiceModel(available_spks=[])
        monkeypatch.setattr(runner, "_get_model", lambda config: fake_model)

        with pytest.raises(RuntimeError, match="no SFT speakers"):
            runner._synthesize_speech("hello", None, None)

    def test_config_sft_speaker_id_pins_the_preset_speaker(self, monkeypatch):
        runner = _load_runner("cosyvoice")
        runner._config = {"sft_speaker_id": "中文男"}
        fake_model = _FakeCosyVoiceModel(available_spks=["中文女", "中文男", "粤语女"])
        monkeypatch.setattr(runner, "_get_model", lambda config: fake_model)

        runner._synthesize_speech("hello", None, None)

        assert fake_model.sft_calls == [{"text": "hello", "spk": "中文男", "stream": False, "speed": 1.0}]

    def test_unknown_config_sft_speaker_id_raises_with_the_available_list(self, monkeypatch):
        runner = _load_runner("cosyvoice")
        runner._config = {"sft_speaker_id": "NoSuchSpeaker"}
        fake_model = _FakeCosyVoiceModel(available_spks=["中文女", "中文男"])
        monkeypatch.setattr(runner, "_get_model", lambda config: fake_model)

        with pytest.raises(RuntimeError, match="sft_speaker_id=.*not a speaker"):
            runner._synthesize_speech("hello", None, None)

    def test_synthesize_forwards_init_config_to_get_model(self, monkeypatch):
        runner = _load_runner("cosyvoice")
        runner._config = {"model_dir": "/custom/sft/ckpt"}
        seen_configs = []
        fake_model = _FakeCosyVoiceModel(available_spks=["中文女"])
        monkeypatch.setattr(
            runner, "_get_model", lambda config: (seen_configs.append(config), fake_model)[1]
        )

        runner._synthesize_speech("hello", None, None)

        assert seen_configs == [{"model_dir": "/custom/sft/ckpt"}]

    def test_configured_speed_is_passed_to_zero_shot_inference(self, monkeypatch):
        runner = _load_runner("cosyvoice")
        runner._config = {"speed": 1.8}
        fake_model = _FakeCosyVoiceModel()
        monkeypatch.setattr(runner, "_get_model", lambda config: fake_model)

        runner._synthesize_speech("你好", "/voices/a.wav", "参考文本")

        assert fake_model.zero_shot_calls == [
            {"text": "你好", "ref_text": "参考文本", "voice_path": "/voices/a.wav", "stream": False, "speed": 1.8}
        ]

    def test_configured_speed_is_passed_to_sft_inference(self, monkeypatch):
        runner = _load_runner("cosyvoice")
        runner._config = {"speed": 0.75}
        fake_model = _FakeCosyVoiceModel(available_spks=["中文女"])
        monkeypatch.setattr(runner, "_get_model", lambda config: fake_model)

        runner._synthesize_speech("hello", None, None)

        assert fake_model.sft_calls == [
            {"text": "hello", "spk": "中文女", "stream": False, "speed": 0.75}
        ]

    def test_missing_speed_config_defaults_to_1_0(self, monkeypatch):
        runner = _load_runner("cosyvoice")
        runner._config = {}
        fake_model = _FakeCosyVoiceModel(available_spks=["中文女"])
        monkeypatch.setattr(runner, "_get_model", lambda config: fake_model)

        runner._synthesize_speech("hello", None, None)

        assert fake_model.sft_calls == [
            {"text": "hello", "spk": "中文女", "stream": False, "speed": 1.0}
        ]


class TestHandleInit:
    def test_success_returns_ok_with_features_and_the_real_samplerate(self, monkeypatch):
        runner = _load_runner("cosyvoice")
        _patch_automodel(monkeypatch, lambda model_dir=None: _FakeCosyVoiceModel(model_dir=model_dir))

        response = runner.handle_init({"request_id": "r1", "config": {}})

        assert response == {
            "request_id": "r1",
            "status": "OK",
            "supported_features": ["zero_shot", "sft"],
            "samplerate": 22050,
        }

    def test_model_load_failure_returns_error_without_raising(self, monkeypatch):
        runner = _load_runner("cosyvoice")

        def _boom(model_dir):
            raise RuntimeError("checkpoint dir not found")

        monkeypatch.setattr(runner, "_load_model", _boom)

        response = runner.handle_init({"request_id": "r2", "config": {}})

        assert response["request_id"] == "r2"
        assert response["status"] == "ERROR"
        assert "checkpoint dir not found" in response["error"]


class TestHandleSynthesize:
    def test_success_writes_wav_and_reports_duration_at_the_model_rate(self, monkeypatch, tmp_path):
        runner = _load_runner("cosyvoice")
        fake_model = _FakeCosyVoiceModel()
        monkeypatch.setattr(runner, "_get_model", lambda config: fake_model)
        speech = torch.zeros(1, 2205)
        monkeypatch.setattr(runner, "_synthesize_speech", lambda text, voice_path, ref_text: speech)
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
        assert response["samplerate"] == 22050
        assert response["duration_seconds"] == pytest.approx(0.1)
        assert output_file.exists()

    def test_leading_silence_is_prepended_at_the_model_rate(self, monkeypatch, tmp_path):
        runner = _load_runner("cosyvoice")
        fake_model = _FakeCosyVoiceModel()
        monkeypatch.setattr(runner, "_get_model", lambda config: fake_model)
        speech = torch.zeros(1, 2205)
        monkeypatch.setattr(runner, "_synthesize_speech", lambda text, voice_path, ref_text: speech)
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

    def test_empty_text_and_no_silence_still_writes_a_valid_empty_wav(self, monkeypatch, tmp_path):
        runner = _load_runner("cosyvoice")
        fake_model = _FakeCosyVoiceModel()
        monkeypatch.setattr(runner, "_get_model", lambda config: fake_model)
        output_file = tmp_path / "out.wav"

        response = runner.handle_synthesize({
            "request_id": "r5",
            "text": "",
            "voice_path": None,
            "ref_text": None,
            "leading_silence_ms": 0,
            "output_file": str(output_file),
        })

        assert response["status"] == "OK"
        assert response["duration_seconds"] == 0.0
        assert output_file.exists()

    def test_synthesis_failure_returns_error_without_crashing_the_worker(self, monkeypatch, tmp_path):
        runner = _load_runner("cosyvoice")
        fake_model = _FakeCosyVoiceModel()
        monkeypatch.setattr(runner, "_get_model", lambda config: fake_model)

        def _boom(text, voice_path, ref_text):
            raise RuntimeError("cosyvoice zero-shot cloning requires ref_text")

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
        runner = _load_runner("cosyvoice")
        fake_model = _FakeCosyVoiceModel()
        monkeypatch.setattr(runner, "_get_model", lambda config: fake_model)
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
        runner = _load_runner("cosyvoice")
        fake_model = _FakeCosyVoiceModel()
        monkeypatch.setattr(runner, "_get_model", lambda config: fake_model)
        calls = []
        monkeypatch.setattr(runner, "release_accelerator_memory", lambda log=None, reason="": calls.append(reason))
        speech = torch.zeros(1, 2205)
        monkeypatch.setattr(runner, "_synthesize_speech", lambda text, voice_path, ref_text: speech)
        output_file = tmp_path / "out.wav"

        response = runner.handle_synthesize({
            "request_id": "r9",
            "text": "hello",
            "voice_path": "/voices/a.wav",
            "ref_text": "reference",
            "leading_silence_ms": 0,
            "output_file": str(output_file),
        })

        assert response["status"] == "OK"
        assert calls == ["synthesize_ok"]
