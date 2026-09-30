from __future__ import annotations

import importlib.util
import sys
import wave
from pathlib import Path
from types import ModuleType
from typing import cast

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


_RUNNER = _load_runner("f5tts")
_REAL_RESOLVE_VOCODER = _RUNNER._resolve_local_vocoder
_REAL_RESOLVE_CHECKPOINT = _RUNNER._resolve_local_checkpoint


@pytest.fixture(autouse=True)
def _reset_module_state(monkeypatch):
    runner = _load_runner("f5tts")
    monkeypatch.setattr(
        runner.model_registry, "apply_offline_env_if_cached", lambda *a, **k: False
    )
    monkeypatch.setattr(runner, "_resolve_local_vocoder", lambda _spec: None)
    monkeypatch.setattr(runner, "_resolve_local_checkpoint", lambda _spec, _name: None)
    for var in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        monkeypatch.delenv(var, raising=False)
    runner._model = None
    runner._config = {}
    runner._sample_rate = runner.DEFAULT_SAMPLE_RATE
    yield
    runner._model = None
    runner._config = {}
    runner._sample_rate = runner.DEFAULT_SAMPLE_RATE


class _FakeF5TTS:
    def __init__(self, model=None, device=None, hf_cache_dir=None,
                 vocoder_local_path=None, ckpt_file=None, **kwargs):
        import numpy as np

        self.model = model
        self.device = device
        self.hf_cache_dir = hf_cache_dir
        self.vocoder_local_path = vocoder_local_path
        self.ckpt_file = ckpt_file
        self.extra_kwargs = kwargs
        self.target_sample_rate = 24000
        self.infer_calls = []
        self._wav = np.zeros(10, dtype="float32")

    def infer(self, **kwargs):
        self.infer_calls.append(kwargs)
        return self._wav, self.target_sample_rate, None


def _patch_f5tts_class(monkeypatch, fake_class=_FakeF5TTS):
    fake_module = ModuleType("f5_tts.api")
    fake_module.F5TTS = fake_class
    monkeypatch.setitem(sys.modules, "f5_tts.api", fake_module)
    fake_pkg = ModuleType("f5_tts")
    monkeypatch.setitem(sys.modules, "f5_tts", fake_pkg)


class TestStdoutIsReservedForNdjson:
    def test_respond_writes_to_the_captured_real_handle_not_current_stdout(self, monkeypatch):
        import io

        runner = _load_runner("f5tts")
        real = io.StringIO()
        monkeypatch.setattr(runner, "_REAL_STDOUT", real)
        monkeypatch.setattr(sys, "stdout", io.StringIO())

        runner._respond({"request_id": "r1", "status": "OK"})

        assert '"request_id": "r1"' in real.getvalue()
        assert sys.stdout.getvalue() == ""

    def test_main_points_stdout_at_stderr_for_the_request_loop(self, monkeypatch):
        import io

        runner = _load_runner("f5tts")
        monkeypatch.setattr(sys, "stdin", io.StringIO(""))
        monkeypatch.setattr(sys, "stdout", io.StringIO())
        monkeypatch.setattr(runner, "_REAL_STDOUT", sys.stdout)
        original_stderr = sys.stderr

        runner.main()

        assert sys.stdout is original_stderr


class TestLoadModel:
    def test_default_model_and_device(self, monkeypatch):
        runner = _load_runner("f5tts")
        _patch_f5tts_class(monkeypatch)

        model = runner._load_model({})

        assert isinstance(model, _FakeF5TTS)
        assert model.model == "F5TTS_v1_Base"
        assert model.device == "cpu"

    def test_model_and_device_overrides(self, monkeypatch):
        runner = _load_runner("f5tts")
        _patch_f5tts_class(monkeypatch)

        model = runner._load_model({"model_name": "F5TTS_Small", "device": "cuda"})

        assert model.model == "F5TTS_Small"
        assert model.device == "cuda"

    def test_get_model_reads_target_sample_rate_once_loaded(self, monkeypatch):
        runner = _load_runner("f5tts")
        _patch_f5tts_class(monkeypatch)

        runner._get_model({})

        assert runner._sample_rate == 24000

    def test_get_model_only_loads_once(self, monkeypatch):
        runner = _load_runner("f5tts")
        _patch_f5tts_class(monkeypatch)
        load_calls = []
        original = runner._load_model
        monkeypatch.setattr(runner, "_load_model", lambda config: (load_calls.append(1), original(config))[1])

        first = runner._get_model({})
        second = runner._get_model({})

        assert first is second
        assert len(load_calls) == 1


class TestLoadModelLocalPaths:
    def test_resolved_local_paths_flow_into_F5TTS(self, monkeypatch, tmp_path):
        runner = _load_runner("f5tts")
        _patch_f5tts_class(monkeypatch)
        voc = tmp_path / "vocos_snapshot"
        voc.mkdir()
        ckpt = tmp_path / "F5TTS_v1_Base" / "model_1250000.safetensors"
        ckpt.parent.mkdir()
        ckpt.write_bytes(b"x")
        monkeypatch.setattr(runner, "_resolve_local_vocoder", lambda _spec: voc)
        monkeypatch.setattr(runner, "_resolve_local_checkpoint", lambda _spec, _name: ckpt)

        model = runner._load_model({})

        assert model.vocoder_local_path == str(voc)
        assert model.ckpt_file == str(ckpt)

    def test_offline_and_missing_raises_an_actionable_error(self, monkeypatch):
        runner = _load_runner("f5tts")
        _patch_f5tts_class(monkeypatch)
        monkeypatch.setenv("SYNTRIVE_TTS_OFFLINE", "1")

        with pytest.raises(RuntimeError, match=r"tools/model_dl\.py f5tts"):
            runner._load_model({})

    def test_online_and_missing_falls_back_to_the_F5TTS_download(self, monkeypatch):
        runner = _load_runner("f5tts")
        _patch_f5tts_class(monkeypatch)

        model = runner._load_model({})

        assert model.vocoder_local_path is None
        assert model.ckpt_file is None

    def test_real_resolvers_find_files_in_an_hf_cache_layout(self, monkeypatch, tmp_path):
        runner = _load_runner("f5tts")
        monkeypatch.setattr(runner, "_resolve_local_vocoder", _REAL_RESOLVE_VOCODER)
        monkeypatch.setattr(runner, "_resolve_local_checkpoint", _REAL_RESOLVE_CHECKPOINT)
        monkeypatch.setattr(runner.model_registry, "TTS_CACHE_DIR", tmp_path)

        spec = runner.model_registry.resolve("f5tts", "f5tts-v1-base")
        voc_snap = runner.model_registry.extra_repo_dirs(spec)[0] / "snapshots" / "s1"
        voc_snap.mkdir(parents=True)
        (voc_snap / "config.yaml").write_text("x")
        (voc_snap / "pytorch_model.bin").write_bytes(b"x")
        ckpt_dir = spec.local_path() / "snapshots" / "c1" / "F5TTS_v1_Base"
        ckpt_dir.mkdir(parents=True)
        (ckpt_dir / "model_1250000.safetensors").write_bytes(b"x")

        assert runner._resolve_local_vocoder(spec) == voc_snap
        assert runner._resolve_local_checkpoint(spec, "F5TTS_v1_Base") == (
            ckpt_dir / "model_1250000.safetensors"
        )

    def test_real_resolver_rejects_a_broken_snapshot_symlink(self, monkeypatch, tmp_path):
        runner = _load_runner("f5tts")
        monkeypatch.setattr(runner, "_resolve_local_vocoder", _REAL_RESOLVE_VOCODER)
        monkeypatch.setattr(runner.model_registry, "TTS_CACHE_DIR", tmp_path)

        spec = runner.model_registry.resolve("f5tts", "f5tts-v1-base")
        voc_snap = runner.model_registry.extra_repo_dirs(spec)[0] / "snapshots" / "s1"
        voc_snap.mkdir(parents=True)
        (voc_snap / "config.yaml").write_text("x")
        try:
            (voc_snap / "pytorch_model.bin").symlink_to(tmp_path / "does_not_exist")
        except (OSError, NotImplementedError):
            pytest.skip("filesystem/OS does not allow symlink creation")

        assert runner._resolve_local_vocoder(spec) is None


class TestSynthesizeSpeech:
    def test_minimum_inference_duration_uses_reference_wav_length(self, tmp_path):
        runner = _load_runner("f5tts")
        reference_path = tmp_path / "reference.wav"
        reference_audio = cast(wave.Wave_write, wave.open(str(reference_path), "wb"))
        with reference_audio:
            reference_audio.setnchannels(1)
            reference_audio.setsampwidth(2)
            reference_audio.setframerate(24000)
            reference_audio.writeframes(b"\x00\x00" * 12000)

        assert runner._minimum_inference_duration(str(reference_path)) == 1.5

    def test_missing_voice_path_raises_before_touching_the_model(self):
        runner = _load_runner("f5tts")

        with pytest.raises(RuntimeError, match="requires voice_path"):
            runner._synthesize_speech("hello", None, "some ref text")

    def test_missing_ref_text_raises_before_touching_the_model(self):
        runner = _load_runner("f5tts")

        with pytest.raises(RuntimeError, match="requires ref_text"):
            runner._synthesize_speech("hello", "/some/voice.wav", None)

    def test_empty_ref_text_also_raises(self):
        runner = _load_runner("f5tts")

        with pytest.raises(RuntimeError, match="requires ref_text"):
            runner._synthesize_speech("hello", "/some/voice.wav", "")

    def test_only_config_keys_actually_present_are_forwarded(self, monkeypatch):
        runner = _load_runner("f5tts")
        runner._config = {"nfe_step": 16, "speed": 1.2}
        fake_model = _FakeF5TTS()
        monkeypatch.setattr(runner, "_get_model", lambda config: fake_model)
        long_text = "hello, this is a sufficiently long sentence for this test"

        wav_tensor = runner._synthesize_speech(long_text, "/voices/a.wav", "reference text")

        call = fake_model.infer_calls[0]
        assert call["nfe_step"] == 16
        assert call["speed"] == 1.2
        assert "cfg_strength" not in call
        assert call["ref_file"] == "/voices/a.wav"
        assert call["ref_text"] == "reference text"
        assert call["gen_text"] == long_text
        assert wav_tensor.shape == (1, 10)

    def test_short_gen_text_gets_a_slower_speed_override(self, monkeypatch):
        runner = _load_runner("f5tts")
        runner._config = {}
        fake_model = _FakeF5TTS()
        monkeypatch.setattr(runner, "_get_model", lambda config: fake_model)

        runner._synthesize_speech("Muse of Fire", "/voices/a.wav", "reference text")

        assert fake_model.infer_calls[0]["speed"] == runner._SHORT_TEXT_SPEED

    def test_explicit_faster_than_override_speed_is_still_overridden_for_short_text(self, monkeypatch):
        runner = _load_runner("f5tts")
        runner._config = {"speed": 1.2}
        fake_model = _FakeF5TTS()
        monkeypatch.setattr(runner, "_get_model", lambda config: fake_model)

        runner._synthesize_speech("Muse of Fire", "/voices/a.wav", "reference text")

        assert fake_model.infer_calls[0]["speed"] == runner._SHORT_TEXT_SPEED

    def test_explicit_already_slow_speed_is_not_overridden_for_short_text(self, monkeypatch):
        runner = _load_runner("f5tts")
        runner._config = {"speed": 0.2}
        fake_model = _FakeF5TTS()
        monkeypatch.setattr(runner, "_get_model", lambda config: fake_model)

        runner._synthesize_speech("Muse of Fire", "/voices/a.wav", "reference text")

        assert fake_model.infer_calls[0]["speed"] == 0.2

    def test_ultra_short_text_gets_a_reference_relative_duration_floor(self, monkeypatch):
        runner = _load_runner("f5tts")
        runner._config = {}
        fake_model = _FakeF5TTS()
        monkeypatch.setattr(runner, "_get_model", lambda config: fake_model)
        monkeypatch.setattr(runner, "_minimum_inference_duration", lambda voice: 7.5)

        runner._synthesize_speech("1", "/voices/a.wav", "reference text")

        assert fake_model.infer_calls[0]["speed"] == runner._SHORT_TEXT_SPEED
        assert fake_model.infer_calls[0]["fix_duration"] == 7.5

    def test_explicit_fixed_duration_is_preserved_for_ultra_short_text(self, monkeypatch):
        runner = _load_runner("f5tts")
        runner._config = {"fix_duration": 9.25}
        fake_model = _FakeF5TTS()
        monkeypatch.setattr(runner, "_get_model", lambda config: fake_model)
        monkeypatch.setattr(
            runner,
            "_minimum_inference_duration",
            lambda voice: pytest.fail("explicit fix_duration must win"),
        )

        runner._synthesize_speech("1", "/voices/a.wav", "reference text")

        assert fake_model.infer_calls[0]["fix_duration"] == 9.25

    def test_long_gen_text_is_not_affected_by_the_short_text_override(self, monkeypatch):
        runner = _load_runner("f5tts")
        runner._config = {}
        fake_model = _FakeF5TTS()
        monkeypatch.setattr(runner, "_get_model", lambda config: fake_model)
        long_text = (
            "This is a much longer sentence that should not trigger the "
            "short-text speed override at all."
        )

        runner._synthesize_speech(long_text, "/voices/a.wav", "reference text")

        assert "speed" not in fake_model.infer_calls[0]


class TestHandleInit:
    def test_success_returns_ok_with_features_and_samplerate(self, monkeypatch):
        runner = _load_runner("f5tts")
        _patch_f5tts_class(monkeypatch)

        response = runner.handle_init({"request_id": "r1", "config": {"device": "cpu"}})

        assert response == {
            "request_id": "r1",
            "status": "OK",
            "supported_features": ["zero_shot"],
            "samplerate": 24000,
        }

    def test_model_load_failure_returns_error_without_raising(self, monkeypatch):
        runner = _load_runner("f5tts")

        def _boom(config):
            raise RuntimeError("checkpoint download failed")

        monkeypatch.setattr(runner, "_load_model", _boom)

        response = runner.handle_init({"request_id": "r2", "config": {}})

        assert response["request_id"] == "r2"
        assert response["status"] == "ERROR"
        assert "checkpoint download failed" in response["error"]


class TestHandleSynthesize:
    def test_success_writes_wav_and_reports_duration(self, monkeypatch, tmp_path):
        runner = _load_runner("f5tts")
        runner._sample_rate = 24000
        speech = torch.zeros(1, 2400)
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
        assert response["samplerate"] == 24000
        assert response["duration_seconds"] == pytest.approx(0.1)
        assert output_file.exists()

    def test_leading_silence_is_prepended(self, monkeypatch, tmp_path):
        runner = _load_runner("f5tts")
        runner._sample_rate = 24000
        speech = torch.zeros(1, 2400)
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

    def test_empty_text_and_no_silence_still_writes_a_valid_empty_wav(self, tmp_path):
        runner = _load_runner("f5tts")
        runner._sample_rate = 24000
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
        runner = _load_runner("f5tts")

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
        runner = _load_runner("f5tts")
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
        runner = _load_runner("f5tts")
        runner._sample_rate = 24000
        calls = []
        monkeypatch.setattr(runner, "release_accelerator_memory", lambda log=None, reason="": calls.append(reason))
        speech = torch.zeros(1, 2400)
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
