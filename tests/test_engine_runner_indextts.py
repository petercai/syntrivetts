from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch
import torchaudio

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
    runner = _load_runner("indextts")
    runner._model = None
    runner._config = {}
    runner._sample_rate = runner.DEFAULT_SAMPLE_RATE
    yield
    runner._model = None
    runner._config = {}
    runner._sample_rate = runner.DEFAULT_SAMPLE_RATE


class _FakeIndexTTS2:
    def __init__(self, cfg_path=None, model_dir=None, device=None,
                 use_fp16=None, use_cuda_kernel=None, use_deepspeed=None):
        self.cfg_path = cfg_path
        self.model_dir = model_dir
        self.device = device
        self.use_fp16 = use_fp16
        self.use_cuda_kernel = use_cuda_kernel
        self.use_deepspeed = use_deepspeed
        self.infer_calls: list = []
        self.output_sample_rate = 16000

    def infer(self, spk_audio_prompt=None, text=None, output_path=None, verbose=False, **kwargs):
        self.infer_calls.append(
            {"spk_audio_prompt": spk_audio_prompt, "text": text, "output_path": output_path, **kwargs}
        )
        torchaudio.save(output_path, torch.zeros(1, 10), self.output_sample_rate)
        return output_path


class _FakeIndexTTS2_5(_FakeIndexTTS2):
    def __init__(self, cfg_path=None, model_dir=None, device=None,
                 use_bf16=None, use_cuda_kernel=None, use_deepspeed=None):
        super().__init__(cfg_path=cfg_path, model_dir=model_dir, device=device,
                          use_cuda_kernel=use_cuda_kernel, use_deepspeed=use_deepspeed)
        self.use_bf16 = use_bf16

    def infer(self, spk_audio_prompt=None, text=None, output_path=None, lang=None, verbose=False, **kwargs):
        return super().infer(spk_audio_prompt=spk_audio_prompt, text=text, output_path=output_path,
                              verbose=verbose, lang=lang, **kwargs)


_FakeIndexTTS2_5.__module__ = "indextts.infer_v2_5"


def _patch_indextts_class(monkeypatch, v2_5_class=_FakeIndexTTS2_5):
    v2_5_module = ModuleType("indextts.infer_v2_5")
    v2_5_module.IndexTTS2 = v2_5_class
    monkeypatch.setitem(sys.modules, "indextts.infer_v2_5", v2_5_module)
    fake_pkg = ModuleType("indextts")
    monkeypatch.setitem(sys.modules, "indextts", fake_pkg)


class TestStdoutIsReservedForNdjson:
    def test_respond_writes_to_the_captured_real_handle_not_current_stdout(self, monkeypatch):
        import io

        runner = _load_runner("indextts")
        real = io.StringIO()
        monkeypatch.setattr(runner, "_REAL_STDOUT", real)
        monkeypatch.setattr(sys, "stdout", io.StringIO())

        runner._respond({"request_id": "r1", "status": "OK"})

        assert '"request_id": "r1"' in real.getvalue()
        assert sys.stdout.getvalue() == ""

    def test_main_points_stdout_at_stderr_for_the_request_loop(self, monkeypatch):
        import io

        runner = _load_runner("indextts")
        monkeypatch.setattr(sys, "stdin", io.StringIO(""))
        monkeypatch.setattr(sys, "stdout", io.StringIO())
        monkeypatch.setattr(runner, "_REAL_STDOUT", sys.stdout)
        original_stderr = sys.stderr

        runner.main()

        assert sys.stdout is original_stderr


class TestLoadModel:
    def test_default_model_dir_routes_to_v2_5_and_uses_the_shared_default_dir(self, monkeypatch):
        runner = _load_runner("indextts")
        _patch_indextts_class(monkeypatch)

        model = runner._load_model({})

        assert isinstance(model, _FakeIndexTTS2_5)
        assert model.model_dir == str(runner._DEFAULT_MODEL_DIR)
        assert model.cfg_path == str(runner._DEFAULT_MODEL_DIR / "config.yaml")
        assert model.device == "cpu"

    def test_legacy_indextts_2_model_id_still_loads_the_v2_5_class(self, monkeypatch):
        runner = _load_runner("indextts")
        _patch_indextts_class(monkeypatch)

        model = runner._load_model({"model": "indextts-2"})

        assert isinstance(model, _FakeIndexTTS2_5)

    def test_explicit_indextts_2_5_routes_to_the_v2_5_class(self, monkeypatch):
        runner = _load_runner("indextts")
        _patch_indextts_class(monkeypatch)

        model = runner._load_model({"model": "indextts-2.5"})

        assert isinstance(model, _FakeIndexTTS2_5)

    def test_missing_infer_v2_5_module_raises_an_actionable_reprovision_error(self, monkeypatch):
        runner = _load_runner("indextts")
        monkeypatch.setitem(sys.modules, "indextts", ModuleType("indextts"))
        monkeypatch.delitem(sys.modules, "indextts.infer_v2_5", raising=False)

        with pytest.raises(RuntimeError, match="setup_tts_envs.py indextts"):
            runner._load_model({"model": "indextts-2.5"})

    def test_model_dir_and_device_overrides(self, monkeypatch):
        runner = _load_runner("indextts")
        _patch_indextts_class(monkeypatch)

        model = runner._load_model({"model_dir": "/custom/ckpt", "device": "cuda"})

        assert model.model_dir == "/custom/ckpt"
        assert Path(model.cfg_path) == Path("/custom/ckpt") / "config.yaml"
        assert model.device == "cuda"

    def test_cfg_path_override_wins_over_model_dir_derived_default(self, monkeypatch):
        runner = _load_runner("indextts")
        _patch_indextts_class(monkeypatch)

        model = runner._load_model({"model_dir": "/custom/ckpt", "cfg_path": "/elsewhere/my_config.yaml"})

        assert model.cfg_path == "/elsewhere/my_config.yaml"

    def test_use_fp16_is_forced_off_on_cpu_regardless_of_config(self, monkeypatch):
        runner = _load_runner("indextts")
        _patch_indextts_class(monkeypatch)

        model = runner._load_model({"device": "cpu", "use_fp16": True})

        assert model.use_bf16 is False

    def test_use_fp16_maps_to_use_bf16_for_indextts_2_5(self, monkeypatch):
        runner = _load_runner("indextts")
        _patch_indextts_class(monkeypatch)

        model = runner._load_model({"model": "indextts-2.5", "device": "cuda", "use_fp16": True})

        assert model.use_bf16 is True
        assert not hasattr(model, "use_fp16") or model.use_fp16 is None

    def test_use_cuda_kernel_and_use_deepspeed_always_false(self, monkeypatch):
        runner = _load_runner("indextts")
        _patch_indextts_class(monkeypatch)

        model = runner._load_model({"device": "cuda"})

        assert model.use_cuda_kernel is False
        assert model.use_deepspeed is False

    def test_get_model_only_loads_once(self, monkeypatch):
        runner = _load_runner("indextts")
        _patch_indextts_class(monkeypatch)
        load_calls = []
        original = runner._load_model
        monkeypatch.setattr(runner, "_load_model", lambda config: (load_calls.append(1), original(config))[1])

        first = runner._get_model({})
        second = runner._get_model({})

        assert first is second
        assert len(load_calls) == 1


class TestSynthesizeSpeech:
    def test_missing_voice_path_raises_before_touching_the_model(self):
        runner = _load_runner("indextts")

        with pytest.raises(RuntimeError, match="requires voice_path"):
            runner._synthesize_speech("hello", None)

    def test_real_call_writes_via_scratch_file_and_reads_back_the_real_rate(self, monkeypatch):
        runner = _load_runner("indextts")
        fake_model = _FakeIndexTTS2()
        monkeypatch.setattr(runner, "_get_model", lambda config: fake_model)

        waveform, sr = runner._synthesize_speech("hello", "/voices/a.wav")

        assert sr == 16000
        assert waveform.shape[0] == 1
        assert waveform.shape[1] > 0
        call = fake_model.infer_calls[0]
        assert call["spk_audio_prompt"] == "/voices/a.wav"
        assert call["text"] == "hello"
        assert call["output_path"] is not None
        assert call["lang"] == "en"
        assert call["duration_factor"] == pytest.approx(1.0)

    def test_v2_5_model_receives_the_configured_language_code(self, monkeypatch):
        runner = _load_runner("indextts")
        fake_model = _FakeIndexTTS2_5()
        monkeypatch.setattr(runner, "_get_model", lambda config: fake_model)
        runner._config = {"language": "zh"}

        runner._synthesize_speech("你好", "/voices/a.wav")

        assert fake_model.infer_calls[0]["lang"] == "zh"

    def test_v2_5_model_defaults_to_en_when_no_language_is_configured(self, monkeypatch):
        runner = _load_runner("indextts")
        fake_model = _FakeIndexTTS2_5()
        monkeypatch.setattr(runner, "_get_model", lambda config: fake_model)
        runner._config = {}

        runner._synthesize_speech("hello", "/voices/a.wav")

        assert fake_model.infer_calls[0]["lang"] == "en"

    def test_duration_factor_is_inverted_for_a_faster_speed(self, monkeypatch):
        runner = _load_runner("indextts")
        fake_model = _FakeIndexTTS2_5()
        monkeypatch.setattr(runner, "_get_model", lambda config: fake_model)
        runner._config = {"speed": 2.0}

        runner._synthesize_speech("hello", "/voices/a.wav")

        assert fake_model.infer_calls[0]["duration_factor"] == pytest.approx(0.5)

    def test_duration_factor_is_inverted_for_a_slower_speed(self, monkeypatch):
        runner = _load_runner("indextts")
        fake_model = _FakeIndexTTS2_5()
        monkeypatch.setattr(runner, "_get_model", lambda config: fake_model)
        runner._config = {"speed": 0.5}

        runner._synthesize_speech("hello", "/voices/a.wav")

        assert fake_model.infer_calls[0]["duration_factor"] == pytest.approx(2.0)

    def test_duration_factor_defaults_to_identity_when_speed_is_not_configured(self, monkeypatch):
        runner = _load_runner("indextts")
        fake_model = _FakeIndexTTS2_5()
        monkeypatch.setattr(runner, "_get_model", lambda config: fake_model)
        runner._config = {}

        runner._synthesize_speech("hello", "/voices/a.wav")

        assert fake_model.infer_calls[0]["duration_factor"] == pytest.approx(1.0)

    def test_ref_text_is_never_part_of_this_functions_signature(self):
        import inspect

        runner = _load_runner("indextts")
        params = inspect.signature(runner._synthesize_speech).parameters
        assert "ref_text" not in params


class TestHandleInit:
    def test_success_returns_ok_with_features_and_samplerate(self, monkeypatch):
        runner = _load_runner("indextts")
        _patch_indextts_class(monkeypatch)

        response = runner.handle_init({"request_id": "r1", "config": {"device": "cpu"}})

        assert response == {
            "request_id": "r1",
            "status": "OK",
            "supported_features": ["zero_shot"],
            "samplerate": runner.DEFAULT_SAMPLE_RATE,
        }

    def test_model_load_failure_returns_error_without_raising(self, monkeypatch):
        runner = _load_runner("indextts")

        def _boom(config):
            raise RuntimeError("checkpoint not found -- run `python tools/model_dl.py indextts indextts-2` first")

        monkeypatch.setattr(runner, "_load_model", _boom)

        response = runner.handle_init({"request_id": "r2", "config": {}})

        assert response["request_id"] == "r2"
        assert response["status"] == "ERROR"
        assert "checkpoint not found" in response["error"]

    def test_model_load_failure_releases_accelerator_memory(self, monkeypatch):
        runner = _load_runner("indextts")
        calls = []
        monkeypatch.setattr(runner, "release_accelerator_memory", lambda log=None, reason="": calls.append(reason))
        monkeypatch.setattr(runner, "_load_model", lambda config: (_ for _ in ()).throw(RuntimeError("oom")))

        runner.handle_init({"request_id": "r8", "config": {}})

        assert calls == ["init_failed"]


class TestHandleSynthesize:
    def test_success_writes_wav_and_reports_duration(self, monkeypatch, tmp_path):
        runner = _load_runner("indextts")
        speech = torch.zeros(1, 2205)
        monkeypatch.setattr(runner, "_synthesize_speech", lambda text, voice_path: (speech, 22050))
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
        assert response["samplerate"] == 22050
        assert response["duration_seconds"] == pytest.approx(0.1)
        assert output_file.exists()

    def test_leading_silence_is_prepended_using_the_real_rate_learned_from_this_call(self, monkeypatch, tmp_path):
        runner = _load_runner("indextts")
        speech = torch.zeros(1, 2205)
        monkeypatch.setattr(runner, "_synthesize_speech", lambda text, voice_path: (speech, 22050))
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
        runner = _load_runner("indextts")
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
        runner = _load_runner("indextts")

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
        runner = _load_runner("indextts")
        calls = []
        monkeypatch.setattr(runner, "release_accelerator_memory", lambda log=None, reason="": calls.append(reason))

        def _boom(text, voice_path):
            raise RuntimeError("MPS backend out of memory")

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
        runner = _load_runner("indextts")
        calls = []
        monkeypatch.setattr(runner, "release_accelerator_memory", lambda log=None, reason="": calls.append(reason))
        speech = torch.zeros(1, 2205)
        monkeypatch.setattr(runner, "_synthesize_speech", lambda text, voice_path: (speech, 22050))
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
