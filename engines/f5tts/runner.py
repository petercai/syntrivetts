from __future__ import annotations

import json
import sys
import wave
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _shared import model_registry  # noqa: E402
from _shared.runtime_env import (  # noqa: E402
    TTS_CACHE_DIR,
    apply_offline_mode,
    configure_model_env,
    release_accelerator_memory,
)

ENGINE_NAME = "f5tts"
DEFAULT_SAMPLE_RATE = 24000
SUPPORTED_FEATURES = ["zero_shot"]

_SHORT_TEXT_BYTE_THRESHOLD = 30
_SHORT_TEXT_SPEED = 0.3
_ULTRA_SHORT_TEXT_BYTE_THRESHOLD = 10
_MIN_GENERATED_DURATION_SECONDS = 1.0
_MAX_REFERENCE_DURATION_SECONDS = 12.0

_DEFAULT_CACHE_DIR = TTS_CACHE_DIR

_model = None
_config: dict = {}
_sample_rate = DEFAULT_SAMPLE_RATE

_REAL_STDOUT = sys.stdout


def _log(message: str) -> None:
    print(f"[{ENGINE_NAME}] {message}", file=sys.stderr, flush=True)


def _respond(response: dict) -> None:
    _REAL_STDOUT.write(json.dumps(response, ensure_ascii=False) + "\n")
    _REAL_STDOUT.flush()


def _resolve_local_vocoder(spec) -> Optional[Path]:
    if spec is None:
        return None
    for repo_dir in model_registry.extra_repo_dirs(spec):
        if "vocos" not in repo_dir.name:
            continue
        snapshot = model_registry.resolved_snapshot_dir(repo_dir)
        if snapshot is None:
            continue
        if (snapshot / "config.yaml").is_file() and (snapshot / "pytorch_model.bin").is_file():
            return snapshot
    return None


def _resolve_local_checkpoint(spec, model_name: str) -> Optional[Path]:
    if spec is None:
        return None
    snapshot = model_registry.resolved_snapshot_dir(spec.local_path())
    if snapshot is None:
        return None
    for pattern in ("model_*.safetensors", "model_*.pt"):
        matches = sorted((snapshot / model_name).glob(pattern))
        if matches:
            return matches[0]
    return None


def _load_model(config: dict):
    offline = apply_offline_mode(log=_log)
    if not offline:
        offline = model_registry.apply_offline_env_if_cached(
            ENGINE_NAME, config.get("model"), log=_log
        )
    from f5_tts.api import F5TTS

    registry_kw = model_registry.runner_kwargs(ENGINE_NAME, config.get("model"))
    model_name = config.get("model_name") or registry_kw.get("model") or "F5TTS_v1_Base"
    device = config.get("device") or "cpu"
    hf_cache_dir = config.get("cache_dir") or str(_DEFAULT_CACHE_DIR)

    spec = model_registry.resolve(ENGINE_NAME, config.get("model"))
    f5_kwargs: dict = {}
    vocoder_dir = _resolve_local_vocoder(spec)
    if vocoder_dir is not None:
        f5_kwargs["vocoder_local_path"] = str(vocoder_dir)
        _log(f"vocos: local path {vocoder_dir}")
    else:
        _log("vocos: no local copy found -> F5TTS will fetch charactr/vocos-mel-24khz")
    checkpoint_file = _resolve_local_checkpoint(spec, model_name)
    if checkpoint_file is not None:
        f5_kwargs["ckpt_file"] = str(checkpoint_file)
        _log(f"checkpoint: local file {checkpoint_file}")
    else:
        _log(f"checkpoint: no local copy found -> F5TTS will fetch SWivid/F5-TTS/{model_name}")

    if offline and ("vocoder_local_path" not in f5_kwargs or "ckpt_file" not in f5_kwargs):
        missing = []
        if "vocoder_local_path" not in f5_kwargs:
            missing.append("charactr/vocos-mel-24khz (vocoder)")
        if "ckpt_file" not in f5_kwargs:
            missing.append(f"SWivid/F5-TTS ({model_name} checkpoint)")
        model_id = config.get("model") or "f5tts-v1-base"
        raise RuntimeError(
            f"offline: F5-TTS is missing local files for {', '.join(missing)}. "
            f"On a networked machine run `python tools/model_dl.py f5tts {model_id}` "
            f"(fetches the checkpoint AND its bundled vocos vocoder), then copy BOTH "
            f"models/tts/models--SWivid--F5-TTS/ and "
            f"models/tts/models--charactr--vocos-mel-24khz/ here, preserving symlinks "
            f"(rsync -a / cp -a). Looked under cache_dir={hf_cache_dir}."
        )

    _log(
        f"loading F5-TTS model: model={model_name} device={device} "
        f"hf_cache_dir={hf_cache_dir} offline={offline} "
        f"local_vocoder={'vocoder_local_path' in f5_kwargs} "
        f"local_ckpt={'ckpt_file' in f5_kwargs}"
    )
    model = F5TTS(model=model_name, device=device, hf_cache_dir=hf_cache_dir, **f5_kwargs)
    _log(f"model loaded: device={device} target_sample_rate={model.target_sample_rate}")
    return model


def _get_model(config: dict):
    global _model, _sample_rate
    if _model is None:
        _model = _load_model(config)
        _sample_rate = getattr(_model, "target_sample_rate", DEFAULT_SAMPLE_RATE)
    return _model


def _minimum_inference_duration(voice_path: str) -> Optional[float]:
    try:
        with wave.open(voice_path, "rb") as reference_audio:
            reference_duration = reference_audio.getnframes() / reference_audio.getframerate()
    except (OSError, wave.Error, ZeroDivisionError) as exc:
        _log(f"ultra_short_duration_floor_unavailable: voice_path={voice_path} error={exc}")
        return None

    return min(reference_duration, _MAX_REFERENCE_DURATION_SECONDS) + _MIN_GENERATED_DURATION_SECONDS


def _synthesize_speech(text: str, voice_path: Optional[str], ref_text: Optional[str]):
    if not voice_path:
        raise RuntimeError(
            f"{ENGINE_NAME} inference requires voice_path (reference-audio cloning) -- none given"
        )
    if not ref_text:
        raise RuntimeError(
            f"{ENGINE_NAME} inference requires ref_text -- missing or empty sibling <voice>.txt "
            "for this reference voice"
        )

    import torch

    model = _get_model(_config)

    fine_tuned_params = {
        key: cast_type(_config[key])
        for key, cast_type in {
            "nfe_step": int, "speed": float, "cross_fade_duration": float,
            "sway_sampling_coef": float, "cfg_strength": float, "target_rms": float,
            "fix_duration": float, "remove_silence": bool, "seed": int,
        }.items()
        if key in _config
    }

    requested_speed = fine_tuned_params.get("speed", 1.0)
    if requested_speed > _SHORT_TEXT_SPEED and len(text.encode("utf-8")) < _SHORT_TEXT_BYTE_THRESHOLD:
        fine_tuned_params["speed"] = _SHORT_TEXT_SPEED
        _log(
            f"short_text_speed_override: text_len={len(text)} "
            f"requested_speed={requested_speed} speed={_SHORT_TEXT_SPEED}"
        )

    text_byte_length = len(text.encode("utf-8"))
    if text_byte_length < _ULTRA_SHORT_TEXT_BYTE_THRESHOLD and "fix_duration" not in fine_tuned_params:
        minimum_duration = _minimum_inference_duration(voice_path)
        if minimum_duration is not None:
            fine_tuned_params["fix_duration"] = minimum_duration
            _log(
                f"ultra_short_duration_floor: text_len={len(text)} text_bytes={text_byte_length} "
                f"fix_duration={minimum_duration:.3f}s"
            )

    wav, _sr, _spec = model.infer(
        ref_file=voice_path,
        ref_text=ref_text,
        gen_text=text,
        file_wave=None,
        **fine_tuned_params,
    )
    return torch.from_numpy(wav).float().unsqueeze(0)


def handle_init(request: dict) -> dict:
    global _config
    configure_model_env(log=_log)
    _config = request.get("config") or {}
    offline = apply_offline_mode(log=_log)
    _log(f"INIT received: config={_config} offline={offline}")
    try:
        _get_model(_config)
    except Exception as exc:  # noqa: BLE001 -- must surface as a structured ERROR response, not crash the worker
        _log(f"INIT failed: error={exc}")
        release_accelerator_memory(log=_log, reason="init_failed")
        return {"request_id": request.get("request_id"), "status": "ERROR", "error": f"model load failed: {exc}"}

    return {
        "request_id": request.get("request_id"),
        "status": "OK",
        "supported_features": SUPPORTED_FEATURES,
        "samplerate": _sample_rate,
    }


def handle_synthesize(request: dict) -> dict:
    import torch
    import torchaudio

    request_id = request.get("request_id")
    text = request.get("text") or ""
    voice_path = request.get("voice_path")
    ref_text = request.get("ref_text")
    leading_silence_ms = int(request.get("leading_silence_ms") or 0)
    output_file = request.get("output_file")

    _log(
        f"SYNTHESIZE received: text_len={len(text)} voice_path={voice_path} "
        f"ref_text_len={len(ref_text) if ref_text else 0} "
        f"leading_silence_ms={leading_silence_ms} output_file={output_file} text={text!r}"
    )

    try:
        segments = []
        if leading_silence_ms > 0:
            silence_samples = int(_sample_rate * leading_silence_ms / 1000)
            segments.append(torch.zeros(1, silence_samples))

        if text:
            segments.append(_synthesize_speech(text, voice_path, ref_text))
        elif not segments:
            segments.append(torch.zeros(1, 0))

        combined = segments[0] if len(segments) == 1 else torch.cat(segments, dim=1)
        torchaudio.save(output_file, combined, _sample_rate)
        duration_seconds = combined.shape[1] / _sample_rate

        _log(f"SYNTHESIZE ok: duration_seconds={duration_seconds:.3f} output_file={output_file}")
        release_accelerator_memory(log=_log, reason="synthesize_ok")
        return {
            "request_id": request_id,
            "status": "OK",
            "output_file": output_file,
            "duration_seconds": duration_seconds,
            "samplerate": _sample_rate,
        }
    except Exception as exc:  # noqa: BLE001 -- must surface as a structured ERROR response, not crash the worker
        _log(f"SYNTHESIZE failed: error={exc}")
        release_accelerator_memory(log=_log, reason="synthesize_failed")
        return {"request_id": request_id, "status": "ERROR", "error": str(exc)}


DISPATCH = {
    "INIT": handle_init,
    "SYNTHESIZE": handle_synthesize,
}


def main() -> None:
    global _REAL_STDOUT
    _REAL_STDOUT = sys.stdout
    sys.stdout = sys.stderr

    _log("worker starting")
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError as exc:
            _log(f"malformed NDJSON request: {line!r} error={exc}")
            _respond({"status": "ERROR", "error": f"malformed NDJSON: {exc}"})
            continue

        handler = DISPATCH.get(request.get("cmd"))
        if handler is None:
            _log(f"unknown cmd: {request.get('cmd')!r}")
            _respond({
                "request_id": request.get("request_id"),
                "status": "ERROR",
                "error": f"unknown cmd: {request.get('cmd')!r}",
            })
            continue

        _respond(handler(request))

    _log("worker exiting (stdin closed)")


if __name__ == "__main__":
    main()
