from __future__ import annotations

import json
import sys
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

ENGINE_NAME = "xtts"
SAMPLE_RATE = 24000
SUPPORTED_FEATURES = ["zero_shot"]

_EXPECTED_CHARS_PER_SECOND = 12.0
_MIN_EXPECTED_SECONDS = 0.5

_DEFAULT_MODEL_REPO = "coqui/XTTS-v2"
_DEFAULT_CACHE_DIR = TTS_CACHE_DIR

_model = None
_config: dict = {}
_latent_cache: dict = {}


def _log(message: str) -> None:
    print(f"[{ENGINE_NAME}] {message}", file=sys.stderr, flush=True)


def _respond(response: dict) -> None:
    sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _resolve_hf_file(repo_id: str, filename: str, cache_dir: str, *, offline: bool) -> str:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    try:
        path = hf_hub_download(
            repo_id=repo_id, filename=filename, cache_dir=cache_dir, local_files_only=True
        )
        _log(f"reusing cached {repo_id}/{filename}")
        return path
    except LocalEntryNotFoundError:
        if offline:
            raise RuntimeError(
                f"offline mode: {repo_id}/{filename} is not in the local cache "
                f"({cache_dir}); run once online to populate it before going offline"
            )
        _log(f"cache miss: downloading {repo_id}/{filename} -> {cache_dir}")
        return hf_hub_download(repo_id=repo_id, filename=filename, cache_dir=cache_dir)


def _load_model(config: dict):
    offline = apply_offline_mode(log=_log)
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts

    registry_kw = model_registry.runner_kwargs(ENGINE_NAME, config.get("model"))
    repo = config.get("model_repo") or registry_kw.get("model_repo") or _DEFAULT_MODEL_REPO
    sub = config.get("model_sub") or ""
    cache_dir = config.get("cache_dir") or str(_DEFAULT_CACHE_DIR)

    _log(f"loading XTTS model: repo={repo} sub={sub!r} cache_dir={cache_dir} offline={offline}")
    config_path = _resolve_hf_file(repo, f"{sub}config.json", cache_dir, offline=offline)
    checkpoint_path = _resolve_hf_file(repo, f"{sub}model.pth", cache_dir, offline=offline)
    vocab_path = _resolve_hf_file(repo, f"{sub}vocab.json", cache_dir, offline=offline)

    xtts_config = XttsConfig()
    xtts_config.load_json(config_path)
    model = Xtts.init_from_config(xtts_config)
    model.load_checkpoint(
        xtts_config,
        checkpoint_path=checkpoint_path,
        vocab_path=vocab_path,
        use_deepspeed=bool(config.get("use_deepspeed", False)),
        eval=True,
    )

    device = config.get("device") or "cpu"
    if device == "cuda":
        model.cuda()
    else:
        model.to(device)

    _log(f"model loaded: device={device} repo={repo}")
    return model


def _get_model(config: dict):
    global _model
    if _model is None:
        _model = _load_model(config)
    return _model


def _get_conditioning_latents(model, voice_path: str):
    if voice_path in _latent_cache:
        return _latent_cache[voice_path]
    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=[voice_path])
    _latent_cache[voice_path] = (gpt_cond_latent, speaker_embedding)
    return gpt_cond_latent, speaker_embedding


def _replace_periods_for_xtts(text: str) -> str:
    return text.replace(".", " —")


def _estimate_expected_seconds(text: str) -> float:
    return max(len(text) / _EXPECTED_CHARS_PER_SECOND, _MIN_EXPECTED_SECONDS)


def _synthesize_speech(text: str, voice_path: Optional[str]):
    if not voice_path:
        raise RuntimeError(
            f"{ENGINE_NAME} inference requires voice_path (reference-audio cloning) -- none given"
        )

    import torch

    model = _get_model(_config)
    gpt_cond_latent, speaker_embedding = _get_conditioning_latents(model, voice_path)

    fine_tuned_params = {
        key: cast_type(_config[key])
        for key, cast_type in {
            "temperature": float, "length_penalty": float, "num_beams": int,
            "repetition_penalty": float, "top_k": int, "top_p": float,
            "speed": float, "enable_text_splitting": bool,
        }.items()
        if key in _config
    }

    synth_text = _replace_periods_for_xtts(text)

    with torch.no_grad():
        result = model.inference(
            text=synth_text,
            language=_config.get("language") or "en",
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
            **fine_tuned_params,
        )
    wav = result.get("wav")
    if wav is None:
        raise RuntimeError(f"{ENGINE_NAME} inference returned no waveform")

    actual_seconds = len(wav) / SAMPLE_RATE
    expected_seconds = _estimate_expected_seconds(synth_text)
    ratio = actual_seconds / expected_seconds
    ratio_threshold = float(_config.get("retry_badcase_ratio_threshold", 6.0))
    if ratio > ratio_threshold:
        _log(
            f"badcase_detected: text_len={len(synth_text)} expected_seconds={expected_seconds:.2f} "
            f"actual_seconds={actual_seconds:.2f} ratio={ratio:.2f} threshold={ratio_threshold:.2f} "
            f"(accepted as-is -- no retry for xtts, see _synthesize_speech docstring)"
        )
    return torch.tensor(wav, dtype=torch.float32).unsqueeze(0)


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
        "samplerate": SAMPLE_RATE,
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
            silence_samples = int(SAMPLE_RATE * leading_silence_ms / 1000)
            segments.append(torch.zeros(1, silence_samples))

        if text:
            segments.append(_synthesize_speech(text, voice_path))
        elif not segments:
            segments.append(torch.zeros(1, 0))

        combined = segments[0] if len(segments) == 1 else torch.cat(segments, dim=1)
        torchaudio.save(output_file, combined, SAMPLE_RATE)
        duration_seconds = combined.shape[1] / SAMPLE_RATE

        _log(f"SYNTHESIZE ok: duration_seconds={duration_seconds:.3f} output_file={output_file}")
        release_accelerator_memory(log=_log, reason="synthesize_ok")
        return {
            "request_id": request_id,
            "status": "OK",
            "output_file": output_file,
            "duration_seconds": duration_seconds,
            "samplerate": SAMPLE_RATE,
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
