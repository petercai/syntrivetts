from __future__ import annotations

from typing import Optional

from syntrive.adapters.tts.engines.base import BaseTTSEngine
from syntrive.adapters.tts.engines.cosyvoice import CosyVoiceEngineAdapter
from syntrive.adapters.tts.engines.f5tts import F5TTSEngineAdapter
from syntrive.adapters.tts.engines.indextts import IndexTTSEngineAdapter
from syntrive.adapters.tts.engines.qwen3tts import Qwen3TTSEngineAdapter
from syntrive.adapters.tts.engines.voxcpm import VoxCPMEngineAdapter
from syntrive.adapters.tts.engines.xtts import XTTSEngineAdapter

_ENGINE_REGISTRY = {
    "xtts": XTTSEngineAdapter,
    "cosyvoice": CosyVoiceEngineAdapter,
    "indextts": IndexTTSEngineAdapter,
    "f5tts": F5TTSEngineAdapter,
    "qwen3tts": Qwen3TTSEngineAdapter,
    "voxcpm": VoxCPMEngineAdapter,
}


class UnknownEngineError(ValueError):
    pass


def get_engine(engine_name: str, options: Optional[dict] = None) -> BaseTTSEngine:
    engine_class = _ENGINE_REGISTRY.get(engine_name)
    if engine_class is None:
        raise UnknownEngineError(
            f"get_engine: unknown or not-yet-synthesis-ready engine {engine_name!r} "
            f"(available: {sorted(_ENGINE_REGISTRY)})"
        )
    return engine_class(options=options)
