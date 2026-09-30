from typing import Optional

NATIVE_SPEED_ENGINES = frozenset({"cosyvoice", "indextts"})

SHORT_ENGINE_CODES = {
    "cosyvoice": "cosy",
    "indextts": "idx",
    "qwen3tts": "qwen",
    "voxcpm": "vox",
}


def _format_speed(speed: float) -> str:
    text = f"{speed:.2f}".rstrip("0")
    return text if not text.endswith(".") else f"{text}0"


def format_provenance_tag(engine_name: str, speed: Optional[float] = None) -> str:
    short_code = SHORT_ENGINE_CODES.get(engine_name, engine_name)
    effective_speed = 1.0 if speed is None else speed
    return f"syntrive ({short_code}-{_format_speed(effective_speed)})"
