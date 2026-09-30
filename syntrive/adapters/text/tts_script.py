from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Union

from syntrive.adapters.text.sml import DEFAULT_SML_DURATIONS_MS  # noqa: E402,F401  (re-exported)

_MARKER_RE = re.compile(r"‡(voice:(\d+)|break|pause)‡")


@dataclass(frozen=True)
class TextSegment:
    text: str
    voice_id: int


@dataclass(frozen=True)
class SilenceMarker:
    duration_ms: int


TtsScriptToken = Union[TextSegment, SilenceMarker]


def parse_tts_script(content: str, sml_durations_ms: dict = None) -> List[TtsScriptToken]:
    durations = {**DEFAULT_SML_DURATIONS_MS, **(sml_durations_ms or {})}
    tokens: List[TtsScriptToken] = []
    active_voice = 0

    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        pos = 0
        for match in _MARKER_RE.finditer(line):
            text_before = line[pos:match.start()].strip()
            if text_before:
                tokens.append(TextSegment(text=text_before, voice_id=active_voice))

            kind = match.group(1)
            if kind.startswith("voice:"):
                active_voice = int(match.group(2))
            elif kind == "break":
                tokens.append(SilenceMarker(duration_ms=durations["break"]))
            elif kind == "pause":
                tokens.append(SilenceMarker(duration_ms=durations["pause"]))
            pos = match.end()

        trailing = line[pos:].strip()
        if trailing:
            tokens.append(TextSegment(text=trailing, voice_id=active_voice))

    return tokens
