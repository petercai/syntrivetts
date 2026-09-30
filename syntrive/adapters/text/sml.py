from __future__ import annotations

BREAK = "‡break‡"
PAUSE = "‡pause‡"

DEFAULT_SML_DURATIONS_MS = {
    "break": 1000,
    "pause": 500,
}

_STRENGTH = {PAUSE: 1, BREAK: 2}

_TERMINALS = frozenset("。！？…；：.!?;:")
_CLOSERS = frozenset("”’」』）》】\"')]")


def is_marker(token: str) -> bool:
    return token in _STRENGTH


def strongest(a: str, b: str) -> str:
    return a if _STRENGTH[a] >= _STRENGTH[b] else b


def ends_with_terminal_punctuation(text: str) -> bool:
    stripped = text.rstrip()
    while stripped and stripped[-1] in _CLOSERS:
        stripped = stripped[:-1].rstrip()
    return bool(stripped) and stripped[-1] in _TERMINALS


def sentence_period(language: str) -> str:
    return "。" if language == "zh" else "."


_TITLE_MAX_WORDS = 4
_TITLE_MAX_CHARS = 30
_TITLE_MAX_CHARS_ZH = 15
_CONTINUATION_TAIL = frozenset(",，、")


def is_title_like(text: str, language: str) -> bool:
    stripped = text.strip()
    if not stripped or stripped[-1] in _CONTINUATION_TAIL:
        return False
    first = stripped[0]
    if language == "zh":
        return len(stripped) <= _TITLE_MAX_CHARS_ZH
    return (
        len(stripped) <= _TITLE_MAX_CHARS
        and len(stripped.split()) <= _TITLE_MAX_WORDS
        and (first.isupper() or first.isdigit())
    )
