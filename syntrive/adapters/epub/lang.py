from __future__ import annotations

_LANGUAGE_ALIASES: dict[str, frozenset[str]] = {
    "en": frozenset({
        "en", "eng", "english",
        "en-us", "en-gb", "en-au", "en-ca", "en-nz", "en-ie", "en-za",
        "en-in", "en-sg", "en-ph",
    }),
    "zh": frozenset({
        "zh", "zho", "chinese",
        "zh-cn", "zh-tw", "zh-hk", "zh-mo", "zh-sg",
        "zh-hans", "zh-hant",
        "cmn",
        "cn",
    }),
}


def normalize_language(raw: str | None) -> str | None:
    if not raw:
        return None
    key = raw.strip().lower()
    for canonical, aliases in _LANGUAGE_ALIASES.items():
        if key in aliases:
            return canonical
    return None


def supported_languages() -> list[str]:
    return sorted(_LANGUAGE_ALIASES.keys())
