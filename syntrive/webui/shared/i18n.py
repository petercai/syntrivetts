from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Mapping

logger = logging.getLogger(__name__)

LANGS = ("en", "zh")
DEFAULT_LANG = "en"
LANG_COOKIE = "lang"

_WEBUI_DIR = Path(__file__).resolve().parent.parent


def _catalog_dirs() -> Iterable[Path]:
    yield _WEBUI_DIR / "shared" / "i18n"
    yield from sorted((_WEBUI_DIR / "features").glob("*/i18n"))


@lru_cache(maxsize=None)
def catalog(lang: str) -> Mapping[str, str]:
    merged: dict[str, str] = {}
    for folder in _catalog_dirs():
        path = folder / f"{lang}.json"
        if path.is_file():
            merged.update(json.loads(path.read_text(encoding="utf-8")))
    logger.debug("webui_i18n_loaded: lang=%s keys=%d", lang, len(merged))
    return merged


def translate(lang: str, key: str, **fields: object) -> str:
    text = catalog(lang).get(key) or catalog(DEFAULT_LANG).get(key) or key
    if not fields:
        return text
    try:
        return text.format(**fields)
    except (KeyError, IndexError, ValueError):
        logger.warning("webui_i18n_format_failed: lang=%s key=%s", lang, key)
        return text


def translate_or(lang: str, key: str, fallback: str) -> str:
    return catalog(lang).get(key) or catalog(DEFAULT_LANG).get(key) or fallback


def pick_lang(cookie_value: str | None, accept_language: str | None) -> str:
    if cookie_value in LANGS:
        return cookie_value
    first = (accept_language or "").split(",")[0].strip().lower()
    return "zh" if first.startswith("zh") else DEFAULT_LANG
