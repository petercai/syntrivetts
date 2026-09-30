from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from syntrive.adapters.text.sml import (
    BREAK,
    PAUSE,
    ends_with_terminal_punctuation,
    is_marker,
    is_title_like,
    sentence_period,
    strongest,
)

logger = logging.getLogger(__name__)

_NO_SPACE_LANGUAGES = {"zh"}

_HEADING_TAGS = frozenset(["h1", "h2", "h3", "h4", "h5", "h6"])
_BREAK_TAGS = _HEADING_TAGS | frozenset(["p"])
_PAUSE_TAGS = frozenset(["ol", "ul", "li", "div", "br", "hr"])
_BLOCK_TAGS = _BREAK_TAGS | _PAUSE_TAGS
_VOID_BREAK_TAGS = frozenset(["br", "hr"])
_CONTAINER_SCAN_TAGS = _BLOCK_TAGS - _VOID_BREAK_TAGS

_AUTO_DETECT_THRESHOLD = 0.5


def _auto_detect_mode(soup) -> str:
    paragraphs = soup.find_all("p", class_="normaltext")
    if not paragraphs:
        return "none"
    annotated = sum(1 for p in paragraphs if p.find("b", class_="calibre3"))
    ratio = annotated / len(paragraphs)
    logger.debug(
        "_auto_detect_mode: total_p=%d annotated=%d ratio=%.2f",
        len(paragraphs), annotated, ratio,
    )
    return "html_tag" if ratio >= _AUTO_DETECT_THRESHOLD else "none"


_BR_SENTINEL = "BR"


@dataclass(frozen=True)
class ReviewItem:
    element_id: str
    preview: str


def _text_with_line_breaks(elem, separator: str) -> str:
    if elem.find("br") is None:
        return elem.get_text(separator=separator, strip=True)
    import copy

    work = copy.copy(elem)
    for br in work.find_all("br"):
        br.replace_with(_BR_SENTINEL)
    return work.get_text(separator=separator, strip=True)


def _extract_html_tag_role(elem, separator: str) -> tuple[str, str, bool, str]:
    import copy
    elem_copy = copy.copy(elem)
    b_tag = elem_copy.find("b", class_="calibre3")
    if b_tag:
        role_raw = b_tag.get_text(strip=True)
        role_id = role_raw.rstrip("：:").strip()
        b_tag.decompose()
        text = _text_with_line_breaks(elem_copy, separator)
        if role_id and text:
            return role_id, text, True, "html_tag"
    text = _text_with_line_breaks(elem, separator)
    return "narrator", text, False, "default"


def extract_tts_script_from_html(
    html: str,
    language: str = "en",
    extraction_mode: str = "none",
    role_voice_map: Optional[dict[str, int]] = None,
    review_sink: Optional[list] = None,
) -> str:
    from bs4 import BeautifulSoup, Tag

    if role_voice_map is None:
        role_voice_map = {}
    role_voice_map.setdefault("narrator", 0)

    separator = "" if language in _NO_SPACE_LANGUAGES else " "
    soup = BeautifulSoup(html, "html.parser")

    effective_mode = extraction_mode
    if extraction_mode == "auto":
        effective_mode = _auto_detect_mode(soup)
    elif extraction_mode == "dialogue_detect":
        logger.warning(
            "extract_tts_script_from_html: dialogue_detect not yet implemented -- "
            "falling back to none"
        )
        effective_mode = "none"

    lines: list[str] = []
    current_role: Optional[str] = None

    period = sentence_period(language)

    def _token_for(tag_name: str) -> str:
        return BREAK if tag_name in _BREAK_TAGS else PAUSE

    def _push_marker(token: str) -> None:
        if not lines:
            return
        if is_marker(lines[-1]):
            lines[-1] = strongest(lines[-1], token)
        else:
            lines.append(token)

    def _last_line_is_open_text() -> bool:
        return bool(lines) and not is_marker(lines[-1]) and not lines[-1].startswith("‡voice:")

    def _join_hard_lines(text: str) -> str:
        parts = [seg.strip() for seg in text.replace(_BR_SENTINEL, "\n").splitlines()]
        parts = [seg for seg in parts if seg]
        if len(parts) <= 1:
            return parts[0] if parts else ""
        closed = [p if ends_with_terminal_punctuation(p) else p + period for p in parts]
        return separator.join(closed)

    def _emit_leaf_text(elem: Tag, tag_name: str) -> bool:
        nonlocal current_role
        if effective_mode == "html_tag":
            role_id, text, _is_dialogue, _source = _extract_html_tag_role(elem, separator)
        else:
            role_id, text = "narrator", _text_with_line_breaks(elem, separator)
        body = _join_hard_lines(text)
        if not body:
            return False

        if tag_name in _HEADING_TAGS and _last_line_is_open_text():
            _push_marker(BREAK)

        if role_id not in role_voice_map:
            role_voice_map[role_id] = max(role_voice_map.values()) + 1
        if role_id != current_role:
            lines.append(f"‡voice:{role_voice_map[role_id]}‡")
            current_role = role_id

        if tag_name == "p" and not ends_with_terminal_punctuation(body) and is_title_like(body, language):
            logger.info("extract_tts_script_from_html: title-like <p> %r closed with a period", body)
            body += period

        lines.append(body)
        if tag_name == "p" and not ends_with_terminal_punctuation(body):
            element_id = str(elem.get("data-para-id") or elem.get("id") or "")
            preview = body[:60]
            logger.warning(
                "extract_tts_script_from_html: <p> without terminal punctuation "
                "(no marker emitted, review needed) id=%r preview=%r", element_id, preview,
            )
            if review_sink is not None:
                review_sink.append(ReviewItem(element_id=element_id, preview=preview))
        else:
            _push_marker(_token_for(tag_name))
        return True

    def _walk(node) -> bool:
        produced = False
        for child in node.children:
            if not isinstance(child, Tag):
                continue
            name = (child.name or "").lower()
            if child.get("data-table-summary"):
                continue

            if name in _VOID_BREAK_TAGS:
                _push_marker(_token_for(name))
                produced = True
                continue

            if name in _BLOCK_TAGS:
                if child.find(list(_CONTAINER_SCAN_TAGS)) is not None:
                    if _walk(child):
                        _push_marker(_token_for(name))
                        produced = True
                elif _emit_leaf_text(child, name):
                    produced = True
            elif _walk(child):
                produced = True

        return produced

    _walk(soup)

    logger.debug(
        "extract_tts_script_from_html: mode=%s lang=%s lines=%d roles=%d",
        effective_mode, language, len(lines), len(role_voice_map),
    )
    return "\n".join(lines)


def extract_text_from_html(html: str, language: str = "en") -> str:
    from bs4 import BeautifulSoup

    separator = "" if language in _NO_SPACE_LANGUAGES else " "

    soup = BeautifulSoup(html, "html.parser")
    paragraphs: list[str] = []

    for elem in soup.find_all(["p", "h1", "h2", "h3", "h4", "h5", "h6"]):
        if elem.get("data-table-summary"):
            continue
        text = elem.get_text(separator=separator, strip=True)
        if text:
            paragraphs.append(text)

    if not paragraphs:
        logger.debug(
            "extract_text_from_html: no block elements found — using full get_text fallback"
        )
        return soup.get_text(separator="\n", strip=True)

    return "\n".join(paragraphs)
