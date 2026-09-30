from __future__ import annotations

import hashlib
import logging
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


@dataclass
class NumeralRule:
    when_p_type: list[str] = field(default_factory=list)
    when_p_class: list[str] = field(default_factory=list)
    numeral_type: str = "roman"
    mode: str = "inline"
    lang: str = "en"
    add_prefix: str = ""
    add_suffix: str = ""

    def matches(self, p_type: str, p_class: Optional[str]) -> bool:
        if self.when_p_type and p_type in self.when_p_type:
            return True
        if self.when_p_class and p_class in self.when_p_class:
            return True
        return False

    def to_dict(self) -> dict:
        d: dict = {}
        if self.when_p_type:
            d["p_type"] = self.when_p_type
        if self.when_p_class:
            d["p_class"] = self.when_p_class
        d["type"] = self.numeral_type
        d["mode"] = self.mode
        d["lang"] = self.lang
        if self.add_prefix:
            d["add_prefix"] = self.add_prefix
        if self.add_suffix:
            d["add_suffix"] = self.add_suffix
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "NumeralRule":
        p_type = data.get("p_type", [])
        p_class = data.get("p_class", [])
        if isinstance(p_type, str):
            p_type = [p_type]
        if isinstance(p_class, str):
            p_class = [p_class]
        return cls(
            when_p_type=p_type,
            when_p_class=p_class,
            numeral_type=data.get("type", "roman"),
            mode=data.get("mode", "inline"),
            lang=data.get("lang", "en"),
            add_prefix=data.get("add_prefix", ""),
            add_suffix=data.get("add_suffix", ""),
        )

_PARA_TAGS = frozenset({"p", "li", "dt", "dd", "blockquote", "h1", "h2", "h3", "h4", "h5", "h6"})

_ID_ATTR = "data-para-id"

_PREVIEW_CHARS = 80

_WS_RE = re.compile(r"\s+")

_ARABIC_RE = re.compile(r"^\s*\d+[.。．)）\s]")

_ROMAN_RE = re.compile(r"^\s*[IVXLCDMivxlcdm]{1,10}[.\s]")

_CHINESE_NUMERAL_CHARS = "零一二三四五六七八九十百千万亿〇"
_CHINESE_RE = re.compile(
    r"^\s*(?:[第]?[" + _CHINESE_NUMERAL_CHARS + r"]+[章节卷部回]|[" +
    _CHINESE_NUMERAL_CHARS + r"]+[、。\s])"
)


@dataclass
class ParaEntry:
    id: str
    order: int
    p_type: str
    p_class: Optional[str]
    text_preview: str
    sequence_type: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "order": self.order,
            "p_type": self.p_type,
            "p_class": self.p_class,
            "text_preview": self.text_preview,
            "sequence_type": self.sequence_type,
        }


@dataclass
class ParaIdMap:
    version: str = "2"
    hash_algo: str = "sha256"
    chapter_id: str = ""
    paragraphs: list[ParaEntry] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "hash_algo": self.hash_algo,
            "chapter_id": self.chapter_id,
            "paragraphs": [e.to_dict() for e in self.paragraphs],
        }

    def to_yaml(self) -> str:
        return yaml.dump(
            self.to_dict(),
            allow_unicode=True,
            default_flow_style=False,
            sort_keys=False,
        )

    @classmethod
    def from_yaml(cls, text: str) -> "ParaIdMap":
        data = yaml.safe_load(text) or {}
        entries = [
            ParaEntry(
                id=p["id"],
                order=p["order"],
                p_type=p.get("p_type", "p"),
                p_class=p.get("p_class"),
                text_preview=p.get("text_preview", ""),
                sequence_type=p.get("sequence_type"),
            )
            for p in data.get("paragraphs", [])
        ]
        return cls(
            version=data.get("version", "2"),
            hash_algo=data.get("hash_algo", "sha256"),
            chapter_id=data.get("chapter_id", ""),
            paragraphs=entries,
        )


class ParagraphIdMapper:
    def tag(self, soup: BeautifulSoup, chapter_id: str = "") -> ParaIdMap:
        seen_hashes: dict[str, int] = {}
        entries: list[ParaEntry] = []

        for order, tag in enumerate(soup.find_all(_PARA_TAGS)):
            text = tag.get_text()
            if not text.strip():
                continue

            para_id = _make_para_id(text, seen_hashes)
            tag.attrs[_ID_ATTR] = para_id

            classes = tag.get("class") or []
            p_class = classes[0] if classes else None

            preview = _normalize(text)[:_PREVIEW_CHARS]
            seq_type = _detect_sequence_type(preview)

            entries.append(ParaEntry(
                id=para_id,
                order=order,
                p_type=tag.name,
                p_class=p_class,
                text_preview=preview,
                sequence_type=seq_type,
            ))

        collision_count = sum(1 for c in seen_hashes.values() if c > 1)
        logger.info(
            "ParagraphIdMapper: chapter=%s tagged=%d hash_collisions=%d",
            chapter_id, len(entries), collision_count,
        )
        return ParaIdMap(chapter_id=chapter_id, paragraphs=entries)


def _make_para_id(text: str, seen: dict[str, int]) -> str:
    normalised = _normalize(text)
    digest = hashlib.sha256(normalised.encode("utf-8")).hexdigest()[:8]
    count = seen.get(digest, 0) + 1
    seen[digest] = count
    return f"p{digest}" if count == 1 else f"p{digest}_{count}"


def _normalize(text: str) -> str:
    return _WS_RE.sub(" ", text.strip())


def _detect_sequence_type(text: str) -> Optional[str]:
    if not text:
        return None
    if _CHINESE_RE.match(text):
        return "chinese"
    if _ARABIC_RE.match(text):
        return "arabic"
    if _ROMAN_RE.match(text) and not text[:1].islower():
        stripped = text.strip()
        match = _ROMAN_RE.match(stripped)
        if match and match.end() > 2:
            return "roman"
    return None


NUMERAL_RULES_FILENAME = "numeral_rules.yaml"

_NUMERAL_RULES_TEMPLATE = """\
# numeral_rules.yaml — Global numeral conversion rules for this book.
#
# Edit this file, then press Ctrl+R in the TUI (REVIEW_HTML step) to re-apply
# all cleaning rules. This file is never overwritten by the pipeline.
#
# Rules are applied to EVERY chapter in order; first-match wins per paragraph.
# Matching uses OR logic: a paragraph is matched when p_type OR p_class matches.
#
# Fields:
#   p_type:     HTML tag names, e.g. [h1, h2]    (omit to skip tag matching)
#   p_class:    CSS class names, e.g. [calibre1]  (omit to skip class matching)
#   type:       roman | arabic
#   mode:       inline  — replace numeral tokens within text
#               whole   — entire paragraph text is a numeral
#               (Note: arabic + inline is not supported; use text normalisation)
#   lang:       en | zh
#   add_prefix: (optional) text prepended before the converted word (whole mode)
#               e.g. add_prefix: "Section"  →  "I" becomes "Section one"
#   add_suffix: (optional) text appended after the converted word (whole mode)
#
# To find p_class values open a para_map.yaml in transcript_html/manifest/ and
# look at the p_class field of the paragraphs you want to target.
#
# Example (uncomment and adjust class names to match your EPUB):
#
# rules:
#   # Inline roman numerals in headings: "PART I" → "PART one"
#   - p_type: [h2]
#     p_class: [calibre1, calibre5]
#     type: roman
#     mode: inline
#     lang: en
#   # Stand-alone roman numeral paragraphs: "I" → "Section one"
#   - p_class: [calibreclass4]
#     type: roman
#     mode: whole
#     lang: en
#     add_prefix: "Section"

rules: []
"""


def load_global_numeral_rules(path: Path) -> list[NumeralRule]:
    if not path.exists():
        return []
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        rules_data = data.get("rules") or []
        rules = [NumeralRule.from_dict(r) for r in rules_data]
        logger.info(
            "NumeralRules: loaded %d rule(s) from %s", len(rules), path.name
        )
        return rules
    except Exception as exc:
        logger.warning("NumeralRules: could not read %s: %s", path, exc)
        return []


def create_numeral_rules_template(path: Path) -> None:
    if path.exists():
        return
    try:
        path.write_text(_NUMERAL_RULES_TEMPLATE, encoding="utf-8")
        logger.info("NumeralRules: created template at %s", path)
    except Exception as exc:
        logger.warning("NumeralRules: could not write template %s: %s", path, exc)
