from __future__ import annotations

import logging
import re
from typing import Optional, TYPE_CHECKING

from bs4 import BeautifulSoup, NavigableString

if TYPE_CHECKING:
    from syntrive.adapters.html.para_id_mapper import NumeralRule

logger = logging.getLogger(__name__)

_PARA_TAGS = frozenset({"p", "li", "dt", "dd", "blockquote", "h1", "h2", "h3", "h4", "h5", "h6"})

_ROMAN_INLINE_RE = re.compile(r"(?<![A-Za-z])([IVXLCDM]+)(?![A-Za-z])")

_ROMAN_VALUES: tuple[tuple[str, int], ...] = (
    ("M", 1000), ("CM", 900), ("D", 500), ("CD", 400),
    ("C", 100),  ("XC", 90),  ("L", 50),  ("XL", 40),
    ("X", 10),   ("IX", 9),   ("V", 5),   ("IV", 4), ("I", 1),
)

_EN_WORD: dict[int, str] = {
    1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
    6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten",
    11: "eleven", 12: "twelve", 13: "thirteen", 14: "fourteen", 15: "fifteen",
    16: "sixteen", 17: "seventeen", 18: "eighteen", 19: "nineteen", 20: "twenty",
    21: "twenty-one", 22: "twenty-two", 23: "twenty-three", 24: "twenty-four",
    25: "twenty-five", 26: "twenty-six", 27: "twenty-seven", 28: "twenty-eight",
    29: "twenty-nine", 30: "thirty", 31: "thirty-one", 32: "thirty-two",
    33: "thirty-three", 34: "thirty-four", 35: "thirty-five", 36: "thirty-six",
    37: "thirty-seven", 38: "thirty-eight", 39: "thirty-nine",
}

_ZH_WORD: dict[int, str] = {
    1: "一", 2: "二", 3: "三", 4: "四", 5: "五",
    6: "六", 7: "七", 8: "八", 9: "九", 10: "十",
    11: "十一", 12: "十二", 13: "十三", 14: "十四", 15: "十五",
    16: "十六", 17: "十七", 18: "十八", 19: "十九", 20: "二十",
    21: "二十一", 22: "二十二", 23: "二十三", 24: "二十四", 25: "二十五",
    26: "二十六", 27: "二十七", 28: "二十八", 29: "二十九", 30: "三十",
    31: "三十一", 32: "三十二", 33: "三十三", 34: "三十四", 35: "三十五",
    36: "三十六", 37: "三十七", 38: "三十八", 39: "三十九",
}


class NumeralConversionRule:
    name = "numeral_conversion"

    def __init__(self, rules: list["NumeralRule"]) -> None:
        self._rules = rules

    def apply(self, soup: BeautifulSoup) -> tuple[BeautifulSoup, int]:
        if not self._rules:
            return soup, 0

        total_changed = 0

        for tag in soup.find_all(_PARA_TAGS):
            classes = tag.get("class") or []
            p_class: Optional[str] = classes[0] if classes else None
            p_type: str = tag.name

            matched_rule = self._find_rule(p_type, p_class)
            if matched_rule is None:
                continue

            original_text = tag.get_text()
            converted_text = _convert(original_text, matched_rule)

            if converted_text != original_text:
                _replace_tag_text(tag, converted_text)
                diff = abs(len(original_text) - len(converted_text))
                total_changed += diff
                logger.debug(
                    "NumeralConversionRule: %s.%s %r -> %r",
                    p_type, p_class, original_text.strip(), converted_text.strip(),
                )

        return soup, total_changed

    def _find_rule(self, p_type: str, p_class: Optional[str]) -> Optional["NumeralRule"]:
        for rule in self._rules:
            if rule.matches(p_type, p_class):
                return rule
        return None


def _convert(text: str, rule: "NumeralRule") -> str:
    if rule.numeral_type == "roman":
        if rule.mode == "whole":
            return _roman_whole(text, rule.lang, rule.add_prefix, rule.add_suffix)
        return _roman_inline(text, rule.lang)

    if rule.numeral_type == "arabic":
        if rule.mode == "whole":
            return _arabic_whole(text, rule.lang, rule.add_prefix, rule.add_suffix)
        logger.warning(
            "NumeralConversionRule: arabic inline mode is not supported; "
            "use text normalisation for in-text arabic number conversion"
        )
        return text

    return text


def _roman_whole(text: str, lang: str, add_prefix: str = "", add_suffix: str = "") -> str:
    stripped = text.strip()
    n = _roman_to_int(stripped)
    if n <= 0:
        logger.debug("NumeralConversionRule: not a valid roman numeral %r, skipping", stripped)
        return text
    word = _int_to_word(n, lang)
    result = _with_affixes(word, add_prefix, add_suffix)
    prefix_ws = text[: len(text) - len(text.lstrip())]
    suffix_ws = text[len(text.rstrip()):]
    return prefix_ws + result + suffix_ws


def _roman_inline(text: str, lang: str) -> str:
    def replace_token(m: re.Match) -> str:
        token = m.group(1)
        n = _roman_to_int(token)
        if n <= 0:
            logger.debug("NumeralConversionRule: skipping invalid roman token %r", token)
            return m.group(0)
        return _int_to_word(n, lang)

    return _ROMAN_INLINE_RE.sub(replace_token, text)


def _arabic_whole(text: str, lang: str, add_prefix: str = "", add_suffix: str = "") -> str:
    stripped = text.strip()
    if not stripped.lstrip("-+").isdigit():
        return text
    try:
        n = int(stripped)
    except ValueError:
        return text
    word = _int_to_word(n, lang)
    result = _with_affixes(word, add_prefix, add_suffix)
    prefix_ws = text[: len(text) - len(text.lstrip())]
    suffix_ws = text[len(text.rstrip()):]
    return prefix_ws + result + suffix_ws


def _with_affixes(word: str, add_prefix: str, add_suffix: str) -> str:
    parts = [p for p in (add_prefix, word, add_suffix) if p]
    return " ".join(parts)


def _is_valid_roman(s: str) -> bool:
    if not s:
        return False
    if s != s.upper() or not all(c in "IVXLCDM" for c in s):
        return False
    return _roman_to_int_unsafe(s) > 0


def _roman_to_int(s: str) -> int:
    s = s.upper().strip()
    if not s or not all(c in "IVXLCDM" for c in s):
        return 0
    return _roman_to_int_unsafe(s)


def _roman_to_int_unsafe(s: str) -> int:
    total = 0
    i = 0
    while i < len(s):
        matched = False
        for roman, value in _ROMAN_VALUES:
            if s[i: i + len(roman)] == roman:
                total += value
                i += len(roman)
                matched = True
                break
        if not matched:
            return 0
    return total


def _int_to_word(n: int, lang: str) -> str:
    if not lang.startswith("zh"):
        try:
            from num2words import num2words
            return num2words(n, lang=lang)
        except Exception as exc:
            logger.warning(
                "NumeralConversionRule: num2words failed n=%d lang=%s: %s — using fallback table",
                n, lang, exc,
            )
    table = _ZH_WORD if lang.startswith("zh") else _EN_WORD
    return table.get(n, str(n))


def _replace_tag_text(tag, new_text: str) -> None:
    children = list(tag.children)
    if len(children) == 1 and isinstance(children[0], NavigableString):
        children[0].replace_with(NavigableString(new_text))
    else:
        tag.clear()
        tag.append(NavigableString(new_text))
