from __future__ import annotations

import pytest
from bs4 import BeautifulSoup

from syntrive.adapters.text import sml


class TestMarkerDefinition:
    def test_durations_are_swapped_relative_to_the_pre_2026_09_20_definition(self):
        assert sml.DEFAULT_SML_DURATIONS_MS == {"break": 1000, "pause": 500}

    def test_strongest_prefers_break(self):
        assert sml.strongest(sml.PAUSE, sml.BREAK) == sml.BREAK
        assert sml.strongest(sml.BREAK, sml.PAUSE) == sml.BREAK
        assert sml.strongest(sml.PAUSE, sml.PAUSE) == sml.PAUSE

    def test_is_marker(self):
        assert sml.is_marker(sml.BREAK) and sml.is_marker(sml.PAUSE)
        assert not sml.is_marker("‡voice:1‡") and not sml.is_marker("text")


class TestTerminalPunctuation:
    @pytest.mark.parametrize("text", [
        "It ended.", "Really?", "No!", "Wait…", "End;", "Note:", "好。", "什么？", "他说：「走。」",
        'He said, "Go."', "(see above.)", "end.”",
    ])
    def test_terminated(self, text):
        assert sml.ends_with_terminal_punctuation(text)

    @pytest.mark.parametrize("text", ["Chapter One", "half a sentence,", "", "   ", "无标点", '"open quote'])
    def test_not_terminated(self, text):
        assert not sml.ends_with_terminal_punctuation(text)

    def test_sentence_period_by_language(self):
        assert sml.sentence_period("zh") == "。"
        assert sml.sentence_period("en") == "."
        assert sml.sentence_period("fr") == "."


class TestIsTitleLike:
    @pytest.mark.parametrize("text", ["Section one", "PART one", "THE END", "1949", "Chapter Twenty-One"])
    def test_titles_en(self, text):
        assert sml.is_title_like(text, "en")

    @pytest.mark.parametrize("text", [
        "", "and then,", "cut by layout", "Winston Smith, his chin nuzzled", "A sentence cut by the layout",
        "Section one,", "x" * 31,
    ])
    def test_non_titles_en(self, text):
        assert not sml.is_title_like(text, "en")

    def test_zh(self):
        assert sml.is_title_like("第一章", "zh")
        assert not sml.is_title_like("他只好拖着沉重的脚步一级一级地往上爬", "zh")
        assert not sml.is_title_like("第一章，", "zh")


class TestHardNewlineToBrRule:
    def _apply(self, html):
        from syntrive.adapters.htmlclean.rules.hard_newline_rule import HardNewlineToBrRule

        soup = BeautifulSoup(html, "html.parser")
        soup, removed = HardNewlineToBrRule().apply(soup)
        return str(soup), removed

    def test_slogan_paragraph_gets_br_between_lines_and_none_at_the_end(self):
        html, removed = self._apply(
            '<p class="c" id="cfs_165">\nWAR IS PEACE \nFREEDOM IS SLAVERY \nIGNORANCE IS STRENGTH \n</p>'
        )
        assert html == (
            '<p class="c" id="cfs_165">WAR IS PEACE<br/> FREEDOM IS SLAVERY<br/> IGNORANCE IS STRENGTH</p>'
        )
        assert removed > 0

    def test_newline_only_at_node_edges_is_left_alone(self):
        html, removed = self._apply("<p>\nOne plain paragraph.\n</p>")
        assert html == "<p>\nOne plain paragraph.\n</p>" and removed == 0

    def test_container_paragraph_and_pre_are_skipped(self):
        html, removed = self._apply("<pre><p>a\nb</p></pre><p>x\ny<div>d</div></p>")
        assert "<br" not in html and removed == 0
        assert "a\nb" in html and "x\ny" in html

    def test_inline_sibling_keeps_a_single_space(self):
        html, _ = self._apply("<p>first line\nsecond <b>bold</b> tail</p>")
        assert html == "<p>first line<br/> second <b>bold</b> tail</p>"

    def test_rule_is_registered_in_seed_and_fallback_chain(self):
        from syntrive.adapters.htmlclean.engine import _build_rule_chain
        from syntrive.db.seed import DEFAULT_CLEANING_RULES

        assert any(r["name"] == "hard_newline_to_br" and r["sort_order"] == 55 for r in DEFAULT_CLEANING_RULES)
        assert "hard_newline_to_br" in [r.name for r in _build_rule_chain("en")]

    def test_end_to_end_cleaning_then_extraction_yields_sentences(self):
        from syntrive.adapters.htmlclean.engine import HtmlCleaningEngine
        from syntrive.adapters.text.extractor import extract_tts_script_from_html

        cleaned, _ = HtmlCleaningEngine().run("<p>\nWAR IS PEACE\nFREEDOM IS SLAVERY\nIGNORANCE IS STRENGTH\n</p>", language="en")
        lines = extract_tts_script_from_html(cleaned, language="en").split("\n")
        assert lines[1] == "WAR IS PEACE. FREEDOM IS SLAVERY. IGNORANCE IS STRENGTH."
        assert lines[2] == sml.BREAK
