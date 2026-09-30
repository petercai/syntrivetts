from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_COMMON_SCRIPT_DIR = _REPO_ROOT / "syntrive" / "shared" / "skills" / "tts-text-normalize-common" / "script"

sys.path.insert(0, str(_COMMON_SCRIPT_DIR))
import common  # noqa: E402


def _load_profile(language: str) -> ModuleType:
    module_name = f"_test_tts_text_normalize_profile_{language}"
    if module_name in sys.modules:
        return sys.modules[module_name]
    script_dir = _REPO_ROOT / "syntrive" / "shared" / "skills" / f"tts-text-normalize-{language}" / "script"
    spec = importlib.util.spec_from_file_location(module_name, script_dir / "_profile.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(params=["en", "cn"])
def profile(request):
    return _load_profile(request.param).PROFILE


class TestGenericProcessBlockVoiceMarkerPlacement:
    def test_voice_marker_is_followed_by_text_on_the_same_line(self, profile):
        output, _ = common.generic_process_block(
            ["‡voice:0‡", "hello"], profile.min_units, profile.max_units, profile
        )

        assert len(output) == 1
        assert output[0].startswith("‡voice:0‡")
        assert "hello" in output[0]

    def test_orphaned_voice_marker_at_a_self_inserted_break_is_deferred_not_dropped(self, profile):
        output, _ = common.generic_process_block(
            ["‡voice:5‡", "‡break‡", "real text after break"],
            profile.min_units, profile.max_units, profile,
        )

        assert output == ["‡voice:5‡", "‡break‡", "real text after break"]

    def test_orphaned_voice_marker_persists_across_a_real_block_boundary(self, profile):
        block1, _ = common.generic_process_block(["‡voice:5‡"], profile.min_units, profile.max_units, profile)
        block2, _ = common.generic_process_block(
            ["real text after break"], profile.min_units, profile.max_units, profile
        )

        result = common.eliminate_marker_only_lines(block1 + ["‡break‡"] + block2, profile)

        assert result == [f"‡voice:5‡{profile.join_sep}real text after break"]

    def test_consecutive_voice_switches_only_the_last_one_survives(self, profile):
        output, _ = common.generic_process_block(
            ["‡voice:1‡", "‡voice:2‡", "text under voice 2"],
            profile.min_units, profile.max_units, profile,
        )

        assert len(output) == 1
        assert "‡voice:1‡" not in output[0]
        assert output[0].startswith("‡voice:2‡")

    def test_no_output_line_is_ever_a_bare_voice_marker(self, profile):
        output, _ = common.generic_process_block(
            ["‡voice:0‡", "‡voice:1‡", "‡voice:2‡", "final text"],
            profile.min_units, profile.max_units, profile,
        )

        for line in output:
            assert not common.VOICE_RE.match(line.strip())


class TestGenericProcessBlockPausePlacement:
    def test_inline_pause_between_two_phrases_forces_a_line_break_after_it(self, profile):
        output, _ = common.generic_process_block(
            ["‡voice:0‡", "PART one ‡pause‡ Section one"],
            profile.min_units, profile.max_units, profile,
        )

        assert len(output) == 2
        assert output[0].endswith("‡pause‡")
        assert "Section one" not in output[0]
        assert output[1].strip() == "Section one"

    def test_no_output_line_contains_pause_or_break_anywhere_but_the_end(self, profile):
        output, _ = common.generic_process_block(
            ["‡voice:0‡", "first phrase ‡pause‡ second phrase ‡pause‡ third phrase"],
            profile.min_units, profile.max_units, profile,
        )

        for line in output:
            stripped = line.strip()
            for match in common.SML_RE.finditer(stripped):
                if not common.VOICE_RE.match(match.group(0)):
                    assert match.end() == len(stripped), f"marker not at line end: {line!r}"

    def test_normal_paragraph_break_boundary_is_unaffected(self, profile):
        output, _ = common.generic_process_block(
            ["first paragraph text", "‡break‡", "second paragraph text"],
            profile.min_units, profile.max_units, profile,
        )

        assert output == ["first paragraph text", "‡break‡", "second paragraph text"]

    def test_leading_pause_with_nothing_accumulated_in_this_call_becomes_its_own_line(self, profile):
        output, _ = common.generic_process_block(
            ["‡pause‡ Do the audaciousness and hubris that drive him"],
            profile.min_units, profile.max_units, profile,
        )

        assert output[0] == "‡pause‡"
        assert "Do the audaciousness" not in output[0]
        assert any("Do the audaciousness" in line for line in output[1:])


class TestScanQualityContractViolations:
    def test_flags_a_bare_voice_only_line(self, profile):
        r = common.scan_quality(["‡voice:0‡", "some real text here"], profile)

        assert len(r["voice_only"]) == 1
        assert r["voice_only"][0][0] == 1

    def test_flags_a_misplaced_inline_pause(self, profile):
        r = common.scan_quality(["PART one ‡pause‡ Section one"], profile)

        assert len(r["misplaced_marker"]) == 1

    def test_flags_a_voice_marker_not_at_line_start(self, profile):
        r = common.scan_quality(["some text ‡voice:1‡ more text"], profile)

        assert len(r["misplaced_marker"]) == 1

    def test_correctly_normalized_lines_pass_clean(self, profile):
        r = common.scan_quality(["‡voice:0‡ PART one ‡pause‡", "Section one"], profile)

        assert r["voice_only"] == []
        assert r["misplaced_marker"] == []
        assert r["unterminated"] == []

    def test_line_ending_with_pause_exempt_from_under_range(self, profile):
        r = common.scan_quality(["short heading ‡pause‡", "next line in the same block"], profile)

        assert r["unterminated"] == []

    def test_a_trailing_pause_at_line_end_is_not_flagged(self, profile):
        r = common.scan_quality(["some real spoken text ‡pause‡"], profile)

        assert r["misplaced_marker"] == []

    def test_flags_a_standalone_bare_pause_line(self, profile):
        r = common.scan_quality(["‡pause‡", "some real spoken text"], profile)

        assert r["voice_only"] == []
        assert r["misplaced_marker"] == []
        assert len(r["marker_only_line"]) == 1
        assert r["marker_only_line"][0][0] == 1

    def test_flags_a_line_with_two_markers_stacked_and_no_text(self, profile):
        combined = f"‡break‡{profile.join_sep}‡pause‡"
        r = common.scan_quality([combined], profile)

        assert len(r["marker_only_line"]) == 1

    def test_flags_a_standalone_bare_break_line(self, profile):
        r = common.scan_quality(["first paragraph", "‡break‡", "second paragraph"], profile)

        assert len(r["marker_only_line"]) == 1
        assert r["marker_only_line"][0] == (2, "‡break‡")

    def test_a_short_line_ending_in_break_is_still_block_final_exempt(self, profile):
        short_final_line = f"short block-final line{profile.join_sep}‡break‡"
        r = common.scan_quality([short_final_line, "next block's first line"], profile)

        assert r["unterminated"] == []

    def test_a_fully_eliminated_file_has_zero_marker_only_lines(self, profile):
        raw = [
            "first paragraph", "‡break‡", "‡voice:1‡", "‡break‡",
            "second paragraph ‡pause‡", "third paragraph",
        ]
        cleaned = common.eliminate_marker_only_lines(raw, profile)

        r = common.scan_quality(cleaned, profile)
        assert r["voice_only"] == []
        assert r["misplaced_marker"] == []
        assert r["marker_only_line"] == []


class TestEnsureLineEndMarkers:
    def test_appends_pause_to_unmarked_lines_only(self, profile):
        sep = profile.join_sep
        result = common.ensure_line_end_markers(
            ["one.", "two. ‡pause‡", "three. ‡break‡", "‡voice:1‡ four."], profile
        )
        assert result == [
            "one." + sep + "‡pause‡",
            "two. ‡pause‡",
            "three. ‡break‡",
            "‡voice:1‡ four." + sep + "‡pause‡",
        ]

    def test_idempotent(self, profile):
        once = common.ensure_line_end_markers(["one.", "two."], profile)
        assert common.ensure_line_end_markers(once, profile) == once


class TestStrongestMarkerWins:
    def test_dedupe_collapses_mixed_run_to_break(self):
        lines, removed = common.dedupe_consecutive_tokens(
            ["a.", "‡pause‡", "‡break‡", "‡pause‡", "b."]
        )
        assert lines == ["a.", "‡break‡", "b."]
        assert removed == 2

    def test_dedupe_keeps_a_lone_pause(self):
        lines, removed = common.dedupe_consecutive_tokens(["a.", "‡pause‡", "b."])
        assert lines == ["a.", "‡pause‡", "b."] and removed == 0

    def test_attach_marker_upgrades_a_pause_tail_to_break(self):
        assert common.attach_marker("text ‡pause‡", "‡break‡", " ") == "text ‡break‡"

    def test_attach_marker_never_downgrades_a_break_tail(self):
        assert common.attach_marker("text ‡break‡", "‡pause‡", " ") == "text ‡break‡"

    def test_attach_marker_plain_append(self):
        assert common.attach_marker("text", "‡pause‡", " ") == "text ‡pause‡"


class TestEliminateMarkerOnlyLines:
    def test_merges_a_standalone_break_onto_the_previous_line(self, profile):
        result = common.eliminate_marker_only_lines(
            ["first paragraph text", "‡break‡", "second paragraph text"], profile
        )

        assert result == [f"first paragraph text{profile.join_sep}‡break‡", "second paragraph text"]

    def test_merges_a_standalone_pause_onto_the_previous_line(self, profile):
        result = common.eliminate_marker_only_lines(["some text", "‡pause‡", "more text"], profile)

        assert result == [f"some text{profile.join_sep}‡pause‡", "more text"]

    def test_drops_a_leading_marker_with_nothing_before_it(self, profile):
        result = common.eliminate_marker_only_lines(["‡break‡", "first real text"], profile)

        assert result == ["first real text"]

    def test_attaches_a_standalone_voice_to_the_front_of_the_next_real_line(self, profile):
        result = common.eliminate_marker_only_lines(["‡voice:2‡", "spoken text"], profile)

        assert result == [f"‡voice:2‡{profile.join_sep}spoken text"]

    def test_consecutive_standalone_voice_lines_collapse_to_the_latest(self, profile):
        result = common.eliminate_marker_only_lines(["‡voice:1‡", "‡voice:2‡", "spoken text"], profile)

        assert result == [f"‡voice:2‡{profile.join_sep}spoken text"]

    def test_drops_a_trailing_voice_with_no_text_ever_following(self, profile):
        result = common.eliminate_marker_only_lines(["spoken text", "‡voice:3‡"], profile)

        assert result == ["spoken text"]

    def test_two_markers_stacked_on_one_physical_line_both_attach_correctly(self, profile):
        combined = f"‡break‡{profile.join_sep}‡pause‡"
        result = common.eliminate_marker_only_lines(["real text before", combined, "real text after"], profile)

        assert result == [f"real text before{profile.join_sep}‡break‡", "real text after"]

    def test_no_output_line_is_ever_marker_only(self, profile):
        raw = [
            "‡voice:0‡", "opening narration", "‡break‡", "‡voice:1‡", "‡break‡",
            "‡voice:2‡", "dialogue line", "‡pause‡", "‡break‡", "closing narration",
        ]
        result = common.eliminate_marker_only_lines(raw, profile)

        for line in result:
            assert not common.SML_RE.fullmatch(line.strip()), f"marker-only line survived: {line!r}"


def _is_cn(profile) -> bool:
    return profile.label.endswith("-cn")


def _sentence(profile, units: int, end: str | None = None) -> str:
    if _is_cn(profile):
        return "汉" * units + (end if end is not None else "。")
    return " ".join(["Abcde"] + ["abcde"] * (units // 5 - 1)) + (end if end is not None else ".")


def _clause_sentence(profile, clauses: int, units_each: int) -> str:
    comma = "，" if _is_cn(profile) else ","
    parts = [_sentence(profile, units_each, comma) for _ in range(clauses)]
    parts[-1] = parts[-1][:-1] + ("。" if _is_cn(profile) else ".")
    return profile.join_sep.join(parts)


def _plain(profile, lines) -> str:
    return "".join(common.strip_sml("".join(lines)).split())


class TestSentenceFirstSegmentation:
    def _run(self, profile, block):
        return common.generic_process_block(block, profile.min_units, profile.max_units, profile)

    def test_unpunctuated_lines_are_joined_into_a_whole_sentence(self, profile):
        first, second = ("他走到了门口", "看见了一个人。") if _is_cn(profile) else ("The quick brown fox", "jumped over the dog.")
        output, _ = self._run(profile, [first, second])

        assert output == [f"{first}{profile.join_sep}{second}"]

    def test_sentences_are_packed_until_the_soft_target_then_cut_at_a_sentence_end(self, profile):
        n = 28 if _is_cn(profile) else 45
        sentences = [_sentence(profile, n) for _ in range(5)]
        output, breaks = self._run(profile, sentences)

        assert breaks == 0
        assert output == [
            profile.join_sep.join(sentences[:3]),
            profile.join_sep.join(sentences[3:]),
        ]
        assert profile.count_units(output[0]) > profile.max_units

    def test_every_line_ends_after_punctuation_and_no_text_is_lost(self, profile):
        n = 28 if _is_cn(profile) else 45
        sentences = [_sentence(profile, n) for _ in range(7)]
        output, _ = self._run(profile, sentences)

        assert _plain(profile, output) == _plain(profile, sentences)
        for line in output:
            assert line.rstrip()[-1] in ".。", f"line cut inside a sentence: {line!r}"

    def test_one_over_long_sentence_is_split_only_at_clause_punctuation(self, profile):
        units = 42 if _is_cn(profile) else 70
        sentence = _clause_sentence(profile, 3, units)
        output, breaks = self._run(profile, [sentence])

        assert breaks >= 1
        assert _plain(profile, output) == _plain(profile, [sentence])
        fragments = [ln for ln in output if ln != common.PAUSE_STR]
        assert len(fragments) >= 2
        for frag in fragments:
            assert frag.rstrip()[-1] in ",，.。", f"fragment cut inside a clause: {frag!r}"

    def test_over_long_sentence_without_any_punctuation_is_kept_whole(self, profile):
        long_no_punct = _sentence(profile, profile.max_units * 3, "").rstrip()
        output, breaks = self._run(profile, [long_no_punct])

        assert breaks == 0
        assert output == [long_no_punct]

    def test_break_boundary_still_stops_merging_even_without_terminal_punctuation(self, profile):
        output, _ = self._run(profile, ["Heading", common.BREAK_STR, "Body text."])

        assert output == ["Heading", common.BREAK_STR, "Body text."]

    def test_english_abbreviation_and_lowercase_continuation_do_not_end_a_sentence(self):
        en = _load_profile("en").PROFILE
        text = 'Mr. Smith said "Stop!" he cried. Then it ended.'

        cuts = en.sentence_ends(text)

        assert [text[:c].strip() for c in cuts] == ['Mr. Smith said "Stop!" he cried.']

    def test_chinese_fullwidth_closing_paren_is_absorbed_into_the_preceding_sentence_end(self):
        cn = _load_profile("cn").PROFILE
        text = "你最低要多少钱？（你不能购买疫苗。）正如你所料，这笔钱数目不小。"
        paren_close = text.index("）")

        cuts = cn.sentence_ends(text)

        assert paren_close not in cuts, "cut must not land BEFORE ）, splitting it onto its own fragment"
        assert paren_close + 1 in cuts, "sentence end must swallow the trailing ）, not stop before it"

    def test_colon_and_semicolon_count_as_sentence_ends(self, profile):
        text = "甲乙丙：丁戊己；庚辛。" if _is_cn(profile) else "First part: second part; third part."

        assert len(profile.sentence_ends(text)) == 2

    def test_normalizing_a_file_twice_is_idempotent(self, profile, tmp_path):
        n = 28 if _is_cn(profile) else 45
        lines = ["Heading", common.BREAK_STR] if not _is_cn(profile) else ["章节标题", common.BREAK_STR]
        lines += [_sentence(profile, n)[:-1] for _ in range(4)]
        lines += [_sentence(profile, n) for _ in range(3)]
        f = tmp_path / "ch_0001.txt"
        f.write_text("\n".join(lines) + "\n", encoding="utf-8")

        common.generic_normalize_file(f, profile)
        once = f.read_text(encoding="utf-8")
        common.generic_normalize_file(f, profile)

        assert f.read_text(encoding="utf-8") == once


class TestScanQualitySentenceContract:
    def test_flags_a_line_cut_inside_a_sentence(self, profile):
        cut = "in an effort to" if not _is_cn(profile) else "他努力地想要"
        r = common.scan_quality([cut, "the rest of it."], profile)

        assert r["unterminated"] == [(1, cut)]

    def test_marker_terminated_heading_and_punctuated_lines_pass(self, profile):
        r = common.scan_quality(["Prologue ‡break‡", "A full sentence.", "Next line."], profile)

        assert r["unterminated"] == []


class TestBreakPauseContract:
    def test_length_cuts_end_in_pause_paragraph_end_stays_break_and_second_pass_is_a_no_op(
        self, profile, tmp_path
    ):
        units = 42 if _is_cn(profile) else 70
        long_sentence = _clause_sentence(profile, 3, units)
        script = tmp_path / "ch_0001.txt"
        script.write_text(
            "\n".join(["‡voice:0‡", long_sentence, "‡break‡", "Heading" if not _is_cn(profile) else "标题", "‡break‡"])
            + "\n",
            encoding="utf-8",
        )

        assert common.generic_normalize_file(script, profile) is not None
        lines = script.read_text(encoding="utf-8").splitlines()

        text_lines = [ln for ln in lines if common.is_text_line(ln)]
        assert len(text_lines) >= 3
        assert all(ln.rstrip().endswith(("‡pause‡", "‡break‡")) for ln in text_lines)
        assert text_lines[0].rstrip().endswith("‡pause‡")
        assert text_lines[-2].rstrip().endswith("‡break‡")
        assert text_lines[-1].rstrip().endswith("‡break‡")
        assert not any("‡break‡" in ln and "‡pause‡" in ln for ln in lines)
        assert common.generic_normalize_file(script, profile) is None

    def test_marker_less_line_is_joined_with_the_next_and_a_heading_break_stays_hard(self, profile):
        first = "cut by the layout" if not _is_cn(profile) else "被排版切断的半句"
        second = "and it continues." if not _is_cn(profile) else "这里继续。"
        output, _ = common.generic_process_block(
            [first, second], profile.min_units, profile.max_units, profile
        )
        assert len(output) == 1 and first in output[0] and second in output[0]


class TestMinSynthFloor:
    def _short_and_ellipsis(self, profile):
        if _is_cn(profile):
            return "嗯。", "……"
        return "OK.", "..."

    def test_a_fragment_below_the_floor_is_padded_with_the_language_filler(self, profile):
        short, _ = self._short_and_ellipsis(profile)
        line = profile.join_sep.join(["‡voice:2‡", short, "‡break‡"])
        new_line, dropped, padded = common.enforce_min_synth_floor(line, profile)
        assert padded is True and dropped is False
        assert new_line.startswith(f"‡voice:2‡{profile.join_sep}{short}")
        assert new_line.rstrip().endswith("‡break‡")
        plain = common.strip_sml(new_line).strip()
        assert len(plain) >= profile.min_synth_chars

    def test_a_pure_punctuation_fragment_becomes_a_bare_pause_not_text(self, profile):
        _, ellipsis_only = self._short_and_ellipsis(profile)
        line = profile.join_sep.join(["‡voice:2‡", ellipsis_only, "‡break‡"])
        new_line, dropped, padded = common.enforce_min_synth_floor(line, profile)
        assert dropped is True and padded is False
        assert ellipsis_only not in new_line
        assert "‡pause‡" in new_line

    def test_a_fragment_already_at_the_floor_is_left_unchanged(self, profile):
        long_enough = ("是的的确如此。" if _is_cn(profile) else "Yes, that is quite correct.")
        line = profile.join_sep.join(["‡voice:1‡", long_enough, "‡break‡"])
        new_line, dropped, padded = common.enforce_min_synth_floor(line, profile)
        assert dropped is False and padded is False
        assert new_line == line

    def test_bare_break_boundary_line_is_a_no_op(self, profile):
        new_line, dropped, padded = common.enforce_min_synth_floor("‡break‡", profile)
        assert new_line == "‡break‡" and not dropped and not padded

    def test_second_pass_on_already_padded_text_does_not_grow_further(self, profile):
        short, _ = self._short_and_ellipsis(profile)
        line = profile.join_sep.join(["‡voice:2‡", short, "‡break‡"])
        once, _, _ = common.enforce_min_synth_floor(line, profile)
        twice, dropped2, padded2 = common.enforce_min_synth_floor(once, profile)
        assert twice == once and not dropped2 and not padded2

    def test_full_file_normalize_pads_a_standalone_short_paragraph(self, profile, tmp_path):
        short, ellipsis_only = self._short_and_ellipsis(profile)
        heading = "Introduction" if not _is_cn(profile) else "引言"
        script = tmp_path / "ch_0001.txt"
        script.write_text(
            "\n".join([
                heading, "‡break‡",
                profile.join_sep.join(["‡voice:2‡", short]), "‡break‡",
                profile.join_sep.join(["‡voice:2‡", ellipsis_only]), "‡break‡",
            ])
            + "\n",
            encoding="utf-8",
        )
        assert common.generic_normalize_file(script, profile) is not None
        lines = script.read_text(encoding="utf-8").splitlines()
        r = common.scan_quality(lines, profile)
        assert r["synth_unsafe"] == []
        assert common.generic_normalize_file(script, profile) is None

    def test_scan_quality_flags_a_synth_unsafe_line_before_the_rule_runs(self, profile):
        short, _ = self._short_and_ellipsis(profile)
        lines = [profile.join_sep.join(["‡voice:2‡", short, "‡break‡"])]
        r = common.scan_quality(lines, profile)
        assert len(r["synth_unsafe"]) == 1
        assert r["synth_unsafe"][0][2] == "too_short"
