from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace


class TestExtractTtsScriptFromHtml:
    def test_paragraph_emits_break_token(self) -> None:
        from syntrive.adapters.text.extractor import extract_tts_script_from_html
        from syntrive.adapters.text.sml import BREAK

        html = "<p>Hello world.</p>"
        out = extract_tts_script_from_html(html, language="en")
        lines = out.split("\n")
        assert lines[0] == "‡voice:0‡"
        assert "Hello world." in lines
        assert BREAK in lines

    def test_heading_emits_break_token_even_without_punctuation(self) -> None:
        from syntrive.adapters.text.extractor import extract_tts_script_from_html
        from syntrive.adapters.text.sml import BREAK, PAUSE

        out = extract_tts_script_from_html("<h1>Chapter One</h1>", language="en")
        assert "Chapter One" in out
        assert BREAK in out.split("\n")
        assert PAUSE not in out

    def test_every_paragraph_gets_its_own_break_token(self) -> None:
        from syntrive.adapters.text.extractor import extract_tts_script_from_html
        from syntrive.adapters.text.sml import BREAK

        html = "<p>First.</p><p>Second.</p><p>Third.</p>"
        lines = extract_tts_script_from_html(html, language="en").split("\n")
        assert lines == ["‡voice:0‡", "First.", BREAK, "Second.", BREAK, "Third.", BREAK]

    def test_ul_items_and_list_collapse_into_single_pauses(self) -> None:
        from syntrive.adapters.text.extractor import extract_tts_script_from_html
        from syntrive.adapters.text.sml import PAUSE

        html = "<ul><li>Item A</li><li>Item B</li></ul>"
        lines = extract_tts_script_from_html(html, language="en").split("\n")
        assert lines == ["‡voice:0‡", "Item A", PAUSE, "Item B", PAUSE]

    def test_hard_linebreaks_inside_one_p_become_one_paragraph_of_sentences(self) -> None:
        from syntrive.adapters.text.extractor import extract_tts_script_from_html
        from syntrive.adapters.text.sml import BREAK

        html = "<p>WAR IS PEACE \nFREEDOM IS SLAVERY \nIGNORANCE IS STRENGTH \n</p>"
        lines = extract_tts_script_from_html(html, language="en").split("\n")
        assert lines == [
            "‡voice:0‡",
            "WAR IS PEACE. FREEDOM IS SLAVERY. IGNORANCE IS STRENGTH.",
            BREAK,
        ]

    def test_br_inside_p_becomes_a_period_en_and_zh(self) -> None:
        from syntrive.adapters.text.extractor import extract_tts_script_from_html
        from syntrive.adapters.text.sml import BREAK

        en = extract_tts_script_from_html(
            "<p>WAR IS PEACE<br> FREEDOM IS SLAVERY<br> IGNORANCE IS STRENGTH</p>", language="en"
        ).split("\n")
        assert en == ["‡voice:0‡", "WAR IS PEACE. FREEDOM IS SLAVERY. IGNORANCE IS STRENGTH.", BREAK]

        zh = extract_tts_script_from_html("<p>战争即和平<br>自由即奴役！<br>无知即力量</p>", language="zh").split("\n")
        assert zh == ["‡voice:0‡", "战争即和平。自由即奴役！无知即力量。", BREAK]

    def test_div_wrapper_does_not_duplicate_nested_paragraph_text(self) -> None:
        from syntrive.adapters.text.extractor import extract_tts_script_from_html
        from syntrive.adapters.text.sml import BREAK

        html = "<div><p>Wrapped paragraph.</p></div>"
        lines = extract_tts_script_from_html(html, language="en").split("\n")
        assert lines.count("Wrapped paragraph.") == 1
        assert lines == ["‡voice:0‡", "Wrapped paragraph.", BREAK]

    def test_br_and_hr_emit_pause_that_collapses_into_a_preceding_break(self) -> None:
        from syntrive.adapters.text.extractor import extract_tts_script_from_html
        from syntrive.adapters.text.sml import BREAK, PAUSE

        out = extract_tts_script_from_html("<p>Before.</p><br/><hr/><p>After.</p>", language="en")
        lines = out.split("\n")
        assert lines == ["‡voice:0‡", "Before.", BREAK, "After.", BREAK]
        assert PAUSE not in lines

    def test_p_with_trailing_br_keeps_its_own_text(self) -> None:
        from syntrive.adapters.text.extractor import extract_tts_script_from_html
        from syntrive.adapters.text.sml import BREAK

        html = (
            '<p class="calibre13">启发式问题？<br class="calibre3"/></p>'
            '<p class="calibre13">想到垂死的海豚时，我的情绪波动有多大？<br class="calibre3"/></p>'
        )
        lines = extract_tts_script_from_html(html, language="zh").split("\n")
        assert lines == [
            "‡voice:0‡", "启发式问题？", BREAK,
            "想到垂死的海豚时，我的情绪波动有多大？", BREAK,
        ]

    def test_p_without_terminal_punctuation_gets_no_marker_and_is_reported(self) -> None:
        from syntrive.adapters.text.extractor import ReviewItem, extract_tts_script_from_html
        from syntrive.adapters.text.sml import BREAK

        sink: list = []
        html = '<p id="p1">A sentence cut by the layout</p><p>and its continuation.</p>'
        lines = extract_tts_script_from_html(html, language="en", review_sink=sink).split("\n")
        assert lines == ["‡voice:0‡", "A sentence cut by the layout", "and its continuation.", BREAK]
        assert sink == [ReviewItem(element_id="p1", preview="A sentence cut by the layout")]

    def test_closing_quote_and_ellipsis_count_as_terminal_punctuation(self) -> None:
        from syntrive.adapters.text.extractor import extract_tts_script_from_html

        sink: list = []
        html = '<p>He said, "Go."</p><p>Wait for it…</p><p>「好。」</p>'
        extract_tts_script_from_html(html, language="en", review_sink=sink)
        assert sink == []

    def test_heading_closes_a_preceding_unpunctuated_paragraph(self) -> None:
        from syntrive.adapters.text.extractor import extract_tts_script_from_html
        from syntrive.adapters.text.sml import BREAK

        lines = extract_tts_script_from_html(
            "<p>trailing fragment of a sentence</p><h2>Next Section</h2>", language="en"
        ).split("\n")
        assert lines == ["‡voice:0‡", "trailing fragment of a sentence", BREAK, "Next Section", BREAK]

    def test_title_like_unpunctuated_p_gets_a_period_and_a_break(self) -> None:
        from syntrive.adapters.text.extractor import extract_tts_script_from_html
        from syntrive.adapters.text.sml import BREAK

        sink: list = []
        html = "<p>Section one</p><p>It was a bright cold day in April.</p><p>THE END</p>"
        lines = extract_tts_script_from_html(html, language="en", review_sink=sink).split("\n")
        assert lines == [
            "‡voice:0‡", "Section one.", BREAK,
            "It was a bright cold day in April.", BREAK, "THE END.", BREAK,
        ]
        assert sink == []

    def test_title_like_zh_and_non_titles(self) -> None:
        from syntrive.adapters.text.extractor import extract_tts_script_from_html
        from syntrive.adapters.text.sml import BREAK

        sink: list = []
        zh = extract_tts_script_from_html("<p>第一章</p>", language="zh").split("\n")
        assert zh == ["‡voice:0‡", "第一章。", BREAK]
        html = "<p>Winston Smith, his chin nuzzled into</p><p>and then,</p><p>cut by layout</p>"
        lines = extract_tts_script_from_html(html, language="en", review_sink=sink).split("\n")
        assert BREAK not in lines and len(sink) == 3

    def test_html_tag_mode_assigns_voice_per_role(self) -> None:
        from syntrive.adapters.text.extractor import extract_tts_script_from_html

        html = (
            '<p class="normaltext"><b class="calibre3">Alice：</b>Hello there.</p>'
            '<p class="normaltext"><b class="calibre3">Bob：</b>Hi Alice.</p>'
        )
        out = extract_tts_script_from_html(html, language="en", extraction_mode="html_tag")
        lines = out.split("\n")
        assert "\u2021voice:1\u2021" in lines
        assert "\u2021voice:2\u2021" in lines
        assert "Hello there." in lines
        assert "Hi Alice." in lines

    def test_role_voice_map_persists_across_calls(self) -> None:
        from syntrive.adapters.text.extractor import extract_tts_script_from_html

        role_voice_map: dict[str, int] = {}
        html1 = '<p class="normaltext"><b class="calibre3">Alice：</b>Chapter 1 line.</p>'
        html2 = '<p class="normaltext"><b class="calibre3">Alice：</b>Chapter 2 line.</p>'
        extract_tts_script_from_html(
            html1, language="en", extraction_mode="html_tag", role_voice_map=role_voice_map
        )
        out2 = extract_tts_script_from_html(
            html2, language="en", extraction_mode="html_tag", role_voice_map=role_voice_map
        )
        assert role_voice_map["Alice"] == 1
        assert "\u2021voice:1\u2021" in out2.split("\n")

    def test_table_summary_elements_are_skipped(self) -> None:
        from syntrive.adapters.text.extractor import extract_tts_script_from_html

        html = '<p data-table-summary="true">Table summary text.</p><p>Real text.</p>'
        out = extract_tts_script_from_html(html, language="en")
        assert "Table summary text." not in out
        assert "Real text." in out


def _write_minimal_cleaned_toc(process_dir: Path, chapter_id: str = "ch_0001") -> None:
    from syntrive.adapters.epub.merged_toc_writer import MergedTocEntry, MergedTocMeta, MergedTocWriter

    cleaned = process_dir / "transcript_html" / "cleaned"
    cleaned.mkdir(parents=True, exist_ok=True)
    (cleaned / f"{chapter_id}.html").write_text(
        "<h1>Chapter One</h1><p>Narrator paragraph text.</p>", encoding="utf-8"
    )
    entry = MergedTocEntry(
        chapter_id=chapter_id,
        title="Chapter One",
        source_files=["part0001.html"],
        excluded=False,
        is_discard=False,
    )
    MergedTocWriter().write([entry], MergedTocMeta(), cleaned / "0_toc.md")


def _mock_job(process_dir: Path) -> SimpleNamespace:
    return SimpleNamespace(
        id=1,
        book_id=1,
        process_dir=str(process_dir),
        epub_path=str(process_dir / "book.epub"),
    )


class TestTextExtractionStageMultiFormat:
    def test_missing_cleaned_toc_returns_error(self, tmp_path: Path) -> None:
        from syntrive.pipeline.text_extraction_stage import TextExtractionStage

        stage = TextExtractionStage(job=_mock_job(tmp_path), db_path=tmp_path / "db.sqlite")
        result = stage.run_extract()
        assert result.success is False
        assert "cleaned/0_toc.md not found" in (result.error or "")

    def test_default_formats_write_raw_and_tts_script_only(self, tmp_path: Path) -> None:
        from syntrive.pipeline.text_extraction_stage import TextExtractionStage

        _write_minimal_cleaned_toc(tmp_path)
        stage = TextExtractionStage(job=_mock_job(tmp_path), db_path=tmp_path / "db.sqlite")
        result = stage.run_extract()

        assert result.success is True
        assert (tmp_path / "transcript_text" / "raw" / "ch_0001.txt").exists()
        assert (tmp_path / "transcript_text" / "raw" / "0_toc.md").exists()
        assert (tmp_path / "transcript_text" / "tts_script" / "ch_0001.txt").exists()
        assert (tmp_path / "transcript_text" / "tts_script" / "0_toc.md").exists()
        assert result.artifacts.get("raw") == 1
        assert result.artifacts.get("tts_script") == 1

    def test_raw_output_has_no_sml_tokens(self, tmp_path: Path) -> None:
        from syntrive.pipeline.text_extraction_stage import TextExtractionStage
        from lib.models import TTS_SML

        _write_minimal_cleaned_toc(tmp_path)
        stage = TextExtractionStage(job=_mock_job(tmp_path), db_path=tmp_path / "db.sqlite")
        stage.run_extract(formats={"raw"}, primary_format="raw")

        raw_text = (tmp_path / "transcript_text" / "raw" / "ch_0001.txt").read_text(encoding="utf-8")
        assert TTS_SML["break"] not in raw_text
        assert TTS_SML["pause"] not in raw_text
        assert "Narrator paragraph text." in raw_text

    def test_tts_script_output_has_sml_tokens(self, tmp_path: Path) -> None:
        from syntrive.pipeline.text_extraction_stage import TextExtractionStage
        from lib.models import TTS_SML

        _write_minimal_cleaned_toc(tmp_path)
        stage = TextExtractionStage(job=_mock_job(tmp_path), db_path=tmp_path / "db.sqlite")
        stage.run_extract(formats={"tts_script"}, primary_format="tts_script")

        script_text = (
            tmp_path / "transcript_text" / "tts_script" / "ch_0001.txt"
        ).read_text(encoding="utf-8")
        assert TTS_SML["pause"] not in script_text
        assert TTS_SML["break"] in script_text

    def test_unknown_primary_falls_back_to_selected_default(self, tmp_path: Path) -> None:
        from syntrive.pipeline.text_extraction_stage import TextExtractionStage

        _write_minimal_cleaned_toc(tmp_path)
        stage = TextExtractionStage(job=_mock_job(tmp_path), db_path=tmp_path / "db.sqlite")
        result = stage.run_extract(formats={"raw"}, primary_format="tts_script")
        assert result.success is True
        assert "raw" in (result.notes or "")


class TestTextExtractionAuditTable:
    def test_raw_audit_table_counts_plain_text_characters(self, tmp_path: Path) -> None:
        from syntrive.adapters.toc_audit import file_char_count
        from syntrive.pipeline.text_extraction_stage import TextExtractionStage

        _write_minimal_cleaned_toc(tmp_path)
        stage = TextExtractionStage(job=_mock_job(tmp_path), db_path=tmp_path / "db.sqlite")
        stage.run_extract(formats={"raw"}, primary_format="raw")

        raw_dir = tmp_path / "transcript_text" / "raw"
        toc_text = (raw_dir / "0_toc.md").read_text(encoding="utf-8")
        assert "## File Size & Character Count (audit)" in toc_text
        assert (
            "| chapter | file | sequence_number | chapter_number | size_bytes | char_count |"
            in toc_text
        )
        expected = file_char_count(raw_dir / "ch_0001.txt")
        size = (raw_dir / "ch_0001.txt").stat().st_size
        assert f"| ch_0001.txt | 0001 | 0001 | {size} | {expected} |" in toc_text

    def test_tts_script_audit_table_strips_sml_tokens(self, tmp_path: Path) -> None:
        from syntrive.adapters.toc_audit import file_char_count
        from syntrive.pipeline.text_extraction_stage import TextExtractionStage

        _write_minimal_cleaned_toc(tmp_path)
        stage = TextExtractionStage(job=_mock_job(tmp_path), db_path=tmp_path / "db.sqlite")
        stage.run_extract(formats={"tts_script"}, primary_format="tts_script")

        script_dir = tmp_path / "transcript_text" / "tts_script"
        script_path = script_dir / "ch_0001.txt"
        toc_text = (script_dir / "0_toc.md").read_text(encoding="utf-8")
        expected_stripped = file_char_count(script_path, strip_sml=True)
        expected_unstripped = file_char_count(script_path, strip_sml=False)
        assert expected_stripped < expected_unstripped
        assert f"| {script_path.stat().st_size} | {expected_stripped} |" in toc_text


def _write_cleaned_toc_with_gaps(process_dir: Path) -> None:
    from syntrive.adapters.epub.merged_toc_writer import MergedTocEntry, MergedTocMeta, MergedTocWriter

    cleaned = process_dir / "transcript_html" / "cleaned"
    cleaned.mkdir(parents=True, exist_ok=True)
    chapters = [
        ("ch_0001", "Chapter One", "First chapter text."),
        ("ch_0003", "Chapter Two", "Second chapter text."),
        ("ch_0004", "Chapter Three", "Third chapter text."),
    ]
    entries = []
    for chapter_id, title, text in chapters:
        (cleaned / f"{chapter_id}.html").write_text(
            f"<h1>{title}</h1><p>{text}</p>", encoding="utf-8"
        )
        entries.append(MergedTocEntry(
            chapter_id=chapter_id,
            title=title,
            source_files=["part0001.html"],
            excluded=False,
            is_discard=False,
            sequence_number=chapter_id.split("_", 1)[1],
        ))
    MergedTocWriter().write(entries, MergedTocMeta(), cleaned / "0_toc.md")


class TestTextExtractionRenumbering:
    def test_output_filenames_are_gap_free(self, tmp_path: Path) -> None:
        from syntrive.pipeline.text_extraction_stage import TextExtractionStage

        _write_cleaned_toc_with_gaps(tmp_path)
        stage = TextExtractionStage(job=_mock_job(tmp_path), db_path=tmp_path / "db.sqlite")
        result = stage.run_extract(formats={"raw"}, primary_format="raw")

        assert result.success is True
        raw_dir = tmp_path / "transcript_text" / "raw"
        assert (raw_dir / "ch_0001.txt").exists()
        assert (raw_dir / "ch_0002.txt").exists()
        assert (raw_dir / "ch_0003.txt").exists()
        assert not (raw_dir / "ch_0004.txt").exists()
        assert "First chapter text." in (raw_dir / "ch_0001.txt").read_text(encoding="utf-8")
        assert "Second chapter text." in (raw_dir / "ch_0002.txt").read_text(encoding="utf-8")
        assert "Third chapter text." in (raw_dir / "ch_0003.txt").read_text(encoding="utf-8")

    def test_toc_sequence_and_chapter_number_are_contiguous(self, tmp_path: Path) -> None:
        from syntrive.pipeline.text_extraction_stage import TextExtractionStage

        _write_cleaned_toc_with_gaps(tmp_path)
        stage = TextExtractionStage(job=_mock_job(tmp_path), db_path=tmp_path / "db.sqlite")
        stage.run_extract(formats={"raw"}, primary_format="raw")

        toc_text = (tmp_path / "transcript_text" / "raw" / "0_toc.md").read_text(encoding="utf-8")
        assert "## [Chapter One](ch_0001.txt)" in toc_text
        assert "## [Chapter Two](ch_0002.txt)" in toc_text
        assert "## [Chapter Three](ch_0003.txt)" in toc_text
        sequence_numbers = re.findall(r"- sequence_number: (\d{4})", toc_text)
        assert sequence_numbers == ["0001", "0002", "0003"]
        chapter_numbers = re.findall(r"- chapter_number: (\d{4})", toc_text)
        assert chapter_numbers == ["0001", "0002", "0003"]

    def test_all_formats_share_the_same_renumbering(self, tmp_path: Path) -> None:
        from syntrive.pipeline.text_extraction_stage import TextExtractionStage

        _write_cleaned_toc_with_gaps(tmp_path)
        stage = TextExtractionStage(job=_mock_job(tmp_path), db_path=tmp_path / "db.sqlite")
        stage.run_extract(formats={"raw", "tts_script"}, primary_format="raw")

        for fmt in ("raw", "tts_script"):
            fmt_dir = tmp_path / "transcript_text" / fmt
            assert (fmt_dir / "ch_0001.txt").exists()
            assert (fmt_dir / "ch_0002.txt").exists()
            assert (fmt_dir / "ch_0003.txt").exists()

    def test_rerun_removes_stale_files_from_previous_numbering(self, tmp_path: Path) -> None:
        from syntrive.pipeline.text_extraction_stage import TextExtractionStage

        _write_cleaned_toc_with_gaps(tmp_path)
        stage = TextExtractionStage(job=_mock_job(tmp_path), db_path=tmp_path / "db.sqlite")
        stage.run_extract(formats={"raw"}, primary_format="raw")
        raw_dir = tmp_path / "transcript_text" / "raw"
        assert (raw_dir / "ch_0003.txt").exists()

        (raw_dir / "ch_0099.txt").write_text("stale leftover", encoding="utf-8")

        from syntrive.adapters.epub.merged_toc_writer import MergedTocEntry, MergedTocMeta, MergedTocWriter
        cleaned = tmp_path / "transcript_html" / "cleaned"
        entries = [
            MergedTocEntry(chapter_id="ch_0001", title="Chapter One",
                            source_files=["part0001.html"], excluded=False, is_discard=False),
            MergedTocEntry(chapter_id="ch_0003", title="Chapter Two",
                            source_files=["part0001.html"], excluded=False, is_discard=False),
        ]
        MergedTocWriter().write(entries, MergedTocMeta(), cleaned / "0_toc.md")

        stage.run_extract(formats={"raw"}, primary_format="raw")

        assert not (raw_dir / "ch_0003.txt").exists()
        assert not (raw_dir / "ch_0099.txt").exists()
        assert (raw_dir / "ch_0001.txt").exists()
        assert (raw_dir / "ch_0002.txt").exists()


def _write_cleaned_toc_with_front_matter_and_volume(process_dir: Path) -> None:
    from syntrive.adapters.epub.merged_toc_writer import MergedTocEntry, MergedTocMeta, MergedTocWriter

    cleaned = process_dir / "transcript_html" / "cleaned"
    cleaned.mkdir(parents=True, exist_ok=True)
    specs = [
        ("ch_0001", "Preface", "", "", "Preface text."),
        ("ch_0002", "Chapter 1", "Volume One", "0001", "Chapter one text."),
        ("ch_0003", "Chapter 2", "Volume One", "0001", "Chapter two text."),
    ]
    entries = []
    for chapter_id, title, vol_name, vol_num, text in specs:
        (cleaned / f"{chapter_id}.html").write_text(
            f"<h1>{title}</h1><p>{text}</p>", encoding="utf-8"
        )
        entries.append(MergedTocEntry(
            chapter_id=chapter_id,
            title=title,
            source_files=["part0001.html"],
            excluded=False,
            is_discard=False,
            volume_name=vol_name,
            volume_number=vol_num,
        ))
    MergedTocWriter().write(entries, MergedTocMeta(), cleaned / "0_toc.md")


class TestTextExtractionChapterNumberVsSequenceNumber:
    def test_non_chapter_front_matter_has_empty_chapter_number(self, tmp_path: Path) -> None:
        from syntrive.pipeline.text_extraction_stage import TextExtractionStage

        _write_cleaned_toc_with_front_matter_and_volume(tmp_path)
        stage = TextExtractionStage(job=_mock_job(tmp_path), db_path=tmp_path / "db.sqlite")
        stage.run_extract(formats={"raw"}, primary_format="raw")

        toc_text = (tmp_path / "transcript_text" / "raw" / "0_toc.md").read_text(encoding="utf-8")
        preface_block = toc_text.split("## [Preface]")[1].split("## [Chapter 1]")[0]
        assert "  - sequence_number: 0001" in preface_block
        assert "  - chapter_number: \n" in preface_block

    def test_real_chapters_count_from_0001_ignoring_front_matter(self, tmp_path: Path) -> None:
        from syntrive.pipeline.text_extraction_stage import TextExtractionStage

        _write_cleaned_toc_with_front_matter_and_volume(tmp_path)
        stage = TextExtractionStage(job=_mock_job(tmp_path), db_path=tmp_path / "db.sqlite")
        stage.run_extract(formats={"raw"}, primary_format="raw")

        toc_text = (tmp_path / "transcript_text" / "raw" / "0_toc.md").read_text(encoding="utf-8")
        ch1_block = toc_text.split("## [Chapter 1]")[1].split("## [Chapter 2]")[0]
        assert "  - sequence_number: 0002" in ch1_block
        assert "  - chapter_number: 0001" in ch1_block

        ch2_block = toc_text.split("## [Chapter 2]")[1]
        assert "  - sequence_number: 0003" in ch2_block
        assert "  - chapter_number: 0002" in ch2_block

    def test_audit_table_mirrors_toc_heading_numbers(self, tmp_path: Path) -> None:
        from syntrive.pipeline.text_extraction_stage import TextExtractionStage

        _write_cleaned_toc_with_front_matter_and_volume(tmp_path)
        stage = TextExtractionStage(job=_mock_job(tmp_path), db_path=tmp_path / "db.sqlite")
        stage.run_extract(formats={"raw"}, primary_format="raw")

        toc_text = (tmp_path / "transcript_text" / "raw" / "0_toc.md").read_text(encoding="utf-8")
        audit_block = toc_text.split("## File Size & Character Count (audit)")[1]
        assert (
            "| volume | chapter | file | sequence_number | chapter_number "
            "| volume_number | size_bytes | char_count |"
        ) in audit_block
        audit_rows = [ln for ln in audit_block.splitlines() if "](ch_000" in ln]
        assert audit_rows[0].startswith(
            "|  | [Preface](ch_0001.txt) | ch_0001.txt | 0001 |  |  |"
        )
        assert audit_rows[1].startswith(
            "| Volume One | [Chapter 1](ch_0002.txt) | ch_0002.txt | 0002 | 0001 | 0001 |"
        )

    def test_audit_table_omits_volume_columns_for_flat_book(self, tmp_path: Path) -> None:
        from syntrive.pipeline.text_extraction_stage import TextExtractionStage

        _write_cleaned_toc_with_gaps(tmp_path)
        stage = TextExtractionStage(job=_mock_job(tmp_path), db_path=tmp_path / "db.sqlite")
        stage.run_extract(formats={"raw"}, primary_format="raw")

        toc_text = (tmp_path / "transcript_text" / "raw" / "0_toc.md").read_text(encoding="utf-8")
        audit_block = toc_text.split("## File Size & Character Count (audit)")[1]
        assert (
            "| chapter | file | sequence_number | chapter_number | size_bytes | char_count |"
            in audit_block
        )
        assert "volume" not in audit_block

    def test_flat_book_chapter_number_still_mirrors_sequence_number(self, tmp_path: Path) -> None:
        from syntrive.pipeline.text_extraction_stage import TextExtractionStage

        _write_cleaned_toc_with_gaps(tmp_path)
        stage = TextExtractionStage(job=_mock_job(tmp_path), db_path=tmp_path / "db.sqlite")
        stage.run_extract(formats={"raw"}, primary_format="raw")

        toc_text = (tmp_path / "transcript_text" / "raw" / "0_toc.md").read_text(encoding="utf-8")
        assert re.findall(r"- sequence_number: (\d{4})", toc_text) == ["0001", "0002", "0003"]
        assert re.findall(r"- chapter_number: (\d{4})", toc_text) == ["0001", "0002", "0003"]


class TestTextTocSourceField:
    def test_source_field_matches_cleaned_html_filename(self, tmp_path: Path) -> None:
        from syntrive.pipeline.text_extraction_stage import TextExtractionStage

        _write_cleaned_toc_with_gaps(tmp_path)
        stage = TextExtractionStage(job=_mock_job(tmp_path), db_path=tmp_path / "db.sqlite")
        stage.run_extract(formats={"raw"}, primary_format="raw")

        toc_text = (tmp_path / "transcript_text" / "raw" / "0_toc.md").read_text(encoding="utf-8")
        assert re.findall(r"- source: (\S+)", toc_text) == [
            "ch_0001.html", "ch_0003.html", "ch_0004.html",
        ]


class TestCountTranscriptLines:
    def test_counts_every_non_blank_line(self, tmp_path: Path) -> None:
        from syntrive.pipeline.text_extraction_stage import _count_transcript_lines

        f = tmp_path / "ch_0001.txt"
        f.write_text(
            "\u2021voice:0\u2021\n"
            "Chapter One\n"
            "\u2021pause\u2021\n"
            "First sentence.\n"
            "\u2021break\u2021\n"
            "\n"
            "Second sentence.\n"
            "\u2021break\u2021\n",
            encoding="utf-8",
        )
        assert _count_transcript_lines(f) == 7

    def test_raw_format_file_has_no_sml_tokens_to_exclude(self, tmp_path: Path) -> None:
        from syntrive.pipeline.text_extraction_stage import _count_transcript_lines

        f = tmp_path / "ch_0001.txt"
        f.write_text("Chapter One\nFirst sentence.\nSecond sentence.\n", encoding="utf-8")
        assert _count_transcript_lines(f) == 3

    def test_missing_file_returns_zero(self, tmp_path: Path) -> None:
        from syntrive.pipeline.text_extraction_stage import _count_transcript_lines

        assert _count_transcript_lines(tmp_path / "does_not_exist.txt") == 0


def _insert_job_and_chapter_row(db_path: Path, job_id: int, chapter_id: str) -> None:
    from syntrive.db.session import get_db_session
    from syntrive.db.models import Book, Job, TranscriptChapter

    with get_db_session(db_path) as db:
        book = Book(title="Test Book")
        db.add(book)
        db.flush()
        job = Job(
            id=job_id,
            book_id=book.id,
            process_dir=str(db_path.parent),
            epub_path=str(db_path.parent / "book.epub"),
        )
        db.add(job)
        db.flush()
        db.add(TranscriptChapter(job_id=job_id, chapter_id=chapter_id))


class TestPersistTranscriptPathsSetsTranscriptLines:
    def test_updates_transcript_lines_for_tts_script_selection(self, tmp_path: Path) -> None:
        from syntrive.pipeline.text_extraction_stage import TextExtractionStage
        from syntrive.db.session import get_db_session
        from syntrive.db.models import TranscriptChapter

        _write_minimal_cleaned_toc(tmp_path, chapter_id="ch_0001")
        db_path = tmp_path / "db.sqlite"
        _insert_job_and_chapter_row(db_path, job_id=1, chapter_id="ch_0001")

        stage = TextExtractionStage(job=_mock_job(tmp_path), db_path=db_path)
        stage.run_extract(formats={"tts_script"}, primary_format="tts_script")
        result = stage.persist_transcript_paths("tts_script")

        assert result["success"] is True
        assert result["updated"] == 1

        script_file = tmp_path / "transcript_text" / "tts_script" / "ch_0001.txt"
        expected_lines = sum(
            1 for ln in script_file.read_text(encoding="utf-8").splitlines() if ln.strip()
        )

        with get_db_session(db_path) as db:
            record = (
                db.query(TranscriptChapter)
                .filter_by(job_id=1, chapter_id="ch_0001")
                .first()
            )
            assert record.transcript_path == "transcript_text/tts_script/ch_0001.txt"
            assert record.transcript_lines == expected_lines
            assert record.transcript_lines > 0
            assert record.sequence_number == "0001"
            assert record.chapter_name == "Chapter One"
            assert record.chapter_number == "0001"
            assert record.volume == ""
            assert record.volume_number == ""

    def test_updates_transcript_lines_for_raw_selection(self, tmp_path: Path) -> None:
        from syntrive.pipeline.text_extraction_stage import TextExtractionStage
        from syntrive.db.session import get_db_session
        from syntrive.db.models import TranscriptChapter

        _write_minimal_cleaned_toc(tmp_path, chapter_id="ch_0001")
        db_path = tmp_path / "db.sqlite"
        _insert_job_and_chapter_row(db_path, job_id=1, chapter_id="ch_0001")

        stage = TextExtractionStage(job=_mock_job(tmp_path), db_path=db_path)
        stage.run_extract(formats={"raw"}, primary_format="raw")
        stage.persist_transcript_paths("raw")

        raw_file = tmp_path / "transcript_text" / "raw" / "ch_0001.txt"
        expected_lines = sum(
            1 for ln in raw_file.read_text(encoding="utf-8").splitlines() if ln.strip()
        )

        with get_db_session(db_path) as db:
            record = (
                db.query(TranscriptChapter)
                .filter_by(job_id=1, chapter_id="ch_0001")
                .first()
            )
            assert record.transcript_lines == expected_lines


class TestPersistTranscriptPathsVolumeFields:
    def test_volume_fields_populated_for_volume_split_book(self, tmp_path: Path) -> None:
        from syntrive.adapters.epub.merged_toc_writer import (
            MergedTocEntry, MergedTocMeta, MergedTocWriter,
        )
        from syntrive.pipeline.text_extraction_stage import TextExtractionStage
        from syntrive.db.session import get_db_session
        from syntrive.db.models import TranscriptChapter

        cleaned = tmp_path / "transcript_html" / "cleaned"
        cleaned.mkdir(parents=True, exist_ok=True)
        (cleaned / "ch_0002.html").write_text(
            "<h2>Volume One</h2>", encoding="utf-8"
        )
        (cleaned / "ch_0003.html").write_text(
            "<h1>Chapter One</h1><p>Narrator paragraph text.</p>", encoding="utf-8"
        )
        entries = [
            MergedTocEntry(
                chapter_id="ch_0002", title="Volume One", source_files=["part0001.html"],
                excluded=True, is_discard=False, volume_name="", volume_number="",
            ),
            MergedTocEntry(
                chapter_id="ch_0003", title="Chapter One", source_files=["part0002.html"],
                excluded=False, is_discard=False,
                volume_name="Volume One", volume_number="0001",
            ),
        ]
        MergedTocWriter().write(entries, MergedTocMeta(), cleaned / "0_toc.md")

        db_path = tmp_path / "db.sqlite"
        _insert_job_and_chapter_row(db_path, job_id=1, chapter_id="ch_0003")

        stage = TextExtractionStage(job=_mock_job(tmp_path), db_path=db_path)
        stage.run_extract(formats={"raw"}, primary_format="raw")
        result = stage.persist_transcript_paths("raw")

        assert result["success"] is True
        assert result["updated"] == 1

        with get_db_session(db_path) as db:
            record = (
                db.query(TranscriptChapter)
                .filter_by(job_id=1, chapter_id="ch_0003")
                .first()
            )
            assert record.volume == "Volume One"
            assert record.volume_number == "0001"
            assert record.sequence_number == "0001"
            assert record.chapter_name == "Chapter One"
            assert record.chapter_number == "0001"

    def test_missing_selected_format_toc_fails_gracefully(self, tmp_path: Path) -> None:
        from syntrive.pipeline.text_extraction_stage import TextExtractionStage

        _write_minimal_cleaned_toc(tmp_path, chapter_id="ch_0001")
        db_path = tmp_path / "db.sqlite"
        _insert_job_and_chapter_row(db_path, job_id=1, chapter_id="ch_0001")

        stage = TextExtractionStage(job=_mock_job(tmp_path), db_path=db_path)
        stage.run_extract(formats={"raw"}, primary_format="raw")
        result = stage.persist_transcript_paths("tts_script")

        assert result["success"] is False
        assert "tts_script/0_toc.md" in (result["error"] or "")


class TestTextTocReader:
    def test_missing_file_returns_empty_list(self, tmp_path: Path) -> None:
        from syntrive.adapters.text.toc_writer import TextTocReader

        assert TextTocReader().read(tmp_path / "does_not_exist.md") == []

    def test_round_trips_writer_output(self, tmp_path: Path) -> None:
        from syntrive.adapters.epub.merged_toc_writer import MergedTocEntry
        from syntrive.adapters.text.toc_writer import TextTocReader, TextTocWriter

        entries = [
            MergedTocEntry(
                chapter_id="ch_0002", title="Chapter One", source_files=[],
                excluded=False, is_discard=False,
                volume_name="Volume One", volume_number="0001",
                chapter_number="0001", sequence_number="0001",
            ),
        ]
        toc_path = tmp_path / "0_toc.md"
        TextTocWriter().write(entries, toc_path, ext="txt")

        parsed = TextTocReader().read(toc_path)
        assert len(parsed) == 1
        entry = parsed[0]
        assert entry.chapter_id == "ch_0002"
        assert entry.title == "Chapter One"
        assert entry.volume_name == "Volume One"
        assert entry.volume_number == "0001"
        assert entry.chapter_number == "0001"
        assert entry.sequence_number == "0001"
        assert entry.excluded is False

    def test_stops_before_audit_table(self, tmp_path: Path) -> None:
        from syntrive.adapters.epub.merged_toc_writer import MergedTocEntry
        from syntrive.adapters.text.toc_writer import TextTocReader, TextTocWriter

        entries = [
            MergedTocEntry(
                chapter_id="ch_0001", title="Chapter One", source_files=[],
                excluded=False, is_discard=False,
                chapter_number="0001", sequence_number="0001",
            ),
        ]
        toc_path = tmp_path / "0_toc.md"
        toc_path.parent.mkdir(parents=True, exist_ok=True)
        (toc_path.parent / "ch_0001.txt").write_text("Some text.", encoding="utf-8")
        TextTocWriter().write(entries, toc_path, ext="txt")

        parsed = TextTocReader().read(toc_path)
        assert len(parsed) == 1




class TestReviewReportForUnpunctuatedParagraphs:
    def _run(self, tmp_path: Path, html: str):
        from syntrive.pipeline.text_extraction_stage import TextExtractionStage

        _write_minimal_cleaned_toc(tmp_path)
        (tmp_path / "transcript_html" / "cleaned" / "ch_0001.html").write_text(html, encoding="utf-8")
        stage = TextExtractionStage(job=_mock_job(tmp_path), db_path=tmp_path / "db.sqlite")
        return stage.run_extract(formats={"tts_script"}, primary_format="tts_script")

    def test_report_written_and_noted_when_a_paragraph_lacks_punctuation(self, tmp_path: Path) -> None:
        result = self._run(tmp_path, '<p id="cfs_9">cut by the layout</p><p>Fine.</p>')

        report = tmp_path / "transcript_text" / "tts_script" / "review_unpunctuated.md"
        assert report.exists()
        text = report.read_text(encoding="utf-8")
        assert "ch_0001" in text and "cfs_9" in text and "cut by the layout" in text
        assert "REVIEW NEEDED: 1 paragraph(s)" in (result.notes or "")

    def test_no_report_and_stale_report_removed_when_clean(self, tmp_path: Path) -> None:
        report = tmp_path / "transcript_text" / "tts_script" / "review_unpunctuated.md"
        report.parent.mkdir(parents=True)
        report.write_text("stale", encoding="utf-8")

        result = self._run(tmp_path, "<p>All good.</p>")

        assert not report.exists()
        assert "REVIEW NEEDED" not in (result.notes or "")
