from __future__ import annotations

from syntrive.adapters.toc_audit import (
    AUDIT_SECTION_HEADING,
    AuditTocMeta,
    build_audit_table_lines,
    file_char_count,
    file_size_bytes,
)


class TestFileSizeBytes:
    def test_returns_actual_size_for_existing_file(self, tmp_path):
        p = tmp_path / "a.txt"
        p.write_text("hello", encoding="utf-8")
        assert file_size_bytes(p) == 5

    def test_returns_zero_for_missing_file(self, tmp_path):
        assert file_size_bytes(tmp_path / "missing.txt") == 0


class TestFileCharCount:
    def test_plain_text_counts_all_characters(self, tmp_path):
        p = tmp_path / "a.txt"
        p.write_text("你好，世界", encoding="utf-8")
        assert file_char_count(p) == 5

    def test_strip_html_extracts_visible_text_only(self, tmp_path):
        p = tmp_path / "a.html"
        p.write_text("<html><body><p>你好</p><p>世界</p></body></html>", encoding="utf-8")
        assert file_char_count(p, strip_html=True) == 4

    def test_strip_sml_removes_tokens_before_counting(self, tmp_path):
        p = tmp_path / "a.txt"
        p.write_text("‡voice:0‡你好‡break‡世界‡pause‡", encoding="utf-8")
        assert file_char_count(p, strip_sml=True) == 4

    def test_returns_zero_for_missing_file(self, tmp_path):
        assert file_char_count(tmp_path / "missing.txt") == 0


class TestBuildAuditTableLines:
    def test_empty_rows_returns_empty_list(self):
        assert build_audit_table_lines([]) == []

    def test_populated_rows_produce_table_with_expected_shape(self, tmp_path):
        p = tmp_path / "ch_0001.txt"
        p.write_text("你好世界", encoding="utf-8")
        lines = build_audit_table_lines([("Chapter One", "ch_0001.txt", p)])

        assert lines[0] == ""
        assert lines[1] == "---"
        assert lines[3] == AUDIT_SECTION_HEADING
        assert lines[5] == "| chapter | file | size_bytes | char_count |"
        assert lines[6] == "|---|---|---|---|"
        data_row = lines[7]
        assert data_row.startswith("| [Chapter One](ch_0001.txt) | ch_0001.txt |")
        assert str(p.stat().st_size) in data_row
        assert data_row.rstrip().endswith("| 4 |")

    def test_title_brackets_are_escaped(self, tmp_path):
        p = tmp_path / "ch_0001.txt"
        p.write_text("x", encoding="utf-8")
        lines = build_audit_table_lines([("Weird [Title]", "ch_0001.txt", p)])
        data_row = next(ln for ln in lines if ln.startswith("|") and "Weird" in ln)
        assert "\\[Title\\]" in data_row

    def test_toc_meta_inserts_number_columns_after_file(self, tmp_path):
        p1 = tmp_path / "ch_0001.txt"
        p1.write_text("你好", encoding="utf-8")
        p2 = tmp_path / "ch_0002.txt"
        p2.write_text("世界世界", encoding="utf-8")
        lines = build_audit_table_lines(
            [("Preface", "ch_0001.txt", p1), ("Chapter 1", "ch_0002.txt", p2)],
            toc_meta=[
                AuditTocMeta("0001", ""),
                AuditTocMeta("0002", "0001"),
            ],
        )
        assert lines[5] == (
            "| chapter | file | sequence_number | chapter_number | size_bytes | char_count |"
        )
        assert lines[6] == "|---|---|---|---|---|---|"
        assert lines[7].startswith("| [Preface](ch_0001.txt) | ch_0001.txt | 0001 |  |")
        assert lines[8].startswith("| [Chapter 1](ch_0002.txt) | ch_0002.txt | 0002 | 0001 |")

    def test_toc_meta_volume_columns_only_when_book_has_volumes(self, tmp_path):
        p1 = tmp_path / "ch_0001.txt"
        p1.write_text("你好", encoding="utf-8")
        p2 = tmp_path / "ch_0002.txt"
        p2.write_text("世界", encoding="utf-8")
        lines = build_audit_table_lines(
            [("Preface", "ch_0001.txt", p1), ("Chapter 1", "ch_0002.txt", p2)],
            toc_meta=[
                AuditTocMeta("0001", "", "", ""),
                AuditTocMeta("0002", "0001", "Volume One", "0001"),
            ],
        )
        assert lines[5] == (
            "| volume | chapter | file | sequence_number | chapter_number "
            "| volume_number | size_bytes | char_count |"
        )
        assert lines[6] == "|---|---|---|---|---|---|---|---|"
        assert lines[7].startswith("|  | [Preface](ch_0001.txt) | ch_0001.txt | 0001 |  |  |")
        assert lines[8].startswith(
            "| Volume One | [Chapter 1](ch_0002.txt) | ch_0002.txt | 0002 | 0001 | 0001 |"
        )

    def test_toc_meta_no_volume_columns_for_flat_book(self, tmp_path):
        p = tmp_path / "ch_0001.txt"
        p.write_text("你好", encoding="utf-8")
        lines = build_audit_table_lines(
            [("Chapter One", "ch_0001.txt", p)],
            toc_meta=[AuditTocMeta("0001", "0001", "", "")],
        )
        assert lines[5] == (
            "| chapter | file | sequence_number | chapter_number | size_bytes | char_count |"
        )

    def test_toc_meta_length_mismatch_is_ignored(self, tmp_path):
        p = tmp_path / "ch_0001.txt"
        p.write_text("x", encoding="utf-8")
        lines = build_audit_table_lines(
            [("Chapter One", "ch_0001.txt", p)],
            toc_meta=[AuditTocMeta("0001", "0001"), AuditTocMeta("0002", "0002")],
        )
        assert lines[5] == "| chapter | file | size_bytes | char_count |"
