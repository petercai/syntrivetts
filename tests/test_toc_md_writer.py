from __future__ import annotations

from syntrive.adapters.epub.toc_md_writer import TocMdWriter


class TestTocMdWriterHeadings:
    def test_h1_uses_book_title(self, tmp_path):
        out = tmp_path / "0_toc.md"
        TocMdWriter().write(
            [("Chapter 1", "ch1.html", 2)], [], out, book_title="思考,快与慢"
        )
        assert out.read_text(encoding="utf-8").splitlines()[0] == "# 思考,快与慢"

    def test_h1_falls_back_to_toc_when_no_book_title(self, tmp_path):
        out = tmp_path / "0_toc.md"
        TocMdWriter().write([("Chapter 1", "ch1.html", 2)], [], out)
        assert out.read_text(encoding="utf-8").splitlines()[0] == "# TOC"

    def test_entries_written_at_their_own_heading_level(self, tmp_path):
        out = tmp_path / "0_toc.md"
        entries = [
            ("序言", "part0004.html", 2),
            ("第一部分 系统1，系统2", "part0008.html", 2),
            ("第1章 一张愤怒的脸和一道乘法题", "part0009.html", 3),
        ]
        TocMdWriter().write(entries, [], out, book_title="思考,快与慢")
        heading_lines = [
            ln for ln in out.read_text(encoding="utf-8").splitlines()
            if ln.startswith("#") and "[" in ln
        ]
        assert heading_lines == [
            "## [序言](part0004.html)",
            "## [第一部分 系统1，系统2](part0008.html)",
            "### [第1章 一张愤怒的脸和一道乘法题](part0009.html)",
        ]

    def test_heading_level_clamped_to_h6(self, tmp_path):
        out = tmp_path / "0_toc.md"
        TocMdWriter().write([("Deep", "deep.html", 9)], [], out)
        heading_line = next(
            ln for ln in out.read_text(encoding="utf-8").splitlines() if "[Deep]" in ln
        )
        assert heading_line.startswith("###### [Deep]")

    def test_orphan_section_unaffected(self, tmp_path):
        out = tmp_path / "0_toc.md"
        TocMdWriter().write(
            [("Chapter 1", "ch1.html", 2)], ["cover.html"], out, book_title="Book"
        )
        lines = out.read_text(encoding="utf-8").splitlines()
        assert "- [cover.html](cover.html)" in lines


class TestTocMdWriterAuditTable:
    def test_audit_table_appended_with_size_and_char_count(self, tmp_path):
        (tmp_path / "ch1.html").write_text(
            "<html><body><p>你好世界</p></body></html>", encoding="utf-8"
        )
        out = tmp_path / "0_toc.md"
        TocMdWriter().write([("Chapter 1", "ch1.html", 2)], [], out, book_title="Book")
        text = out.read_text(encoding="utf-8")

        assert "## File Size & Character Count (audit)" in text
        assert "| [Chapter 1](ch1.html) | ch1.html |" in text
        size = (tmp_path / "ch1.html").stat().st_size
        assert f"| [Chapter 1](ch1.html) | ch1.html | {size} | 4 |" in text

    def test_no_audit_table_when_no_toc_entries(self, tmp_path):
        out = tmp_path / "0_toc.md"
        TocMdWriter().write([], [], out, book_title="Book")
        assert "File Size & Character Count" not in out.read_text(encoding="utf-8")

