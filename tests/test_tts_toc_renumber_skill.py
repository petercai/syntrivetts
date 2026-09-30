from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from syntrive.adapters.epub.merged_toc_writer import MergedTocEntry
from syntrive.adapters.text.toc_writer import TextTocWriter

_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "syntrive" / "shared" / "skills" / "tts-toc-renumber" / "script" / "renumber.py"
)


def _load():
    spec = importlib.util.spec_from_file_location("_ttc_renumber", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


rn = _load()


def _entry(seq: str, chapter_number: str, title: str, *, vol="", voln="") -> MergedTocEntry:
    cid = f"ch_{seq}"
    return MergedTocEntry(
        chapter_id=cid, title=title, source_files=[f"{cid}.html"],
        excluded=False, is_discard=False, volume_name=vol, volume_number=voln,
        sequence_number=seq, chapter_number=chapter_number,
    )


def _write_toc(dirpath: Path, entries: list[MergedTocEntry]) -> Path:
    dirpath.mkdir(parents=True, exist_ok=True)
    for e in entries:
        (dirpath / f"ch_{e.sequence_number}.txt").write_text("body text\n", encoding="utf-8")
    out = dirpath / "0_toc.md"
    TextTocWriter().write(entries, out, ext="txt")
    return out


class TestPushChapterNumberDownRule:
    def test_shifts_column_down_one_row(self):
        entries = [
            rn.Entry("ch_0001.txt", "0001", "0001"),
            rn.Entry("ch_0002.txt", "0002", "0002"),
            rn.Entry("ch_0003.txt", "0003", "0003"),
        ]
        out = rn._rule_push_chapter_number_down(entries)
        assert out == {"chapter_number": ["", "0001", "0002"]}

    def test_only_touches_chapter_number(self):
        entries = [rn.Entry("ch_0001.txt", "0001", "0001", "Vol", "0001")]
        assert set(rn._rule_push_chapter_number_down(entries)) == {"chapter_number"}


class TestProcessFileFlatBook:
    @pytest.fixture
    def toc(self, tmp_path: Path) -> Path:
        return _write_toc(
            tmp_path / "transcript_text" / "raw",
            [
                _entry("0001", "0001", "Prologue"),
                _entry("0002", "0002", "Chapter 1"),
                _entry("0003", "0003", "Chapter 2"),
            ],
        )

    def test_dry_run_writes_nothing(self, toc: Path):
        before = toc.read_text(encoding="utf-8")
        res = rn.process_file(
            toc, "push-chapter-number-down", dry_run=True, force=False, today="2026-09-03"
        )
        assert res.written is False
        assert toc.read_text(encoding="utf-8") == before
        assert res.heading_changes == 3 and res.audit_changes == 3

    def test_apply_updates_both_surfaces(self, toc: Path):
        rn.process_file(
            toc, "push-chapter-number-down", dry_run=False, force=False, today="2026-09-03"
        )
        text = toc.read_text(encoding="utf-8")

        head_lines, audit_lines = rn._split_sections(text)
        heading = [(e.filename, e.chapter_number) for e in rn.parse_entries(head_lines)]
        assert heading == [("ch_0001.txt", ""), ("ch_0002.txt", "0001"), ("ch_0003.txt", "0002")]

        h_idx, fd, stop = rn._audit_table_bounds(audit_lines)
        ci = rn._col_index(audit_lines[h_idx], "chapter_number")
        fi = rn._col_index(audit_lines[h_idx], "file")
        audit = [
            (rn._row_cells(audit_lines[r])[fi + 1].strip(),
             rn._row_cells(audit_lines[r])[ci + 1].strip())
            for r in range(fd, stop)
        ]
        assert audit == heading

        assert "- sequence_number: 0001" in text and "- sequence_number: 0003" in text
        assert text.count("| ch_000") == 3

    def test_marker_inserted_and_reapply_refused(self, toc: Path):
        rn.process_file(toc, "push-chapter-number-down", dry_run=False, force=False, today="2026-09-03")
        assert "<!-- toc-renumber applied: push-chapter-number-down (2026-09-03) -->" \
            in toc.read_text(encoding="utf-8")
        with pytest.raises(rn.TocFormatError, match="already applied"):
            rn.process_file(toc, "push-chapter-number-down", dry_run=False, force=False, today="2026-09-03")

    def test_reapply_with_force_shifts_again(self, toc: Path):
        for _ in range(2):
            rn.process_file(toc, "push-chapter-number-down", dry_run=False, force=True, today="2026-09-03")
        head_lines, _ = rn._split_sections(toc.read_text(encoding="utf-8"))
        nums = [e.chapter_number for e in rn.parse_entries(head_lines)]
        assert nums == ["", "", "0001"]

    def test_only_expected_lines_change(self, toc: Path):
        before = toc.read_text(encoding="utf-8").split("\n")
        rn.process_file(toc, "push-chapter-number-down", dry_run=False, force=False, today="2026-09-03")
        after = toc.read_text(encoding="utf-8").split("\n")
        assert len(after) == len(before) + 1
        import difflib
        for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=before, b=after, autojunk=False).get_opcodes():
            if tag == "equal":
                continue
            for ln in before[i1:i2] + after[j1:j2]:
                s = ln.strip()
                assert (
                    s.startswith("- chapter_number:")
                    or s.startswith("<!-- toc-renumber applied:")
                    or (s.startswith("|") and "](ch_" in s)
                ), f"unexpected changed line: {ln!r}"


class TestProcessFileVolumeBook:
    def test_volume_audit_shape_chapter_number_still_shifts(self, tmp_path: Path):
        toc = _write_toc(
            tmp_path / "transcript_text" / "tts_script",
            [
                _entry("0001", "", "Preface"),
                _entry("0002", "0001", "V1C1", vol="Volume One", voln="0001"),
                _entry("0003", "0002", "V1C2", vol="Volume One", voln="0001"),
            ],
        )
        rn.process_file(toc, "push-chapter-number-down", dry_run=False, force=False, today="2026-09-03")
        text = toc.read_text(encoding="utf-8")
        head_lines, audit_lines = rn._split_sections(text)

        h_idx, fd, stop = rn._audit_table_bounds(audit_lines)
        hdr = [c.strip() for c in rn._row_cells(audit_lines[h_idx])][1:-1]
        assert hdr[0] == "volume" and "volume_number" in hdr

        heading = [(e.filename, e.chapter_number) for e in rn.parse_entries(head_lines)]
        assert heading == [("ch_0001.txt", ""), ("ch_0002.txt", ""), ("ch_0003.txt", "0001")]

        ci = rn._col_index(audit_lines[h_idx], "chapter_number")
        fi = rn._col_index(audit_lines[h_idx], "file")
        audit = [
            (rn._row_cells(audit_lines[r])[fi + 1].strip(),
             rn._row_cells(audit_lines[r])[ci + 1].strip())
            for r in range(fd, stop)
        ]
        assert audit == heading
        assert text.count("| Volume One |") == 2


class TestScopeGuard:
    @pytest.mark.parametrize("rel", [
        "transcript_html/cleaned/0_toc.md",
        "transcript_text/other/0_toc.md",
        "transcript_text/raw/other.md",
        "0_toc.md",
    ])
    def test_out_of_scope_paths_refused(self, tmp_path: Path, rel: str):
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("# TOC\n", encoding="utf-8")
        with pytest.raises(rn.TocFormatError, match="scope"):
            rn.process_file(p, "push-chapter-number-down", dry_run=True, force=False, today="x")

    @pytest.mark.parametrize("sub", ["raw", "tts_script"])
    def test_in_scope_paths_accepted(self, tmp_path: Path, sub: str):
        toc = _write_toc(
            tmp_path / "transcript_text" / sub,
            [_entry("0001", "0001", "C1"), _entry("0002", "0002", "C2")],
        )
        res = rn.process_file(
            toc, "push-chapter-number-down", dry_run=True, force=False, today="x"
        )
        assert res.entries == 2
