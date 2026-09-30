from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_tts_batch():
    module_name = "_test_tts_batch_cli_module"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, _REPO_ROOT / "tts_batch.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _fake_pausable_book(job_id: int, chapters: list) -> SimpleNamespace:
    ch_objs = [
        SimpleNamespace(
            chapter_db_id=cid,
            chapter_id=f"ch_{i:04d}",
            sequence_number=f"{i:04d}",
            chapter_name=f"Ch {i}",
            synthesis_status=status,
            synthesis_batch_id=cid + 1000,
            paused=paused,
        )
        for i, (cid, status, paused) in enumerate(chapters, start=1)
    ]
    return SimpleNamespace(
        job_id=job_id, book_title=f"Book {job_id}", chapters=ch_objs,
        paused_count=sum(1 for c in ch_objs if c.paused),
    )


class TestExpandTreeSelection:
    def test_selecting_a_book_expands_to_all_its_chapters(self):
        from syntrive.cli.tree_selection import expand_tree_selection

        books = [_fake_pausable_book(1, [(10, "queued", False), (11, "combining", True)])]
        assert expand_tree_selection(books, [("book", 1)]) == [10, 11]

    def test_individual_chapters_union_and_dedupe_with_a_book(self):
        from syntrive.cli.tree_selection import expand_tree_selection

        books = [
            _fake_pausable_book(1, [(10, "queued", False), (11, "queued", False)]),
            _fake_pausable_book(2, [(20, "queued", False)]),
        ]
        got = expand_tree_selection(books, [("book", 1), ("chapter", 11), ("chapter", 20)])
        assert got == [10, 11, 20]

    def test_empty_selection_resolves_to_nothing(self):
        from syntrive.cli.tree_selection import expand_tree_selection

        assert expand_tree_selection([_fake_pausable_book(1, [(10, "queued", False)])], []) == []


class TestReviewTreeGroups:
    def test_one_group_per_book_and_a_paused_marker_on_the_leaf(self):
        tts_batch = _load_tts_batch()
        books = [_fake_pausable_book(1, [(10, "queued", False), (11, "combining", True)])]

        groups = tts_batch._review_tree_groups(books)
        assert len(groups) == 1
        assert groups[0].value == ("book", 1)
        assert [leaf.value for leaf in groups[0].leaves] == [("chapter", 10), ("chapter", 11)]
        assert "[PAUSED]" not in groups[0].leaves[0].title
        assert "[PAUSED]" in groups[0].leaves[1].title
        assert "[queued]" in groups[0].leaves[0].title
        assert "1 paused" in groups[0].title

    def test_large_book_opens_collapsed(self):
        tts_batch = _load_tts_batch()
        from syntrive.cli.collapsible_tree import DEFAULT_AUTO_COLLAPSE_THRESHOLD

        big = _fake_pausable_book(
            1, [(i, "queued", False) for i in range(DEFAULT_AUTO_COLLAPSE_THRESHOLD + 1)],
        )
        small = _fake_pausable_book(
            2, [(100 + i, "queued", False) for i in range(DEFAULT_AUTO_COLLAPSE_THRESHOLD)],
        )
        groups = tts_batch._review_tree_groups([big, small])
        assert groups[0].collapsed is True
        assert groups[1].collapsed is False
