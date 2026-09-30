from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_tts_task():
    module_name = "_test_tts_task_cli_module"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, _REPO_ROOT / "tts_task.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _fake_book(job_id: int, chapter_ids: list) -> SimpleNamespace:
    chapters = [
        SimpleNamespace(
            chapter_db_id=cid,
            sequence_number=f"{i:04d}",
            chapter_name=f"Ch {i}",
            chapter_id=f"ch_{i:04d}",
        )
        for i, cid in enumerate(chapter_ids, start=1)
    ]
    return SimpleNamespace(job_id=job_id, book_title=f"Book {job_id}", chapters=chapters)


class TestResolveTreeSelection:
    def test_selecting_a_book_expands_to_all_its_chapters(self):
        tts_task = _load_tts_task()
        books = [_fake_book(1, [10, 11, 12])]

        resolved = tts_task._resolve_tree_selection(books, [("book", 1)])
        assert resolved == [10, 11, 12]

    def test_selecting_individual_chapters_without_the_book(self):
        tts_task = _load_tts_task()
        books = [_fake_book(1, [10, 11, 12])]

        resolved = tts_task._resolve_tree_selection(books, [("chapter", 11)])
        assert resolved == [11]

    def test_book_and_individual_chapter_selections_union_and_dedupe(self):
        tts_task = _load_tts_task()
        books = [_fake_book(1, [10, 11]), _fake_book(2, [20, 21])]

        resolved = tts_task._resolve_tree_selection(books, [("book", 1), ("chapter", 11), ("chapter", 20)])
        assert resolved == [10, 11, 20]

    def test_empty_selection_resolves_to_nothing(self):
        tts_task = _load_tts_task()
        books = [_fake_book(1, [10])]

        assert tts_task._resolve_tree_selection(books, []) == []

    def test_multiple_books_each_expand_independently(self):
        tts_task = _load_tts_task()
        books = [_fake_book(1, [10, 11]), _fake_book(2, [20, 21])]

        resolved = tts_task._resolve_tree_selection(books, [("book", 1), ("book", 2)])
        assert resolved == [10, 11, 20, 21]


class TestTreeGroups:
    def test_one_group_per_book_with_one_leaf_per_chapter(self):
        tts_task = _load_tts_task()
        books = [_fake_book(1, [10, 11])]

        groups = tts_task._tree_groups(books)
        assert len(groups) == 1
        assert groups[0].value == ("book", 1)
        assert [leaf.value for leaf in groups[0].leaves] == [("chapter", 10), ("chapter", 11)]

    def test_small_book_opens_expanded_large_book_opens_collapsed(self):
        tts_task = _load_tts_task()
        threshold = tts_task.DEFAULT_AUTO_COLLAPSE_THRESHOLD
        books = [
            _fake_book(1, list(range(100, 100 + threshold))),
            _fake_book(2, list(range(200, 200 + threshold + 1))),
        ]

        groups = tts_task._tree_groups(books)
        assert groups[0].collapsed is False
        assert groups[1].collapsed is True
