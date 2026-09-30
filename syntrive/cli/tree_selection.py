from __future__ import annotations

from typing import Protocol, Sequence


class _BookLike(Protocol):
    job_id: int
    chapters: Sequence["_ChapterLike"]


class _ChapterLike(Protocol):
    chapter_db_id: int


def expand_tree_selection(books: Sequence[_BookLike], selected: Sequence) -> list[int]:
    book_job_ids = {value for kind, value in selected if kind == "book"}
    resolved = {value for kind, value in selected if kind == "chapter"}
    for book in books:
        if book.job_id in book_job_ids:
            resolved.update(entry.chapter_db_id for entry in book.chapters)
    return sorted(resolved)
