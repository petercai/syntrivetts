from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

_FINISHED_STATUSES = frozenset({"completed", "done"})


def is_archived(job: Any) -> bool:
    return getattr(job, "archived_at", None) is not None


@dataclass(frozen=True)
class JobListView:
    visible: tuple
    archived_count: int
    show_archived: bool
    active_count: int
    incomplete_count: int

    @property
    def hint(self) -> str:
        if self.archived_count == 0:
            return ""
        if self.show_archived:
            return f"Showing {self.archived_count} archived (a to hide)"
        return f"{self.archived_count} archived hidden (a to show)"

    @property
    def counter(self) -> str:
        return f"{self.incomplete_count} incomplete / {self.active_count} active  ↺ live"


def build_job_list_view(jobs: Sequence[Any], *, show_archived: bool) -> JobListView:
    active = [job for job in jobs if not is_archived(job)]
    return JobListView(
        visible=tuple(jobs if show_archived else active),
        archived_count=len(jobs) - len(active),
        show_archived=show_archived,
        active_count=len(active),
        incomplete_count=sum(1 for job in active if job.status not in _FINISHED_STATUSES),
    )


def book_title(job: Any) -> str:
    book = getattr(job, "book", None)
    return (book.title if book is not None and book.title else None) or f"Job #{job.id}"


def read_only_warning(job: Any) -> str:
    return f"“{book_title(job)}” is archived and read-only. Press x to restore it, then continue."


def archived_notice(job: Any, *, show_archived: bool) -> str:
    if show_archived:
        return f"Archived “{book_title(job)}”. tts_task will no longer offer it."
    return f"Archived “{book_title(job)}”. It is hidden now; press a to show archived jobs."


def restored_notice(job: Any) -> str:
    return f"Restored “{book_title(job)}”. It is back in the active list."
